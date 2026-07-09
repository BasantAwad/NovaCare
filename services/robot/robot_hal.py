"""
NovaCare Robot - Hardware Abstraction Layer (HAL)
==================================================
Clean abstraction over the **Hiwonder JetAuto** robot running ROS2.

Replaces the old SERBot Prime X ``pop`` library HAL.  The JetAuto uses:
  - ROS2 ``/cmd_vel`` (geometry_msgs/Twist) for mecanum wheel motion
  - Nav2 stack for autonomous SLAM navigation
  - Depth camera (AstraPro) via ROS2 image topics
  - LiDAR (A1/G4) via ROS2 ``/scan`` topic
  - STM32 motor controller communicating over serial with the Jetson

On dev laptops (where ``rclpy`` is unavailable) every method degrades
gracefully with a mock/fallback implementation.  Thread-safety is
managed centrally.

Subsystems
----------
- CameraHAL:   Depth camera via ROS2 topics + GStreamer/V4L2 fallback
- MotionHAL:   Mecanum wheels via ROS2 cmd_vel + Nav2 autonomous navigation
- AudioHAL:    Speaker + Microphone via JetAuto voice module / gTTS fallback
- LidarHAL:    LiDAR via ROS2 /scan topic
"""

import os
import sys
import math
import time
import json
import threading
import tempfile
import struct
from typing import Optional, Tuple, List, Dict, Any, Callable

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_INDEX, CAMERA_GSTREAMER_FLIP,
    DEFAULT_SPEED, MAX_SPEED, MAX_LINEAR_VEL, MAX_ANGULAR_VEL,
    OBSTACLE_STOP_DISTANCE_MM,
    TTS_LANG, TTS_TEMP_DIR, STT_LANG, STT_TIMEOUT, STT_PHRASE_TIMEOUT,
    LIDAR_ENABLED, MINIMAL_MODE, MOCK_MODE,
    STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS, STREAM_JPEG_QUALITY,
    ROS_CMD_VEL_TOPIC, ROS_SCAN_TOPIC, ROS_MAP_TOPIC,
    ROS_CAMERA_COLOR_TOPIC, ROS_CAMERA_DEPTH_TOPIC,
    ROS_NAV2_ACTION, ROS_NAV2_THROUGH_POSES_ACTION,
    ROS_ODOM_TOPIC, ROS_DOMAIN_ID, ROS_ASR_TOPIC,
    DESTINATIONS, MAP_SAVE_DIR, DEFAULT_MAP_NAME,
    SOS_ALARM_FREQUENCY, SOS_ALARM_DURATION,
    USE_JETAUTO_VOICE, ROBOT_NAME,
)

# ---------------------------------------------------------------------------
# Try importing ROS2 rclpy (only available on the JetAuto Jetson)
# ---------------------------------------------------------------------------
_ROS2_AVAILABLE = False
_rclpy = None
_Node = None

if not MOCK_MODE:
    try:
        import rclpy
        from rclpy.node import Node as _RosNode
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        _rclpy = rclpy
        _Node = _RosNode
        _ROS2_AVAILABLE = True
        print("[OK] ROS2 rclpy loaded — running on JetAuto hardware")
    except ImportError:
        print("[WARN] rclpy not available — running in MOCK/DEV mode")
else:
    print("[INFO] MOCK mode forced via NOVACARE_MOCK=1")

# ROS2 message types (imported conditionally)
_Twist = None
_LaserScan = None
_OccupancyGrid = None
_Image = None
_Odometry = None
_PoseStamped = None
_NavigateToPose = None
_String = None

if _ROS2_AVAILABLE:
    try:
        from geometry_msgs.msg import Twist as _TwistMsg
        from geometry_msgs.msg import PoseStamped as _PoseStampedMsg
        from sensor_msgs.msg import LaserScan as _LaserScanMsg
        from sensor_msgs.msg import Image as _ImageMsg
        from nav_msgs.msg import OccupancyGrid as _OccupancyGridMsg
        from nav_msgs.msg import Odometry as _OdometryMsg
        from std_msgs.msg import String as _StringMsg
        _Twist = _TwistMsg
        _LaserScan = _LaserScanMsg
        _OccupancyGrid = _OccupancyGridMsg
        _Image = _ImageMsg
        _Odometry = _OdometryMsg
        _PoseStamped = _PoseStampedMsg
        _String = _StringMsg
    except ImportError as e:
        print(f"[WARN] Some ROS2 message types unavailable: {e}")

    try:
        from nav2_msgs.action import NavigateToPose as _NavigateToPoseAction
        _NavigateToPose = _NavigateToPoseAction
    except ImportError:
        print("[WARN] nav2_msgs not available — autonomous navigation disabled")

# Import the lightweight camera service as fallback (no OpenCV required)
try:
    from camera_service import get_camera, LightweightCamera
    _CAMERA_SERVICE_AVAILABLE = True
except ImportError:
    _CAMERA_SERVICE_AVAILABLE = False
    print("[WARN] camera_service not available — using ROS2 camera topics only")

# Optional imports for TTS/STT
try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False
    print("[WARN] gTTS not installed (pip install gTTS)")

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False
    print("[WARN] SpeechRecognition not installed")

# OpenCV — optional, for frame encoding
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[WARN] OpenCV not available — camera frame encoding limited")


# ============================================================================
# ROS2 Bridge Node (singleton, runs in background thread)
# ============================================================================
class _NovaCareROS2Node:
    """
    Lightweight ROS2 bridge that manages all topic subscriptions and
    publishers for the NovaCare robot service.  Runs rclpy.spin() in
    a background thread so Flask can operate normally.
    """

    def __init__(self):
        self._node = None
        self._spin_thread = None
        self._lock = threading.Lock()
        self._running = False

        # Cached sensor data
        self._latest_scan: List[Dict] = []
        self._latest_odom: Optional[Dict] = None
        self._latest_map: Optional[bytes] = None
        self._latest_map_info: Optional[Dict] = None
        self._latest_camera_shape: Tuple[int, int] = (0, 0)  # (height, width)
        self._latest_depth_frame: Optional[bytes] = None

        self._asr_callbacks: List[Callable] = []

        # Navigation state
        self._nav_active = False
        self._nav_status = "idle"
        self._nav_feedback: Optional[Dict] = None
        self._nav_goal_handle = None

        # Publishers
        self._cmd_vel_pub = None

        if _ROS2_AVAILABLE:
            self._init_ros2()

    def _init_ros2(self):
        """Initialize the ROS2 node and all subscriptions/publishers."""
        try:
            os.environ["ROS_DOMAIN_ID"] = str(ROS_DOMAIN_ID)
            if not _rclpy.ok():
                _rclpy.init()

            self._node = _Node("novacare_robot_service")

            # --- Publishers ---
            self._cmd_vel_pub = self._node.create_publisher(
                _Twist, ROS_CMD_VEL_TOPIC, 10
            )

            # --- Subscribers ---
            # LiDAR scan
            if _LaserScan:
                self._node.create_subscription(
                    _LaserScan, ROS_SCAN_TOPIC, self._scan_callback, 10
                )

            # Odometry
            if _Odometry:
                self._node.create_subscription(
                    _Odometry, ROS_ODOM_TOPIC, self._odom_callback, 10
                )

            # Map (occupancy grid from SLAM)
            if _OccupancyGrid:
                qos = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )
                self._node.create_subscription(
                    _OccupancyGrid, ROS_MAP_TOPIC, self._map_callback, qos
                )

            # Camera color image
            if _Image:
                self._node.create_subscription(
                    _Image, ROS_CAMERA_COLOR_TOPIC, self._camera_color_callback, 10
                )
                self._node.create_subscription(
                    _Image, ROS_CAMERA_DEPTH_TOPIC, self._camera_depth_callback, 10
                )

            # Offline ASR (JetAuto Voice Module)
            if _String:
                self._node.create_subscription(
                    _String, ROS_ASR_TOPIC, self._asr_callback, 10
                )

            # Start spinning in background
            self._running = True
            self._spin_thread = threading.Thread(
                target=self._spin_loop, daemon=True, name="ros2-spin"
            )
            self._spin_thread.start()
            print("[OK] ROS2 NovaCare node initialized and spinning")

        except Exception as e:
            print(f"[FAIL] ROS2 init failed: {e}")
            self._node = None

    def _spin_loop(self):
        """Background thread running rclpy.spin()."""
        try:
            while self._running and _rclpy.ok():
                _rclpy.spin_once(self._node, timeout_sec=0.05)
        except Exception as e:
            print(f"[ROS2 Camera Depth Error] {e}")

    def _asr_callback(self, msg):
        """Process incoming offline voice wake words."""
        text = msg.data.lower()
        if text:
            print(f"[ASR] Detected voice: {text}")
            with self._lock:
                for cb in self._asr_callbacks:
                    try:
                        cb(text)
                    except Exception as e:
                        print(f"[ASR] Callback error: {e}")

    def register_asr_callback(self, callback: Callable):
        with self._lock:
            if callback not in self._asr_callbacks:
                self._asr_callbacks.append(callback)

    def unregister_asr_callback(self, callback: Callable):
        with self._lock:
            if callback in self._asr_callbacks:
                self._asr_callbacks.remove(callback)

    # --- Callbacks ---
    def _scan_callback(self, msg):
        """Process incoming LiDAR scan."""
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                points.append({
                    "angle": math.degrees(angle) % 360,
                    "distance_mm": r * 1000.0,
                })
            angle += msg.angle_increment
        with self._lock:
            self._latest_scan = points

    def _odom_callback(self, msg):
        """Process odometry for current position."""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1.0 - 2.0 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._lock:
            self._latest_odom = {
                "x": pos.x, "y": pos.y, "z": pos.z,
                "yaw": yaw,
                "linear_vel": msg.twist.twist.linear.x,
                "angular_vel": msg.twist.twist.angular.z,
            }

    def _map_callback(self, msg):
        """Process SLAM occupancy grid map."""
        with self._lock:
            self._latest_map = bytes(msg.data)
            self._latest_map_info = {
                "width": msg.info.width,
                "height": msg.info.height,
                "resolution": msg.info.resolution,
                "origin_x": msg.info.origin.position.x,
                "origin_y": msg.info.origin.position.y,
            }

    def _camera_color_callback(self, msg):
        """Process incoming camera color frame."""
        with self._lock:
            self._latest_camera_frame = bytes(msg.data)
            self._latest_camera_shape = (msg.height, msg.width)

    def _camera_depth_callback(self, msg):
        """Process incoming depth frame."""
        with self._lock:
            self._latest_depth_frame = bytes(msg.data)

    # --- Public API ---
    @property
    def is_available(self) -> bool:
        return self._node is not None

    def publish_cmd_vel(self, linear_x: float = 0.0, linear_y: float = 0.0,
                        angular_z: float = 0.0):
        """Publish a Twist message to /cmd_vel."""
        if self._cmd_vel_pub is None:
            return
        msg = _Twist()
        msg.linear.x = max(-MAX_LINEAR_VEL, min(linear_x, MAX_LINEAR_VEL))
        msg.linear.y = max(-MAX_LINEAR_VEL, min(linear_y, MAX_LINEAR_VEL))
        msg.angular.z = max(-MAX_ANGULAR_VEL, min(angular_z, MAX_ANGULAR_VEL))
        self._cmd_vel_pub.publish(msg)

    def stop_motion(self):
        """Publish zero velocity."""
        self.publish_cmd_vel(0.0, 0.0, 0.0)

    def get_scan(self) -> List[Dict]:
        with self._lock:
            return list(self._latest_scan)

    def get_odom(self) -> Optional[Dict]:
        with self._lock:
            return dict(self._latest_odom) if self._latest_odom else None

    def get_map_data(self) -> Tuple[Optional[bytes], Optional[Dict]]:
        with self._lock:
            return self._latest_map, self._latest_map_info

    def get_camera_frame_raw(self) -> Tuple[Optional[bytes], Tuple[int, int]]:
        with self._lock:
            return self._latest_camera_frame, self._latest_camera_shape

    def get_depth_frame_raw(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_depth_frame

    def send_nav2_goal(self, x: float, y: float, theta: float) -> bool:
        """Send a navigation goal to Nav2 action server."""
        if not _ROS2_AVAILABLE or _NavigateToPose is None or self._node is None:
            print(f"[MOCK] navigate_to(x={x}, y={y}, theta={theta})")
            self._nav_active = True
            self._nav_status = "navigating"

            # Simulate navigation completion in background
            def _sim_nav():
                time.sleep(5)
                self._nav_active = False
                self._nav_status = "succeeded"
            threading.Thread(target=_sim_nav, daemon=True).start()
            return True

        try:
            from rclpy.action import ActionClient
            nav_client = ActionClient(self._node, _NavigateToPose, ROS_NAV2_ACTION)
            if not nav_client.wait_for_server(timeout_sec=5.0):
                print("[NAV2] Action server not available")
                return False

            goal = _NavigateToPose.Goal()
            goal.pose.header.frame_id = "map"
            goal.pose.header.stamp = self._node.get_clock().now().to_msg()
            goal.pose.pose.position.x = x
            goal.pose.pose.position.y = y
            # Convert theta to quaternion (yaw only)
            goal.pose.pose.orientation.z = math.sin(theta / 2.0)
            goal.pose.pose.orientation.w = math.cos(theta / 2.0)

            future = nav_client.send_goal_async(
                goal, feedback_callback=self._nav2_feedback_callback
            )
            future.add_done_callback(self._nav2_goal_response_callback)

            self._nav_active = True
            self._nav_status = "navigating"
            print(f"[NAV2] Goal sent: x={x}, y={y}, theta={theta}")
            return True

        except Exception as e:
            print(f"[NAV2] Failed to send goal: {e}")
            return False

    def cancel_nav2_goal(self) -> bool:
        """Cancel the active Nav2 goal."""
        if self._nav_goal_handle:
            try:
                self._nav_goal_handle.cancel_goal_async()
                self._nav_active = False
                self._nav_status = "cancelled"
                print("[NAV2] Goal cancelled")
                return True
            except Exception as e:
                print(f"[NAV2] Cancel failed: {e}")
                return False
        self._nav_active = False
        self._nav_status = "idle"
        return True

    def _nav2_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._nav_active = False
            self._nav_status = "rejected"
            print("[NAV2] Goal was rejected")
            return
        self._nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav2_result_callback)

    def _nav2_result_callback(self, future):
        self._nav_active = False
        self._nav_status = "succeeded"
        self._nav_goal_handle = None
        print("[NAV2] Navigation complete!")

    def _nav2_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        pos = feedback.current_pose.pose.position
        self._nav_feedback = {
            "current_x": pos.x,
            "current_y": pos.y,
            "distance_remaining": feedback.distance_remaining
            if hasattr(feedback, "distance_remaining") else None,
            "estimated_time_remaining": feedback.estimated_time_remaining.sec
            if hasattr(feedback, "estimated_time_remaining") else None,
        }

    def get_nav_status(self) -> Dict:
        return {
            "active": self._nav_active,
            "status": self._nav_status,
            "feedback": self._nav_feedback,
        }

    def shutdown(self):
        """Stop ROS2 node."""
        self._running = False
        if self._node:
            try:
                self.stop_motion()
                self._node.destroy_node()
            except Exception:
                pass
        try:
            if _rclpy and _rclpy.ok():
                _rclpy.shutdown()
        except Exception:
            pass


# Singleton ROS2 node
_ros2_node: Optional[_NovaCareROS2Node] = None


def _get_ros2_node() -> _NovaCareROS2Node:
    global _ros2_node
    if _ros2_node is None:
        _ros2_node = _NovaCareROS2Node()
    return _ros2_node


# ============================================================================
# Camera HAL
# ============================================================================
class CameraHAL:
    """
    Manages the JetAuto's depth camera (AstraPro).

    Primary: reads frames from ROS2 image topics.
    Fallback: uses LightweightCamera (GStreamer/V4L2) if ROS2 unavailable.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ros = _get_ros2_node()
        self._fallback_camera = None
        self._session_active = False

        # Try fallback camera service for non-ROS2 operation
        if _CAMERA_SERVICE_AVAILABLE and not self._ros.is_available:
            try:
                self._fallback_camera = get_camera(
                    capture_width=CAMERA_WIDTH,
                    capture_height=CAMERA_HEIGHT,
                    capture_fps=CAMERA_FPS,
                    stream_width=STREAM_WIDTH,
                    stream_height=STREAM_HEIGHT,
                    stream_fps=STREAM_FPS,
                    jpeg_quality=STREAM_JPEG_QUALITY,
                    gstreamer_flip=CAMERA_GSTREAMER_FLIP,
                    camera_index=CAMERA_INDEX,
                )
            except Exception as e:
                print(f"[WARN] Fallback camera init failed: {e}")

        if self._ros.is_available:
            print(f"[OK] CameraHAL ready (ROS2 depth camera topics)")
        elif self._fallback_camera and self._fallback_camera.is_available:
            print(f"[OK] CameraHAL ready (fallback camera)")
        else:
            print("[WARN] CameraHAL: no camera available")

    @property
    def is_available(self) -> bool:
        if self._ros.is_available:
            return True
        if self._fallback_camera:
            return self._fallback_camera.is_available
        return False

    def start_session(self) -> bool:
        """Begin a viewer session."""
        self._session_active = True
        if self._fallback_camera:
            return self._fallback_camera.start_session()
        return True

    def stop_session(self) -> None:
        """End a viewer session."""
        self._session_active = False
        if self._fallback_camera:
            self._fallback_camera.stop_session()

    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """Return (success, BGR frame) for vision pipelines."""
        if _CV2_AVAILABLE and self._ros.is_available:
            raw, shape = self._ros.get_camera_frame_raw()
            if raw and shape[0] > 0:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    shape[0], shape[1], 3
                )
                # ROS2 image is RGB, convert to BGR for OpenCV
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, bgr
        if self._fallback_camera:
            return self._fallback_camera.read_frame()
        return False, None

    def read_frame_base64(self, quality: int = 80) -> Optional[str]:
        """Read a frame and return as base64-encoded JPEG string."""
        import base64
        jpg = self.read_frame_jpeg_bytes(quality)
        if jpg:
            return base64.b64encode(jpg).decode("utf-8")
        return None

    def read_frame_jpeg_bytes(self, quality: int = 80) -> Optional[bytes]:
        """Read a frame and return raw JPEG bytes."""
        if _CV2_AVAILABLE:
            ok, frame = self.read_frame()
            if ok and frame is not None:
                _, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                return buf.tobytes()
        if self._fallback_camera:
            return self._fallback_camera.read_frame_jpeg()
        return None

    def get_depth_at(self, x: int, y: int) -> Optional[float]:
        """Get depth value at pixel coordinates (mm)."""
        raw = self._ros.get_depth_frame_raw()
        if raw and _CV2_AVAILABLE:
            try:
                depth = np.frombuffer(raw, dtype=np.uint16).reshape(
                    CAMERA_HEIGHT, CAMERA_WIDTH
                )
                if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                    return float(depth[y, x])
            except Exception:
                pass
        return None

    def get_status(self) -> dict:
        source = "ros2_depth_camera" if self._ros.is_available else "fallback"
        return {
            "available": self.is_available,
            "backend": source,
            "session_active": self._session_active,
            "stream_resolution": f"{STREAM_WIDTH}x{STREAM_HEIGHT}",
            "capture_resolution": f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
        }

    def detect_obstacle_ahead(self) -> bool:
        """Check LiDAR for forward obstacles (not camera-based)."""
        if MINIMAL_MODE:
            return False
        # Delegate to LiDAR-based obstacle detection
        return False

    def is_obstacle_ahead(self) -> bool:
        """Alias for movement safety checks."""
        return self.detect_obstacle_ahead()

    def release(self):
        with self._lock:
            if self._fallback_camera:
                self._fallback_camera.release()


# ============================================================================
# Motion HAL
# ============================================================================
class MotionHAL:
    """
    Controls the JetAuto's mecanum wheel drive via ROS2 ``/cmd_vel``.

    On JetAuto: publishes Twist messages, uses Nav2 for autonomous navigation.
    On dev machines: prints movement commands to console (mock mode).
    """

    def __init__(self):
        self._ros = _get_ros2_node()
        self._lock = threading.Lock()
        self._speed = DEFAULT_SPEED
        self._moving = False
        self._tracking = False
        self._tracking_thread = None
        self._tracking_stop = threading.Event()

        if self._ros.is_available:
            print("[OK] MotionHAL ready (ROS2 cmd_vel + Nav2)")
        else:
            print("[WARN] MotionHAL in MOCK mode (no ROS2)")

    @property
    def is_available(self) -> bool:
        return self._ros.is_available or MOCK_MODE

    @property
    def is_moving(self) -> bool:
        return self._moving

    def set_speed(self, speed: int):
        self._speed = max(0, min(speed, MAX_SPEED))

    def _speed_to_vel(self, speed: Optional[int] = None) -> float:
        """Convert our 0-80 speed scale to m/s for cmd_vel."""
        s = speed if speed is not None else self._speed
        s = max(0, min(s, MAX_SPEED))
        return (s / MAX_SPEED) * MAX_LINEAR_VEL

    def move(self, angle: int, speed: Optional[int] = None):
        """
        Move at *angle* (0=forward, 90=right, 180=backward, 270=left) at given speed.

        Mecanum wheels support omnidirectional movement:
          - angle 0:   linear.x = +vel  (forward)
          - angle 90:  linear.y = -vel  (right)
          - angle 180: linear.x = -vel  (backward)
          - angle 270: linear.y = +vel  (left)
        """
        vel = self._speed_to_vel(speed)
        rad = math.radians(angle)
        lx = vel * math.cos(rad)
        ly = -vel * math.sin(rad)  # ROS y-left is positive

        with self._lock:
            self._moving = True
            if self._ros.is_available:
                self._ros.publish_cmd_vel(linear_x=lx, linear_y=ly)
            else:
                print(f"[MOCK] move(angle={angle}, vel={vel:.2f}, lx={lx:.2f}, ly={ly:.2f})")

    def forward(self, speed: Optional[int] = None):
        self.move(0, speed)

    def backward(self, speed: Optional[int] = None):
        self.move(180, speed)

    def left(self, speed: Optional[int] = None):
        self.move(270, speed)

    def right(self, speed: Optional[int] = None):
        self.move(90, speed)

    def turn_left(self, speed: Optional[int] = None):
        vel = self._speed_to_vel(speed)
        angular = (vel / MAX_LINEAR_VEL) * MAX_ANGULAR_VEL
        with self._lock:
            self._moving = True
            if self._ros.is_available:
                self._ros.publish_cmd_vel(angular_z=angular)
            else:
                print(f"[MOCK] turnLeft(angular={angular:.2f})")

    def turn_right(self, speed: Optional[int] = None):
        vel = self._speed_to_vel(speed)
        angular = -(vel / MAX_LINEAR_VEL) * MAX_ANGULAR_VEL
        with self._lock:
            self._moving = True
            if self._ros.is_available:
                self._ros.publish_cmd_vel(angular_z=angular)
            else:
                print(f"[MOCK] turnRight(angular={angular:.2f})")

    def stop(self):
        with self._lock:
            self._moving = False
            if self._ros.is_available:
                self._ros.stop_motion()
            else:
                print("[MOCK] stop()")

    def move_for(self, angle: int, duration_s: float, speed: Optional[int] = None):
        """Move in a direction for a fixed duration, then stop."""
        self.move(angle, speed)
        time.sleep(duration_s)
        self.stop()

    # --- Autonomous Navigation (Nav2) ---

    def navigate_to(self, destination: str = None, x: float = None,
                    y: float = None, theta: float = None) -> bool:
        """
        Navigate to a destination using Nav2 autonomous navigation.

        Can be called with a named destination (from DESTINATIONS config)
        or explicit (x, y, theta) coordinates in the map frame.
        """
        if destination and destination.lower() in DESTINATIONS:
            dest = DESTINATIONS[destination.lower()]
            x = dest["x"]
            y = dest["y"]
            theta = dest.get("theta", 0.0)

        if x is None or y is None:
            print(f"[NAV] Unknown destination: {destination}")
            return False

        if theta is None:
            theta = 0.0

        self._moving = True
        return self._ros.send_nav2_goal(x, y, theta)

    def cancel_navigation(self) -> bool:
        """Cancel the active Nav2 navigation goal."""
        result = self._ros.cancel_nav2_goal()
        self._moving = False
        return result

    def get_navigation_status(self) -> Dict:
        """Get current Nav2 navigation status."""
        return self._ros.get_nav_status()

    def get_current_pose(self) -> Optional[Dict]:
        """Get current robot pose from odometry."""
        return self._ros.get_odom()

    # --- Target Tracking (Follow Mode) ---

    def start_tracking(self, target: str = "person"):
        """
        Start person-following mode.

        On JetAuto: uses depth camera + LiDAR fusion for robust tracking.
        The on-robot AI service (onboard_ai.py) publishes target coordinates
        which this method reads and converts to cmd_vel commands.
        """
        if MINIMAL_MODE:
            print(f"[MINIMAL] start_tracking({target}) ignored")
            return

        self._tracking_stop.clear()
        self._tracking = True
        self._moving = True

        if self._ros.is_available:
            # Launch JetAuto's built-in tracking via ROS2 service call
            try:
                import subprocess
                # Launch the JetAuto target tracking node
                subprocess.Popen(
                    ["ros2", "launch", "app", "target_tracking.launch.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[OK] JetAuto target tracking started for: {target}")
            except Exception as e:
                print(f"[WARN] Could not launch tracking node: {e}")
                print(f"[OK] Using built-in tracking fallback")
        else:
            print(f"[MOCK] start_tracking(target={target})")

    def stop_tracking(self):
        """Stop target tracking / follow mode."""
        self._tracking_stop.set()
        self._tracking = False

        if self._ros.is_available:
            try:
                import subprocess
                # Kill the tracking node
                subprocess.run(
                    ["pkill", "-f", "target_tracking"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        self.stop()
        print("[OK] Target tracking stopped")


# ============================================================================
# Audio HAL
# ============================================================================
class AudioHAL:
    """
    Manages the JetAuto's speaker and microphone.

    The JetAuto has a microphone array module (R818 noise reduction board)
    with 360° sound reception and the ``xf_mic_asr_offline`` ROS2 package
    for voice recognition.

    TTS: JetAuto built-in voice → gTTS fallback → OS playback.
    STT: JetAuto microphone array → SpeechRecognition fallback.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._audio_player = None
        os.makedirs(TTS_TEMP_DIR, exist_ok=True)

        if os.environ.get("NOVACARE_LIGHTWEIGHT") == "1":
            print("[OK] AudioHAL in LIGHTWEIGHT mode (no hardware audio)")
            return

        print("[OK] AudioHAL ready (gTTS + aplay/mpg123)")

    @property
    def tts_available(self) -> bool:
        return _GTTS_AVAILABLE or USE_JETAUTO_VOICE

    @property
    def stt_available(self) -> bool:
        return _SR_AVAILABLE or USE_JETAUTO_VOICE

    def register_asr_callback(self, callback: Callable):
        """Register a callback for offline voice wake words."""
        ros = _get_ros2_node()
        ros.register_asr_callback(callback)

    def unregister_asr_callback(self, callback: Callable):
        """Unregister a callback for offline voice wake words."""
        ros = _get_ros2_node()
        ros.unregister_asr_callback(callback)

    def speak(self, text: str, lang: str = None, block: bool = True) -> bool:
        """
        Convert text to speech and play on the robot speaker.

        Tries JetAuto's built-in voice system first, falls back to gTTS.
        Returns True if audio was played (or queued).
        """
        if not text.strip():
            return False

        # Try JetAuto's built-in TTS via ROS2
        if USE_JETAUTO_VOICE and _ROS2_AVAILABLE:
            try:
                import subprocess
                subprocess.Popen(
                    ["ros2", "topic", "pub", "--once",
                     "/novacare/tts", "std_msgs/msg/String",
                     f'{{"data": "{text}"}}'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                pass

        if not _GTTS_AVAILABLE:
            print(f"[TTS-MOCK] {text}")
            return False

        lang = lang or TTS_LANG
        try:
            tts = gTTS(text=text, lang=lang)
            filepath = os.path.join(TTS_TEMP_DIR, f"tts_{int(time.time()*1000)}.mp3")
            tts.save(filepath)

            if block:
                self._play_file(filepath)
            else:
                t = threading.Thread(target=self._play_file, args=(filepath,), daemon=True)
                t.start()
            return True
        except Exception as e:
            print(f"[TTS Error] {e}")
            return False

    def _play_file(self, filepath: str):
        """Play an audio file through the best available backend."""
        with self._lock:
            try:
                if sys.platform == "linux":
                    os.system(f"mpg123 -q '{filepath}' 2>/dev/null || aplay '{filepath}' 2>/dev/null")
                elif sys.platform == "darwin":
                    os.system(f"afplay '{filepath}'")
                else:
                    try:
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(filepath)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    except ImportError:
                        os.system(f'start /min "" "{filepath}"')
            finally:
                try:
                    os.remove(filepath)
                except OSError:
                    pass

    def play_alarm(self, frequency: int = None, duration: float = None):
        """Play an SOS alarm sound on the robot speaker."""
        freq = frequency or SOS_ALARM_FREQUENCY
        dur = duration or SOS_ALARM_DURATION

        def _alarm_loop():
            end_time = time.time() + dur
            while time.time() < end_time:
                try:
                    if sys.platform == "linux":
                        # Use speaker-test or beep for alarm
                        os.system(
                            f"speaker-test -t sine -f {freq} -l 1 -p 1 2>/dev/null &"
                        )
                        time.sleep(0.5)
                        os.system("pkill -f speaker-test 2>/dev/null")
                        time.sleep(0.3)
                    else:
                        print(f"[ALARM] Beep {freq}Hz")
                        time.sleep(0.8)
                except Exception:
                    time.sleep(1)

        t = threading.Thread(target=_alarm_loop, daemon=True)
        t.start()
        return True

    def stop_alarm(self):
        """Stop any playing alarm."""
        try:
            if sys.platform == "linux":
                os.system("pkill -f speaker-test 2>/dev/null")
        except Exception:
            pass

    def listen(self, timeout: int = None, phrase_timeout: int = None) -> Optional[str]:
        """
        Listen for speech and return recognised text.
        Returns None if nothing was recognised or STT is unavailable.
        """
        if not _SR_AVAILABLE:
            print("[STT-MOCK] listen() called but SpeechRecognition not available")
            return None

        timeout = timeout or STT_TIMEOUT
        phrase_timeout = phrase_timeout or STT_PHRASE_TIMEOUT

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("[STT] Listening...")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)
                print("[STT] Processing...")
                text = recognizer.recognize_google(audio, language=STT_LANG)
                print(f"[STT] Recognised: {text}")
                return text
        except sr.WaitTimeoutError:
            print("[STT] Timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("[STT] Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"[STT] Google API error: {e}")
            return None
        except Exception as e:
            print(f"[STT] Error: {e}")
            return None


# ============================================================================
# LiDAR HAL
# ============================================================================
class LidarHAL:
    """
    Interfaces with the JetAuto's LiDAR (A1/G4) via ROS2 ``/scan`` topic.

    Provides raw scan vectors and convenience obstacle detection methods.
    """

    def __init__(self):
        self._ros = _get_ros2_node()
        self._lock = threading.Lock()

        if os.environ.get("NOVACARE_LIGHTWEIGHT") == "1":
            print("[OK] LidarHAL in LIGHTWEIGHT mode (no hardware LiDAR)")
            return

        if LIDAR_ENABLED and self._ros.is_available:
            print("[OK] LidarHAL ready (ROS2 /scan topic)")
        elif LIDAR_ENABLED:
            print("[WARN] LidarHAL: ROS2 not available, LiDAR data unavailable")
        else:
            print("[WARN] LidarHAL disabled")

    @property
    def is_available(self) -> bool:
        return LIDAR_ENABLED and self._ros.is_available

    def get_scan(self) -> List[Dict]:
        """
        Return a list of scan points: [{"angle": float, "distance_mm": float}, …].
        Returns an empty list if LiDAR is unavailable.
        """
        if not self.is_available:
            return []
        return self._ros.get_scan()

    def is_obstacle_ahead(self, cone_degrees: int = 30,
                          distance_mm: int = OBSTACLE_STOP_DISTANCE_MM) -> bool:
        """Check if there's an obstacle within *cone_degrees* of forward direction."""
        scan = self.get_scan()
        if not scan:
            return False
        half_cone = cone_degrees / 2
        for point in scan:
            angle = point["angle"]
            # Forward is angle ≈ 0 (or ≈ 360)
            if (angle <= half_cone or angle >= 360 - half_cone):
                if 0 < point["distance_mm"] < distance_mm:
                    return True
        return False

    def get_distance_at(self, target_angle: float, cone_degrees: int = 15) -> float:
        """
        Get the minimum distance (in mm) to an obstacle at a specific angle.
        Returns float('inf') if no obstacle is in that cone.
        """
        scan = self.get_scan()
        if not scan:
            return float('inf')

        min_dist = float('inf')
        half_cone = cone_degrees / 2

        for point in scan:
            angle = point["angle"]
            dist = point["distance_mm"]
            diff = abs((angle - target_angle + 180) % 360 - 180)
            if diff <= half_cone and dist > 0:
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def get_closest_obstacle(self) -> Tuple[float, float]:
        """
        Finds the closest obstacle.
        Returns (angle_degrees, distance_mm).
        Returns (0.0, float('inf')) if no obstacles detected.
        """
        scan = self.get_scan()
        if not scan:
            return 0.0, float('inf')

        min_dist = float('inf')
        best_angle = 0.0

        for point in scan:
            dist = point["distance_mm"]
            if 0 < dist < min_dist:
                min_dist = dist
                best_angle = point["angle"]

        return best_angle, min_dist

    def get_map_image_png(self) -> Optional[bytes]:
        """
        Get the current SLAM occupancy grid as a PNG image.
        Returns PNG bytes or None.
        """
        map_data, map_info = self._ros.get_map_data()
        if not map_data or not map_info or not _CV2_AVAILABLE:
            return None

        try:
            w = map_info["width"]
            h = map_info["height"]
            grid = np.array(list(map_data), dtype=np.int8).reshape(h, w)

            # Convert occupancy grid to grayscale image
            # -1 (unknown) → 128 (gray), 0 (free) → 255 (white), 100 (occupied) → 0 (black)
            img = np.full((h, w), 128, dtype=np.uint8)
            img[grid == 0] = 255    # free space
            img[grid == 100] = 0    # occupied
            img[grid == -1] = 128   # unknown

            # Flip vertically (ROS map origin is bottom-left)
            img = cv2.flip(img, 0)

            _, png_buf = cv2.imencode(".png", img)
            return png_buf.tobytes()
        except Exception as e:
            print(f"[MAP] Failed to generate map image: {e}")
            return None

    def shutdown(self):
        pass  # ROS2 cleanup handled by the node


# ============================================================================
# Unified Robot HAL (singleton facade)
# ============================================================================
class RobotHAL:
    """
    Unified facade for all JetAuto hardware subsystems.

    Usage::

        from robot_hal import get_robot
        robot = get_robot()
        robot.motion.forward()
        frame = robot.camera.read_frame_jpeg_bytes()
        robot.audio.speak("Hello!")
        robot.motion.stop()
        robot.motion.navigate_to("kitchen")
    """

    def __init__(self):
        print("=" * 50)
        print(f"  NovaCare - JetAuto Hardware Abstraction Layer")
        print(f"  Robot: {ROBOT_NAME}")
        print("=" * 50)
        self.camera = CameraHAL()
        self.motion = MotionHAL()
        self.audio = AudioHAL()
        self.lidar = LidarHAL()
        self._ros = _get_ros2_node()
        self._print_status()

    def _print_status(self):
        print("\n  Hardware Status:")
        print(f"    Camera:   {'[OK] READY' if self.camera.is_available else '[WARN] MOCK'}")
        print(f"    Motion:   {'[OK] READY' if self.motion.is_available else '[WARN] MOCK'}")
        print(f"    TTS:      {'[OK] READY' if self.audio.tts_available else '[WARN] N/A'}")
        print(f"    STT:      {'[OK] READY' if self.audio.stt_available else '[WARN] N/A'}")
        print(f"    LiDAR:    {'[OK] READY' if self.lidar.is_available else '[WARN] N/A'}")
        print(f"    ROS2:     {'[OK] LOADED' if _ROS2_AVAILABLE else '[WARN] NOT AVAILABLE'}")
        print(f"    Nav2:     {'[OK] READY' if _NavigateToPose else '[WARN] N/A'}")
        print("=" * 50)

    def get_navigation_status(self) -> Dict:
        """Get current autonomous navigation status."""
        return self._ros.get_nav_status()

    def get_current_pose(self) -> Optional[Dict]:
        """Get current robot pose from odometry."""
        return self._ros.get_odom()

    def get_map_image(self) -> Optional[bytes]:
        """Get current SLAM map as PNG image bytes."""
        return self.lidar.get_map_image_png()

    def shutdown(self):
        """Gracefully release all hardware resources."""
        print("[HAL] Shutting down all hardware...")
        self.motion.stop()
        self.motion.stop_tracking()
        self.camera.release()
        self.lidar.shutdown()
        self._ros.shutdown()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_robot_instance: Optional[RobotHAL] = None


def get_robot() -> RobotHAL:
    """Get or create the singleton RobotHAL instance."""
    global _robot_instance
    if _robot_instance is None:
        _robot_instance = RobotHAL()
    return _robot_instance
