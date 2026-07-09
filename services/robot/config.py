"""
NovaCare Robot — Configuration
===============================
Hardware and service configuration for the Hiwonder JetAuto (ROS2).

Replaces the old SERBot Prime X (pop library) configuration.
The JetAuto uses ROS2 for motor control (cmd_vel), Nav2 for autonomous
navigation, and a depth camera (AstraPro) + LiDAR for SLAM mapping.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# Runtime mode
# ---------------------------------------------------------------------------
# MINIMAL_MODE: run as I/O bridge only — no AI, no watch, no Nav2
MINIMAL_MODE = os.getenv("NOVACARE_MINIMAL", "0").lower() in ("1", "true", "yes")
# MOCK_MODE: true when running off-robot (dev laptop) — all hardware mocked
MOCK_MODE = os.getenv("NOVACARE_MOCK", "0").lower() in ("1", "true", "yes")

ROBOT_SERVICE_HOST = os.getenv("ROBOT_SERVICE_HOST", "0.0.0.0")
ROBOT_SERVICE_PORT = int(os.getenv("ROBOT_SERVICE_PORT", "9000"))

# ---------------------------------------------------------------------------
# JetAuto Robot Identity
# ---------------------------------------------------------------------------
ROBOT_NAME = os.getenv("ROBOT_NAME", "JETAUTO-NC-001")
ROBOT_TYPE = os.getenv("ROBOT_TYPE", "jetauto_pro")   # jetauto | jetauto_pro

# ---------------------------------------------------------------------------
# ROS2 Configuration
# ---------------------------------------------------------------------------
# ROS2 domain ID (match the JetAuto's ROS_DOMAIN_ID)
ROS_DOMAIN_ID = int(os.getenv("ROS_DOMAIN_ID", "0"))

# Topics
ROS_CMD_VEL_TOPIC = os.getenv("ROS_CMD_VEL_TOPIC", "/cmd_vel")
ROS_ODOM_TOPIC = os.getenv("ROS_ODOM_TOPIC", "/odom")
ROS_SCAN_TOPIC = os.getenv("ROS_SCAN_TOPIC", "/scan")
ROS_MAP_TOPIC = os.getenv("ROS_MAP_TOPIC", "/map")
ROS_CAMERA_COLOR_TOPIC = os.getenv("ROS_CAMERA_COLOR_TOPIC", "/camera/color/image_raw")
ROS_CAMERA_DEPTH_TOPIC = os.getenv("ROS_CAMERA_DEPTH_TOPIC", "/camera/depth/image_raw")
ROS_IMU_TOPIC = os.getenv("ROS_IMU_TOPIC", "/imu/data")
ROS_ASR_TOPIC = os.getenv("ROS_ASR_TOPIC", "/xf_mic_asr_offline/voice_words")

# Nav2 action server
ROS_NAV2_ACTION = os.getenv("ROS_NAV2_ACTION", "/navigate_to_pose")
ROS_NAV2_THROUGH_POSES_ACTION = os.getenv("ROS_NAV2_THROUGH_POSES_ACTION", "/navigate_through_poses")

# ---------------------------------------------------------------------------
# Camera (Depth Camera — AstraPro / AstraPro Plus)
# ---------------------------------------------------------------------------
CAMERA_TYPE = os.getenv("CAMERA_TYPE", "astra_pro")  # astra_pro | astra_pro_plus | monocular
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
# Camera index fallback (used when ROS2 topics are unavailable)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
# Jetson CSI flip-method for GStreamer / nvarguscamerasrc
CAMERA_GSTREAMER_FLIP = int(os.getenv("CAMERA_GSTREAMER_FLIP", "0"))

# ---------------------------------------------------------------------------
# Live Stream
# ---------------------------------------------------------------------------
STREAM_WIDTH = int(os.getenv("STREAM_WIDTH", "320"))
STREAM_HEIGHT = int(os.getenv("STREAM_HEIGHT", "240"))
STREAM_FPS = int(os.getenv("STREAM_FPS", "15"))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "65"))

# ---------------------------------------------------------------------------
# Movement (Mecanum Wheel Drive via STM32 + ROS2 cmd_vel)
# ---------------------------------------------------------------------------
DEFAULT_SPEED = int(os.getenv("DEFAULT_SPEED", "30"))
MAX_SPEED = int(os.getenv("MAX_SPEED", "80"))
# Max linear velocity for ROS2 cmd_vel (m/s)
MAX_LINEAR_VEL = float(os.getenv("MAX_LINEAR_VEL", "0.5"))
# Max angular velocity for ROS2 cmd_vel (rad/s)
MAX_ANGULAR_VEL = float(os.getenv("MAX_ANGULAR_VEL", "1.5"))
# Safety: minimum distance (mm) before obstacle-stop
OBSTACLE_STOP_DISTANCE_MM = int(os.getenv("OBSTACLE_STOP_DISTANCE_MM", "300"))

# ---------------------------------------------------------------------------
# Audio / TTS / STT
# ---------------------------------------------------------------------------
TTS_LANG = os.getenv("TTS_LANG", "en")
TTS_TEMP_DIR = os.getenv("TTS_TEMP_DIR", "/tmp/novacare_tts")
STT_LANG = os.getenv("STT_LANG", "en-US")
STT_TIMEOUT = int(os.getenv("STT_TIMEOUT", "10"))  # seconds to listen
STT_PHRASE_TIMEOUT = int(os.getenv("STT_PHRASE_TIMEOUT", "5"))
# JetAuto voice module (xf_mic_asr_offline)
USE_JETAUTO_VOICE = os.getenv("USE_JETAUTO_VOICE", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# LiDAR (A1 / A2 / G4 / S2L / LD14P)
# ---------------------------------------------------------------------------
LIDAR_ENABLED = os.getenv("LIDAR_ENABLED", "true").lower() in ("true", "1", "yes")
LIDAR_MODEL = os.getenv("LIDAR_MODEL", "A1")  # A1 | A2 | G4 | S2L | LD14P

# ---------------------------------------------------------------------------
# On-Robot AI Processing (Jetson GPU)
# ---------------------------------------------------------------------------
ONBOARD_FALL_DETECTION = os.getenv("ONBOARD_FALL_DETECTION", "true").lower() in ("true", "1", "yes")
ONBOARD_PERSON_TRACKING = os.getenv("ONBOARD_PERSON_TRACKING", "true").lower() in ("true", "1", "yes")
ONBOARD_EMOTION_DETECTION = os.getenv("ONBOARD_EMOTION_DETECTION", "false").lower() in ("true", "1", "yes")
# Confidence threshold for fall detection trigger
FALL_DETECTION_THRESHOLD = float(os.getenv("FALL_DETECTION_THRESHOLD", "0.7"))
# Consecutive frames required to confirm a fall
FALL_DETECTION_CONFIRM_FRAMES = int(os.getenv("FALL_DETECTION_CONFIRM_FRAMES", "2"))

# ---------------------------------------------------------------------------
# SLAM / Mapping
# ---------------------------------------------------------------------------
MAP_SAVE_DIR = os.getenv("MAP_SAVE_DIR", "/home/ubuntu/novacare/maps")
DEFAULT_MAP_NAME = os.getenv("DEFAULT_MAP_NAME", "home_map")

# ---------------------------------------------------------------------------
# Destinations (for autonomous navigation)
# Real coordinate poses from SLAM map (x, y, theta in map frame)
# These are overridden at runtime when the user saves destinations
# via the /api/map/destinations endpoint.
# ---------------------------------------------------------------------------
import json

_dest_file = os.path.join(
    os.getenv("MAP_SAVE_DIR", os.path.join(os.path.dirname(__file__), "maps")),
    "destinations.json",
)


def _load_destinations() -> dict:
    """Load saved destinations from disk, fall back to defaults."""
    defaults = {
        "kitchen":  {"x": 2.5,  "y": 0.3, "theta": 0.0,   "label": "Kitchen"},
        "bathroom": {"x": 0.2,  "y": 3.1, "theta": 1.57,  "label": "Bathroom"},
        "living":   {"x": 0.0,  "y": 0.0, "theta": 0.0,   "label": "Living Room"},
        "bedroom":  {"x": -1.8, "y": 2.0, "theta": 3.14,  "label": "Bedroom"},
        "dining":   {"x": 3.0,  "y": 2.5, "theta": 0.78,  "label": "Dining Room"},
        "entrance": {"x": 4.0,  "y": 0.0, "theta": -0.78, "label": "Entrance"},
        "dock":     {"x": 0.0,  "y": 0.0, "theta": 0.0,   "label": "Charging Dock"},
    }
    try:
        if os.path.exists(_dest_file):
            with open(_dest_file, "r") as f:
                saved = json.load(f)
            if saved:
                return saved
    except Exception:
        pass
    return defaults


def save_destinations(destinations: dict) -> bool:
    """Persist destinations to disk."""
    try:
        os.makedirs(os.path.dirname(_dest_file), exist_ok=True)
        with open(_dest_file, "w") as f:
            json.dump(destinations, f, indent=2)
        return True
    except Exception as e:
        print(f"[CONFIG] Failed to save destinations: {e}")
        return False


DESTINATIONS = _load_destinations()

# ---------------------------------------------------------------------------
# SOS / Emergency Configuration
# ---------------------------------------------------------------------------
SOS_ALARM_FREQUENCY = int(os.getenv("SOS_ALARM_FREQUENCY", "880"))   # Hz
SOS_ALARM_DURATION = float(os.getenv("SOS_ALARM_DURATION", "30.0"))  # seconds
SOS_LED_FLASH = os.getenv("SOS_LED_FLASH", "true").lower() in ("true", "1", "yes")
# Firebase Cloud Messaging server key for caregiver push notifications
FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY", "")

# ---------------------------------------------------------------------------
# Network / Connectivity
# ---------------------------------------------------------------------------
# JetAuto default AP IP is 192.168.149.1; set to actual IP in STA LAN mode
JETAUTO_IP = os.getenv("JETAUTO_IP", "192.168.149.1")
