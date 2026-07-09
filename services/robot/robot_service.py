"""
NovaCare Robot REST Service (JetAuto)
======================================
Flask API exposing JetAuto robot hardware capabilities over HTTP.

Port: 9000 (configurable via ROBOT_SERVICE_PORT)

Endpoints
---------
Camera
    GET  /api/camera/frame           base64 JPEG frame
    GET  /api/camera/stream          MJPEG stream (multipart/x-mixed-replace)
    POST /api/camera/session/start   register viewer, warm up capture pipeline
    POST /api/camera/session/stop    unregister viewer
    GET  /api/camera/status          backend + session status
    GET  /api/depth/frame            base64 depth frame

Movement
    POST /api/move                   body: {"direction": str, "speed": int, "duration": float}
    POST /api/move/stop              stop all movement

Navigation (Nav2 Autonomous)
    POST /api/navigate               body: {"destination": str} OR {"x": float, "y": float, "theta": float}
    GET  /api/navigation/status      current Nav2 status, pose, ETA
    POST /api/navigation/cancel      cancel active Nav2 goal
    POST /api/navigation/waypoints   body: {"waypoints": [{"x","y","theta"}...]}
    GET  /api/navigation/pose        current robot pose from odometry

Follow User
    POST /api/follow/start           start person-following mode
    POST /api/follow/stop            stop person-following mode

Audio
    POST /api/tts/speak              body: {"text": str, "lang": str}
    POST /api/play_audio             body: {"name": str, "audio_base64": str, "mime": str}
    POST /api/stt/listen             listen for speech, return text
    GET  /api/stt/status             check STT availability

LiDAR
    GET  /api/lidar/scan             full scan data
    GET  /api/lidar/obstacle         obstacle-ahead check

Map / SLAM
    GET  /api/map/current            current SLAM map as PNG image
    POST /api/map/save               save current SLAM map
    POST /api/map/load               load a named map
    GET  /api/map/destinations       list saved destinations
    POST /api/map/destinations       save/update a destination

SOS / Emergency
    POST /api/sos/trigger            trigger full SOS sequence
    POST /api/sos/cancel             cancel SOS alarm

Fall Detection
    POST /api/fall-detection/start   start on-robot fall detection
    POST /api/fall-detection/stop    stop on-robot fall detection
    GET  /api/fall-detection/status  fall detection state

Vitals & Health
    GET  /api/vitals/heart-rate      latest heart rate from smart watch
    GET  /api/vitals/current         all current vitals (HR, steps, battery)
    GET  /health                     service health + hardware status + vitals
"""

import os
import sys
import time
import signal
import threading
import tempfile
import base64
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, Response, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from config import (
    ROBOT_SERVICE_HOST, ROBOT_SERVICE_PORT, DESTINATIONS,
    DEFAULT_SPEED, OBSTACLE_STOP_DISTANCE_MM,
    STREAM_FPS, MINIMAL_MODE, ROBOT_NAME, ROBOT_TYPE,
    ONBOARD_FALL_DETECTION, MAP_SAVE_DIR, DEFAULT_MAP_NAME,
    SOS_ALARM_FREQUENCY, SOS_ALARM_DURATION, FCM_SERVER_KEY,
    save_destinations,
)
from robot_hal import get_robot

try:
    from watch_integration import (
        init_watch_integration,
        get_watch_manager,
        get_current_vitals,
    )
    _WATCH_AVAILABLE = True
except ImportError:
    _WATCH_AVAILABLE = False
    print("[WARN] watch_integration not available")

    class _FakeVitals:
        heart_rate = None
        timestamp = None
        def to_dict(self):
            return {"heart_rate": None, "steps": None, "battery": None}

    def get_current_vitals():
        return _FakeVitals()

    def init_watch_integration(**kwargs):
        pass

    def get_watch_manager():
        return None

app = Flask(__name__)
CORS(app)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# ---------------------------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("NOVACARE_API_KEY", "novacare-secure-key-2026")

@app.before_request
def check_api_key():
    """Ensure all API endpoints are authenticated."""
    if request.method == "OPTIONS":
        return
    if request.path in ["/health", "/", "/ui", "/RobotUI.css", "/static/RobotUI.css", "/optimized_runtime/robot_ui/RobotUI.css"]:
        return

    key = request.headers.get("X-API-Key")
    if key != API_KEY and request.args.get("api_key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

# ---------------------------------------------------------------------------
# Robot UI Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
@app.get("/ui")
def robot_ui():
    """Serve the active robot face UI."""
    try:
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_robot_ui.html")
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    except Exception as e:
        return f"Error loading Robot UI: {e}", 500

@app.get("/RobotUI.css")
@app.get("/static/RobotUI.css")
@app.get("/optimized_runtime/robot_ui/RobotUI.css")
def robot_ui_css():
    """Serve robot face stylesheet (using the root RobotUI.css)."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return send_from_directory(base_dir, "RobotUI.css", mimetype="text/css")
    except Exception as e:
        return f"Error loading Robot UI Stylesheet: {e}", 500

# ---------------------------------------------------------------------------
# Robot & AI instances (lazy init on first request)
# ---------------------------------------------------------------------------
_robot = None
_ai_service = None

def robot():
    global _robot
    if _robot is None:
        _robot = get_robot()
    return _robot

def ai_service():
    global _ai_service
    if _ai_service is None:
        from onboard_ai import OnboardAIService
        _ai_service = OnboardAIService(
            camera_read_fn=lambda: robot().camera.read_frame(),
            alarm_fn=lambda f, d: robot().audio.play_alarm(f, d),
            speak_fn=lambda t, b: robot().audio.speak(t, block=b),
            notify_fn=lambda t, s, d: None # Simplification for now
        )
    return _ai_service

# ============================================================================
# Camera Endpoints
# ============================================================================

@app.get("/api/camera/frame")
def camera_frame():
    """Return a single camera frame as base64 JPEG."""
    b64 = robot().camera.read_frame_base64(quality=80)
    if b64 is None:
        return jsonify({"error": "Camera not available or no frame captured"}), 503
    return jsonify({"image": b64, "status": "success"})


@app.get("/api/camera/stream")
def camera_stream():
    """MJPEG streaming endpoint for live video feed."""
    def generate():
        frame_interval = 1.0 / max(1, STREAM_FPS)
        while True:
            jpg = robot().camera.read_frame_jpeg_bytes(quality=70)
            if jpg is None:
                time.sleep(0.1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(frame_interval)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/camera/session/start")
def camera_session_start():
    """Register a mobile/web viewer and warm up the camera pipeline."""
    cam = robot().camera
    if not cam.is_available:
        return jsonify({"error": "Camera not available"}), 503
    if not cam.start_session():
        return jsonify({"error": "Failed to start camera session"}), 503
    status = cam.get_status()
    return jsonify({
        "status": "success",
        "stream_url": "/api/camera/stream",
        "resolution": status.get("stream_resolution"),
        "backend": status.get("backend"),
    })


@app.post("/api/camera/session/stop")
def camera_session_stop():
    """Unregister a viewer session."""
    robot().camera.stop_session()
    return jsonify({"status": "success"})


@app.get("/api/camera/status")
def camera_status():
    """Return camera backend and session status."""
    return jsonify({"status": "success", **robot().camera.get_status()})


@app.get("/api/depth/frame")
def depth_frame():
    """Return a depth camera frame as base64."""
    cam = robot().camera
    depth_val = cam.get_depth_at(320, 240)  # center point sample
    b64 = cam.read_frame_base64(quality=70)
    return jsonify({
        "status": "success",
        "image": b64,
        "center_depth_mm": depth_val,
    })

# ============================================================================
# Gesture Endpoints
# ============================================================================

@app.post("/api/gestures/start")
def start_gestures():
    """Start the gesture recognition AI loop."""
    ok = ai_service().start_gesture_control()
    if ok:
        return jsonify({"status": "success", "message": "Gesture control started"})
    return jsonify({"status": "error", "message": "Failed to start gesture control"}), 500

@app.post("/api/gestures/stop")
def stop_gestures():
    """Stop the gesture recognition AI loop."""
    ai_service().stop_gesture_control()
    return jsonify({"status": "success", "message": "Gesture control stopped"})

@app.get("/api/gestures/response")
def get_gesture_response():
    """Poll for the user's latest Yes/No confirmation gesture."""
    val = ai_service().get_last_binary_response()
    return jsonify({"status": "success", "binary_response": val})


# ============================================================================
# Lidar Guarding Endpoints
# ============================================================================

@app.post("/api/guarding/start")
def api_start_guarding():
    """Start Lidar Guarding mode."""
    ok = ai_service().start_guarding()
    if ok:
        return jsonify({"status": "success", "message": "Guarding mode started"})
    return jsonify({"status": "error", "message": "Failed to start guarding mode"}), 500

@app.post("/api/guarding/stop")
def api_stop_guarding():
    """Stop Lidar Guarding mode."""
    ai_service().stop_guarding()
    return jsonify({"status": "success", "message": "Guarding mode stopped"})


# ============================================================================
# ASL Recognition Endpoints
# ============================================================================

@app.post("/api/asl/start")
def start_asl():
    """Start ASL Recognition mode."""
    ok = ai_service().start_asl_recognition()
    if ok:
        return jsonify({"status": "success", "message": "ASL recognition started"})
    return jsonify({"status": "error", "message": "Failed to start ASL recognition"}), 500

@app.post("/api/asl/stop")
def stop_asl():
    """Stop ASL Recognition mode."""
    ai_service().stop_asl_recognition()
    return jsonify({"status": "success", "message": "ASL recognition stopped"})

@app.get("/api/asl/status")
def get_asl_status():
    """Get the latest ASL prediction."""
    pred = ai_service().current_asl_prediction
    if pred:
        return jsonify({"status": "success", "prediction": pred})
    return jsonify({"status": "success", "prediction": None})


# ============================================================================
# Movement Endpoints
# ============================================================================

DIRECTION_MAP = {
    "forward":    0,
    "backward":   180,
    "left":       270,
    "right":      90,
    "turn_left":  -1,   # special handling
    "turn_right": -2,   # special handling
}


@app.post("/api/move")
def move():
    """
    Move the robot in a direction.

    Body: {"direction": "forward"|"backward"|"left"|"right"|"turn_left"|"turn_right",
           "speed": int (0-80), "duration": float (seconds, 0=indefinite)}
    """
    data = request.get_json(silent=True) or {}
    direction = data.get("direction", "").lower()
    speed = data.get("speed", DEFAULT_SPEED)
    duration = data.get("duration", 0)

    if direction not in DIRECTION_MAP:
        return jsonify({"error": f"Unknown direction: {direction}",
                        "valid": list(DIRECTION_MAP.keys())}), 400

    # LiDAR-based obstacle check for forward movement
    if direction == "forward" and robot().lidar.is_obstacle_ahead():
        robot().motion.stop()
        return jsonify({
            "status": "blocked",
            "message": "Obstacle detected ahead — movement stopped for safety",
        }), 200

    angle = DIRECTION_MAP[direction]

    if angle == -1:
        if duration > 0:
            robot().motion.turn_left(speed)
            time.sleep(duration)
            robot().motion.stop()
        else:
            robot().motion.turn_left(speed)
    elif angle == -2:
        if duration > 0:
            robot().motion.turn_right(speed)
            time.sleep(duration)
            robot().motion.stop()
        else:
            robot().motion.turn_right(speed)
    elif duration > 0:
        robot().motion.move_for(angle, duration, speed)
    else:
        robot().motion.move(angle, speed)

    return jsonify({
        "status": "moving",
        "direction": direction,
        "speed": speed,
        "duration": duration,
    })


@app.post("/api/move/stop")
def move_stop():
    """Stop all movement immediately."""
    robot().motion.stop()
    return jsonify({"status": "stopped"})


# ============================================================================
# Autonomous Navigation Endpoints (Nav2)
# ============================================================================

@app.post("/api/navigate")
def navigate():
    """
    Navigate to a destination using JetAuto's Nav2 autonomous navigation.

    Body: {"destination": "kitchen"|"bathroom"|...}
    OR:   {"x": float, "y": float, "theta": float}
    """
    data = request.get_json(silent=True) or {}
    dest_name = data.get("destination", "").lower() if data.get("destination") else None
    x = data.get("x")
    y = data.get("y")
    theta = data.get("theta", 0.0)

    if dest_name:
        if dest_name not in DESTINATIONS:
            return jsonify({
                "error": f"Unknown destination: {dest_name}",
                "valid": list(DESTINATIONS.keys()),
            }), 400

        dest = DESTINATIONS[dest_name]
        x = dest["x"]
        y = dest["y"]
        theta = dest.get("theta", 0.0)
        label = dest.get("label", dest_name)
    elif x is not None and y is not None:
        label = f"({x:.1f}, {y:.1f})"
    else:
        return jsonify({
            "error": "Provide 'destination' name or 'x','y','theta' coordinates",
            "valid_destinations": list(DESTINATIONS.keys()),
        }), 400

    success = robot().motion.navigate_to(x=x, y=y, theta=theta)
    if success:
        robot().audio.speak(f"Navigating to {label}.", block=False)

    return jsonify({
        "status": "navigating" if success else "failed",
        "destination": dest_name or label,
        "x": x,
        "y": y,
        "theta": theta,
        "message": f"Nav2 autonomous navigation to {label}" if success else "Navigation failed to start",
    })


@app.get("/api/navigation/status")
def navigation_status():
    """Return current Nav2 navigation status, pose, and ETA."""
    nav = robot().get_navigation_status()
    pose = robot().get_current_pose()
    return jsonify({
        "status": "success",
        "navigation": nav,
        "current_pose": pose,
    })


@app.post("/api/navigation/cancel")
def navigation_cancel():
    """Cancel the active Nav2 navigation goal."""
    success = robot().motion.cancel_navigation()
    robot().motion.stop()
    return jsonify({
        "status": "cancelled" if success else "no_active_goal",
    })


@app.post("/api/navigation/waypoints")
def navigation_waypoints():
    """
    Navigate through multiple waypoints (Nav2 through-poses).

    Body: {"waypoints": [{"x": float, "y": float, "theta": float}, ...]}
    """
    data = request.get_json(silent=True) or {}
    waypoints = data.get("waypoints", [])

    if not waypoints:
        return jsonify({"error": "No waypoints provided"}), 400

    # Navigate to each waypoint sequentially via Nav2
    # For now, navigate to the first waypoint; multi-point will be chained
    first = waypoints[0]
    success = robot().motion.navigate_to(
        x=first.get("x", 0), y=first.get("y", 0), theta=first.get("theta", 0)
    )

    return jsonify({
        "status": "navigating" if success else "failed",
        "total_waypoints": len(waypoints),
        "current_waypoint": 0,
    })


@app.get("/api/navigation/pose")
def navigation_pose():
    """Return current robot pose from odometry."""
    pose = robot().get_current_pose()
    if pose is None:
        return jsonify({"status": "unavailable", "pose": None}), 503
    return jsonify({"status": "success", "pose": pose})


# ============================================================================
# Follow User Endpoints
# ============================================================================

@app.post("/api/follow/start")
def follow_start():
    """Start person-following mode using depth camera + LiDAR tracking."""
    if MINIMAL_MODE:
        return jsonify({"error": "Follow mode disabled in minimal robot mode"}), 501
    robot().motion.start_tracking("person")
    return jsonify({
        "status": "following",
        "message": "Person tracking active — robot will follow the nearest person",
    })


@app.post("/api/follow/stop")
def follow_stop():
    """Stop person-following mode."""
    robot().motion.stop_tracking()
    return jsonify({"status": "stopped"})


# ============================================================================
# Audio / TTS / STT Endpoints
# ============================================================================

@app.post("/api/tts/speak")
def tts_speak():
    """
    Speak text on the robot speaker.

    Body: {"text": str, "lang": str (default "en")}
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    lang = data.get("lang", "en")
    success = robot().audio.speak(text, lang=lang, block=False)
    return jsonify({"status": "speaking" if success else "tts_unavailable", "text": text})


@app.post("/api/play_audio")
def play_audio():
    """Accept base64 audio and play it on the robot speaker.

    Body: {"name": str, "audio_base64": str, "mime": str}
    """
    data = request.get_json(silent=True) or {}
    audio_b64 = data.get("audio_base64")
    name = data.get("name", f"phone_audio_{int(time.time())}")
    mime = data.get("mime", "audio/mpeg")

    if not audio_b64:
        return jsonify({"error": "audio_base64 required"}), 400

    try:
        raw = base64.b64decode(audio_b64)
    except Exception as e:
        return jsonify({"error": f"invalid base64: {e}"}), 400

    try:
        tf = os.path.join(tempfile.gettempdir(), f"{name}")
        ext = ".mp3" if "mpeg" in mime or "mp3" in mime else ".wav"
        tf = tf + ext
        with open(tf, "wb") as f:
            f.write(raw)

        t = threading.Thread(target=robot().audio._play_file, args=(tf,), daemon=True)
        t.start()
        return jsonify({"status": "playing", "file": os.path.basename(tf)})
    except Exception as e:
        return jsonify({"error": f"playback failed: {e}"}), 500


@app.post("/api/stt/listen")
def stt_listen():
    """
    Listen for speech and return recognised text.

    Body (optional): {"timeout": int, "phrase_timeout": int}
    """
    data = request.get_json(silent=True) or {}
    timeout = data.get("timeout")
    phrase_timeout = data.get("phrase_timeout")

    text = robot().audio.listen(timeout=timeout, phrase_timeout=phrase_timeout)
    if text is None:
        return jsonify({"status": "no_speech", "text": None})
    return jsonify({"status": "success", "text": text})


@app.get("/api/stt/status")
def stt_status():
    """Check STT availability."""
    return jsonify({
        "stt_available": robot().audio.stt_available,
        "tts_available": robot().audio.tts_available,
    })


# ============================================================================
# LiDAR Endpoints
# ============================================================================

@app.get("/api/lidar/scan")
def lidar_scan():
    """Return full LiDAR scan data."""
    scan = robot().lidar.get_scan()
    return jsonify({"points": scan, "count": len(scan)})


@app.get("/api/lidar/obstacle")
def lidar_obstacle():
    """Check if there's an obstacle ahead."""
    blocked = robot().lidar.is_obstacle_ahead()
    closest_angle, closest_dist = robot().lidar.get_closest_obstacle()
    return jsonify({
        "obstacle_ahead": blocked,
        "closest_obstacle": {
            "angle": closest_angle,
            "distance_mm": closest_dist,
        },
    })


# ============================================================================
# Map / SLAM Endpoints
# ============================================================================

@app.get("/api/map/current")
def map_current():
    """Return current SLAM map as PNG image."""
    png_bytes = robot().get_map_image()
    if png_bytes is None:
        return jsonify({"error": "No map available"}), 503

    import io
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        download_name="novacare_map.png",
    )


@app.get("/api/map/current/json")
def map_current_json():
    """Return current SLAM map metadata as JSON."""
    from robot_hal import _get_ros2_node
    _, map_info = _get_ros2_node().get_map_data()
    if map_info is None:
        return jsonify({"error": "No map available"}), 503

    pose = robot().get_current_pose()
    return jsonify({
        "status": "success",
        "map_info": map_info,
        "robot_pose": pose,
    })


@app.post("/api/map/save")
def map_save():
    """Save current SLAM map to disk."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", DEFAULT_MAP_NAME)

    os.makedirs(MAP_SAVE_DIR, exist_ok=True)
    png = robot().get_map_image()
    if png:
        path = os.path.join(MAP_SAVE_DIR, f"{name}.png")
        with open(path, "wb") as f:
            f.write(png)
        return jsonify({"status": "saved", "name": name, "path": path})

    return jsonify({"error": "No map to save"}), 503


@app.post("/api/map/load")
def map_load():
    """Load a named SLAM map for navigation."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", DEFAULT_MAP_NAME)

    path = os.path.join(MAP_SAVE_DIR, f"{name}.png")
    if not os.path.exists(path):
        return jsonify({"error": f"Map '{name}' not found"}), 404

    return jsonify({"status": "loaded", "name": name})


@app.get("/api/map/destinations")
def map_destinations():
    """List all saved navigation destinations."""
    return jsonify({
        "status": "success",
        "destinations": DESTINATIONS,
    })


@app.post("/api/map/destinations")
def map_destinations_save():
    """Save or update a navigation destination.

    Body: {"name": str, "x": float, "y": float, "theta": float, "label": str}
    """
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").lower()
    if not name:
        return jsonify({"error": "name is required"}), 400

    DESTINATIONS[name] = {
        "x": data.get("x", 0.0),
        "y": data.get("y", 0.0),
        "theta": data.get("theta", 0.0),
        "label": data.get("label", name.title()),
    }

    # Also save current robot pose as the destination if no coordinates given
    if data.get("x") is None:
        pose = robot().get_current_pose()
        if pose:
            DESTINATIONS[name]["x"] = pose["x"]
            DESTINATIONS[name]["y"] = pose["y"]
            DESTINATIONS[name]["theta"] = pose["yaw"]

    save_destinations(DESTINATIONS)
    return jsonify({
        "status": "saved",
        "destination": DESTINATIONS[name],
    })


# ============================================================================
# SOS / Emergency Endpoints
# ============================================================================

_sos_active = False
_sos_thread = None


@app.post("/api/sos/trigger")
def sos_trigger():
    """
    Trigger full SOS emergency sequence:
    1. Sound alarm on robot speaker
    2. Flash LEDs (if available)
    3. Announce emergency via TTS
    4. Send push notifications to caregivers via Firebase
    5. Record event with timestamp and vitals
    """
    global _sos_active, _sos_thread

    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "unknown")
    location = data.get("location", "unknown")

    _sos_active = True
    r = robot()

    # 1. Sound alarm
    r.audio.play_alarm(SOS_ALARM_FREQUENCY, SOS_ALARM_DURATION)

    # 2. Announce emergency
    r.audio.speak("Emergency alert activated! Help is on the way!", block=False)

    # 3. Collect vitals for the emergency record
    vitals = get_current_vitals()
    vitals_dict = vitals.to_dict() if vitals else {}

    # 4. Send Firebase push notification to caregivers
    notification_sent = _send_sos_notification(user_id, location, vitals_dict)

    # 5. Record event
    sos_event = {
        "timestamp": time.time(),
        "user_id": user_id,
        "location": location,
        "vitals": vitals_dict,
        "notification_sent": notification_sent,
    }

    return jsonify({
        "status": "sos_active",
        "message": "Emergency sequence activated",
        "alarm_active": True,
        "notification_sent": notification_sent,
        "event": sos_event,
    })


@app.post("/api/sos/cancel")
def sos_cancel():
    """Cancel the SOS alarm."""
    global _sos_active
    _sos_active = False
    robot().audio.stop_alarm()
    robot().audio.speak("Emergency alert cancelled.", block=False)
    return jsonify({"status": "cancelled"})


def _send_sos_notification(user_id: str, location: str, vitals: dict) -> bool:
    """Send push notification to caregivers via Firebase Cloud Messaging."""
    if not FCM_SERVER_KEY:
        print("[SOS] No FCM_SERVER_KEY configured — skipping push notification")
        return False

    try:
        import urllib.request
        import urllib.error

        payload = {
            "to": "/topics/novacare_caregivers",
            "notification": {
                "title": "🚨 NovaCare SOS Emergency",
                "body": f"Emergency alert from {user_id} at {location}. Heart rate: {vitals.get('heart_rate', 'N/A')}",
            },
            "data": {
                "type": "sos_emergency",
                "user_id": user_id,
                "location": location,
                "timestamp": str(time.time()),
                "vitals": json.dumps(vitals),
            },
            "priority": "high",
        }

        req = urllib.request.Request(
            "https://fcm.googleapis.com/fcm/send",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"key={FCM_SERVER_KEY}",
            },
        )
        resp = urllib.request.urlopen(req, timeout=5)
        print(f"[SOS] FCM notification sent: {resp.status}")
        return resp.status == 200
    except Exception as e:
        print(f"[SOS] FCM notification failed: {e}")
        return False


# ============================================================================
# Fall Detection Endpoints
# ============================================================================

_fall_detection_active = False
_fall_detection_thread = None
_last_fall_event = None


@app.post("/api/fall-detection/start")
def fall_detection_start():
    """Start on-robot fall detection pipeline."""
    global _fall_detection_active, _fall_detection_thread

    if _fall_detection_active:
        return jsonify({"status": "already_active"})

    _fall_detection_active = True

    def _run_detection():
        global _fall_detection_active, _last_fall_event
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fall-detection"))
            from fall_detection import FallDetector
            detector = FallDetector()
            print("[FALL] On-robot fall detection started (GPU-accelerated)")

            while _fall_detection_active:
                ok, frame = robot().camera.read_frame()
                if not ok or frame is None:
                    time.sleep(0.1)
                    continue

                result = detector.analyze_frame(frame)
                if result.get("fall_detected"):
                    print(f"[FALL] FALL DETECTED! Confidence: {result['confidence']:.2f}")
                    _last_fall_event = {
                        "timestamp": time.time(),
                        "confidence": result["confidence"],
                        "method": result["method"],
                        "message": result["message"],
                    }
                    # Auto-trigger SOS
                    robot().audio.play_alarm(SOS_ALARM_FREQUENCY, 10)
                    robot().audio.speak("Fall detected! Are you okay? Alerting caregivers.", block=False)
                    _send_sos_notification("fall_detection", "robot_camera", {
                        "fall_confidence": result["confidence"],
                        "method": result["method"],
                    })

                time.sleep(0.1)  # ~10 fps analysis

        except Exception as e:
            print(f"[FALL] Detection error: {e}")
        finally:
            _fall_detection_active = False
            print("[FALL] Fall detection stopped")

    _fall_detection_thread = threading.Thread(target=_run_detection, daemon=True)
    _fall_detection_thread.start()

    return jsonify({
        "status": "started",
        "message": "Fall detection active — monitoring camera feed",
    })


@app.post("/api/fall-detection/stop")
def fall_detection_stop():
    """Stop on-robot fall detection pipeline."""
    global _fall_detection_active
    _fall_detection_active = False
    return jsonify({"status": "stopped"})


@app.get("/api/fall-detection/status")
def fall_detection_status():
    """Return fall detection state and last event."""
    return jsonify({
        "active": _fall_detection_active,
        "last_event": _last_fall_event,
        "onboard_gpu": ONBOARD_FALL_DETECTION,
    })


# ============================================================================
# Vitals & Health
# ============================================================================

@app.get("/api/vitals/heart-rate")
def get_heart_rate():
    """Get latest heart rate from smart watch."""
    vitals = get_current_vitals()
    if vitals.heart_rate is None:
        return jsonify({
            "status": "unavailable",
            "heart_rate": None,
            "message": "Heart rate data not available"
        }), 503

    return jsonify({
        "status": "success",
        "heart_rate": vitals.heart_rate,
        "timestamp": vitals.timestamp.isoformat() if vitals.timestamp else None,
    })


@app.get("/api/vitals/current")
def get_all_vitals():
    """Get all current vitals from smart watch."""
    vitals = get_current_vitals()
    return jsonify({
        "status": "success",
        **vitals.to_dict()
    })


@app.get("/health")
def health():
    """Service health check with hardware status."""
    r = robot()
    vitals = get_current_vitals()
    nav_status = r.get_navigation_status()
    pose = r.get_current_pose()

    return jsonify({
        "status": "healthy",
        "service": "NovaCare Robot Service",
        "robot": {
            "name": ROBOT_NAME,
            "type": ROBOT_TYPE,
            "platform": "JetAuto",
        },
        "hardware": {
            "camera": r.camera.is_available,
            "camera_status": r.camera.get_status(),
            "motion": r.motion.is_available,
            "tts": r.audio.tts_available,
            "stt": r.audio.stt_available,
            "lidar": r.lidar.is_available,
            "moving": r.motion.is_moving,
        },
        "navigation": nav_status,
        "pose": pose,
        "vitals": vitals.to_dict() if vitals else None,
        "fall_detection": {
            "active": _fall_detection_active,
            "last_event": _last_fall_event,
        },
        "sos_active": _sos_active,
    })


# ============================================================================
# Main
# ============================================================================

def _cleanup(signum, frame):
    """Handle SIGINT/SIGTERM — stop robot gracefully."""
    if _WATCH_AVAILABLE:
        watch_mgr = get_watch_manager()
        if watch_mgr:
            watch_mgr.stop()
    if _robot:
        _robot.shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, _cleanup)
signal.signal(signal.SIGTERM, _cleanup)


if __name__ == "__main__":
    print("=" * 50)
    print(f"  NovaCare JetAuto Robot REST Service")
    print(f"  Robot: {ROBOT_NAME} ({ROBOT_TYPE})")
    print(f"  Listening on {ROBOT_SERVICE_HOST}:{ROBOT_SERVICE_PORT}")
    if MINIMAL_MODE:
        print("  Mode: MINIMAL (I/O bridge — no watch, no AI)")
    print("=" * 50)

    if MINIMAL_MODE:
        print("[OK] Skipping watch integration (NOVACARE_MINIMAL=1)")
    elif _WATCH_AVAILABLE:
        watch_address = os.getenv("WATCH_ADDRESS", "C2:FC:28:B7:1C:1B")
        simulation_mode = os.getenv("WATCH_SIMULATION", "true").lower() == "true"
        print(f"\n Initializing watch integration (simulation={simulation_mode})...")
        init_watch_integration(device_address=watch_address, simulation_mode=simulation_mode)
        watch_mgr = get_watch_manager()
        if watch_mgr:
            watch_mgr.start()
            print(" Watch monitoring started\n")

    # Auto-start fall detection if configured
    if ONBOARD_FALL_DETECTION and not MINIMAL_MODE:
        print("[OK] Auto-starting on-robot fall detection...")
        with app.test_request_context():
            fall_detection_start()

    app.run(host=ROBOT_SERVICE_HOST, port=ROBOT_SERVICE_PORT, threaded=True)
