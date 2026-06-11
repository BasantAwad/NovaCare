"""
NovaCare Robot  REST Service
==============================
Flask API exposing robot hardware capabilities over HTTP.

Port: 9000 (configurable via ROBOT_SERVICE_PORT)

Endpoints
---------
Camera
    GET  /api/camera/frame           base64 JPEG frame
    GET  /api/camera/stream          MJPEG stream (multipart/x-mixed-replace)

Movement
    POST /api/move                   body: {"direction": str, "speed": int, "duration": float}
    POST /api/move/stop              stop all movement
    POST /api/navigate               body: {"destination": str}
    POST /api/follow/start           start follow-user mode
    POST /api/follow/stop            stop follow-user mode

Audio
    POST /api/tts/speak              body: {"text": str, "lang": str}
    POST /api/stt/listen             listen for speech, return text
    GET  /api/stt/status             check STT availability

LiDAR
    GET  /api/lidar/scan             full scan data
    GET  /api/lidar/obstacle         obstacle-ahead check

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

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from config import (
    ROBOT_SERVICE_HOST, ROBOT_SERVICE_PORT, DESTINATIONS,
    DEFAULT_SPEED, OBSTACLE_STOP_DISTANCE_MM,
)
from robot_hal import get_robot
from watch_integration import (
    init_watch_integration,
    get_watch_manager,
    get_current_vitals,
)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("NOVACARE_API_KEY", "novacare-secure-key-2026")

@app.before_request
def check_api_key():
    """Ensure all API endpoints are authenticated."""
    if request.method == "OPTIONS":
        return
    if request.path in ["/health", "/", "/ui", "/optimized_runtime/robot_ui/RobotUI.css"]:
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

@app.get("/optimized_runtime/robot_ui/RobotUI.css")
def robot_ui_css():
    """Serve the styles for the robot face UI."""
    try:
        # Dynamically resolve optimized_runtime/robot_ui/RobotUI.css
        # Works with flat 'robot/' and nested 'services/robot/' structures
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(curr_dir)
        css_path = os.path.join(base_dir, "optimized_runtime", "robot_ui", "RobotUI.css")
        if not os.path.exists(css_path):
            parent_of_parent = os.path.dirname(base_dir)
            css_path = os.path.join(parent_of_parent, "optimized_runtime", "robot_ui", "RobotUI.css")
            
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/css"}
    except Exception as e:
        return f"Error loading Robot UI Stylesheet: {e}", 500

# ---------------------------------------------------------------------------
# Robot instance (lazy init on first request)
# ---------------------------------------------------------------------------
_robot = None


def robot():
    global _robot
    if _robot is None:
        _robot = get_robot()
    return _robot


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
        while True:
            jpg = robot().camera.read_frame_jpeg_bytes(quality=70)
            if jpg is None:
                time.sleep(0.1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(1.0 / 15)  # ~15 FPS

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


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
    direction = data.get("direction")
    speed = data.get("speed", DEFAULT_SPEED)
    duration = data.get("duration", 0)

    angle = None
    if isinstance(direction, int):
        angle = direction
    elif isinstance(direction, str) and direction.strip().isdigit():
        angle = int(direction.strip())
    elif isinstance(direction, str) and direction.lower() in DIRECTION_MAP:
        angle = DIRECTION_MAP[direction.lower()]

    if angle is None:
        return jsonify({"error": f"Unknown direction: {direction}",
                        "valid": list(DIRECTION_MAP.keys()) + ["integer angle"]}), 400


    # Check for obstacles before moving forward
    if direction == "forward" and robot().camera.is_obstacle_ahead():
        robot().motion.stop()
        return jsonify({
            "status": "blocked",
            "message": "Obstacle detected ahead  movement stopped for safety",
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


@app.post("/api/navigate")
def navigate():
    """
    Navigate to a predefined destination using SERBot's built-in SLAM/navigation.

    Body: {"destination": "kitchen"|"bathroom"|...}
    """
    data = request.get_json(silent=True) or {}
    dest_name = data.get("destination", "").lower()

    if dest_name not in DESTINATIONS:
        return jsonify({"error": f"Unknown destination: {dest_name}",
                        "valid": list(DESTINATIONS.keys())}), 400

    robot().motion.navigate_to(dest_name)
    robot().audio.speak(f"Navigating to {dest_name}.", block=False)

    return jsonify({
        "status": "navigating",
        "destination": dest_name,
        "message": "Handed off to SERBot built-in navigation subsystem."
    })


# ---------------------------------------------------------------------------
# Follow User
# ---------------------------------------------------------------------------

@app.post("/api/follow/start")
def follow_start():
    """Start follow-user mode (using SERBot's built-in tracking)."""
    robot().motion.start_tracking("face")
    return jsonify({"status": "following", "message": "Using SERBot built-in tracking"})

@app.post("/api/follow/stop")
def follow_stop():
    """Stop follow-user mode."""
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
    # Speak asynchronously so the response returns immediately
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

    import base64
    try:
        raw = base64.b64decode(audio_b64)
    except Exception as e:
        return jsonify({"error": f"invalid base64: {e}"}), 400

    # Write to temp file and play
    try:
        tf = os.path.join(tempfile.gettempdir(), f"{name}")
        # choose extension heuristically
        ext = ".mp3" if "mpeg" in mime or "mp3" in mime else ".wav"
        tf = tf + ext
        with open(tf, "wb") as f:
            f.write(raw)

        # Play asynchronously so client gets immediate response
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
    return jsonify({"obstacle_ahead": blocked})


# ============================================================================
# Health / System
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
    return jsonify({
        "status": "healthy",
        "service": "NovaCare Robot Service",
        "hardware": {
            "camera": r.camera.is_available,
            "motion": r.motion.is_available,
            "tts": r.audio.tts_available,
            "stt": r.audio.stt_available,
            "lidar": r.lidar.is_available,
            "moving": r.motion.is_moving,
        },
        "vitals": vitals.to_dict() if vitals else None,
    })


# ============================================================================
# Main
# ============================================================================

def _cleanup(signum, frame):
    """Handle SIGINT/SIGTERM  stop robot gracefully."""
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
    print("  NovaCare  Robot REST Service")
    print(f"  Listening on {ROBOT_SERVICE_HOST}:{ROBOT_SERVICE_PORT}")
    print("=" * 50)
    
    # Initialize watch integration in simulation mode by default
    # (set simulation_mode=False to connect to real HRYFINE watch)
    watch_address = os.getenv("WATCH_ADDRESS", "C2:FC:28:B7:1C:1B")
    simulation_mode = os.getenv("WATCH_SIMULATION", "true").lower() == "true"
    
    print(f"\n Initializing watch integration (simulation={simulation_mode})...")
    init_watch_integration(device_address=watch_address, simulation_mode=simulation_mode)
    
    watch_mgr = get_watch_manager()
    if watch_mgr:
        watch_mgr.start()
        print(" Watch monitoring started\n")
    
    app.run(host=ROBOT_SERVICE_HOST, port=ROBOT_SERVICE_PORT, threaded=True)
