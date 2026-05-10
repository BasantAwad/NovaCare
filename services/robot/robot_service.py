"""
NovaCare Robot — REST Service
==============================
Flask API exposing robot hardware capabilities over HTTP.

Port: 9000 (configurable via ROBOT_SERVICE_PORT)

Endpoints
---------
Camera
    GET  /api/camera/frame          → base64 JPEG frame
    GET  /api/camera/stream         → MJPEG stream (multipart/x-mixed-replace)

Movement
    POST /api/move                  → body: {"direction": str, "speed": int, "duration": float}
    POST /api/move/stop             → stop all movement
    POST /api/navigate              → body: {"destination": str}
    POST /api/follow/start          → start follow-user mode
    POST /api/follow/stop           → stop follow-user mode

Audio
    POST /api/tts/speak             → body: {"text": str, "lang": str}
    POST /api/stt/listen            → listen for speech, return text
    GET  /api/stt/status            → check STT availability

LiDAR
    GET  /api/lidar/scan            → full scan data
    GET  /api/lidar/obstacle        → obstacle-ahead check

System
    GET  /health                    → service health + hardware status
"""

import os
import sys
import time
import signal
import threading

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from config import (
    ROBOT_SERVICE_HOST, ROBOT_SERVICE_PORT, DESTINATIONS,
    DEFAULT_SPEED, OBSTACLE_STOP_DISTANCE_MM,
)
from robot_hal import get_robot

app = Flask(__name__)
CORS(app)

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
    direction = data.get("direction", "").lower()
    speed = data.get("speed", DEFAULT_SPEED)
    duration = data.get("duration", 0)

    if direction not in DIRECTION_MAP:
        return jsonify({"error": f"Unknown direction: {direction}",
                        "valid": list(DIRECTION_MAP.keys())}), 400

    # Check for obstacles before moving forward
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


@app.post("/api/navigate")
def navigate():
    """
    Navigate to a predefined destination.

    Body: {"destination": "kitchen"|"bathroom"|...}
    """
    data = request.get_json(silent=True) or {}
    dest_name = data.get("destination", "").lower()

    if dest_name not in DESTINATIONS:
        return jsonify({"error": f"Unknown destination: {dest_name}",
                        "valid": list(DESTINATIONS.keys())}), 400

    dest = DESTINATIONS[dest_name]
    angle = dest["angle"]
    distance_cm = dest["distance_cm"]
    # Rough estimate: ~10 cm/s at default speed
    duration = distance_cm / 10.0

    # Run navigation in background thread so it doesn't block the response
    def _nav():
        r = robot()
        # Check for obstacles periodically during navigation
        elapsed = 0.0
        step = 0.5  # check every 0.5 seconds
        r.motion.move(angle, DEFAULT_SPEED)
        while elapsed < duration:
            time.sleep(step)
            elapsed += step
            if r.lidar.is_obstacle_ahead():
                r.motion.stop()
                r.audio.speak("Obstacle detected. Waiting.", block=False)
                # Wait for obstacle to clear
                for _ in range(20):  # max 10 seconds
                    time.sleep(0.5)
                    if not r.lidar.is_obstacle_ahead():
                        r.motion.move(angle, DEFAULT_SPEED)
                        break
                else:
                    r.audio.speak("Path is blocked. Navigation cancelled.", block=False)
                    return
        r.motion.stop()
        r.audio.speak(f"Arrived at {dest_name}.", block=False)

    threading.Thread(target=_nav, daemon=True).start()

    return jsonify({
        "status": "navigating",
        "destination": dest_name,
        "estimated_duration_s": round(duration, 1),
    })


# ---------------------------------------------------------------------------
# Follow User
# ---------------------------------------------------------------------------
_follow_thread: threading.Thread = None
_follow_running = False


@app.post("/api/follow/start")
def follow_start():
    """Start follow-user mode (camera-based person tracking)."""
    global _follow_thread, _follow_running

    if _follow_running:
        return jsonify({"status": "already_following"})

    _follow_running = True

    def _follow_loop():
        global _follow_running
        r = robot()
        # Simple centre-tracking: if a person's face is left/right of centre,
        # rotate towards them and move forward.
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        while _follow_running:
            ret, frame = r.camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.2)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

            if len(faces) == 0:
                r.motion.stop()
                time.sleep(0.1)
                continue

            # Track the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            frame_w = frame.shape[1]
            face_cx = x + w // 2
            error = face_cx - frame_w // 2  # positive = person is to the right

            # Dead zone
            if abs(error) < frame_w * 0.1:
                # Person is centred — move forward if no obstacle
                if not r.lidar.is_obstacle_ahead():
                    r.motion.forward(20)
                else:
                    r.motion.stop()
            elif error > 0:
                r.motion.turn_right(15)
            else:
                r.motion.turn_left(15)

            time.sleep(0.1)

        r.motion.stop()

    import cv2  # needed inside the function
    _follow_thread = threading.Thread(target=_follow_loop, daemon=True)
    _follow_thread.start()

    return jsonify({"status": "following"})


@app.post("/api/follow/stop")
def follow_stop():
    """Stop follow-user mode."""
    global _follow_running
    _follow_running = False
    robot().motion.stop()
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

@app.get("/health")
def health():
    """Service health check with hardware status."""
    r = robot()
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
    })


# ============================================================================
# Main
# ============================================================================

def _cleanup(signum, frame):
    """Handle SIGINT/SIGTERM — stop robot gracefully."""
    global _follow_running
    _follow_running = False
    if _robot:
        _robot.shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, _cleanup)
signal.signal(signal.SIGTERM, _cleanup)


if __name__ == "__main__":
    print("=" * 50)
    print("  NovaCare — Robot REST Service")
    print(f"  Listening on {ROBOT_SERVICE_HOST}:{ROBOT_SERVICE_PORT}")
    print("=" * 50)
    app.run(host=ROBOT_SERVICE_HOST, port=ROBOT_SERVICE_PORT, threaded=True)
