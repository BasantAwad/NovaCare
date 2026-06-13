"""
NovaCare Robot — Configuration
===============================
Hardware and service configuration for the SERBot Prime X.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# Minimal robot mode (SerBot = I/O bridge only, no AI / watch / vision logic)
# ---------------------------------------------------------------------------
MINIMAL_MODE = os.getenv("NOVACARE_MINIMAL", "0").lower() in ("1", "true", "yes")
ROBOT_SERVICE_HOST = os.getenv("ROBOT_SERVICE_HOST", "0.0.0.0")
ROBOT_SERVICE_PORT = int(os.getenv("ROBOT_SERVICE_PORT", "9000"))

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
# Camera index fallback (used when pop.Util is unavailable)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
# Jetson CSI flip-method for Util.gstrmer / nvarguscamerasrc (manual §4.1.3.1)
CAMERA_GSTREAMER_FLIP = int(os.getenv("CAMERA_GSTREAMER_FLIP", "0"))

# ---------------------------------------------------------------------------
# Live Stream (lightweight camera service)
# ---------------------------------------------------------------------------
STREAM_WIDTH = int(os.getenv("STREAM_WIDTH", "320"))
STREAM_HEIGHT = int(os.getenv("STREAM_HEIGHT", "240"))
STREAM_FPS = int(os.getenv("STREAM_FPS", "10"))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "60"))

# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------
DEFAULT_SPEED = int(os.getenv("DEFAULT_SPEED", "30"))
MAX_SPEED = int(os.getenv("MAX_SPEED", "80"))
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

# ---------------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------
LIDAR_ENABLED = os.getenv("LIDAR_ENABLED", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Destinations (for navigation)
# ---------------------------------------------------------------------------
DESTINATIONS = {
    "kitchen":    {"angle": 0,   "distance_cm": 150},
    "bathroom":   {"angle": 90,  "distance_cm": 80},
    "living":     {"angle": 180, "distance_cm": 100},
    "bedroom":    {"angle": 270, "distance_cm": 120},
    "dining":     {"angle": 45,  "distance_cm": 180},
    "entrance":   {"angle": 315, "distance_cm": 200},
}
