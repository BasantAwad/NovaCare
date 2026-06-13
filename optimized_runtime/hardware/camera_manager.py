"""
Shared camera manager for optimized_runtime services.

Prefers the unified ``services/robot/camera_service`` when importable;
falls back to manual-aligned GStreamer capture (SerBot manual §4.1.3).
"""

import logging
import os
import sys
import threading
import time
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# Allow imports from services/robot when running from repo root
_ROBOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "services", "robot")
)
if _ROBOT_DIR not in sys.path and os.path.isdir(_ROBOT_DIR):
    sys.path.insert(0, _ROBOT_DIR)


def _try_robot_camera():
    """Reuse the HAL-configured camera singleton when robot service is loaded."""
    try:
        from robot_hal import get_robot
        return get_robot().camera.get_lightweight_camera()
    except Exception as exc:
        logger.debug("Robot HAL camera unavailable: %s", exc)
        return None


class CameraManager:
    """
    Singleton camera manager for optimized_runtime consumers.

    Delegates to ``LightweightCamera`` on the robot; otherwise uses
    ``pop.Util.gstrmer`` / V4L2 index fallback locally.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._robot_camera = None
        self.camera_index = 0
        self.cap: Any = None
        self.frame: Any = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.fps = 0.0
        self.last_time = time.time()
        self._initialized = True

    def start(self, camera_index: int = 0):
        if self.running:
            return True

        if self._robot_camera is None:
            self._robot_camera = _try_robot_camera()

        if self._robot_camera is not None:
            self._robot_camera.start_session()
            self.running = True
            self.thread = threading.Thread(target=self._delegated_loop, daemon=True)
            self.thread.start()
            logger.info("CameraManager started via robot camera_service")
            return True

        self.camera_index = camera_index
        try:
            from camera_service import get_camera

            self._robot_camera = get_camera()
            self._robot_camera.start_session()
            self.running = True
            self.thread = threading.Thread(target=self._delegated_loop, daemon=True)
            self.thread.start()
            logger.info("CameraManager started via camera_service (no OpenCV)")
            return True
        except Exception as exc:
            logger.error("CameraManager fallback failed: %s", exc)
            return False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        if self._robot_camera is not None:
            self._robot_camera.stop_session()
        elif self.cap:
            self.cap.release()
            self.cap = None

        logger.info("CameraManager stopped")

    def _delegated_loop(self):
        while self.running and self._robot_camera is not None:
            ok, frame = self._robot_camera.read_frame()
            if ok and frame is not None:
                self.frame = frame
                now = time.time()
                self.fps = 1.0 / max(0.001, now - self.last_time)
                self.last_time = now
            else:
                time.sleep(0.1)

    def _update_loop(self):
        while self.running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                now = time.time()
                self.fps = 1.0 / max(0.001, now - self.last_time)
                self.last_time = now
            else:
                logger.warning("Failed to capture frame")
                time.sleep(0.1)

    def get_frame(self):
        return self.frame

    def get_frame_encoded(self):
        if self._robot_camera is not None:
            return self._robot_camera.read_frame_jpeg()
        return None


def get_camera_manager() -> CameraManager:
    return CameraManager()
