import cv2
import threading
import time
import logging
from typing import Optional, Generator, Any

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Singleton Camera Manager to optimize resource usage on SERBot.
    Provides shared access to the camera feed for multiple services.
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
        
        self.camera_index = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Any = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._initialized = True
        self.fps = 0
        self.last_time = time.time()

    def start(self, camera_index: int = 0):
        """Start the camera capture thread"""
        if self.running:
            return
        
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {camera_index}")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info(f"CameraManager started on index {camera_index}")
        return True

    def stop(self):
        """Stop the camera capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        logger.info("CameraManager stopped")

    def _update_loop(self):
        """Background thread to read frames from camera"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                # Calculate FPS
                now = time.time()
                self.fps = 1.0 / (now - self.last_time)
                self.last_time = now
            else:
                logger.warning("Failed to capture frame")
                time.sleep(0.1)

    def get_frame(self):
        """Get the latest frame"""
        return self.frame

    def get_frame_encoded(self):
        """Get latest frame encoded as JPEG for streaming"""
        if self.frame is None:
            return None
        
        ret, buffer = cv2.imencode('.jpg', self.frame)
        if ret:
            return buffer.tobytes()
        return None

def get_camera_manager() -> CameraManager:
    """Helper to get the singleton instance"""
    return CameraManager()
