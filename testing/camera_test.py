"""
NovaCare Testing - Camera Module
Uses PC webcam to simulate robot camera for development.
DELETE THIS FILE when integrating with actual robot hardware.
"""
import cv2
import numpy as np
import threading
from datetime import datetime
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestCamera:
    """
    Test camera using PC webcam.
    Replace with robot camera interface in production.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.face_cascade = None
        self._load_face_detector()

    def _load_face_detector(self):
        """Load OpenCV face cascade for face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("[TestCamera] Face detector loaded")
        except Exception as e:
            print(f"[TestCamera] Could not load face detector: {e}")

    def start(self):
        """Start capturing from webcam"""
        if self.is_running:
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print("[TestCamera] Could not open webcam")
                return False
            
            self.is_running = True
            print(f"[TestCamera] Webcam started (index {self.camera_index})")
            return True
        except Exception as e:
            print(f"[TestCamera] Error starting camera: {e}")
            return False

    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[TestCamera] Camera stopped")

    def get_frame(self):
        """Get current frame from camera"""
        if not self.is_running or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.latest_frame = frame
            return frame
        return None

    def get_frame_as_jpeg(self):
        """Get frame encoded as JPEG bytes (for streaming)"""
        frame = self.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                return jpeg.tobytes()
        return None

    def detect_faces(self, frame=None):
        """
        Detect faces in frame
        :param frame: Optional frame, uses latest if not provided
        :return: List of face regions (x, y, w, h)
        """
        if frame is None:
            frame = self.get_frame()
        if frame is None or self.face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces.tolist() if len(faces) > 0 else []

    def get_face_for_emotion(self, frame=None):
        """
        Extract face region preprocessed for emotion detection
        :return: 48x48 grayscale face image or None
        """
        if frame is None:
            frame = self.get_frame()
        if frame is None:
            return None

        faces = self.detect_faces(frame)
        if not faces:
            return None

        # Get largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest

        # Extract and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_region = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (48, 48))
        
        return face_resized


# Singleton
_test_camera_instance = None

def get_test_camera():
    global _test_camera_instance
    if _test_camera_instance is None:
        _test_camera_instance = TestCamera()
    return _test_camera_instance


if __name__ == "__main__":
    # Quick test
    cam = TestCamera()
    if cam.start():
        print("Press 'q' to quit")
        while True:
            frame = cam.get_frame()
            if frame is not None:
                faces = cam.detect_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow('Test Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cam.stop()
        cv2.destroyAllWindows()
