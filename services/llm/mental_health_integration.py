"""
Mental Health Integration — Bridge Module
==========================================
Connects the camera-based emotion detection with the multi-API therapy pipeline.
This module is imported by ConversationalAI.chat() — its interface is kept
backward-compatible with the original MentalHealthOrchestrator.
"""

import os
import threading
import time

import cv2
import base64
import json
import urllib.request
import numpy as np

# Robot camera REST API URL (robot_service.py on port 9000)
ROBOT_CAMERA_URL = os.getenv("ROBOT_CAMERA_URL", "http://10.174.134.247:9000/api/camera/frame")


# ===========================================================================
# Camera Emotion Poller — Robot-Integrated
# ===========================================================================
class CameraEmotionPoller:
    """
    Polls camera frames for emotion detection and fall detection.

    Integration priority:
      1. Robot camera via REST API (ROBOT_CAMERA_URL)
      2. Local webcam fallback (cv2.VideoCapture(0))
    """

    def __init__(self):
        self.latest_emotion = "neutral"
        self.latest_confidence = 0.0
        self.running = False
        self.thread = None
        self.analyzer = None
        self.fall_detector = None
        self._use_robot_camera = True  # Try robot camera first
        self._local_cap = None

    def start(self):
        if self.running:
            return

        try:
            from emotion_detection import get_analyzer
            self.analyzer = get_analyzer()
        except ImportError:
            print("Warning: emotion_detection module not found.")
            return

        try:
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            fd_dir = os.path.join(parent_dir, "fall-detection")
            if fd_dir not in sys.path:
                sys.path.insert(0, fd_dir)
            from fall_detection import FallDetector
            self.fall_detector = FallDetector()
            print("✓ FallDetector successfully loaded and initialized in CameraEmotionPoller")
        except Exception as e:
            print(f"Warning: Could not initialize FallDetector: {e}")
            self.fall_detector = None

        self.running = True
        self.thread = threading.Thread(target=self._poll_camera, daemon=True)
        self.thread.start()

    def _fetch_robot_frame(self):
        """Fetch a frame from the robot camera REST API."""
        try:
            req = urllib.request.Request(ROBOT_CAMERA_URL, method="GET")
            req.add_header("X-API-Key", "novacare-secure-key-2026")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "success" and data.get("image"):
                img_bytes = base64.b64decode(data["image"])
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return frame
        except Exception:
            pass
        return None

    def _poll_camera(self):
        robot_failures = 0
        while self.running:
            frame = None

            # Try robot camera first
            if self._use_robot_camera:
                frame = self._fetch_robot_frame()
                if frame is not None:
                    robot_failures = 0
                else:
                    robot_failures += 1
                    # After 5 consecutive failures, fall back to local webcam
                    if robot_failures >= 5:
                        print("[CameraPoller] Robot camera unavailable — falling back to local webcam")
                        self._use_robot_camera = False

            # Fallback: local webcam
            if frame is None and not self._use_robot_camera:
                if os.getenv("ALLOW_LOCAL_WEBCAM_POLLING") == "true":
                    if self._local_cap is None:
                        self._local_cap = cv2.VideoCapture(0)
                    ret, frame = self._local_cap.read()
                    if not ret:
                        frame = None
                else:
                    # Do not lock the local webcam by default, as it breaks frontend camera features like ASL
                    time.sleep(1)
                    continue

            # Run emotion detection
            if frame is not None and self.analyzer:
                try:
                    result = self.analyzer.predict(frame, detect_face=True)
                    if result.get("emotion") != "unknown":
                        self.latest_emotion = result["emotion"].lower()
                        self.latest_confidence = result["confidence"]
                except Exception as e:
                    print(f"Emotion polling error: {e}")

            # Run fall detection on the exact same frame
            if frame is not None and self.fall_detector is not None:
                try:
                    fall_result = self.fall_detector.analyze_frame(frame)
                    if fall_result.get("fall_detected"):
                        print(f"🚨 [Fall Detected] Confidence: {fall_result.get('confidence')}")
                        # Issue verbal speech alert on the robot!
                        try:
                            speak_url = ROBOT_CAMERA_URL.replace("/api/camera/frame", "/api/tts/speak")
                            speak_data = json.dumps({
                                "text": "Warning: Fall detected! Please remain calm. I am notifying your caretaker immediately."
                            }).encode("utf-8")
                            speak_req = urllib.request.Request(
                                speak_url,
                                data=speak_data,
                                headers={
                                    "Content-Type": "application/json",
                                    "X-API-Key": "novacare-secure-key-2026"
                                },
                                method="POST"
                            )
                            with urllib.request.urlopen(speak_req, timeout=3) as speak_resp:
                                pass
                        except Exception as se:
                            print(f"Error triggering robot verbal fall alert: {se}")
                except Exception as fe:
                    print(f"Fall detection execution error: {fe}")

            time.sleep(0.5)

        # Cleanup
        if self._local_cap is not None:
            self._local_cap.release()
            self._local_cap = None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()


# Global poller instance
poller = CameraEmotionPoller()
poller.start()


# ===========================================================================
# MentalHealthOrchestrator — now backed by the multi-API pipeline
# ===========================================================================
class MentalHealthOrchestrator:
    """
    Drop-in replacement for the original orchestrator.
    process() signature returns the same (bypass, reply, prefix) tuple
    so ConversationalAI.chat() needs zero changes.
    """

    def __init__(self):
        self.target_emotions = {"sad", "angry", "fear", "disgust"}
        self.threshold = 0.65
        self._pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Lazy-import so missing deps don't crash the whole server."""
        try:
            from mental_health_pipeline import get_pipeline
            self._pipeline = get_pipeline()
            print("✓ MentalHealthOrchestrator: Multi-API therapy pipeline loaded")
        except Exception as e:
            print(f"⚠ MentalHealthOrchestrator: Pipeline unavailable ({e})")
            self._pipeline = None

    def process(self, user_message: str, conversation_history=None, frontend_emotion: str = "unknown", frontend_confidence: float = 0.0):
        """
        Returns (bypass: bool, reply: str, prefix_to_add: str)
        - bypass=True  → use reply directly, skip standard LLM
        - bypass=False, prefix set → prepend prefix to standard LLM reply
        - bypass=False, prefix="" → no mental-health signal
        """
        # Prioritize explicit frontend emotion, fallback to global poller
        if frontend_emotion and frontend_emotion.lower() != "unknown":
            emotion = frontend_emotion.lower()
            confidence = frontend_confidence
        else:
            emotion = poller.latest_emotion.lower()
            confidence = poller.latest_confidence

        # If the pipeline is available, use it (full multi-API flow)
        if self._pipeline and self._pipeline.is_available:
            result = self._pipeline.process(
                user_message=user_message,
                emotion=emotion,
                emotion_confidence=confidence,
                conversation_history=conversation_history,
            )

            # Log pipeline stages
            if result.stages_log:
                print(f"[MH Pipeline] {' | '.join(result.stages_log)}")

            if result.triggered and result.response:
                return True, result.response, ""

            # Even if pipeline didn't trigger on text, emotion may warrant a gentle prefix
            if emotion in self.target_emotions and confidence > self.threshold:
                return False, "", "I noticed you might be feeling a bit down. "

            return False, "", ""

        # ---- Fallback: emotion-only prefix (no APIs configured) ----
        if emotion in self.target_emotions and confidence > self.threshold:
            prefix = "I noticed you seem a bit down today, are you doing okay? "
            return False, "", prefix

        return False, "", ""


# ===========================================================================
# Singleton (backward-compatible with existing imports)
# ===========================================================================
_orchestrator_instance = None


def get_orchestrator():
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MentalHealthOrchestrator()
    return _orchestrator_instance
