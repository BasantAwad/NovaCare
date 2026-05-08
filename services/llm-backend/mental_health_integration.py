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


# ===========================================================================
# Camera Emotion Poller (unchanged from original)
# ===========================================================================
class CameraEmotionPoller:
    def __init__(self):
        self.latest_emotion = "neutral"
        self.latest_confidence = 0.0
        self.running = False
        self.thread = None
        self.analyzer = None

    def start(self):
        if self.running:
            return
        
        try:
            from emotion_detection import get_analyzer
            self.analyzer = get_analyzer()
        except ImportError:
            print("Warning: emotion_detection module not found.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._poll_camera, daemon=True)
        self.thread.start()

    def _poll_camera(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret and self.analyzer:
                try:
                    result = self.analyzer.predict(frame, detect_face=True)
                    if result.get("emotion") != "unknown":
                        self.latest_emotion = result["emotion"].lower()
                        self.latest_confidence = result["confidence"]
                except Exception as e:
                    print(f"Emotion polling error: {e}")
            time.sleep(0.5)
        cap.release()

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

    def process(self, user_message: str, conversation_history=None):
        """
        Returns (bypass: bool, reply: str, prefix_to_add: str)
        - bypass=True  → use reply directly, skip standard LLM
        - bypass=False, prefix set → prepend prefix to standard LLM reply
        - bypass=False, prefix="" → no mental-health signal
        """
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
