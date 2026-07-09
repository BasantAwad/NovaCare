#!/usr/bin/env python3
"""
NovaCare On-Robot AI Service
==============================
Runs GPU-accelerated AI inference directly on the JetAuto's Jetson board
for minimum-latency response times.

Features:
  - Fall Detection (MediaPipe Pose + CUDA)  → auto-triggers SOS
  - Person Tracking (depth camera + LiDAR fusion) → follow mode
  - Emotion Detection (face classifier) → adaptive robot behavior

Response time target: < 500ms from event to alert.

This module can run standalone as a background service or be imported
by robot_service.py for integrated operation.
"""

import os
import sys
import time
import json
import threading
from typing import Optional, Dict, Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FALL_DETECTION_THRESHOLD,
    FALL_DETECTION_CONFIRM_FRAMES,
    ONBOARD_FALL_DETECTION,
    ONBOARD_PERSON_TRACKING,
    ONBOARD_EMOTION_DETECTION,
    SOS_ALARM_FREQUENCY,
    FCM_SERVER_KEY,
    ROBOT_NAME,
)

# Try importing the fall detector
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fall-detection"))
    from fall_detection import FallDetector
    from gesture_recognition import GestureRecognizer
    _FALL_DETECTOR_AVAILABLE = True
except ImportError:
    _FALL_DETECTOR_AVAILABLE = False
    print("[AI] Fall detection / Gesture recognition modules not available")

# Try importing MediaPipe for person tracking
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# Try OpenCV
try:
    import cv2
except ImportError:
    _CV2_AVAILABLE = False

# Try importing Edge Models (Emotion & ASL)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "emotion"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "asl", "inference"))
    from face_predictor import FaceEmotionAnalyzer
    from predictor import ASLPredictor
    _EDGE_MODELS_AVAILABLE = True
except ImportError as e:
    _EDGE_MODELS_AVAILABLE = False
    print(f"[AI] Edge models (Emotion/ASL) not available: {e}")

class OnboardAIService:
    """
    Manages all on-robot AI processing pipelines.
    Runs on the Jetson GPU for fast inference.
    """

    def __init__(self, camera_read_fn: Callable = None,
                 alarm_fn: Callable = None,
                 speak_fn: Callable = None,
                 notify_fn: Callable = None):
        """
        Args:
            camera_read_fn: Callable that returns (bool, np.ndarray) frame.
            alarm_fn: Callable to trigger robot alarm.
            speak_fn: Callable to speak text on robot.
            notify_fn: Callable to send push notifications.
        """
        self._camera_read = camera_read_fn
        self._alarm = alarm_fn
        self._speak = speak_fn
        self._notify = notify_fn

        # Fall detection state
        self._fall_detector: Optional[FallDetector] = None
        self._fall_active = False
        self._fall_thread: Optional[threading.Thread] = None
        self._fall_stop = threading.Event()
        self._last_fall_event: Optional[Dict] = None
        self._fall_cooldown = 30.0  # seconds between fall alerts

        # Person tracking state
        self._tracking_active = False
        self._tracking_thread: Optional[threading.Thread] = None
        self._tracking_stop = threading.Event()
        self._track_target: Optional[Dict] = None  # {x, y, depth, confidence}

        # Emotion detection state
        self._emotion_active = False
        self._emotion_thread: Optional[threading.Thread] = None
        self._emotion_stop = threading.Event()
        self._current_emotion: Optional[str] = None

        # Gesture control state
        self._gesture_recognizer: Optional[GestureRecognizer] = None
        self._gesture_active = False
        self._gesture_thread: Optional[threading.Thread] = None
        self._gesture_stop = threading.Event()
        self._last_binary_response: Optional[bool] = None

        # Guarding mode state
        self._guarding_active = False
        self._guarding_thread: Optional[threading.Thread] = None
        self._guarding_stop = threading.Event()

        # ASL Recognition state
        self._asl_active = False
        self._asl_thread: Optional[threading.Thread] = None
        self._asl_stop = threading.Event()
        self._current_asl_prediction: Optional[Dict] = None

        self._lock = threading.Lock()

        # Voice Wake-Up setup
        self._voice_setup_done = False
        self._setup_voice_callbacks()

    def _setup_voice_callbacks(self):
        """Register offline voice wake-up callbacks."""
        try:
            from robot_hal import get_robot
            r = get_robot()
            r.audio.register_asr_callback(self._on_voice_detected)
            self._voice_setup_done = True
            print("[AI] Voice wake-up handler registered")
        except Exception as e:
            print(f"[AI] Failed to register voice handler: {e}")

    def _on_voice_detected(self, text: str):
        """Handle offline ASR voice words."""
        text = text.lower()
        if "help me" in text:
            print("[AI] ⚠ WAKE WORD SOS DETECTED!")
            if self._alarm:
                self._alarm(SOS_ALARM_FREQUENCY, 10)
            if self._speak:
                self._speak("Emergency voice command detected. Alerting caregivers.", False)
            if self._notify:
                self._notify("voice_wakeup", "robot_mic", {
                    "event": "sos_voice_detected",
                    "text": text
                })
        elif "nova care" in text or "novacare" in text:
            print("[AI] Wake word detected")
            if self._speak:
                self._speak("Yes, I am here. How can I help?", False)

    # ========================================================================
    # Fall Detection
    # ========================================================================

    def start_fall_detection(self) -> bool:
        """Start the fall detection pipeline."""
        if self._fall_active:
            return True

        if not _FALL_DETECTOR_AVAILABLE:
            print("[AI] Cannot start fall detection — module not available")
            return False

        self._fall_detector = FallDetector()
        self._fall_stop.clear()
        self._fall_active = True

        self._fall_thread = threading.Thread(
            target=self._fall_detection_loop, daemon=True, name="fall-detect"
        )
        self._fall_thread.start()
        print("[AI] Fall detection started (GPU-accelerated)")
        return True

    def stop_fall_detection(self):
        """Stop the fall detection pipeline."""
        self._fall_stop.set()
        self._fall_active = False
        if self._fall_thread:
            self._fall_thread.join(timeout=2.0)
        self._fall_detector = None
        print("[AI] Fall detection stopped")

    def _fall_detection_loop(self):
        """Main fall detection loop — runs at ~10 fps on Jetson GPU."""
        last_alert_time = 0

        while not self._fall_stop.is_set():
            if not self._camera_read:
                time.sleep(0.5)
                continue

            try:
                ok, frame = self._camera_read()
                if not ok or frame is None:
                    time.sleep(0.1)
                    continue

                result = self._fall_detector.analyze_frame(frame)

                if result.get("fall_detected"):
                    now = time.time()
                    # Cooldown to prevent alert spam
                    if now - last_alert_time > self._fall_cooldown:
                        last_alert_time = now
                        self._handle_fall_event(result)

            except Exception as e:
                print(f"[AI] Fall detection error: {e}")

            time.sleep(0.1)  # ~10 fps

    def _handle_fall_event(self, result: Dict):
        """Handle a confirmed fall detection event."""
        event = {
            "timestamp": time.time(),
            "confidence": result.get("confidence", 0),
            "method": result.get("method", "unknown"),
            "spine_angle": result.get("spine_angle"),
            "velocity": result.get("velocity"),
        }
        self._last_fall_event = event
        print(f"[AI] ⚠ FALL DETECTED! Confidence: {event['confidence']:.2f}")

        # Trigger alarm
        if self._alarm:
            self._alarm(SOS_ALARM_FREQUENCY, 10)

        # Announce
        if self._speak:
            self._speak("Fall detected! Are you okay? I am alerting your caregivers.", False)

        # Send push notification
        if self._notify:
            self._notify("fall_detection", "robot_camera", {
                "event": "fall_detected",
                "confidence": event["confidence"],
                "method": event["method"],
            })

    @property
    def fall_detection_active(self) -> bool:
        return self._fall_active

    @property
    def last_fall_event(self) -> Optional[Dict]:
        return self._last_fall_event

    # ========================================================================
    # Person Tracking (for Follow Mode)
    # ========================================================================

    def start_person_tracking(self) -> bool:
        """Start person tracking for follow mode."""
        if self._tracking_active:
            return True

        if not _MEDIAPIPE_AVAILABLE or not _CV2_AVAILABLE:
            print("[AI] Cannot start tracking — MediaPipe/OpenCV not available")
            return False

        self._tracking_stop.clear()
        self._tracking_active = True

        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True, name="person-track"
        )
        self._tracking_thread.start()
        print("[AI] Person tracking started")
        return True

    def stop_person_tracking(self):
        """Stop person tracking."""
        self._tracking_stop.set()
        self._tracking_active = False
        self._track_target = None
        if self._tracking_thread:
            self._tracking_thread.join(timeout=2.0)
        print("[AI] Person tracking stopped")

    def _tracking_loop(self):
        """Track the nearest person using pose detection."""
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Fastest model for tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        while not self._tracking_stop.is_set():
            if not self._camera_read:
                time.sleep(0.1)
                continue

            try:
                ok, frame = self._camera_read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    # Track the center of the person (average of hips)
                    l_hip = results.pose_landmarks.landmark[23]
                    r_hip = results.pose_landmarks.landmark[24]
                    cx = (l_hip.x + r_hip.x) / 2.0
                    cy = (l_hip.y + r_hip.y) / 2.0

                    with self._lock:
                        self._track_target = {
                            "x": cx,  # normalized 0-1 (0.5 = center)
                            "y": cy,
                            "pixel_x": int(cx * w),
                            "pixel_y": int(cy * h),
                            "confidence": min(l_hip.visibility, r_hip.visibility),
                            "timestamp": time.time(),
                        }
                else:
                    with self._lock:
                        self._track_target = None

            except Exception as e:
                print(f"[AI] Tracking error: {e}")

            time.sleep(0.05)  # ~20 fps for responsive tracking

        pose.close()

    def get_track_target(self) -> Optional[Dict]:
        """Get the current tracking target position."""
        with self._lock:
            return self._track_target

    @property
    def tracking_active(self) -> bool:
        return self._tracking_active

    # ========================================================================
    # Emotion Detection
    # ========================================================================

    def start_emotion_detection(self) -> bool:
        """Start emotion detection from camera feed."""
        if self._emotion_active:
            return True

        if not _CV2_AVAILABLE:
            print("[AI] Cannot start emotion detection — OpenCV not available")
            return False

        self._emotion_stop.clear()
        self._emotion_active = True

        self._emotion_thread = threading.Thread(
            target=self._emotion_loop, daemon=True, name="emotion-detect"
        )
        self._emotion_thread.start()
        print("[AI] Emotion detection started")
        return True

    def stop_emotion_detection(self):
        """Stop emotion detection."""
        self._emotion_stop.set()
        self._emotion_active = False
        self._current_emotion = None
        if self._emotion_thread:
            self._emotion_thread.join(timeout=2.0)
        print("[AI] Emotion detection stopped")

    def _emotion_loop(self):
        """Detect emotions using Edge AI GPU model."""
        if not _EDGE_MODELS_AVAILABLE:
            print("[AI] Edge models unavailable for emotion detection")
            self._emotion_active = False
            return

        analyzer = None
        try:
            analyzer = FaceEmotionAnalyzer(use_huggingface=True, device="cuda")
        except Exception as e:
            print(f"[AI] Failed to load FaceEmotionAnalyzer: {e}")
            self._emotion_active = False
            return

        while not self._emotion_stop.is_set():
            if not self._camera_read:
                time.sleep(0.5)
                continue

            try:
                ok, frame = self._camera_read()
                if not ok or frame is None:
                    time.sleep(0.2)
                    continue

                # Run GPU inference
                result = analyzer.predict(frame, detect_face=True)
                if result.get("face_detected"):
                    self._current_emotion = result.get("emotion")
                else:
                    self._current_emotion = None

            except Exception as e:
                print(f"[AI] Emotion detection error: {e}")

            time.sleep(0.2)  # ~5 fps for emotion (less time-critical)

    @property
    def current_emotion(self) -> Optional[str]:
        return self._current_emotion

    # ========================================================================
    # Gesture Control
    # ========================================================================

    def start_gesture_control(self) -> bool:
        if self._gesture_active:
            return True

        if not _FALL_DETECTOR_AVAILABLE:
            print("[AI] Cannot start gesture control — module not available")
            return False

        self._gesture_recognizer = GestureRecognizer()
        self._gesture_stop.clear()
        self._gesture_active = True

        self._gesture_thread = threading.Thread(
            target=self._gesture_loop, daemon=True, name="gesture-detect"
        )
        self._gesture_thread.start()
        print("[AI] Gesture control started")
        return True

    def stop_gesture_control(self):
        self._gesture_stop.set()
        self._gesture_active = False
        if self._gesture_thread:
            self._gesture_thread.join(timeout=2.0)
        self._gesture_recognizer = None
        print("[AI] Gesture control stopped")

    def _gesture_loop(self):
        """Analyze frames for hand gestures and execute commands."""
        # For robot control
        from robot_hal import get_robot
        r = get_robot()

        while not self._gesture_stop.is_set():
            if not self._camera_read:
                time.sleep(0.5)
                continue

            try:
                ok, frame = self._camera_read()
                if not ok or frame is None:
                    time.sleep(0.1)
                    continue

                result = self._gesture_recognizer.analyze_frame(frame)
                gesture = result.get("gesture")
                
                if gesture == "attention":
                    if self._speak:
                        self._speak("I am listening.", block=False)
                
                elif gesture == "sos":
                    print("[AI] ⚠ GESTURE SOS DETECTED!")
                    if self._alarm:
                        self._alarm(SOS_ALARM_FREQUENCY, 10)
                    if self._speak:
                        self._speak("Emergency gesture detected. Alerting caregivers.", False)
                    if self._notify:
                        self._notify("gesture_control", "robot_camera", {
                            "event": "sos_gesture_detected"
                        })
                        
                elif gesture == "push_back":
                    print("[AI] Gesture: Push Back")
                    r.motion.move(180, 40) # move backward
                    time.sleep(0.5)
                    r.motion.stop()
                    
                elif gesture == "flat_hand":
                    y_delta = result.get("y_delta", 0.0)
                    # Y delta positive means hand moving down -> reduce speed
                    # Y delta negative means hand moving up -> increase speed
                    if abs(y_delta) > 0.05:
                        direction = "Decreasing" if y_delta > 0 else "Increasing"
                        print(f"[AI] Gesture: Speed/Proximity Adjust ({direction})")
                        # In a full implementation, this would adjust a shared variable
                        
                elif gesture in ["thumbs_up", "thumbs_down"]:
                    val = result.get("binary_response")
                    with self._lock:
                        self._last_binary_response = val
                    print(f"[AI] Gesture: Binary Confirmation -> {'YES' if val else 'NO'}")

            except Exception as e:
                print(f"[AI] Gesture detection error: {e}")

            time.sleep(0.1)  # ~10 fps

    def get_last_binary_response(self) -> Optional[bool]:
        """Fetch and clear the last binary confirmation from the user."""
        with self._lock:
            val = self._last_binary_response
            self._last_binary_response = None
            return val

    # ========================================================================
    # Lidar Guarding
    # ========================================================================

    def start_guarding(self) -> bool:
        """Start the Lidar Guarding mode to track nearby movement."""
        if self._guarding_active:
            return True

        self._guarding_stop.clear()
        self._guarding_active = True

        self._guarding_thread = threading.Thread(
            target=self._guarding_loop, daemon=True, name="lidar-guard"
        )
        self._guarding_thread.start()
        print("[AI] Lidar guarding mode started")
        return True

    def stop_guarding(self):
        self._guarding_stop.set()
        self._guarding_active = False
        if self._guarding_thread:
            self._guarding_thread.join(timeout=2.0)
        print("[AI] Lidar guarding mode stopped")

    def _guarding_loop(self):
        """Poll Lidar for closest obstacle and rotate to face it."""
        from robot_hal import get_robot
        r = get_robot()

        while not self._guarding_stop.is_set():
            if not getattr(r, 'lidar', None) or not r.lidar.is_available:
                time.sleep(1.0)
                continue

            try:
                angle, dist = r.lidar.get_closest_obstacle()
                
                # If obstacle is within 2 meters (2000mm)
                if dist < 2000:
                    # Angle 0 is front. Normalize to -180 to 180
                    target_angle = angle
                    if target_angle > 180:
                        target_angle -= 360
                        
                    # If angle is outside a small deadband, rotate
                    if abs(target_angle) > 15:
                        direction = "right" if target_angle < 0 else "left"
                        angular_z = 0.5 if direction == "left" else -0.5
                        r.motion.move(0, 0, angular_z, duration=0.2)
                        
            except Exception as e:
                print(f"[AI] Guarding loop error: {e}")

            time.sleep(0.2)  # ~5 fps

    # ========================================================================
    # ASL Recognition
    # ========================================================================

    def start_asl_recognition(self) -> bool:
        if self._asl_active:
            return True

        if not _EDGE_MODELS_AVAILABLE:
            print("[AI] Cannot start ASL — edge models not available")
            return False

        self._asl_stop.clear()
        self._asl_active = True

        self._asl_thread = threading.Thread(
            target=self._asl_loop, daemon=True, name="asl-detect"
        )
        self._asl_thread.start()
        print("[AI] ASL recognition started on Edge GPU")
        return True

    def stop_asl_recognition(self):
        self._asl_stop.set()
        self._asl_active = False
        with self._lock:
            self._current_asl_prediction = None
        if self._asl_thread:
            self._asl_thread.join(timeout=2.0)
        print("[AI] ASL recognition stopped")

    def _asl_loop(self):
        predictor = None
        try:
            predictor = ASLPredictor(device="cuda")
        except Exception as e:
            print(f"[AI] Failed to load ASLPredictor: {e}")
            self._asl_active = False
            return

        while not self._asl_stop.is_set():
            if not self._camera_read:
                time.sleep(0.1)
                continue

            try:
                ok, frame = self._camera_read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                res = predictor.predict_frame(frame)
                with self._lock:
                    self._current_asl_prediction = {
                        "letter": res.letter,
                        "confidence": res.confidence,
                        "is_confirmed": res.is_confirmed
                    }
            except Exception as e:
                print(f"[AI] ASL detection error: {e}")

            time.sleep(0.05)  # ~20 fps for fluid ASL tracking

    @property
    def current_asl_prediction(self) -> Optional[Dict]:
        with self._lock:
            return self._current_asl_prediction

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def start_all(self):
        """Start all configured AI pipelines."""
        if ONBOARD_FALL_DETECTION:
            self.start_fall_detection()
        if ONBOARD_PERSON_TRACKING:
            self.start_person_tracking()
        if ONBOARD_EMOTION_DETECTION:
            self.start_emotion_detection()

    def stop_all(self):
        """Stop all AI pipelines."""
        self.stop_fall_detection()
        self.stop_person_tracking()
        self.stop_emotion_detection()
        self.stop_gesture_control()
        self.stop_guarding()
        self.stop_asl_recognition()

    def get_status(self) -> Dict:
        """Get status of all AI pipelines."""
        return {
            "fall_detection": {
                "active": self._fall_active,
                "available": _FALL_DETECTOR_AVAILABLE,
                "last_event": self._last_fall_event,
            },
            "person_tracking": {
                "active": self._tracking_active,
                "available": _MEDIAPIPE_AVAILABLE,
                "target": self._track_target,
            },
            "emotion_detection": {
                "active": self._emotion_active,
                "available": _CV2_AVAILABLE,
                "current_emotion": self._current_emotion,
            },
            "gesture_control": {
                "active": self._gesture_active,
                "locked": getattr(self._gesture_recognizer, "is_locked", True) if self._gesture_recognizer else True,
            },
            "guarding_mode": {
                "active": self._guarding_active
            },
            "asl_recognition": {
                "active": self._asl_active,
                "current_prediction": self.current_asl_prediction
            }
        }


# ---------------------------------------------------------------------------
# Standalone entry point (run as background service on JetAuto)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from robot_hal import get_robot

    print("=" * 50)
    print("  NovaCare On-Robot AI Service")
    print(f"  Robot: {ROBOT_NAME}")
    print("=" * 50)

    robot = get_robot()

    ai = OnboardAIService(
        camera_read_fn=robot.camera.read_frame,
        alarm_fn=robot.audio.play_alarm,
        speak_fn=robot.audio.speak,
    )

    ai.start_all()
    print("[AI] All configured pipelines running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(5)
            status = ai.get_status()
            active = [k for k, v in status.items() if v.get("active")]
            print(f"[AI] Active pipelines: {', '.join(active) or 'none'}")
    except KeyboardInterrupt:
        print("\n[AI] Shutting down...")
        ai.stop_all()
        robot.shutdown()
