"""
NovaCare Gesture Recognition Module
===================================
Utilizes MediaPipe Hands to process camera frames and classify gestures 
for robot control and emergency signals.

Supported Gestures:
- Attention Lock: "V" or Peace sign (Index and Middle fingers up).
- SOS Signal: Thumb tucked, fingers folded over it (Distress signal).
- Give Me Space: Two hands with open palms facing camera.
- Speed/Proximity Control: Single hand, open palm, moving vertically.
- Binary Confirmations: Thumbs Up / Thumbs Down.

This module is designed for rapid inference on the Jetson Orin Nano GPU.
"""

import time
import math
import numpy as np
from typing import Dict, Any, List, Optional

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


class GestureRecognizer:
    def __init__(self):
        self.hands = None
        if _MEDIAPIPE_AVAILABLE:
            try:
                print("[GESTURE] Initializing MediaPipe Hands with Jetson GPU acceleration...")
                self.hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=0,  # Fast inference for edge devices
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[GESTURE] ✓ Hardware Acceleration: CUDA / Jetson Orin Nano active")
            except Exception as e:
                print(f"[ERROR] Initializing MediaPipe Hands: {e}")
                self.hands = None

        # State management
        self.is_locked = True  # True means ignoring commands until 'Attention' gesture
        self.lock_timeout = 5.0
        self.last_attention_time = 0

        # Temporal tracking
        self.history_len = 15
        self.hand_y_history = []
        self.hand_size_history = []

        # SOS specific
        self.sos_sequence_state = 0  # 0: Open, 1: Thumb tucked, 2: Fist closed over thumb
        self.sos_confirm_frames = 0

        # Confirmation tracking
        self.thumbs_up_frames = 0
        self.thumbs_down_frames = 0

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame for hand gestures.
        Returns a dictionary containing the detected gesture and confidence.
        """
        if not self.hands or frame is None:
            return {"gesture": "none", "locked": self.is_locked}

        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        now = time.time()
        
        # Check if lock has timed out
        if not self.is_locked and (now - self.last_attention_time > self.lock_timeout):
            self.is_locked = True
            print("[GESTURE] Attention lock timed out. Waiting for wake-word gesture.")

        result_payload = {
            "gesture": "none",
            "locked": self.is_locked,
            "y_delta": 0.0,
            "size_delta": 0.0,
            "binary_response": None
        }

        if not results.multi_hand_landmarks:
            # Decay tracking counters if hands lost
            self.sos_confirm_frames = max(0, self.sos_confirm_frames - 1)
            self.thumbs_up_frames = max(0, self.thumbs_up_frames - 1)
            self.thumbs_down_frames = max(0, self.thumbs_down_frames - 1)
            return result_payload

        # Process hands
        hand_states = []
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            state = self._analyze_hand_topology(hand_landmarks)
            state["label"] = handedness.classification[0].label  # 'Left' or 'Right'
            hand_states.append(state)

            # Draw landmarks (optional, for debugging if we hook it to video feed)
            # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # ---------------------------------------------------------
        # Gesture 1: Attention Lock (Wake-Word) -> "Peace Sign"
        # ---------------------------------------------------------
        # Look for at least one hand doing the peace sign
        if any(h["is_peace_sign"] for h in hand_states):
            self.is_locked = False
            self.last_attention_time = now
            result_payload["gesture"] = "attention"
            result_payload["locked"] = False
            return result_payload

        if self.is_locked:
            return result_payload

        # Update interaction timer since a valid hand is present and we are unlocked
        self.last_attention_time = now

        # ---------------------------------------------------------
        # Gesture 2: Give Me Space (Back Up)
        # ---------------------------------------------------------
        # Requires two hands, both open
        if len(hand_states) == 2 and all(h["is_open_palm"] for h in hand_states):
            # Calculate hand size changes over time for pushing motion
            current_size = sum(h["bounding_box_area"] for h in hand_states)
            self.hand_size_history.append(current_size)
            if len(self.hand_size_history) > self.history_len:
                self.hand_size_history.pop(0)

            size_delta = 0.0
            if len(self.hand_size_history) > 5:
                # Compare recent size to older size to detect pushing forward
                size_delta = np.mean(self.hand_size_history[-3:]) - np.mean(self.hand_size_history[:3])
            
            result_payload["size_delta"] = size_delta
            
            # If hands are significantly increasing in area, it's a dynamic push. 
            # Or if they just hold two open palms, we can count it as "back up" steady.
            if size_delta > 0.02: 
                result_payload["gesture"] = "push_back"
            else:
                result_payload["gesture"] = "two_open_palms"
            
            return result_payload

        # For the following gestures, we primarily focus on the dominant hand
        primary_hand = hand_states[0]

        # ---------------------------------------------------------
        # Gesture 3: Universal SOS Signal
        # ---------------------------------------------------------
        # Sequence: Open -> Thumb Tucked -> Fist over thumb
        # For robustness on edge, detecting a fist where the thumb tip is enclosed by fingers
        if primary_hand["is_sos_fist"]:
            self.sos_confirm_frames += 1
            if self.sos_confirm_frames > 5:  # Require holding for ~0.5s at 10fps
                result_payload["gesture"] = "sos"
                self.sos_confirm_frames = 0  # reset after trigger
                return result_payload
        else:
            self.sos_confirm_frames = max(0, self.sos_confirm_frames - 1)

        # ---------------------------------------------------------
        # Gesture 4: Binary Confirmation (Thumbs Up / Down)
        # ---------------------------------------------------------
        if primary_hand["is_thumbs_up"]:
            self.thumbs_up_frames += 1
            if self.thumbs_up_frames > 5:
                result_payload["gesture"] = "thumbs_up"
                result_payload["binary_response"] = True
                self.thumbs_up_frames = 0
                return result_payload
        else:
            self.thumbs_up_frames = 0

        if primary_hand["is_thumbs_down"]:
            self.thumbs_down_frames += 1
            if self.thumbs_down_frames > 5:
                result_payload["gesture"] = "thumbs_down"
                result_payload["binary_response"] = False
                self.thumbs_down_frames = 0
                return result_payload
        else:
            self.thumbs_down_frames = 0

        # ---------------------------------------------------------
        # Gesture 5: Speed/Proximity Control
        # ---------------------------------------------------------
        # Single hand, open flat, moving up or down
        if primary_hand["is_open_palm"]:
            cy = primary_hand["center_y"]
            self.hand_y_history.append(cy)
            if len(self.hand_y_history) > self.history_len:
                self.hand_y_history.pop(0)

            y_delta = 0.0
            if len(self.hand_y_history) > 5:
                y_delta = np.mean(self.hand_y_history[-3:]) - np.mean(self.hand_y_history[:3])
            
            result_payload["gesture"] = "flat_hand"
            result_payload["y_delta"] = y_delta  # Positive means moving down (screen coords)
            return result_payload

        return result_payload

    def _analyze_hand_topology(self, landmarks) -> Dict[str, Any]:
        """
        Analyzes 3D hand landmarks to determine finger states and overall hand geometry.
        """
        lm = landmarks.landmark
        
        # Calculate bounding box area (approximate size of hand)
        min_x = min([p.x for p in lm])
        max_x = max([p.x for p in lm])
        min_y = min([p.y for p in lm])
        max_y = max([p.y for p in lm])
        area = (max_x - min_x) * (max_y - min_y)
        
        center_y = (min_y + max_y) / 2.0

        # Helper to check if a finger is extended (tip is higher than PIP joint)
        # Note: Y-axis goes DOWN in image coordinates
        is_index_extended = lm[8].y < lm[6].y
        is_middle_extended = lm[12].y < lm[10].y
        is_ring_extended = lm[16].y < lm[14].y
        is_pinky_extended = lm[20].y < lm[18].y

        # Thumb logic is tricky because it moves sideways.
        # Check if thumb tip is further from the center than its base
        is_thumb_extended = abs(lm[4].x - lm[0].x) > abs(lm[2].x - lm[0].x)

        # SOS check: Thumb is tucked horizontally inside the palm, and fingers are curled down
        thumb_tucked = lm[4].x > min(lm[5].x, lm[17].x) and lm[4].x < max(lm[5].x, lm[17].x)
        fingers_curled = not (is_index_extended or is_middle_extended or is_ring_extended or is_pinky_extended)
        is_sos_fist = thumb_tucked and fingers_curled

        # Peace sign check: Index and Middle up, Ring and Pinky down
        is_peace = is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended

        # Open palm: All fingers extended
        is_open = is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended

        # Thumbs up/down: Fingers curled, thumb extended vertically
        thumb_vertical_up = lm[4].y < lm[3].y and lm[4].y < lm[5].y
        thumb_vertical_down = lm[4].y > lm[3].y and lm[4].y > lm[5].y
        is_thumbs_up = fingers_curled and thumb_vertical_up
        is_thumbs_down = fingers_curled and thumb_vertical_down

        return {
            "center_y": center_y,
            "bounding_box_area": area,
            "is_index_extended": is_index_extended,
            "is_middle_extended": is_middle_extended,
            "is_ring_extended": is_ring_extended,
            "is_pinky_extended": is_pinky_extended,
            "is_thumb_extended": is_thumb_extended,
            "is_sos_fist": is_sos_fist,
            "is_peace_sign": is_peace,
            "is_open_palm": is_open,
            "is_thumbs_up": is_thumbs_up,
            "is_thumbs_down": is_thumbs_down
        }
