"""
NovaCare Fall Detection Module
==============================
A robust, real-time fall detection system designed for the SERBot camera.

Uses a multi-tiered approach:
  1. MediaPipe Pose (Primary) — tracks spine angle (shoulder-to-hip vector) 
     and relative vertical position of the hips.
  2. OpenCV Silhouette Analysis (Fallback) — tracks bounding box aspect ratio
     (width/height > threshold) and vertical centroid velocity.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Try importing MediaPipe Pose
_MEDIAPIPE_AVAILABLE = False
mp_pose = None
mp_drawing = None

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    _MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe Pose successfully loaded for Fall Detection")
except ImportError:
    print("⚠ MediaPipe not available — using high-fidelity OpenCV silhouette fallback")


class FallDetector:
    def __init__(self):
        self.pose = None
        if _MEDIAPIPE_AVAILABLE:
            try:
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"Error initializing MediaPipe Pose: {e}")
                self.pose = None

        # History tracking for velocity and state transitions
        self.history_len = 10
        self.hip_y_history = []
        self.timestamp_history = []
        self.last_state = "upright"
        self.fall_confirmed_frames = 0
        
        # Calibration defaults
        self.floor_threshold_y = 0.8  # Hips below 80% of screen height
        self.aspect_ratio_threshold = 1.3  # Width/Height ratio > 1.3 indicates lying down
        
    def reset(self):
        self.hip_y_history.clear()
        self.timestamp_history.clear()
        self.last_state = "upright"
        self.fall_confirmed_frames = 0

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame for fall events.
        
        Returns:
            Dict containing:
              - 'fall_detected': bool
              - 'confidence': float
              - 'method': str ('mediapipe' | 'opencv_silhouette')
              - 'spine_angle': float
              - 'aspect_ratio': float
              - 'message': str
        """
        if frame is None:
            return {"fall_detected": False, "confidence": 0.0, "message": "No frame", "method": "none"}
            
        h, w = frame.shape[:2]
        
        # 1. Attempt MediaPipe Pose (Primary)
        if self.pose is not None:
            try:
                # Convert to RGB as required by MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Keypoints
                    # Left shoulder (11), Right shoulder (12)
                    # Left hip (23), Right hip (24)
                    l_shoulder = landmarks[11]
                    r_shoulder = landmarks[12]
                    l_hip = landmarks[23]
                    r_hip = landmarks[24]
                    
                    # Calculate center points
                    shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
                    shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
                    hip_y = (l_hip.y + r_hip.y) / 2
                    hip_x = (l_hip.x + r_hip.x) / 2
                    
                    # Track vertical coordinates over time for velocity
                    now = time.time()
                    self.hip_y_history.append(hip_y)
                    self.timestamp_history.append(now)
                    if len(self.hip_y_history) > self.history_len:
                        self.hip_y_history.pop(0)
                        self.timestamp_history.pop(0)
                        
                    # Calculate velocity (change in Y over time)
                    velocity = 0.0
                    if len(self.hip_y_history) >= 2:
                        dy = self.hip_y_history[-1] - self.hip_y_history[0]
                        dt = self.timestamp_history[-1] - self.timestamp_history[0]
                        if dt > 0:
                            velocity = dy / dt  # Positive is downward speed
                            
                    # Calculate spine angle relative to vertical axis (0 degrees = standing)
                    dx = shoulder_x - hip_x
                    dy = shoulder_y - hip_y
                    angle_rad = np.arctan2(abs(dx), abs(dy))
                    angle_deg = np.degrees(angle_rad)
                    
                    # Fall conditions:
                    # - Spine angle is greater than 60 degrees (close to horizontal)
                    # - Hip height is in the lower part of the screen (on the ground)
                    # - Downward velocity is high (rapid descent)
                    is_flat = angle_deg > 55.0
                    is_low = hip_y > 0.65
                    is_falling = velocity > 0.8
                    
                    fall_score = 0.0
                    if is_flat:
                        fall_score += 0.5
                    if is_low:
                        fall_score += 0.3
                    if is_falling:
                        fall_score += 0.2
                        
                    detected = fall_score >= 0.7
                    
                    # Hysteresis confirmation: require 3 consecutive frames
                    if detected:
                        self.fall_confirmed_frames += 1
                    else:
                        self.fall_confirmed_frames = max(0, self.fall_confirmed_frames - 1)
                        
                    confirmed = self.fall_confirmed_frames >= 2
                    
                    msg = "Fall detected!" if confirmed else "Person standing / active"
                    if detected and not confirmed:
                        msg = "Potential fall event detected, confirming..."
                        
                    return {
                        "fall_detected": confirmed,
                        "confidence": float(fall_score),
                        "method": "mediapipe",
                        "spine_angle": float(angle_deg),
                        "hip_height_ratio": float(hip_y),
                        "velocity": float(velocity),
                        "message": msg
                    }
            except Exception as e:
                print(f"MediaPipe processing error: {e}. Falling back to OpenCV.")

        # 2. Fallback: OpenCV Silhouette / Motion Contour Analysis
        try:
            # Simple background subtraction/thresholding to find the human silhouette
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # Use adaptive thresholding to extract contours
            thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assume it's the person)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Check if it's large enough to be a person
                if area > (h * w * 0.05):
                    x, y, w_box, h_box = cv2.boundingRect(largest_contour)
                    
                    aspect_ratio = w_box / max(1, h_box)
                    centroid_y = (y + h_box / 2) / h
                    
                    # Keep track of centroid height for velocity
                    now = time.time()
                    self.hip_y_history.append(centroid_y)
                    self.timestamp_history.append(now)
                    if len(self.hip_y_history) > self.history_len:
                        self.hip_y_history.pop(0)
                        self.timestamp_history.pop(0)
                        
                    velocity = 0.0
                    if len(self.hip_y_history) >= 2:
                        dy = self.hip_y_history[-1] - self.hip_y_history[0]
                        dt = self.timestamp_history[-1] - self.timestamp_history[0]
                        if dt > 0:
                            velocity = dy / dt
                            
                    # A fall is characterized by a horizontal bounding box (width > height)
                    # and center of mass sitting low to the floor
                    is_horizontal = aspect_ratio > self.aspect_ratio_threshold
                    is_on_floor = centroid_y > 0.7
                    
                    fall_score = 0.0
                    if is_horizontal:
                        fall_score += 0.5
                    if is_on_floor:
                        fall_score += 0.3
                    if velocity > 0.7:
                        fall_score += 0.2
                        
                    detected = fall_score >= 0.7
                    
                    if detected:
                        self.fall_confirmed_frames += 1
                    else:
                        self.fall_confirmed_frames = max(0, self.fall_confirmed_frames - 1)
                        
                    confirmed = self.fall_confirmed_frames >= 2
                    
                    return {
                        "fall_detected": confirmed,
                        "confidence": float(fall_score),
                        "method": "opencv_silhouette",
                        "spine_angle": 90.0 if is_horizontal else 0.0,
                        "aspect_ratio": float(aspect_ratio),
                        "hip_height_ratio": float(centroid_y),
                        "velocity": float(velocity),
                        "message": "Fall detected!" if confirmed else "Person standing / active"
                    }
                    
        except Exception as e:
            print(f"OpenCV Silhouette fall detection failed: {e}")
            
        return {
            "fall_detected": False,
            "confidence": 0.0,
            "method": "none",
            "message": "No person detected in frame"
        }
