"""
Simple camera vision test for obstacle detection.
Run this on the SERBot (without NOVACARE_LIGHTWEIGHT) to print edge densities
for left/center/right regions of the lower half of the camera frame.
"""

import time
import sys
import os

# Ensure services package is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from services.robot.robot_hal import get_robot

import cv2
import numpy as np

robot = get_robot()

print("Starting vision test. Press Ctrl+C to stop.")
try:
    while True:
        ret, frame = robot.camera.read_frame()
        if not ret or frame is None:
            print("No frame available")
            time.sleep(0.5)
            continue

        h, w = frame.shape[:2]
        roi = frame[int(h * 0.45):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        third = edges.shape[1] // 3
        left_density = np.count_nonzero(edges[:, :third]) / (edges.shape[0] * max(1, third))
        center_density = np.count_nonzero(edges[:, third:2*third]) / (edges.shape[0] * max(1, third))
        right_density = np.count_nonzero(edges[:, 2*third:]) / (edges.shape[0] * max(1, third))

        print(f"L={left_density:.4f}  C={center_density:.4f}  R={right_density:.4f}")
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Vision test stopped")
    robot.camera.release()
