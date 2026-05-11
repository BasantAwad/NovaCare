# SERBot Hardware Integration & Progress Guide

**Purpose:** This document is the source of truth for the SERBot hardware integration. Every team member MUST read this before starting work and MUST update the `Progress Log` before finishing their session.

## 📅 Progress Log

### [2026-05-11] - Diagnostic Testing & AI Tracking Implementation

**Completed by:** Basant , Nadira

- **Camera Feed:** ✅ **FULLY WORKING.** Integrated high-speed face tracking using **MediaPipe**. Implemented an optimized fallback that uses downscaled OpenCV Haar Cascades (4x reduction) to eliminate lag and prevent system crashes on the Jetson Nano.
- **LiDAR / SLAM:** ❌ **HARDWARE FAILURE.** Initializing `startMotor()` causes a consistent C-level Segmentation Fault. Diagnostics confirmed no software locks on `/dev/ttyUSB0`. Suspected physical defect or power failure in the LiDAR unit. Dashboard now uses a resilient **multiprocessing worker** to provide mock SLAM data for UI testing.
- **Obstacle Avoidance:** ✅ **IMPLEMENTED.** Manual forward movement is now programmatically intercepted if the (mock/real) LiDAR detects an object within 300mm.
- **Speaker & Mic:** ⚠️ **UNSURE / HARDWARE ISSUES.**
  - **Speaker:** Software (espeak) reports success, but no physical sound is heard.
  - **Microphone:** `arecord` confirms hardware is missing/disconnected (`Unknown PCM ArrayUAC10`).
- **Next Steps:** Contact Hiwonder support regarding the peripheral expansion board failure. Continue software development using the mock SLAM fallback.

---

### [2026-05-10] - Initial Hardware Connection & API Upgrades

**Completed by:** Basant, Nadira, Ramez

- **What we did:** We successfully connected to the SERBot and started testing its movement and camera using the simple Flask UI in `test_robot.py`.
- **Upgrades made:** Upgraded the brittle OpenCV face tracking and dead-reckoning navigation in the main API (`services/robot/robot_service.py`) to utilize SERBot's built-in native tracking and SLAM navigation functions (`bot.tracking` and `bot.navigation`). Added an API key security layer (`NOVACARE_API_KEY`) to the robot endpoints to block unauthorized access over the network.
- **Issues/Blockers:** We ran out of time to fully test the camera feed over the network.
- **Next Steps:** The next member needs to test the camera feed, verify the native `navigate` and `follow` endpoints work properly on the physical robot, and document the results here.

---

## 🛠️ Hardware Integration Files

These are the core files responsible for translating web/software commands into physical robot actions.

| File                              | Purpose                                                                          | How to Use It                                                                                                                                                                                                                                 |
| --------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_robot.py`                   | A lightweight, standalone Flask script for quick hardware validation.            | Run `python test_robot.py` on the robot to get a basic web UI (on port 5000) with arrow buttons to test the motors and view the raw camera feed without booting the entire NovaCare backend.                                                  |
| `services/robot/robot_hal.py`     | Hardware Abstraction Layer (HAL). It wraps the proprietary `pop` Python library. | Do not import `pop` anywhere else. If you need to add a new hardware feature (e.g., read a sensor), add it to a class here (like `MotionHAL` or `CameraHAL`) so that it gracefully mocks itself if run on a laptop without the `pop` library. |
| `services/robot/robot_service.py` | The REST API that exposes the HAL to the rest of the NovaCare network.           | Runs on port 9000. It provides secure endpoints like `POST /api/navigate` and `POST /api/follow/start`. This is what the Next.js frontend calls to move the robot.                                                                            |

---

## 🗺️ Hardware Roadmap

| Task                                          | Status     | Priority | Notes                                                          |
| --------------------------------------------- | ---------- | -------- | -------------------------------------------------------------- |
| Secure API endpoints with API Key             | ✅ Done    | High     | Implemented in `robot_service.py`                              |
| Migrate from OpenCV to native SERBot Tracking | ✅ Done    | High     | `bot.tracking()` implemented in HAL                            |
| Migrate from Dead-Reckoning to native SLAM    | ✅ Done    | High     | `bot.navigation()` implemented in HAL                          |
| Validate Camera Feed                          | 🔄 Next    | High     | Use `test_robot.py` to ensure GStreamer/cv2 works on the robot |
| Test Native Tracking / SLAM on physical floor | ⬜ Pending | High     | Verify the new HAL implementations actually move the wheels    |
| Integrate Frontend with API Key               | ⬜ Pending | High     | Update Next.js API calls to include `X-API-Key` header         |
| Connect ROS 2 (Optional)                      | ⬜ Pending | Low      | Only if the `pop` library's native SLAM is insufficient        |

---

## 📝 Instructions for the Next Member

1. **Read** the progress log above.
2. **Review** the code updates in `services/robot/robot_hal.py` and `services/robot/robot_service.py`.
3. **Task:** Boot the robot, run `test_robot.py` and verify the camera feed. Then boot the main `robot_service.py` (port 9000) and test the `/api/follow/start` and `/api/navigate` endpoints via an HTTP client (Postman/curl) passing the `X-API-Key` header.
4. **Update** the Progress Log at the top of this document with your findings before you log off!
