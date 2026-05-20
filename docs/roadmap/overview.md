# NovaCare — Roadmap & Tasks

> **Deadline:** May 31, 2026 · **Last Updated:** May 11, 2026 · **Team Size:** 5

---

## Status Legend

| Icon | Meaning |
|------|---------|
| ✅ | Done |
| 🔄 | In Progress |
| ⬜ | Not Started |
| 🔴 | Blocked |

---

## Phase 1: Core Software (✅ Complete)

> Foundation services and UI dashboards — all functional.

| Task | Status | Service |
|------|--------|---------|
| ASL fingerspelling model (PyTorch + MediaPipe) | ✅ | ASL Model |
| ASL FastAPI with `/predict` endpoints | ✅ | ASL Model |
| NovaBot conversational AI (Ollama + Hugging Face, dual routing) | ✅ | LLM Backend |
| Facial emotion detection (ViT model) | ✅ | LLM Backend |
| Chat API (`/api/chat` with `llm_profile` / `prefer_quality`, `/api/clear`) | ✅ | LLM Backend |
| Emotion API (`/api/emotion/detect`) | ✅ | LLM Backend |
| Rover dashboard (talk, health, meds, emergency, entertainment, navigate, help, settings) | ✅ | Frontend |
| Guardian dashboard (activity, communication, medications, settings) | ✅ | Frontend |
| Medical dashboard (vitals, records, care-plan, appointments, medications, settings) | ✅ | Frontend |
| Login & signup pages | ✅ | Frontend |
| Browser STT/TTS integration | ✅ | Frontend |
| ASL modal component | ✅ | Frontend |
| Emotion detection modal component | ✅ | Frontend |
| UI component library (Button, Card, Modal, Alert, etc.) | ✅ | Frontend |

---

## Phase 2: Repository & Infrastructure (🔄 In Progress)

> Cleaning up the codebase, unifying services, and preparing for integration.

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Unify 3 repos into single project structure | ✅ | — | `novacare/` folder created |
| One-click startup scripts (`start_all.bat`, `start_all.ps1`) | ✅ | — | Working |
| Project documentation (`docs/`) | ✅ | — | Unified guide + hardware docs updated |
| Set up `.gitignore` for unified repo | ✅ | — | |
| Push unified repo to GitHub | ⬜ | | |
| Upgrade Next.js to stable patched version | ⬜ | | Currently on 14.2.35, consider stable v14 |
| Set up CI/CD pipeline (GitHub Actions) | ⬜ | | Lint + test on PR |
| Add environment variable validation on startup | ⬜ | | Fail fast if keys missing |

---

## Phase 3: Feature Completion (⬜ Not Started)

> Must-have features to complete before deadline.

| Task | Status | Priority | Service | Notes |
|------|--------|----------|---------|-------|
| **Navigation module** (robot control via web UI) | ⬜ | 🔴 Must-have | Frontend + ROS | Core rover feature |
| **Notification service** (real-time guardian alerts) | ⬜ | 🔴 Must-have | Backend | WebSocket or push notifications |
| **Fall detection** integration | ⬜ | 🔴 Must-have | ASL Model / new service | Pose estimation + alert pipeline |
| **Database migration** (SQLite → PostgreSQL) | ⬜ | 🟡 High | Frontend backend | Use Alembic for migrations |
| **Authentication middleware** (JWT/sessions/API Keys) | 🔄 | 🟡 High | Frontend backend | Robot REST API secured; UI auth pending |
| **RAG medical queries** | ⬜ | 🟡 Medium | LLM Backend | Vector search on medical KB |
| Speech emotion recognition | ⬜ | 🟢 Nice-to-have | LLM Backend | Wav2Vec 2.0 model |
| Text emotion recognition | ⬜ | 🟢 Nice-to-have | LLM Backend | RoBERTa model |
| Multi-language support | ⬜ | 🟢 Nice-to-have | All | Arabic at minimum |

---

## Phase 4: Robot Integration (🔄 In Progress)

> Hardware integration with SERBot Prime X.

| Task | Status | Notes |
|------|--------|-------|
| Hardware Abstraction Layer (`robot_hal.py`) | ✅ | Camera, Motion, Audio, LiDAR abstractions |
| Robot REST Service (`robot_service.py` port 9000) | ✅ | Full REST API for hardware control |
| Camera integration (GStreamer + fallback) | ✅ | Replaces `cv2.VideoCapture(0)` |
| Robot TTS (gTTS → pop.AudioPlay) | 🔴 | Issues: Software success but no physical sound output |
| Robot STT (SpeechRecognition) | 🔴 | Issues: Hardware missing (`ArrayUAC10` disconnected) |
| Movement API (pop.Pilot.SerBot) | ✅ | Forward, backward, left, right, turn |
| Navigation with LiDAR obstacle avoidance | ✅ | Software logic ready; physical LiDAR unit failing |
| Follow-user mode (face tracking + movement) | ✅ | Implemented high-speed MediaPipe + OpenCV fallback |
| Frontend robot-api.ts client | ✅ | TypeScript API for all robot endpoints |
| Navigate page → real movement API | ✅ | Buttons call robot REST API |
| Talk page → robot TTS/STT | 🔴 | Blocked by physical speaker/mic failure |
| Emotion modal → robot camera | ✅ | MJPEG stream or frame polling |
| ASL modal → robot camera | ✅ | Frame capture from robot camera |
| `CameraEmotionPoller` → robot camera | ✅ | REST API with local webcam fallback |
| Robot startup script (Linux) | ✅ | `scripts/jetson/start_robot.sh` |
| Chromium kiosk mode for touchscreen | ✅ | Auto-launch on 7-inch display |
| Set up ROS 2 on JetAuto Kit | ⬜ | Optional: for advanced SLAM |
| Camera feed streaming to guardian dashboard | 🔄 | MJPEG endpoint ready |
| SLAM + obstacle avoidance integration | 🔄 | Logic functional; blocked by LiDAR hardware |
| On-device vs cloud deployment decision | ✅ | Hybrid: on-device HAL + cloud LLM APIs |

---

## Phase 5: Testing & Polish (⬜ Not Started)

> Final validation before graduation demo.

| Task | Status | Notes |
|------|--------|-------|
| Write API tests for all endpoints | ⬜ | pytest + requests |
| Write frontend component tests | ⬜ | React Testing Library |
| Integration testing (all 3 services) | ⬜ | End-to-end flow |
| Performance benchmarks (latency, accuracy) | ⬜ | ASL < 200ms, LLM < 3s |
| Accessibility audit (WCAG 2.1) | ⬜ | Screen reader, contrast, keyboard nav |
| User acceptance testing | ⬜ | Target ≥ 4.0/5.0 rating |
| Security review (secrets, CORS, auth) | ⬜ | |
| Final demo preparation | ⬜ | Presentation + live demo |

---

## Suggested Weekly Milestones

| Week | Dates | Focus |
|------|-------|-------|
| **W1** | Mar 9–15 | Finish repo cleanup, push to GitHub, set up CI |
| **W2** | Mar 16–22 | Navigation module + notification service skeleton |
| **W3** | Mar 23–29 | Fall detection + database migration |
| **W4** | Mar 30–Apr 5 | Authentication + RAG medical queries |
| **W5** | Apr 6–12 | Robot integration begins (ROS 2 setup) |
| **W6** | Apr 13–19 | Edge deployment + camera streaming |
| **W7** | Apr 20–26 | SLAM + navigation + follow-user |
| **W8** | Apr 27–May 3 | Integration testing + bug fixes |
| **W9** | May 4–10 | Performance tuning + accessibility audit |
| **W10** | May 11–17 | User testing + polish |
| **W11** | May 18–24 | Final demo prep + documentation |
| **W12** | May 25–31 | **🎓 Submission deadline** |

---

> **Note:** This roadmap should be updated weekly during team meetings. Mark tasks as 🔄 when started and ✅ when complete.
