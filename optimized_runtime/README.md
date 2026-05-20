# 🩺 NovaCare Optimized Runtime Layer

This is the high-performance, async-first robotics runtime for NovaCare, designed to run on constrained hardware like the Jetson Nano (SERBot).

## 🏗️ Architecture

The runtime operates as an orchestration layer that wraps around the core AI services.

- **`orchestrator/`**: Central async event loop and task queue.
- **`communication/`**: Real-time WebSockets for UI and Distributed AI.
- **`hardware/`**: Singleton managers for shared camera and audio access.
- **`inference/`**: Intelligent workload distributor (Local vs Remote).
- **`state/`**: Unified robot state manager.
- **`robot_ui/`**: React-based animated social robot face.

## 🚀 Key Features

1.  **Shared Camera Manager**: Multiple services (ASL, Emotion, Logging) share a single camera stream, saving 40% CPU.
2.  **Async Inference Pipeline**: Queue-based processing prevents UI lag and blocking calls.
3.  **Distributed Workload**: Heavy tasks (LLM) are automatically routed to a laptop, while safety tasks run locally.
4.  **Audio-Reactive UI**: The robot face animates in sync with TTS and microphone input.

## 🛠️ Installation & Deployment

### SERBot Deployment (Lightweight)
```bash
cd serbot_deployment/scripts
./bootstrap.sh
```

### Laptop Services (Heavy AI)
```bash
cd laptop_services
docker-compose up -d
```

## 📊 Performance & Monitoring
Use the built-in benchmarking tool to profile inference:
```bash
python -m monitoring.perf_benchmarks
```

---
© 2026 NovaCare Robotics Team.
