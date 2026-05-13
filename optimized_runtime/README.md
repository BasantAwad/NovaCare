# NovaCare Optimized Robotics Runtime v2.0

> A clean, async-first robotics runtime layer for the NovaCare SERBot system.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Status: Active Development](https://img.shields.io/badge/status-active-green)]()

---

## 🎯 Overview

The optimized runtime is a modern, async-first robotics framework designed specifically for the NovaCare SERBot hardware. It provides:

- **Centralized Orchestration** - Async task pipelines for audio, camera, inference, and animation
- **Unified State Management** - Single source of truth for all robot state
- **Real-time Communication** - WebSocket server for live UI updates
- **Service Integration** - Adapters for existing microservices (no modifications needed)
- **Lightweight Robot UI** - Animated fullscreen interface with emotion display
- **Distributed Architecture** - Smart split between edge (SERBot) and cloud (Laptop)
- **Production Ready** - Docker deployment, health monitoring, comprehensive logging

### Key Features

✅ **Non-destructive** - Existing code remains completely untouched
✅ **Async-first** - Optimized for performance on constrained hardware  
✅ **Composable** - Clean module architecture with clear separation of concerns  
✅ **Observable** - All state changes broadcast in real-time via WebSocket
✅ **Distributed** - Intelligent task distribution between edge and cloud
✅ **Monitored** - Built-in health checks and performance tracking
✅ **Deployable** - Docker, Docker Compose, SCP deployment scripts included

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt.runtime
```

### 2. Start the Runtime

```bash
# Debug mode (recommended for development)
python -m optimized_runtime.runtime.launcher --mode debug

# SERBot edge mode
python -m optimized_runtime.runtime.launcher --mode serbot

# Laptop heavy AI mode
python -m optimized_runtime.runtime.launcher --mode laptop
```

### 3. Open Robot UI

```bash
cd optimized_runtime/robot_ui
npm install && npm start
```

Visit `http://localhost:3000` in your browser.

### 4. Health Check

```bash
bash optimized_runtime/scripts/health_check.sh
```

---

## 📋 Architecture

```
┌─────────────────┐
│    Robot UI     │ ← Animated eyes, emotion display, status
│ (Browser/React) │
└────────┬────────┘
         │ WebSocket (ws://9999)
         ▼
┌────────────────────────────────────────┐
│      SERBOT (Edge Device)              │
│  ┌──────────────────────────────────┐  │
│  │   RuntimeOrchestrator             │  │ ← Central task coordinator
│  │  - Audio Pipeline                │  │
│  │  - Camera Pipeline               │  │
│  │  - Inference Pipeline            │  │
│  │  - Animation Pipeline            │  │
│  └──────────────────────────────────┘  │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │   RobotState                      │  │ ← Unified state container
│  │  - Emotion, Audio, Hardware       │  │
│  │  - User context, Animations      │  │
│  └──────────────────────────────────┘  │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │  WebSocketServer                  │  │ ← Real-time broadcasts
│  │  - State updates (10 Hz)          │  │
│  │  - Command handling               │  │
│  └──────────────────────────────────┘  │
└────────┬─────────────────────────────────┘
         │ HTTP Requests
         ▼
┌────────────────────────────────────────┐
│    Existing NovaCare Services          │
│  (LLM, ASL, Robot HAL, TTS)            │ ← Via service adapters
│  (Completely unchanged)                │
└────────────────────────────────────────┘
```

For detailed architecture, see [ARCHITECTURE.md](optimized_runtime/ARCHITECTURE.md).

---

## 📁 Project Structure

```
optimized_runtime/
├── orchestrator/              # Central task orchestrator
│   └── runtime_orchestrator.py
├── state/                     # Unified robot state
│   └── robot_state.py
├── communication/             # WebSocket server
│   └── websocket_server.py
├── adapters/                  # Service wrappers (non-destructive)
│   └── service_adapters.py
├── runtime/                   # Runtime lifecycle
│   └── launcher.py
├── robot_ui/                  # Lightweight animated UI
│   ├── RobotUI.tsx
│   ├── RobotUI.css
│   └── config.py
├── hardware/                  # Hardware abstraction (future)
├── inference/                 # Inference pipeline (future)
├── monitoring/                # Health monitoring
│   └── health.py
├── deployment/                # Deployment config
├── docker/                    # Docker configuration
│   ├── Dockerfile.serbot
│   ├── Dockerfile.laptop
│   └── docker-compose.yml
├── scripts/                   # Deployment scripts
│   ├── deploy_serbot.sh
│   ├── startup.sh
│   └── health_check.sh
├── ARCHITECTURE.md            # Detailed architecture guide
├── QUICKSTART.md              # Getting started guide
└── INTEGRATION.md             # Integration with existing system
```

---

## 🔌 Integration

The runtime **wraps existing services** with adapters—no modifications required:

```python
from optimized_runtime.adapters import get_service_registry

registry = get_service_registry()
await registry.initialize_all()

# Use wrapped services seamlessly
llm = registry.get("llm")
response = await llm.chat("Hello robot")

asl = registry.get("asl")
result = await asl.predict_asl(frame_data)
```

- **Existing Flask LLM API** (port 5000) → `LLMServiceAdapter`
- **Existing FastAPI ASL** (port 8000) → `ASLServiceAdapter`
- **Existing Robot HAL** (port 9000) → `RobotServiceAdapter`
- **Existing TTS Service** (port 8002) → `TTSServiceAdapter`

See [INTEGRATION.md](optimized_runtime/INTEGRATION.md) for complete details.

---

## 🎮 Core Components

### RuntimeOrchestrator

Manages async pipelines and task queuing:

```python
from optimized_runtime import RuntimeOrchestrator

orch = RuntimeOrchestrator()
await orch.initialize()
await orch.start()

# Queue tasks with priority
await orch.queue_stt(audio_data, priority=10)
await orch.queue_llm_inference("Hello", priority=10)
await orch.queue_emotion_detection(frame, priority=3)
```

### RobotState

Unified state with observer pattern:

```python
from optimized_runtime import RobotState, EmotionType

state = get_robot_state()

# Update state
await state.set_emotion(EmotionType.HAPPY)
await state.set_listening(True)
await state.update_battery(85.5, charging=False)

# Listen for changes
state.on_state_change("emotion", callback)

# Get full state
state_dict = state.get_full_state()
```

### WebSocketServer

Real-time bidirectional communication:

```python
from optimized_runtime.communication import get_websocket_server

ws = get_websocket_server()
await ws.start()

# Broadcast to all clients
await ws.broadcast_emotion("happy")
await ws.broadcast_state(state_dict)

# Handle incoming commands
ws.register_handler("command", handle_command)
```

### Service Adapters

Wrap existing microservices:

```python
from optimized_runtime.adapters import (
    LLMServiceAdapter,
    ASLServiceAdapter,
    RobotServiceAdapter,
)

llm = LLMServiceAdapter()
await llm.initialize()

response = await llm.chat("What's your name?")
await llm.clear_history()
```

---

## 📊 Performance

### Latency (Estimated)

| Operation | Latency |
|-----------|---------|
| State update → WebSocket broadcast | < 100ms |
| Audio capture → STT dispatch | < 50ms |
| Camera frame → inference dispatch | < 100ms |

### Memory (Estimated)

| Component | Memory |
|-----------|--------|
| RuntimeOrchestrator | ~10 MB |
| RobotState | ~5 MB |
| WebSocketServer (10 clients) | ~20 MB |
| **Total (SERBot)** | **~50 MB** |

### CPU (Estimated)

| Component | CPU |
|-----------|-----|
| Idle | < 1% |
| Audio active | 5-15% |
| Camera 30 FPS | 10-20% |
| WebSocket broadcasts | 2-5% |
| **Total (active)** | **20-40%** |

---

## 🐳 Deployment

### Docker

```bash
# Build images
docker build -t novacare-runtime-serbot \
  -f optimized_runtime/docker/Dockerfile.serbot .

docker build -t novacare-runtime-laptop \
  -f optimized_runtime/docker/Dockerfile.laptop .

# Run with docker-compose
docker-compose -f optimized_runtime/docker/docker-compose.yml up -d
```

### SERBot Hardware

```bash
# One-command deployment
bash optimized_runtime/scripts/deploy_serbot.sh 192.168.1.100 ubuntu

# Or manual
scp -r optimized_runtime ubuntu@192.168.1.100:~/
ssh ubuntu@192.168.1.100 "cd optimized_runtime && python -m optimized_runtime.runtime.launcher --mode serbot"
```

### Health Check

```bash
bash optimized_runtime/scripts/health_check.sh

# Or remote
bash optimized_runtime/scripts/health_check.sh 192.168.1.100
```

---

## 📖 Documentation

- **[ARCHITECTURE.md](optimized_runtime/ARCHITECTURE.md)** - Detailed system design and data flow
- **[QUICKSTART.md](optimized_runtime/QUICKSTART.md)** - Getting started guide with examples
- **[INTEGRATION.md](optimized_runtime/INTEGRATION.md)** - Using existing services without modification
- **API Reference** - Docstrings in each module
- **Examples** - Usage examples throughout

---

## 🔧 Configuration

### Environment Variables

```bash
export ROBOT_MODE=serbot              # serbot, laptop, or debug
export ORCHESTRATOR_PORT=9999
export LLM_SERVICE_URL=http://localhost:5000
export ASL_SERVICE_URL=http://localhost:8000
export LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
```

### Config File

Edit `optimized_runtime/robot_ui/config.json`:

```json
{
  "websocket": {
    "url": "ws://localhost:9999",
    "reconnect_interval": 1000
  },
  "ui": {
    "fullscreen": true,
    "fps": 60,
    "theme": "dark"
  }
}
```

---

## 🧪 Testing

### Run Health Checks

```python
from optimized_runtime.monitoring import HealthMonitor

monitor = HealthMonitor()
health = await monitor.run_checks()
print(health)
```

### Check Metrics

```python
metrics = orchestrator.get_metrics()
print(f"Tasks processed: {metrics['tasks_processed']}")
print(f"Queue depth: {metrics['queue_depth']}")
```

### Test Service Adapters

```python
registry = get_service_registry()
health = await registry.health_check_all()
print(health)  # {'llm': True, 'asl': False, ...}
```

---

## 🚨 Troubleshooting

### WebSocket won't start
```bash
lsof -i :9999          # Check port
kill -9 <PID>          # Kill process using port
```

### Service adapters unavailable
```bash
bash optimized_runtime/scripts/health_check.sh
curl http://localhost:5000/health       # Test each service
```

### High CPU/Memory
- Reduce WebSocket broadcast frequency
- Lower camera FPS
- Reduce queue size
- Profile with Python profiler

See [QUICKSTART.md](optimized_runtime/QUICKSTART.md#troubleshooting) for more.

---

## 🔮 Future Enhancements

- [ ] Model optimization (ONNX, TensorRT, quantization)
- [ ] Advanced priority queueing algorithms
- [ ] Service auto-scaling
- [ ] Comprehensive monitoring dashboard
- [ ] Distributed inference across multiple devices
- [ ] Edge model optimization for ARM/Jetson
- [ ] Advanced caching strategies
- [ ] Request batching for efficiency
- [ ] WebSocket compression
- [ ] Multi-robot coordination

---

## 🤝 Contributing

This runtime layer is designed to be extended:

1. **Add new pipeline** - Create async worker in `orchestrator/`
2. **Add service adapter** - Extend `ServiceAdapter` in `adapters/`
3. **Add UI component** - Extend React UI in `robot_ui/`
4. **Improve performance** - Profile and optimize pipelines

---

## 📝 License

MIT License - See [LICENSE](LICENSE) (if present)

---

## 🙏 Acknowledgments

Built for the NovaCare project using:
- Python asyncio for efficient concurrency
- WebSockets for real-time communication
- React for the UI
- Docker for reproducible deployments

---

## 📞 Support

For questions and issues:

1. Check documentation: [ARCHITECTURE.md](optimized_runtime/ARCHITECTURE.md), [QUICKSTART.md](optimized_runtime/QUICKSTART.md)
2. Review module docstrings and examples
3. Run health checks: `bash optimized_runtime/scripts/health_check.sh`
4. Check logs: `tail -f logs/runtime_*.log`
5. Enable debug logging: `--log-level DEBUG`

---

## Version

**v2.0.0** - First production release (May 2026)

- Complete async orchestration
- Service adapters for all existing microservices
- Lightweight Robot UI
- Docker deployment
- SERBot hardware optimization
- Distributed edge/cloud architecture

---

**Built with ❤️ for social robots**
