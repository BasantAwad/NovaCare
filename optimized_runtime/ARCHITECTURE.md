# NovaCare Optimized Runtime — Architecture Guide

## Overview

The optimized runtime is a clean, organized robotics runtime layer built alongside the existing NovaCare system. It does NOT replace existing code—it WRAPS and ORCHESTRATES it.

### Design Philosophy

✓ **Non-destructive**: Original code remains untouched
✓ **Compositional**: Uses adapters to wrap existing services
✓ **Async-first**: Optimized for performance on constrained hardware
✓ **Distributed**: Smart split between edge (SERBot) and cloud (Laptop)
✓ **Observable**: All state changes broadcast in real-time
✓ **Modular**: Clean separation of concerns

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Robot UI (Browser)                       │
│                         :8080 / ws://9999                        │
└────────┬─────────────────────────────────────────────────────────┘
         │ WebSocket
         │ State Updates
         ▼
┌────────────────────────────────────────────────────────────────────┐
│                    SERBOT (Edge Device)                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │             RuntimeOrchestrator                               │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │  │
│  │  │ Audio    │ │ Camera   │ │Inference │ │Animation │        │  │
│  │  │ Pipeline │ │ Pipeline │ │ Pipeline │ │ Pipeline │        │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌──────────────────────────┴──────────────────────────────┐       │
│  │                                                           │       │
│  ▼                                                           ▼       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐     │
│  │  RobotState      │  │ WebSocketServer  │  │  Adapters    │     │
│  │                  │  │                  │  │              │     │
│  │ • emotion        │  │ • broadcast      │  │ • LLM        │     │
│  │ • audio          │  │ • listen         │  │ • ASL        │     │
│  │ • hardware       │  │ • commands       │  │ • Robot HAL  │     │
│  │ • user context   │  │ • health         │  │ • TTS        │     │
│  └──────────────────┘  └──────────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────────────┘
         │ HTTP                    │ HTTP
         │ Requests                │ Requests
         ▼                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                   LAPTOP (Heavy AI Services)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ LLM Service  │  │ ASL Service  │  │ TTS Service  │             │
│  │ (port 5000)  │  │ (port 8000)  │  │ (port 8002)  │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  • Large language models                                           │
│  • ASL recognition (transformers)                                 │
│  • Advanced emotion detection                                     │
│  • GPU-accelerated inference                                      │
└────────────────────────────────────────────────────────────────────┘
         │ HTTP Responses
         │
         └──────────► Results streamed to SERBot
```

---

## Core Components

### 1. RuntimeOrchestrator (`orchestrator/`)

**Purpose**: Central async task coordinator

**Responsibilities**:
- Manage async pipelines (audio, camera, inference, animation)
- Queue and prioritize tasks
- Dispatch work to appropriate handlers
- Track metrics and performance
- Coordinate state updates

**Key Features**:
- Priority-based task queue (higher priority tasks first)
- Per-pipeline worker async tasks
- Callback-based event system
- Built-in performance metrics

**Usage**:
```python
from optimized_runtime.orchestrator import RuntimeOrchestrator

orchestrator = RuntimeOrchestrator()
await orchestrator.initialize()
await orchestrator.start()

# Queue tasks
await orchestrator.queue_stt(audio_data)
await orchestrator.queue_llm_inference("Hello robot")
```

---

### 2. RobotState (`state/`)

**Purpose**: Unified state container

**Manages**:
- Emotion and animation states
- Audio states (listening, speaking, amplitude)
- Hardware states (battery, CPU, sensors)
- Service availability
- User context
- Animation queues

**Features**:
- Thread-safe async operations
- Observer pattern for state changes
- Callback registration for specific state changes
- Full state snapshots
- Metadata (update count, timestamps)

**Usage**:
```python
from optimized_runtime.state import get_robot_state

state = get_robot_state()
await state.set_emotion(EmotionType.HAPPY)
await state.set_listening(True)
await state.update_battery(85.5, charging=False)

# Get full state
state_dict = state.get_full_state()
```

---

### 3. WebSocketServer (`communication/`)

**Purpose**: Real-time bidirectional communication

**Capabilities**:
- Accept multiple WebSocket connections
- Broadcast state updates to all clients
- Handle incoming commands
- Event emitter for different message types
- Health monitoring of connections

**Broadcasting**:
- Full state updates (10 Hz)
- Emotion changes
- Audio state changes
- Animation updates
- Hardware state changes

**Usage**:
```python
from optimized_runtime.communication import get_websocket_server

ws = get_websocket_server()
await ws.start()

# Broadcast to all clients
await ws.broadcast_emotion("happy")
await ws.broadcast_state(state_dict)
```

---

### 4. Service Adapters (`adapters/`)

**Purpose**: Wrapper for existing microservices

**Adapters**:
- `LLMServiceAdapter` - Wraps Flask LLM API (port 5000)
- `ASLServiceAdapter` - Wraps FastAPI ASL model (port 8000)
- `RobotServiceAdapter` - Wraps robot HAL (port 9000)
- `TTSServiceAdapter` - Wraps TTS service (port 8002)

**Features**:
- Unified HTTP client
- Health checking
- Request marshaling
- Error handling
- Service registry

**Usage**:
```python
from optimized_runtime.adapters import get_service_registry

registry = get_service_registry()
await registry.initialize_all()

# Use adapters
llm = registry.get("llm")
response = await llm.chat("Hello")

# Check health
health = await registry.health_check_all()
```

---

### 5. Robot UI (`robot_ui/`)

**Purpose**: Lightweight fullscreen animated interface

**Features**:
- Animated eyes with blinking
- Emotion-based color/animation changes
- Audio level visualization
- Battery and connection indicators
- Real-time WebSocket state sync

**Tech Stack**:
- React for UI
- TypeScript for type safety
- Framer Motion for animations
- SVG for crisp rendering
- CSS animations for effects

**Files**:
- `RobotUI.tsx` - Main React component
- `RobotUI.css` - Animations and styling
- `config.py` - Configuration management

---

### 6. RuntimeLauncher (`runtime/`)

**Purpose**: Lifecycle management and startup

**Responsibilities**:
- Initialize all subsystems
- Start orchestrator
- Handle graceful shutdown
- Support different runtime modes (serbot, laptop, debug)

**Modes**:
- `serbot` - Lightweight edge runtime
- `laptop` - Heavy AI service runtime
- `debug` - Full diagnostics and logging

**Usage**:
```python
from optimized_runtime.runtime import RuntimeLauncher

launcher = RuntimeLauncher(mode="serbot")
await launcher.run()  # Runs until KeyboardInterrupt
```

---

### 7. Monitoring (`monitoring/`)

**Purpose**: Health tracking and performance metrics

**Components**:
- `HealthMonitor` - CPU, memory, disk checks
- `PerformanceTracker` - Operation latency, error rates

**Usage**:
```python
from optimized_runtime.monitoring import HealthMonitor, PerformanceTracker

monitor = HealthMonitor()
health = await monitor.run_checks()

tracker = PerformanceTracker()
tracker.record_latency("stt", 250.5)
summary = tracker.get_summary()
```

---

## Data Flow Examples

### Example 1: User speaks to robot

```
1. Audio arrives at microphone
2. Orchestrator.queue_audio_capture() enqueued
3. Audio pipeline processes
4. Orchestrator.queue_stt() sent to STT service
5. LLMServiceAdapter.chat() calls laptop LLM
6. Response queued to TTS
7. Orchestrator.queue_tts() plays audio
8. State updated (speaking=true, amplitude changes)
9. WebSocket broadcasts to UI
10. Robot UI shows audio bars and animated mouth
```

### Example 2: Emotion detection

```
1. Camera captures frame
2. Orchestrator.queue_camera_capture() enqueued
3. Camera pipeline processes
4. Orchestrator.queue_emotion_detection() sent
5. ASLServiceAdapter calls laptop emotion model
6. Result: detected_emotion = "happy"
7. state.set_emotion(EmotionType.HAPPY) called
8. State observers notified
9. WebSocket broadcasts emotion change
10. Robot UI updates eye colors and animations
```

### Example 3: LLM response

```
1. User message STT → "What's your name?"
2. Orchestrator.queue_llm_inference() with message
3. LLMServiceAdapter.chat() calls Flask LLM
4. Response: "I'm SERBot"
5. Orchestrator.queue_tts() with response
6. Audio synthesized and played
7. state.set_speaking() updates state
8. WebSocket broadcasts speaking amplitude
9. Robot UI shows audio visualization
```

---

## Integration Points

### Existing Services (NO CHANGES REQUIRED)

1. **Flask LLM Backend** (api_server.py:5000)
   - Wraps via LLMServiceAdapter
   - No code changes needed
   - Continues operating independently

2. **FastAPI ASL Service** (services/asl-model:8000)
   - Wraps via ASLServiceAdapter
   - No code changes needed

3. **Robot HAL** (services/robot:9000)
   - Wraps via RobotServiceAdapter
   - No code changes needed

4. **Original Frontend** (frontend/:3000)
   - Continues operating independently
   - Not modified
   - Can coexist with Robot UI

5. **Mobile App** (novacare_app/)
   - Continues operating independently
   - Not modified

---

## Deployment Architecture

### SERBot Deployment

```
SERBot Device
├── optimized_runtime/
│   ├── orchestrator/
│   ├── state/
│   ├── communication/
│   ├── runtime/
│   └── adapters/
├── robot_ui/  (Node.js/Electron)
└── startup.sh
```

**What runs on SERBot**:
- RuntimeOrchestrator (lightweight)
- RobotState (in-memory)
- WebSocketServer
- Camera/Audio capture threads
- Robot UI server
- Service adapters (HTTP clients only)

**What doesn't run on SERBot**:
- Large models
- Heavy GPU inference
- Full LLM processing

### Laptop Deployment

```
Laptop Workstation
├── LLM Service (Flask:5000)
├── ASL Service (FastAPI:8000)
├── TTS Service (port 8002)
└── Robot HAL (port 9000)
```

**What runs on Laptop**:
- All heavy AI inference
- LLM model inference
- ASL recognition
- Emotion detection models
- TTS synthesis
- Optional: orchestrator in "laptop" mode

---

## Performance Characteristics

### Latency (Estimated)

| Operation | Latency |
|-----------|---------|
| State update → WebSocket broadcast | < 100ms |
| Audio capture → STT dispatch | < 50ms |
| Camera frame → inference dispatch | < 100ms |
| Task queue time (idle) | < 10ms |
| WebSocket state update cycle | ~100ms |

### Memory (Estimated)

| Component | Memory |
|-----------|--------|
| RuntimeOrchestrator | ~10 MB |
| RobotState | ~5 MB |
| WebSocketServer (10 clients) | ~20 MB |
| Task queue (100 items) | ~5 MB |
| **Total (SERBot)** | **~50 MB** |

### CPU (Estimated)

| Component | CPU |
|-----------|-----|
| Idle orchestrator | < 1% |
| Audio pipeline (active) | 5-15% |
| Camera pipeline (30 FPS) | 10-20% |
| WebSocket broadcasts | 2-5% |
| **Total (SERBot, active)** | **20-40%** |

---

## Extensibility

### Adding a New Pipeline

```python
# In orchestrator/runtime_orchestrator.py
class RuntimeOrchestrator:
    async def _custom_pipeline(self):
        while self._running:
            # Process custom tasks
            pass
    
    async def queue_custom_task(self, data):
        return await self.queue_task(
            Task(PipelineTask.CUSTOM, data, priority=5)
        )
```

### Adding a New Service Adapter

```python
# In adapters/service_adapters.py
class CustomServiceAdapter(ServiceAdapter):
    async def do_something(self, params):
        response = await self._make_request(
            "POST",
            "/endpoint",
            json=params
        )
        return response

# Register
registry = get_service_registry()
registry.register("custom", CustomServiceAdapter("http://localhost:8003"))
```

### Adding State Change Listeners

```python
from optimized_runtime.state import get_robot_state

state = get_robot_state()

async def on_emotion_change(old, new):
    print(f"Emotion: {old} → {new}")

state.on_state_change("emotion", on_emotion_change)
```

---

## Troubleshooting

### Runtime won't start

1. Check Python version (3.8+)
2. Install dependencies: `pip install -r requirements.txt`
3. Check log file: `tail -f logs/runtime_*.log`
4. Verify port availability: `netstat -tulpn | grep 9999`

### WebSocket not connecting

1. Check if server is running: `curl http://localhost:9999`
2. Verify firewall allows port 9999
3. Check browser console for errors
4. Restart both SERBot and UI

### Service adapters failing

1. Run health checks: `bash scripts/health_check.sh`
2. Verify services are running on expected ports
3. Check network connectivity
4. Review service logs

---

## Future Enhancements

- [ ] Model optimization (ONNX, TensorRT)
- [ ] Advanced priority queueing
- [ ] Service autoscaling
- [ ] Advanced monitoring dashboards
- [ ] Distributed inference across multiple devices
- [ ] Edge model inference optimization
- [ ] Advanced caching strategies
- [ ] Request batching for efficiency
