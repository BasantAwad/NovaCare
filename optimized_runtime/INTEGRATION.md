# Integration Guide — Using Optimized Runtime with Existing NovaCare System

## Important: Non-Destructive Integration

The optimized runtime **does not modify, replace, or break** the existing NovaCare system.

✓ Original code remains untouched
✓ Original services continue running
✓ Original frontend unaffected
✓ Original database untouched
✓ Original APIs unchanged

---

## How Existing Services Integrate

### Current System (Unchanged)

```
ROOT FILES:
├── api_server.py              ← Flask LLM API (port 5000)
├── requirements.txt           ← Python dependencies
└── ... (other original files)

SERVICES:
├── services/llm-backend/      ← LLM Flask app
├── services/asl-model/        ← ASL FastAPI
├── services/robot/            ← Robot HAL
└── services/edge-tts-proxy/   ← TTS service

FRONTENDS:
├── frontend/                  ← Next.js dashboard
├── src/                       ← TypeScript source
└── novacare_app/              ← Flutter mobile app
```

### New Optimized Runtime (Alongside)

```
optimized_runtime/
├── orchestrator/              ← Task orchestration
├── state/                     ← Unified state
├── communication/             ← WebSocket server
├── adapters/                  ← Service wrappers
├── runtime/                   ← Launcher
├── robot_ui/                  ← New Robot UI
├── monitoring/                ← Health checks
├── docker/                    ← Docker configs
└── scripts/                   ← Deployment

Robot UI:
├── http://localhost:8080      ← New lightweight UI
└── ws://localhost:9999        ← State streaming
```

---

## Service Adapter Pattern

### What Service Adapters Do

Service adapters act as **lightweight HTTP clients** to existing services. They:

1. **Do NOT modify** the original service code
2. **Do NOT require changes** to original services
3. **Do NOT replace** original functionality
4. **Provide a unified interface** to disparate services

### How It Works

```
RuntimeOrchestrator
    │
    ├─→ LLMServiceAdapter
    │   └─→ HTTP GET/POST to http://localhost:5000/api/chat
    │       (Original Flask app, completely unchanged)
    │
    ├─→ ASLServiceAdapter
    │   └─→ HTTP GET/POST to http://localhost:8000/predict
    │       (Original FastAPI, completely unchanged)
    │
    ├─→ RobotServiceAdapter
    │   └─→ HTTP GET/POST to http://localhost:9000/api/*
    │       (Original Robot HAL, completely unchanged)
    │
    └─→ TTSServiceAdapter
        └─→ HTTP GET/POST to http://localhost:8002/tts
            (Original TTS service, completely unchanged)
```

### Example: Using LLM Service

**Original (unchanged)**:
```bash
# Terminal 1 - Start Flask server
cd services/llm-backend
python api_server.py  # Runs on port 5000
```

**New runtime uses it via adapter**:
```python
from optimized_runtime.adapters import get_service_registry

registry = get_service_registry()
llm_adapter = registry.get("llm")

# This makes an HTTP call to the unchanged Flask app
response = await llm_adapter.chat("Hello robot")
```

---

## Running Existing Services + Optimized Runtime

### Option 1: Keep Everything Running Separately

```bash
# Terminal 1 - Existing LLM backend (unchanged)
cd services/llm-backend
python api_server.py

# Terminal 2 - Existing ASL service (unchanged)
cd services/asl-model
python -m api.main --port 8000

# Terminal 3 - Existing Robot HAL (unchanged)
cd services/robot
python robot_service.py

# Terminal 4 - Optimized runtime
python -m optimized_runtime.runtime.launcher --mode serbot

# Terminal 5 - Robot UI (optional)
cd optimized_runtime/robot_ui
npm start
```

### Option 2: Use Docker Compose

```bash
# Start all services (original + optimized)
docker-compose -f optimized_runtime/docker/docker-compose.yml up -d
```

This Docker setup includes the optimized runtime and can wrap existing services.

### Option 3: Gradual Migration

1. Keep running existing services
2. Deploy optimized runtime alongside
3. Gradually move functionality to new runtime
4. Keep original frontend running for compatibility

---

## Migration Path (Non-Destructive)

### Phase 1: Parallel Operation (Week 1)

```
├─ Keep all existing services running
├─ Deploy optimized runtime
├─ Test service adapters
├─ Validate all existing functionality still works
└─ Deploy new Robot UI
```

### Phase 2: Service Wrapping (Week 2-3)

```
├─ Use optimized runtime for new features
├─ Keep using existing APIs for old features
├─ Gradually migrate features to new runtime
├─ Maintain backward compatibility
└─ Test extensively
```

### Phase 3: Full Integration (Week 4+)

```
├─ All services accessible via adapters
├─ Original APIs still available directly
├─ New features use optimized runtime
├─ Legacy features use original system
└─ Both coexist perfectly
```

---

## Checking Integration

### 1. Verify Services Are Running

```bash
# Check each service
curl http://localhost:5000/health      # LLM
curl http://localhost:8000/health      # ASL
curl http://localhost:9000/health      # Robot HAL
curl http://localhost:8002/health      # TTS (optional)
```

### 2. Run Health Check

```bash
bash optimized_runtime/scripts/health_check.sh localhost
```

**Expected output**:
```
[1] WebSocket server (9999): OK
[2] Robot UI (8080): OK
[3] LLM Service (5000): OK
[4] ASL Service (8000): OK
```

### 3. Test Service Adapters

```python
from optimized_runtime.adapters import get_service_registry
import asyncio

async def test():
    registry = get_service_registry()
    await registry.initialize_all()
    
    # Check health
    health = await registry.health_check_all()
    print("Service Health:")
    for service, available in health.items():
        status = "✓" if available else "✗"
        print(f"  {status} {service}")

asyncio.run(test())
```

### 4. Test Individual Adapters

```python
from optimized_runtime.adapters import (
    LLMServiceAdapter,
    ASLServiceAdapter,
)
import asyncio

async def test():
    # Test LLM
    llm = LLMServiceAdapter()
    await llm.initialize()
    health = await llm.health_check()
    print(f"LLM available: {health}")
    
    # Test ASL
    asl = ASLServiceAdapter()
    await asl.initialize()
    health = await asl.health_check()
    print(f"ASL available: {health}")

asyncio.run(test())
```

---

## Coexisting Systems

### Original Frontend (http://localhost:3000)

**Unchanged**. Can coexist with:
- Optimized runtime
- New Robot UI
- All existing APIs

```bash
# Start as before
cd frontend
npm run dev  # Still runs on port 3000
```

### New Robot UI (http://localhost:8080)

**New component**. Lightweight and optimized for SERBot.

```bash
# Start Robot UI
cd optimized_runtime/robot_ui
npm start  # Runs on port 8080
```

### Original Mobile App (novacare_app/)

**Unchanged**. Can access:
- Original backend APIs
- Optimized runtime APIs
- Both simultaneously

---

## Custom Integrations

### Adding New Service Adapter

If you need to integrate another service:

```python
# In optimized_runtime/adapters/service_adapters.py

class MyCustomServiceAdapter(ServiceAdapter):
    def __init__(self, base_url: str = "http://localhost:9999"):
        super().__init__("Custom Service", base_url)
    
    async def do_something(self, data):
        response = await self._make_request(
            "POST",
            "/api/custom",
            json=data
        )
        return response

# Register
registry = get_service_registry()
registry.register("custom", MyCustomServiceAdapter())
```

### Using Orchestrator with Existing Code

```python
from optimized_runtime.orchestrator import RuntimeOrchestrator
from existing_code.chat import get_chat_response

orch = RuntimeOrchestrator()

# Set up callback to use existing code
async def inference_handler(data):
    prompt = data.get("prompt")
    response = get_chat_response(prompt)  # Use existing function
    return response

orch.set_inference_callback(inference_handler)
```

### Listening to State Changes

```python
from optimized_runtime.state import get_robot_state

state = get_robot_state()

# Register for emotion changes
async def on_emotion_change(old, new):
    print(f"Robot emotion: {old} → {new}")
    # Could integrate with existing code here

state.on_state_change("emotion", on_emotion_change)
```

---

## API Compatibility

### Existing APIs Still Work

All original APIs continue to work:

```bash
# Original LLM API
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# Original ASL API
curl -X POST http://localhost:8000/predict \
  -F "file=@frame.jpg"

# Original Robot HAL API
curl -X POST http://localhost:9000/api/move \
  -H "Content-Type: application/json" \
  -d '{"direction": "forward", "distance": 1.0}'
```

### New Adapter APIs

Service adapters provide Pythonic async interfaces:

```python
llm = get_service_registry().get("llm")
response = await llm.chat("Hello")

asl = get_service_registry().get("asl")
result = await asl.predict_asl(frame_data)

robot = get_service_registry().get("robot")
await robot.move_forward(1.0)
```

---

## Troubleshooting Integration

### Issue: Service adapters can't reach services

**Cause**: Services not running or on wrong port

**Fix**:
1. Verify services are running
2. Check port numbers match
3. Update adapter URLs if different

```python
# If service on different port
LLMServiceAdapter(base_url="http://localhost:5001")
```

### Issue: WebSocket can't broadcast

**Cause**: Service adapters failing

**Fix**:
1. Run health checks
2. Verify each service responds
3. Check network connectivity

```bash
bash optimized_runtime/scripts/health_check.sh
```

### Issue: State updates not propagating

**Cause**: Listeners not registered

**Fix**:
1. Register state change listeners early
2. Ensure async functions are awaited
3. Check logging for errors

```python
state.on_state_change("emotion", async_callback)
```

---

## Best Practices

1. **Keep services running** - Don't stop original services
2. **Test separately** - Test each service independently first
3. **Monitor health** - Regularly run health checks
4. **Use docker-compose** - Simplifies multi-service management
5. **Log everything** - Enable debug logging during integration
6. **Test APIs first** - Curl before using adapters
7. **Graceful degradation** - Handle service failures gracefully

---

## Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design
2. Read [QUICKSTART.md](QUICKSTART.md) for setup
3. Deploy optimized runtime alongside existing services
4. Test all service integrations
5. Deploy Robot UI
6. Monitor and optimize

---

## Questions?

Check:
- `optimized_runtime/__init__.py` - Main module overview
- Individual module docstrings
- Test files for usage examples
- Docker configuration for deployment
