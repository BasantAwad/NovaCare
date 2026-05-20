# Quick Start Guide — NovaCare Optimized Runtime

## Prerequisites

- Python 3.8+
- pip/poetry for package management
- Docker & Docker Compose (for deployment)
- Node.js 16+ (for Robot UI)
- SERBot hardware running Ubuntu 20.04+

## Installation

### 1. Clone/Extract the Project

```bash
cd /path/to/NovaCare-1
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt.runtime
```

**Key dependencies**:
- `asyncio` - Async runtime (built-in)
- `websockets` - WebSocket server
- `httpx` - Async HTTP client
- `psutil` - System monitoring

### 3. Install Robot UI Dependencies (optional)

```bash
cd optimized_runtime/robot_ui
npm install
```

---

## Running the Runtime

### Local Development (Debug Mode)

```bash
python -m optimized_runtime.runtime.launcher --mode debug --log-level DEBUG
```

**Output**:
```
2026-05-13 14:23:45,123 - __main__ - INFO - Starting NovaCare optimized runtime (mode=debug)
2026-05-13 14:23:45,456 - __main__ - INFO - Initializing RuntimeOrchestrator...
2026-05-13 14:23:45,789 - __main__ - INFO - WebSocket server started on ws://0.0.0.0:9999
...
```

### SERBot Edge Runtime

```bash
python -m optimized_runtime.runtime.launcher --mode serbot --log-level INFO
```

### Laptop Heavy AI Services

```bash
python -m optimized_runtime.runtime.launcher --mode laptop --log-level INFO
```

---

## Accessing the Robot UI

### 1. Start a Local Server

```bash
cd optimized_runtime/robot_ui
npm start
```

Or use the development build:

```bash
npm run dev
```

### 2. Open in Browser

```
http://localhost:3000
```

### 3. Expected UI

- Two animated eyes in center
- Status bar at bottom showing:
  - Connection status
  - Battery level
- Emotion text displayed
- Audio visualization when speaking/listening

---

## Docker Deployment

### Build Images

```bash
# Build SERBot runtime
docker build -t novacare-runtime-serbot \
  -f optimized_runtime/docker/Dockerfile.serbot .

# Build Laptop services
docker build -t novacare-runtime-laptop \
  -f optimized_runtime/docker/Dockerfile.laptop .
```

### Run with Docker Compose

```bash
docker-compose -f optimized_runtime/docker/docker-compose.yml up -d
```

### Stop Services

```bash
docker-compose -f optimized_runtime/docker/docker-compose.yml down
```

---

## Deploying to SERBot Device

### Prerequisites

- SERBot IP address (e.g., 192.168.1.100)
- SSH access (user: ubuntu, or your user)
- Network connectivity

### One-Command Deployment

```bash
bash optimized_runtime/scripts/deploy_serbot.sh 192.168.1.100 ubuntu
```

**This**:
1. Creates remote directory
2. Copies runtime code
3. Copies Docker files
4. Copies startup script
5. Starts the runtime

### Manual Step-by-Step

```bash
# 1. SSH into SERBot
ssh ubuntu@192.168.1.100

# 2. Create directory
mkdir -p ~/novacare-runtime

# 3. Copy files (from your PC)
scp -r optimized_runtime ubuntu@192.168.1.100:~/novacare-runtime/

# 4. Start runtime
ssh ubuntu@192.168.1.100 "cd ~/novacare-runtime && python3 -m optimized_runtime.runtime.launcher --mode serbot"
```

---

## Health Checking

### Quick Check

```bash
bash optimized_runtime/scripts/health_check.sh localhost
```

### Remote Health Check

```bash
bash optimized_runtime/scripts/health_check.sh 192.168.1.100
```

**Output**:
```
[1] Checking WebSocket server...
✓ WebSocket server (9999): OK

[2] Checking Robot UI server...
✓ Robot UI (8080): OK

[3] Checking LLM service...
✓ LLM Service (5000): OK

[4] Checking ASL service...
✓ ASL Service (8000): OK
```

---

## Debugging

### Enable Debug Logging

```bash
python -m optimized_runtime.runtime.launcher --mode debug --log-level DEBUG 2>&1 | tee runtime.log
```

### Monitor WebSocket Connections

```python
# In a Python script
from optimized_runtime.communication import get_websocket_server

ws = get_websocket_server()
print(f"Active connections: {ws.connection_count}")
```

### Check System Health

```python
from optimized_runtime.monitoring import HealthMonitor
import asyncio

async def check():
    monitor = HealthMonitor()
    health = await monitor.run_checks()
    print(health)

asyncio.run(check())
```

### Check Orchestrator Metrics

```python
from optimized_runtime.orchestrator import RuntimeOrchestrator
import asyncio

async def check():
    orch = RuntimeOrchestrator()
    await orch.initialize()
    
    # Simulate some operations
    await asyncio.sleep(1)
    
    metrics = orch.get_metrics()
    print(metrics)

asyncio.run(check())
```

---

## Common Issues & Solutions

### Issue: WebSocket server won't start

**Symptom**: `OSError: Address already in use`

**Solution**:
```bash
# Find process using port 9999
lsof -i :9999

# Kill it
kill -9 <PID>
```

### Issue: Service adapters report unavailable

**Symptom**: `Service health: {'llm': False, 'asl': False, ...}`

**Solution**:
1. Verify services are running on expected ports
2. Check firewall settings
3. If using Docker: verify network mode

```bash
# Check services
curl http://localhost:5000/health
curl http://localhost:8000/health
```

### Issue: Robot UI doesn't connect

**Symptom**: UI shows "Disconnected" status

**Solution**:
1. Verify WebSocket server is running
2. Check browser console for errors
3. Verify network connectivity
4. Check firewall port 9999

### Issue: High CPU/Memory usage

**Symptom**: 100% CPU, OOM errors

**Solution**:
1. Reduce WebSocket broadcast frequency
2. Enable frame skipping on camera
3. Reduce priority queue size
4. Profile with Python profiler

---

## Configuration

### Environment Variables

```bash
# Runtime mode
export ROBOT_MODE=serbot  # or laptop, debug

# Ports
export ORCHESTRATOR_HOST=0.0.0.0
export ORCHESTRATOR_PORT=9999

# Service endpoints
export LLM_SERVICE_URL=http://localhost:5000
export ASL_SERVICE_URL=http://localhost:8000
export ROBOT_HAL_URL=http://localhost:9000
export TTS_SERVICE_URL=http://localhost:8002

# Logging
export LOG_LEVEL=INFO
```

### Config File

Edit `optimized_runtime/robot_ui/config.json`:

```json
{
  "robot": {
    "name": "SERBot",
    "version": "2.0"
  },
  "websocket": {
    "url": "ws://localhost:9999",
    "reconnect_interval": 1000,
    "max_reconnect_attempts": 10
  },
  "ui": {
    "fullscreen": true,
    "fps": 60,
    "theme": "dark"
  }
}
```

---

## Performance Tuning

### For Slow Hardware (SERBot)

```python
# In orchestrator
orch.max_queue_size = 50  # Reduce queue
camera_fps = 15  # Lower FPS
ws_broadcast_hz = 5  # Less frequent updates
```

### For GPU Acceleration

```bash
# Enable GPU inference
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_LAUNCH_BLOCKING=1
```

### Memory Optimization

```bash
# Run with memory limits
python -m memory_profiler optimized_runtime/runtime/launcher.py
```

---

## Next Steps

1. **Read full architecture guide**: [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Explore code examples**: See individual module docstrings
3. **Deploy to hardware**: Follow deployment section above
4. **Integrate existing services**: See INTEGRATION.md
5. **Optimize for your hardware**: Performance tuning section

---

## Support & Documentation

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Integration**: [INTEGRATION.md](INTEGRATION.md) (coming)
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md) (coming)
- **API Reference**: See docstrings in each module
- **Examples**: See test files

---

## Quick Command Reference

| Task | Command |
|------|---------|
| Start (debug mode) | `python -m optimized_runtime.runtime.launcher --mode debug` |
| Start (SERBot) | `python -m optimized_runtime.runtime.launcher --mode serbot` |
| Start (Laptop) | `python -m optimized_runtime.runtime.launcher --mode laptop` |
| Deploy to SERBot | `bash optimized_runtime/scripts/deploy_serbot.sh 192.168.1.100 ubuntu` |
| Health check | `bash optimized_runtime/scripts/health_check.sh` |
| Start UI | `cd optimized_runtime/robot_ui && npm start` |
| Docker build | `docker-compose -f optimized_runtime/docker/docker-compose.yml build` |
| Docker start | `docker-compose -f optimized_runtime/docker/docker-compose.yml up -d` |
| View logs | `tail -f logs/runtime_*.log` |

---

Good luck! 🤖
