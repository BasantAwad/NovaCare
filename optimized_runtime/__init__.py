"""
NovaCare Optimized Runtime — v2.0

A clean, organized robotics runtime layer for the NovaCare SERBot system.

This module provides:
- Centralized async orchestration (RuntimeOrchestrator)
- Unified robot state management (RobotState)
- Real-time WebSocket communication (WebSocketServer)
- Service adapters for existing microservices
- Lightweight robot UI framework
- Performance monitoring and health checks

STRUCTURE:
└── optimized_runtime/
    ├── orchestrator/       # Central task orchestrator
    ├── state/             # Unified robot state
    ├── communication/     # WebSocket server
    ├── adapters/          # Service wrappers (existing services)
    ├── runtime/           # Runtime lifecycle
    ├── robot_ui/          # Robot UI components
    ├── hardware/          # Hardware abstraction layer
    ├── inference/         # Inference pipeline management
    ├── monitoring/        # Health & performance tracking
    ├── deployment/        # Deployment configs
    ├── docker/            # Docker files
    └── scripts/           # Deployment scripts

QUICK START:

1. Start the runtime:
   python -m optimized_runtime.runtime.launcher --mode serbot

2. Connect robot UI:
   - Open http://localhost:8080
   - Should connect via WebSocket to ws://localhost:9999

3. Monitor health:
   bash optimized_runtime/scripts/health_check.sh

KEY FEATURES:

✓ Async-first design for optimal SERBot performance
✓ Service adapters preserve existing system compatibility
✓ Centralized state with observer pattern
✓ Real-time WebSocket broadcasting
✓ Task queueing for inference pipelines
✓ Distributed architecture (SERBot + Laptop)
✓ Performance monitoring and metrics
✓ Docker deployment ready

INTEGRATION WITH EXISTING SYSTEM:

The optimized runtime WRAPS existing services:
- Existing API servers continue running unchanged
- Service adapters communicate with existing APIs
- No breaking changes to original codebase
- Original frontend, mobile app, databases untouched

DISTRIBUTED ARCHITECTURE:

SERBot (Edge):
  - Lightweight runtime
  - Camera/audio capture
  - Robot UI
  - WebSocket server
  - Hardware control
  - Wake-word detection
  - Local state management

Laptop (Heavy AI):
  - LLM inference
  - ASL recognition
  - Emotion detection
  - Advanced models
  - GPU inference
  - Results streamed back to SERBot

COMMUNICATION:

WebSocket messages (ws://robot:9999):
  {
    "type": "state_update",
    "data": {
      "emotion": "happy",
      "audio": {"speaking": true, "amplitude": 0.8},
      "hardware": {"battery_level": 85}
    }
  }

PERFORMANCE OPTIMIZATIONS:

- Async task pipelines prevent blocking
- Priority-based task queuing
- Lazy model loading
- Frame skipping on video
- Model caching
- Efficient state updates
- Minimal WebSocket overhead

DEPLOYMENT:

1. SERBot deployment:
   bash scripts/deploy_serbot.sh 192.168.1.100 ubuntu

2. Laptop services:
   docker-compose -f docker/docker-compose.yml up

3. Health check:
   bash scripts/health_check.sh 192.168.1.100

MONITORING:

Access metrics via:
- RuntimeOrchestrator.get_metrics()
- HealthMonitor.run_checks()
- PerformanceTracker.get_summary()

"""

__version__ = "2.0.0"
__author__ = "NovaCare Team"
__all__ = [
    "RuntimeOrchestrator",
    "RobotState",
    "WebSocketServer",
    "ServiceRegistry",
    "RuntimeLauncher",
]

# Import core classes for convenience
from .orchestrator import RuntimeOrchestrator
from .state import RobotState, get_robot_state
from .communication import WebSocketServer, get_websocket_server
from .adapters import ServiceRegistry, get_service_registry
from .runtime import RuntimeLauncher
