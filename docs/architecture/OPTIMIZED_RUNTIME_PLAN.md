# 🩺 NovaCare Optimized Runtime — Implementation Plan

## 🎯 Goal
Create a production-grade, async-first robotics runtime layer that optimizes performance on SERBot hardware while maintaining compatibility with existing AI services.

## 🏗️ Architecture Overview

### 1. Hardware Layer (`hardware/`)
- **`CameraManager`**: Singleton for shared camera access, reducing CPU/Memory overhead.
- **`AudioManager`**: Unified interface for microphone (capture) and speaker (playback).

### 2. Orchestration Layer (`orchestrator/`)
- **`RuntimeOrchestrator`**: (Partially implemented) Async coordinator for all events.
- **`ServiceRegistry`**: Dynamic discovery of local and remote services.

### 3. Inference Layer (`inference/`)
- **`InferenceManager`**: Central dispatcher for AI tasks.
- **`DistributedAdapter`**: Routes heavy tasks (LLM, ASL) to Laptop and light tasks to SERBot.

### 4. Communication Layer (`communication/`)
- **`WebSocketHub`**: High-performance state broadcasting and command receiving.

### 5. UI Layer (`robot_ui/`)
- **`RobotFace`**: React/Framer-Motion based social robot interface.

## 🚀 Phases

### Phase 1: Hardware Abstraction (MANDATORY)
- Implement `camera_manager.py` (OpenCV + ThreadPool).
- Implement `audio_manager.py` (Pygame/Pyaudio + Async).

### Phase 2: Distributed Inference
- Implement `remote_adapter.py` for WebSocket-based communication with Laptop.
- Implement `workload_distributor.py`.

### Phase 3: Optimized Deployment
- Create `serbot_deployment/` structure.
- Create `laptop_services/` structure.
- Implement unified `start_system.sh`.

### Phase 4: Validation
- Test on SERBot hardware (simulated or real).
- Verify Mobile App & Robot UI connectivity.
