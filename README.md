# 🩺 NovaCare — Unified AI Healthcare Companion

NovaCare is an integrated AI-powered assistant designed to empower independence for individuals with physical or sensory disabilities through intelligent, multimodal interaction.

## 🚀 Repository Structure

This repository is organized into a clean, professional, and scalable structure:

```text
NovaCare/
├── apps/                    # User-facing applications
│   ├── frontend/            # Next.js Dashboard (Guardian/Medical/Patient)
│   ├── mobile/              # Flutter-based mobile application
│   └── robot_ui/            # Specialized robot interface components
├── services/                # Core microservices
│   ├── robot/               # Hardware Abstraction Layer (HAL) & Control
│   ├── asl/                 # Sign Language Recognition (FastAPI)
│   ├── llm/                 # Conversational AI & Emotion Detection
│   ├── auth/                # Authentication & User Management
│   ├── tts/                 # Edge TTS Proxy service
│   ├── robot-runtime/       # Advanced optimized robotics runtime
│   └── app-backend/         # Python backend logic for the main app
├── shared/                  # Shared resources
│   ├── models/              # Common data models & DB schemas
│   ├── utils/               # Shared utility functions
│   ├── configs/             # Global configuration templates
│   └── assets/              # Shared static assets (images, icons)
├── infrastructure/          # DevOps and Deployment
│   ├── docker/              # Dockerfiles and Compose configs
│   ├── deployment/          # Cloud/Hardware deployment scripts
│   ├── database/            # SQL migrations and database tools
│   └── scripts/             # Startup and maintenance scripts
├── docs/                    # Centralized documentation
├── tests/                   # Consolidated test suites
├── archive/                 # Deprecated or legacy files (preserved safely)
└── docker-compose.yml       # Primary orchestration
```

## 🛠️ Getting Started

### 1. Prerequisites
- **Node.js** v18+ & **npm**
- **Python** 3.10+
- **Docker & Docker Compose** (recommended for deployment)

### 2. Quick Setup (Unified)
Run the unified launcher to install dependencies and start all services:

**Windows:**
```powershell
.\infrastructure\scripts\start_all.ps1
```

**Linux/Jetson:**
```bash
./infrastructure/scripts/start_all.sh
```

### 3. Manual Startup
If you wish to run services individually, refer to the [Setup Guide](docs/setup/HOW_TO_RUN.md).

### 4. SERBot Integration (Deploy + Runtime)

To deploy the optimized runtime to the SERBot device and start local services from your development machine, use the provided PowerShell integration script. It prefers the `ROBOT_IP` environment variable but accepts a CLI argument.

Windows example (uses env var override):
```powershell
$env:ROBOT_IP="192.168.137.206"
.\run_serbot_integration.ps1
```

The integration script will:
- Start local laptop services (ASL, LLM) when not using Docker.
- SCP the `optimized_runtime` bundle to the robot and attempt to start it.
- Prefer starting `docker-compose` on the robot if Docker is available, otherwise fall back to the runtime `startup.sh`.

Ensure the robot has Docker if you want the robot UI and runtime started via containers.

## 📚 Documentation Index

| Category | Description | Link |
|----------|-------------|------|
| **Setup** | Installation and first run | [docs/setup/](docs/setup/) |
| **Architecture** | System design and data flow | [docs/architecture/](docs/architecture/) |
| **Hardware** | SERBot hardware integration | [docs/hardware/](docs/hardware/) |
| **APIs** | Service interface documentation | [docs/APIs/](docs/APIs/) |
| **Roadmap** | Project progress and future plans | [docs/roadmap/](docs/roadmap/) |

---

## 👥 The NovaCare Team
* **Basant Awad**
* **Nadira El-Sirafy**
* **Noureen Yasser**
* **Muhammad Mustafa**
* **Ramez Asaad**

---
© 2026 NovaCare. All rights reserved. Professional, Production-Grade Healthcare Robotics.
