# NovaCare - Complete Intelligent Care Rover System

[![TypeScript](https://img.shields.io/badge/TypeScript-49%25-blue)](https://github.com/BasantAwad/NovaCare)
[![Python](https://img.shields.io/badge/Python-34.1%25-blue)](https://github.com/BasantAwad/NovaCare)
[![Dart/Flutter](https://img.shields.io/badge/Dart-7.6%25-blue)](https://github.com/BasantAwad/NovaCare)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen)](https://github.com/BasantAwad/NovaCare)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A comprehensive AI-powered assistive rover system designed for individuals with disabilities, featuring mental health therapy, emotion detection, sign language recognition, and real-time robot control.

## 🌟 Project Overview

NovaCare is an integrated ecosystem combining:
- **AI/ML Services**: LLM conversational therapy, emotion detection, ASL recognition
- **Mobile App**: Flutter-based caregiver & patient interfaces with SOS alerts
- **Web Dashboard**: Real-time monitoring and control for guardians, rovers, and medical professionals
- **Hardware Integration**: ESP32-based rover control, Jetson Orin robotics
- **Cloud Services**: Firebase real-time database, multi-provider LLM backends

---

## 📁 Repository Structure

```
NovaCare/
├── frontend/                      # Next.js web application
│   ├── src/app/
│   │   ├── rover/                 # Rover control dashboard
│   │   ├── guardian/              # Guardian monitoring interface  
│   │   ├── medical/               # Medical professional dashboard
│   │   ├── admin/                 # System administration
│   │   └── auth/                  # Authentication pages
│   ├── src/lib/
│   │   ├── *-api.ts               # API client libraries
│   │   ├── speech.ts              # TTS/STT services
│   │   └── utils.ts               # Dynamic URL routing
│   └── public/                    # Static assets
│
├── mobile/                        # Flutter mobile app
│   ├── lib/
│   │   ├── screens/               # Caregiver & patient interfaces
│   │   ├── providers/             # State management
│   │   ├── services/              # BLE, Firebase, voice services
│   │   └── widgets/               # Reusable UI components
│   ├── android/                   # Android native config
│   └── ios/                       # iOS native config
│
├── services/                      # Microservices architecture
│   ├── auth/                      # JWT authentication (Flask)
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── routes/auth.py     # Authentication endpoints
│   │   │   ├── db_controller.py   # MySQL operations
│   │   │   └── utils/
│   │   │       ├── password.py    # PBKDF2 SHA-256 hashing
│   │   │       └── tokens.py      # JWT generation
│   │   ├── config.py              # Config management
│   │   ├── run.py                 # Entry point
│   │   └── requirements.txt
│   │
│   ├── llm/                       # NovaBot LLM Backend (Flask)
│   │   ├── app.py                 # Main application
│   │   ├── mental_health_pipeline.py  # 4-stage therapy pipeline
│   │   │   ├── Pattern Recognition
│   │   │   ├── Risk Assessment
│   │   │   ├── Therapy Generation
│   │   │   └── Validation
│   │   ├── start_server.py        # Server startup
│   │   ├── requirements.txt       # Groq, Cerebras, SambaNova, Gemini
│   │   └── Dockerfile
│   │
│   ├── asl/                       # ASL Recognition (FastAPI/PyTorch)
│   │   ├── api/main.py            # FastAPI endpoints
│   │   ├── data/prepare_dataset.py    # Landmark extraction (MediaPipe)
│   │   ├── models/landmark_classifier.py  # Transformer classifier
│   │   ├── training/train.py      # Model training
│   │   ├── inference/predictor.py # Real-time prediction
│   │   ├── checkpoints/           # Pre-trained models
│   │   └── requirements.txt
│   │
│   ├── emotion/                   # Multimodal Emotion Detection
│   │   ├── face_emotion.py        # Face-based emotion (pre-trained)
│   │   ├── audio_emotion.py       # Audio emotion (Wav2Vec 2.0)
│   │   ├── text_emotion.py        # Text-based emotion (BERT)
│   │   ├── aggregator.py          # Multi-model fusion
│   │   ├── utils.py               # Video/audio processing
│   │   └── requirements.txt
│   │
│   ├── robot/                     # Robot HAL & REST API (Python)
│   │   ├── robot_service.py       # Main REST API service
│   │   ├── robot_hal.py           # Hardware abstraction layer
│   │   ├── movement_controller.py # Motor control
│   │   ├── camera_controller.py   # Live camera feed
│   │   ├── lidar_controller.py    # LiDAR/obstacle detection
│   │   ├── audio_controller.py    # Speaker/microphone
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── tts/                       # Text-to-Speech Proxy (Flask)
│   │   ├── app.py                 # TTS proxy to Pocket TTS
│   │   ├── requirements.txt       # Flask, CORS wrapper
│   │   └── Dockerfile
│   │
│   └── dashboard/                 # Health Telemetry (Flask)
│       ├── dashboard.py           # REST endpoints for metrics
│       ├── database_models.py     # Sleep, hydration, mood logs
│       ├── requirements.txt
│       └── Dockerfile
│
├── optimized_runtime/             # Streamlined runtime for SERBot
│   ├── docker/                    # Docker composition
│   │   ├── docker-compose.yml     # Production container orchestration
│   │   ├── Dockerfile.robot       # Robot service image
│   │   └── Dockerfile.services    # Unified services image
│   ├── scripts/
│   │   ├── startup.sh             # Runtime startup script
│   │   ├── deploy_serbot.sh       # SCP deployment to robot
│   │   └── health_check.sh        # Service health monitoring
│   ├── robot_ui/                  # Robot-side React UI (if applicable)
│   └── summon/                    # Mobile summon service
│       ├── summon_controller.py   # Summon request handling
│       ├── rssi_tracker.py        # RSSI/heading tracking for homing
│       └── README.md              # Summon integration guide
│
├── scripts/                       # Utility & deployment scripts
│   ├── deploy_to_rover.py         # SSH/SCP deployment
│   ├── check_firewall.py          # Network diagnostics
│   ├── diagnose_rover.py          # System diagnostics
│   ├── start_rover_backend.py     # Remote service start
│   └── check_robot_service.py     # Health checks
│
├── shared/                        # Shared configuration
│   ├── configs/
│   │   ├── robot_config.py        # Robot IP, credentials
│   │   ├── api_config.py          # API endpoints & timeouts
│   │   └── db_config.py           # Database connection strings
│   └── schemas/                   # Shared data structures
│       └── telemetry.py           # Common telemetry models
│
├── docker-compose.yml             # Local development stack
├── run_serbot_integration.ps1     # PowerShell SERBot integration
├── run_launcher.sh                # Linux launcher
└── docs/                          # Documentation
    ├── API.md                     # API reference
    ├── SETUP.md                   # Installation guide
    ├── DEPLOYMENT.md              # Deployment procedures
    └── ARCHITECTURE.md            # System architecture
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for frontend)
- Python 3.10+ (for services)
- Flutter 3.2+ (for mobile)
- Git

### 1. Clone Repository

```bash
git clone https://github.com/BasantAwad/NovaCare.git
cd NovaCare
```

### 2. Local Development (Docker)

```bash
# Start all services (frontend, backend, LLM, robot API)
docker-compose up -d

# Check service health
curl http://localhost:5001/health   # Auth service
curl http://localhost:5000/health   # LLM service
curl http://localhost:8001/health   # ASL service
curl http://localhost:3000          # Web frontend
```

### 3. Frontend Development

```bash
cd frontend
npm install
npm run dev    # Starts Next.js on http://localhost:3000
```

### 4. Mobile App

```bash
cd mobile
flutter pub get
flutter run -d <device_id>  # Run on phone/emulator
```

### 5. Service Launchers

**Windows (PowerShell):**
```powershell
$env:ROBOT_IP="192.168.137.206"
.\run_serbot_integration.ps1   # Auto-creates venvs, starts services
```

**Linux:**
```bash
chmod +x run_launcher.sh
./run_launcher.sh              # Launches all services
```

---

## 🤖 Services

### Authentication Service (5001)
- JWT-based role access control
- Multi-role support: rover, caregiver, doctor
- Password: PBKDF2 SHA-256
- OAuth: Google integration ready

### LLM Backend (5000)
**4-Stage Mental Health Therapy Pipeline:**
1. **Pattern Recognition** - Analyze conversation history
2. **Risk Assessment** - Detect emotional distress
3. **Therapy Generation** - Multi-provider cascade:
   - Groq (fastest)
   - Cerebras (efficient)
   - SambaNova (balanced)
   - Google Gemini (fallback)
4. **Validation** - Ensure therapeutic quality

**Endpoints:**
- `POST /api/chat` - Send message, receive therapy response
- `POST /api/mental-health/*` - Direct therapy endpoints
- `GET /health` - Service health check

### ASL Recognition (8001)
- Real-time fingerspelling recognition
- 29 classes (A-Z, space, delete, nothing)
- MediaPipe landmarks + Transformer classifier
- Letter accumulation into words/sentences

**Endpoints:**
- `POST /predict` - Send frame (base64), get letter
- `POST /accumulator/add` - Add letter to word
- `POST /accumulator/clear` - Clear accumulated text
- `GET /health` - Service health

### Emotion Detection
- **Face**: Pre-trained CNN (7 emotions)
- **Audio**: Wav2Vec 2.0 (8 emotions)
- **Text**: BERT-based (multi-label)
- **Aggregation**: Weighted fusion

### Robot HAL API (9000)
- Motor/movement control
- Live camera streaming
- LiDAR obstacle detection
- Audio playback & recording
- Battery monitoring

### Text-to-Speech Proxy (8765)
- CORS-friendly wrapper around Pocket TTS
- Falls back to browser SpeechSynthesis
- Audio caching & streaming

### Dashboard API (5002)
- Health telemetry: sleep, hydration, mood, battery
- Dynamic database integration
- Guardian/medical dashboards

---

## 📱 Mobile App Features

### Patient Interface
- **Big Button Layout**: SOS, medication, home, follow-me
- **Real-time Telemetry**: Battery, heart rate, temperature, location
- **Voice Control**: Conversational chat with therapy bot
- **ASL Input**: Hand gesture recognition for commands
- **High Contrast Mode**: WCAG AAA accessibility
- **Multi-language**: English + Arabic (Egyptian dialect)

### Caregiver Interface
- **Role-based Auth**: Signup/login with SharedPreferences persistence
- **SOS Alert Feed**: Live notifications from patients
- **Patient Dashboard**: Overview, health metrics, location
- **Remote Control**: Send commands to rover
- **Settings**: Account info, accessibility toggles

---

## 🌐 Web Dashboard Routes

### Rover Dashboard (`/rover`)
- Remote camera feed
- Real-time movement control
- Conversational AI chat
- Mood/emotion tracking
- Navigate to locations
- Follow-me mode toggle

### Guardian Dashboard (`/guardian`)
- Patient location tracking
- Health metrics overview
- Medication reminders
- SOS alert history
- Communication logs

### Medical Dashboard (`/medical`)
- Comprehensive health records
- Telemetry visualization
- Therapy session history
- Risk assessment reports
- Treatment recommendations

### Admin Dashboard (`/admin`)
- Service health monitoring
- User management
- System diagnostics
- Log aggregation
- Configuration controls

---

## 🔧 Deployment

### Local Development Stack
```bash
docker-compose up -d
```
Starts: Frontend, Auth, LLM, ASL, Robot API, TTS, Dashboard services.

### SERBot Robot Deployment

**Via PowerShell (Windows):**
```powershell
$env:ROBOT_IP="10.34.19.247"
.\run_serbot_integration.ps1 -DeployOnly:$false
```
This:
1. Creates local venvs for ASL, LLM, Robot services
2. Starts services on development machine
3. SCPs optimized_runtime to robot
4. Starts docker-compose or startup.sh on robot

**Via Bash (Linux/Mac):**
```bash
export ROBOT_IP="10.34.19.247"
./optimized_runtime/scripts/deploy_serbot.sh
```

**Docker-based (if robot has Docker):**
```bash
docker-compose -f optimized_runtime/docker/docker-compose.yml up -d
```

### Production Deployment
1. **Environment**: Set `DEPLOYMENT_ENV=production`
2. **Database**: Migrate from SQLite to PostgreSQL
3. **LLM Models**: Use quantized versions or API keys
4. **Secrets**: Store API keys in environment variables
5. **Monitoring**: Enable health checks & log aggregation

---

## 🔄 Unified Branch Merge Summary

This main branch consolidates work from:

| Branch | Contributor | Key Features | Commit |
|--------|-------------|--------------|--------|
| **ramez-unified-branch** | Ramez Asaad | API service layer, dynamic URL routing, TTS/speech services | fe533e21 |
| **clean-opt-version** | BasantAwad | SERBot deployment, runtime optimization, startup scripts | 504521c0 |
| **Nova-app** | Muhammed Farag | Flutter mobile app, caregiver auth, SOS alerts, local persistence | d94f08d8 |
| **nadira** | NadiraElsirafy | Medical dashboards, health telemetry, data visualization | fac1e95d |
| **upgrading-Conv-LLM** | Ramez Asaad | Multi-provider LLM pipeline, Groq/Cerebras/SambaNova/Gemini | 73c6ada* |
| **upgrade-tts** | Ramez Asaad | Enhanced TTS, Pocket TTS integration, fallback handling | 73c6ada* |
| **backend-stitch** | (merged) | Backend service integration, API orchestration | 0f7f135 |
| **hardware-integration** | (merged) | Rover HAL, motor control, sensor integration | 8d43468 |

*Latest unique commits per branch included in unification

---

## 📚 Documentation

- **[Setup Guide](docs/SETUP.md)** - Detailed installation instructions
- **[API Reference](docs/API.md)** - Complete endpoint documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design & data flow
- **[Deployment](docs/DEPLOYMENT.md)** - Production deployment guide
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State**: Redux/Context API
- **Real-time**: Socket.io, Firebase

### Backend
- **Language**: Python 3.10+
- **Frameworks**: Flask, FastAPI
- **Auth**: JWT, OAuth2
- **Database**: MySQL, SQLite (dev)
- **LLM Providers**: Groq, Cerebras, SambaNova, Google Gemini

### Mobile
- **Framework**: Flutter 3.2+
- **Language**: Dart
- **State**: Provider
- **BLE**: flutter_blue_plus
- **Cloud**: Firebase Realtime DB

### DevOps
- **Containerization**: Docker & Docker Compose
- **Orchestration**: Docker Compose (dev), Kubernetes-ready
- **CI/CD**: GitHub Actions ready
- **Deployment**: SSH/SCP scripts, PowerShell integration

---

## 🔐 Security

- ✅ JWT-based authentication with role-based access control
- ✅ PBKDF2 SHA-256 password hashing
- ✅ CORS hardening for production
- ✅ Secrets management via environment variables
- ✅ HTTPS/TLS ready
- ✅ OAuth2 integration support

---

## 📞 Support & Issues

- **Bug Reports**: [GitHub Issues](https://github.com/BasantAwad/NovaCare/issues)
- **Feature Requests**: [Discussions](https://github.com/BasantAwad/NovaCare/discussions)
- **Documentation**: See `/docs` directory

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👥 Contributors

- **BasantAwad** - Project lead, architecture, backend optimization
- **Ramez Asaad** - Frontend APIs, LLM pipeline, TTS services, unified branch
- **NadiraElsirafy** - Medical dashboards, telemetry systems
- **Muhammed Farrag** - Flutter mobile application, caregiver features

---

## 🙏 Acknowledgments

- OpenAI, Google, Anthropic for LLM APIs
- MediaPipe for hand landmark detection
- PyTorch for deep learning framework
- Flutter team for mobile framework

---

**Last Updated**: June 5, 2026 | **Status**: Production Ready ✅
