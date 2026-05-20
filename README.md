# 🩺 NovaCare — Unified AI Healthcare Companion

NovaCare is an integrated AI-powered assistant designed to empower independence for individuals with physical or sensory disabilities through intelligent, multimodal interaction.

## 🚀 Unified Services Architecture

| Service | Port | Tech | Description |
|---------|------|------|-------------|
| 🖥️ Frontend | `3000` | Next.js | Modern dashboard for Guardians, Medical Pros, and Patients |
| 🤖 LLM Backend | `5000` | Flask | Conversational AI (NovaBot) with emotional support |
| 🤖 Robot Service | `9000` | Flask | Hardware Abstraction (Camera, Motors, LiDAR, Audio) |
| 🖐️ ASL Model API | `8001` | FastAPI | Real-time ASL fingerspelling recognition |

---

## ✨ Key Features

*   💬 **Conversational AI**: Emotional support and medical Q&A via NovaBot.
*   🖐️ **ASL Recognition**: Communicate with the robot using sign language.
*   😊 **Emotion Detection**: The robot understands how you feel.
*   📊 **Vital Monitoring**: Real-time heart rate, activity, and medication tracking.
*   🚨 **Safety Alerts**: Advanced fall detection and emergency hazard alerts.
*   🚗 **Autonomous Navigation**: Robot can move to specific locations (Kitchen, Bed, etc.).

---

## 📁 Project Structure

```
NovaCare/
├── frontend/                ← Next.js frontend application
├── services/
│   ├── asl-model/          ← FastAPI ASL recognition service
│   ├── llm-backend/        ← Flask LLM chatbot & Emotion service
│   ├── robot/              ← Robot hardware service (SERBot Prime X)
│   └── edge-tts-proxy/     ← CORS proxy for Pocket TTS
├── novacare_app/           ← Flutter-based mobile application
├── docs/                    ← Project-level documentation (Roadmap, Spec, Architecture)
├── scripts/                 ← Deployment and utility scripts
├── start_all.ps1            ← Unified launcher for Windows
└── README.md                ← This file
```

---

## 🛠️ Quick Start

### 1. Prerequisites
- **Node.js** v18+ & **npm**
- **Python** 3.10+
- **Ollama** (optional, for local LLM)

### 2. Automatic Setup
Run the unified launcher to install dependencies and start all services:
```powershell
.\start_all.ps1
```

### 3. Manual Setup (Frontend)
```bash
cd frontend
npm install
npm run dev
```

---

## 👥 The Team

*   **Basant Awad** (22101405)
*   **Nadira El-Sirafy** (22101377)
*   **Noureen Yasser** (22101109)
*   **Muhammad Mustafa** (22101336)
*   **Ramez Asaad** (22100506)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture & data flow |
| [docs/product_spec.md](docs/product_spec.md) | Requirements & tech stack |
| [docs/roadmap.md](docs/roadmap.md) | Timeline & progress tracking |
| [docs/design_guidelines.md](docs/design_guidelines.md) | UI/UX guidelines |

---

© 2026 NovaCare. All rights reserved. HIPAA Compliant.
