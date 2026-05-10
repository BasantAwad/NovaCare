# 🩺 NovaCare — Unified Project

NovaCare is an AI-powered healthcare companion application with three integrated services:

| Service | Port | Tech | Description |
|---------|------|------|-------------|
| 🖐️ ASL Model API | `8001` | FastAPI | Real-time ASL fingerspelling recognition |
| 🤖 LLM Backend | `5000` | Flask | Conversational AI chatbot (NovaBot) |
| 🤖 Robot Service | `9000` | Flask | Hardware abstraction (camera, motors, audio, LiDAR) |
| 🖥️ Frontend | `3000` | Next.js | User interface with multiple dashboards |

---

## 📁 Project Structure

```
novacare/
├── services/
│   ├── asl-model/          ← FastAPI ASL recognition service
│   │   ├── api/            ← API routes & entry point
│   │   ├── config/         ← Model configuration
│   │   ├── inference/      ← Prediction & webcam logic
│   │   ├── models/         ← Neural network architecture
│   │   ├── training/       ← Training & evaluation scripts
│   │   └── requirements.txt
│   │
│   ├── edge-tts-proxy/     ← CORS proxy for Pocket TTS (Jetson / edge)
│   ├── robot/              ← Robot hardware service (SERBot Prime X)
│   │   ├── robot_hal.py    ← Hardware Abstraction Layer (pop library)
│   │   ├── robot_service.py ← Flask REST API for hardware control
│   │   ├── config.py       ← Robot configuration
│   │   └── requirements.txt
│   └── llm-backend/        ← Flask LLM chatbot service
│       ├── README.md       ← Env vars, Ollama + Hugging Face routing, API
│       ├── LLMs/           ← Conversational AI logic
│       ├── static/js/      ← Client-side JS (NovaBotClient, STT, TTS)
│       ├── templates/      ← Test HTML templates
│       └── requirements.txt
│
├── frontend/                ← Next.js frontend application
│   ├── src/                ← React pages & components
│   │   ├── app/            ← Next.js App Router pages
│   │   ├── components/     ← Reusable UI components
│   │   ├── lib/            ← API clients & utilities
│   │   └── types/          ← TypeScript type definitions
│   └── package.json
│
├── deploy/
│   └── jetson/              ← systemd units for edge TTS (Pocket + proxy)
├── docs/                    ← Project-level documentation
├── scripts/
│   └── jetson/              ← Robot startup + benchmark scripts
├── start_all.sh             ← One-click launcher (macOS / Linux)
├── start_all.bat            ← One-click launcher (CMD)
├── start_all.ps1            ← One-click launcher (PowerShell)
└── README.md                ← This file
```

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** v18+ and **npm**
- **Python** 3.10+
- For **NovaBot chat** (LLM Backend): either **[Ollama](https://ollama.com/)** with a pulled model (e.g. `ollama pull qwen2.5:0.5b`) and/or a **Hugging Face API key** (see `services/llm-backend/README.md`)

### First-Time Setup

#### 1. ASL Model API
```powershell
cd services\asl-model

# Copy your existing venv here, OR create a new one:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# For CUDA support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Note:** You also need model checkpoints in `services/asl-model/checkpoints/`.
> Copy them from your original `ASL-model/model/checkpoints/` directory.

#### 2. LLM Backend
```powershell
cd services\llm-backend

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Create .env — use Ollama for fast local chat, Hugging Face for quality / fallback:
#   OLLAMA_MODEL=qwen2.5:0.5b
#   HUGGINGFACE_API_KEY=hf_your_key_here
# Optional: CHAT_LLM_DEFAULT_PROFILE=fast|quality, MODEL_NAME=..., OLLAMA_BASE_URL=...
```

See **`services/llm-backend/README.md`** for dual routing, `POST /api/chat` fields (`llm_profile`, `prefer_quality`), and defaults.

#### 3. Frontend
```powershell
cd frontend

npm install

# Minimum: NovaBot API URL
echo NEXT_PUBLIC_NOVABOT_API_URL=http://localhost:5000 > .env.local
```

Optional: edit `frontend/.env.local` and add **`NEXT_PUBLIC_POCKET_TTS_URL`** or **`NEXT_PUBLIC_EDGE_TTS_URL`** — see **[`services/edge-tts-proxy/README.md`](services/edge-tts-proxy/README.md)** for TTS details.

### ⚡ Start All Services (One Command!)

**Option A — macOS / Linux:**
```bash
chmod +x start_all.sh   # first time only
./start_all.sh
```
On macOS this opens **3 Terminal.app windows**, one per service. On Linux it runs all three in the background.

**Option B — Windows (cmd):**
```
start_all.bat
```

**Option C — Windows (PowerShell):**
```powershell
.\start_all.ps1
```

All launchers will:
- Activate the correct virtual environment (or create one)
- Check for missing dependencies and install them
- Warn you about missing `.env` files
- Start the service

### ✅ Verify Everything Works

| Service | URL | What to Expect |
|---------|-----|----------------|
| ASL Model API | http://localhost:8000/docs | FastAPI Swagger docs |
| LLM Backend | http://localhost:5000 | Server response |
| Frontend | http://localhost:3000 | NovaCare app loads |

### Voice / TTS (Jetson & local dev)

Optional **Kyutai Pocket TTS** in the Next.js app (e.g. **Rover → Talk to Nova**): set **`NEXT_PUBLIC_POCKET_TTS_URL`** for direct Pocket, or **`NEXT_PUBLIC_EDGE_TTS_URL`** for the NovaCare CORS proxy. See **[`services/edge-tts-proxy/README.md`](services/edge-tts-proxy/README.md)** for full TTS setup, CORS, Jetson deployment, and troubleshooting.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture, service details, data flow diagrams |
| [docs/product_spec.md](docs/product_spec.md) | Product requirements, features, and tech stack |
| [docs/roadmap.md](docs/roadmap.md) | Detailed task breakdown and timeline |
| [docs/design_guidelines.md](docs/design_guidelines.md) | UI/UX design guidelines |
| [docs/dev_rules.md](docs/dev_rules.md) | Development rules and conventions |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA not available` | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| `mediapipe` import error | `pip uninstall mediapipe && pip install mediapipe==0.10.9` |
| ASL not detecting hands | Check lighting, keep hand in frame |
| Frontend can't reach LLM API | Ensure `.env.local` has `NEXT_PUBLIC_NOVABOT_API_URL=http://localhost:5000` |
| Frontend can't reach ASL API | Ensure ASL server is running on port `8000` |
| TTS CORS / no audio from Pocket | Match dev URL (`localhost` vs `127.0.0.1`) with Pocket allowlist, or use the edge proxy — see [`services/edge-tts-proxy/README.md`](services/edge-tts-proxy/README.md) |
| Reply spoken twice (Pocket then system voice) | Fixed in current `speech.ts` / `TTS.js` (audio teardown). Pull latest. |
| `venv` not found | Run `python -m venv venv` first |
| PowerShell execution policy | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
