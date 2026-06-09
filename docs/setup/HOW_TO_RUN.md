# 🚀 NovaCare — How to Run Everything

This guide covers starting all services needed to run the full NovaCare application.

| Service             | Port   | Tech              |
|---------------------|--------|-------------------|
| 🖥️ Frontend         | `3000` | Next.js 14        |
| 🤖 LLM Backend      | `5000` | Flask             |
| 🖐️ ASL Model API    | `8000` | FastAPI           |
| 🔊 TTS Proxy        | `8002` | FastAPI (optional)|

> **You need 3 terminals** — one for each core service.

---

## Prerequisites

- **Node.js** v18+ and **npm**
- **Python** 3.10+
- A **Hugging Face API key** (for the LLM backend)
- Project root: `e:\NovaCare`

---

## Terminal 1 — ASL Model API (port 8000)

```powershell
cd e:\NovaCare\services\asl

# Activate the virtual environment (create it first if needed)
python -m venv venv
venv\Scripts\activate

# (First time only) Install dependencies
pip install -r requirements.txt

# Start the API server
python -m api.main --port 8000
```

✅ Verify: open http://localhost:8000/docs — FastAPI docs should appear.

---

## Terminal 2 — LLM Backend (port 5000)

```powershell
cd e:\NovaCare\services\llm

# Activate the virtual environment
python -m venv venv
venv\Scripts\activate

# (First time only) Install dependencies
pip install -r requirements.txt

# Ensure .env exists with your Hugging Face key:
#   HUGGINGFACE_API_KEY=hf_your_key_here

# Start the Flask server
python -c "from api_server import app; app.run(host='0.0.0.0', port=5000, debug=False)"
```

✅ Verify: open http://localhost:5000 — server should respond.

---

## Terminal 3 — Frontend (port 3000)

```powershell
cd e:\NovaCare\apps\frontend

# (First time only) Install dependencies
npm install

# Ensure .env.local exists at e:\NovaCare\apps\frontend\.env.local with:
#   NEXT_PUBLIC_NOVABOT_API_URL=http://localhost:5000
#   HUGGINGFACE_API_KEY=hf_your_key_here
#
# Optional — enable Nova voice debug panel in the browser:
#   NEXT_PUBLIC_VOICE_DEBUG=true

# Start the dev server
npm run dev
```

✅ Verify: open http://localhost:3000 — the NovaCare app should load.

---

## ⚡ Quick Start (after first-time setup)

Once everything is installed, run these 3 commands in 3 separate terminals:

```powershell
# Terminal 1 — ASL Model
cd e:\NovaCare\services\asl && venv\Scripts\activate && python -m api.main --port 8000

# Terminal 2 — LLM Backend
cd e:\NovaCare\services\llm && venv\Scripts\activate && python -c "from api_server import app; app.run(host='0.0.0.0', port=5000, debug=False)"

# Terminal 3 — Frontend
cd e:\NovaCare\apps\frontend && npm run dev
```

---

## 🎙️ Nova AI Voice Assistant

Nova is built into the frontend and requires no separate process.

Once the frontend is running:

1. Open http://localhost:3000 and log in.
2. Click the **microphone button** (bottom-right corner) to activate Nova.
3. Say **"Hey Nova"** to wake the assistant, then speak your command.
4. You can also type commands in the Nova chat input.

### Example Voice Commands

| Command | What Happens |
|---------|-------------|
| `"open settings"` | Navigates to your role's settings page |
| `"go to appointments"` | Opens `/medical/appointments` |
| `"dashboard"` | Opens your role's dashboard |
| `"scroll down"` | Scrolls the current page down |
| `"go back"` | Browser back navigation |

> **Role-aware navigation:** Nova detects your current role from the URL
> (`/medical/*`, `/admin/*`, `/guardian/*`, `/rover/*`) and resolves
> commands to the correct role-specific page — no 404s.

### Nova Debug Panel (dev only)

To see real-time route resolution logs, add this to `.env.local`:

```
NEXT_PUBLIC_VOICE_DEBUG=true
```

A **"Nova Debug"** button will appear at the bottom-left of the screen showing:
- Detected intent
- Resolved route path
- Route validity check
- Navigation result (`success` / `blocked`)

---

## 🤖 SERBot Integration

To deploy the optimized runtime to the SERBot device and run local services:

```powershell
$env:ROBOT_IP = "192.168.137.206"
.\run_serbot_integration.ps1
```

The script will:
- Start local laptop services (ASL, LLM)
- SCP the `optimized_runtime` bundle to the robot
- Start the robot runtime via Docker or `startup.sh`

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA not available` | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| `mediapipe` import error | `pip uninstall mediapipe && pip install mediapipe==0.10.9` |
| ASL not detecting hands | Check lighting, keep hand fully in frame |
| Frontend can't reach LLM | Check `.env.local`: `NEXT_PUBLIC_NOVABOT_API_URL=http://localhost:5000` |
| Frontend can't reach ASL | Confirm ASL server is running on port `8000` |
| `venv` not found | Run `python -m venv venv` first |
| Nova navigates to 404 | This is fixed — Nova now uses the centralized route registry |
| Nova says "I couldn't find that page" | The spoken page name doesn't match any alias; check `routeRegistry.ts` |
| `npm run dev` fails | Delete `e:\NovaCare\apps\frontend\.next` and retry |
| Port already in use | `netstat -ano \| findstr :3000` then `taskkill /PID <pid> /F` |
