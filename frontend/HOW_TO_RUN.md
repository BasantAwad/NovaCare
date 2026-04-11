# 🚀 NovaCare — How to Run Everything

This guide covers starting all three services from the **unified `novacare` repository** (paths below are relative to the repo root).

| Service          | Port   | Tech         |
|------------------|--------|--------------|
| 🖥️ Frontend      | `3000` | Next.js      |
| 🤖 LLM Backend   | `5000` | Flask        |
| 🖐️ ASL Model API | `8000` | FastAPI      |

> **You need three separate terminals** — one for each service.

---

## Prerequisites

- **Node.js** (v18+) and **npm**
- **Python** (3.10+)
- For **LLM Backend** chat: **[Ollama](https://ollama.com/)** with a pulled model (optional, fast path) and/or a **Hugging Face API key** (quality / fallback). See `services/llm-backend/README.md`.

---

## Terminal 1 — ASL Model API (port 8000)

```powershell
cd path\to\novacare\services\asl-model
.\venv\Scripts\activate

# (First time only)
# python -m venv venv
# pip install -r requirements.txt

python -m api.main --port 8000
```

✅ Verify: open http://localhost:8000/docs — you should see the FastAPI docs.

---

## Terminal 2 — LLM Backend (port 5000)

```powershell
cd path\to\novacare\services\llm-backend
.\venv\Scripts\activate

# (First time only)
# python -m venv venv
# pip install -r requirements.txt

# .env example (see services/llm-backend/README.md):
#   OLLAMA_MODEL=qwen2.5:0.5b
#   HUGGINGFACE_API_KEY=hf_your_key_here

python start_server.py
```

✅ Verify: open http://localhost:5000 — the server should respond. `GET /health` returns an `llm` object describing Ollama/Hugging Face configuration.

---

## Terminal 3 — Frontend (port 3000)

```powershell
cd path\to\novacare\frontend

# (First time only)
npm install

# .env.local — at minimum:
#   NEXT_PUBLIC_NOVABOT_API_URL=http://localhost:5000

npm run dev
```

✅ Verify: open http://localhost:3000 — the app should load.

---

## ⚡ Quick start (after first-time setup)

```powershell
# Terminal 1 — ASL
cd path\to\novacare\services\asl-model; .\venv\Scripts\activate; python -m api.main --port 8000

# Terminal 2 — LLM
cd path\to\novacare\services\llm-backend; .\venv\Scripts\activate; python start_server.py

# Terminal 3 — Frontend
cd path\to\novacare\frontend; npm run dev
```

Or use **`start_all.bat`** / **`start_all.ps1`** from the repo root.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA not available` | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| `mediapipe` import error | `pip uninstall mediapipe && pip install mediapipe==0.10.9` |
| ASL not detecting hands | Check lighting, keep hand in frame |
| Frontend can't reach LLM API | Ensure `.env.local` has `NEXT_PUBLIC_NOVABOT_API_URL=http://localhost:5000` and the LLM Backend is running |
| Chat errors on LLM Backend | Configure `OLLAMA_MODEL` + run Ollama and/or set `HUGGINGFACE_API_KEY` (see `services/llm-backend/README.md`) |
| Frontend can't reach ASL API | Ensure the ASL server is running on port `8000` |
| `venv` not found | Run `python -m venv venv` first to create it |
