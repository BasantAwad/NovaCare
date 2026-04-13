# NovaCare LLM Backend

Flask service on **port 5000**: NovaBot chat, test UI (STT/TTS), and facial emotion detection.

## Conversational AI (dual routing)

Chat uses **two backends** with automatic fallbacks:

| Profile | Primary | Fallback |
|--------|---------|----------|
| **fast** | [Ollama](https://ollama.com/) (local) when `OLLAMA_MODEL` is set | Hugging Face Inference API if `HUGGINGFACE_API_KEY` is set and Ollama fails |
| **quality** | Hugging Face Inference API | Ollama if `OLLAMA_MODEL` is set and Hugging Face fails |

If `OLLAMA_MODEL` is **not** set, the **fast** profile uses Hugging Face only (same as a cloud-only setup).

### Default profile (no request override)

1. `CHAT_LLM_DEFAULT_PROFILE` = `fast` or `quality` if set in `.env`.
2. Otherwise: **fast** when `OLLAMA_MODEL` is set; else **quality** when `HUGGINGFACE_API_KEY` is set; else **fast**.

You need **at least one** of: `OLLAMA_MODEL` (with Ollama running) or `HUGGINGFACE_API_KEY`.

### Environment variables (`services/llm-backend/.env`)

| Variable | Description |
|----------|-------------|
| `HUGGINGFACE_API_KEY` | Token for Hugging Face Inference API (required for **quality** unless only using Ollama after HF errors). |
| `MODEL_NAME` | Hugging Face model id (default: `meta-llama/Meta-Llama-3-8B-Instruct`). |
| `OLLAMA_BASE_URL` | Ollama server (default: `http://127.0.0.1:11434`). |
| `OLLAMA_MODEL` | Ollama model name (e.g. `qwen2.5:0.5b`). Must be pulled locally: `ollama pull <name>`. |
| `CHAT_LLM_DEFAULT_PROFILE` | Optional: `fast` or `quality` to override automatic default. |

### HTTP API

**`POST /api/chat`**

JSON body:

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | User message (required). |
| `llm_profile` | `"fast"` \| `"quality"` | Optional; selects routing profile. |
| `prefer_quality` | boolean | If `true`, same as `llm_profile: "quality"`. |

Response includes:

| Field | Description |
|-------|-------------|
| `response` | Assistant text. |
| `llm_profile` | Profile used (`fast` or `quality`). |
| `llm_route` | e.g. `ollama`, `huggingface`, `huggingface_fallback`, `ollama_fallback`. |

**`GET /health`**

Returns `llm` with `default_profile`, `ollama` (`enabled`, `base_url`, `model`), and `huggingface` (`enabled`, `model`).

### Run

```powershell
cd services\llm-backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python start_server.py
```

Open `http://localhost:5000` for the bundled test page. For voice/text tests, use the **Cloud model (higher quality, slower)** checkbox to send `prefer_quality: true`.

### Test UI: remote TTS (Pocket / proxy)

The bundled page can use the same HTTP TTS stack as the Next.js rover (optional). Configure in HTML (see comments in `templates/test_novabot.html`) or before loading scripts:

| Global | Purpose |
|--------|---------|
| `window.NOVACARE_POCKET_TTS_URL` | Pocket base URL → browser **`POST …/tts`** (wins if set). |
| `window.NOVACARE_POCKET_TTS_VOICE_URL` | Optional Pocket `voice_url` form field. |
| `window.NOVACARE_EDGE_TTS_URL` | NovaCare edge proxy → **`POST …/api/speak`** (used if Pocket URL unset). |

Implementation: `static/js/TTS.js`. Precedence: Pocket direct → edge proxy → Web Speech (browser fallback). For full TTS architecture, CORS, and Jetson deployment, see **[`../edge-tts-proxy/README.md`](../edge-tts-proxy/README.md)**.

## Other endpoints

- `POST /api/clear` — clear server-side chat history.
- `POST /api/emotion/detect` — facial emotion from base64 image (ViT).
- `GET /api/emotion/health` — emotion model status.

