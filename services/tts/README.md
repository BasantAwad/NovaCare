# NovaCare — Edge TTS Proxy

CORS-friendly wrapper around **Kyutai Pocket TTS** for the NovaCare rover.

Pocket TTS ships a fixed CORS allowlist. This proxy sits between the browser and Pocket on the LAN so that Chromium (on the Jetson or during local dev) can speak assistant replies without CORS errors.

## How It Works

```
Browser (Next.js / LLM test page)
    │  POST /api/speak  { "text": "Hello" }
    ▼
┌─────────────────────┐
│  edge-tts-proxy     │  ← This service (Flask, port 8765)
│  (CORS: allow all)  │
└────────┬────────────┘
         │  POST /tts  (form: text=Hello)
         ▼
┌─────────────────────┐
│  Pocket TTS         │  ← Upstream (port 8800 by default)
│  (kyutai)           │
└─────────────────────┘
         │
         ▼
    audio/wav stream → proxied back to browser
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/speak` | JSON `{"text": "...", "voice_url?": "..."}` → proxied to Pocket `POST /tts` → returns `audio/wav` stream |
| `GET` | `/health` | Returns `{"status": "healthy", "upstream": "...", "pocket_reachable": true/false}` |

## Quick Start

```bash
cd services/edge-tts-proxy

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Listens on **`0.0.0.0:8765`** by default.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POCKET_TTS_UPSTREAM` | `http://127.0.0.1:8800` | Pocket TTS base URL |
| `EDGE_TTS_PROXY_HOST` | `0.0.0.0` | Bind address |
| `EDGE_TTS_PROXY_PORT` | `8765` | Bind port |
| `EDGE_TTS_PROXY_CONNECT_TIMEOUT` | `30` | Upstream connect timeout (seconds) |
| `EDGE_TTS_PROXY_READ_TIMEOUT` | `120` | Upstream read timeout (seconds) |
| `EDGE_TTS_CORS_ORIGINS` | `*` | Allowed CORS origins (tighten for production) |

---

## Kyutai Pocket TTS (Upstream)

- **Repo:** [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
- **CLI:** [serve](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/serve/)
- **Runtime:** Python **3.10+**, **PyTorch 2.5+** (CPU is fine for many setups). On Jetson, pin wheels per board.

### Pocket HTTP contract

- **`POST /tts`** — `application/x-www-form-urlencoded` or `multipart/form-data` with:
  - **`text`** (required)
  - **`voice_url`** (optional; e.g. `hf://kyutai/tts-voices/...`)
- **Response:** `audio/wav` (streamed).
- **`GET /health`** — e.g. `{"status":"healthy"}`.

NovaCare's browser client uses **`application/x-www-form-urlencoded`** with a `URLSearchParams` body.

### CORS

Pocket's built-in allowlist includes **`http://localhost:3000`** (see upstream `pocket_tts/main.py`). If you open the app as **`http://127.0.0.1:3000`**, the browser origin differs and Pocket will block requests. Solutions:
- Use `localhost` consistently
- Use this **edge-tts-proxy**
- Patch Pocket's CORS upstream

---

## TTS Fallback Ladder (Full System)

The proxy is one piece of the NovaCare TTS stack. The full precedence (applies to both the Next.js frontend and the LLM Backend test page):

```
1.  Pocket TTS direct (NEXT_PUBLIC_POCKET_TTS_URL set?)
    └─ YES → Browser calls Pocket POST /tts (form-encoded, wins)
    └─ NO  → continue

2.  Edge TTS Proxy (NEXT_PUBLIC_EDGE_TTS_URL set?)
    └─ YES → Browser calls this proxy POST /api/speak (JSON)
    └─ NO  → continue

3.  Web Speech API fallback (browser speechSynthesis)
```

### Default ports (reference)

| Service | Example port | Bind |
|---------|--------------|------|
| Pocket `serve` (systemd) | **8800** | `127.0.0.1` |
| Edge TTS proxy | **8765** | `0.0.0.0` (Chromium / LAN) |
| ASL Model API | **8000** | per `services/asl-model` |

---

## Jetson Nano Deployment

1. Create user `novabot` (or edit units to your user).
2. Install Pocket in a venv (see upstream; aarch64 may need JetPack-aligned PyTorch).
3. Adjust paths in:
   - [`deploy/jetson/novacare-pocket-tts.service`](../../deploy/jetson/novacare-pocket-tts.service)
   - [`deploy/jetson/novacare-edge-tts-proxy.service`](../../deploy/jetson/novacare-edge-tts-proxy.service)
4. Proxy deps: `pip install -r services/edge-tts-proxy/requirements.txt`
5. `sudo systemctl enable --now novacare-pocket-tts.service` then `novacare-edge-tts-proxy.service`
6. Benchmark: `python scripts/jetson/benchmark_tts_latency.py --url http://127.0.0.1:8765`

**Memory:** Pocket supports **`--quantize`** (int8); prefer on 4GB Nano if you hit OOM.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| Browser shows CORS error on `POST /tts` | Origin not in Pocket allowlist | Use `localhost` vs `127.0.0.1` consistently, or use this proxy |
| No speech from HTTP path | Pocket down, wrong URL, or timeout | Check Pocket `GET /health`; increase `NEXT_PUBLIC_EDGE_TTS_TIMEOUT_MS` |
| Reply spoken **twice** (Pocket then system voice) | Was a bug: `<audio>` teardown fired `error` after a successful `ended` | Fixed in current `speech.ts` / `TTS.js` by clearing handlers before revoking blob URL. Pull latest. |
