# Text-to-speech (NovaCare)

End-to-end guide for **Kyutai Pocket TTS** in NovaCare: browser → Pocket (direct or via proxy) → optional **Web Speech** fallback.

## Summary

| Mode | Where it runs | When used |
|------|----------------|-----------|
| **Web Speech API** | Browser (`speechSynthesis`) | Default if no HTTP TTS URL is set, or if HTTP TTS fails / times out |
| **Pocket TTS direct** | Your `pocket-tts serve` instance | Set **`NEXT_PUBLIC_POCKET_TTS_URL`** to the Pocket base URL (e.g. `http://127.0.0.1:8000`). The app calls **`POST {base}/tts`** from the browser (no NovaCare proxy). |
| **Edge TTS proxy** | [`services/edge-tts-proxy`](../services/edge-tts-proxy) | Set **`NEXT_PUBLIC_EDGE_TTS_URL`** when you need CORS headers or JSON **`POST /api/speak`**. **Ignored** if **`NEXT_PUBLIC_POCKET_TTS_URL`** is set (direct Pocket wins). |

## Where it is used

| Surface | Code | Configuration |
|---------|------|----------------|
| **Rover — Talk to Nova** | [`frontend/src/app/rover/talk/page.tsx`](../frontend/src/app/rover/talk/page.tsx) | Uses `TTSService` from `speech.ts`; speaks each assistant reply when **Audio On** is enabled. |
| **Shared TTS library** | [`frontend/src/lib/speech.ts`](../frontend/src/lib/speech.ts) | `TTSService`: env + optional constructor overrides (`TTSOptions`). |
| **LLM Backend test UI** | [`services/llm-backend/static/js/TTS.js`](../services/llm-backend/static/js/TTS.js) | `window.NOVACARE_POCKET_TTS_URL` / `window.NOVACARE_EDGE_TTS_URL` (see [LLM test page](#llm-backend-test-page)). |

## Kyutai Pocket (upstream)

- **Repo:** [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)  
- **CLI:** [serve](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/serve/)  
- **Runtime:** Python **3.10+**, **PyTorch 2.5+** (CPU is fine for many setups). On Jetson, pin wheels / `pip freeze` per board.

### HTTP contract (Pocket)

- **`POST /tts`** — `application/x-www-form-urlencoded` **or** `multipart/form-data` with:
  - **`text`** (required)
  - **`voice_url`** (optional; e.g. `hf://kyutai/tts-voices/...`)
- **Response:** `audio/wav` (streamed).
- **`GET /health`** — e.g. `{"status":"healthy"}`.

NovaCare’s browser client uses **`application/x-www-form-urlencoded`** with a `URLSearchParams` body (equivalent fields to a small `FormData`).

### CORS

Pocket’s built-in allowlist includes **`http://localhost:3000`** (see upstream `pocket_tts/main.py`). If you open the app as **`http://127.0.0.1:3000`**, the browser origin differs and Pocket may block requests — use the **edge-tts-proxy**, adjust Pocket/upstream CORS, or use the same host the allowlist expects.

## NovaCare edge-tts-proxy (optional)

[`services/edge-tts-proxy/app.py`](../services/edge-tts-proxy/app.py):

- **`POST /api/speak`** — JSON `{"text":"...", "voice_url?": "..."}` → forwards to Pocket **`POST /tts`**.
- **`GET /health`** — includes `pocket_reachable`.

Proxy env **`EDGE_TTS_CORS_ORIGINS`** (default `*`) controls CORS; tighten in production if the UI origin is fixed.

## Default ports (reference)

Avoid clashing NovaCare **ASL** on **8000** when you run multiple services locally:

| Service | Example port | Bind |
|---------|--------------|------|
| Pocket `serve` (systemd example) | **8800** | `127.0.0.1` |
| Edge TTS proxy | **8765** | `0.0.0.0` (Chromium / LAN) |
| ASL Model API (NovaCare) | **8000** | per `services/asl-model` |

Using **8000** for Pocket is fine for **direct** mode if ASL is not on the same port at the same time.

## Jetson Nano deployment

1. Create user `novabot` (or edit units to your user).
2. Install Pocket in a venv (see upstream; aarch64 may need JetPack-aligned PyTorch).
3. Adjust paths in:
   - [`deploy/jetson/novacare-pocket-tts.service`](../deploy/jetson/novacare-pocket-tts.service)
   - [`deploy/jetson/novacare-edge-tts-proxy.service`](../deploy/jetson/novacare-edge-tts-proxy.service)
4. Proxy deps: `pip install -r services/edge-tts-proxy/requirements.txt`
5. `sudo systemctl enable --now novacare-pocket-tts.service` then `novacare-edge-tts-proxy.service`
6. Benchmark: `python scripts/jetson/benchmark_tts_latency.py --url http://127.0.0.1:8765`

**Memory:** Pocket supports **`--quantize`** (int8); prefer on 4GB Nano if you hit OOM.

## Frontend (Next.js)

Configure **`.env.local`** on the machine where the **browser** runs (values are `NEXT_PUBLIC_*`).

### Option A — Pocket direct (no NovaCare proxy)

```bash
NEXT_PUBLIC_POCKET_TTS_URL=http://127.0.0.1:8000
# Optional Pocket voice (same idea as Pocket CLI / voice_url):
# NEXT_PUBLIC_POCKET_TTS_VOICE_URL=hf://kyutai/tts-voices/...
```

### Option B — NovaCare proxy

```bash
NEXT_PUBLIC_EDGE_TTS_URL=http://127.0.0.1:8765
```

Do **not** set both unless you intend **direct Pocket** to take precedence (see summary table).

### Optional: HTTP timeout (both paths)

Uses the same variable name as the edge feature; applies to **Pocket direct** and **proxy** fetches:

```bash
NEXT_PUBLIC_EDGE_TTS_TIMEOUT_MS=60000
```

### Optional: code overrides

`TTSService` constructor accepts `TTSOptions` to override env without changing files:

| Field | Overrides env |
|-------|----------------|
| `pocketTtsBaseUrl` | `NEXT_PUBLIC_POCKET_TTS_URL` |
| `pocketTtsVoiceUrl` | `NEXT_PUBLIC_POCKET_TTS_VOICE_URL` |
| `edgeTtsBaseUrl` | `NEXT_PUBLIC_EDGE_TTS_URL` |
| `edgeTtsTimeoutMs` | `NEXT_PUBLIC_EDGE_TTS_TIMEOUT_MS` |

## LLM Backend test page

[`services/llm-backend/static/js/TTS.js`](../services/llm-backend/static/js/TTS.js) mirrors the Next.js precedence:

1. **`window.NOVACARE_POCKET_TTS_URL`** — direct Pocket; optional **`window.NOVACARE_POCKET_TTS_VOICE_URL`**.
2. Else **`window.NOVACARE_EDGE_TTS_URL`** — proxy JSON **`/api/speak`**.
3. Else **Web Speech**.

Optional snippets live as HTML comments in [`services/llm-backend/templates/test_novabot.html`](../services/llm-backend/templates/test_novabot.html).

## Fallback ladder

1. Pocket direct **or** edge proxy + Pocket (WAV → `<audio>` playback).
2. **Web Speech API** if HTTP fails, times out, or `audio.play()` is blocked.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| Browser shows CORS error on `POST /tts` | Origin not in Pocket allowlist | Use `localhost` vs `127.0.0.1` consistently, or use **edge-tts-proxy** / custom CORS. |
| No speech from HTTP path | Pocket down, wrong URL, or timeout | Check Pocket **`GET /health`**; increase **`NEXT_PUBLIC_EDGE_TTS_TIMEOUT_MS`**. |
| Assistant reply spoken **twice** (Pocket then system voice) | Was a bug: `<audio>` teardown fired `error` after a successful `ended` | Fixed in **`speech.ts`** / **`TTS.js`** by clearing `onplay` / `onended` / `onerror` before clearing `src` and revoking the blob URL. Update to current `main` if you still see it. |

## Maintainer notes

When you change TTS behavior, env names, ports, or proxy contracts:

1. Update **`docs/tts.md`** (this file).
2. Update root **`README.md`** if quick-start or troubleshooting should mention it.
3. Update **`services/llm-backend/README.md`** if the test UI or static JS contract changes.

See also the repo rule **“Documentation with features”** in `.cursor/rules/`.
