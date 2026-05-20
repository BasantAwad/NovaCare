"""
NovaCare edge TTS proxy — CORS-friendly wrapper around Kyutai Pocket TTS `POST /tts`.

Pocket TTS ships a fixed CORS allowlist; this proxy binds on the robot LAN and forwards
multipart form requests to Pocket (loopback) so Chromium on the Jetson can speak replies.

Upstream: https://github.com/kyutai-labs/pocket-tts (POST /tts, Form: text, optional voice_url)
"""
import os

import requests
from flask import Flask, Response, request
from flask_cors import CORS

UPSTREAM = os.environ.get("POCKET_TTS_UPSTREAM", "http://127.0.0.1:8800").rstrip("/")
TIMEOUT_CONNECT = float(os.environ.get("EDGE_TTS_PROXY_CONNECT_TIMEOUT", "30"))
TIMEOUT_READ = float(os.environ.get("EDGE_TTS_PROXY_READ_TIMEOUT", "120"))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.environ.get("EDGE_TTS_CORS_ORIGINS", "*")}})


@app.get("/health")
def health():
    try:
        r = requests.get(f"{UPSTREAM}/health", timeout=5)
        pocket_ok = r.ok
    except OSError:
        pocket_ok = False
    return {
        "status": "healthy",
        "upstream": UPSTREAM,
        "pocket_reachable": pocket_ok,
    }


@app.post("/api/speak")
def speak():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return {"error": "text is required"}, 400

    data = {"text": text}
    voice_url = payload.get("voice_url")
    if voice_url:
        data["voice_url"] = str(voice_url)

    try:
        upstream = requests.post(
            f"{UPSTREAM}/tts",
            data=data,
            stream=True,
            timeout=(TIMEOUT_CONNECT, TIMEOUT_READ),
        )
    except requests.RequestException as e:
        return {"error": f"upstream: {e!s}"}, 502

    if not upstream.ok:
        body = upstream.text[:2000]
        return {"error": f"pocket_tts HTTP {upstream.status_code}", "detail": body}, 502

    def generate():
        for chunk in upstream.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return Response(
        generate(),
        mimetype=upstream.headers.get("Content-Type", "audio/wav"),
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    host = os.environ.get("EDGE_TTS_PROXY_HOST", "0.0.0.0")
    port = int(os.environ.get("EDGE_TTS_PROXY_PORT", "8765"))
    app.run(host=host, port=port, threaded=True)
