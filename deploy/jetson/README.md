# Jetson deployment snippets (NovaCare)

Systemd units in this folder assume:

- Install tree under `/opt/novacare/` (adjust paths in the unit files to match your layout).
- A dedicated Unix user `novabot` (or change `User=` / `Group=`).
- Pocket TTS venv at `/opt/novacare/edge-tts/venv` and proxy venv at `/opt/novacare/edge-tts-proxy/venv`.

Copy unit files:

```bash
sudo cp novacare-pocket-tts.service novacare-edge-tts-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now novacare-pocket-tts.service
sudo systemctl enable --now novacare-edge-tts-proxy.service
```

Full procedure and benchmark: [docs/tts.md](../../docs/tts.md).
