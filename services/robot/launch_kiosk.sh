#!/usr/bin/env bash
# Opens the robot face UI fullscreen on the SerBot LCD (auto-restart on exit).

ROBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROBOT_DIR}/logs"
mkdir -p "$LOG_DIR"

export DISPLAY="${DISPLAY:-:0}"
export ROBOT_SERVICE_PORT="${ROBOT_SERVICE_PORT:-9000}"
export UI_URL="${UI_URL:-http://127.0.0.1:${ROBOT_SERVICE_PORT}/ui?minimal=1}"

# Force kill any lingering browsers to ensure new flags (camera/mic permissions) are applied
killall -9 chromium-browser chromium google-chrome chrome 2>/dev/null || true

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [kiosk] $*"
}

find_browser() {
  for cmd in chromium-browser chromium google-chrome chrome; do
    command -v "$cmd" >/dev/null 2>&1 && echo "$cmd" && return 0
  done
  return 1
}

wait_for_http() {
  local url=$1
  local tries=${2:-90}
  local i=1
  while [ "$i" -le "$tries" ]; do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 1
    i=$((i + 1))
  done
  return 1
}

browser="$(find_browser)" || {
  log "No Chromium/Chrome found"
  exit 1
}

wait_for_http "http://127.0.0.1:${ROBOT_SERVICE_PORT}/health" 90 || \
  log "REST API slow to start — opening UI anyway"

while true; do
  log "Opening ${UI_URL}"
  "$browser" \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --no-first-run \
    --disable-translate \
    --disable-session-crashed-bubble \
    --check-for-update-interval=31536000 \
    --no-sandbox \
    --user-data-dir=/tmp/chromium_kiosk \
    --unsafely-treat-insecure-origin-as-secure=http://10.174.134.241:3000 \
    --use-fake-ui-for-media-stream \
    --allow-running-insecure-content \
    --autoplay-policy=no-user-gesture-required \
    "${UI_URL}" >> "${LOG_DIR}/kiosk.log" 2>&1 || true
  log "Browser closed — relaunch in 5s"
  sleep 5
done
