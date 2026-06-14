#!/usr/bin/env bash
# =============================================================================
# NovaCare SerBot — minimal fault-tolerant launcher
# =============================================================================
# SCP this entire services/robot folder to the SerBot, then run:
#   chmod +x run_serbot.sh stop_serbot.sh
#   ./run_serbot.sh
#
# Starts (each auto-restarts on crash):
#   - TCP command server  (port 5555, MJPEG 5557)
#   - REST / robot face UI (port 9000, /ui)
#   - Chromium kiosk on the robot LCD (no user interaction)
#
# No AI, no watch sync, no on-robot vision logic (NOVACARE_MINIMAL=1).
# =============================================================================

set -u

ROBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROBOT_DIR}/logs"
PID_DIR="${ROBOT_DIR}/.pids"
VENV="${ROBOT_DIR}/venv"

mkdir -p "$LOG_DIR" "$PID_DIR"

# ---------------------------------------------------------------------------
# Minimal I/O mode — SerBot as input/output bridge only
# ---------------------------------------------------------------------------
export NOVACARE_MINIMAL=1
export NOVACARE_LIGHTWEIGHT=1
export NOVACARE_USE_OPENCV=0
export LIDAR_ENABLED=false
export DISPLAY="${DISPLAY:-:0}"
export ROBOT_SERVICE_HOST="${ROBOT_SERVICE_HOST:-0.0.0.0}"
export ROBOT_SERVICE_PORT="${ROBOT_SERVICE_PORT:-9000}"
export UI_URL="${UI_URL:-http://127.0.0.1:${ROBOT_SERVICE_PORT}/ui?minimal=1}"

# Load local .env if present (does not override exports above unless unset)
if [ -f "${ROBOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROBOT_DIR}/.env"
  set +a
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

activate_venv() {
  cd "$ROBOT_DIR" || exit 1
  if [ ! -f "$VENV/bin/activate" ]; then
    log "Creating Python venv..."
    rm -rf "$VENV"
    python3 -m venv --system-site-packages "$VENV" || return 1
    # shellcheck disable=SC1091
    source "$VENV/bin/activate" || return 1
    pip install --upgrade pip wheel setuptools >/dev/null 2>&1 || true
    pip install -r requirements.txt >> "${LOG_DIR}/setup.log" 2>&1 || true
  else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate" || return 1
  fi
  if [ ! -f "${ROBOT_DIR}/.env" ] && [ -f "${ROBOT_DIR}/.env.example" ]; then
    cp "${ROBOT_DIR}/.env.example" "${ROBOT_DIR}/.env"
    log "Created .env from .env.example"
  fi
}

wait_for_http() {
  local url=$1
  local label=$2
  local tries=${3:-60}
  local i=1
  while [ "$i" -le "$tries" ]; do
    if curl -sf "$url" >/dev/null 2>&1; then
      log "$label is up ($url)"
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  log "WARN: $label did not respond at $url (continuing anyway)"
  return 1
}

find_browser() {
  for cmd in chromium-browser chromium google-chrome chrome; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "$cmd"
      return 0
    fi
  done
  return 1
}

# Run a command in a restart loop. Isolated — one crash does not stop others.
run_supervised() {
  local name=$1
  shift
  local pidfile="${PID_DIR}/${name}.pid"
  local logfile="${LOG_DIR}/${name}.log"

  (
    while true; do
      log "Starting ${name}..."
      "$@" >> "$logfile" 2>&1
      local code=$?
      log "${name} exited (code ${code}) — restart in 3s"
      sleep 3
    done
  ) &

  echo $! > "$pidfile"
  log "${name} supervisor PID $(cat "$pidfile") → ${logfile}"
}

stop_existing() {
  if [ -x "${ROBOT_DIR}/stop_serbot.sh" ]; then
    bash "${ROBOT_DIR}/stop_serbot.sh" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  log "Shutting down SerBot services..."
  stop_existing
  exit 0
}

trap cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
main() {
  log "NovaCare SerBot launcher"
  log "Robot dir: ${ROBOT_DIR}"
  log "Display:   ${DISPLAY}"
  log "UI URL:    ${UI_URL}"

  stop_existing
  activate_venv || {
    log "ERROR: Could not activate Python environment"
    exit 1
  }

  run_supervised tcp bash -c "cd '${ROBOT_DIR}' && source '${VENV}/bin/activate' && exec python3 tcp_command_server.py"
  run_supervised rest bash -c "cd '${ROBOT_DIR}' && source '${VENV}/bin/activate' && exec python3 robot_service.py"
  run_supervised watch bash -c "cd '${ROBOT_DIR}' && source '${VENV}/bin/activate' && exec python3 watch_integration.py"
  run_supervised kiosk bash "${ROBOT_DIR}/launch_kiosk.sh"

  log "All supervisors started. Logs: ${LOG_DIR}/"
  log "TCP port 5555 | REST/UI port ${ROBOT_SERVICE_PORT} | MJPEG port 5557"
  log "Press Ctrl+C to stop."

  wait
}

main "$@"
