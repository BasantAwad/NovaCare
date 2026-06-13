#!/usr/bin/env bash
# Stop all processes started by run_serbot.sh

ROBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${ROBOT_DIR}/.pids"

stop_pidfile() {
  local name=$1
  local pidfile="${PID_DIR}/${name}.pid"
  if [ -f "$pidfile" ]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
    echo "Stopped ${name} supervisor"
  fi
}

stop_pidfile tcp
stop_pidfile rest
stop_pidfile kiosk

# Kill any leftover service processes from this folder
pkill -f "${ROBOT_DIR}/tcp_command_server.py" 2>/dev/null || true
pkill -f "${ROBOT_DIR}/robot_service.py" 2>/dev/null || true
pkill -f "${ROBOT_DIR}/launch_kiosk.sh" 2>/dev/null || true

# Kill kiosk browsers showing our UI (best effort)
pkill -f "http://127.0.0.1:.*/ui" 2>/dev/null || true

echo "SerBot services stopped."
