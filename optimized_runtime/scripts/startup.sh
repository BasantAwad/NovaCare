#!/bin/bash

# SERBot Startup Script
# Initializes and starts the optimized NovaCare runtime

set -e

echo "=========================================="
echo "NovaCare SERBot Startup"
echo "=========================================="

# Environment setup
export PYTHONUNBUFFERED=1
export ROBOT_MODE=serbot
export ORCHESTRATOR_HOST=0.0.0.0
export ORCHESTRATOR_PORT=9999

# Directories
RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$RUNTIME_DIR/logs"

# Create log directory
mkdir -p $LOG_DIR

# Set up logging
LOG_FILE="$LOG_DIR/runtime_$(date +%Y%m%d_%H%M%S).log"

echo "Log file: $LOG_FILE"
echo ""

# Function for cleanup on exit
cleanup() {
    echo "Shutting down runtime..."
    # Add cleanup commands here
}

trap cleanup EXIT

# Start the runtime
echo "Starting optimized runtime..."
start_ui() {
    # Attempt to start Robot UI with increasing fallbacks:
    # 1) docker compose (if available and compose file present)
    # 2) npm start (if Node/npm available in PATH)
    # 3) python simple HTTP server fallback (serves files under robot_ui)

    UI_LOG="$LOG_DIR/robot_ui_$(date +%Y%m%d_%H%M%S).log"

    # Try Docker compose inside runtime/docker
    if command -v docker >/dev/null 2>&1; then
        if [ -f "$RUNTIME_DIR/docker/docker-compose.yml" ]; then
            echo "[UI] Docker detected; attempting docker compose up" >> "$UI_LOG" 2>&1 || true
            docker compose -f "$RUNTIME_DIR/docker/docker-compose.yml" up -d --build >> "$UI_LOG" 2>&1 || docker-compose -f "$RUNTIME_DIR/docker/docker-compose.yml" up -d --build >> "$UI_LOG" 2>&1 || true
            return
        fi
    fi

    # Try Node/npm start (development mode)
    if command -v npm >/dev/null 2>&1 && [ -d "$RUNTIME_DIR/robot_ui" ]; then
        echo "[UI] npm detected; starting robot_ui via npm start" >> "$UI_LOG" 2>&1
        (cd "$RUNTIME_DIR/robot_ui" && nohup npm start >> "$UI_LOG" 2>&1 &) || true
        return
    fi

    # Fallback: serve the robot_ui directory with a simple Python HTTP server
    if [ -d "$RUNTIME_DIR/robot_ui" ]; then
        echo "[UI] Falling back to python http.server to serve robot_ui" >> "$UI_LOG" 2>&1
        nohup python3 -m http.server 8080 --directory "$RUNTIME_DIR/robot_ui" >> "$UI_LOG" 2>&1 &
        return
    fi

    echo "[UI] No UI available to start" >> "$UI_LOG" 2>&1
}

# Start Robot UI (non-blocking)
echo "Starting Robot UI (if available)..."
start_ui
echo "Waiting a few seconds for UI to initialize..."
sleep 4

# Ensure Python can locate the `optimized_runtime` package by using the repo root
REPO_ROOT="$(cd "$RUNTIME_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
cd "$REPO_ROOT"

# Launch runtime (foreground)
python3 -m optimized_runtime.runtime.launcher \
    --mode serbot \
    --log-level INFO \
    2>&1 | tee -a $LOG_FILE

echo "Runtime exited"
