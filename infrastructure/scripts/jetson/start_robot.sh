#!/bin/bash
# ===========================================================================
# NovaCare — Robot Startup Script
# ===========================================================================
# Launches all services on the SERBot Prime X hardware.
# Run with: sudo bash scripts/jetson/start_robot.sh
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo ""
echo "  ============================================"
echo "   NovaCare — SERBot Startup"
echo "  ============================================"
echo "   Project: $PROJECT_DIR"
echo "  ============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Robot Service (Hardware HAL + REST API, port 9000)
# ---------------------------------------------------------------------------
echo "[1/4] Starting Robot Service (port 9000)..."
cd "$PROJECT_DIR/services/robot"

if [ ! -d "venv" ]; then
    echo "  [*] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Copy .env if not present
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  [!] Created .env from template — review settings"
fi

python robot_service.py &
ROBOT_PID=$!
echo "  [OK] Robot Service started (PID: $ROBOT_PID)"
sleep 2

# ---------------------------------------------------------------------------
# 2. ASL Model API (FastAPI, port 8000)
# ---------------------------------------------------------------------------
echo "[2/4] Starting ASL Model API (port 8001)..."
cd "$PROJECT_DIR/services/asl-model"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "  [!] No venv found for ASL model — skipping"
fi

python -m api.main --port 8001 &
ASL_PID=$!
echo "  [OK] ASL Model API started (PID: $ASL_PID)"
sleep 2

# ---------------------------------------------------------------------------
# 3. LLM Backend (Flask, port 5000)
# ---------------------------------------------------------------------------
echo "[3/4] Starting LLM Backend (port 5000)..."
cd "$PROJECT_DIR/services/llm-backend"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "  [!] No venv found for LLM backend — skipping"
fi

python start_server.py &
LLM_PID=$!
echo "  [OK] LLM Backend started (PID: $LLM_PID)"
sleep 2

# ---------------------------------------------------------------------------
# 4. Frontend (Next.js, port 3000)
# ---------------------------------------------------------------------------
echo "[4/4] Starting Frontend (port 3000)..."
cd "$PROJECT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "  [*] Installing npm dependencies..."
    npm install
fi

npm run dev &
FE_PID=$!
echo "  [OK] Frontend started (PID: $FE_PID)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "  ============================================"
echo "   All Services Running!"
echo "  ============================================"
echo ""
echo "   Robot Service:  http://localhost:9000/health"
echo "   ASL Model API:  http://localhost:8001/docs"
echo "   LLM Backend:    http://localhost:5000"
echo "   Frontend:       http://localhost:3000"
echo ""
echo "   PIDs: Robot=$ROBOT_PID ASL=$ASL_PID LLM=$LLM_PID FE=$FE_PID"
echo ""
echo "   Press Ctrl+C to stop all services"
echo "  ============================================"
echo ""

# ---------------------------------------------------------------------------
# Launch Chromium in Kiosk Mode (robot touchscreen)
# ---------------------------------------------------------------------------
echo "[*] Launching Chromium in kiosk mode on touchscreen..."
sleep 5  # Wait for frontend to be ready

# Check if running in a graphical environment
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    chromium-browser --kiosk --noerrdialogs --disable-translate \
        --no-first-run --fast --fast-start --disable-infobars \
        --use-fake-ui-for-media-stream \
        --unsafely-treat-insecure-origin-as-secure="http://localhost:3000" \
        --disable-features=TranslateUI --disk-cache-dir=/dev/null \
        "http://localhost:3000/rover" 2>/dev/null &
    echo "  [OK] Chromium kiosk launched"
else
    echo "  [!] No display detected — skipping kiosk mode"
fi

# ---------------------------------------------------------------------------
# Wait for all background jobs
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[*] Shutting down all services..."
    kill $ROBOT_PID $ASL_PID $LLM_PID $FE_PID 2>/dev/null
    wait
    echo "[OK] All services stopped."
}

trap cleanup SIGINT SIGTERM

wait
