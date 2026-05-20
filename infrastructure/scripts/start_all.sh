#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  NovaCare — One-Command Service Launcher (macOS / Linux)
# ──────────────────────────────────────────────────────────────
#  Starts all 3 NovaCare services:
#    1. ASL Model API     (FastAPI  → port 8000)
#    2. LLM Backend       (Flask    → port 5000)
#    3. Frontend           (Next.js  → port 3000)
#
#  Usage:
#    chmod +x start_all.sh
#    ./start_all.sh
#
#  On macOS this opens three new Terminal.app tabs.
#  On Linux it falls back to running all three in the background
#  within this terminal (logs interleaved).
# ──────────────────────────────────────────────────────────────

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Colors ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

banner() {
  echo ""
  echo -e "${CYAN}  ============================================${NC}"
  echo -e "${BOLD}   NovaCare — Starting All Services${NC}"
  echo -e "${CYAN}  ============================================${NC}"
  echo ""
  echo -e "   [1] ASL Model API     ${YELLOW}(port 8000)${NC}"
  echo -e "   [2] LLM Backend       ${YELLOW}(port 5000)${NC}"
  echo -e "   [3] Frontend           ${YELLOW}(port 3000)${NC}"
  echo ""
  echo -e "${CYAN}  ============================================${NC}"
  echo ""
}

# ── Service runner scripts (written to tmp files) ────────────

asl_script() {
  cat <<'INNER'
echo "=== NovaCare — ASL Model API ==="
cd "__ROOT__/services/asl-model"

if [ ! -d "venv" ]; then
  echo "[!] No venv found. Creating..."
  python3 -m venv venv
  source venv/bin/activate
  echo "[*] Installing dependencies..."
  pip install -r requirements.txt
  echo "[OK] Dependencies installed"
else
  source venv/bin/activate
  echo "[OK] ASL Model venv activated"
fi

echo "[*] Starting FastAPI on port 8000..."
python -m api.main --port 8000
INNER
}

llm_script() {
  cat <<'INNER'
echo "=== NovaCare — LLM Backend ==="
cd "__ROOT__/services/llm-backend"

if [ ! -d "venv" ]; then
  echo "[*] Creating venv..."
  python3 -m venv venv
  source venv/bin/activate
  echo "[*] Installing dependencies..."
  pip install -r requirements.txt
  echo "[OK] Dependencies installed"
else
  source venv/bin/activate
  echo "[OK] LLM Backend venv activated"
fi

if [ ! -f ".env" ]; then
  echo "[!] WARNING: No .env file found!"
  echo "[!] Create .env with OLLAMA_MODEL and/or HUGGINGFACE_API_KEY (see README.md)"
fi

echo "[*] Starting Flask on port 5000..."
python start_server.py
INNER
}

fe_script() {
  cat <<'INNER'
echo "=== NovaCare — Frontend ==="
cd "__ROOT__/frontend"

if [ ! -d "node_modules" ]; then
  echo "[*] Installing npm dependencies..."
  npm install
  echo "[OK] Dependencies installed"
else
  echo "[OK] node_modules found"
fi

if [ ! -f ".env.local" ]; then
  echo "[!] WARNING: No .env.local file found!"
  echo "[!] Create .env.local with at least:"
  echo "    NEXT_PUBLIC_NOVABOT_API_URL=http://localhost:5000"
fi

echo "[*] Starting Next.js on port 3000..."
npm run dev
INNER
}

# ── Write temp scripts ───────────────────────────────────────

write_script() {
  local name="$1"
  local content="$2"
  local tmp_file
  tmp_file="$(mktemp "/tmp/novacare_${name}_XXXXXX.sh")"
  echo "$content" | sed "s|__ROOT__|${ROOT}|g" > "$tmp_file"
  chmod +x "$tmp_file"
  echo "$tmp_file"
}

# ── macOS: open in Terminal.app tabs ─────────────────────────

open_mac_tab() {
  local title="$1"
  local script_path="$2"
  osascript <<EOF
tell application "Terminal"
  activate
  do script "echo -e '\\033]0;${title}\\007'; bash '${script_path}'"
end tell
EOF
}

# ── Linux: run in background within this terminal ────────────

run_linux_bg() {
  local title="$1"
  local script_path="$2"
  echo -e "${CYAN}[*] Starting ${title} in background...${NC}"
  bash "$script_path" &
}

# ── Main ─────────────────────────────────────────────────────

banner

ASL_SCRIPT="$(write_script "asl" "$(asl_script)")"
LLM_SCRIPT="$(write_script "llm" "$(llm_script)")"
FE_SCRIPT="$(write_script "fe"  "$(fe_script)")"

if [[ "$(uname)" == "Darwin" ]]; then
  echo -e "${GREEN}[*] Detected macOS — opening Terminal tabs...${NC}"
  echo ""

  open_mac_tab "NovaCare — ASL Model API (8000)" "$ASL_SCRIPT"
  sleep 2
  open_mac_tab "NovaCare — LLM Backend (5000)" "$LLM_SCRIPT"
  sleep 2
  open_mac_tab "NovaCare — Frontend (3000)" "$FE_SCRIPT"
else
  echo -e "${GREEN}[*] Detected Linux — starting services in background...${NC}"
  echo ""

  run_linux_bg "ASL Model API" "$ASL_SCRIPT"
  sleep 2
  run_linux_bg "LLM Backend" "$LLM_SCRIPT"
  sleep 2
  run_linux_bg "Frontend" "$FE_SCRIPT"

  echo ""
  echo -e "${YELLOW}[*] All services started in background. Use 'jobs' to list them.${NC}"
  wait
fi

echo ""
echo -e "${CYAN}  ============================================${NC}"
echo -e "${GREEN}   All services launched!${NC}"
echo -e "${CYAN}  ============================================${NC}"
echo ""
echo -e "   ASL Model API:  ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "   LLM Backend:    ${YELLOW}http://localhost:5000${NC}"
echo -e "   Frontend:       ${YELLOW}http://localhost:3000${NC}"
echo ""
