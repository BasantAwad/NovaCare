#!/bin/bash

# NovaCare Omni-Runner
# Starts both the Backend Auth Server and the Frontend Next.js Server

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}      Starting NovaCare Platform       ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Define paths
ROOT_DIR=$(pwd)
BACKEND_DIR="$ROOT_DIR/services/auth-backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

# Cleanup function to kill background processes when stopping
cleanup() {
    echo -e "\n${YELLOW}Shutting down NovaCare cleanly...${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit
}

trap cleanup EXIT INT TERM

# --- Start Backend ---
echo -e "${GREEN}>>> Starting Auth Backend (Port 5001)${NC}"
cd $BACKEND_DIR

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi
python run.py &
BACKEND_PID=$!

sleep 2

# --- Start Frontend ---
echo -e "${GREEN}>>> Starting Next.js Frontend (Port 3000)${NC}"
cd $FRONTEND_DIR
npm run dev

# Process waits here
wait
