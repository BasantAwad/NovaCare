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
python3 -m optimized_runtime.runtime.launcher \
    --mode serbot \
    --log-level INFO \
    2>&1 | tee -a $LOG_FILE

echo "Runtime exited"
