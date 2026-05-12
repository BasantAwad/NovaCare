#!/bin/bash
# ===========================================================================
# NovaCare — Robot Deployment Script (Docker)
# ===========================================================================
# Redepoys the entire NovaCare stack on the Jetson Nano.
# ===========================================================================

set -e

# Detect Docker Compose command (support both v1 and v2)
if docker compose version >/dev/null 2>&1; then
    DOCKER_CMD="docker compose"
elif docker-compose version >/dev/null 2>&1; then
    DOCKER_CMD="docker-compose"
else
    echo "❌ Error: Docker Compose is not installed. Please install it first."
    exit 1
fi

echo "🚀 Starting NovaCare Deployment using $DOCKER_CMD..."

# 1. Pull latest changes
echo "[1/3] Fetching latest code from GitHub..."
# Ensure we are on the correct branch and sync it
git fetch origin ramez-unified-branch
git reset --hard origin/ramez-unified-branch

# 2. Restart Services
echo "[2/3] Building and starting containers in the background..."
$DOCKER_CMD down || true
$DOCKER_CMD up --build -d

# 3. Finalizing
echo "[3/3] Deployment complete!"
echo "----------------------------------------------------"
echo "  Frontend:  http://localhost:3000"
echo "  Robot API: http://localhost:9000/health"
echo "  ASL API:   http://localhost:8001/docs"
echo "  LLM API:   http://localhost:5000"
echo "----------------------------------------------------"
echo "Monitor logs: $DOCKER_CMD logs -f"
echo "Stop services: $DOCKER_CMD down"
