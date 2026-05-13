#!/bin/bash

# Deploy SERBot Runtime to Device
# Usage: ./deploy_serbot.sh SERBOT_IP USERNAME

set -e

SERBOT_IP="${1:-192.168.1.100}"
USERNAME="${2:-ubuntu}"
REMOTE_DIR="/home/${USERNAME}/novacare-runtime"

echo "=========================================="
echo "NovaCare SERBot Runtime Deployment"
echo "=========================================="
echo "Target: $USERNAME@$SERBOT_IP"
echo "Directory: $REMOTE_DIR"
echo ""

# 1. Create remote directories
echo "[1/5] Creating remote directories..."
ssh $USERNAME@$SERBOT_IP "mkdir -p $REMOTE_DIR"

# 2. Copy optimized runtime
echo "[2/5] Copying runtime code..."
scp -r ../optimized_runtime $USERNAME@$SERBOT_IP:$REMOTE_DIR/

# 3. Copy Docker files
echo "[3/5] Copying Docker configuration..."
scp -r ../optimized_runtime/docker/* $USERNAME@$SERBOT_IP:$REMOTE_DIR/docker/

# 4. Copy startup scripts
echo "[4/5] Copying startup scripts..."
scp startup.sh $USERNAME@$SERBOT_IP:$REMOTE_DIR/
ssh $USERNAME@$SERBOT_IP "chmod +x $REMOTE_DIR/startup.sh"

# 5. Start the runtime
echo "[5/5] Starting runtime..."
ssh $USERNAME@$SERBOT_IP "cd $REMOTE_DIR && ./startup.sh"

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "Robot UI: http://$SERBOT_IP:8080"
echo "WebSocket: ws://$SERBOT_IP:9999"
echo "=========================================="
