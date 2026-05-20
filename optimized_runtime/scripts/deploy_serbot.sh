#!/bin/bash

# Deploy SERBot Runtime to Device
# Usage: ./deploy_serbot.sh SERBOT_IP USERNAME

set -e

SERBOT_IP="${1:-192.168.1.100}"
USERNAME="${2:-ubuntu}"
REMOTE_DIR="/home/${USERNAME}/novacare-runtime"

SERBOT_IP="${1:-192.168.1.100}"
USERNAME="${2:-ubuntu}"
# Remote base directory on the robot
REMOTE_DIR="/home/${USERNAME}/novacare-runtime"

# Resolve local repo root and optimized_runtime path reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_RUNTIME_DIR="$REPO_ROOT/optimized_runtime"

if [ ! -d "$LOCAL_RUNTIME_DIR" ]; then
	echo "Error: could not find local optimized_runtime at $LOCAL_RUNTIME_DIR"
	exit 2
fi

echo "Using runtime root: $RUNTIME_ROOT"

echo "=========================================="
echo "NovaCare SERBot Runtime Deployment"
echo "=========================================="
echo "[2/5] Copying runtime code from $LOCAL_RUNTIME_DIR..."
scp -r "$LOCAL_RUNTIME_DIR" $USERNAME@$SERBOT_IP:$REMOTE_DIR/
echo ""

echo "[3/5] Ensuring remote docker directory exists..."
ssh $USERNAME@$SERBOT_IP "mkdir -p $REMOTE_DIR/optimized_runtime/docker"
ssh $USERNAME@$SERBOT_IP "mkdir -p $REMOTE_DIR"

# 2. Copy optimized runtime
scp "$SCRIPT_DIR/startup.sh" $USERNAME@$SERBOT_IP:$REMOTE_DIR/
scp -r "$RUNTIME_ROOT" $USERNAME@$SERBOT_IP:$REMOTE_DIR/

# 3. Copy Docker files
echo "[3/5] Copying Docker configuration..."
REMOTE_RUNTIME_START="$REMOTE_DIR/optimized_runtime/scripts/startup.sh"
ssh $USERNAME@$SERBOT_IP "\
	if [ -f '$REMOTE_RUNTIME_START' ]; then \
		if command -v docker >/dev/null 2>&1; then \
			echo 'Docker detected on remote — attempting docker compose in runtime/docker'; \
			if command -v docker-compose >/dev/null 2>&1; then \
				docker-compose -f '$REMOTE_DIR/optimized_runtime/docker/docker-compose.yml' up -d --build || bash '$REMOTE_RUNTIME_START'; \
			else \
				docker compose -f '$REMOTE_DIR/optimized_runtime/docker/docker-compose.yml' up -d --build || bash '$REMOTE_RUNTIME_START'; \
			fi; \
		else \
			echo 'Docker not detected on remote — running runtime startup script'; \
			bash '$REMOTE_RUNTIME_START'; \
		fi; \
	else \
		echo 'Remote runtime startup script not found, aborting'; exit 3; \
	fi"
			docker-compose -f docker/docker-compose.yml up -d --build; \
		else \
			# Try modern `docker compose`
			docker compose -f docker/docker-compose.yml up -d --build || ./startup.sh; \
		fi \
	else \
		echo 'Docker not detected on remote — running startup.sh'; \
		./startup.sh; \
	fi"

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "Robot UI: http://$SERBOT_IP:8080"
echo "WebSocket: ws://$SERBOT_IP:9999"
echo "=========================================="
