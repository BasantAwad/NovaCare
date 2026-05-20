#!/bin/bash

# Health Check Script
# Monitors runtime health and service connectivity

set -e

SERBOT_IP="${1:-localhost}"
TIMEOUT=5

echo "=========================================="
echo "NovaCare Runtime Health Check"
echo "=========================================="
echo "Target: $SERBOT_IP"
echo ""

# Check WebSocket connectivity
echo "[1] Checking WebSocket server..."
if timeout $TIMEOUT bash -c "echo '' > /dev/tcp/$SERBOT_IP/9999" 2>/dev/null; then
    echo "✓ WebSocket server (9999): OK"
else
    echo "✗ WebSocket server (9999): FAILED"
fi

# Check Robot UI server
echo ""
echo "[2] Checking Robot UI server..."
if curl -s -f -m $TIMEOUT http://$SERBOT_IP:8080/health > /dev/null; then
    echo "✓ Robot UI (8080): OK"
else
    echo "✗ Robot UI (8080): FAILED"
fi

# Check LLM service (laptop)
echo ""
echo "[3] Checking LLM service..."
if curl -s -f -m $TIMEOUT http://$SERBOT_IP:5000/health > /dev/null; then
    echo "✓ LLM Service (5000): OK"
else
    echo "✗ LLM Service (5000): FAILED (expected if laptop not running)"
fi

# Check ASL service (laptop)
echo ""
echo "[4] Checking ASL service..."
if curl -s -f -m $TIMEOUT http://$SERBOT_IP:8000/health > /dev/null; then
    echo "✓ ASL Service (8000): OK"
else
    echo "✗ ASL Service (8000): FAILED (expected if laptop not running)"
fi

echo ""
echo "=========================================="
echo "Health check complete"
echo "=========================================="
