#!/bin/bash
# NovaCare SERBot Bootstrap Script
# This script initializes the lightweight runtime on the robot.

echo "🚀 Starting NovaCare SERBot Runtime Bootstrap..."

# 1. Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# 2. Check for Hardware access
echo "🔍 Checking for camera and serial devices..."
[ -e /dev/video0 ] && echo "✅ Camera detected" || echo "⚠️ Camera not found"
[ -e /dev/ttyUSB0 ] && echo "✅ LiDAR/Serial detected" || echo "⚠️ LiDAR not found"

# 3. Pull/Build containers
echo "🛠️ Building lightweight services..."
docker-compose build

# 4. Start services
echo "✨ Launching services..."
docker-compose up -d

echo "✅ SERBot Runtime is now running."
echo "📺 Robot UI: http://localhost:3000 (if running locally)"
echo "📡 WebSocket: ws://localhost:9999"
