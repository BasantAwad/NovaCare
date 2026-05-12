# NovaCare Robot Service - Dockerfile
# Optimized for Jetson Nano / ARM64

FROM python:3.10-slim-bullseye

# Install system dependencies for hardware access and audio
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    vlc \
    aplay \
    espeak \
    alsa-utils \
    libasound2-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* || true

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the service
COPY . .

# Expose the robot service port
EXPOSE 9000

# Run with unbuffered output for logging
ENV PYTHONUNBUFFERED=1

CMD ["python", "robot_service.py"]
