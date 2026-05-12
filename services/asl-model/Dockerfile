# NovaCare ASL Model Service - Dockerfile
# Optimized for Jetson Nano / ARM64

FROM python:3.10-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the service
COPY . .

# Expose the ASL API port
EXPOSE 8001

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "api.main", "--host", "0.0.0.0", "--port", "8001"]
