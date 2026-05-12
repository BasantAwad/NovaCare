# NovaCare LLM Backend - Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio/emotion detection if needed
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the LLM API port
EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "start_server.py"]
