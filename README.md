<p align="center">
  <img src="https://raw.githubusercontent.com/BasantAwad/BasantAwad/main/assets/basant-terminal-banner.png" alt="Terminal-inspired project banner" width="100%" />
</p>

# NovaCare — Intelligent Care Rover System

NovaCare is an AI-powered assistive rover ecosystem for accessible care, combining conversational assistance, emotion-aware interactions, sign-language recognition, real-time telemetry, and hardware control.

## System areas

- AI/ML services for conversational therapy, emotion detection, and ASL recognition.
- Flutter interfaces for patients and caregivers, including SOS workflows.
- Web dashboards for rover, patient, and health telemetry.
- ESP32 and Jetson-oriented rover integration.
- Node.js, Python, and Docker services with Firebase and multi-provider model support.

## Stack

TypeScript, Python, Dart/Flutter, Node.js, MongoDB, WebSockets, BLE, Docker, ESP32, Jetson, Firebase.

## Getting started

Use the root `requirements.txt` and the Docker Compose files for the target deployment profile. The `docs/`, `tests/`, `optimized_runtime/`, `laptop_services/`, and `serbot_deployment/` directories document the supported environments.

```bash
docker compose up --build
```

Hardware and cloud integrations require their own credentials and device configuration; keep secrets in environment variables and never commit them.

## Design focus

NovaCare prioritizes low-latency APIs, continuous BLE ingestion, real-time WebSocket telemetry, modular AI providers, and an accessible interaction model for assistive technology.
