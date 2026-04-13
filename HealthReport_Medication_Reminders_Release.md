# Feature Release: Core API Services & Architecture Refactoring

## Overview
This PR introduces three major decoupled backend features to construct the foundation for the NovaCare platform, adhering to SOLID principles and Clean Architecture (Service & Mock-Repository pattern). We additionally resolved an application bootstrap bug (`ModuleNotFoundError`) allowing the Flask server to map local dependencies correctly.

## 🚀 Features Implemented

### 1. Health Report Generation Service
- **Endpoint:** `GET /api/report/pdf/<patient_id>`
- **Functionality:** Aggregates mocked user vitals (Heart Rate, SpO2) and dynamically streams a structurally valid binary PDF payload via Flask's `send_file()` over HTTP.
- **Security Check:** Implements strict regex (`^[A-Za-z0-9]{1,20}$`) preventing path traversal and injection attacks on the endpoint parameter.
- **Custom Errors:** Developed robust sub-classed exceptions (`HealthReportError`, `UnauthorizedError`) handling logic faults safely without returning 500 stack traces.

### 2. Medication Management Subsystem
- **Endpoints:** `POST /api/medication`, `GET /api/medication/today`, `PATCH /api/medication/<dose_id>/status`
- **Validation:** Integrated **Pydantic** (`MedicationCreateSchema`) to strictly validate bounds, dates, and timestamp types upon creation so malformed arrays are caught instantly (`400 Bad Request`).
- **Conflict Engine:** Added overlapping schedule protection parsing existing `MockMedicationRepository` states to prevent the identical drug schedules intercepting. Returns `409 Conflict`.
- **Safety Flags:** Tracks modified dosage events. Generates passive payload warnings if a drug is checked-in > 2.0 hours away from the original timeline.

### 3. Medication Reminder Engine
- **Engine Logic:** Added Python **APScheduler** to automatically scan upcoming medications within a 15-minute timeframe mapping every 60 seconds natively in the Flask background block.
- **De-coupled Dispatchers:** Implemented an abstract `NotificationProvider` Interface. Currently mocks pushing iOS generic alerts (*"NovaCare Medication Reminder"*) and strips raw sensitive drug data hiding it safely to comply with lock-screen PHI mandates. 
- **Notification Storm Prevention:** Engine references a dictionary lookup prior to routing to prevent looping duplicate logic dispatches across instances.

## 🛠 Refactoring & System Fixes
- **App Factory Structuring Fix:** Solved the persistent `ModuleNotFoundError` encountered starting `run.py`. Updated all routing pointers from tracing `from app.routes` to `from backend.routes` aligning with the absolute file-system architecture. 
- **Registered the Reminders Blueprint:** Dynamically appended `init_reminders_module()` inside `__init__.py` to start the daemon process upon server boot globally.

## 📦 Dependency Updates (`requirements.txt`)
- `loguru>=0.7.2` (Globally structured transparent audit & diagnostic logs).
- `pydantic>=2.5.0` (Strict schema runtime validations).
- `APScheduler>=3.10.4` (Background automated daemons).
