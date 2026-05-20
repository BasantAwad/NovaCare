# NovaCare Backend Enhancements Documentation

## Overview
This document serves as a comprehensive log of the backend architecture enhancements implemented for the NovaCare API services. The core additions heavily revolve around implementing **SOLID principles**, decoupled **Repository Patterns**, and enforcing **Security-First PHI constraints**.

Three major independent feature sets have been fully implemented within the `backend/routes/api/` ecosystem:
1. Health Report Generation Service
2. Medication Management Service
3. Medication Reminder Engine (APScheduler)

Below is an engineering summary of all modifications, logic flows, and added files.

---

## 1. Health Report Generation Service
**File Added/Modified**: `reports.py`

### Goal
Provide a decoupled modular pipeline to aggregate mocked patient vitals, compute their health statuses, and stream the generated Health Report securely as a direct PDF binary output.

### Key Implementations
* **Custom Error Handling**: Defined explicit custom exceptions (`HealthReportError`, `UnauthorizedError`, `PatientNotFoundError`). These ensure unified JSON response handling mapping to strict HTTP Response Codes (`403`, `404`) rather than generic `500` server blunders.
* **Separation of Concerns Layering**:
  * **Data Layer (`fetch_mocked_vitals`)**: Abstracted pure data retrieval logic using Python Type Hints (`List[Dict[str, Any]]`).
  * **Business Logic (`analyze_vitals`)**: Abstracted raw computations measuring abnormal Heart Rates (`>100`) and standard SpO2 bounds (`<92`), maintaining a single responsibility away from the routing tier.
* **PDF Blob Streaming**: The Flask response stream directly attaches a binary output simulating a legitimate `%PDF-1.4` file through the `send_file()` command using `mimetype='application/pdf'` (facilitating frontend downloads).
* **Security Validation**: Intercepts `patient_id` parameter utilizing strict Regex `^[A-Za-z0-9]{1,20}$` checks preventing path-traversals.

---

## 2. Medication Management Service
**File Added/Modified**: `medication.py`

### Goal
A secure RESTful pipeline empowering patients and doctors to process CRUD commands on medication schedules without trusting unsanitized inputs heavily relying upon Pydantic Data validation.

### Key Implementations
* **Mock Repository Pattern**: Decoupled the entire module from the pending (`SQLAlchemy`) database architecture. Developed `MockMedicationRepository` to abstract `add_medication`, `get_todays_schedule`, and `log_dose` routines. 
* **Pydantic Validation**:
  * Added global definitions for `MedicationCreateSchema` and `DoseStatusUpdateSchema`.
  * Protects business logic from malformed dates, invalid timespans natively wrapping exceptions directly to `400 Bad Request` cascades.
* **Complex Safety Overrides**:
  * **Overlapping Guard**: Rejects incoming schedules returning `409 Conflict` if the patient attempts to log an overlapping schedule timeslot for the exact same drug name.
  * **Safety Flag Alerts**: When updating a dose's status (`Taken`/`Skipped`), the service inspects differential time gaps. If the patient logs > `2.0 hours` outside the original threshold, a `safety_warning` payload surfaces inside the JSON warning of adverse medical actions.
* **Auth Integration Middleware (`@auth_required`)**: Rejects incoming body-spoofing by blindly overriding target payloads in favor of reading identifiers solely from `current_user.id`.

---

## 3. Medication Reminder Engine
**File Added/Modified**: `reminders.py`

### Goal
Initiate an automated, real-time background scanning agent broadcasting privacy-constrained notifications prior to upcoming user medication schedules (15-minute lead bounds). 

### Key Implementations
* **APScheduler Integration**: Python's `BackgroundScheduler` directly wired natively into Flask executing intervals every `60 seconds` checking an aggregated array of queued doses. 
* **Abstract Provider Interface (Protocol)**: Uses Python's runtime abstractions via `NotificationProvider` interface (`send(patient_id, title...)`). This allows dev-teams to effortlessly hot-swap delivery carriers (e.g. Twilio SMS, APNS Push, Email) without rewriting the primary scheduler mechanism.
* **Rate-Limiting & Notification Storm Defense**: The internal `.check_exists()` state mechanism aborts iteration dispatches if an alert representing the exactly equivalent payload object (`patient_id` + `med_id` + `scheduled_time`) has already been logged.
* **PHI Security Architecture**: Hardened for mobile locked-screens. Sends generic title/body frames (*"NovaCare Medication Reminder"*), while stripping raw drug and dosage metrics out bounding them natively strictly to the `"hidden_details"` parameter for secure authenticated app parsing.

---

## 4. Dependencies Updated
**File Modified**: `requirements.txt`

The following global Python packages were formally added to orchestrate these capabilities:

1. **`loguru>=0.7.2`**: Replaced standard unformatted prints with fully traceable structured system logging, tagging transactions identically. (Audit trails).
2. **`pydantic>=2.5.0`**: Brought V2-grade JSON bounds schema validation strictly enforcing Type Safety and Date boundaries in HTTP Post actions.
3. **`APScheduler>=3.10.4`**: Used entirely for executing out-of-band asynchronous jobs securely natively inside Flask application bounds without relying prematurely on full-blown `Celery` / `Redis` pipelines.
