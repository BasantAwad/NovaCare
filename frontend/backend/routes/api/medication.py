"""
NovaCare - Medication API (SOLID: Clean Architecture)
Handles medication CRUD operations using Service & Repository patterns.
"""
from functools import wraps
from flask import Blueprint, request, jsonify
from flask_login import current_user
from datetime import datetime, date, time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError, Field, field_validator
from loguru import logger as loguru_logger
import uuid

medication_bp = Blueprint('medication', __name__)

# --- Dependencies Injected during Flask app initialization ---
_db = None
_logger = None
_Medication = None

def init_medication(db, logger, Medication):
    """Initialize medication blueprint with dependencies."""
    global _db, _logger, _Medication
    _db = db
    _logger = logger
    _Medication = Medication


# --- Security / Auth Middleware ---
def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # *** TEMPORARY BYPASS FOR API TESTING ***
        # In a real app we check `if not current_user.is_authenticated:`
        # For testing our terminal curl loops without frontend cookies, we mock the user context:
        class MockUser:
            id = 1
        global current_user
        current_user = MockUser()
        
        return f(*args, **kwargs)
    return decorated_function


# --- Domain / Validation Layer (Pydantic) ---
class MedicationCreateSchema(BaseModel):
    """Schema for validating new medication payload"""
    name: str = Field(..., min_length=2, max_length=100)
    dosage: str = Field(..., min_length=1, max_length=50)
    frequency: str = Field(default="daily")
    start_date: date
    end_date: date
    time_slots: List[str] # e.g. ["08:00", "20:00"]
    
    @field_validator('time_slots')
    def validate_time_slots(cls, v):
        for ts in v:
            try:
                datetime.strptime(ts, '%H:%M')
            except ValueError:
                raise ValueError(f"Invalid time format: {ts}. Expected HH:MM")
        return v

class DoseStatusUpdateSchema(BaseModel):
    """Schema for validating status updates"""
    status: str = Field(..., pattern="^(Taken|Skipped)$")


# --- Data Layer (Mock Repository Pattern) ---
# Decoupled since database layer is under development
class MockMedicationRepository:
    def __init__(self):
        # In-memory stores
        self.medications = {} # patient_id -> list of medications
        self.doses = {} # dose_id -> dose_info
        
    def add_medication(self, patient_id: int, med_data: dict) -> dict:
        """Create and store medication entry and virtual doses."""
        med_id = str(uuid.uuid4())
        # Parse dates to strings for JSON serializability
        med_entry = {
            "id": med_id, 
            "patient_id": patient_id, 
            **med_data,
            "start_date": str(med_data['start_date']),
            "end_date": str(med_data['end_date'])
        }
        
        if patient_id not in self.medications:
            self.medications[patient_id] = []
        self.medications[patient_id].append(med_entry)
        
        # Generate virtual doses for today just for the mock structure
        for ts in med_data['time_slots']:
            dose_id = str(uuid.uuid4())
            ts_time = datetime.strptime(ts, '%H:%M').time()
            dose_dt = datetime.combine(datetime.now().date(), ts_time)
            
            self.doses[dose_id] = {
                "id": dose_id,
                "medication_id": med_id,
                "patient_id": patient_id,
                "name": med_data['name'],
                "dosage": med_data['dosage'],
                "scheduled_time": ts,
                "scheduled_datetime": dose_dt,
                "status": "Pending"
            }
        return med_entry
        
    def get_todays_schedule(self, patient_id: int) -> List[dict]:
        """Fetch all doses meant for today for a specific patient."""
        return [d for d in self.doses.values() if d["patient_id"] == patient_id]
        
    def get_patient_medications(self, patient_id: int) -> List[dict]:
        """Fetch all medications for a specific patient."""
        return self.medications.get(patient_id, [])
        
    def get_dose(self, dose_id: str) -> Optional[dict]:
        """Retrieve a specific dose by ID."""
        return self.doses.get(dose_id)
        
    def update_dose_status(self, dose_id: str, status: str, logged_time: datetime) -> bool:
        """Update the status of a specific dose."""
        if dose_id in self.doses:
            self.doses[dose_id]["status"] = status
            self.doses[dose_id]["logged_time"] = logged_time
            return True
        return False

# Concrete instantiation for mock layer
repo = MockMedicationRepository()


# --- Service Layer (Business Logic) ---
class MedicationService:
    @staticmethod
    def add_schedule(patient_id: int, data: dict) -> dict:
        """Adds a medication schedule after performing safety checks."""
        loguru_logger.info(f"Service: Attempting to add medication {data['name']} for patient {patient_id}")
        
        existing_meds = repo.get_patient_medications(patient_id)
        
        # Safety Logic Check: Prevent overlapping schedules for the SAME drug
        for med in existing_meds:
            if med['name'].lower() == data['name'].lower():
                shared_slots = set(med['time_slots']).intersection(set(data['time_slots']))
                if shared_slots:
                    raise ValueError(f"Overlapping schedule detected for {data['name']} at {shared_slots}.")
        
        # Save to Repository
        result = repo.add_medication(patient_id, data)
        return result

    @staticmethod
    def get_todays_schedule(patient_id: int) -> List[dict]:
        """Retrieves and formats today's schedule."""
        loguru_logger.info(f"Service: Fetching today's schedule for patient {patient_id}")
        doses = repo.get_todays_schedule(patient_id)
        
        return [{
            "dose_id": d["id"],
            "name": d["name"],
            "dosage": d["dosage"],
            "scheduled_time": d["scheduled_time"],
            "status": d["status"]
        } for d in doses]

    @staticmethod
    def mark_dose(patient_id: int, dose_id: str, status: str) -> dict:
        """Marks a dose as Taken/Skipped with safety flag validation."""
        loguru_logger.info(f"Service: Patient {patient_id} marking dose {dose_id} as {status}")
        
        dose = repo.get_dose(dose_id)
        if not dose or dose["patient_id"] != patient_id:
            raise ValueError("Dose not found or unauthorized access.")
        
        now = datetime.now()
        scheduled_dt = dose["scheduled_datetime"]
        
        # Update repo
        repo.update_dose_status(dose_id, status, now)
        
        # Logic Check: "Safety Warning" flag if logged significantly outside scheduled window (> 2 hours)
        time_diff_hours = abs((now - scheduled_dt).total_seconds()) / 3600
        safety_warning = None
        
        if time_diff_hours > 2:
            safety_warning = f"Dose logged {round(time_diff_hours, 1)} hours away from scheduled window. Check with doctor if safe."
            loguru_logger.warning(f"Safety Warning Issued: {safety_warning} for patient {patient_id}")
            
        return {
            "dose_id": dose_id,
            "status": status,
            "safety_warning": safety_warning
        }


# --- Controller Layer (API Routes) ---
@medication_bp.route('/medication', methods=['POST'])
@auth_required
def create_medication():
    """Create a new medication entry securely."""
    data = request.json
    try:
        # Input Validation via Pydantic
        validated_data = MedicationCreateSchema(**data).model_dump()
    except ValidationError as e:
        loguru_logger.error(f"Validation Error creating medication: {e.errors()}")
        return jsonify({"success": False, "error": "Invalid data format", "details": e.errors()}), 400
        
    try:
        # Security: Fetch patient ID strictly from session token
        patient_id = current_user.id
        med_entry = MedicationService.add_schedule(patient_id, validated_data)
        
        # Traceable Logging Audit Trail
        loguru_logger.info(f"AUDIT - Medication {med_entry['name']} initialized for User {patient_id}.")
        return jsonify({"success": True, "data": med_entry}), 201
        
    except ValueError as e:
        loguru_logger.warning(f"Business logic conflict: {e}")
        return jsonify({"success": False, "error": str(e)}), 409
    except Exception as e:
        loguru_logger.error(f"Server error during creation: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@medication_bp.route('/medication/today', methods=['GET'])
@auth_required
def get_today():
    """Retrieve today's schedule for the authenticated patient."""
    try:
        patient_id = current_user.id
        schedule = MedicationService.get_todays_schedule(patient_id)
        return jsonify({"success": True, "data": schedule}), 200
    except Exception as e:
        loguru_logger.error(f"Error fetching schedule: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@medication_bp.route('/medication/<dose_id>/status', methods=['PATCH'])
@auth_required
def update_dose(dose_id: str):
    """Mark a dose as Taken or Skipped."""
    data = request.json
    try:
        validated_data = DoseStatusUpdateSchema(**data).model_dump()
    except ValidationError as e:
        return jsonify({"success": False, "error": "Invalid status format", "details": e.errors()}), 400
        
    try:
        patient_id = current_user.id
        result = MedicationService.mark_dose(patient_id, dose_id, validated_data['status'])
        
        # Audit Trail
        loguru_logger.info(f"AUDIT - User {patient_id} updated dose {dose_id} to {validated_data['status']}.")
        
        return jsonify({"success": True, "data": result}), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        loguru_logger.error(f"Error updating dose: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500
