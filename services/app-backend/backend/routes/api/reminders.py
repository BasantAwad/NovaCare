"""
NovaCare - Medication Reminder Engine
Handles automated scanning, minimal-PHI notifications via abstract providers, and status management.
"""
from functools import wraps
from flask import Blueprint, request, jsonify
from flask_login import current_user
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Protocol
from pydantic import BaseModel
from loguru import logger as loguru_logger
import uuid
from apscheduler.schedulers.background import BackgroundScheduler

reminders_bp = Blueprint('reminders', __name__)

# --- Auth Middleware ---
def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Security Note: Ensures caller is the authenticated owner of the account
        if not current_user or not current_user.is_authenticated:
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# --- Data & DTO Layer ---
class ReminderDTO(BaseModel):
    """DTO defining exactly what the reminder object looks like for Frontend"""
    id: str
    patient_id: int
    medication_id: str
    status: str # Pending, Sent, Failed, Acknowledged
    scheduled_time: datetime
    created_at: datetime

class MockReminderRepository:
    """Mock repository decoupled from SQL layer."""
    def __init__(self):
        self.reminders: Dict[str, dict] = {} # reminder_id -> reminder dict
        
    def add_reminder(self, reminder: dict) -> dict:
        self.reminders[reminder['id']] = reminder
        return reminder
        
    def get_patient_reminders(self, patient_id: int) -> List[dict]:
        return [r for r in self.reminders.values() if r['patient_id'] == patient_id]
        
    def get_reminder(self, reminder_id: str) -> Optional[dict]:
        return self.reminders.get(reminder_id)
        
    def update_status(self, reminder_id: str, status: str):
        if reminder_id in self.reminders:
            self.reminders[reminder_id]['status'] = status
            
    def check_exists(self, patient_id: int, medication_id: str, scheduled_time: datetime) -> bool:
        """Rate Limiting / Storm prevention check: Prevent dupes for the exact same dose."""
        for r in self.reminders.values():
            if (r['patient_id'] == patient_id and 
                r['medication_id'] == medication_id and 
                r['scheduled_time'] == scheduled_time):
                return True
        return False

# Concrete instantiation
reminder_repo = MockReminderRepository()

# --- Notification Abstraction Layer ---
class NotificationProvider(Protocol):
    """Abstract decoupled Notification delivery provider."""
    def send(self, patient_id: int, title: str, generic_body: str, hidden_details: dict) -> bool: ...

class PushNotificationProvider:
    """Simulated Native Push Notification Provider"""
    def send(self, patient_id: int, title: str, generic_body: str, hidden_details: dict) -> bool:
        # Security: Minimal PHI in Notifications
        # title/body are generic. hidden_details only accessible inside the native app itself
        loguru_logger.info(f"Notification sent to PatientID: {patient_id}. Content: [{title}] {generic_body}")
        
        # Simulate successful transmission 
        return True

# Active interface injection
notification_provider: NotificationProvider = PushNotificationProvider()

# --- Engine / System Logic Layer ---

# MOCK_UPCOMING_DOSES serves as a mocked query return since actual DB isn't ready. 
MOCK_UPCOMING_DOSES = []

def _generate_mock_doses_for_testing():
    """Seed data: Generates mocked doses randomly due in the next 5 mins for testing."""
    if not MOCK_UPCOMING_DOSES:
        MOCK_UPCOMING_DOSES.append({
            "patient_id": 1, 
            "medication_id": "med-123", 
            "drug_name": "Lisinopril",
            "dosage": "10mg",
            "scheduled_datetime": datetime.now() + timedelta(minutes=5)
        })

class MedicationReminderEngine:
    @staticmethod
    def scan_and_process():
        """Scans mocked data determining doses due in the upcoming 15 minute window."""
        loguru_logger.info("Scanning upcoming medications...")
        _generate_mock_doses_for_testing()
        
        now = datetime.now()
        scan_window = now + timedelta(minutes=15)
        
        for dose in MOCK_UPCOMING_DOSES:
            sched_time = dose["scheduled_datetime"]
            
            # Check if within window and not missed
            if now <= sched_time <= scan_window:
                patient_id = dose["patient_id"]
                med_id = dose["medication_id"]
                
                # Check for duplicates to prevent Notification Storms
                if reminder_repo.check_exists(patient_id, med_id, sched_time):
                    continue 
                    
                loguru_logger.info(f"Reminder scheduled for {sched_time.strftime('%H:%M:%S')}")
                
                rem_id = str(uuid.uuid4())
                reminder = {
                    "id": rem_id,
                    "patient_id": patient_id,
                    "medication_id": med_id,
                    "status": "Pending",
                    "scheduled_time": sched_time,
                    "created_at": now
                }
                reminder_repo.add_reminder(reminder)
                
                # Execute dispatch logic explicitly separated
                MedicationReminderEngine.send_reminder(patient_id, dose, rem_id)

    @staticmethod
    def send_reminder(patient_id: int, medication_details: dict, reminder_id: str):
        """Prepares and delegates execution to generic interface provider securely."""
        # Critical Security Strategy: Never show drug details in outer notifications
        title = "NovaCare Medication Reminder"
        generic_body = "It is almost time for your designated medication. Tap to view specifics in the secure app."
        
        hidden_details = {
            "drug_name": medication_details["drug_name"],
            "dosage": medication_details["dosage"]
        }
        
        try:
            success = notification_provider.send(patient_id, title, generic_body, hidden_details)
            if success:
                reminder_repo.update_status(reminder_id, "Sent")
            else:
                loguru_logger.error(f"Failed to deliver to PatientID: {patient_id}")
                reminder_repo.update_status(reminder_id, "Failed")
        except Exception as e:
            loguru_logger.error(f"Error dispatching notification to {patient_id}: {str(e)}")
            reminder_repo.update_status(reminder_id, "Failed")


# Start Background Scheduler Daemon (APScheduler)
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(MedicationReminderEngine.scan_and_process, 'interval', minutes=1)

def init_reminders_module():
    """To be called centrally during Flask initialization."""
    scheduler.start()
    loguru_logger.info("APScheduler initialized for Medication Reminder Engine.")


# --- Controller Layer (API Routes) ---
@reminders_bp.route('/reminders', methods=['GET'])
@auth_required
def get_reminders():
    """Retrieve all reminders for the authenticated user."""
    patient_id = current_user.id
    rems = reminder_repo.get_patient_reminders(patient_id)
    return jsonify({"success": True, "data": rems}), 200

@reminders_bp.route('/reminders/<rem_id>/acknowledge', methods=['POST'])
@auth_required
def acknowledge_reminder(rem_id: str):
    """Mark a reminder as acknowledged by the user."""
    rem = reminder_repo.get_reminder(rem_id)
    if not rem or rem['patient_id'] != current_user.id:
        return jsonify({"success": False, "error": "Reminder not found or unauthorized access."}), 404
        
    reminder_repo.update_status(rem_id, "Acknowledged")
    loguru_logger.info(f"User {current_user.id} consciously acknowledged reminder {rem_id}")
    return jsonify({"success": True, "status": "Acknowledged"}), 200
