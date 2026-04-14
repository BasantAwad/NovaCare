"""
NovaCare Auth Backend — In-Memory Mock Database

Placeholder data store that mirrors the 3NF normalized schema.
Each "table" is a dict keyed by UUID string.  When a real database
is integrated, replace this module with SQLAlchemy models.

Pre-seeded with demo accounts so the frontend works immediately.
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta, timezone
from app.utils.password import hash_password

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
_now = datetime.now(timezone.utc).isoformat()
_demo_password = hash_password("NovaCare2026!")


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Reference / Lookup Tables
# ---------------------------------------------------------------------------

# ---- Countries ----
COUNTRY_EGYPT    = _uuid()
COUNTRY_US       = _uuid()
COUNTRY_UK       = _uuid()
COUNTRY_GERMANY  = _uuid()

countries: Dict[str, dict] = {
    COUNTRY_EGYPT:   {"id": COUNTRY_EGYPT,   "name": "Egypt",          "iso_code": "EG", "phone_code": "+20",  "currency_code": "EGP"},
    COUNTRY_US:      {"id": COUNTRY_US,      "name": "United States",  "iso_code": "US", "phone_code": "+1",   "currency_code": "USD"},
    COUNTRY_UK:      {"id": COUNTRY_UK,      "name": "United Kingdom", "iso_code": "GB", "phone_code": "+44",  "currency_code": "GBP"},
    COUNTRY_GERMANY: {"id": COUNTRY_GERMANY, "name": "Germany",        "iso_code": "DE", "phone_code": "+49",  "currency_code": "EUR"},
}

# ---- Verification Statuses ----
VS_PENDING  = _uuid()
VS_VERIFIED = _uuid()
VS_REJECTED = _uuid()
VS_ADDINFO  = _uuid()

verification_statuses: Dict[str, dict] = {
    VS_PENDING:  {"id": VS_PENDING,  "status_name": "pending",         "display_label": "Pending",          "description": "Awaiting admin review"},
    VS_VERIFIED: {"id": VS_VERIFIED, "status_name": "verified",        "display_label": "Verified",         "description": "Identity verified"},
    VS_REJECTED: {"id": VS_REJECTED, "status_name": "rejected",        "display_label": "Rejected",         "description": "Verification rejected"},
    VS_ADDINFO:  {"id": VS_ADDINFO,  "status_name": "additional_info", "display_label": "Additional Info",  "description": "More information needed"},
}

# ---- Approval Statuses ----
AS_PENDING  = _uuid()
AS_APPROVED = _uuid()
AS_DENIED   = _uuid()

approval_statuses: Dict[str, dict] = {
    AS_PENDING:  {"id": AS_PENDING,  "status_name": "pending",  "display_label": "Pending"},
    AS_APPROVED: {"id": AS_APPROVED, "status_name": "approved", "display_label": "Approved"},
    AS_DENIED:   {"id": AS_DENIED,   "status_name": "denied",   "display_label": "Denied"},
}

# ---- ID Types (Government) ----
IDT_PASSPORT   = _uuid()
IDT_DRIVERS    = _uuid()
IDT_NATIONAL   = _uuid()

id_types: Dict[str, dict] = {
    IDT_PASSPORT: {"id": IDT_PASSPORT, "name": "Passport",         "country_id": None,          "description": "International passport"},
    IDT_DRIVERS:  {"id": IDT_DRIVERS,  "name": "Driver's License", "country_id": None,          "description": "Government-issued driving license"},
    IDT_NATIONAL: {"id": IDT_NATIONAL, "name": "National ID",      "country_id": COUNTRY_EGYPT, "description": "National identity card"},
}

# ---- Relationship Types ----
RT_PARENT  = _uuid()
RT_SPOUSE  = _uuid()
RT_SIBLING = _uuid()
RT_PROF    = _uuid()

relationship_types: Dict[str, dict] = {
    RT_PARENT:  {"id": RT_PARENT,  "relationship": "parent",       "description": "Parent of the patient"},
    RT_SPOUSE:  {"id": RT_SPOUSE,  "relationship": "spouse",       "description": "Spouse or partner"},
    RT_SIBLING: {"id": RT_SIBLING, "relationship": "sibling",      "description": "Brother or sister"},
    RT_PROF:    {"id": RT_PROF,    "relationship": "professional", "description": "Professional caregiver"},
}

# ---- Specializations ----
SPEC_CARDIO  = _uuid()
SPEC_NEURO   = _uuid()
SPEC_GENERAL = _uuid()
SPEC_GERI    = _uuid()
SPEC_ORTHO   = _uuid()

specializations: Dict[str, dict] = {
    SPEC_CARDIO:  {"id": SPEC_CARDIO,  "name": "Cardiology",        "description": "Heart and cardiovascular system", "code": "CARD"},
    SPEC_NEURO:   {"id": SPEC_NEURO,   "name": "Neurology",         "description": "Nervous system disorders",        "code": "NEUR"},
    SPEC_GENERAL: {"id": SPEC_GENERAL, "name": "General Practice",  "description": "Primary care physician",          "code": "GP"},
    SPEC_GERI:    {"id": SPEC_GERI,    "name": "Geriatrics",        "description": "Care for elderly patients",        "code": "GER"},
    SPEC_ORTHO:   {"id": SPEC_ORTHO,   "name": "Orthopedics",       "description": "Musculoskeletal system",           "code": "ORTH"},
}

# ---- Access Levels ----
AL_FULL     = _uuid()
AL_LIMITED  = _uuid()
AL_READONLY = _uuid()

access_levels: Dict[str, dict] = {
    AL_FULL:     {"id": AL_FULL,     "level_name": "full",     "display_label": "Full Access",     "description": "Complete patient access",    "can_view": True, "can_edit": True,  "can_prescribe": True,  "can_message": True},
    AL_LIMITED:  {"id": AL_LIMITED,  "level_name": "limited",  "display_label": "Limited Access",  "description": "View and message only",      "can_view": True, "can_edit": False, "can_prescribe": False, "can_message": True},
    AL_READONLY: {"id": AL_READONLY, "level_name": "readonly", "display_label": "Read Only",       "description": "View only",                  "can_view": True, "can_edit": False, "can_prescribe": False, "can_message": False},
}

# ---- Health Conditions ----
HC_DIABETES = _uuid()
HC_HYPERT   = _uuid()
HC_ASTHMA   = _uuid()
HC_ARTHRITIS = _uuid()

health_conditions: Dict[str, dict] = {
    HC_DIABETES:  {"id": HC_DIABETES,  "name": "Type 2 Diabetes",  "icd10_code": "E11",   "description": "Chronic metabolic disorder",        "severity_levels": "mild,moderate,severe"},
    HC_HYPERT:    {"id": HC_HYPERT,    "name": "Hypertension",     "icd10_code": "I10",   "description": "High blood pressure",               "severity_levels": "stage1,stage2,crisis"},
    HC_ASTHMA:    {"id": HC_ASTHMA,    "name": "Asthma",           "icd10_code": "J45",   "description": "Chronic respiratory condition",      "severity_levels": "intermittent,mild,moderate,severe"},
    HC_ARTHRITIS: {"id": HC_ARTHRITIS, "name": "Osteoarthritis",   "icd10_code": "M15",   "description": "Degenerative joint disease",         "severity_levels": "mild,moderate,severe"},
}

# ---- Medications ----
MED_METFORMIN = _uuid()
MED_LISINOP   = _uuid()
MED_ASPIRIN   = _uuid()

medications: Dict[str, dict] = {
    MED_METFORMIN: {"id": MED_METFORMIN, "name": "Metformin",    "brand_names": "Glucophage",  "atc_code": "A10BA02", "dosage_form": "tablet", "strength": "500mg",  "description": "Antidiabetic medication", "side_effects": "nausea,diarrhea"},
    MED_LISINOP:   {"id": MED_LISINOP,   "name": "Lisinopril",  "brand_names": "Zestril",     "atc_code": "C09AA03", "dosage_form": "tablet", "strength": "10mg",   "description": "ACE inhibitor",          "side_effects": "cough,dizziness"},
    MED_ASPIRIN:   {"id": MED_ASPIRIN,   "name": "Aspirin",      "brand_names": "Bayer",       "atc_code": "B01AC06", "dosage_form": "tablet", "strength": "81mg",   "description": "Blood thinner",          "side_effects": "bleeding,stomach upset"},
}

# ---- Allergies ----
ALG_PENICILLIN = _uuid()
ALG_PEANUTS    = _uuid()
ALG_LATEX      = _uuid()

allergies: Dict[str, dict] = {
    ALG_PENICILLIN: {"id": ALG_PENICILLIN, "name": "Penicillin", "category": "medication", "description": "Antibiotic allergy",      "severity_levels": "mild,moderate,severe,anaphylaxis"},
    ALG_PEANUTS:    {"id": ALG_PEANUTS,    "name": "Peanuts",    "category": "food",       "description": "Peanut and tree nut allergy", "severity_levels": "mild,moderate,severe,anaphylaxis"},
    ALG_LATEX:      {"id": ALG_LATEX,      "name": "Latex",      "category": "material",   "description": "Natural rubber latex allergy", "severity_levels": "mild,moderate,severe"},
}

# ---- Message Types ----
MT_GENERAL  = _uuid()
MT_URGENT   = _uuid()
MT_MEDICAL  = _uuid()

message_types: Dict[str, dict] = {
    MT_GENERAL: {"id": MT_GENERAL, "type_name": "general",  "display_label": "General",  "priority_level": 3},
    MT_URGENT:  {"id": MT_URGENT,  "type_name": "urgent",   "display_label": "Urgent",   "priority_level": 5},
    MT_MEDICAL: {"id": MT_MEDICAL, "type_name": "medical",  "display_label": "Medical",  "priority_level": 4},
}

# ---- Notification Types ----
NT_MEDICATION = _uuid()
NT_APPOINTMENT = _uuid()
NT_ALERT      = _uuid()
NT_SYSTEM     = _uuid()

notification_types: Dict[str, dict] = {
    NT_MEDICATION:  {"id": NT_MEDICATION,  "type_name": "medication_reminder", "display_label": "Medication Reminder", "priority": 4, "is_urgent": False},
    NT_APPOINTMENT: {"id": NT_APPOINTMENT, "type_name": "appointment",         "display_label": "Appointment",         "priority": 3, "is_urgent": False},
    NT_ALERT:       {"id": NT_ALERT,       "type_name": "health_alert",        "display_label": "Health Alert",        "priority": 5, "is_urgent": True},
    NT_SYSTEM:      {"id": NT_SYSTEM,      "type_name": "system",              "display_label": "System",              "priority": 2, "is_urgent": False},
}

# ---- Appointment Types & Statuses ----
APT_CHECKUP  = _uuid()
APT_FOLLOWUP = _uuid()
APT_EMERGENCY = _uuid()

appointment_types: Dict[str, dict] = {
    APT_CHECKUP:   {"id": APT_CHECKUP,   "type_name": "checkup",    "display_label": "Check-up"},
    APT_FOLLOWUP:  {"id": APT_FOLLOWUP,  "type_name": "follow_up",  "display_label": "Follow-up"},
    APT_EMERGENCY: {"id": APT_EMERGENCY, "type_name": "emergency",  "display_label": "Emergency"},
}

APTS_SCHEDULED = _uuid()
APTS_COMPLETED = _uuid()
APTS_CANCELLED = _uuid()

appointment_statuses: Dict[str, dict] = {
    APTS_SCHEDULED: {"id": APTS_SCHEDULED, "status_name": "scheduled", "display_label": "Scheduled"},
    APTS_COMPLETED: {"id": APTS_COMPLETED, "status_name": "completed", "display_label": "Completed"},
    APTS_CANCELLED: {"id": APTS_CANCELLED, "status_name": "cancelled", "display_label": "Cancelled"},
}

# ---- Medical Note Types ----
MNT_DIAGNOSIS = _uuid()
MNT_RECOMMEND = _uuid()
MNT_PROGRESS  = _uuid()

medical_note_types: Dict[str, dict] = {
    MNT_DIAGNOSIS: {"id": MNT_DIAGNOSIS, "note_type": "diagnosis",      "display_label": "Diagnosis"},
    MNT_RECOMMEND: {"id": MNT_RECOMMEND, "note_type": "recommendation", "display_label": "Recommendation"},
    MNT_PROGRESS:  {"id": MNT_PROGRESS,  "note_type": "progress",       "display_label": "Progress Note"},
}

# ---- Action Types (Audit) ----
ACT_LOGIN   = _uuid()
ACT_SIGNUP  = _uuid()
ACT_VIEW    = _uuid()
ACT_UPDATE  = _uuid()

action_types: Dict[str, dict] = {
    ACT_LOGIN:  {"id": ACT_LOGIN,  "action_name": "user_login",  "description": "User logged in",           "requires_logging": True},
    ACT_SIGNUP: {"id": ACT_SIGNUP, "action_name": "user_signup", "description": "New user registered",      "requires_logging": True},
    ACT_VIEW:   {"id": ACT_VIEW,   "action_name": "data_view",   "description": "User viewed a resource",   "requires_logging": False},
    ACT_UPDATE: {"id": ACT_UPDATE, "action_name": "data_update", "description": "User updated a resource",  "requires_logging": True},
}

# ---- Action Statuses (Audit) ----
ACTS_SUCCESS = _uuid()
ACTS_FAILURE = _uuid()
ACTS_DENIED  = _uuid()

action_statuses: Dict[str, dict] = {
    ACTS_SUCCESS: {"id": ACTS_SUCCESS, "status_name": "success", "display_label": "Success"},
    ACTS_FAILURE: {"id": ACTS_FAILURE, "status_name": "failure", "display_label": "Failure"},
    ACTS_DENIED:  {"id": ACTS_DENIED,  "status_name": "denied",  "display_label": "Denied"},
}

# ---- Clinic Organizations ----
CLINIC_MAIN = _uuid()

clinic_organizations: Dict[str, dict] = {
    CLINIC_MAIN: {
        "id": CLINIC_MAIN,
        "name": "NovaCare Medical Center",
        "address_id": None,
        "phone_number": "+20-100-000-0001",
        "email": "info@novacaremedical.com",
        "website": "https://novacaremedical.com",
        "license_number": "CLN-2024-001",
        "is_verified": True,
        "created_at": _now,
        "updated_at": _now,
    },
}

# ---------------------------------------------------------------------------
# Core Tables (Mutable – populated with demo data)
# ---------------------------------------------------------------------------

# ---- Users ----
USER_SARAH  = _uuid()   # Rover / Patient
USER_JOHN   = _uuid()   # Caregiver
USER_SMITH  = _uuid()   # Doctor

users: Dict[str, dict] = {
    USER_SARAH: {
        "id": USER_SARAH,
        "email": "sarah@novacare.demo",
        "hashed_password": _demo_password,
        "google_id": None,
        "first_name": "Sarah",
        "last_name": "Johnson",
        "profile_picture_url": None,
        "is_email_verified": True,
        "email_verified_at": _now,
        "created_at": _now,
        "updated_at": _now,
        "is_active": True,
        "last_login_at": _now,
    },
    USER_JOHN: {
        "id": USER_JOHN,
        "email": "john.guardian@novacare.demo",
        "hashed_password": _demo_password,
        "google_id": None,
        "first_name": "John",
        "last_name": "Guardian",
        "profile_picture_url": None,
        "is_email_verified": True,
        "email_verified_at": _now,
        "created_at": _now,
        "updated_at": _now,
        "is_active": True,
        "last_login_at": _now,
    },
    USER_SMITH: {
        "id": USER_SMITH,
        "email": "dr.smith@novacare.demo",
        "hashed_password": _demo_password,
        "google_id": None,
        "first_name": "David",
        "last_name": "Smith",
        "profile_picture_url": None,
        "is_email_verified": True,
        "email_verified_at": _now,
        "created_at": _now,
        "updated_at": _now,
        "is_active": True,
        "last_login_at": _now,
    },
}

# ---- User Roles ----
ROLE_SARAH_R  = _uuid()
ROLE_JOHN_C   = _uuid()
ROLE_SMITH_D  = _uuid()

user_roles: Dict[str, dict] = {
    ROLE_SARAH_R: {"id": ROLE_SARAH_R, "user_id": USER_SARAH, "role": "rover",     "assigned_at": _now, "assigned_by_id": None, "is_active": True, "deactivated_at": None},
    ROLE_JOHN_C:  {"id": ROLE_JOHN_C,  "user_id": USER_JOHN,  "role": "caregiver", "assigned_at": _now, "assigned_by_id": None, "is_active": True, "deactivated_at": None},
    ROLE_SMITH_D: {"id": ROLE_SMITH_D, "user_id": USER_SMITH, "role": "doctor",    "assigned_at": _now, "assigned_by_id": None, "is_active": True, "deactivated_at": None},
}

# ---- Sessions (populated on login) ----
sessions: Dict[str, dict] = {}

# ---- Device Info ----
device_info: Dict[str, dict] = {}

# ---- Password Reset Tokens ----
password_reset_tokens: Dict[str, dict] = {}

# ---- Email Verification Tokens ----
email_verification_tokens: Dict[str, dict] = {}

# ---- Addresses ----
ADDR_SARAH = _uuid()
ADDR_JOHN  = _uuid()

addresses: Dict[str, dict] = {
    ADDR_SARAH: {"id": ADDR_SARAH, "street_address": "123 Nile St",   "city": "Cairo",      "state_province": "Cairo",      "postal_code": "11511", "country_id": COUNTRY_EGYPT, "created_at": _now, "updated_at": _now},
    ADDR_JOHN:  {"id": ADDR_JOHN,  "street_address": "456 Sphinx Ave", "city": "Alexandria", "state_province": "Alexandria", "postal_code": "21500", "country_id": COUNTRY_EGYPT, "created_at": _now, "updated_at": _now},
}

# ---- Rovers (Patient Profiles) ----
ROVER_SARAH = _uuid()

rovers: Dict[str, dict] = {
    ROVER_SARAH: {
        "id": ROVER_SARAH,
        "user_id": USER_SARAH,
        "date_of_birth": "1958-03-15",
        "gender": "female",
        "phone_number": "+20-100-555-0001",
        "address_id": ADDR_SARAH,
        "blood_type": "A+",
        "needs_caregiver": True,
        "primary_caregiver_id": None,  # set after caregiver record
        "caregiver_approved_at": _now,
        "created_at": _now,
        "updated_at": _now,
    },
}

# ---- Caregivers ----
CG_JOHN = _uuid()

caregivers: Dict[str, dict] = {
    CG_JOHN: {
        "id": CG_JOHN,
        "user_id": USER_JOHN,
        "date_of_birth": "1985-07-22",
        "phone_number": "+20-100-555-0002",
        "address_id": ADDR_JOHN,
        "government_id_type_id": IDT_NATIONAL,
        "government_id_number": "29507221234567",
        "id_expiry_date": "2030-12-31",
        "verification_status_id": VS_VERIFIED,
        "verified_at": _now,
        "verification_notes": "Identity verified by admin",
        "verified_by_admin_id": None,
        "rejection_reason": None,
        "certification_url": None,
        "created_at": _now,
        "updated_at": _now,
    },
}

# Link Sarah's primary caregiver to John
rovers[ROVER_SARAH]["primary_caregiver_id"] = CG_JOHN

# ---- Identity Documents ----
identity_documents: Dict[str, dict] = {
    _uuid(): {
        "id": _uuid(),
        "caregiver_id": CG_JOHN,
        "id_type_id": IDT_NATIONAL,
        "document_url": "https://storage.novacare.demo/docs/john-national-id.jpg",
        "document_hash": "sha256:placeholder",
        "upload_date": _now,
        "is_primary": True,
        "ocr_extracted_text": None,
        "is_verified": True,
        "created_at": _now,
    },
}

# ---- Caregiver-Rover Assignments ----
CRA_JOHN_SARAH = _uuid()

caregiver_rover_assignments: Dict[str, dict] = {
    CRA_JOHN_SARAH: {
        "id": CRA_JOHN_SARAH,
        "caregiver_id": CG_JOHN,
        "rover_id": ROVER_SARAH,
        "relationship_type_id": RT_PROF,
        "approval_status_id": AS_APPROVED,
        "requested_at": _now,
        "approved_at": _now,
        "approved_by_id": USER_SARAH,
        "denied_at": None,
        "denial_reason": None,
        "is_active": True,
        "created_at": _now,
    },
}

# ---- Doctors ----
DOC_SMITH = _uuid()

doctors: Dict[str, dict] = {
    DOC_SMITH: {
        "id": DOC_SMITH,
        "user_id": USER_SMITH,
        "specialization_id": SPEC_GERI,
        "medical_license_num": "MED-EG-2024-00042",
        "license_country_id": COUNTRY_EGYPT,
        "license_expiry_date": "2028-06-30",
        "board_reg_number": "BRD-00042",
        "clinic_organization_id": CLINIC_MAIN,
        "verification_status_id": VS_VERIFIED,
        "verified_at": _now,
        "verification_notes": "License verified",
        "verified_by_admin_id": None,
        "rejection_reason": None,
        "professional_id_url": "https://storage.novacare.demo/docs/dr-smith-license.pdf",
        "created_at": _now,
        "updated_at": _now,
    },
}

# ---- Medical License Documents ----
medical_license_documents: Dict[str, dict] = {}

# ---- Doctor-Patient Access ----
DPA_SMITH_SARAH = _uuid()

doctor_patient_access: Dict[str, dict] = {
    DPA_SMITH_SARAH: {
        "id": DPA_SMITH_SARAH,
        "doctor_id": DOC_SMITH,
        "rover_id": ROVER_SARAH,
        "access_level_id": AL_FULL,
        "approval_status_id": AS_APPROVED,
        "requested_at": _now,
        "approved_at": _now,
        "approved_by_id": USER_SARAH,
        "revoked_at": None,
        "revoked_by_id": None,
        "revocation_reason": None,
        "is_active": True,
        "created_at": _now,
    },
}

# ---- Rover Health Conditions ----
rover_health_conditions: Dict[str, dict] = {
    _uuid(): {"id": _uuid(), "rover_id": ROVER_SARAH, "condition_id": HC_HYPERT,   "severity": "stage1",  "diagnosed_date": "2020-01-15", "notes": "Controlled with medication", "is_active": True, "created_at": _now, "updated_at": _now},
    _uuid(): {"id": _uuid(), "rover_id": ROVER_SARAH, "condition_id": HC_ARTHRITIS, "severity": "moderate", "diagnosed_date": "2019-05-10", "notes": "Affects knees and hips",    "is_active": True, "created_at": _now, "updated_at": _now},
}

# ---- Rover Medications ----
rover_medications: Dict[str, dict] = {
    _uuid(): {"id": _uuid(), "rover_id": ROVER_SARAH, "medication_id": MED_LISINOP,  "dosage": "10mg", "frequency": "once daily",  "route": "oral", "start_date": "2020-02-01", "end_date": None, "prescriber_id": DOC_SMITH, "is_active": True, "notes": "Take in the morning",        "created_at": _now, "updated_at": _now},
    _uuid(): {"id": _uuid(), "rover_id": ROVER_SARAH, "medication_id": MED_ASPIRIN,   "dosage": "81mg", "frequency": "once daily",  "route": "oral", "start_date": "2020-02-01", "end_date": None, "prescriber_id": DOC_SMITH, "is_active": True, "notes": "Take with food",             "created_at": _now, "updated_at": _now},
}

# ---- Rover Allergies ----
rover_allergies: Dict[str, dict] = {
    _uuid(): {"id": _uuid(), "rover_id": ROVER_SARAH, "allergy_id": ALG_PENICILLIN, "severity": "severe", "reaction": "Rash and difficulty breathing", "is_active": True, "created_at": _now},
}

# ---- Vital Signs (sample time-series) ----
vital_signs: Dict[str, dict] = {}
_base_time = datetime.now(timezone.utc) - timedelta(hours=24)
for i in range(24):
    _vid = _uuid()
    vital_signs[_vid] = {
        "id": _vid,
        "rover_id": ROVER_SARAH,
        "heart_rate": 68 + (i % 5),
        "spo2": 97 + (i % 2),
        "temperature": 36.5 + (i % 3) * 0.1,
        "systolic_bp": 125 + (i % 4),
        "diastolic_bp": 80 + (i % 3),
        "respiratory_rate": 16 + (i % 2),
        "blood_glucose": 105 + (i % 6),
        "weight": 65.0,
        "measurement_device_id": None,
        "measured_at": (_base_time + timedelta(hours=i)).isoformat(),
        "created_at": _now,
        "is_synced": True,
    }

# ---- Measurement Devices ----
measurement_devices: Dict[str, dict] = {}

# ---- Emergency Contacts ----
emergency_contacts: Dict[str, dict] = {
    _uuid(): {
        "id": _uuid(),
        "rover_id": ROVER_SARAH,
        "name": "John Guardian",
        "relationship": "Professional Caregiver",
        "phone_number": "+20-100-555-0002",
        "alt_phone": None,
        "email": "john.guardian@novacare.demo",
        "is_primary": True,
        "is_active": True,
        "created_at": _now,
        "updated_at": _now,
    },
}

# ---- Secure Messages ----
secure_messages: Dict[str, dict] = {}

# ---- AI Interactions ----
ai_interactions: Dict[str, dict] = {}

# ---- Emotion Tracking ----
emotion_tracking: Dict[str, dict] = {}

# ---- Notifications ----
notifications: Dict[str, dict] = {}

# ---- Appointments ----
_appt1 = _uuid()
appointments: Dict[str, dict] = {
    _appt1: {
        "id": _appt1,
        "rover_id": ROVER_SARAH,
        "doctor_id": DOC_SMITH,
        "caregiver_id": None,
        "appointment_type_id": APT_CHECKUP,
        "scheduled_at": (datetime.now(timezone.utc) + timedelta(days=3)).isoformat(),
        "duration_minutes": 30,
        "status_id": APTS_SCHEDULED,
        "cancellation_reason": None,
        "notes": "Routine check-up",
        "created_at": _now,
        "updated_at": _now,
    },
}

# ---- Medical Notes ----
medical_notes: Dict[str, dict] = {
    _uuid(): {
        "id": _uuid(),
        "doctor_id": DOC_SMITH,
        "rover_id": ROVER_SARAH,
        "note_type_id": MNT_PROGRESS,
        "note_content": "Patient's blood pressure has been stable over the last month. Continue current medication regimen.",
        "is_encrypted": False,
        "created_at": _now,
        "updated_at": _now,
        "updated_by_id": None,
    },
}

# ---- Audit Logs ----
audit_logs: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helper functions for querying the mock database
# ---------------------------------------------------------------------------

def find_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email address."""
    for user in users.values():
        if user["email"].lower() == email.lower():
            return user
    return None


def get_user_roles(user_id: str) -> List[dict]:
    """Get all active roles for a user."""
    return [r for r in user_roles.values() if r["user_id"] == user_id and r["is_active"]]


def get_user_role_names(user_id: str) -> List[str]:
    """Get role name strings for a user."""
    return [r["role"] for r in get_user_roles(user_id)]


def get_rover_profile(user_id: str) -> Optional[dict]:
    """Get rover profile by user_id."""
    for r in rovers.values():
        if r["user_id"] == user_id:
            return r
    return None


def get_caregiver_profile(user_id: str) -> Optional[dict]:
    """Get caregiver profile by user_id."""
    for c in caregivers.values():
        if c["user_id"] == user_id:
            return c
    return None


def get_doctor_profile(user_id: str) -> Optional[dict]:
    """Get doctor profile by user_id."""
    for d in doctors.values():
        if d["user_id"] == user_id:
            return d
    return None
