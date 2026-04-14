"""
NovaCare Auth Backend — Auth Routes

Handles signup (rover/caregiver/doctor), login, token refresh,
email verification, password reset, and session management.
"""
import uuid
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g

from app import mock_db as db
from app.utils.password import hash_password, verify_password
from app.utils.tokens import (
    generate_access_token,
    generate_refresh_token,
    decode_token,
    generate_verification_token,
)
from app.middleware.auth_middleware import require_auth

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


# ---------------------------------------------------------------------------
# POST /api/auth/signup/rover
# ---------------------------------------------------------------------------
@auth_bp.route("/signup/rover", methods=["POST"])
def signup_rover():
    """Register a new Rover (Patient) account."""
    data = request.get_json(silent=True) or {}

    # --- validate required fields ---
    required = ["email", "first_name", "last_name", "date_of_birth", "gender"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"status": "error", "error": f"Missing fields: {', '.join(missing)}"}), 400

    if not data.get("google_id") and not data.get("password"):
        return jsonify({"status": "error", "error": "Password is required if not using Google Auth"}), 400

    if db.find_user_by_email(data["email"]):
        return jsonify({"status": "error", "error": "Email already registered"}), 409

    # --- create user ---
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    hashed_pw = hash_password(data["password"]) if data.get("password") else ""
    user = {
        "id": user_id,
        "email": data["email"].lower().strip(),
        "hashed_password": hashed_pw,
        "google_id": data.get("google_id"),
        "first_name": data["first_name"],
        "last_name": data["last_name"],
        "profile_picture_url": None,
        "is_email_verified": False,
        "email_verified_at": None,
        "created_at": now,
        "updated_at": now,
        "is_active": True,
        "last_login_at": None,
    }
    db.users[user_id] = user

    # --- assign rover role ---
    role_id = str(uuid.uuid4())
    db.user_roles[role_id] = {
        "id": role_id,
        "user_id": user_id,
        "role": "rover",
        "assigned_at": now,
        "assigned_by_id": None,
        "is_active": True,
        "deactivated_at": None,
    }

    # --- create address (optional) ---
    address_id = None
    if data.get("address"):
        address_id = str(uuid.uuid4())
        addr = data["address"]
        db.addresses[address_id] = {
            "id": address_id,
            "street_address": addr.get("street", ""),
            "city": addr.get("city", ""),
            "state_province": addr.get("state", ""),
            "postal_code": addr.get("postal_code", ""),
            "country_id": addr.get("country_id"),
            "created_at": now,
            "updated_at": now,
        }

    # --- create rover profile ---
    rover_id = str(uuid.uuid4())
    db.rovers[rover_id] = {
        "id": rover_id,
        "user_id": user_id,
        "date_of_birth": data["date_of_birth"],
        "gender": data["gender"],
        "phone_number": data.get("phone_number", ""),
        "address_id": address_id,
        "blood_type": data.get("blood_type", ""),
        "needs_caregiver": data.get("needs_caregiver", False),
        "primary_caregiver_id": None,
        "caregiver_approved_at": None,
        "created_at": now,
        "updated_at": now,
    }

    # --- create health conditions ---
    for cond in data.get("health_conditions", []):
        hc_id = str(uuid.uuid4())
        db.rover_health_conditions[hc_id] = {
            "id": hc_id,
            "rover_id": rover_id,
            "condition_id": cond.get("condition_id", ""),
            "severity": cond.get("severity", "mild"),
            "diagnosed_date": cond.get("diagnosed_date"),
            "notes": cond.get("notes", ""),
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        }

    # --- create allergies ---
    for alg in data.get("allergies", []):
        alg_id = str(uuid.uuid4())
        db.rover_allergies[alg_id] = {
            "id": alg_id,
            "rover_id": rover_id,
            "allergy_id": alg.get("allergy_id", ""),
            "severity": alg.get("severity", "mild"),
            "reaction": alg.get("reaction", ""),
            "is_active": True,
            "created_at": now,
        }

    # --- create emergency contact ---
    if data.get("emergency_contact"):
        ec = data["emergency_contact"]
        ec_id = str(uuid.uuid4())
        db.emergency_contacts[ec_id] = {
            "id": ec_id,
            "rover_id": rover_id,
            "name": ec.get("name", ""),
            "relationship": ec.get("relationship", ""),
            "phone_number": ec.get("phone_number", ""),
            "alt_phone": ec.get("alt_phone"),
            "email": ec.get("email"),
            "is_primary": True,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        }

    # --- generate tokens ---
    roles = db.get_user_role_names(user_id)
    access_token = generate_access_token(user_id, roles)
    refresh_token = generate_refresh_token(user_id)

    # --- create session ---
    session_id = str(uuid.uuid4())
    db.sessions[session_id] = {
        "id": session_id,
        "user_id": user_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "device_info_id": None,
        "ip_address": request.remote_addr,
        "user_agent": request.user_agent.string,
        "expires_at": None,
        "refresh_expires_at": None,
        "created_at": now,
        "is_active": True,
        "revoked_at": None,
    }

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }), 201


# ---------------------------------------------------------------------------
# POST /api/auth/signup/caregiver
# ---------------------------------------------------------------------------
@auth_bp.route("/signup/caregiver", methods=["POST"])
def signup_caregiver():
    """Register a new Caregiver account."""
    data = request.get_json(silent=True) or {}

    required = ["email", "first_name", "last_name", "date_of_birth", "phone_number"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"status": "error", "error": f"Missing fields: {', '.join(missing)}"}), 400

    if not data.get("google_id") and not data.get("password"):
        return jsonify({"status": "error", "error": "Password is required if not using Google Auth"}), 400

    if db.find_user_by_email(data["email"]):
        return jsonify({"status": "error", "error": "Email already registered"}), 409

    now = datetime.now(timezone.utc).isoformat()
    user_id = str(uuid.uuid4())

    # --- create user ---
    hashed_pw = hash_password(data["password"]) if data.get("password") else ""
    user = {
        "id": user_id,
        "email": data["email"].lower().strip(),
        "hashed_password": hashed_pw,
        "google_id": data.get("google_id"),
        "first_name": data["first_name"],
        "last_name": data["last_name"],
        "profile_picture_url": None,
        "is_email_verified": False,
        "email_verified_at": None,
        "created_at": now,
        "updated_at": now,
        "is_active": True,
        "last_login_at": None,
    }
    db.users[user_id] = user

    # --- assign caregiver role ---
    role_id = str(uuid.uuid4())
    db.user_roles[role_id] = {
        "id": role_id,
        "user_id": user_id,
        "role": "caregiver",
        "assigned_at": now,
        "assigned_by_id": None,
        "is_active": True,
        "deactivated_at": None,
    }

    # --- create address ---
    address_id = None
    if data.get("address"):
        address_id = str(uuid.uuid4())
        addr = data["address"]
        db.addresses[address_id] = {
            "id": address_id,
            "street_address": addr.get("street", ""),
            "city": addr.get("city", ""),
            "state_province": addr.get("state", ""),
            "postal_code": addr.get("postal_code", ""),
            "country_id": addr.get("country_id"),
            "created_at": now,
            "updated_at": now,
        }

    # --- create caregiver profile (starts as "pending" verification) ---
    cg_id = str(uuid.uuid4())
    db.caregivers[cg_id] = {
        "id": cg_id,
        "user_id": user_id,
        "date_of_birth": data["date_of_birth"],
        "phone_number": data["phone_number"],
        "address_id": address_id,
        "government_id_type_id": data.get("government_id_type_id"),
        "government_id_number": data.get("government_id_number", ""),
        "id_expiry_date": data.get("id_expiry_date"),
        "verification_status_id": db.VS_PENDING,
        "verified_at": None,
        "verification_notes": None,
        "verified_by_admin_id": None,
        "rejection_reason": None,
        "certification_url": data.get("certification_url"),
        "created_at": now,
        "updated_at": now,
    }

    # --- handle ID document upload (placeholder URL) ---
    if data.get("document_url"):
        doc_id = str(uuid.uuid4())
        db.identity_documents[doc_id] = {
            "id": doc_id,
            "caregiver_id": cg_id,
            "id_type_id": data.get("government_id_type_id"),
            "document_url": data["document_url"],
            "document_hash": "sha256:placeholder",
            "upload_date": now,
            "is_primary": True,
            "ocr_extracted_text": None,
            "is_verified": False,
            "created_at": now,
        }

    # --- if rover_email provided, create assignment request ---
    if data.get("rover_email"):
        rover_user = db.find_user_by_email(data["rover_email"])
        if rover_user:
            rover_profile = db.get_rover_profile(rover_user["id"])
            if rover_profile:
                assignment_id = str(uuid.uuid4())
                db.caregiver_rover_assignments[assignment_id] = {
                    "id": assignment_id,
                    "caregiver_id": cg_id,
                    "rover_id": rover_profile["id"],
                    "relationship_type_id": data.get("relationship_type_id", db.RT_PROF),
                    "approval_status_id": db.AS_PENDING,
                    "requested_at": now,
                    "approved_at": None,
                    "approved_by_id": None,
                    "denied_at": None,
                    "denial_reason": None,
                    "is_active": False,
                    "created_at": now,
                }

    # --- generate tokens ---
    roles = db.get_user_role_names(user_id)
    access_token = generate_access_token(user_id, roles)
    refresh_token = generate_refresh_token(user_id)

    session_id = str(uuid.uuid4())
    db.sessions[session_id] = {
        "id": session_id, "user_id": user_id, "access_token": access_token,
        "refresh_token": refresh_token, "device_info_id": None,
        "ip_address": request.remote_addr, "user_agent": request.user_agent.string,
        "expires_at": None, "refresh_expires_at": None, "created_at": now,
        "is_active": True, "revoked_at": None,
    }

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "verification_status": "pending",
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }), 201


# ---------------------------------------------------------------------------
# POST /api/auth/signup/doctor
# ---------------------------------------------------------------------------
@auth_bp.route("/signup/doctor", methods=["POST"])
def signup_doctor():
    """Register a new Doctor account."""
    data = request.get_json(silent=True) or {}

    required = ["email", "first_name", "last_name", "specialization_id", "medical_license_num"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"status": "error", "error": f"Missing fields: {', '.join(missing)}"}), 400

    if not data.get("google_id") and not data.get("password"):
        return jsonify({"status": "error", "error": "Password is required if not using Google Auth"}), 400

    if db.find_user_by_email(data["email"]):
        return jsonify({"status": "error", "error": "Email already registered"}), 409

    now = datetime.now(timezone.utc).isoformat()
    user_id = str(uuid.uuid4())

    hashed_pw = hash_password(data["password"]) if data.get("password") else ""
    user = {
        "id": user_id,
        "email": data["email"].lower().strip(),
        "hashed_password": hashed_pw,
        "google_id": data.get("google_id"),
        "first_name": data["first_name"],
        "last_name": data["last_name"],
        "profile_picture_url": None,
        "is_email_verified": False,
        "email_verified_at": None,
        "created_at": now,
        "updated_at": now,
        "is_active": True,
        "last_login_at": None,
    }
    db.users[user_id] = user

    role_id = str(uuid.uuid4())
    db.user_roles[role_id] = {
        "id": role_id, "user_id": user_id, "role": "doctor",
        "assigned_at": now, "assigned_by_id": None, "is_active": True, "deactivated_at": None,
    }

    doc_id = str(uuid.uuid4())
    db.doctors[doc_id] = {
        "id": doc_id,
        "user_id": user_id,
        "specialization_id": data["specialization_id"],
        "medical_license_num": data["medical_license_num"],
        "license_country_id": data.get("license_country_id"),
        "license_expiry_date": data.get("license_expiry_date"),
        "board_reg_number": data.get("board_reg_number", ""),
        "clinic_organization_id": data.get("clinic_organization_id"),
        "verification_status_id": db.VS_PENDING,
        "verified_at": None,
        "verification_notes": None,
        "verified_by_admin_id": None,
        "rejection_reason": None,
        "professional_id_url": data.get("professional_id_url"),
        "created_at": now,
        "updated_at": now,
    }

    roles = db.get_user_role_names(user_id)
    access_token = generate_access_token(user_id, roles)
    refresh_token = generate_refresh_token(user_id)

    session_id = str(uuid.uuid4())
    db.sessions[session_id] = {
        "id": session_id, "user_id": user_id, "access_token": access_token,
        "refresh_token": refresh_token, "device_info_id": None,
        "ip_address": request.remote_addr, "user_agent": request.user_agent.string,
        "expires_at": None, "refresh_expires_at": None, "created_at": now,
        "is_active": True, "revoked_at": None,
    }

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "verification_status": "pending",
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }), 201


# ---------------------------------------------------------------------------
# POST /api/auth/login
# ---------------------------------------------------------------------------
@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate user with email + password.  Auto-detects roles."""
    data = request.get_json(silent=True) or {}

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"status": "error", "error": "Email and password are required"}), 400

    user = db.find_user_by_email(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return jsonify({"status": "error", "error": "Invalid email or password"}), 401

    if not user["is_active"]:
        return jsonify({"status": "error", "error": "Account is deactivated"}), 403

    # Update last login
    user["last_login_at"] = datetime.now(timezone.utc).isoformat()

    roles = db.get_user_role_names(user["id"])
    access_token = generate_access_token(user["id"], roles)
    refresh_token = generate_refresh_token(user["id"])

    now = datetime.now(timezone.utc).isoformat()
    session_id = str(uuid.uuid4())
    db.sessions[session_id] = {
        "id": session_id, "user_id": user["id"], "access_token": access_token,
        "refresh_token": refresh_token, "device_info_id": None,
        "ip_address": request.remote_addr, "user_agent": request.user_agent.string,
        "expires_at": None, "refresh_expires_at": None, "created_at": now,
        "is_active": True, "revoked_at": None,
    }

    # Build profile summary based on roles
    profile = {}
    if "rover" in roles:
        profile["rover"] = db.get_rover_profile(user["id"])
    if "caregiver" in roles:
        profile["caregiver"] = db.get_caregiver_profile(user["id"])
    if "doctor" in roles:
        profile["doctor"] = db.get_doctor_profile(user["id"])

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "profile": profile,
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }), 200


@auth_bp.route("/login/google", methods=["POST"])
def login_google():
    """Authenticate user with Google ID."""
    data = request.get_json(silent=True) or {}
    google_id = data.get("google_id")
    email = data.get("email", "").strip().lower()

    if not google_id or not email:
        return jsonify({"status": "error", "error": "Google ID and email are required"}), 400

    user = db.find_user_by_email(email)
    if not user or user.get("google_id") != google_id:
        return jsonify({"status": "error", "error": "User not found or Google account not linked. Please sign up first."}), 401

    if not user["is_active"]:
        return jsonify({"status": "error", "error": "Account is deactivated"}), 403

    user["last_login_at"] = datetime.now(timezone.utc).isoformat()

    roles = db.get_user_role_names(user["id"])
    access_token = generate_access_token(user["id"], roles)
    refresh_token = generate_refresh_token(user["id"])

    now = datetime.now(timezone.utc).isoformat()
    session_id = str(uuid.uuid4())
    db.sessions[session_id] = {
        "id": session_id, "user_id": user["id"], "access_token": access_token,
        "refresh_token": refresh_token, "device_info_id": None,
        "ip_address": request.remote_addr, "user_agent": request.user_agent.string,
        "expires_at": None, "refresh_expires_at": None, "created_at": now,
        "is_active": True, "revoked_at": None,
    }

    profile = {}
    if "rover" in roles:
        profile["rover"] = db.get_rover_profile(user["id"])
    if "caregiver" in roles:
        profile["caregiver"] = db.get_caregiver_profile(user["id"])
    if "doctor" in roles:
        profile["doctor"] = db.get_doctor_profile(user["id"])

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "profile": profile,
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }), 200

# ---------------------------------------------------------------------------
# POST /api/auth/refresh
# ---------------------------------------------------------------------------
@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    """Generate a new access token from a valid refresh token."""
    data = request.get_json(silent=True) or {}
    token = data.get("refresh_token", "")

    if not token:
        return jsonify({"status": "error", "error": "Refresh token required"}), 400

    payload = decode_token(token)
    if not payload or payload.get("type") != "refresh":
        return jsonify({"status": "error", "error": "Invalid or expired refresh token"}), 401

    user_id = payload["sub"]
    user = db.users.get(user_id)
    if not user or not user["is_active"]:
        return jsonify({"status": "error", "error": "User not found or deactivated"}), 401

    roles = db.get_user_role_names(user_id)
    new_access = generate_access_token(user_id, roles)

    return jsonify({
        "status": "success",
        "data": {"access_token": new_access},
    }), 200


# ---------------------------------------------------------------------------
# GET /api/auth/me
# ---------------------------------------------------------------------------
@auth_bp.route("/me", methods=["GET"])
@require_auth
def me():
    """Return the current authenticated user's info and profiles."""
    user_id = g.current_user_id
    user = db.users.get(user_id)

    if not user:
        return jsonify({"status": "error", "error": "User not found"}), 404

    roles = db.get_user_role_names(user_id)

    profile = {}
    if "rover" in roles:
        rp = db.get_rover_profile(user_id)
        if rp:
            profile["rover"] = rp
    if "caregiver" in roles:
        cp = db.get_caregiver_profile(user_id)
        if cp:
            profile["caregiver"] = cp
            # resolve verification status name
            vs = db.verification_statuses.get(cp.get("verification_status_id", ""), {})
            profile["caregiver_verification"] = vs.get("status_name", "unknown")
    if "doctor" in roles:
        dp = db.get_doctor_profile(user_id)
        if dp:
            profile["doctor"] = dp
            vs = db.verification_statuses.get(dp.get("verification_status_id", ""), {})
            profile["doctor_verification"] = vs.get("status_name", "unknown")
            # resolve specialization
            spec = db.specializations.get(dp.get("specialization_id", ""), {})
            profile["specialization"] = spec.get("name", "")

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "profile": profile,
        },
    }), 200


# ---------------------------------------------------------------------------
# POST /api/auth/logout
# ---------------------------------------------------------------------------
@auth_bp.route("/logout", methods=["POST"])
@require_auth
def logout():
    """Revoke the current session."""
    user_id = g.current_user_id
    now = datetime.now(timezone.utc).isoformat()

    for session in db.sessions.values():
        if session["user_id"] == user_id and session["is_active"]:
            session["is_active"] = False
            session["revoked_at"] = now

    return jsonify({"status": "success", "data": {"message": "Logged out successfully"}}), 200


# ---------------------------------------------------------------------------
# POST /api/auth/forgot-password
# ---------------------------------------------------------------------------
@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Request a password reset token (simulated - always returns success)."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    # Always return success to prevent email enumeration
    if email:
        user = db.find_user_by_email(email)
        if user:
            token = generate_verification_token()
            token_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            db.password_reset_tokens[token_id] = {
                "id": token_id,
                "user_id": user["id"],
                "token_hash": token,
                "created_at": now,
                "expires_at": now,  # In production, add 48 hours
                "used_at": None,
                "ip_address": request.remote_addr,
            }

    return jsonify({
        "status": "success",
        "data": {"message": "If an account with that email exists, a reset link has been sent."},
    }), 200


# ---------------------------------------------------------------------------
# POST /api/auth/reset-password
# ---------------------------------------------------------------------------
@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    """Reset password with a valid token (simplified for mock)."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")
    new_password = data.get("new_password", "")

    if not token or not new_password:
        return jsonify({"status": "error", "error": "Token and new password required"}), 400

    # Find token in mock DB
    reset_entry = None
    for entry in db.password_reset_tokens.values():
        if entry["token_hash"] == token and not entry["used_at"]:
            reset_entry = entry
            break

    if not reset_entry:
        return jsonify({"status": "error", "error": "Invalid or expired reset token"}), 400

    user = db.users.get(reset_entry["user_id"])
    if not user:
        return jsonify({"status": "error", "error": "User not found"}), 404

    user["hashed_password"] = hash_password(new_password)
    user["updated_at"] = datetime.now(timezone.utc).isoformat()
    reset_entry["used_at"] = datetime.now(timezone.utc).isoformat()

    return jsonify({"status": "success", "data": {"message": "Password reset successfully"}}), 200


# ---------------------------------------------------------------------------
# POST /api/auth/verify-email
# ---------------------------------------------------------------------------
@auth_bp.route("/verify-email", methods=["POST"])
def verify_email():
    """Verify email with token (simplified for mock — auto-verifies)."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")
    # For demo purposes, accept any non-empty token and verify the first unverified user
    # In production, match against email_verification_tokens table

    if not token:
        return jsonify({"status": "error", "error": "Verification token required"}), 400

    # Check verification tokens
    ver_entry = None
    for entry in db.email_verification_tokens.values():
        if entry["token_hash"] == token and not entry["verified_at"]:
            ver_entry = entry
            break

    if not ver_entry:
        return jsonify({"status": "error", "error": "Invalid or expired verification token"}), 400

    user = db.users.get(ver_entry["user_id"])
    if user:
        now = datetime.now(timezone.utc).isoformat()
        user["is_email_verified"] = True
        user["email_verified_at"] = now
        ver_entry["verified_at"] = now

    return jsonify({"status": "success", "data": {"message": "Email verified successfully"}}), 200


# ---------------------------------------------------------------------------
# GET /api/auth/reference-data
# ---------------------------------------------------------------------------
@auth_bp.route("/reference-data", methods=["GET"])
def reference_data():
    """Return lookup tables needed by signup forms (countries, specializations, etc.)."""
    return jsonify({
        "status": "success",
        "data": {
            "countries": list(db.countries.values()),
            "specializations": list(db.specializations.values()),
            "health_conditions": list(db.health_conditions.values()),
            "medications": list(db.medications.values()),
            "allergies": list(db.allergies.values()),
            "id_types": list(db.id_types.values()),
            "relationship_types": list(db.relationship_types.values()),
            "clinic_organizations": list(db.clinic_organizations.values()),
        },
    }), 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sanitize_user(user: dict) -> dict:
    """Remove sensitive fields from user dict before sending to client."""
    return {k: v for k, v in user.items() if k not in ("hashed_password",)}
