"""
NovaCare Auth Backend — Auth Routes
Refactored to use live MySQL database (db_controller).
All queries match the verified live schema on 192.168.1.164.

Verified table schemas:
  users:              id, email, hashed_password, google_id, first_name, last_name, profile_picture_url, is_email_verified, email_verified_at, is_active, created_at, updated_at
  user_roles:         id, user_id, role, is_active
  rovers:             id, user_id, date_of_birth, gender, address_id, primary_caregiver_id
  caregivers:         id, user_id, phone_number, address_id, government_id_number, verification_status_id
  doctors:            id, user_id, medical_license_num, specialization_id, verification_status_id
  sessions:           id, user_id, access_token, refresh_token, device_info_id, ip_address, expires_at, is_active
  emergency_contacts: id, rover_id, name, relationship, phone_number, is_primary
"""
import uuid
import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g

from app.db_controller import db
from app.utils.password import hash_password, verify_password
from app.utils.tokens import (
    generate_access_token,
    generate_refresh_token,
    decode_token,
    generate_verification_token,
)
from app.middleware.auth_middleware import require_auth

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def _sanitize_user(user: dict) -> dict:
    """Remove sensitive fields and convert non-serializable types."""
    safe = {}
    for k, v in user.items():
        if k in ("hashed_password",):
            continue
        if isinstance(v, datetime):
            safe[k] = v.isoformat()
        else:
            safe[k] = v
    return safe


def _get_roles(user_id: str) -> list:
    rows = db.fetch_all("SELECT role FROM user_roles WHERE user_id = %s", (user_id,))
    return [r["role"] for r in rows] if rows else []


def _get_profile(user_id: str, roles: list) -> dict:
    profile = {}
    try:
        if "rover" in roles:
            rp = db.fetch_one("SELECT * FROM rovers WHERE user_id = %s", (user_id,))
            if rp:
                profile["rover"] = rp
        if "caregiver" in roles:
            cp = db.fetch_one("SELECT * FROM caregivers WHERE user_id = %s", (user_id,))
            if cp:
                profile["caregiver"] = cp
        if "doctor" in roles:
            dp = db.fetch_one("SELECT * FROM doctors WHERE user_id = %s", (user_id,))
            if dp:
                profile["doctor"] = dp
    except Exception as e:
        logger.error(f"Error building profile: {e}")
    return profile


def _create_session(user_id, access_token, refresh_token):
    """sessions: id, user_id, access_token, refresh_token, device_info_id, ip_address, expires_at, is_active"""
    session_id = str(uuid.uuid4())
    try:
        db.execute_query(
            "INSERT INTO sessions (id, user_id, access_token, refresh_token, ip_address, is_active) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (session_id, user_id, access_token, refresh_token, request.remote_addr, True)
        )
    except Exception as e:
        logger.error(f"Session creation error: {e}")


# ---------------------------------------------------------------------------
# POST /api/auth/login
# ---------------------------------------------------------------------------
@auth_bp.route("/login", methods=["POST"])
def login():
    """Authenticate user with email + password."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"status": "error", "error": "Email and password are required"}), 400

    try:
        user = db.fetch_one("SELECT * FROM users WHERE email = %s", (email,))
    except Exception as e:
        logger.error(f"DB Error fetching user: {e}")
        return jsonify({"status": "error", "error": "Database connection error"}), 500

    if not user or not verify_password(password, user["hashed_password"]):
        return jsonify({"status": "error", "error": "Invalid email or password"}), 401

    if not user.get("is_active"):
        return jsonify({"status": "error", "error": "Account is deactivated"}), 403

    # Update updated_at as login marker (users table has no last_login_at)
    try:
        db.execute_query("UPDATE users SET updated_at = %s WHERE id = %s", (_now(), user["id"]))
    except Exception as e:
        logger.error(f"Failed updating timestamp: {e}")

    roles = _get_roles(user["id"])
    access_token = generate_access_token(user["id"], roles)
    refresh_token = generate_refresh_token(user["id"])
    _create_session(user["id"], access_token, refresh_token)

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "profile": _get_profile(user["id"], roles),
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }), 200


# ---------------------------------------------------------------------------
# POST /api/auth/login/google
# ---------------------------------------------------------------------------
@auth_bp.route("/login/google", methods=["POST"])
def login_google():
    """Authenticate user with Google ID."""
    data = request.get_json(silent=True) or {}
    google_id = data.get("google_id")
    email = data.get("email", "").strip().lower()

    if not google_id or not email:
        return jsonify({"status": "error", "error": "Google ID and email are required"}), 400

    try:
        user = db.fetch_one("SELECT * FROM users WHERE email = %s", (email,))
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return jsonify({"status": "error", "error": "Database connection error"}), 500

    if not user or user.get("google_id") != google_id:
        return jsonify({"status": "error", "error": "User not found or Google account not linked. Please sign up first."}), 401

    if not user.get("is_active"):
        return jsonify({"status": "error", "error": "Account is deactivated"}), 403

    roles = _get_roles(user["id"])
    access_token = generate_access_token(user["id"], roles)
    refresh_token = generate_refresh_token(user["id"])
    _create_session(user["id"], access_token, refresh_token)

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "profile": _get_profile(user["id"], roles),
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }), 200


# ---------------------------------------------------------------------------
# GET /api/auth/me
# ---------------------------------------------------------------------------
@auth_bp.route("/me", methods=["GET"])
@require_auth
def me():
    """Return the current authenticated user's info and profiles."""
    user_id = g.current_user_id
    try:
        user = db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
        if not user:
            return jsonify({"status": "error", "error": "User not found"}), 404

        roles = _get_roles(user_id)
        profile = _get_profile(user_id, roles)

    except Exception as e:
        logger.error(f"DB Error: {e}")
        return jsonify({"status": "error", "error": "Database retrieval error"}), 500

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "profile": profile,
        },
    }), 200


# ---------------------------------------------------------------------------
# POST /api/auth/signup/rover
# ---------------------------------------------------------------------------
@auth_bp.route("/signup/rover", methods=["POST"])
def signup_rover():
    """Register a new Rover (Patient) account.
    users:  id, email, hashed_password, google_id, first_name, last_name, is_email_verified, is_active, created_at, updated_at
    rovers: id, user_id, date_of_birth, gender, address_id, primary_caregiver_id
    """
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    required = ["email", "first_name", "last_name", "date_of_birth", "gender"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"status": "error", "error": f"Missing fields: {', '.join(missing)}"}), 400

    if not data.get("google_id") and not data.get("password"):
        return jsonify({"status": "error", "error": "Password is required if not using Google Auth"}), 400

    try:
        existing = db.fetch_one("SELECT id FROM users WHERE email = %s", (email,))
        if existing:
            return jsonify({"status": "error", "error": "Email already registered"}), 409

        user_id = str(uuid.uuid4())
        now = _now()
        hashed_pw = hash_password(data["password"]) if data.get("password") else ""

        # Insert user
        db.execute_query(
            "INSERT INTO users (id, email, hashed_password, google_id, first_name, last_name, "
            "is_email_verified, is_active, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (user_id, email, hashed_pw, data.get("google_id"), data["first_name"],
             data["last_name"], False, True, now, now)
        )

        # Assign role
        db.execute_query(
            "INSERT INTO user_roles (id, user_id, role, is_active) VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), user_id, "rover", True)
        )

        # Create rover profile
        rover_id = str(uuid.uuid4())
        db.execute_query(
            "INSERT INTO rovers (id, user_id, date_of_birth, gender) VALUES (%s, %s, %s, %s)",
            (rover_id, user_id, data["date_of_birth"], data["gender"])
        )

        # Create emergency contact if provided
        if data.get("emergency_contact"):
            ec = data["emergency_contact"]
            if ec.get("name") and ec.get("phone_number"):
                db.execute_query(
                    "INSERT INTO emergency_contacts (id, rover_id, name, relationship, phone_number, is_primary) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (str(uuid.uuid4()), rover_id, ec["name"], ec.get("relationship", ""),
                     ec["phone_number"], True)
                )

        # Health Conditions (Primary + Additional)
        # First, process the custom/primary condition if provided
        primary_name = data.get("primary_condition_name")
        if primary_name:
            # check if it exists in health_conditions
            hc = db.fetch_one("SELECT id FROM health_conditions WHERE name = %s", (primary_name,))
            if hc:
                hc_id = hc["id"]
            else:
                hc_id = str(uuid.uuid4())
                db.execute_query("INSERT INTO health_conditions (id, name) VALUES (%s, %s)", (hc_id, primary_name))
            
            try:
                db.execute_query(
                    "INSERT INTO rover_health_conditions (id, rover_id, condition_id, severity, notes) VALUES (%s, %s, %s, %s, %s)",
                    (str(uuid.uuid4()), rover_id, hc_id, 'severe', 'Primary Condition')
                )
            except Exception as e:
                logger.error(f"Failed to insert primary condition: {e}")

        # Process additional health conditions from checkboxes
        health_conds = data.get("health_conditions", [])
        if isinstance(health_conds, list):
            for hc in health_conds:
                c_id = hc.get("condition_id")
                # Skip if we already inserted it as primary (to avoid duplicates if they selected it in both)
                if primary_name and c_id == locals().get('hc_id'):
                    continue
                if c_id:
                    try:
                        db.execute_query(
                            "INSERT INTO rover_health_conditions (id, rover_id, condition_id, severity) VALUES (%s, %s, %s, %s)",
                            (str(uuid.uuid4()), rover_id, c_id, hc.get("severity", "mild"))
                        )
                    except Exception as e:
                        logger.error(f"Failed to insert condition {c_id}: {e}")

        # Allergies
        allergies = data.get("allergies", [])
        if isinstance(allergies, list):
            for alg in allergies:
                a_id = alg.get("allergy_id")
                a_name = alg.get("allergy_name")
                
                if a_name and not a_id:
                    # Fallback text input: find or create allergy
                    existing_alg = db.fetch_one("SELECT id FROM allergies WHERE name = %s", (a_name,))
                    if existing_alg:
                        a_id = existing_alg["id"]
                    else:
                        a_id = str(uuid.uuid4())
                        db.execute_query("INSERT INTO allergies (id, name, category) VALUES (%s, %s, 'medication')", (a_id, a_name))

                if a_id:
                    try:
                        db.execute_query(
                            "INSERT INTO rover_allergies (id, rover_id, allergy_id, severity, is_active) VALUES (%s, %s, %s, %s, %s)",
                            (str(uuid.uuid4()), rover_id, a_id, alg.get("severity", "mild"), True)
                        )
                    except Exception as e:
                        logger.error(f"Failed to insert allergy: {e}")

        # Generate tokens so user is logged in immediately after signup
        roles = ["rover"]
        access_token = generate_access_token(user_id, roles)
        refresh_token = generate_refresh_token(user_id)
        _create_session(user_id, access_token, refresh_token)

        user = db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))

    except Exception as e:
        logger.error(f"DB Error during rover signup: {e}")
        return jsonify({"status": "error", "error": f"Signup failed: {e}"}), 500

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
    """Register a new Caregiver account.
    caregivers: id, user_id, phone_number, address_id, government_id_number, verification_status_id
    """
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    required = ["email", "first_name", "last_name", "phone_number"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"status": "error", "error": f"Missing fields: {', '.join(missing)}"}), 400

    if not data.get("google_id") and not data.get("password"):
        return jsonify({"status": "error", "error": "Password is required if not using Google Auth"}), 400

    try:
        existing = db.fetch_one("SELECT id FROM users WHERE email = %s", (email,))
        if existing:
            return jsonify({"status": "error", "error": "Email already registered"}), 409

        user_id = str(uuid.uuid4())
        now = _now()
        hashed_pw = hash_password(data["password"]) if data.get("password") else ""

        db.execute_query(
            "INSERT INTO users (id, email, hashed_password, google_id, first_name, last_name, "
            "is_email_verified, is_active, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (user_id, email, hashed_pw, data.get("google_id"), data["first_name"],
             data["last_name"], False, True, now, now)
        )

        db.execute_query(
            "INSERT INTO user_roles (id, user_id, role, is_active) VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), user_id, "caregiver", True)
        )

        # Look up pending verification status
        vs_pending = db.fetch_one("SELECT id FROM verification_statuses WHERE status_name = %s", ("pending",))
        vs_id = vs_pending["id"] if vs_pending else None

        cg_id = str(uuid.uuid4())
        db.execute_query(
            "INSERT INTO caregivers (id, user_id, phone_number, government_id_number, verification_status_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (cg_id, user_id, data["phone_number"], data.get("government_id_number", ""), vs_id)
        )

        roles = ["caregiver"]
        access_token = generate_access_token(user_id, roles)
        refresh_token = generate_refresh_token(user_id)
        _create_session(user_id, access_token, refresh_token)

        user = db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))

    except Exception as e:
        logger.error(f"DB Error during caregiver signup: {e}")
        return jsonify({"status": "error", "error": f"Signup failed: {e}"}), 500

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
    """Register a new Doctor account.
    doctors: id, user_id, medical_license_num, specialization_id, verification_status_id
    """
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    required = ["email", "first_name", "last_name", "specialization_id", "medical_license_num"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"status": "error", "error": f"Missing fields: {', '.join(missing)}"}), 400

    if not data.get("google_id") and not data.get("password"):
        return jsonify({"status": "error", "error": "Password is required if not using Google Auth"}), 400

    try:
        existing = db.fetch_one("SELECT id FROM users WHERE email = %s", (email,))
        if existing:
            return jsonify({"status": "error", "error": "Email already registered"}), 409

        user_id = str(uuid.uuid4())
        now = _now()
        hashed_pw = hash_password(data["password"]) if data.get("password") else ""

        db.execute_query(
            "INSERT INTO users (id, email, hashed_password, google_id, first_name, last_name, "
            "is_email_verified, is_active, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (user_id, email, hashed_pw, data.get("google_id"), data["first_name"],
             data["last_name"], False, True, now, now)
        )

        db.execute_query(
            "INSERT INTO user_roles (id, user_id, role, is_active) VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), user_id, "doctor", True)
        )

        vs_pending = db.fetch_one("SELECT id FROM verification_statuses WHERE status_name = %s", ("pending",))
        vs_id = vs_pending["id"] if vs_pending else None

        doc_id = str(uuid.uuid4())
        db.execute_query(
            "INSERT INTO doctors (id, user_id, medical_license_num, specialization_id, verification_status_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (doc_id, user_id, data["medical_license_num"], data["specialization_id"], vs_id)
        )

        roles = ["doctor"]
        access_token = generate_access_token(user_id, roles)
        refresh_token = generate_refresh_token(user_id)
        _create_session(user_id, access_token, refresh_token)

        user = db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))

    except Exception as e:
        logger.error(f"DB Error during doctor signup: {e}")
        return jsonify({"status": "error", "error": f"Signup failed: {e}"}), 500

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
    try:
        user = db.fetch_one("SELECT * FROM users WHERE id = %s AND is_active = 1", (user_id,))
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return jsonify({"status": "error", "error": "Database error"}), 500

    if not user:
        return jsonify({"status": "error", "error": "User not found or deactivated"}), 401

    roles = _get_roles(user_id)
    new_access = generate_access_token(user_id, roles)

    return jsonify({
        "status": "success",
        "data": {"access_token": new_access},
    }), 200


# ---------------------------------------------------------------------------
# POST /api/auth/logout
# ---------------------------------------------------------------------------
@auth_bp.route("/logout", methods=["POST"])
@require_auth
def logout():
    """Revoke the current session."""
    user_id = g.current_user_id
    try:
        db.execute_query(
            "UPDATE sessions SET is_active = 0 WHERE user_id = %s AND is_active = 1",
            (user_id,)
        )
    except Exception as e:
        logger.error(f"Error revoking sessions: {e}")

    return jsonify({"status": "success", "data": {"message": "Logged out successfully"}}), 200


# ---------------------------------------------------------------------------
# POST /api/auth/forgot-password
# ---------------------------------------------------------------------------
@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Request a password reset token (always returns success to prevent email enumeration)."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    if email:
        try:
            user = db.fetch_one("SELECT id FROM users WHERE email = %s", (email,))
            if user:
                token = generate_verification_token()
                logger.info(f"Password reset token generated for {email}: {token}")
        except Exception as e:
            logger.error(f"Error in forgot-password: {e}")

    return jsonify({
        "status": "success",
        "data": {"message": "If an account with that email exists, a reset link has been sent."},
    }), 200


# ---------------------------------------------------------------------------
# POST /api/auth/reset-password
# ---------------------------------------------------------------------------
@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    """Reset password with a valid token."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")
    new_password = data.get("new_password", "")

    if not token or not new_password:
        return jsonify({"status": "error", "error": "Token and new password required"}), 400

    return jsonify({"status": "error", "error": "Invalid or expired reset token"}), 400


# ---------------------------------------------------------------------------
# POST /api/auth/verify-email
# ---------------------------------------------------------------------------
@auth_bp.route("/verify-email", methods=["POST"])
def verify_email():
    """Verify email with token."""
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")

    if not token:
        return jsonify({"status": "error", "error": "Verification token required"}), 400

    return jsonify({"status": "error", "error": "Invalid or expired verification token"}), 400


# ---------------------------------------------------------------------------
# GET /api/auth/reference-data
# ---------------------------------------------------------------------------
@auth_bp.route("/reference-data", methods=["GET"])
def reference_data():
    """Return lookup tables needed by signup forms."""
    try:
        countries = db.fetch_all("SELECT * FROM countries") or []
        specializations = db.fetch_all("SELECT * FROM specializations") or []
        relationship_types = db.fetch_all("SELECT * FROM relationship_types") or []
        medication_catalog = db.fetch_all("SELECT * FROM medication_catalog") or []

        try:
            health_conditions = db.fetch_all("SELECT * FROM health_conditions") or []
        except Exception:
            health_conditions = []

        try:
            allergies = db.fetch_all("SELECT * FROM allergies") or []
        except Exception:
            allergies = []

        return jsonify({
            "status": "success",
            "data": {
                "countries": countries,
                "specializations": specializations,
                "relationship_types": relationship_types,
                "medication_catalog": medication_catalog,
                "health_conditions": health_conditions,
                "allergies": allergies,
                "medications": [],
                "id_types": [],
                "clinic_organizations": [],
            }
        }), 200
    except Exception as e:
        logger.error(f"DB Error fetching reference data: {e}")
        # Return empty data gracefully so the UI doesn't break
        return jsonify({
            "status": "success",
            "data": {
                "countries": [], "specializations": [], "relationship_types": [],
                "medication_catalog": [], "health_conditions": [], "medications": [],
                "allergies": [], "id_types": [], "clinic_organizations": [],
            }
        }), 200
