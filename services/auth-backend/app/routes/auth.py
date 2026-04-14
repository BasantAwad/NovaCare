"""
NovaCare Auth Backend — Auth Routes
Refactored to use live MySQL database (db_controller).
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
        return jsonify({"status": "error", "error": "Database connection drop"}), 500

    if not user or not verify_password(password, user["hashed_password"]):
        return jsonify({"status": "error", "error": "Invalid email or password"}), 401

    if not user.get("is_active"):
        return jsonify({"status": "error", "error": "Account is deactivated"}), 403

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    try:
        db.execute_query("UPDATE users SET last_login_at = %s WHERE id = %s", (now, user["id"]))
    except Exception as e:
        logger.error(f"Failed updating login timestamp: {e}")

    try:
        user_roles_db = db.fetch_all("SELECT role FROM user_roles WHERE user_id = %s", (user["id"],))
        roles = [r["role"] for r in user_roles_db] if user_roles_db else []
    except Exception as e:
        logger.error(f"DB Error fetching roles: {e}")
        roles = []

    access_token = generate_access_token(user["id"], roles)
    refresh_token = generate_refresh_token(user["id"])
    session_id = str(uuid.uuid4())

    try:
        db.execute_query(
            "INSERT INTO sessions (id, user_id, access_token, refresh_token, ip_address, user_agent, created_at, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (session_id, user["id"], access_token, refresh_token, request.remote_addr, request.user_agent.string, now, True)
        )
    except Exception as e:
        logger.error(f"DB Error creating session: {e}")
        return jsonify({"status": "error", "error": "Could not create session"}), 500

    return jsonify({
        "status": "success",
        "data": {
            "user": _sanitize_user(user),
            "roles": roles,
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    }), 200


@auth_bp.route("/me", methods=["GET"])
@require_auth
def me():
    """Return the current authenticated user's info and profile."""
    user_id = g.current_user_id
    try:
        user = db.fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
        if not user:
            return jsonify({"status": "error", "error": "User not found"}), 404
            
        roles_data = db.fetch_all("SELECT role FROM user_roles WHERE user_id = %s", (user_id,))
        roles = [r["role"] for r in roles_data] if roles_data else []
        
        profile = {}
        if "rover" in roles:
            rp = db.fetch_one("SELECT * FROM rovers WHERE user_id = %s", (user_id,))
            if rp: profile["rover"] = rp
        if "caregiver" in roles:
            cp = db.fetch_one("SELECT * FROM caregivers WHERE user_id = %s", (user_id,))
            if cp: profile["caregiver"] = cp
        if "doctor" in roles:
            dp = db.fetch_one("SELECT * FROM doctors WHERE user_id = %s", (user_id,))
            if dp: profile["doctor"] = dp
            
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


@auth_bp.route("/signup/rover", methods=["POST"])
def signup_rover():
    """Register a new Rover (Patient) account."""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    
    try:
        existing = db.fetch_one("SELECT id FROM users WHERE email = %s", (email,))
        if existing:
            return jsonify({"status": "error", "error": "Email already registered"}), 409
            
        user_id = str(uuid.uuid4())
        hashed_pw = hash_password(data.get("password", ""))
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        db.execute_query(
            "INSERT INTO users (id, email, hashed_password, first_name, last_name, is_active, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (user_id, email, hashed_pw, data.get("first_name"), data.get("last_name"), True, now)
        )
        
        rover_id = str(uuid.uuid4())
        db.execute_query(
            "INSERT INTO rovers (id, user_id, date_of_birth, gender, created_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (rover_id, user_id, data.get("date_of_birth"), data.get("gender"), now)
        )
        
        db.execute_query(
            "INSERT INTO user_roles (id, user_id, role, assigned_at) VALUES (%s, %s, %s, %s)",
            (str(uuid.uuid4()), user_id, "rover", now)
        )
        
    except Exception as e:
        logger.error(f"DB Error during signup: {e}")
        return jsonify({"status": "error", "error": "Database connection drop"}), 500
        
    return jsonify({"status": "success", "message": "Rover account created. Please log in."}), 201


@auth_bp.route("/reference-data", methods=["GET"])
def reference_data():
    """Return lookup tables safely via database without querying empty 0-row tables."""
    try:
        countries = db.fetch_all("SELECT * FROM countries")
        specializations = db.fetch_all("SELECT * FROM specializations")
        catalog = db.fetch_all("SELECT * FROM medication_catalog")
        return jsonify({
            "status": "success", 
            "data": {
                "countries": countries, 
                "specializations": specializations, 
                "medication_catalog": catalog
            }
        })
    except Exception as e:
        logger.error(f"DB Error referencing data: {e}")
        return jsonify({"status": "error", "error": "Reference fetch error due to drop"}), 500


def _sanitize_user(user: dict) -> dict:
    return {k: v for k, v in user.items() if k not in ("hashed_password",)}
