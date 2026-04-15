"""
NovaCare Auth Backend — Dashboard Routes
Refactored to use live MySQL database (db_controller).
"""
import logging
from flask import Blueprint, jsonify, g
from app.db_controller import db
from app.middleware.auth_middleware import require_auth

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")

@dashboard_bp.route("/vitals", methods=["GET"])
@require_auth
def get_dashboard_vitals():
    user_id = g.current_user_id
    try:
        rover = db.fetch_one("SELECT id FROM rovers WHERE user_id = %s", (user_id,))
        if not rover:
            return jsonify({"status": "error", "error": "Not authorized as Rover"}), 403
            
        vitals = db.fetch_all(
            "SELECT * FROM vital_signs WHERE rover_id = %s ORDER BY measured_at DESC LIMIT 50", 
            (rover["id"],)
        )
        return jsonify({"status": "success", "data": vitals}), 200
    except Exception as e:
        logger.error(f"DB Error fetching vitals: {e}")
        return jsonify({"status": "error", "error": "Connection drop while fetching vitals"}), 500

@dashboard_bp.route("/profile", methods=["GET"])
@require_auth
def get_dashboard_profile():
    user_id = g.current_user_id
    try:
        user = db.fetch_one("SELECT id, email, first_name, last_name FROM users WHERE id = %s", (user_id,))
        roles = db.fetch_all("SELECT role FROM user_roles WHERE user_id = %s", (user_id,))
        role_list = [r["role"] for r in roles] if roles else []
        
        profile = {"user": user, "role": role_list[0] if role_list else "unknown"}
        
        if "rover" in role_list:
            prof = db.fetch_one("""
                SELECT r.*, u.first_name, u.last_name, u.email 
                FROM rovers r
                JOIN users u ON u.id = r.user_id
                WHERE r.user_id = %s
            """, (user_id,))
            profile["details"] = prof
            
        elif "doctor" in role_list:
            prof = db.fetch_one("""
                SELECT d.*, u.first_name, u.last_name, u.email 
                FROM doctors d
                JOIN users u ON u.id = d.user_id
                WHERE d.user_id = %s
            """, (user_id,))
            profile["details"] = prof

        elif "caregiver" in role_list:
            prof = db.fetch_one("""
                SELECT c.*, u.first_name, u.last_name, u.email 
                FROM caregivers c
                JOIN users u ON u.id = c.user_id
                WHERE c.user_id = %s
            """, (user_id,))
            profile["details"] = prof
            
        return jsonify({"status": "success", "data": profile}), 200
        
    except Exception as e:
        logger.error(f"DB error fetching profile: {e}")
        return jsonify({"status": "error", "error": "Connection drop while fetching profile"}), 500

@dashboard_bp.route("/notes", methods=["GET"])
@require_auth
def get_medical_notes():
    user_id = g.current_user_id
    try:
        roles = db.fetch_all("SELECT role FROM user_roles WHERE user_id = %s", (user_id,))
        role_list = [r["role"] for r in roles] if roles else []

        if "doctor" in role_list:
            doctor = db.fetch_one("SELECT id FROM doctors WHERE user_id = %s", (user_id,))
            if not doctor:
                return jsonify({"status": "error", "error": "Doctor profile not found"}), 404
            notes = db.fetch_all("SELECT * FROM medical_notes WHERE doctor_id = %s ORDER BY created_at DESC", (doctor["id"],))
        
        elif "rover" in role_list:
            rover = db.fetch_one("SELECT id FROM rovers WHERE user_id = %s", (user_id,))
            if not rover:
                return jsonify({"status": "error", "error": "Rover profile not found"}), 404
            notes = db.fetch_all("SELECT * FROM medical_notes WHERE rover_id = %s ORDER BY created_at DESC", (rover["id"],))
            
        else:
            return jsonify({"status": "error", "error": "Unauthorized"}), 403

        return jsonify({"status": "success", "data": notes}), 200
        
    except Exception as e:
        logger.error(f"DB Error fetching notes: {e}")
        return jsonify({"status": "error", "error": "Connection drop while fetching notes"}), 500
