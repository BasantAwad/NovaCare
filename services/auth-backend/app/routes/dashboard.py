"""
NovaCare Auth Backend — Dashboard Routes
Refactored to use live MySQL database (db_controller).
"""
import logging
from datetime import datetime, timezone, timedelta
from flask import Blueprint, jsonify, g, request
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


# ---------------------------------------------------------------------------
# Helper: Resolve rover_id for the authenticated user (rover, caregiver, or doctor)
# ---------------------------------------------------------------------------
def _resolve_rover_id(user_id):
    """
    Returns (rover_id, error_response) tuple.
    If rover_id is found, error_response is None. Otherwise, rover_id is None.
    """
    roles = db.fetch_all("SELECT role FROM user_roles WHERE user_id = %s", (user_id,))
    role_list = [r["role"] for r in roles] if roles else []

    if "rover" in role_list:
        rover = db.fetch_one("SELECT id FROM rovers WHERE user_id = %s", (user_id,))
        if rover:
            return rover["id"], None
        return None, (jsonify({"status": "error", "error": "Rover profile not found"}), 404)

    if "caregiver" in role_list:
        # Caregiver sees their linked rover (primary_caregiver_id points to the caregiver)
        caregiver = db.fetch_one("SELECT id FROM caregivers WHERE user_id = %s", (user_id,))
        if caregiver:
            rover = db.fetch_one(
                "SELECT id FROM rovers WHERE primary_caregiver_id = %s LIMIT 1",
                (caregiver["id"],)
            )
            if rover:
                return rover["id"], None
        return None, (jsonify({"status": "error", "error": "No linked rover found for caregiver"}), 404)

    if "doctor" in role_list:
        # Doctor sees the first assigned rover (TODO: support multiple patients via query param)
        doctor = db.fetch_one("SELECT id FROM doctors WHERE user_id = %s", (user_id,))
        if doctor:
            # Try to find a rover linked through medical_notes
            note = db.fetch_one("SELECT rover_id FROM medical_notes WHERE doctor_id = %s LIMIT 1", (doctor["id"],))
            if note and note.get("rover_id"):
                return note["rover_id"], None
        return None, (jsonify({"status": "error", "error": "No linked rover found for doctor"}), 404)

    return None, (jsonify({"status": "error", "error": "Unauthorized role"}), 403)


# ---------------------------------------------------------------------------
# GET /api/dashboard/medications
# ---------------------------------------------------------------------------
@dashboard_bp.route("/medications", methods=["GET"])
@require_auth
def get_medications():
    """Fetch medication schedules for the authenticated user's linked rover."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        medications = db.fetch_all("""
            SELECT ms.id, ms.rover_id, ms.medication_id, ms.dosage, ms.frequency,
                   ms.scheduled_time, ms.scheduled_date, ms.instructions,
                   ms.status, ms.taken_at, ms.is_active,
                   mc.brand_name as medication_name, mc.generic_name,
                   CONCAT(u.first_name, ' ', u.last_name) as prescribed_by,
                   rm.start_date, rm.end_date
            FROM medication_schedules ms
            LEFT JOIN medication_catalog mc ON mc.med_id = ms.medication_id
            LEFT JOIN rover_medications rm ON rm.id = ms.rover_medication_id
            LEFT JOIN doctors d ON d.id = ms.prescribed_by
            LEFT JOIN users u ON u.id = d.user_id
            WHERE ms.rover_id = %s AND ms.is_active = 1
            ORDER BY ms.scheduled_date DESC, ms.scheduled_time ASC
        """, (rover_id,))

        return jsonify({"status": "success", "data": medications or []}), 200

    except Exception as e:
        logger.error(f"DB Error fetching medications: {e}")
        return jsonify({"status": "error", "error": "Error fetching medications"}), 500


# ---------------------------------------------------------------------------
# GET /api/dashboard/activities
# ---------------------------------------------------------------------------
@dashboard_bp.route("/activities", methods=["GET"])
@require_auth
def get_activities():
    """Fetch activity logs for the authenticated user's linked rover."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        activities = db.fetch_all("""
            SELECT id, rover_id, type, title, description, priority, timestamp
            FROM activity_logs
            WHERE rover_id = %s
            ORDER BY timestamp DESC
            LIMIT 50
        """, (rover_id,))

        return jsonify({"status": "success", "data": activities or []}), 200

    except Exception as e:
        logger.error(f"DB Error fetching activities: {e}")
        return jsonify({"status": "error", "error": "Error fetching activities"}), 500


# ---------------------------------------------------------------------------
# GET /api/dashboard/linked-rover
# ---------------------------------------------------------------------------
@dashboard_bp.route("/linked-rover", methods=["GET"])
@require_auth
def get_linked_rover():
    """For caregivers/doctors: fetch linked rover's profile and status."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        rover_profile = db.fetch_one("""
            SELECT r.id as rover_id, r.user_id, u.first_name, u.last_name, u.email,
                   r.date_of_birth, r.gender
            FROM rovers r
            JOIN users u ON u.id = r.user_id
            WHERE r.id = %s
        """, (rover_id,))

        if not rover_profile:
            return jsonify({"status": "error", "error": "Rover profile not found"}), 404

        # Check last session activity for online status
        last_session = db.fetch_one("""
            SELECT created_at FROM sessions
            WHERE user_id = %s AND is_active = 1
            ORDER BY created_at DESC LIMIT 1
        """, (rover_profile["user_id"],))

        # datetime imported at module level
        status = "offline"
        last_check_in = None
        if last_session and last_session.get("created_at"):
            session_time = last_session["created_at"]
            if isinstance(session_time, str):
                session_time = datetime.fromisoformat(session_time)
            if session_time.tzinfo is None:
                session_time = session_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            diff = now - session_time
            if diff < timedelta(minutes=10):
                status = "online"
            elif diff < timedelta(hours=1):
                status = "resting"
            last_check_in = session_time.isoformat()

        result = {
            "rover_id": rover_profile["rover_id"],
            "user_id": rover_profile["user_id"],
            "first_name": rover_profile["first_name"],
            "last_name": rover_profile["last_name"],
            "status": status,
            "last_check_in": last_check_in,
        }

        return jsonify({"status": "success", "data": result}), 200

    except Exception as e:
        logger.error(f"DB Error fetching linked rover: {e}")
        return jsonify({"status": "error", "error": "Error fetching linked rover"}), 500


# ---------------------------------------------------------------------------
# GET /api/dashboard/medication-stats
# ---------------------------------------------------------------------------
@dashboard_bp.route("/medication-stats", methods=["GET"])
@require_auth
def get_medication_stats():
    """Fetch medication compliance stats for today."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        stats = db.fetch_one("""
            SELECT
                COUNT(*) as total_doses,
                SUM(CASE WHEN status = 'taken' THEN 1 ELSE 0 END) as taken_doses,
                SUM(CASE WHEN status = 'missed' THEN 1 ELSE 0 END) as missed_doses,
                SUM(CASE WHEN status IN ('upcoming', 'due') THEN 1 ELSE 0 END) as upcoming_doses
            FROM medication_schedules
            WHERE rover_id = %s AND scheduled_date = CURDATE()
        """, (rover_id,))

        return jsonify({"status": "success", "data": stats or {
            "total_doses": 0, "taken_doses": 0, "missed_doses": 0, "upcoming_doses": 0
        }}), 200

    except Exception as e:
        logger.error(f"DB Error fetching medication stats: {e}")
        return jsonify({"status": "error", "error": "Error fetching medication stats"}), 500


# ---------------------------------------------------------------------------
# POST /api/dashboard/medications/take
# ---------------------------------------------------------------------------
@dashboard_bp.route("/medications/take", methods=["POST"])
@require_auth
def take_medication():
    """Mark a medication schedule entry as taken."""
    user_id = g.current_user_id
    try:
        data = request.get_json(silent=True) or {}
        schedule_id = data.get("schedule_id")
        if not schedule_id:
            return jsonify({"status": "error", "error": "schedule_id is required"}), 400

        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        db.execute_query("""
            UPDATE medication_schedules
            SET status = 'taken', taken_at = %s
            WHERE id = %s AND rover_id = %s
        """, (now, schedule_id, rover_id))

        return jsonify({"status": "success", "data": {"taken_at": now}}), 200

    except Exception as e:
        logger.error(f"DB Error marking medication as taken: {e}")
        return jsonify({"status": "error", "error": "Error updating medication status"}), 500


# ---------------------------------------------------------------------------
# GET /api/dashboard/sleep
# ---------------------------------------------------------------------------
@dashboard_bp.route("/sleep", methods=["GET"])
@require_auth
def get_sleep_logs():
    """Fetch sleep logs for the authenticated user's linked rover."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        logs = db.fetch_all("""
            SELECT id, rover_id, date, bed_time, wake_time, duration_hours,
                   quality, deep_sleep_minutes, light_sleep_minutes,
                   rem_sleep_minutes, awakenings, notes
            FROM sleep_logs
            WHERE rover_id = %s
            ORDER BY date DESC
            LIMIT 30
        """, (rover_id,))

        return jsonify({"status": "success", "data": logs or []}), 200

    except Exception as e:
        logger.error(f"DB Error fetching sleep logs: {e}")
        return jsonify({"status": "error", "error": "Error fetching sleep logs"}), 500


# ---------------------------------------------------------------------------
# GET /api/dashboard/hydration
# ---------------------------------------------------------------------------
@dashboard_bp.route("/hydration", methods=["GET"])
@require_auth
def get_hydration_logs():
    """Fetch hydration logs for the authenticated user's linked rover."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        logs = db.fetch_all("""
            SELECT id, rover_id, date, glasses, total_ml, goal_glasses
            FROM hydration_logs
            WHERE rover_id = %s
            ORDER BY date DESC
            LIMIT 30
        """, (rover_id,))

        return jsonify({"status": "success", "data": logs or []}), 200

    except Exception as e:
        logger.error(f"DB Error fetching hydration logs: {e}")
        return jsonify({"status": "error", "error": "Error fetching hydration logs"}), 500


# ---------------------------------------------------------------------------
# POST /api/dashboard/hydration/log
# ---------------------------------------------------------------------------
@dashboard_bp.route("/hydration/log", methods=["POST"])
@require_auth
def log_hydration():
    """Log a glass of water for today."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        import uuid
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        # Check if entry exists for today
        existing = db.fetch_one("""
            SELECT id, glasses, total_ml FROM hydration_logs
            WHERE rover_id = %s AND date = %s
        """, (rover_id, today))

        if existing:
            new_glasses = existing["glasses"] + 1
            new_ml = new_glasses * 250
            db.execute_query("""
                UPDATE hydration_logs SET glasses = %s, total_ml = %s
                WHERE id = %s
            """, (new_glasses, new_ml, existing["id"]))
        else:
            log_id = str(uuid.uuid4())
            db.execute_query("""
                INSERT INTO hydration_logs (id, rover_id, date, glasses, total_ml, goal_glasses)
                VALUES (%s, %s, %s, 1, 250, 8)
            """, (log_id, rover_id, today))
            new_glasses = 1

        return jsonify({"status": "success", "data": {"glasses": new_glasses}}), 200

    except Exception as e:
        logger.error(f"DB Error logging hydration: {e}")
        return jsonify({"status": "error", "error": "Error logging hydration"}), 500


# ---------------------------------------------------------------------------
# GET /api/dashboard/weight
# ---------------------------------------------------------------------------
@dashboard_bp.route("/weight", methods=["GET"])
@require_auth
def get_weight_logs():
    """Fetch weight logs for the authenticated user's linked rover."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        logs = db.fetch_all("""
            SELECT id, rover_id, date, weight_kg, weight_lbs, target_weight_kg, bmi
            FROM weight_logs
            WHERE rover_id = %s
            ORDER BY date DESC
            LIMIT 30
        """, (rover_id,))

        return jsonify({"status": "success", "data": logs or []}), 200

    except Exception as e:
        logger.error(f"DB Error fetching weight logs: {e}")
        return jsonify({"status": "error", "error": "Error fetching weight logs"}), 500


# ---------------------------------------------------------------------------
# GET /api/dashboard/battery
# ---------------------------------------------------------------------------
@dashboard_bp.route("/battery", methods=["GET"])
@require_auth
def get_battery_status():
    """Fetch rover battery status."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        status = db.fetch_one("""
            SELECT id, rover_id, battery_percent, is_charging,
                   estimated_remaining_minutes, recorded_at
            FROM rover_battery_status
            WHERE rover_id = %s
            ORDER BY recorded_at DESC
            LIMIT 1
        """, (rover_id,))

        return jsonify({"status": "success", "data": status}), 200

    except Exception as e:
        logger.error(f"DB Error fetching battery: {e}")
        return jsonify({"status": "error", "error": "Error fetching battery status"}), 500


# ---------------------------------------------------------------------------
# GET /api/dashboard/mood
# ---------------------------------------------------------------------------
@dashboard_bp.route("/mood", methods=["GET"])
@require_auth
def get_mood_logs():
    """Fetch mood logs for the authenticated user's linked rover."""
    user_id = g.current_user_id
    try:
        rover_id, err = _resolve_rover_id(user_id)
        if err:
            return err

        logs = db.fetch_all("""
            SELECT id, rover_id, date, mood, energy_level,
                   anxiety_level, notes, emoji
            FROM mood_logs
            WHERE rover_id = %s
            ORDER BY date DESC
            LIMIT 30
        """, (rover_id,))

        return jsonify({"status": "success", "data": logs or []}), 200

    except Exception as e:
        logger.error(f"DB Error fetching mood logs: {e}")
        return jsonify({"status": "error", "error": "Error fetching mood logs"}), 500
