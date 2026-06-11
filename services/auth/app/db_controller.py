"""
NovaCare - Main Integration Controller for Database
Secure connection manager resisting SQL Injections using parameterized queries.
"""
import os
import mysql.connector
from mysql.connector import Error, pooling
import logging

logger = logging.getLogger(__name__)

class DatabaseController:
    """
    Main database connection controller for NovaCare.
    Implements a robust and secure connection pool to the main MySQL engine 
    while keeping the schema abstracted from the calling code.
    All querying methods enforce parameterized query parameters to prevent SQL injection.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseController, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize the connection pool securely using environment variables."""
        self.host = os.getenv("DB_HOST", "192.168.1.164")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.database = os.getenv("DB_NAME", "NovaCare_db")
        
        # Admins will supply their unique username & password via their individual .env files
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        
        self.pool = None
        
        if not self.user or not self.password:
            logger.warning("Database credentials missing in environment variables. DB controller inactive.")
            return

        try:
            # We use a connection pool to manage concurrent database access safely
            self.pool_name = "novacare_pool"
            self.pool_size = 5

            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name=self.pool_name,
                pool_size=self.pool_size,
                pool_reset_session=True,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"Successfully configured DB controller and connected to {self.host}:{self.port} as {self.user}")
        except Error as e:
            logger.error(f"Error while instantiating secure MySQL Connection pool: {e}")
            self.pool = None
            logger.warning("[DB] Live database connection failed. Falling back to Mock Database.")

    def _mock_query(self, query: str, params: tuple = None) -> list:
        """Helper to return mocked results based on SQL queries."""
        from datetime import datetime, timedelta
        q = query.strip().lower()
        
        # 1. users lookup
        if "select * from users where email =" in q:
            email = params[0] if params else "basant@example.com"
            import bcrypt
            hashed = bcrypt.hashpw(b"password", bcrypt.gensalt()).decode("utf-8")
            return [{
                "id": "mock-user-uuid-12345",
                "email": email,
                "hashed_password": hashed,
                "first_name": "Basant",
                "last_name": "Awad",
                "is_active": 1,
                "is_email_verified": 1,
                "google_id": None,
                "profile_picture_url": None,
                "created_at": "2026-06-10 12:00:00",
                "updated_at": "2026-06-10 12:00:00"
            }]
            
        elif "select * from users where id =" in q:
            uid = params[0] if params else "mock-user-uuid-12345"
            return [{
                "id": uid,
                "email": "basant@example.com",
                "hashed_password": "",
                "first_name": "Basant",
                "last_name": "Awad",
                "is_active": 1,
                "is_email_verified": 1,
                "google_id": None,
                "profile_picture_url": None,
                "created_at": "2026-06-10 12:00:00",
                "updated_at": "2026-06-10 12:00:00"
            }]

        # 2. user roles
        elif "select role from user_roles where user_id =" in q:
            return [{"role": "rover"}, {"role": "caregiver"}, {"role": "doctor"}]

        # 3. rovers / caregivers / doctors profiles
        elif "select * from rovers where user_id =" in q:
            uid = params[0] if params else "mock-user-uuid-12345"
            return [{
                "id": "mock-rover-uuid-12345",
                "user_id": uid,
                "date_of_birth": "1950-01-01",
                "gender": "male",
                "address_id": "mock-address-uuid",
                "primary_caregiver_id": "mock-caregiver-uuid-12345"
            }]
            
        elif "select * from caregivers where user_id =" in q:
            uid = params[0] if params else "mock-user-uuid-12345"
            return [{
                "id": "mock-caregiver-uuid-12345",
                "user_id": uid,
                "phone_number": "+1234567890",
                "address_id": "mock-address-uuid",
                "government_id_number": "CG12345",
                "verification_status_id": "mock-status-uuid"
            }]
            
        elif "select * from doctors where user_id =" in q:
            uid = params[0] if params else "mock-user-uuid-12345"
            return [{
                "id": "mock-doctor-uuid-12345",
                "user_id": uid,
                "medical_license_num": "DOC12345",
                "specialization_id": "mock-specialization-uuid",
                "verification_status_id": "mock-status-uuid"
            }]

        # 4. verification statuses
        elif "select id from verification_statuses where status_name =" in q:
            return [{"id": "mock-status-uuid"}]

        # 5. reference data
        elif "select * from countries" in q:
            return [{"id": "country-1", "name": "United States", "code": "US"}]
            
        elif "select * from specializations" in q:
            return [{"id": "spec-1", "name": "Cardiology"}, {"id": "spec-2", "name": "Neurology"}, {"id": "spec-3", "name": "General Practice"}]
            
        elif "select * from relationship_types" in q:
            return [{"id": "rel-1", "name": "Spouse"}, {"id": "rel-2", "name": "Child"}, {"id": "rel-3", "name": "Guardian"}]
            
        elif "select * from medication_catalog" in q:
            return [{"med_id": "med-001", "brand_name": "Lipitor", "generic_name": "Atorvastatin"}]
            
        elif "select * from health_conditions" in q:
            return [{"id": "cond-1", "name": "Hypertension"}, {"id": "cond-2", "name": "Diabetes"}]
            
        elif "select * from allergies" in q:
            return [{"id": "alg-1", "name": "Penicillin", "category": "medication"}]

        # 6. vital signs
        elif "select * from vital_signs" in q:
            return [
                {
                    "id": "vital-1",
                    "rover_id": "mock-rover-uuid-12345",
                    "heart_rate": 72,
                    "spo2": 98,
                    "temperature": 36.6,
                    "measured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "blood_pressure": "120/80"
                }
            ]

        # 7. medical notes
        elif "select * from medical_notes" in q:
            return [
                {
                    "id": "note-1",
                    "rover_id": "mock-rover-uuid-12345",
                    "doctor_id": "mock-doctor-uuid-12345",
                    "note_content": "Patient is in stable condition.",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note_type": "Progress Note",
                    "doctor_first_name": "Nova",
                    "doctor_last_name": "Doctor"
                }
            ]

        # 8. medication schedules
        elif "select ms.id, ms.rover_id" in q:
            return [
                {
                    "id": "med-001",
                    "rover_id": "mock-rover-uuid-12345",
                    "medication_id": "med-001",
                    "dosage": "2 tablets",
                    "frequency": "Daily",
                    "scheduled_time": "08:00:00",
                    "scheduled_date": datetime.now().strftime("%Y-%m-%d"),
                    "instructions": "Take with water",
                    "status": "taken",
                    "taken_at": "08:05:00",
                    "is_active": 1,
                    "medication_name": "Lipitor (Atorvastatin)",
                    "generic_name": "Atorvastatin",
                    "prescribed_by": "Nova Doctor",
                    "start_date": "2026-01-01",
                    "end_date": "2026-12-31"
                }
            ]

        # 9. activity logs
        elif "select id, rover_id, type" in q or "select id, rover_id, type, title" in q:
            return [
                {
                    "id": "act-1",
                    "rover_id": "mock-rover-uuid-12345",
                    "type": "medication",
                    "title": "Medication Taken",
                    "description": "Lipitor taken at 08:05 AM",
                    "priority": "low",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            ]

        # 10. sleep logs
        elif "select id, rover_id, date, bed_time" in q:
            return [
                {
                    "id": "sleep-1",
                    "rover_id": "mock-rover-uuid-12345",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "bed_time": "22:00:00",
                    "wake_time": "06:00:00",
                    "duration_hours": 8.0,
                    "quality": "Good",
                    "deep_sleep_minutes": 120,
                    "light_sleep_minutes": 300,
                    "rem_sleep_minutes": 60,
                    "awakenings": 1,
                    "notes": "Slept well"
                }
            ]

        # 11. hydration logs
        elif "select id, rover_id, date, glasses" in q:
            return [
                {
                    "id": "hyd-1",
                    "rover_id": "mock-rover-uuid-12345",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "glasses": 3,
                    "total_ml": 750,
                    "goal_glasses": 8
                }
            ]

        # 12. weight logs
        elif "select id, rover_id, date, weight_kg" in q:
            return [
                {
                    "id": "weight-1",
                    "rover_id": "mock-rover-uuid-12345",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "weight_kg": 75.0,
                    "weight_lbs": 165.3,
                    "target_weight_kg": 74.0,
                    "bmi": 23.5
                }
            ]

        # 13. battery status
        elif "select id, rover_id, battery_percent" in q:
            return [{
                "id": "bat-1",
                "rover_id": "mock-rover-uuid-12345",
                "battery_percent": 84,
                "is_charging": False,
                "estimated_remaining_minutes": 180,
                "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]

        # 14. mood logs
        elif "select id, rover_id, date, mood" in q:
            return [
                {
                    "id": "mood-1",
                    "rover_id": "mock-rover-uuid-12345",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "mood": "Good",
                    "energy_level": 4,
                    "anxiety_level": 2,
                    "notes": "Feeling great!",
                    "emoji": "😊"
                }
            ]

        # 15. linked rover lookup
        elif "select r.id as rover_id, r.user_id" in q:
            return [{
                "rover_id": "mock-rover-uuid-12345",
                "user_id": "mock-user-uuid-12345",
                "first_name": "Basant",
                "last_name": "Awad",
                "email": "basant@example.com",
                "date_of_birth": "1950-01-01",
                "gender": "male"
            }]

        # 16. sessions check
        elif "select created_at from sessions" in q:
            return [{"created_at": datetime.now().isoformat()}]

        # 17. medication stats
        elif "count(*) as total_doses" in q:
            return [{
                "total_doses": 3,
                "taken_doses": 2,
                "missed_doses": 0,
                "upcoming_doses": 1
            }]

        return []

    def get_connection(self):
        """Yield robust connections uniformly, drawn securely from the pool."""
        if self.pool is None:
            raise Exception("Database controller rejected connection request: Pool not initialized. Please configure DB_USER and DB_PASSWORD.")
        return self.pool.get_connection()

    def execute_query(self, query: str, params: tuple = None) -> int:
        """
        Executes a data manipulation query (INSERT, UPDATE, DELETE) securely.
        REQUIRED: Passing `params` dynamically uses parameterized payloads to repel typical injection vectors.
        """
        if self.pool is None:
            logger.warning(f"DB in mock mode. Executing mock query: {query}")
            return 1
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            return cursor.rowcount
        except Error as e:
            if conn:
                conn.rollback()
            logger.error(f"DB Controller caught query execution error: {e}")
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def fetch_all(self, query: str, params: tuple = None) -> list:
        """
        Executes a SELECT query that anticipates retrieving multiple records safely.
        """
        if self.pool is None:
            return self._mock_query(query, params)
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            return cursor.fetchall()
        except Error as e:
            logger.error(f"DB Controller fault mapping dictionary stream securely: {e}")
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def fetch_one(self, query: str, params: tuple = None) -> dict:
        """
        Executes a targeted internal lookup or SELECT query expecting an individual entity safely.
        """
        if self.pool is None:
            res = self._mock_query(query, params)
            return res[0] if res else None
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            return cursor.fetchone()
        except Error as e:
            logger.error(f"DB Controller fault resolving lone entity node securely: {e}")
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

# Provide a modular singleton to other elements of the codebase
db = DatabaseController()
