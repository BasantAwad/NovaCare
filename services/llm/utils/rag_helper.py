import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGDataManager:
    """
    RAGDataManager acts as the data retrieval layer for the Conversational AI.
    It queries patient-specific live metrics from the MySQL database.
    Supports intelligent query routing to only fetch relevant data sources.
    If no database connection can be established, it falls back to a local Mock Database.
    """
    def __init__(self):
        self.use_mock = True
        
        self.host = os.getenv("DB_HOST", "127.0.0.1")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.database = os.getenv("DB_NAME", "novacare_db")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        
        if self.user and self.password:
            try:
                import mysql.connector
                self.conn = mysql.connector.connect(
                    host=self.host, port=self.port, database=self.database,
                    user=self.user, password=self.password, connection_timeout=2
                )
                self.conn.close()
                self.use_mock = False
                print(f"[RAG] Successfully connected to live database '{self.database}' at {self.host}:{self.port}")
            except Exception as e:
                print(f"[RAG] Live database connection failed ({e}). Falling back to Mock Database.")
                self.use_mock = True
        else:
            print("[RAG] DB credentials missing. Fallback Mock Database active.")

    # ── Query Router ─────────────────────────────────────────────────────
    
    # Mapping of keywords/intents to data source names
    ROUTE_MAP = {
        "medications": ["medication", "medicine", "pill", "drug", "prescription", "dose", "dosage", "lipitor", "metformin", "synthroid", "take my"],
        "vitals": ["vital", "heart rate", "blood pressure", "oxygen", "spo2", "temperature", "pulse", "bp"],
        "vitals_trend": ["trend", "week", "past few days", "lately", "over time", "history", "been"],
        "appointments": ["appointment", "doctor", "visit", "checkup", "schedule", "follow-up", "therapy session"],
        "health_conditions": ["condition", "diagnosis", "disease", "parkinson", "stroke", "injury", "health condition"],
        "allergies": ["allergy", "allergic", "allergies", "penicillin", "peanut", "latex"],
        "emergency_contacts": ["emergency", "contact", "call someone", "who to call", "family", "guardian"],
        "medical_notes": ["doctor said", "doctor note", "lab result", "prescription note", "medical note", "last visit"],
        "emotion_history": ["feeling", "mood", "emotion", "sentiment", "how have i been", "distress"],
        "notifications": ["notification", "reminder", "alert", "pending", "unread"],
    }
    
    def route_query(self, user_query):
        """Determine which data sources are relevant for a given user query."""
        query_lower = user_query.lower()
        matched_sources = set()
        
        for source, keywords in self.ROUTE_MAP.items():
            for kw in keywords:
                if kw in query_lower:
                    matched_sources.add(source)
                    break
        
        # If nothing matched, return all sources (safe fallback)
        if not matched_sources:
            return list(self.ROUTE_MAP.keys())
        
        return list(matched_sources)

    # ── Connection helper ────────────────────────────────────────────────

    def get_connection(self):
        import mysql.connector
        return mysql.connector.connect(
            host=self.host, port=self.port, database=self.database,
            user=self.user, password=self.password
        )

    # ── Mock DB helpers ──────────────────────────────────────────────────

    def _read_mock_db(self):
        import json
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "novacare_mock_db.json")
        try:
            if os.path.exists(db_path):
                with open(db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[RAG Mock DB Error] Could not read mock database file: {e}")
        return {
            "medications": [], "vitals": {"heart_rate": 72, "blood_pressure": "120/80", "oxygen_level": 98, "temperature": 36.6},
            "hydration": {"glasses": 3, "total_ml": 750, "goal_glasses": 8},
            "battery": {"battery_percent": 84, "is_charging": False, "estimated_remaining_minutes": 180}
        }

    def _write_mock_db(self, data):
        import json
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "novacare_mock_db.json")
        try:
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[RAG Mock DB Error] Could not write mock database file: {e}")

    # ── Data Retrieval Methods ───────────────────────────────────────────

    def get_medications(self, rover_id="RV001"):
        print(f"\n[RAG] Fetching medications for '{rover_id}' ({'MOCK' if self.use_mock else 'SQL'})")
        if self.use_mock:
            db = self._read_mock_db()
            meds = db.get("medications", [])
            print(f"   -> {len(meds)} mock medications")
            return meds
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT rm.dosage, rm.frequency as scheduled_time, 'pending' as status, NULL as taken_at,
                       mc.brand_name as medication_name
                FROM rover_medications rm
                LEFT JOIN medication_catalog mc ON mc.med_id = rm.medication_id
                WHERE rm.rover_id = %s AND rm.is_active = 1
            """, (rover_id,))
            results = cursor.fetchall()
            for row in results:
                if row.get("scheduled_time") and not isinstance(row["scheduled_time"], str):
                    row["scheduled_time"] = str(row["scheduled_time"])
            print(f"   -> [DB] {len(results)} active medications")
            return results
        except Exception as e:
            print(f"   -> [DB Error] medications: {e}")
            logger.error(f"[RAG] medications failed: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_vitals(self, rover_id="RV001"):
        print(f"[RAG] Fetching vitals for '{rover_id}' ({'MOCK' if self.use_mock else 'SQL'})")
        
        # Try Robot API first
        try:
            import requests
            robot_api = os.getenv("ROBOT_API_URL", "http://localhost:9000")
            api_key = os.getenv("ROBOT_API_KEY", "novacare-secure-key-2026")
            res = requests.get(f"{robot_api}/api/vitals/current", headers={"X-API-Key": api_key}, timeout=2)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "success" and data.get("heart_rate") is not None:
                    print(f"   -> [Robot API] HR={data.get('heart_rate')} bpm")
                    return {"heart_rate": data.get("heart_rate"), "blood_pressure": "120/80", "oxygen_level": 98, "temperature": 36.6, "measured_at": "Live (Watch)"}
        except Exception as e:
            print(f"   -> [Robot API] offline: {type(e).__name__}")

        if self.use_mock:
            db = self._read_mock_db()
            vitals = db.get("vitals", {})
            print(f"   -> [Mock] HR={vitals.get('heart_rate')} bpm")
            return vitals
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT heart_rate, spo2 as oxygen_level, temperature, measured_at
                FROM vital_signs WHERE rover_id = %s ORDER BY measured_at DESC LIMIT 1
            """, (rover_id,))
            result = cursor.fetchone()
            if result:
                result["blood_pressure"] = "120/80 (Est)"
                if result.get("measured_at"):
                    result["measured_at"] = result["measured_at"].strftime("%I:%M %p")
                print(f"   -> [DB] HR={result['heart_rate']} bpm, SpO2={result['oxygen_level']}%")
                return result
            print("   -> [DB] No vitals found")
            return None
        except Exception as e:
            print(f"   -> [DB Error] vitals: {e}")
            return None
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_vitals_trend(self, rover_id="RV001", days=7):
        """Fetch vitals over the past N days for trend analysis."""
        print(f"[RAG] Fetching {days}-day vitals trend for '{rover_id}'")
        if self.use_mock:
            print("   -> [Mock] No trend data available")
            return []
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT heart_rate, spo2 as oxygen_level, temperature, measured_at
                FROM vital_signs WHERE rover_id = %s AND measured_at >= NOW() - INTERVAL %s DAY
                ORDER BY measured_at ASC
            """, (rover_id, days))
            results = cursor.fetchall()
            for row in results:
                if row.get("measured_at"):
                    row["measured_at"] = row["measured_at"].strftime("%Y-%m-%d %H:%M")
            print(f"   -> [DB] {len(results)} readings over {days} days")
            return results
        except Exception as e:
            print(f"   -> [DB Error] vitals_trend: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_appointments(self, rover_id="RV001"):
        """Fetch upcoming and recent appointments."""
        print(f"[RAG] Fetching appointments for '{rover_id}'")
        if self.use_mock:
            print("   -> [Mock] No appointment data")
            return []
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT a.scheduled_at, a.notes,
                       at2.type_name as appointment_type,
                       ast.status_name as status,
                       u.first_name as doctor_first_name, u.last_name as doctor_last_name,
                       s.name as specialization
                FROM appointments a
                LEFT JOIN appointment_types at2 ON at2.id = a.appointment_type_id
                LEFT JOIN appointment_statuses ast ON ast.id = a.status_id
                LEFT JOIN doctors d ON d.id = a.doctor_id
                LEFT JOIN users u ON u.id = d.user_id
                LEFT JOIN specializations s ON s.id = d.specialization_id
                WHERE a.rover_id = %s
                ORDER BY a.scheduled_at DESC LIMIT 10
            """, (rover_id,))
            results = cursor.fetchall()
            for row in results:
                if row.get("scheduled_at"):
                    row["scheduled_at"] = row["scheduled_at"].strftime("%Y-%m-%d %I:%M %p")
            print(f"   -> [DB] {len(results)} appointments")
            return results
        except Exception as e:
            print(f"   -> [DB Error] appointments: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_health_conditions(self, rover_id="RV001"):
        """Fetch diagnosed health conditions."""
        print(f"[RAG] Fetching health conditions for '{rover_id}'")
        if self.use_mock:
            print("   -> [Mock] No condition data")
            return []
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT hc.name as condition_name, hc.icd10_code as icd_code, rhc.severity, rhc.notes
                FROM rover_health_conditions rhc
                JOIN health_conditions hc ON hc.id = rhc.condition_id
                WHERE rhc.rover_id = %s
            """, (rover_id,))
            results = cursor.fetchall()
            print(f"   -> [DB] {len(results)} conditions")
            return results
        except Exception as e:
            print(f"   -> [DB Error] health_conditions: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_allergies(self, rover_id="RV001"):
        """Fetch known allergies."""
        print(f"[RAG] Fetching allergies for '{rover_id}'")
        if self.use_mock:
            print("   -> [Mock] No allergy data")
            return []
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT al.name as allergy_name, al.category as allergy_type, ra.severity
                FROM rover_allergies ra
                JOIN allergies al ON al.id = ra.allergy_id
                WHERE ra.rover_id = %s AND ra.is_active = 1
            """, (rover_id,))
            results = cursor.fetchall()
            print(f"   -> [DB] {len(results)} allergies")
            return results
        except Exception as e:
            print(f"   -> [DB Error] allergies: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_emergency_contacts(self, rover_id="RV001"):
        """Fetch emergency contacts."""
        print(f"[RAG] Fetching emergency contacts for '{rover_id}'")
        if self.use_mock:
            print("   -> [Mock] No contact data")
            return []
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT name, relationship, phone_number as phone, is_primary
                FROM emergency_contacts
                WHERE rover_id = %s
                ORDER BY is_primary DESC
            """, (rover_id,))
            results = cursor.fetchall()
            print(f"   -> [DB] {len(results)} emergency contacts")
            return results
        except Exception as e:
            print(f"   -> [DB Error] emergency_contacts: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_medical_notes(self, rover_id="RV001"):
        """Fetch recent doctor notes."""
        print(f"[RAG] Fetching medical notes for '{rover_id}'")
        if self.use_mock:
            print("   -> [Mock] No note data")
            return []
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT mn.note_content, mn.created_at,
                       mnt.note_type as note_type,
                       u.first_name as doctor_first_name, u.last_name as doctor_last_name
                FROM medical_notes mn
                LEFT JOIN medical_note_types mnt ON mnt.id = mn.note_type_id
                LEFT JOIN doctors d ON d.id = mn.doctor_id
                LEFT JOIN users u ON u.id = d.user_id
                WHERE mn.rover_id = %s
                ORDER BY mn.created_at DESC LIMIT 5
            """, (rover_id,))
            results = cursor.fetchall()
            for row in results:
                if row.get("created_at"):
                    row["created_at"] = row["created_at"].strftime("%Y-%m-%d")
            print(f"   -> [DB] {len(results)} medical notes")
            return results
        except Exception as e:
            print(f"   -> [DB Error] medical_notes: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_emotion_history(self, rover_id="RV001"):
        """Fetch recent emotion tracking data."""
        print(f"[RAG] Fetching emotion history for '{rover_id}'")
        if self.use_mock:
            print("   -> [Mock] No emotion history")
            return []
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT date, avg_sentiment, primary_emotion, distress_detected
                FROM emotion_tracking
                WHERE rover_id = %s
                ORDER BY date DESC LIMIT 10
            """, (rover_id,))
            results = cursor.fetchall()
            for row in results:
                if row.get("date"):
                    row["date"] = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
                if row.get("avg_sentiment") is not None:
                    row["avg_sentiment"] = float(row["avg_sentiment"])
            print(f"   -> [DB] {len(results)} emotion entries")
            return results
        except Exception as e:
            print(f"   -> [DB Error] emotion_history: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_notifications(self, rover_id="RV001"):
        """Fetch pending notifications. Maps rover_id -> user_id for the FK."""
        print(f"[RAG] Fetching notifications for '{rover_id}'")
        if self.use_mock:
            print("   -> [Mock] No notification data")
            return []
        
        conn = cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            # Resolve rover_id -> user_id
            cursor.execute("SELECT user_id FROM rovers WHERE id = %s", (rover_id,))
            row = cursor.fetchone()
            if not row:
                print(f"   -> [DB] Rover '{rover_id}' not found in rovers table")
                return []
            user_id = row["user_id"]
            
            cursor.execute("""
                SELECT title, message, is_read, created_at
                FROM notifications
                WHERE recipient_id = %s
                ORDER BY created_at DESC LIMIT 10
            """, (user_id,))
            results = cursor.fetchall()
            for r in results:
                if r.get("created_at"):
                    r["created_at"] = r["created_at"].strftime("%Y-%m-%d %H:%M")
            print(f"   -> [DB] {len(results)} notifications")
            return results
        except Exception as e:
            print(f"   -> [DB Error] notifications: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_hydration(self, rover_id="RV001"):
        """Fetch hydration data (mock-only, not in current DB schema)."""
        print(f"[RAG] Fetching hydration for '{rover_id}' (mock-only)")
        if self.use_mock:
            db = self._read_mock_db()
            hydration = db.get("hydration", {})
            print(f"   -> [Mock] {hydration.get('glasses')}/{hydration.get('goal_glasses')} glasses")
            return hydration
        print("   -> [DB] Hydration table not in schema")
        return None

    def get_battery(self, rover_id="RV001"):
        """Fetch battery data (mock-only, not in current DB schema)."""
        print(f"[RAG] Fetching battery for '{rover_id}' (mock-only)")
        if self.use_mock:
            db = self._read_mock_db()
            battery = db.get("battery", {})
            print(f"   -> [Mock] {battery.get('battery_percent')}%")
            return battery
        print("   -> [DB] Battery table not in schema")
        return None

    # ── Aggregation ──────────────────────────────────────────────────────

    def get_all_context(self, rover_id="RV001"):
        """Aggregate ALL available metrics into a clean dictionary."""
        return {
            "medications": self.get_medications(rover_id),
            "vitals": self.get_vitals(rover_id),
            "vitals_trend": self.get_vitals_trend(rover_id),
            "appointments": self.get_appointments(rover_id),
            "health_conditions": self.get_health_conditions(rover_id),
            "allergies": self.get_allergies(rover_id),
            "emergency_contacts": self.get_emergency_contacts(rover_id),
            "medical_notes": self.get_medical_notes(rover_id),
            "emotion_history": self.get_emotion_history(rover_id),
            "notifications": self.get_notifications(rover_id),
            "hydration": self.get_hydration(rover_id),
            "battery": self.get_battery(rover_id),
        }

    def get_routed_context(self, user_query, rover_id="RV001"):
        """Intelligent retrieval: only fetch data sources relevant to the user's query."""
        sources = self.route_query(user_query)
        print(f"[RAG Router] Query: '{user_query[:60]}...' -> Sources: {sources}")
        
        context = {}
        method_map = {
            "medications": self.get_medications,
            "vitals": self.get_vitals,
            "vitals_trend": self.get_vitals_trend,
            "appointments": self.get_appointments,
            "health_conditions": self.get_health_conditions,
            "allergies": self.get_allergies,
            "emergency_contacts": self.get_emergency_contacts,
            "medical_notes": self.get_medical_notes,
            "emotion_history": self.get_emotion_history,
            "notifications": self.get_notifications,
        }
        
        for source in sources:
            if source in method_map:
                context[source] = method_map[source](rover_id)
        
        return context, sources

# Singleton instance
rag_manager = RAGDataManager()
