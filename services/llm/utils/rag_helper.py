import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGDataManager:
    """
    RAGDataManager acts as the data retrieval layer for the Conversational AI.
    It queries patient-specific live metrics (medication schedules, vitals, hydration, and battery status)
    from the MySQL database. If no database connection can be established or credentials are missing,
    it falls back dynamically to a high-fidelity local Mock Database.
    """
    def __init__(self):
        self.use_mock = True
        
        # Load DB credentials from LLM service env
        self.host = os.getenv("DB_HOST", "127.0.0.1")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.database = os.getenv("DB_NAME", "novacare_db")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        
        # Check if auth-backend database credentials are provided
        if self.user and self.password:
            try:
                import mysql.connector
                # Short timeout so starting up does not block excessively if host is offline
                self.conn = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    connection_timeout=2
                )
                self.conn.close()
                self.use_mock = False
                print(f"[RAG] Successfully connected to live database '{self.database}' at {self.host}:{self.port}")
            except Exception as e:
                print(f"[RAG] Live database connection failed ({e}). Falling back to Mock Database.")
                self.use_mock = True
        else:
            print("[RAG] DB credentials missing. Fallback Mock Database active.")

    def get_connection(self):
        """Secure connection utility (only called when use_mock is False)."""
        import mysql.connector
        return mysql.connector.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )

    def _read_mock_db(self):
        """Read data from local mock database file."""
        import json
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "novacare_mock_db.json")
        try:
            if os.path.exists(db_path):
                with open(db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[RAG Mock DB Error] Could not read mock database file: {e}")
        # Return fallback hardcoded if read fails
        return {
            "medications": [],
            "navigation": {"destination": None, "status": "idle", "progress": 0, "follow_mode": False},
            "vitals": {"heart_rate": 72, "blood_pressure": "120/80", "oxygen_level": 98, "temperature": 36.6, "measured_at": "10:15 AM"},
            "hydration": {"glasses": 3, "total_ml": 750, "goal_glasses": 8},
            "battery": {"battery_percent": 84, "is_charging": False, "estimated_remaining_minutes": 180}
        }

    def _write_mock_db(self, data):
        """Write data to local mock database file."""
        import json
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "novacare_mock_db.json")
        try:
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[RAG Mock DB Error] Could not write mock database file: {e}")

    def get_medications(self, rover_id="RV001"):
        """Fetch today's active medication schedules."""
        print(f"\n📡 [RAG DB Access] Fetching medications for '{rover_id}' ({'MOCK MODE' if self.use_mock else 'MYSQL MODE'})")
        if self.use_mock:
            db = self._read_mock_db()
            meds = db.get("medications", [])
            print(f"   ↳ [Mock File] ✓ Returned {len(meds)} mock medications.")
            return meds
        
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT ms.dosage, ms.scheduled_time, ms.status, ms.taken_at,
                       mc.brand_name as medication_name
                FROM medication_schedules ms
                LEFT JOIN medication_catalog mc ON mc.med_id = ms.medication_id
                WHERE ms.rover_id = %s AND ms.is_active = 1
                ORDER BY ms.scheduled_time ASC
            """
            cursor.execute(query, (rover_id,))
            results = cursor.fetchall()
            # Convert timedelta scheduled_time to string representation
            for row in results:
                if row.get("scheduled_time"):
                    row["scheduled_time"] = str(row["scheduled_time"])
                if row.get("taken_at") and not isinstance(row["taken_at"], str):
                    row["taken_at"] = row["taken_at"].strftime("%H:%M:%S")
            print(f"   ↳ [Live DB] ✓ Retrieved {len(results)} active medications.")
            return results
        except Exception as e:
            print(f"   ↳ ❌ [DB Error] Failed to retrieve medications: {e}")
            logger.error(f"[RAG DB Error] medications retrieval failed: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_vitals(self, rover_id="RV001"):
        """Fetch the patient's latest measured vital signs."""
        print(f"📡 [RAG DB Access] Fetching vital signs for '{rover_id}' ({'MOCK MODE' if self.use_mock else 'MYSQL MODE'})")
        
        # Try fetching from real Watch/Robot API first
        try:
            import requests
            robot_api = os.getenv("ROBOT_API_URL", "http://localhost:9000")
            api_key = os.getenv("ROBOT_API_KEY", "novacare-secure-key-2026")
            res = requests.get(f"{robot_api}/api/vitals/current", headers={"X-API-Key": api_key}, timeout=2)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "success" and data.get("heart_rate") is not None:
                    print(f"   ↳ [Robot API] ✓ Returned live watch vitals: HR={data.get('heart_rate')} bpm")
                    return {
                        "heart_rate": data.get("heart_rate"),
                        "blood_pressure": "120/80", # watch doesn't measure this
                        "oxygen_level": 98,
                        "temperature": 36.6,
                        "measured_at": "Live (Watch)"
                    }
        except Exception as e:
            print(f"   ↳ [Robot API] Could not fetch live watch vitals (falling back): {e}")

        if self.use_mock:
            db = self._read_mock_db()
            vitals = db.get("vitals", {})
            print(f"   ↳ [Mock File] ✓ Returned vitals: HR={vitals.get('heart_rate')} bpm, BP={vitals.get('blood_pressure')}, SpO2={vitals.get('oxygen_level')}%")
            return vitals
        
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT heart_rate, blood_pressure_systolic, blood_pressure_diastolic, 
                       spo2 as oxygen_level, temperature, measured_at
                FROM vital_signs
                WHERE rover_id = %s
                ORDER BY measured_at DESC
                LIMIT 1
            """
            cursor.execute(query, (rover_id,))
            result = cursor.fetchone()
            if result:
                # Merge blood pressures
                result["blood_pressure"] = f"{result['blood_pressure_systolic']}/{result['blood_pressure_diastolic']}"
                if result.get("measured_at"):
                    result["measured_at"] = result["measured_at"].strftime("%I:%M %p")
                print(f"   ↳ [Live DB] ✓ Retrieved vitals: HR={result['heart_rate']} bpm, BP={result['blood_pressure']}, SpO2={result['oxygen_level']}%")
                return result
            print("   ↳ [Live DB] ⚠️ No vital records found.")
            return None
        except Exception as e:
            print(f"   ↳ ❌ [DB Error] Failed to retrieve vitals: {e}")
            logger.error(f"[RAG DB Error] vitals retrieval failed: {e}")
            return None
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_hydration(self, rover_id="RV001"):
        """Fetch today's logs for patient water intake."""
        print(f"📡 [RAG DB Access] Fetching hydration logs for '{rover_id}' ({'MOCK MODE' if self.use_mock else 'MYSQL MODE'})")
        if self.use_mock:
            db = self._read_mock_db()
            hydration = db.get("hydration", {})
            print(f"   ↳ [Mock File] ✓ Returned hydration: {hydration.get('glasses')}/{hydration.get('goal_glasses')} glasses ({hydration.get('total_ml')}ml)")
            return hydration
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT glasses, total_ml, goal_glasses
                FROM hydration_logs
                WHERE rover_id = %s AND date = CURDATE()
                LIMIT 1
            """
            cursor.execute(query, (rover_id,))
            result = cursor.fetchone()
            if result:
                print(f"   ↳ [Live DB] ✓ Retrieved hydration: {result['glasses']}/{result['goal_glasses']} glasses ({result['total_ml']}ml)")
                return result
            print("   ↳ [Live DB] ⚠️ No hydration log found for today.")
            return None
        except Exception as e:
            print(f"   ↳ ❌ [DB Error] Failed to retrieve hydration: {e}")
            logger.error(f"[RAG DB Error] hydration retrieval failed: {e}")
            return None
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_battery(self, rover_id="RV001"):
        """Fetch latest battery runtime status for the physical rover."""
        print(f"📡 [RAG DB Access] Fetching battery status for '{rover_id}' ({'MOCK MODE' if self.use_mock else 'MYSQL MODE'})")
        if self.use_mock:
            db = self._read_mock_db()
            battery = db.get("battery", {})
            charging_str = "charging" if battery.get("is_charging") else "discharging"
            print(f"   ↳ [Mock File] ✓ Returned battery: {battery.get('battery_percent')}% ({charging_str}, {battery.get('estimated_remaining_minutes')} mins left)")
            return battery
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT battery_percent, is_charging, estimated_remaining_minutes
                FROM rover_battery_status
                WHERE rover_id = %s
                ORDER BY recorded_at DESC
                LIMIT 1
            """
            cursor.execute(query, (rover_id,))
            result = cursor.fetchone()
            if result:
                charging_str = "charging" if result["is_charging"] else "discharging"
                print(f"   ↳ [Live DB] ✓ Retrieved battery: {result['battery_percent']}% ({charging_str}, {result['estimated_remaining_minutes']} mins left)\n")
                return result
            print("   ↳ [Live DB] ⚠️ No battery logs found.\n")
            return None
        except Exception as e:
            print(f"   ↳ ❌ [DB Error] Failed to retrieve battery: {e}\n")
            logger.error(f"[RAG DB Error] battery retrieval failed: {e}")
            return None
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_all_context(self, rover_id="RV001"):
        """Aggregate all metrics into a clean dictionary payload."""
        return {
            "medications": self.get_medications(rover_id),
            "vitals": self.get_vitals(rover_id),
            "hydration": self.get_hydration(rover_id),
            "battery": self.get_battery(rover_id)
        }

# Singleton instance
rag_manager = RAGDataManager()
