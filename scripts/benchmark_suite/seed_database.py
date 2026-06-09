"""
NovaCare RAG Database Seeder
Populates the live MySQL instance with realistic patient data for RV001.
Safe to re-run (uses INSERT IGNORE and checks for existing data).
"""
import mysql.connector
import uuid
import random
from datetime import datetime, timedelta

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "basant_admin",
    "password": "NovaCare_2026_B",
    "database": "NovaCare_db"
}

def get_conn():
    return mysql.connector.connect(**DB_CONFIG)

def seed_medications(cur):
    """Seed 8 realistic medications for RV001."""
    print("\n📦 Seeding rover_medications...")
    
    # First check what medication_catalog entries exist
    cur.execute("SELECT med_id, brand_name FROM medication_catalog LIMIT 20")
    catalog = {row[0]: row[1] for row in cur.fetchall()}
    print(f"   Found {len(catalog)} medications in catalog")
    
    # Clear existing RV001 medications to avoid duplicates
    cur.execute("DELETE FROM rover_medications WHERE rover_id = 'RV001'")
    
    meds = [
        # (med_id, dosage, frequency)
        (1, "2 tablets", "08:00:00"),       # Lipitor - morning cholesterol
        (3, "1 capsule", "07:00:00"),       # Synthroid - thyroid, before breakfast
        (2, "1 tablet", "08:00:00"),        # Glucophage/Metformin - with breakfast
        (2, "1 tablet", "18:00:00"),        # Glucophage/Metformin - with dinner (twice daily)
        (4, "1 tablet", "09:00:00"),        # Zestril - blood pressure
        (5, "1 tablet", "21:00:00"),        # Norvasc - blood pressure, evening
    ]
    
    # Add a few more from catalog if available
    if len(catalog) > 5:
        extra_meds = [
            (6, "1 capsule", "12:00:00"),
            (7, "2 tablets", "20:00:00"),
        ]
        meds.extend(extra_meds)
    
    for med_id, dosage, freq in meds:
        if med_id in catalog:
            cur.execute(
                "INSERT INTO rover_medications (id, rover_id, medication_id, dosage, frequency, is_active) "
                "VALUES (%s, %s, %s, %s, %s, 1)",
                (str(uuid.uuid4()), "RV001", med_id, dosage, freq)
            )
            print(f"   ✓ {catalog[med_id]} - {dosage} at {freq}")
    
    print(f"   ✓ Seeded {len(meds)} medications for RV001")

def seed_vitals(cur):
    """Seed 7 days of vital signs time-series data for RV001."""
    print("\n📦 Seeding vital_signs (7-day time series)...")
    
    # Clear old vitals for clean time-series
    cur.execute("DELETE FROM vital_signs WHERE rover_id = 'RV001'")
    
    now = datetime.now()
    count = 0
    
    # 7 days, 4 readings per day (morning, noon, afternoon, evening)
    for day_offset in range(7):
        day = now - timedelta(days=day_offset)
        for hour in [8, 12, 16, 20]:
            ts = day.replace(hour=hour, minute=random.randint(0, 59), second=0, microsecond=0)
            
            # Simulate realistic vital sign variations
            base_hr = 72
            hr_variation = random.gauss(0, 5)  # Normal distribution around base
            if hour == 8:
                hr = int(base_hr - 3 + hr_variation)  # Lower in morning
            elif hour in [12, 16]:
                hr = int(base_hr + 5 + hr_variation)  # Higher during activity
            else:
                hr = int(base_hr + hr_variation)       # Normal in evening
            
            hr = max(55, min(110, hr))  # Clamp to realistic range
            
            spo2 = round(random.uniform(96.0, 99.5), 2)
            temp = round(random.gauss(36.6, 0.3), 2)
            temp = max(36.0, min(37.8, temp))
            
            cur.execute(
                "INSERT INTO vital_signs (id, rover_id, heart_rate, spo2, temperature, measured_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), "RV001", hr, spo2, temp, ts)
            )
            count += 1
    
    # Add a couple of abnormal readings (elevated HR) for trend detection testing
    abnormal_ts = now - timedelta(days=1, hours=3)
    cur.execute(
        "INSERT INTO vital_signs (id, rover_id, heart_rate, spo2, temperature, measured_at) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (str(uuid.uuid4()), "RV001", 102, 95.5, 37.4, abnormal_ts)
    )
    count += 1
    
    print(f"   ✓ Seeded {count} vital sign readings over 7 days")

def seed_appointments(cur):
    """Seed 6 appointments (3 past, 3 upcoming) for RV001."""
    print("\n📦 Seeding appointments...")
    
    cur.execute("DELETE FROM appointments WHERE rover_id = 'RV001'")
    
    now = datetime.now()
    
    appointments = [
        # Past appointments
        {
            "doctor_id": "DOC001",
            "type_id": "AT01",     # Routine Checkup
            "status_id": "AS03",   # Completed
            "scheduled_at": now - timedelta(days=14, hours=-10),
            "notes": "Regular quarterly checkup. Blood pressure stable. Lipitor dosage maintained."
        },
        {
            "doctor_id": "DOC002",
            "type_id": "AT04",     # Physical Therapy
            "status_id": "AS03",   # Completed
            "scheduled_at": now - timedelta(days=7, hours=-14),
            "notes": "PT session completed. Good progress on upper limb mobility. Continue home exercises."
        },
        {
            "doctor_id": "DOC003",
            "type_id": "AT05",     # Lab Work
            "status_id": "AS03",   # Completed
            "scheduled_at": now - timedelta(days=3, hours=-9),
            "notes": "Blood panel drawn. HbA1c results pending. Fasting glucose was 118 mg/dL."
        },
        # Upcoming appointments
        {
            "doctor_id": "DOC001",
            "type_id": "AT09",     # Follow-up
            "status_id": "AS02",   # Confirmed
            "scheduled_at": now + timedelta(days=3, hours=2),
            "notes": "Follow-up on lab results. Review HbA1c and adjust Metformin if needed."
        },
        {
            "doctor_id": "DOC002",
            "type_id": "AT04",     # Physical Therapy
            "status_id": "AS02",   # Confirmed
            "scheduled_at": now + timedelta(days=7, hours=4),
            "notes": "Weekly PT session. Focus on shoulder range of motion."
        },
        {
            "doctor_id": "DOC003",
            "type_id": "AT08",     # Mental Health Session
            "status_id": "AS01",   # Scheduled
            "scheduled_at": now + timedelta(days=10, hours=1),
            "notes": "Monthly mental wellness check-in."
        },
    ]
    
    for appt in appointments:
        cur.execute(
            "INSERT INTO appointments (id, rover_id, doctor_id, appointment_type_id, status_id, scheduled_at, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), "RV001", appt["doctor_id"], appt["type_id"], 
             appt["status_id"], appt["scheduled_at"], appt["notes"])
        )
        delta = appt["scheduled_at"] - now
        direction = "in " + str(abs(delta.days)) + " days" if delta.days > 0 else str(abs(delta.days)) + " days ago"
        print(f"   ✓ {appt['type_id']} with {appt['doctor_id']} — {direction}")
    
    print(f"   ✓ Seeded {len(appointments)} appointments")

def seed_health_conditions(cur):
    """Link health conditions to RV001."""
    print("\n📦 Seeding rover_health_conditions...")
    
    cur.execute("DELETE FROM rover_health_conditions WHERE rover_id = 'RV001'")
    
    # Get existing health condition IDs
    cur.execute("SELECT id, name FROM health_conditions")
    conditions = {row[1]: row[0] for row in cur.fetchall()}
    
    # RV001's conditions: Parkinson's, Stroke Recovery, Traumatic Brain Injury
    target_conditions = [
        ("Parkinsons Disease", "moderate", "Diagnosed March 2024. Early-stage tremor in left hand."),
        ("Stroke Recovery", "severe", "Post-stroke rehabilitation since June 2025. Improving mobility."),
        ("Traumatic Brain Injury", "mild", "Under monitoring since January 2025. Cognitive function stable."),
    ]
    
    for name, severity, notes in target_conditions:
        if name in conditions:
            cur.execute(
                "INSERT INTO rover_health_conditions (id, rover_id, condition_id, severity, notes) "
                "VALUES (%s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), "RV001", conditions[name], severity, notes)
            )
            print(f"   + {name} (severity: {severity})")
    
    print(f"   ✓ Seeded {len(target_conditions)} health conditions")

def seed_allergies(cur):
    """Link allergies to RV001."""
    print("\n📦 Seeding rover_allergies...")
    
    cur.execute("DELETE FROM rover_allergies WHERE rover_id = 'RV001'")
    
    cur.execute("SELECT id, name FROM allergies")
    allergies = {row[1]: row[0] for row in cur.fetchall()}
    
    target_allergies = [
        ("Penicillin", "severe"),
        ("Aspirin", "moderate"),
        ("Peanuts", "severe"),
        ("Latex", "mild"),
    ]
    
    for name, severity in target_allergies:
        if name in allergies:
            cur.execute(
                "INSERT INTO rover_allergies (id, rover_id, allergy_id, severity) "
                "VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), "RV001", allergies[name], severity)
            )
            print(f"   ✓ {name} (severity: {severity})")
    
    print(f"   ✓ Seeded {len(target_allergies)} allergies")

def seed_medical_notes(cur):
    """Seed doctor notes for RV001."""
    print("\n📦 Seeding medical_notes...")
    
    cur.execute("DELETE FROM medical_notes WHERE rover_id = 'RV001'")
    
    now = datetime.now()
    
    notes = [
        {
            "doctor_id": "DOC001",
            "note_type_id": "NT01",  # Diagnosis
            "content": "Patient presents with early-stage Parkinson's tremor in left hand. Initiated Levodopa therapy. "
                       "Recommend follow-up in 4 weeks to assess medication response.",
            "created_at": now - timedelta(days=60)
        },
        {
            "doctor_id": "DOC001",
            "note_type_id": "NT02",  # Progress Note
            "content": "Blood pressure well-controlled on Zestril 10mg. HbA1c dropped from 7.8 to 7.2 — "
                       "Metformin dosage adequate. Continue current regimen.",
            "created_at": now - timedelta(days=14)
        },
        {
            "doctor_id": "DOC002",
            "note_type_id": "NT02",  # Progress Note
            "content": "Physical therapy session #12. Patient showing 15% improvement in upper limb range of motion. "
                       "Grip strength improving. Can now independently transfer from wheelchair. "
                       "Recommend 2 more weeks of intensive PT.",
            "created_at": now - timedelta(days=7)
        },
        {
            "doctor_id": "DOC003",
            "note_type_id": "NT04",  # Lab Results
            "content": "Lab results: Fasting glucose 118 mg/dL (slightly elevated). HbA1c 7.2% (improving). "
                       "Cholesterol panel: Total 195, LDL 120, HDL 55, Triglycerides 150. "
                       "Lipitor effective. Recheck in 3 months.",
            "created_at": now - timedelta(days=3)
        },
        {
            "doctor_id": "DOC003",
            "note_type_id": "NT03",  # Prescription Note
            "content": "Renewed prescriptions: Lipitor 20mg (90-day supply), Metformin 500mg (90-day supply), "
                       "Synthroid 50mcg (90-day supply). All refills authorized for 1 year.",
            "created_at": now - timedelta(days=3)
        },
    ]
    
    for note in notes:
        cur.execute(
            "INSERT INTO medical_notes (id, rover_id, doctor_id, note_type_id, note_content, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), "RV001", note["doctor_id"], note["note_type_id"],
             note["content"], note["created_at"])
        )
        days_ago = (now - note["created_at"]).days
        print(f"   ✓ {note['note_type_id']} by {note['doctor_id']} — {days_ago} days ago")
    
    print(f"   ✓ Seeded {len(notes)} medical notes")

def seed_emotion_tracking(cur):
    """Seed emotion history for RV001."""
    print("\n[Seed] Seeding emotion_tracking...")
    
    cur.execute("DELETE FROM emotion_tracking WHERE rover_id = 'RV001'")
    
    now = datetime.now()
    emotions = ["happy", "neutral", "sad", "angry", "fearful", "surprised", "disgusted"]
    weights = [0.30, 0.35, 0.15, 0.05, 0.05, 0.05, 0.05]
    
    count = 0
    for day_offset in range(15):
        day = (now - timedelta(days=day_offset)).date()
        emotion = random.choices(emotions, weights=weights, k=1)[0]
        sentiment = round(random.uniform(-1.0, 1.0), 2)
        # Map emotions to sentiment range
        if emotion in ["happy", "surprised"]:
            sentiment = round(random.uniform(0.3, 0.95), 2)
        elif emotion == "neutral":
            sentiment = round(random.uniform(-0.1, 0.3), 2)
        else:
            sentiment = round(random.uniform(-0.95, -0.1), 2)
        
        distress = 1 if emotion in ["sad", "angry", "fearful"] else 0
        
        cur.execute(
            "INSERT INTO emotion_tracking (id, rover_id, date, avg_sentiment, primary_emotion, distress_detected) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), "RV001", day, sentiment, emotion, distress)
        )
        count += 1
    
    print(f"   + Seeded {count} emotion tracking entries (15 days)")

def seed_notifications(cur):
    """Seed notifications for RV001."""
    print("\n[Seed] Seeding notifications...")
    
    # notifications.recipient_id references users.id, not rover id
    # RV001's user_id is 'U001'
    cur.execute("DELETE FROM notifications WHERE recipient_id = 'U001'")
    
    now = datetime.now()
    
    notifications = [
        {"title": "Medication Reminder", "message": "Time to take Lipitor (2 tablets) - Morning dose", "is_read": False, "created_at": now - timedelta(hours=2)},
        {"title": "Medication Reminder", "message": "Time to take Synthroid (1 capsule) - Before breakfast", "is_read": False, "created_at": now - timedelta(hours=3)},
        {"title": "Appointment Reminder", "message": "Follow-up appointment with Dr. Ahmed (Cardiology) in 3 days", "is_read": False, "created_at": now - timedelta(hours=6)},
        {"title": "Hydration Alert", "message": "You've only had 2 glasses of water today. Try to drink more!", "is_read": True, "created_at": now - timedelta(hours=8)},
        {"title": "Vital Sign Alert", "message": "Your heart rate was 102 BPM yesterday - slightly elevated. Please rest.", "is_read": True, "created_at": now - timedelta(days=1)},
        {"title": "Physical Therapy", "message": "Don't forget your home exercises today! Focus on shoulder stretches.", "is_read": False, "created_at": now - timedelta(hours=1)},
        {"title": "Medication Taken", "message": "Metformin morning dose marked as taken at 8:15 AM", "is_read": True, "created_at": now - timedelta(hours=4)},
        {"title": "Lab Results Available", "message": "Your latest blood panel results are ready. Ask your doctor for details.", "is_read": True, "created_at": now - timedelta(days=3)},
        {"title": "Battery Low", "message": "NovaCare rover battery at 15%. Please connect to charger.", "is_read": True, "created_at": now - timedelta(days=2)},
        {"title": "Weekly Health Summary", "message": "Your average heart rate this week: 74 BPM. SpO2 average: 97.8%%. Keep it up!", "is_read": True, "created_at": now - timedelta(days=1, hours=12)},
    ]
    
    for notif in notifications:
        cur.execute(
            "INSERT INTO notifications (id, recipient_id, title, message, is_read, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (str(uuid.uuid4()), "U001", notif["title"], notif["message"],
             notif["is_read"], notif["created_at"])
        )
    
    print(f"   + Seeded {len(notifications)} notifications")

def main():
    print("=" * 60)
    print("  NovaCare RAG Database Seeder")
    print("=" * 60)
    
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        seed_medications(cur)
        seed_vitals(cur)
        seed_appointments(cur)
        seed_health_conditions(cur)
        seed_allergies(cur)
        seed_medical_notes(cur)
        seed_emotion_tracking(cur)
        seed_notifications(cur)
        
        conn.commit()
        print("\n" + "=" * 60)
        print("  ✅ Database seeding complete!")
        print("=" * 60)
        
        # Summary
        tables = [
            ("rover_medications", "rover_id"),
            ("vital_signs", "rover_id"),
            ("appointments", "rover_id"),
            ("rover_health_conditions", "rover_id"),
            ("rover_allergies", "rover_id"),
            ("medical_notes", "rover_id"),
            ("emotion_tracking", "rover_id"),
            ("notifications", "recipient_id"),
        ]
        print("\n[Summary] Final row counts for RV001:")
        for t, col in tables:
            rid = 'U001' if t == 'notifications' else 'RV001'
            cur.execute(f"SELECT COUNT(*) FROM {t} WHERE {col} = %s", (rid,))
            print(f"   {t}: {cur.fetchone()[0]} rows")
        
    except Exception as e:
        print(f"\n❌ Seeding failed: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
