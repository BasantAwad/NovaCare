import pytest
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_rover_mood_and_sleep_and_caregiver_latest_vitals(db_conn, seed_entities, test_rover_id, test_caregiver_id):
    # mood_logs
    mood_id = "MD_TST_000000000000000000000000000001"
    await db_conn.execute(
        """
        INSERT INTO mood_logs (id, rover_id, date, mood, energy_level, anxiety_level, notes, emoji, created_at)
        VALUES ($1,$2,CURRENT_DATE,'happy','moderate',2,'Feeling good','😊', NOW())
        """,
        mood_id,
        test_rover_id,
    )
    mood = await db_conn.fetchrow("SELECT mood, energy_level FROM mood_logs WHERE id=$1", mood_id)
    assert mood["mood"] == "happy"
    assert mood["energy_level"] == "moderate"

    # sleep_logs
    sleep_id = "SL_TST_000000000000000000000000000001"
    await db_conn.execute(
        """
        INSERT INTO sleep_logs (id, rover_id, date, bed_time, wake_time, duration_hours, quality,
                                 deep_sleep_minutes, light_sleep_minutes, rem_sleep_minutes, awakenings, notes, created_at)
        VALUES ($1,$2,CURRENT_DATE,'22:30:00','06:00:00',7.5,'good',105,210,90,1,NULL,NOW())
        """,
        sleep_id,
        test_rover_id,
    )
    sleep = await db_conn.fetchrow("SELECT duration_hours, quality FROM sleep_logs WHERE id=$1", sleep_id)
    assert float(sleep["duration_hours"]) == 7.5
    assert sleep["quality"] == "good"

    # caregiver viewing latest vitals
    vital_id = "VT_TST_LATEST_0000000000000000000000000001"
    await db_conn.execute(
        """
        INSERT INTO vital_signs (id, rover_id, heart_rate, spo2, temperature, blood_pressure_systolic, blood_pressure_diastolic, measured_at)
        VALUES ($1,$2,80,99.0,36.7,118,76,NOW())
        """,
        vital_id,
        test_rover_id,
    )

    latest = await db_conn.fetchrow(
        """
        SELECT heart_rate, spo2, temperature
        FROM vital_signs
        WHERE rover_id=$1
        ORDER BY measured_at DESC
        LIMIT 1
        """,
        test_rover_id,
    )
    assert latest is not None
    assert latest["heart_rate"] == 80
    assert float(latest["spo2"]) == 99.0
    assert float(latest["temperature"]) == 36.7

