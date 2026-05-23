from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime
from pydantic import BaseModel
import uuid

from backend.database import get_db
from backend.models import VitalSign, RoverBatteryStatus, ActivityLog

router = APIRouter(prefix="/api/telemetry", tags=["Telemetry"])

class VitalsPayload(BaseModel):
    rover_id: str
    heart_rate: int
    spo2: float
    temperature: float

class BatteryPayload(BaseModel):
    rover_id: str
    battery_percent: int
    is_charging: bool

class ActivityPayload(BaseModel):
    rover_id: str
    type: str
    title: str
    description: str
    priority: str

@router.post("/vitals")
async def log_vitals(payload: VitalsPayload, db: AsyncSession = Depends(get_db)):
    record = VitalSign(
        id=str(uuid.uuid4()),
        rover_id=payload.rover_id,
        heart_rate=payload.heart_rate,
        spo2=payload.spo2,
        temperature=payload.temperature,
        measured_at=datetime.utcnow()
    )
    db.add(record)
    await db.commit()
    return {"success": True, "id": record.id}

@router.get("/vitals/{rover_id}")
async def get_vitals(rover_id: str, db: AsyncSession = Depends(get_db)):
    query = select(VitalSign).where(VitalSign.rover_id == rover_id).order_by(desc(VitalSign.measured_at)).limit(1)
    result = await db.execute(query)
    record = result.scalar_one_or_none()
    if not record:
        return {"success": False, "error": "No vitals found"}
    return {"success": True, "data": {
        "heart_rate": record.heart_rate,
        "spo2": float(record.spo2),
        "temperature": float(record.temperature)
    }}

@router.post("/battery")
async def log_battery(payload: BatteryPayload, db: AsyncSession = Depends(get_db)):
    record = RoverBatteryStatus(
        id=str(uuid.uuid4()),
        rover_id=payload.rover_id,
        battery_percent=payload.battery_percent,
        is_charging=payload.is_charging,
        recorded_at=datetime.utcnow()
    )
    db.add(record)
    await db.commit()
    return {"success": True, "id": record.id}

@router.post("/activity")
async def log_activity(payload: ActivityPayload, db: AsyncSession = Depends(get_db)):
    record = ActivityLog(
        id=str(uuid.uuid4()),
        rover_id=payload.rover_id,
        type=payload.type,
        title=payload.title,
        description=payload.description,
        priority=payload.priority,
        timestamp=datetime.utcnow()
    )
    db.add(record)
    await db.commit()
    return {"success": True, "id": record.id}

@router.get("/activity/{rover_id}")
async def get_activity(rover_id: str, db: AsyncSession = Depends(get_db)):
    query = select(ActivityLog).where(ActivityLog.rover_id == rover_id).order_by(desc(ActivityLog.timestamp)).limit(10)
    result = await db.execute(query)
    rows = result.scalars().all()
    return {"success": True, "data": [{
        "title": r.title,
        "description": r.description,
        "timestamp": r.timestamp.isoformat(),
        "type": r.type
    } for r in rows]}
