from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import date, datetime
from pydantic import BaseModel
from typing import List, Optional

from backend.database import get_db
from backend.models import MedicationSchedule, MedicationCatalog

router = APIRouter(prefix="/api/medication", tags=["Medications"])

class DoseStatusUpdate(BaseModel):
    status: str

@router.get("/today")
async def get_today_schedule(db: AsyncSession = Depends(get_db)):
    """Fetch today's medication schedule from real DB."""
    # For now, hardcode RV001 to get the integration working smoothly
    # A real auth middleware would extract this from JWT
    rover_id = 'RV001'
    today = date.today()
    
    query = (
        select(MedicationSchedule, MedicationCatalog)
        .join(MedicationCatalog, MedicationSchedule.medication_id == MedicationCatalog.drug_id)
        .where(MedicationSchedule.rover_id == rover_id)
        .where(MedicationSchedule.scheduled_date == today)
    )
    result = await db.execute(query)
    rows = result.all()
    
    doses = []
    for schedule, catalog in rows:
        doses.append({
            "dose_id": schedule.id,
            "name": catalog.brand_name,
            "dosage": schedule.dosage,
            "scheduled_time": schedule.scheduled_time.strftime("%H:%M"),
            "status": schedule.status
        })
        
    return {"success": True, "data": doses}

@router.patch("/{dose_id}/status")
async def update_dose_status(dose_id: str, payload: DoseStatusUpdate, db: AsyncSession = Depends(get_db)):
    """Update dose status in real DB."""
    query = select(MedicationSchedule).where(MedicationSchedule.id == dose_id)
    result = await db.execute(query)
    schedule = result.scalar_one_or_none()
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Dose not found")
        
    schedule.status = payload.status
    schedule.taken_at = datetime.utcnow()
    
    await db.commit()
    
    return {"success": True, "data": {"dose_id": dose_id, "status": schedule.status}}
