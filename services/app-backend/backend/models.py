from sqlalchemy import Column, String, Integer, BigInteger, Boolean, DateTime, Date, Time, Numeric, ForeignKey, text
from sqlalchemy.orm import relationship
from backend.database import Base
from datetime import datetime

class User(Base):
    __tablename__ = 'users'
    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    rovers = relationship("Rover", back_populates="user")

class Rover(Base):
    __tablename__ = 'rovers'
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, unique=True)
    user = relationship("User", back_populates="rovers")

class MedicationCatalog(Base):
    __tablename__ = 'medication_catalog'
    drug_id = Column(String(36), primary_key=True)
    brand_name = Column(String(255), nullable=False)
    generic_name = Column(String(255), nullable=False)
    active_strength = Column(Numeric(10, 4))
    strength_unit = Column(String(50))
    manufacturer = Column(String, default="Unknown")
    rx_status = Column(String, default="Unknown")
    therapeutic_class = Column(String, default="Unknown")
    smiles_structure = Column(String, default="Unknown")
    source_origin = Column(String, default="Unknown")

class RoverMedication(Base):
    __tablename__ = 'rover_medications'
    id = Column(String(36), primary_key=True)
    rover_id = Column(String(36), ForeignKey('rovers.id'), nullable=False)
    medication_id = Column(String(36), ForeignKey('medication_catalog.drug_id'), nullable=False)
    dosage = Column(String(100))
    frequency = Column(String(100))
    instructions = Column(String)
    catalog = relationship("MedicationCatalog")

class MedicationSchedule(Base):
    __tablename__ = 'medication_schedules'
    id = Column(String(36), primary_key=True)
    rover_id = Column(String(36), ForeignKey('rovers.id'), nullable=False)
    rover_medication_id = Column(String(36), ForeignKey('rover_medications.id'), nullable=False)
    medication_id = Column(String(36), ForeignKey('medication_catalog.drug_id'), nullable=False)
    scheduled_time = Column(Time, nullable=False)
    scheduled_date = Column(Date, nullable=False)
    status = Column(String, default='upcoming') # 'upcoming', 'due', 'taken', 'missed'
    taken_at = Column(DateTime(timezone=True))
    dosage = Column(String(100))
    
    catalog = relationship("MedicationCatalog")
    rover_medication = relationship("RoverMedication")

class VitalSign(Base):
    __tablename__ = 'vital_signs'
    id = Column(String(36), primary_key=True)
    rover_id = Column(String(36), ForeignKey('rovers.id'), nullable=False)
    heart_rate = Column(Integer)
    spo2 = Column(Numeric(5, 2))
    temperature = Column(Numeric(5, 2))
    measured_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class ActivityLog(Base):
    __tablename__ = 'activity_logs'
    id = Column(String(36), primary_key=True)
    rover_id = Column(String(36), ForeignKey('rovers.id'), nullable=False)
    type = Column(String)
    title = Column(String(200), nullable=False)
    description = Column(String)
    priority = Column(String, default='low')
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)

class RoverBatteryStatus(Base):
    __tablename__ = 'rover_battery_status'
    id = Column(String(36), primary_key=True)
    rover_id = Column(String(36), ForeignKey('rovers.id'), nullable=False)
    battery_percent = Column(Integer, nullable=False)
    is_charging = Column(Boolean, default=False)
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow)
