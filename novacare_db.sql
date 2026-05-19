-- ============================================================================
-- NovaCare Database - PostgreSQL Migration Script
-- ============================================================================
-- This script creates all tables, types, and constraints for NovaCare.
-- Run as a superuser or with CREATE privileges.
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 1. ENUM Types (replaces MySQL ENUM columns)
-- ----------------------------------------------------------------------------
CREATE TYPE gender_type AS ENUM ('male', 'female', 'other');
CREATE TYPE severity_type AS ENUM ('mild', 'moderate', 'severe');
CREATE TYPE user_role_type AS ENUM ('rover', 'caregiver', 'doctor');
CREATE TYPE allergy_category_type AS ENUM ('medication', 'food', 'material');
CREATE TYPE activity_log_type AS ENUM ('medication', 'navigation', 'conversation', 'alert', 'vital', 'system');
CREATE TYPE activity_priority AS ENUM ('low', 'medium', 'high');
CREATE TYPE sleep_quality AS ENUM ('poor', 'fair', 'good', 'excellent');
CREATE TYPE mood_enum AS ENUM ('very_sad', 'sad', 'neutral', 'happy', 'very_happy');
CREATE TYPE energy_level_enum AS ENUM ('very_low', 'low', 'moderate', 'high', 'very_high');
CREATE TYPE medication_schedule_status AS ENUM ('upcoming', 'due', 'taken', 'missed');
CREATE TYPE command_status AS ENUM ('pending', 'acked', 'completed', 'failed');
CREATE TYPE alert_status AS ENUM ('active', 'acknowledged', 'resolved');
CREATE TYPE alert_severity AS ENUM ('low', 'medium', 'high', 'critical');

-- ----------------------------------------------------------------------------
-- 2. Core Tables (existing schema, adapted to PostgreSQL)
-- ----------------------------------------------------------------------------

-- action_statuses
CREATE TABLE action_statuses (
    id VARCHAR(36) PRIMARY KEY,
    status_name VARCHAR(50) NOT NULL UNIQUE
);

-- action_types
CREATE TABLE action_types (
    id VARCHAR(36) PRIMARY KEY,
    action_name VARCHAR(100) NOT NULL UNIQUE
);

-- countries
CREATE TABLE countries (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    iso_code VARCHAR(5) NOT NULL UNIQUE,
    phone_code VARCHAR(10),
    currency_code VARCHAR(10)
);

-- addresses
CREATE TABLE addresses (
    id VARCHAR(36) PRIMARY KEY,
    street_address VARCHAR(255),
    city VARCHAR(100),
    state_province VARCHAR(100),
    postal_code VARCHAR(20),
    country_id VARCHAR(36) REFERENCES countries(id)
);

-- users
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255),
    google_id VARCHAR(255),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    profile_picture_url TEXT,
    is_email_verified BOOLEAN DEFAULT FALSE,
    email_verified_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- verification_statuses
CREATE TABLE verification_statuses (
    id VARCHAR(36) PRIMARY KEY,
    status_name VARCHAR(50) NOT NULL UNIQUE,
    display_label VARCHAR(100)
);

-- rovers
CREATE TABLE rovers (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL UNIQUE REFERENCES users(id),
    date_of_birth DATE,
    gender gender_type,
    address_id VARCHAR(36) REFERENCES addresses(id),
    primary_caregiver_id VARCHAR(36) -- references caregivers, defined later
);

-- caregivers
CREATE TABLE caregivers (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL UNIQUE REFERENCES users(id),
    phone_number VARCHAR(20),
    address_id VARCHAR(36) REFERENCES addresses(id),
    government_id_number VARCHAR(100),
    verification_status_id VARCHAR(36) REFERENCES verification_statuses(id)
);

-- Add foreign key to rovers.primary_caregiver_id after caregivers table exists
ALTER TABLE rovers ADD CONSTRAINT rovers_primary_caregiver_fk
    FOREIGN KEY (primary_caregiver_id) REFERENCES caregivers(id);

-- relationship_types
CREATE TABLE relationship_types (
    id VARCHAR(36) PRIMARY KEY,
    relationship VARCHAR(50) NOT NULL UNIQUE,
    description TEXT
);

-- caregiver_rover_assignments
CREATE TABLE caregiver_rover_assignments (
    id VARCHAR(36) PRIMARY KEY,
    caregiver_id VARCHAR(36) NOT NULL REFERENCES caregivers(id),
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    relationship_type_id VARCHAR(36) NOT NULL REFERENCES relationship_types(id),
    is_active BOOLEAN DEFAULT TRUE
);

-- doctors
CREATE TABLE doctors (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL UNIQUE REFERENCES users(id),
    medical_license_num VARCHAR(100),
    specialization_id VARCHAR(36),
    verification_status_id VARCHAR(36) REFERENCES verification_statuses(id)
);

-- specializations
CREATE TABLE specializations (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);

ALTER TABLE doctors ADD CONSTRAINT doctors_specialization_fk
    FOREIGN KEY (specialization_id) REFERENCES specializations(id);

-- user_roles
CREATE TABLE user_roles (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id),
    role user_role_type NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- ----------------------------------------------------------------------------
-- 3. Medication & Health Condition Tables
-- ----------------------------------------------------------------------------

-- medication_catalog (uses BIGSERIAL as PK)
CREATE TABLE medication_catalog (
    med_id BIGSERIAL PRIMARY KEY,
    brand_name VARCHAR(255) NOT NULL,
    generic_name VARCHAR(255) NOT NULL,
    manufacturer VARCHAR(255),
    therapeutic_class VARCHAR(255),
    is_controlled_substance BOOLEAN DEFAULT FALSE,
    product_external_id VARCHAR(50),
    product_type VARCHAR(100),
    route VARCHAR(100),
    marketing_category VARCHAR(100),
    substance_name TEXT,
    active_strength DECIMAL(10,4),
    strength_unit VARCHAR(50),
    pharm_classes TEXT,
    dea_schedule VARCHAR(10),
    dosage_form VARCHAR(100)
);

-- rover_medications (medication_id is BIGINT to match medication_catalog)
CREATE TABLE rover_medications (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    medication_id BIGINT NOT NULL REFERENCES medication_catalog(med_id),
    dosage VARCHAR(100),
    frequency VARCHAR(100),
    scheduled_time VARCHAR(50),
    instructions TEXT,
    prescribed_by VARCHAR(36) REFERENCES doctors(id),
    start_date DATE,
    end_date DATE,
    is_active BOOLEAN DEFAULT TRUE
);

-- medical_note_types
CREATE TABLE medical_note_types (
    id VARCHAR(36) PRIMARY KEY,
    note_type VARCHAR(50) NOT NULL UNIQUE
);

-- medical_notes
CREATE TABLE medical_notes (
    id VARCHAR(36) PRIMARY KEY,
    doctor_id VARCHAR(36) NOT NULL REFERENCES doctors(id),
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    note_type_id VARCHAR(36) REFERENCES medical_note_types(id),
    note_content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- health_conditions
CREATE TABLE health_conditions (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    icd10_code VARCHAR(20) UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- rover_health_conditions
CREATE TABLE rover_health_conditions (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    condition_id VARCHAR(36) NOT NULL REFERENCES health_conditions(id) ON DELETE CASCADE,
    severity severity_type DEFAULT 'moderate',
    notes TEXT
);

-- allergies
CREATE TABLE allergies (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category allergy_category_type NOT NULL,
    description TEXT
);

-- rover_allergies
CREATE TABLE rover_allergies (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    allergy_id VARCHAR(36) NOT NULL REFERENCES allergies(id),
    severity severity_type,
    is_active BOOLEAN DEFAULT TRUE
);

-- ----------------------------------------------------------------------------
-- 4. Appointments & Statuses
-- ----------------------------------------------------------------------------
CREATE TABLE appointment_statuses (
    id VARCHAR(36) PRIMARY KEY,
    status_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE appointment_types (
    id VARCHAR(36) PRIMARY KEY,
    type_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE appointments (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    doctor_id VARCHAR(36) REFERENCES doctors(id),
    appointment_type_id VARCHAR(36) REFERENCES appointment_types(id),
    status_id VARCHAR(36) REFERENCES appointment_statuses(id),
    scheduled_at TIMESTAMPTZ NOT NULL,
    notes TEXT
);

-- ----------------------------------------------------------------------------
-- 5. Vitals, Devices, Sleep, Hydration, Weight
-- ----------------------------------------------------------------------------
CREATE TABLE measurement_devices (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    device_name VARCHAR(100),
    device_type VARCHAR(50)
);

CREATE TABLE vital_signs (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    heart_rate INT,
    spo2 DECIMAL(5,2),
    temperature DECIMAL(5,2),
    blood_pressure_systolic INT,
    blood_pressure_diastolic INT,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    measurement_device_id VARCHAR(36) REFERENCES measurement_devices(id)
);

CREATE TABLE sleep_logs (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    bed_time TIME,
    wake_time TIME,
    duration_hours DECIMAL(4,2),
    quality sleep_quality,
    deep_sleep_minutes INT,
    light_sleep_minutes INT,
    rem_sleep_minutes INT,
    awakenings INT DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(rover_id, date)
);

CREATE TABLE hydration_logs (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    glasses INT NOT NULL DEFAULT 0,
    total_ml INT,
    goal_glasses INT NOT NULL DEFAULT 8,
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    UNIQUE(rover_id, date)
);

CREATE TABLE weight_logs (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    weight_kg DECIMAL(5,2) NOT NULL,
    weight_lbs DECIMAL(5,1) GENERATED ALWAYS AS (ROUND(weight_kg * 2.20462, 1)) STORED,
    target_weight_kg DECIMAL(5,2),
    bmi DECIMAL(4,1),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(rover_id, date)
);

-- ----------------------------------------------------------------------------
-- 6. AI, Emotion, Mood
-- ----------------------------------------------------------------------------
CREATE TABLE ai_interactions (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    conversation_id VARCHAR(36) NOT NULL,
    user_message TEXT,
    ai_response TEXT,
    emotion_detected VARCHAR(50),
    sentiment_score DECIMAL(3,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE emotion_tracking (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    date DATE NOT NULL,
    avg_sentiment DECIMAL(3,2),
    primary_emotion VARCHAR(50),
    distress_detected BOOLEAN DEFAULT FALSE
);

CREATE TABLE mood_logs (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    mood mood_enum NOT NULL,
    energy_level energy_level_enum,
    anxiety_level INT CHECK (anxiety_level BETWEEN 0 AND 10),
    notes TEXT,
    emoji VARCHAR(10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(rover_id, date)
);

-- ----------------------------------------------------------------------------
-- 7. Communication & Alerts (New tables for bidirectional communication)
-- ----------------------------------------------------------------------------
CREATE TABLE secure_messages (
    id VARCHAR(36) PRIMARY KEY,
    sender_id VARCHAR(36) NOT NULL REFERENCES users(id),
    recipient_id VARCHAR(36) NOT NULL REFERENCES users(id),
    message_body TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE rover_commands (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    command_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    status command_status NOT NULL DEFAULT 'pending',
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    executed_at TIMESTAMPTZ
);

CREATE INDEX idx_rover_commands_pending ON rover_commands(rover_id, status) WHERE status = 'pending';

CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rover_id VARCHAR(36) REFERENCES rovers(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    severity alert_severity NOT NULL,
    message TEXT,
    metadata JSONB,
    status alert_status NOT NULL DEFAULT 'active',
    acknowledged_by VARCHAR(36) REFERENCES users(id),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE configuration (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE data_sync_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    table_name VARCHAR(100) NOT NULL,
    last_sync_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_record_id VARCHAR(36),
    UNIQUE(rover_id, table_name)
);

CREATE TABLE file_attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id VARCHAR(36) NOT NULL, -- references users, rovers, doctors, etc.
    owner_type VARCHAR(50) NOT NULL, -- e.g., 'user', 'rover', 'doctor'
    file_url TEXT NOT NULL,
    file_name VARCHAR(255),
    mime_type VARCHAR(100),
    file_size INT,
    uploaded_by VARCHAR(36) REFERENCES users(id),
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- 8. Notifications & Audit
-- ----------------------------------------------------------------------------
CREATE TABLE notification_types (
    id VARCHAR(36) PRIMARY KEY,
    type_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE notifications (
    id VARCHAR(36) PRIMARY KEY,
    recipient_id VARCHAR(36) NOT NULL REFERENCES users(id),
    notification_type_id VARCHAR(36) REFERENCES notification_types(id),
    title VARCHAR(150),
    message TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE audit_logs (
    id VARCHAR(36) PRIMARY KEY,
    actor_id VARCHAR(36) NOT NULL REFERENCES users(id),
    action_type_id VARCHAR(36) NOT NULL REFERENCES action_types(id),
    resource_type VARCHAR(100),
    resource_id VARCHAR(36),
    old_value JSONB,
    new_value JSONB,
    action_status_id VARCHAR(36) NOT NULL REFERENCES action_statuses(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- 9. Emergency Contacts, Identity Docs, etc.
-- ----------------------------------------------------------------------------
CREATE TABLE emergency_contacts (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id),
    name VARCHAR(255) NOT NULL,
    relationship VARCHAR(100),
    phone_number VARCHAR(20),
    is_primary BOOLEAN DEFAULT FALSE
);

CREATE TABLE id_types (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    country_id VARCHAR(36) REFERENCES countries(id)
);

CREATE TABLE identity_documents (
    id VARCHAR(36) PRIMARY KEY,
    caregiver_id VARCHAR(36) NOT NULL REFERENCES caregivers(id),
    id_type_id VARCHAR(36) REFERENCES id_types(id),
    document_url TEXT NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE
);

CREATE TABLE medical_license_documents (
    id VARCHAR(36) PRIMARY KEY,
    doctor_id VARCHAR(36) NOT NULL REFERENCES doctors(id),
    document_url TEXT NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE
);

-- ----------------------------------------------------------------------------
-- 10. Session & Device Info
-- ----------------------------------------------------------------------------
CREATE TABLE device_info (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id),
    device_name VARCHAR(255),
    device_type VARCHAR(50),
    device_hash VARCHAR(255),
    is_trusted BOOLEAN DEFAULT FALSE
);

CREATE TABLE sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id),
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    device_info_id VARCHAR(36) REFERENCES device_info(id),
    ip_address VARCHAR(45),
    expires_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- ----------------------------------------------------------------------------
-- 11. Medication Schedules (depends on rover_medications)
-- ----------------------------------------------------------------------------
CREATE TABLE medication_schedules (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    rover_medication_id VARCHAR(36) NOT NULL REFERENCES rover_medications(id) ON DELETE CASCADE,
    medication_id BIGINT NOT NULL REFERENCES medication_catalog(med_id),
    dosage VARCHAR(100),
    frequency VARCHAR(100),
    scheduled_time TIME NOT NULL,
    scheduled_date DATE NOT NULL,
    instructions TEXT,
    status medication_schedule_status NOT NULL DEFAULT 'upcoming',
    taken_at TIMESTAMPTZ,
    prescribed_by VARCHAR(36) REFERENCES doctors(id) ON DELETE SET NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- 12. Activity Logs
-- ----------------------------------------------------------------------------
CREATE TABLE activity_logs (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    type activity_log_type NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    priority activity_priority NOT NULL DEFAULT 'low',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- ----------------------------------------------------------------------------
-- 13. Rover Battery Status
-- ----------------------------------------------------------------------------
CREATE TABLE rover_battery_status (
    id VARCHAR(36) PRIMARY KEY,
    rover_id VARCHAR(36) NOT NULL REFERENCES rovers(id) ON DELETE CASCADE,
    battery_percent SMALLINT NOT NULL CHECK (battery_percent BETWEEN 0 AND 100),
    is_charging BOOLEAN NOT NULL DEFAULT FALSE,
    estimated_remaining_minutes INT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- SEED DATA (adapted for PostgreSQL)
-- ============================================================================

-- Insert reference data
INSERT INTO action_statuses (id, status_name) VALUES
    ('S003', 'denied'), ('S002', 'failure'), ('S001', 'success');

INSERT INTO action_types (id, action_name) VALUES
    ('A002', 'data.view'), ('A003', 'emergency.alert'), ('A004', 'medication.take'), ('A001', 'user.login');

INSERT INTO countries (id, name, iso_code, phone_code, currency_code) VALUES
    ('C001', 'Egypt', 'EG', '+20', 'EGP'),
    ('C002', 'United States', 'US', '+1', 'USD'),
    ('C003', 'United Kingdom', 'GB', '+44', 'GBP');

INSERT INTO addresses (id, street_address, city, state_province, postal_code, country_id) VALUES
    ('AD001', '123 Corniche', 'Alexandria', 'Alexandria', '21500', 'C001'),
    ('AD002', '456 Maadi St', 'Cairo', 'Cairo', '11431', 'C001'),
    ('AD003', '789 Rehab City', 'New Cairo', 'Cairo', '11841', 'C001');

INSERT INTO users (id, email, hashed_password, google_id, first_name, last_name, is_email_verified, is_active, created_at, updated_at) VALUES
    ('U001', 'alex.smith@email.com', NULL, NULL, 'Alex', 'Smith', TRUE, TRUE, NOW(), NOW()),
    ('U002', 'sarah.jones@email.com', NULL, NULL, 'Sarah', 'Jones', TRUE, TRUE, NOW(), NOW()),
    ('U003', 'ahmed.doctor@clinic.eg', NULL, NULL, 'Ahmed', 'Shalaby', TRUE, TRUE, NOW(), NOW()),
    ('U004', 'maria.g@provider.net', NULL, NULL, 'Maria', 'Garcia', TRUE, TRUE, NOW(), NOW()),
    ('U005', 'youssef.b@care.com', NULL, NULL, 'Youssef', 'Bedair', TRUE, TRUE, NOW(), NOW()),
    ('U006', 'linda.v@rehab.com', NULL, NULL, 'Linda', 'Vazquez', TRUE, TRUE, NOW(), NOW()),
    ('U007', 'omar.k@proton.me', NULL, NULL, 'Omar', 'Khalil', TRUE, TRUE, NOW(), NOW()),
    ('U008', 'mona.r@family.org', NULL, NULL, 'Mona', 'Rashad', TRUE, TRUE, NOW(), NOW()),
    ('U009', 'zaki.neuro@medical.eg', NULL, NULL, 'Zaki', 'Mansour', TRUE, TRUE, NOW(), NOW()),
    ('U010', 'laila.n@gmail.com', NULL, NULL, 'Laila', 'Nour', TRUE, TRUE, NOW(), NOW()),
    ('U011', 'basant.dev@aiu.edu.eg', NULL, NULL, 'Basant', 'Awad', TRUE, TRUE, NOW(), NOW()),
    ('U012', 'nadira.s@aiu.edu.eg', NULL, NULL, 'Nadira', 'Sami', TRUE, TRUE, NOW(), NOW()),
    ('U013', 'nourine.y@aiu.edu.eg', NULL, NULL, 'Nourine', 'Yasser', TRUE, TRUE, NOW(), NOW()),
    ('U014', 'mohamed.m@aiu.edu.eg', NULL, NULL, 'Mohamed', 'Mosaad', TRUE, TRUE, NOW(), NOW()),
    ('U015', 'ramez.a@aiu.edu.eg', NULL, NULL, 'Ramez', 'Asaad', TRUE, TRUE, NOW(), NOW());

INSERT INTO verification_statuses (id, status_name, display_label) VALUES
    ('V001', 'pending', 'Verification Pending'),
    ('V002', 'verified', 'Identity Verified'),
    ('V003', 'rejected', 'Verification Rejected');

INSERT INTO caregivers (id, user_id, phone_number, address_id, government_id_number, verification_status_id) VALUES
    ('CG001', 'U002', '+201011112222', 'AD001', 'ID-998877', 'V002'),
    ('CG002', 'U005', '+201022223333', 'AD002', 'ID-665544', 'V002'),
    ('CG003', 'U008', '+201033334444', 'AD003', 'ID-332211', 'V001');

INSERT INTO rovers (id, user_id, date_of_birth, gender, address_id, primary_caregiver_id) VALUES
    ('RV001', 'U001', '1995-05-15', 'male', 'AD001', 'CG001'),
    ('RV002', 'U004', '1988-11-20', 'female', 'AD002', 'CG002'),
    ('RV003', 'U007', '2000-01-10', 'male', 'AD003', 'CG003');

INSERT INTO doctors (id, user_id, medical_license_num, specialization_id, verification_status_id) VALUES
    ('DOC001', 'U003', 'EGY-998877', NULL, 'V002'),
    ('DOC002', 'U006', 'EGY-112233', NULL, 'V002'),
    ('DOC003', 'U009', 'EGY-445566', NULL, 'V002');

INSERT INTO user_roles (id, user_id, role, is_active) VALUES
    ('RL001', 'U001', 'rover', TRUE),
    ('RL002', 'U002', 'caregiver', TRUE),
    ('RL003', 'U003', 'doctor', TRUE),
    ('RL004', 'U004', 'rover', TRUE),
    ('RL005', 'U005', 'caregiver', TRUE),
    ('RL006', 'U006', 'doctor', TRUE),
    ('RL007', 'U007', 'rover', TRUE),
    ('RL008', 'U008', 'caregiver', TRUE),
    ('RL009', 'U009', 'doctor', TRUE),
    ('RL010', 'U010', 'rover', TRUE),
    ('RL011', 'U011', 'caregiver', TRUE),
    ('RL012', 'U012', 'doctor', TRUE),
    ('RL013', 'U013', 'rover', TRUE),
    ('RL014', 'U014', 'caregiver', TRUE),
    ('RL015', 'U015', 'doctor', TRUE);

INSERT INTO specializations (id, name, description) VALUES
    ('SP01', 'Cardiology', 'Heart and blood vessel specialist'),
    ('SP02', 'Neurology', 'Brain and nervous system specialist'),
    ('SP15', 'Rehabilitation', 'Physical therapy and recovery');

-- Update doctors with specialization
UPDATE doctors SET specialization_id = 'SP01' WHERE id = 'DOC001';
UPDATE doctors SET specialization_id = 'SP15' WHERE id = 'DOC002';
UPDATE doctors SET specialization_id = 'SP02' WHERE id = 'DOC003';

INSERT INTO relationship_types (id, relationship, description) VALUES
    ('R001', 'parent', NULL),
    ('R002', 'spouse', NULL),
    ('R003', 'sibling', NULL),
    ('R004', 'child', NULL),
    ('R005', 'professional_caregiver', NULL);

INSERT INTO caregiver_rover_assignments (id, caregiver_id, rover_id, relationship_type_id, is_active) VALUES
    ('CRA001', 'CG001', 'RV001', 'R005', TRUE),
    ('CRA002', 'CG002', 'RV002', 'R005', TRUE),
    ('CRA003', 'CG003', 'RV003', 'R005', TRUE);

-- medication_catalog data (includes IDs used in rover_medications and medication_schedules)
INSERT INTO medication_catalog (med_id, brand_name, generic_name, manufacturer, therapeutic_class, is_controlled_substance, product_external_id, product_type, route, marketing_category, substance_name, active_strength, strength_unit, pharm_classes, dea_schedule, dosage_form) VALUES
    (1, 'Lipitor', 'Atorvastatin', 'Pfizer', 'Antihyperlipidemic', FALSE, '00071-0155-23', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ATORVASTATIN CALCIUM', 10.0000, 'mg', 'HMG-CoA Reductase Inhibitor', NULL, 'TABLET'),
    (2, 'Glucophage', 'Metformin', 'Bristol-Myers Squibb', 'Antidiabetic', FALSE, '00087-6060-05', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'METFORMIN HYDROCHLORIDE', 500.0000, 'mg', 'Biguanide', NULL, 'TABLET'),
    (3, 'Synthroid', 'Levothyroxine', 'AbbVie', 'Thyroid Hormone', FALSE, '00074-4341-13', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LEVOTHYROXINE SODIUM', 0.0500, 'mg', 'Thyroid Hormone Replacement', NULL, 'TABLET'),
    (4, 'Zestril', 'Lisinopril', 'AstraZeneca', 'Antihypertensive', FALSE, '00310-0130-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LISINOPRIL', 10.0000, 'mg', 'Angiotensin Converting Enzyme Inhibitor', NULL, 'TABLET'),
    (5, 'Norvasc', 'Amlodipine', 'Pfizer', 'Antihypertensive', FALSE, '00069-1520-68', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'AMLODIPINE BESYLATE', 5.0000, 'mg', 'Calcium Channel Blocker', NULL, 'TABLET'),
    (6, 'Lopressor', 'Metoprolol', 'Novartis', 'Beta-Blocker', FALSE, '00078-0457-05', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'METOPROLOL TARTRATE', 50.0000, 'mg', 'Beta-1 Adrenergic Blocker', NULL, 'TABLET'),
    (7, 'ProAir HFA', 'Albuterol', 'Teva', 'Bronchodilator', FALSE, '00591-2325-04', 'HUMAN PRESCRIPTION DRUG', 'RESPIRATORY (INHALATION)', 'NDA', 'ALBUTEROL SULFATE', 0.0900, 'mg/actuation', 'Beta-2 Adrenergic Agonist', NULL, 'AEROSOL, METERED'),
    (8, 'Cozaar', 'Losartan', 'Merck', 'Antihypertensive', FALSE, '00006-0951-54', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LOSARTAN POTASSIUM', 50.0000, 'mg', 'Angiotensin II Receptor Antagonist', NULL, 'TABLET'),
    (9, 'Neurontin', 'Gabapentin', 'Pfizer', 'Anticonvulsant', FALSE, '00071-0801-24', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'GABAPENTIN', 300.0000, 'mg', 'Gamma-Aminobutyric Acid Analog', NULL, 'CAPSULE'),
    (10, 'Prilosec', 'Omeprazole', 'Procter & Gamble', 'Gastrointestinal', FALSE, '00006-0397-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'OMEPRAZOLE', 20.0000, 'mg', 'Proton Pump Inhibitor', NULL, 'CAPSULE, DELAYED RELEASE'),
    (45, 'Ecotrin', 'Aspirin', 'Medtech', 'Analgesic', FALSE, '00135-0158-01', 'HUMAN OTC DRUG', 'ORAL', 'OTC MONOGRAPH FINAL', 'ASPIRIN', 325.0000, 'mg', 'Salicylate', NULL, 'TABLET, ENTERIC COATED');
-- rover_medications (using correct BIGINT medication_id)
INSERT INTO rover_medications (id, rover_id, medication_id, dosage, frequency, scheduled_time, instructions, prescribed_by, start_date, end_date, is_active) VALUES
    ('RM001', 'RV001', 4, '10mg', 'Once daily', '08:00', 'Take on empty stomach', 'DOC001', '2025-12-01', NULL, TRUE),
    ('RM002', 'RV001', 2, '500mg', 'Twice daily', '12:00', 'Take with meals', 'DOC001', '2025-11-15', NULL, TRUE),
    ('RM003', 'RV001', 45, '81mg', 'Once daily', '18:00', 'Take with water', 'DOC001', '2025-10-01', NULL, TRUE),
    ('RM004', 'RV002', 6, '50mg', 'Once daily', '09:00', 'Take in the morning', 'DOC002', '2025-11-01', NULL, TRUE),
    ('RM005', 'RV002', 1, '10mg', 'Once daily', '21:00', 'Take at bedtime', 'DOC002', '2025-09-15', NULL, TRUE),
    ('RM006', 'RV003', 9, '300mg', 'Three times daily', '08:00', 'Take with food', 'DOC003', '2025-10-01', NULL, TRUE),
    ('RM007', 'RV003', 10, '20mg', 'Once daily', '07:00', 'Take before breakfast', 'DOC003', '2025-08-15', NULL, TRUE);

-- vital_signs with blood pressure
INSERT INTO vital_signs (id, rover_id, heart_rate, spo2, temperature, blood_pressure_systolic, blood_pressure_diastolic, measured_at) VALUES
    ('VT001', 'RV001', 72, 98.50, 36.60, 120, 78, '2026-04-14 18:50:07+00'),
    ('VT002', 'RV001', 75, 98.00, 36.70, 122, 80, '2026-04-14 17:50:07+00'),
    ('VT003', 'RV001', 80, 97.50, 36.80, 118, 76, '2026-04-14 16:50:07+00'),
    ('VT004', 'RV002', 68, 99.00, 36.50, 115, 72, '2026-04-14 18:50:07+00'),
    ('VT005', 'RV002', 70, 99.00, 36.50, 116, 74, '2026-04-14 17:50:07+00'),
    ('VT006', 'RV002', 72, 98.50, 36.60, 118, 76, '2026-04-14 16:50:07+00'),
    ('VT007', 'RV003', 90, 96.00, 37.10, 135, 88, '2026-04-14 18:50:07+00'),
    ('VT008', 'RV003', 92, 95.50, 37.20, 138, 90, '2026-04-14 17:50:07+00'),
    ('VT009', 'RV003', 88, 96.50, 37.00, 132, 86, '2026-04-14 16:50:07+00');

-- medication_schedules (today and historical)
INSERT INTO medication_schedules (id, rover_id, rover_medication_id, medication_id, dosage, frequency, scheduled_time, scheduled_date, instructions, status, taken_at, prescribed_by, is_active) VALUES
    ('MS001', 'RV001', 'RM001', 4, '10mg', 'Once daily', '08:00:00', CURRENT_DATE, 'Take on empty stomach', 'taken', NOW() - INTERVAL '2 hours', 'DOC001', TRUE),
    ('MS002', 'RV001', 'RM002', 2, '500mg', 'Twice daily', '12:00:00', CURRENT_DATE, 'Take with meals', 'taken', NOW() - INTERVAL '1 hour', 'DOC001', TRUE),
    ('MS003', 'RV001', 'RM003', 45, '81mg', 'Once daily', '18:00:00', CURRENT_DATE, 'Take with water', 'upcoming', NULL, 'DOC001', TRUE),
    ('MS004', 'RV001', 'RM002', 2, '500mg', 'Twice daily', '20:00:00', CURRENT_DATE, 'Take with meals', 'upcoming', NULL, 'DOC001', TRUE),
    ('MS005', 'RV001', 'RM001', 4, '10mg', 'Once daily', '08:00:00', CURRENT_DATE - 1, 'Take on empty stomach', 'taken', (NOW() - INTERVAL '1 day' + INTERVAL '10 minutes'), 'DOC001', TRUE),
    ('MS006', 'RV001', 'RM002', 2, '500mg', 'Twice daily', '12:00:00', CURRENT_DATE - 1, 'Take with meals', 'taken', (NOW() - INTERVAL '1 day'), 'DOC001', TRUE),
    ('MS007', 'RV001', 'RM003', 45, '81mg', 'Once daily', '18:00:00', CURRENT_DATE - 1, 'Take with water', 'taken', (NOW() - INTERVAL '1 day' + INTERVAL '5 minutes'), 'DOC001', TRUE),
    ('MS008', 'RV001', 'RM002', 2, '500mg', 'Twice daily', '20:00:00', CURRENT_DATE - 1, 'Take with meals', 'missed', NULL, 'DOC001', TRUE),
    ('MS009', 'RV002', 'RM004', 6, '50mg', 'Once daily', '09:00:00', CURRENT_DATE, 'Take in the morning', 'taken', NOW() - INTERVAL '3 hours', 'DOC002', TRUE),
    ('MS010', 'RV002', 'RM005', 1, '10mg', 'Once daily', '21:00:00', CURRENT_DATE, 'Take at bedtime', 'upcoming', NULL, 'DOC002', TRUE),
    ('MS011', 'RV003', 'RM006', 9, '300mg', 'Three times daily', '08:00:00', CURRENT_DATE, 'Take with food', 'taken', NOW() - INTERVAL '4 hours', 'DOC003', TRUE),
    ('MS012', 'RV003', 'RM007', 10, '20mg', 'Once daily', '07:00:00', CURRENT_DATE, 'Take before breakfast', 'taken', NOW() - INTERVAL '5 hours', 'DOC003', TRUE),
    ('MS013', 'RV003', 'RM006', 9, '300mg', 'Three times daily', '13:00:00', CURRENT_DATE, 'Take with food', 'due', NULL, 'DOC003', TRUE),
    ('MS014', 'RV003', 'RM006', 9, '300mg', 'Three times daily', '19:00:00', CURRENT_DATE, 'Take with food', 'upcoming', NULL, 'DOC003', TRUE);

-- activity_logs
INSERT INTO activity_logs (id, rover_id, type, title, description, priority, timestamp) VALUES
    ('ACT001', 'RV001', 'medication', 'Medication Taken', 'Took Lisinopril (10mg) — morning dose', 'low', NOW() - INTERVAL '2 hours'),
    ('ACT002', 'RV001', 'medication', 'Medication Taken', 'Took Metformin (500mg) — afternoon dose', 'low', NOW() - INTERVAL '4 hours'),
    ('ACT003', 'RV001', 'navigation', 'Navigation Complete', 'Navigated from Bedroom to Kitchen safely', 'low', NOW() - INTERVAL '5 hours'),
    ('ACT004', 'RV001', 'conversation', 'Morning Conversation', 'Had a 15-minute conversation with NovaCare about the weather', 'low', NOW() - INTERVAL '6 hours'),
    ('ACT005', 'RV001', 'vital', 'Vitals Recorded', 'Heart rate 72 bpm, SpO2 98.5%, Temperature 36.6°C', 'low', NOW() - INTERVAL '7 hours'),
    ('ACT006', 'RV001', 'alert', 'Low Battery Warning', 'Rover battery dropped to 15% — now charging', 'high', NOW() - INTERVAL '10 hours'),
    ('ACT007', 'RV001', 'navigation', 'Navigation Complete', 'Navigated from Living Room to Bathroom', 'low', NOW() - INTERVAL '12 hours'),
    ('ACT008', 'RV001', 'medication', 'Medication Missed', 'Missed Metformin (500mg) evening dose yesterday', 'medium', NOW() - INTERVAL '18 hours'),
    ('ACT009', 'RV001', 'alert', 'Fall Detection', 'Possible fall detected in hallway — Patient confirmed OK', 'high', NOW() - INTERVAL '24 hours'),
    ('ACT010', 'RV001', 'medication', 'Medication Taken', 'Took Aspirin (81mg) — evening dose', 'low', NOW() - INTERVAL '26 hours'),
    ('ACT011', 'RV002', 'medication', 'Medication Taken', 'Took Metoprolol (50mg) — morning dose', 'low', NOW() - INTERVAL '3 hours'),
    ('ACT012', 'RV002', 'conversation', 'Video Call', 'Had a 10-minute video call with caregiver Youssef', 'low', NOW() - INTERVAL '5 hours'),
    ('ACT013', 'RV002', 'vital', 'Vitals Recorded', 'Heart rate 68 bpm, SpO2 99.0%, Temperature 36.5°C', 'low', NOW() - INTERVAL '7 hours'),
    ('ACT014', 'RV002', 'navigation', 'Navigation Complete', 'Navigated to Garden area for fresh air', 'low', NOW() - INTERVAL '8 hours'),
    ('ACT015', 'RV002', 'alert', 'Appointment Reminder', 'Upcoming appointment with Dr. Vazquez tomorrow at 10 AM', 'medium', NOW() - INTERVAL '12 hours');

-- sleep_logs
INSERT INTO sleep_logs (id, rover_id, date, bed_time, wake_time, duration_hours, quality, deep_sleep_minutes, light_sleep_minutes, rem_sleep_minutes, awakenings, notes) VALUES
    ('SL001', 'RV001', CURRENT_DATE - 1, '22:30:00', '06:00:00', 7.5, 'good', 105, 210, 90, 1, NULL),
    ('SL002', 'RV001', CURRENT_DATE - 2, '23:00:00', '06:30:00', 7.5, 'good', 100, 220, 85, 0, NULL),
    ('SL003', 'RV001', CURRENT_DATE - 3, '23:15:00', '06:00:00', 6.75, 'fair', 80, 200, 70, 2, 'Woke up briefly twice'),
    ('SL004', 'RV001', CURRENT_DATE - 4, '22:00:00', '06:30:00', 8.5, 'excellent', 130, 230, 110, 0, NULL),
    ('SL008', 'RV002', CURRENT_DATE - 1, '21:30:00', '05:30:00', 8.0, 'excellent', 120, 230, 100, 0, NULL),
    ('SL009', 'RV002', CURRENT_DATE - 2, '22:00:00', '06:00:00', 8.0, 'good', 110, 220, 95, 1, NULL),
    ('SL011', 'RV003', CURRENT_DATE - 1, '00:00:00', '07:30:00', 7.5, 'fair', 85, 210, 75, 2, 'Slight discomfort'),
    ('SL012', 'RV003', CURRENT_DATE - 2, '23:30:00', '07:00:00', 7.5, 'good', 100, 220, 90, 1, NULL);

-- hydration_logs
INSERT INTO hydration_logs (id, rover_id, date, glasses, total_ml, goal_glasses) VALUES
    ('HL001', 'RV001', CURRENT_DATE, 6, 1500, 8),
    ('HL002', 'RV001', CURRENT_DATE - 1, 8, 2000, 8),
    ('HL003', 'RV001', CURRENT_DATE - 2, 7, 1750, 8),
    ('HL008', 'RV002', CURRENT_DATE, 7, 1750, 8),
    ('HL009', 'RV002', CURRENT_DATE - 1, 8, 2000, 8),
    ('HL011', 'RV003', CURRENT_DATE, 4, 1000, 8),
    ('HL012', 'RV003', CURRENT_DATE - 1, 5, 1250, 8);

-- weight_logs
INSERT INTO weight_logs (id, rover_id, date, weight_kg, target_weight_kg, bmi) VALUES
    ('WL001', 'RV001', CURRENT_DATE, 65.8, 65.0, 22.5),
    ('WL002', 'RV001', CURRENT_DATE - 7, 66.0, 65.0, 22.6),
    ('WL005', 'RV002', CURRENT_DATE, 57.2, 55.0, 21.8),
    ('WL006', 'RV002', CURRENT_DATE - 7, 57.5, 55.0, 21.9),
    ('WL007', 'RV003', CURRENT_DATE, 72.0, 70.0, 23.1),
    ('WL008', 'RV003', CURRENT_DATE - 7, 72.5, 70.0, 23.3);

-- mood_logs
INSERT INTO mood_logs (id, rover_id, date, mood, energy_level, anxiety_level, notes, emoji) VALUES
    ('MD001', 'RV001', CURRENT_DATE, 'happy', 'moderate', 2, 'Feeling good today', '😊'),
    ('MD002', 'RV001', CURRENT_DATE - 1, 'happy', 'high', 1, 'Great day, went outside', '😄'),
    ('MD003', 'RV001', CURRENT_DATE - 2, 'neutral', 'moderate', 3, 'Average day', '😐'),
    ('MD008', 'RV002', CURRENT_DATE, 'happy', 'moderate', 2, 'Calm and relaxed', '😊'),
    ('MD009', 'RV002', CURRENT_DATE - 1, 'very_happy', 'high', 0, 'Daughter came to visit', '😄'),
    ('MD011', 'RV003', CURRENT_DATE, 'neutral', 'low', 4, 'Some discomfort but managing', '😐'),
    ('MD012', 'RV003', CURRENT_DATE - 1, 'happy', 'moderate', 3, 'Physical therapy went well', '😊');

-- emotion_tracking
INSERT INTO emotion_tracking (id, rover_id, date, avg_sentiment, primary_emotion, distress_detected) VALUES
    ('ET001', 'RV001', CURRENT_DATE, 0.5, 'Happy', FALSE),
    ('ET002', 'RV001', CURRENT_DATE - 1, 0.3, 'Neutral', FALSE),
    ('ET003', 'RV001', CURRENT_DATE - 2, -0.1, 'Tired', FALSE),
    ('ET004', 'RV002', CURRENT_DATE, 0.6, 'Relieved', FALSE),
    ('ET005', 'RV002', CURRENT_DATE - 1, 0.1, 'Neutral', FALSE),
    ('ET006', 'RV003', CURRENT_DATE, -0.3, 'Pain', TRUE),
    ('ET007', 'RV003', CURRENT_DATE - 1, 0.4, 'Satisfied', FALSE);

-- rover_battery_status
INSERT INTO rover_battery_status (id, rover_id, battery_percent, is_charging, estimated_remaining_minutes, recorded_at) VALUES
    ('RBS001', 'RV001', 85, FALSE, 420, NOW()),
    ('RBS002', 'RV001', 78, FALSE, 380, NOW() - INTERVAL '1 hour'),
    ('RBS003', 'RV001', 15, TRUE, 60, NOW() - INTERVAL '10 hours'),
    ('RBS005', 'RV002', 92, FALSE, 460, NOW()),
    ('RBS006', 'RV002', 88, FALSE, 440, NOW() - INTERVAL '2 hours'),
    ('RBS007', 'RV003', 64, FALSE, 310, NOW()),
    ('RBS008', 'RV003', 70, FALSE, 340, NOW() - INTERVAL '2 hours');

-- emergency_contacts
INSERT INTO emergency_contacts (id, rover_id, name, relationship, phone_number, is_primary) VALUES
    ('EC001', 'RV001', 'Sarah Jones', 'Sister', '+201011112222', TRUE),
    ('EC002', 'RV001', 'John Smith', 'Father', '+201011112223', FALSE),
    ('EC004', 'RV002', 'Youssef Bedair', 'Spouse', '+201022223333', TRUE),
    ('EC007', 'RV003', 'Mona Rashad', 'Guardian', '+201033334444', TRUE);

-- medical_note_types
INSERT INTO medical_note_types (id, note_type) VALUES
    ('NT01', 'Diagnosis'), ('NT02', 'Progress Note'), ('NT10', 'Initial Assessment');

-- medical_notes
INSERT INTO medical_notes (id, doctor_id, rover_id, note_type_id, note_content, created_at) VALUES
    ('MN01', 'DOC001', 'RV001', 'NT01', 'Patient showing early signs of hypertension.', NOW()),
    ('MN02', 'DOC002', 'RV001', 'NT02', 'Recovery from physical therapy is on track.', NOW()),
    ('MN03', 'DOC003', 'RV002', 'NT10', 'Neurological assessment complete; no tremors detected.', NOW());

-- audit_logs
INSERT INTO audit_logs (id, actor_id, action_type_id, resource_type, resource_id, old_value, new_value, action_status_id, created_at) VALUES
    ('AL001', 'U001', 'A001', 'user', 'U001', NULL, NULL, 'S001', NOW()),
    ('AL002', 'U003', 'A002', 'vital_signs', 'RV001', NULL, NULL, 'S001', NOW()),
    ('AL003', 'U002', 'A002', 'vital_signs', 'RV001', NULL, NULL, 'S001', NOW());

-- ai_interactions
INSERT INTO ai_interactions (id, rover_id, conversation_id, user_message, ai_response, emotion_detected, sentiment_score, created_at) VALUES
    ('AI001', 'RV001', 'CONV001', 'How is my heart rate?', 'It is stable at 72 bpm.', 'Neutral', 0.10, NOW() - INTERVAL '2 days'),
    ('AI002', 'RV001', 'CONV001', 'I feel a bit tired.', 'I recommend resting. I have alerted Sarah.', 'Tired', -0.20, NOW() - INTERVAL '2 days');

-- sample rover_command (to show how it works)
INSERT INTO rover_commands (rover_id, command_type, payload, status, created_at) VALUES
    ('RV001', 'navigate', '{"destination": "kitchen"}', 'pending', NOW()),
    ('RV001', 'play_music', '{"playlist": "relax"}', 'completed', NOW() - INTERVAL '1 day');

-- sample alert
INSERT INTO alerts (rover_id, alert_type, severity, message, metadata, status, created_at) VALUES
    ('RV001', 'tachycardia', 'high', 'Heart rate exceeded 100 bpm', '{"heart_rate": 102, "measured_at": "2026-05-19T10:00:00Z"}', 'active', NOW());

-- configuration
INSERT INTO configuration (key, value, description) VALUES
    ('voice_volume', '{"default": 5, "max": 10}', 'Rover voice volume settings'),
    ('auto_navigation', '{"enabled": true, "avoid_stairs": true}', 'Auto-navigation preferences');

-- data_sync_checkpoints
INSERT INTO data_sync_checkpoints (rover_id, table_name, last_sync_at, last_record_id) VALUES
    ('RV001', 'vital_signs', NOW(), 'VT001'),
    ('RV002', 'activity_logs', NOW(), 'ACT011');

-- ----------------------------------------------------------------------------
-- Indexes for performance
-- ----------------------------------------------------------------------------
CREATE INDEX idx_vitals_rover_time ON vital_signs(rover_id, measured_at);
CREATE INDEX idx_med_sched_date ON medication_schedules(scheduled_date);
CREATE INDEX idx_med_sched_rover_status ON medication_schedules(rover_id, status);
CREATE INDEX idx_activity_rover_time ON activity_logs(rover_id, timestamp);
CREATE INDEX idx_alerts_rover_status ON alerts(rover_id, status);
CREATE INDEX idx_commands_rover_status ON rover_commands(rover_id, status);
CREATE INDEX idx_sleep_rover_date ON sleep_logs(rover_id, date);
CREATE INDEX idx_hydration_rover_date ON hydration_logs(rover_id, date);
CREATE INDEX idx_weight_rover_date ON weight_logs(rover_id, date);
CREATE INDEX idx_mood_rover_date ON mood_logs(rover_id, date);
CREATE INDEX idx_emotion_rover_date ON emotion_tracking(rover_id, date);
CREATE INDEX idx_audit_actor ON audit_logs(actor_id);
CREATE INDEX idx_audit_created ON audit_logs(created_at);
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(access_token);
CREATE INDEX idx_messages_recipient_unread ON secure_messages(recipient_id, is_read) WHERE is_read = FALSE;

COMMIT;