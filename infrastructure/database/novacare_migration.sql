-- ============================================================================
-- NovaCare Database Migration — New & Altered Tables
-- ============================================================================
-- Run AFTER the base novacare_db.sql has been imported.
-- This script:
--   1. ALTERs existing tables that need extra columns
--   2. CREATEs new tables that did not exist
--   3. INSERTs realistic seed data for all new tables
-- ============================================================================

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";

-- ============================================================================
-- PART 1: ALTER EXISTING TABLES
-- ============================================================================

-- -------------------------------------------------------
-- 1a. vital_signs — add blood_pressure columns
--     (spo2 already exists but frontend calls it blood_oxygen)
-- -------------------------------------------------------
ALTER TABLE `vital_signs`
  ADD COLUMN `blood_pressure_systolic` int(11) DEFAULT NULL AFTER `temperature`,
  ADD COLUMN `blood_pressure_diastolic` int(11) DEFAULT NULL AFTER `blood_pressure_systolic`;

-- Update existing vital_signs records with blood-pressure values
UPDATE `vital_signs` SET `blood_pressure_systolic` = 120, `blood_pressure_diastolic` = 78 WHERE `id` = 'VT001';
UPDATE `vital_signs` SET `blood_pressure_systolic` = 122, `blood_pressure_diastolic` = 80 WHERE `id` = 'VT002';
UPDATE `vital_signs` SET `blood_pressure_systolic` = 118, `blood_pressure_diastolic` = 76 WHERE `id` = 'VT003';
UPDATE `vital_signs` SET `blood_pressure_systolic` = 115, `blood_pressure_diastolic` = 72 WHERE `id` = 'VT004';
UPDATE `vital_signs` SET `blood_pressure_systolic` = 116, `blood_pressure_diastolic` = 74 WHERE `id` = 'VT005';
UPDATE `vital_signs` SET `blood_pressure_systolic` = 118, `blood_pressure_diastolic` = 76 WHERE `id` = 'VT006';
UPDATE `vital_signs` SET `blood_pressure_systolic` = 135, `blood_pressure_diastolic` = 88 WHERE `id` = 'VT007';
UPDATE `vital_signs` SET `blood_pressure_systolic` = 138, `blood_pressure_diastolic` = 90 WHERE `id` = 'VT008';
UPDATE `vital_signs` SET `blood_pressure_systolic` = 132, `blood_pressure_diastolic` = 86 WHERE `id` = 'VT009';

-- -------------------------------------------------------
-- 1b. rover_medications — add schedule-related columns
--     (table exists but lacks time, instructions, prescribed_by, end_date)
-- -------------------------------------------------------
ALTER TABLE `rover_medications`
  ADD COLUMN `scheduled_time` varchar(50) DEFAULT NULL AFTER `frequency`,
  ADD COLUMN `instructions` text DEFAULT NULL AFTER `scheduled_time`,
  ADD COLUMN `prescribed_by` char(36) DEFAULT NULL AFTER `instructions`,
  ADD COLUMN `end_date` date DEFAULT NULL AFTER `start_date`;

-- ============================================================================
-- PART 2: CREATE NEW TABLES
-- ============================================================================

-- -------------------------------------------------------
-- 2a. medication_schedules — daily dose tracking
-- -------------------------------------------------------
CREATE TABLE `medication_schedules` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `rover_medication_id` char(36) NOT NULL COMMENT 'FK to rover_medications',
  `medication_id` char(36) NOT NULL COMMENT 'FK to medication_catalog (via rover_medications)',
  `dosage` varchar(100) DEFAULT NULL,
  `frequency` varchar(100) DEFAULT NULL,
  `scheduled_time` time NOT NULL,
  `scheduled_date` date NOT NULL,
  `instructions` text DEFAULT NULL,
  `status` enum('upcoming','due','taken','missed') NOT NULL DEFAULT 'upcoming',
  `taken_at` timestamp NULL DEFAULT NULL,
  `prescribed_by` char(36) DEFAULT NULL COMMENT 'FK to doctors',
  `is_active` tinyint(1) NOT NULL DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_ms_rover` (`rover_id`),
  KEY `idx_ms_rover_med` (`rover_medication_id`),
  KEY `idx_ms_date` (`scheduled_date`),
  KEY `idx_ms_status` (`status`),
  CONSTRAINT `ms_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE,
  CONSTRAINT `ms_rover_med_fk` FOREIGN KEY (`rover_medication_id`) REFERENCES `rover_medications` (`id`) ON DELETE CASCADE,
  CONSTRAINT `ms_doctor_fk` FOREIGN KEY (`prescribed_by`) REFERENCES `doctors` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------------------------------------
-- 2b. activity_logs — unified activity timeline
-- -------------------------------------------------------
CREATE TABLE `activity_logs` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `type` enum('medication','navigation','conversation','alert','vital','system') NOT NULL,
  `title` varchar(200) NOT NULL,
  `description` text DEFAULT NULL,
  `priority` enum('low','medium','high') NOT NULL DEFAULT 'low',
  `timestamp` timestamp NOT NULL DEFAULT current_timestamp(),
  `metadata` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`metadata`)),
  PRIMARY KEY (`id`),
  KEY `idx_al_rover` (`rover_id`),
  KEY `idx_al_type` (`type`),
  KEY `idx_al_ts` (`timestamp`),
  CONSTRAINT `al_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------------------------------------
-- 2c. sleep_logs — nightly sleep tracking
-- -------------------------------------------------------
CREATE TABLE `sleep_logs` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `date` date NOT NULL,
  `bed_time` time DEFAULT NULL,
  `wake_time` time DEFAULT NULL,
  `duration_hours` decimal(4,2) DEFAULT NULL,
  `quality` enum('poor','fair','good','excellent') DEFAULT NULL,
  `deep_sleep_minutes` int(11) DEFAULT NULL,
  `light_sleep_minutes` int(11) DEFAULT NULL,
  `rem_sleep_minutes` int(11) DEFAULT NULL,
  `awakenings` int(11) DEFAULT 0,
  `notes` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_sleep_rover_date` (`rover_id`, `date`),
  KEY `idx_sl_rover` (`rover_id`),
  CONSTRAINT `sl_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------------------------------------
-- 2d. hydration_logs — daily water intake tracking
-- -------------------------------------------------------
CREATE TABLE `hydration_logs` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `date` date NOT NULL,
  `glasses` int(11) NOT NULL DEFAULT 0 COMMENT 'Number of ~250ml glasses',
  `total_ml` int(11) DEFAULT NULL,
  `goal_glasses` int(11) NOT NULL DEFAULT 8,
  `logged_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `notes` text DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_hydration_rover_date` (`rover_id`, `date`),
  KEY `idx_hl_rover` (`rover_id`),
  CONSTRAINT `hl_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------------------------------------
-- 2e. weight_logs — body weight tracking
-- -------------------------------------------------------
CREATE TABLE `weight_logs` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `date` date NOT NULL,
  `weight_kg` decimal(5,2) NOT NULL,
  `weight_lbs` decimal(5,1) GENERATED ALWAYS AS (ROUND(`weight_kg` * 2.20462, 1)) VIRTUAL,
  `target_weight_kg` decimal(5,2) DEFAULT NULL,
  `bmi` decimal(4,1) DEFAULT NULL,
  `notes` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_weight_rover_date` (`rover_id`, `date`),
  KEY `idx_wl_rover` (`rover_id`),
  CONSTRAINT `wl_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------------------------------------
-- 2f. rover_battery_status — rover device battery tracking
-- -------------------------------------------------------
CREATE TABLE `rover_battery_status` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `battery_percent` tinyint(3) UNSIGNED NOT NULL,
  `is_charging` tinyint(1) NOT NULL DEFAULT 0,
  `estimated_remaining_minutes` int(11) DEFAULT NULL,
  `recorded_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_rbs_rover` (`rover_id`),
  KEY `idx_rbs_ts` (`recorded_at`),
  CONSTRAINT `rbs_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------------------------------------------------------
-- 2g. mood_logs — daily mood self-reports  
--     (supplements the AI-driven emotion_tracking table)
-- -------------------------------------------------------
CREATE TABLE `mood_logs` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `date` date NOT NULL,
  `mood` enum('very_sad','sad','neutral','happy','very_happy') NOT NULL,
  `energy_level` enum('very_low','low','moderate','high','very_high') DEFAULT NULL,
  `anxiety_level` tinyint(3) UNSIGNED DEFAULT NULL COMMENT '0-10 scale',
  `notes` text DEFAULT NULL,
  `emoji` varchar(10) DEFAULT NULL COMMENT 'Emoji used e.g. 😊',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_mood_rover_date` (`rover_id`, `date`),
  KEY `idx_ml_rover` (`rover_id`),
  CONSTRAINT `ml_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ============================================================================
-- PART 3: SEED DATA
-- ============================================================================

-- -------------------------------------------------------
-- 3a. rover_medications — populate for existing rovers
-- -------------------------------------------------------
INSERT INTO `rover_medications` (`id`, `rover_id`, `medication_id`, `dosage`, `frequency`, `scheduled_time`, `instructions`, `prescribed_by`, `start_date`, `end_date`, `is_active`) VALUES
-- RV001 (Alex Smith) — prescribed by DOC001
('RM001', 'RV001', '4',  '10mg',  'Once daily',   '08:00', 'Take on empty stomach',   'DOC001', '2025-12-01', NULL, 1),
('RM002', 'RV001', '2',  '500mg', 'Twice daily',  '12:00', 'Take with meals',         'DOC001', '2025-11-15', NULL, 1),
('RM003', 'RV001', '45', '81mg',  'Once daily',   '18:00', 'Take with water',         'DOC001', '2025-10-01', NULL, 1),
-- RV002 (Maria Garcia) — prescribed by DOC002
('RM004', 'RV002', '6',  '50mg',  'Once daily',   '09:00', 'Take in the morning',     'DOC002', '2025-11-01', NULL, 1),
('RM005', 'RV002', '1',  '10mg',  'Once daily',   '21:00', 'Take at bedtime',         'DOC002', '2025-09-15', NULL, 1),
-- RV003 (Omar Khalil) — prescribed by DOC003
('RM006', 'RV003', '9',  '300mg', 'Three times daily', '08:00', 'Take with food', 'DOC003', '2025-10-01', NULL, 1),
('RM007', 'RV003', '10', '20mg',  'Once daily',   '07:00', 'Take before breakfast',   'DOC003', '2025-08-15', NULL, 1);

-- -------------------------------------------------------
-- 3b. medication_schedules — today's schedule for each rover
-- -------------------------------------------------------
INSERT INTO `medication_schedules` (`id`, `rover_id`, `rover_medication_id`, `medication_id`, `dosage`, `frequency`, `scheduled_time`, `scheduled_date`, `instructions`, `status`, `taken_at`, `prescribed_by`, `is_active`) VALUES
-- RV001 today
('MS001', 'RV001', 'RM001', '4',  '10mg',  'Once daily',  '08:00:00', CURDATE(), 'Take on empty stomach', 'taken',    DATE_SUB(CONCAT(CURDATE(), ' 08:05:00'), INTERVAL 0 DAY), 'DOC001', 1),
('MS002', 'RV001', 'RM002', '2',  '500mg', 'Twice daily', '12:00:00', CURDATE(), 'Take with meals',       'taken',    DATE_SUB(CONCAT(CURDATE(), ' 12:10:00'), INTERVAL 0 DAY), 'DOC001', 1),
('MS003', 'RV001', 'RM003', '45', '81mg',  'Once daily',  '18:00:00', CURDATE(), 'Take with water',       'upcoming', NULL, 'DOC001', 1),
('MS004', 'RV001', 'RM002', '2',  '500mg', 'Twice daily', '20:00:00', CURDATE(), 'Take with meals',       'upcoming', NULL, 'DOC001', 1),
-- RV001 yesterday (for history)
('MS005', 'RV001', 'RM001', '4',  '10mg',  'Once daily',  '08:00:00', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 'Take on empty stomach', 'taken', DATE_SUB(CONCAT(CURDATE(), ' 08:10:00'), INTERVAL 1 DAY), 'DOC001', 1),
('MS006', 'RV001', 'RM002', '2',  '500mg', 'Twice daily', '12:00:00', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 'Take with meals',       'taken', DATE_SUB(CONCAT(CURDATE(), ' 12:00:00'), INTERVAL 1 DAY), 'DOC001', 1),
('MS007', 'RV001', 'RM003', '45', '81mg',  'Once daily',  '18:00:00', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 'Take with water',       'taken', DATE_SUB(CONCAT(CURDATE(), ' 18:05:00'), INTERVAL 1 DAY), 'DOC001', 1),
('MS008', 'RV001', 'RM002', '2',  '500mg', 'Twice daily', '20:00:00', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 'Take with meals',       'missed', NULL, 'DOC001', 1),
-- RV002 today
('MS009', 'RV002', 'RM004', '6',  '50mg',  'Once daily',  '09:00:00', CURDATE(), 'Take in the morning',   'taken', DATE_SUB(CONCAT(CURDATE(), ' 09:02:00'), INTERVAL 0 DAY), 'DOC002', 1),
('MS010', 'RV002', 'RM005', '1',  '10mg',  'Once daily',  '21:00:00', CURDATE(), 'Take at bedtime',       'upcoming', NULL, 'DOC002', 1),
-- RV003 today
('MS011', 'RV003', 'RM006', '9',  '300mg', 'Three times daily', '08:00:00', CURDATE(), 'Take with food',   'taken', DATE_SUB(CONCAT(CURDATE(), ' 08:15:00'), INTERVAL 0 DAY), 'DOC003', 1),
('MS012', 'RV003', 'RM007', '10', '20mg',  'Once daily',  '07:00:00', CURDATE(), 'Take before breakfast', 'taken', DATE_SUB(CONCAT(CURDATE(), ' 07:00:00'), INTERVAL 0 DAY), 'DOC003', 1),
('MS013', 'RV003', 'RM006', '9',  '300mg', 'Three times daily', '13:00:00', CURDATE(), 'Take with food',   'due',  NULL, 'DOC003', 1),
('MS014', 'RV003', 'RM006', '9',  '300mg', 'Three times daily', '19:00:00', CURDATE(), 'Take with food',   'upcoming', NULL, 'DOC003', 1);

-- -------------------------------------------------------
-- 3c. activity_logs — realistic recent activity
-- -------------------------------------------------------
INSERT INTO `activity_logs` (`id`, `rover_id`, `type`, `title`, `description`, `priority`, `timestamp`) VALUES
-- RV001 (Alex Smith)
('ACT001', 'RV001', 'medication',    'Medication Taken',      'Took Lisinopril (10mg) — morning dose',                    'low',    DATE_SUB(NOW(), INTERVAL 2 HOUR)),
('ACT002', 'RV001', 'medication',    'Medication Taken',      'Took Metformin (500mg) — afternoon dose',                  'low',    DATE_SUB(NOW(), INTERVAL 4 HOUR)),
('ACT003', 'RV001', 'navigation',    'Navigation Complete',   'Navigated from Bedroom to Kitchen safely',                 'low',    DATE_SUB(NOW(), INTERVAL 5 HOUR)),
('ACT004', 'RV001', 'conversation',  'Morning Conversation',  'Had a 15-minute conversation with NovaCare about the weather', 'low', DATE_SUB(NOW(), INTERVAL 6 HOUR)),
('ACT005', 'RV001', 'vital',         'Vitals Recorded',       'Heart rate 72 bpm, SpO2 98.5%, Temperature 36.6°C',        'low',    DATE_SUB(NOW(), INTERVAL 7 HOUR)),
('ACT006', 'RV001', 'alert',         'Low Battery Warning',   'Rover battery dropped to 15% — now charging',              'high',   DATE_SUB(NOW(), INTERVAL 10 HOUR)),
('ACT007', 'RV001', 'navigation',    'Navigation Complete',   'Navigated from Living Room to Bathroom',                   'low',    DATE_SUB(NOW(), INTERVAL 12 HOUR)),
('ACT008', 'RV001', 'medication',    'Medication Missed',     'Missed Metformin (500mg) evening dose yesterday',          'medium', DATE_SUB(NOW(), INTERVAL 18 HOUR)),
('ACT009', 'RV001', 'alert',         'Fall Detection',        'Possible fall detected in hallway — Patient confirmed OK', 'high',   DATE_SUB(NOW(), INTERVAL 24 HOUR)),
('ACT010', 'RV001', 'medication',    'Medication Taken',      'Took Aspirin (81mg) — evening dose',                       'low',    DATE_SUB(NOW(), INTERVAL 26 HOUR)),
-- RV002 (Maria Garcia)
('ACT011', 'RV002', 'medication',    'Medication Taken',      'Took Metoprolol (50mg) — morning dose',                    'low',    DATE_SUB(NOW(), INTERVAL 3 HOUR)),
('ACT012', 'RV002', 'conversation',  'Video Call',            'Had a 10-minute video call with caregiver Youssef',        'low',    DATE_SUB(NOW(), INTERVAL 5 HOUR)),
('ACT013', 'RV002', 'vital',         'Vitals Recorded',       'Heart rate 68 bpm, SpO2 99.0%, Temperature 36.5°C',        'low',    DATE_SUB(NOW(), INTERVAL 7 HOUR)),
('ACT014', 'RV002', 'navigation',    'Navigation Complete',   'Navigated to Garden area for fresh air',                   'low',    DATE_SUB(NOW(), INTERVAL 8 HOUR)),
('ACT015', 'RV002', 'alert',         'Appointment Reminder',  'Upcoming appointment with Dr. Vazquez tomorrow at 10 AM',  'medium', DATE_SUB(NOW(), INTERVAL 12 HOUR)),
-- RV003 (Omar Khalil)
('ACT016', 'RV003', 'medication',    'Medication Taken',      'Took Gabapentin (300mg) — morning dose',                   'low',    DATE_SUB(NOW(), INTERVAL 2 HOUR)),
('ACT017', 'RV003', 'medication',    'Medication Taken',      'Took Omeprazole (20mg) — before breakfast',                'low',    DATE_SUB(NOW(), INTERVAL 3 HOUR)),
('ACT018', 'RV003', 'alert',         'Elevated Heart Rate',   'Heart rate reached 92 bpm — monitoring',                   'high',   DATE_SUB(NOW(), INTERVAL 6 HOUR)),
('ACT019', 'RV003', 'conversation',  'Entertainment Session', 'Played relaxation music playlist for 30 minutes',          'low',    DATE_SUB(NOW(), INTERVAL 8 HOUR)),
('ACT020', 'RV003', 'vital',         'Vitals Recorded',       'Heart rate 90 bpm, SpO2 96.0%, Temperature 37.1°C',        'medium', DATE_SUB(NOW(), INTERVAL 9 HOUR));

-- -------------------------------------------------------
-- 3d. sleep_logs — last 7 days for each rover
-- -------------------------------------------------------
INSERT INTO `sleep_logs` (`id`, `rover_id`, `date`, `bed_time`, `wake_time`, `duration_hours`, `quality`, `deep_sleep_minutes`, `light_sleep_minutes`, `rem_sleep_minutes`, `awakenings`, `notes`) VALUES
-- RV001
('SL001', 'RV001', DATE_SUB(CURDATE(), INTERVAL 1 DAY), '22:30:00', '06:00:00', 7.50, 'good',      105, 210, 90, 1, NULL),
('SL002', 'RV001', DATE_SUB(CURDATE(), INTERVAL 2 DAY), '23:00:00', '06:30:00', 7.50, 'good',      100, 220, 85, 0, NULL),
('SL003', 'RV001', DATE_SUB(CURDATE(), INTERVAL 3 DAY), '23:15:00', '06:00:00', 6.75, 'fair',       80, 200, 70, 2, 'Woke up briefly twice'),
('SL004', 'RV001', DATE_SUB(CURDATE(), INTERVAL 4 DAY), '22:00:00', '06:30:00', 8.50, 'excellent', 130, 230, 110, 0, NULL),
('SL005', 'RV001', DATE_SUB(CURDATE(), INTERVAL 5 DAY), '22:45:00', '06:15:00', 7.50, 'good',      110, 215, 95, 1, NULL),
('SL006', 'RV001', DATE_SUB(CURDATE(), INTERVAL 6 DAY), '01:00:00', '07:00:00', 6.00, 'poor',       60, 180, 50, 3, 'Late bedtime, restless'),
('SL007', 'RV001', DATE_SUB(CURDATE(), INTERVAL 7 DAY), '22:30:00', '06:30:00', 8.00, 'good',      115, 225, 100, 0, NULL),
-- RV002
('SL008', 'RV002', DATE_SUB(CURDATE(), INTERVAL 1 DAY), '21:30:00', '05:30:00', 8.00, 'excellent', 120, 230, 100, 0, NULL),
('SL009', 'RV002', DATE_SUB(CURDATE(), INTERVAL 2 DAY), '22:00:00', '06:00:00', 8.00, 'good',      110, 220, 95, 1, NULL),
('SL010', 'RV002', DATE_SUB(CURDATE(), INTERVAL 3 DAY), '22:30:00', '06:00:00', 7.50, 'good',      105, 215, 90, 0, NULL),
-- RV003
('SL011', 'RV003', DATE_SUB(CURDATE(), INTERVAL 1 DAY), '00:00:00', '07:30:00', 7.50, 'fair',       85, 210, 75, 2, 'Slight discomfort'),
('SL012', 'RV003', DATE_SUB(CURDATE(), INTERVAL 2 DAY), '23:30:00', '07:00:00', 7.50, 'good',      100, 220, 90, 1, NULL),
('SL013', 'RV003', DATE_SUB(CURDATE(), INTERVAL 3 DAY), '23:00:00', '06:30:00', 7.50, 'good',      105, 215, 95, 0, NULL);

-- -------------------------------------------------------
-- 3e. hydration_logs — last 7 days for each rover
-- -------------------------------------------------------
INSERT INTO `hydration_logs` (`id`, `rover_id`, `date`, `glasses`, `total_ml`, `goal_glasses`) VALUES
-- RV001
('HL001', 'RV001', CURDATE(),                            6, 1500, 8),
('HL002', 'RV001', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 8, 2000, 8),
('HL003', 'RV001', DATE_SUB(CURDATE(), INTERVAL 2 DAY), 7, 1750, 8),
('HL004', 'RV001', DATE_SUB(CURDATE(), INTERVAL 3 DAY), 5, 1250, 8),
('HL005', 'RV001', DATE_SUB(CURDATE(), INTERVAL 4 DAY), 9, 2250, 8),
('HL006', 'RV001', DATE_SUB(CURDATE(), INTERVAL 5 DAY), 7, 1750, 8),
('HL007', 'RV001', DATE_SUB(CURDATE(), INTERVAL 6 DAY), 6, 1500, 8),
-- RV002
('HL008', 'RV002', CURDATE(),                            7, 1750, 8),
('HL009', 'RV002', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 8, 2000, 8),
('HL010', 'RV002', DATE_SUB(CURDATE(), INTERVAL 2 DAY), 6, 1500, 8),
-- RV003
('HL011', 'RV003', CURDATE(),                            4, 1000, 8),
('HL012', 'RV003', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 5, 1250, 8),
('HL013', 'RV003', DATE_SUB(CURDATE(), INTERVAL 2 DAY), 6, 1500, 8);

-- -------------------------------------------------------
-- 3f. weight_logs — weekly entries for each rover
-- -------------------------------------------------------
INSERT INTO `weight_logs` (`id`, `rover_id`, `date`, `weight_kg`, `target_weight_kg`, `bmi`) VALUES
-- RV001 (Alex Smith) — target 65kg
('WL001', 'RV001', CURDATE(),                            65.80, 65.00, 22.5),
('WL002', 'RV001', DATE_SUB(CURDATE(), INTERVAL 7 DAY), 66.00, 65.00, 22.6),
('WL003', 'RV001', DATE_SUB(CURDATE(), INTERVAL 14 DAY),66.50, 65.00, 22.8),
('WL004', 'RV001', DATE_SUB(CURDATE(), INTERVAL 21 DAY),67.00, 65.00, 22.9),
-- RV002 (Maria Garcia) — target 55kg
('WL005', 'RV002', CURDATE(),                            57.20, 55.00, 21.8),
('WL006', 'RV002', DATE_SUB(CURDATE(), INTERVAL 7 DAY), 57.50, 55.00, 21.9),
-- RV003 (Omar Khalil) — target 70kg
('WL007', 'RV003', CURDATE(),                            72.00, 70.00, 23.1),
('WL008', 'RV003', DATE_SUB(CURDATE(), INTERVAL 7 DAY), 72.50, 70.00, 23.3);

-- -------------------------------------------------------
-- 3g. rover_battery_status — current + recent entries
-- -------------------------------------------------------
INSERT INTO `rover_battery_status` (`id`, `rover_id`, `battery_percent`, `is_charging`, `estimated_remaining_minutes`, `recorded_at`) VALUES
-- RV001
('RBS001', 'RV001', 85, 0, 420, NOW()),
('RBS002', 'RV001', 78, 0, 380, DATE_SUB(NOW(), INTERVAL 1 HOUR)),
('RBS003', 'RV001', 15, 1, 60,  DATE_SUB(NOW(), INTERVAL 10 HOUR)),
('RBS004', 'RV001', 100, 0, 500, DATE_SUB(NOW(), INTERVAL 12 HOUR)),
-- RV002
('RBS005', 'RV002', 92, 0, 460, NOW()),
('RBS006', 'RV002', 88, 0, 440, DATE_SUB(NOW(), INTERVAL 2 HOUR)),
-- RV003
('RBS007', 'RV003', 64, 0, 310, NOW()),
('RBS008', 'RV003', 70, 0, 340, DATE_SUB(NOW(), INTERVAL 2 HOUR));

-- -------------------------------------------------------
-- 3h. mood_logs — daily mood entries for each rover
-- -------------------------------------------------------
INSERT INTO `mood_logs` (`id`, `rover_id`, `date`, `mood`, `energy_level`, `anxiety_level`, `notes`, `emoji`) VALUES
-- RV001
('MD001', 'RV001', CURDATE(),                            'happy',     'moderate', 2, 'Feeling good today',          '😊'),
('MD002', 'RV001', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 'happy',     'high',     1, 'Great day, went outside',     '😄'),
('MD003', 'RV001', DATE_SUB(CURDATE(), INTERVAL 2 DAY), 'neutral',   'moderate', 3, 'Average day',                 '😐'),
('MD004', 'RV001', DATE_SUB(CURDATE(), INTERVAL 3 DAY), 'sad',       'low',      5, 'Missing family',              '😢'),
('MD005', 'RV001', DATE_SUB(CURDATE(), INTERVAL 4 DAY), 'happy',     'high',     1, 'Family visited!',             '😊'),
('MD006', 'RV001', DATE_SUB(CURDATE(), INTERVAL 5 DAY), 'neutral',   'moderate', 2, NULL,                          '😐'),
('MD007', 'RV001', DATE_SUB(CURDATE(), INTERVAL 6 DAY), 'very_happy','very_high',0, 'Best day in weeks!',          '🤩'),
-- RV002
('MD008', 'RV002', CURDATE(),                            'happy',     'moderate', 2, 'Calm and relaxed',            '😊'),
('MD009', 'RV002', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 'very_happy','high',     0, 'Daughter came to visit',      '😄'),
('MD010', 'RV002', DATE_SUB(CURDATE(), INTERVAL 2 DAY), 'neutral',   'moderate', 3, NULL,                          '😐'),
-- RV003
('MD011', 'RV003', CURDATE(),                            'neutral',   'low',      4, 'Some discomfort but managing','😐'),
('MD012', 'RV003', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 'happy',     'moderate', 3, 'Physical therapy went well',  '😊'),
('MD013', 'RV003', DATE_SUB(CURDATE(), INTERVAL 2 DAY), 'sad',       'very_low', 6, 'Hard day, pain flare-up',     '😢');

-- -------------------------------------------------------
-- 3i. emotion_tracking — populate from AI interaction data
-- -------------------------------------------------------
INSERT INTO `emotion_tracking` (`id`, `rover_id`, `date`, `avg_sentiment`, `primary_emotion`, `distress_detected`) VALUES
('ET001', 'RV001', CURDATE(),                            0.50,  'Happy',   0),
('ET002', 'RV001', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 0.30,  'Neutral', 0),
('ET003', 'RV001', DATE_SUB(CURDATE(), INTERVAL 2 DAY), -0.10, 'Tired',   0),
('ET004', 'RV002', CURDATE(),                            0.60,  'Relieved',0),
('ET005', 'RV002', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 0.10,  'Neutral', 0),
('ET006', 'RV003', CURDATE(),                            -0.30, 'Pain',    1),
('ET007', 'RV003', DATE_SUB(CURDATE(), INTERVAL 1 DAY), 0.40,  'Satisfied', 0);


COMMIT;
