-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Apr 15, 2026 at 09:38 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `novacare_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `action_statuses`
--

CREATE TABLE `action_statuses` (
  `id` char(36) NOT NULL,
  `status_name` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `action_statuses`
--

INSERT INTO `action_statuses` (`id`, `status_name`) VALUES
('S003', 'denied'),
('S002', 'failure'),
('S001', 'success');

-- --------------------------------------------------------

--
-- Table structure for table `action_types`
--

CREATE TABLE `action_types` (
  `id` char(36) NOT NULL,
  `action_name` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `action_types`
--

INSERT INTO `action_types` (`id`, `action_name`) VALUES
('A002', 'data.view'),
('A003', 'emergency.alert'),
('A004', 'medication.take'),
('A001', 'user.login');

-- --------------------------------------------------------

--
-- Table structure for table `addresses`
--

CREATE TABLE `addresses` (
  `id` char(36) NOT NULL,
  `street_address` varchar(255) DEFAULT NULL,
  `city` varchar(100) DEFAULT NULL,
  `state_province` varchar(100) DEFAULT NULL,
  `postal_code` varchar(20) DEFAULT NULL,
  `country_id` char(36) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `addresses`
--

INSERT INTO `addresses` (`id`, `street_address`, `city`, `state_province`, `postal_code`, `country_id`) VALUES
('AD001', '123 Corniche', 'Alexandria', 'Alexandria', '21500', 'C001'),
('AD002', '456 Maadi St', 'Cairo', 'Cairo', '11431', 'C001'),
('AD003', '789 Rehab City', 'New Cairo', 'Cairo', '11841', 'C001');

-- --------------------------------------------------------

--
-- Table structure for table `ai_interactions`
--

CREATE TABLE `ai_interactions` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `conversation_id` char(36) NOT NULL,
  `user_message` text DEFAULT NULL,
  `ai_response` text DEFAULT NULL,
  `emotion_detected` varchar(50) DEFAULT NULL,
  `sentiment_score` decimal(3,2) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `ai_interactions`
--

INSERT INTO `ai_interactions` (`id`, `rover_id`, `conversation_id`, `user_message`, `ai_response`, `emotion_detected`, `sentiment_score`, `created_at`) VALUES
('AI001', 'RV001', 'CONV001', 'How is my heart rate?', 'It is stable at 72 bpm.', 'Neutral', 0.10, '2026-04-14 18:50:07'),
('AI002', 'RV001', 'CONV001', 'I feel a bit tired.', 'I recommend resting. I have alerted Sarah.', 'Tired', -0.20, '2026-04-14 18:50:07'),
('AI003', 'RV002', 'CONV002', 'Is it time for my medicine?', 'Yes, please take your Metformin.', 'Curious', 0.05, '2026-04-14 18:50:07'),
('AI004', 'RV003', 'CONV003', 'Help me!', 'Emergency protocols activated. Contacting EMS.', 'Panic', -0.90, '2026-04-14 18:50:07'),
('AI005', 'RV001', 'CONV001', 'Thank you, Nova.', 'You are welcome, Alex.', 'Happy', 0.80, '2026-04-14 18:50:07'),
('AI006', 'RV002', 'CONV002', 'Good morning.', 'Good morning, Maria. Vitals are normal.', 'Happy', 0.50, '2026-04-14 18:50:07'),
('AI007', 'RV003', 'CONV003', 'I am alone.', 'I am here with you. Sarah is on her way.', 'Sad', -0.40, '2026-04-14 18:50:07'),
('AI008', 'RV001', 'CONV004', 'What is the weather?', 'It is sunny in Alexandria.', 'Neutral', 0.00, '2026-04-14 18:50:07'),
('AI009', 'RV002', 'CONV005', 'Remind me to call the doctor.', 'Reminder set for 2 PM.', 'Neutral', 0.10, '2026-04-14 18:50:07'),
('AI010', 'RV003', 'CONV006', 'I am feeling better.', 'That is great news!', 'Happy', 0.90, '2026-04-14 18:50:07'),
('AI011', 'RV001', 'CONV004', 'Play some music.', 'Playing your relaxation playlist.', 'Calm', 0.60, '2026-04-14 18:50:07'),
('AI012', 'RV002', 'CONV005', 'Is my daughter home?', 'She just arrived at the front door.', 'Relieved', 0.70, '2026-04-14 18:50:07'),
('AI013', 'RV003', 'CONV006', 'I slept well.', 'Your sleep data shows deep recovery.', 'Satisfied', 0.80, '2026-04-14 18:50:07'),
('AI014', 'RV001', 'CONV007', 'Who is at the door?', 'It is your therapist, Linda.', 'Neutral', 0.00, '2026-04-14 18:50:07'),
('AI015', 'RV002', 'CONV008', 'My leg hurts.', 'I have logged this for Dr. Ahmed.', 'Pain', -0.50, '2026-04-14 18:50:07');

-- --------------------------------------------------------

--
-- Table structure for table `allergies`
--

CREATE TABLE `allergies` (
  `id` char(36) NOT NULL,
  `name` varchar(100) NOT NULL,
  `category` enum('medication','food','material') NOT NULL,
  `description` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `allergies`
--

INSERT INTO `allergies` (`id`, `name`, `category`, `description`) VALUES
('cc1e127c-3849-11f1-a697-00155dd53540', 'Penicillin', 'medication', NULL),
('cc1e20aa-3849-11f1-a697-00155dd53540', 'Sulfa Drugs', 'medication', NULL),
('cc1e213b-3849-11f1-a697-00155dd53540', 'Aspirin', 'medication', NULL),
('cc1e217e-3849-11f1-a697-00155dd53540', 'Peanuts', 'food', NULL),
('cc1e21c0-3849-11f1-a697-00155dd53540', 'Dairy', 'food', NULL),
('cc1e21fb-3849-11f1-a697-00155dd53540', 'Shellfish', 'food', NULL),
('cc1e2239-3849-11f1-a697-00155dd53540', 'Gluten', 'food', NULL),
('cc1e2272-3849-11f1-a697-00155dd53540', 'Latex', 'material', NULL),
('cc1e22b3-3849-11f1-a697-00155dd53540', 'Adhesive Tape', 'material', NULL),
('cc1e22ee-3849-11f1-a697-00155dd53540', 'Eggs', 'food', NULL),
('cc1e2328-3849-11f1-a697-00155dd53540', 'Soy', 'food', NULL),
('cc1e2365-3849-11f1-a697-00155dd53540', 'Ibuprofen', 'medication', NULL),
('cc1e23a7-3849-11f1-a697-00155dd53540', 'Codeine', 'medication', NULL),
('cc1e23e1-3849-11f1-a697-00155dd53540', 'Tree Nuts', 'food', NULL),
('cc1e2418-3849-11f1-a697-00155dd53540', 'Nickel', 'material', NULL);

-- --------------------------------------------------------

--
-- Table structure for table `appointments`
--

CREATE TABLE `appointments` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `doctor_id` char(36) DEFAULT NULL,
  `appointment_type_id` char(36) DEFAULT NULL,
  `status_id` char(36) DEFAULT NULL,
  `scheduled_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `notes` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `appointment_statuses`
--

CREATE TABLE `appointment_statuses` (
  `id` char(36) NOT NULL,
  `status_name` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `appointment_statuses`
--

INSERT INTO `appointment_statuses` (`id`, `status_name`) VALUES
('AS15', 'Archived'),
('AS05', 'Cancelled by Doctor'),
('AS04', 'Cancelled by Patient'),
('AS03', 'Completed'),
('AS02', 'Confirmed'),
('AS10', 'Delayed'),
('AS14', 'Follow-up Required'),
('AS08', 'In Progress'),
('AS06', 'No Show'),
('AS12', 'Payment Pending'),
('AS11', 'Pending Approval'),
('AS13', 'Referred Out'),
('AS07', 'Rescheduled'),
('AS01', 'Scheduled'),
('AS09', 'Waiting Room');

-- --------------------------------------------------------

--
-- Table structure for table `appointment_types`
--

CREATE TABLE `appointment_types` (
  `id` char(36) NOT NULL,
  `type_name` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `appointment_types`
--

INSERT INTO `appointment_types` (`id`, `type_name`) VALUES
('AT15', 'Chronic Care'),
('AT10', 'Diagnostic Scan'),
('AT02', 'Emergency'),
('AT09', 'Follow-up'),
('AT13', 'Home Visit'),
('AT05', 'Lab Work'),
('AT08', 'Mental Health Session'),
('AT12', 'Nutrition Counseling'),
('AT04', 'Physical Therapy'),
('AT01', 'Routine Checkup'),
('AT14', 'Script Renewal'),
('AT07', 'Specialist Visit'),
('AT11', 'Surgery Consultation'),
('AT06', 'Vaccination'),
('AT03', 'Video Consultation');

-- --------------------------------------------------------

--
-- Table structure for table `approval_statuses`
--

CREATE TABLE `approval_statuses` (
  `id` char(36) NOT NULL,
  `status_name` varchar(50) NOT NULL,
  `display_label` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `audit_logs`
--

CREATE TABLE `audit_logs` (
  `id` char(36) NOT NULL,
  `actor_id` char(36) NOT NULL,
  `action_type_id` char(36) NOT NULL,
  `resource_type` varchar(100) DEFAULT NULL,
  `resource_id` char(36) DEFAULT NULL,
  `old_value` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`old_value`)),
  `new_value` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`new_value`)),
  `action_status_id` char(36) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `audit_logs`
--

INSERT INTO `audit_logs` (`id`, `actor_id`, `action_type_id`, `resource_type`, `resource_id`, `old_value`, `new_value`, `action_status_id`, `created_at`) VALUES
('AL001', 'U001', 'A001', 'user', 'U001', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL002', 'U003', 'A002', 'vital_signs', 'RV001', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL003', 'U002', 'A002', 'vital_signs', 'RV001', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL004', 'U001', 'A003', 'alert_event', 'RV001', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL005', 'U001', 'A004', 'medication', 'RV001', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL006', 'U004', 'A001', 'user', 'U004', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL007', 'U006', 'A002', 'vital_signs', 'RV002', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL008', 'U007', 'A003', 'alert_event', 'RV003', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL009', 'U011', 'A002', 'audit_logs', 'AL001', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL010', 'U003', 'A002', 'health_report', 'RV001', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL011', 'U015', 'A001', 'user', 'U015', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL012', 'U001', 'A001', 'user', 'U001', NULL, NULL, 'S002', '2026-04-14 18:50:07'),
('AL013', 'U010', 'A001', 'user', 'U010', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL014', 'U009', 'A002', 'vital_signs', 'RV003', NULL, NULL, 'S001', '2026-04-14 18:50:07'),
('AL015', 'U001', 'A004', 'medication', 'RV001', NULL, NULL, 'S001', '2026-04-14 18:50:07');

-- --------------------------------------------------------

--
-- Table structure for table `caregivers`
--

CREATE TABLE `caregivers` (
  `id` char(36) NOT NULL,
  `user_id` char(36) NOT NULL,
  `phone_number` varchar(20) DEFAULT NULL,
  `address_id` char(36) DEFAULT NULL,
  `government_id_number` varchar(100) DEFAULT NULL,
  `verification_status_id` char(36) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `caregivers`
--

INSERT INTO `caregivers` (`id`, `user_id`, `phone_number`, `address_id`, `government_id_number`, `verification_status_id`) VALUES
('CG001', 'U002', '+201011112222', 'AD001', 'ID-998877', 'V002'),
('CG002', 'U005', '+201022223333', 'AD002', 'ID-665544', 'V002'),
('CG003', 'U008', '+201033334444', 'AD003', 'ID-332211', 'V001');

-- --------------------------------------------------------

--
-- Table structure for table `caregiver_rover_assignments`
--

CREATE TABLE `caregiver_rover_assignments` (
  `id` char(36) NOT NULL,
  `caregiver_id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `relationship_type_id` char(36) NOT NULL,
  `is_active` tinyint(1) DEFAULT 1
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `clinic_organizations`
--

CREATE TABLE `clinic_organizations` (
  `id` char(36) NOT NULL,
  `name` varchar(150) NOT NULL,
  `address_id` char(36) DEFAULT NULL,
  `phone_number` varchar(20) DEFAULT NULL,
  `is_verified` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `countries`
--

CREATE TABLE `countries` (
  `id` char(36) NOT NULL,
  `name` varchar(100) NOT NULL,
  `iso_code` varchar(5) NOT NULL,
  `phone_code` varchar(10) DEFAULT NULL,
  `currency_code` varchar(10) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `countries`
--

INSERT INTO `countries` (`id`, `name`, `iso_code`, `phone_code`, `currency_code`) VALUES
('C001', 'Egypt', 'EG', '+20', 'EGP'),
('C002', 'United States', 'US', '+1', 'USD'),
('C003', 'United Kingdom', 'GB', '+44', 'GBP');

-- --------------------------------------------------------

--
-- Table structure for table `device_info`
--

CREATE TABLE `device_info` (
  `id` char(36) NOT NULL,
  `user_id` char(36) NOT NULL,
  `device_name` varchar(255) DEFAULT NULL,
  `device_type` varchar(50) DEFAULT NULL,
  `device_hash` varchar(255) DEFAULT NULL,
  `is_trusted` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `doctors`
--

CREATE TABLE `doctors` (
  `id` char(36) NOT NULL,
  `user_id` char(36) NOT NULL,
  `medical_license_num` varchar(100) DEFAULT NULL,
  `specialization_id` char(36) DEFAULT NULL,
  `verification_status_id` char(36) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `doctors`
--

INSERT INTO `doctors` (`id`, `user_id`, `medical_license_num`, `specialization_id`, `verification_status_id`) VALUES
('DOC001', 'U003', 'EGY-998877', 'SP01', 'V002'),
('DOC002', 'U006', 'EGY-112233', 'SP15', 'V002'),
('DOC003', 'U009', 'EGY-445566', 'SP02', 'V002');

-- --------------------------------------------------------

--
-- Table structure for table `email_verification_tokens`
--

CREATE TABLE `email_verification_tokens` (
  `id` char(36) NOT NULL,
  `user_id` char(36) NOT NULL,
  `token_hash` varchar(255) NOT NULL,
  `expires_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `verified_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `emergency_contacts`
--

CREATE TABLE `emergency_contacts` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `name` varchar(255) NOT NULL,
  `relationship` varchar(100) DEFAULT NULL,
  `phone_number` varchar(20) DEFAULT NULL,
  `is_primary` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `emergency_contacts`
--

INSERT INTO `emergency_contacts` (`id`, `rover_id`, `name`, `relationship`, `phone_number`, `is_primary`) VALUES
('6c9ca6ac-0617-44c0-a093-60bffbdbec31', 'bf3fed8e-d9d8-4f45-837a-f5d028e3a0d8', 'Emergency Contact', '', '+20-100-555-0001', 1),
('EC001', 'RV001', 'Sarah Jones', 'Sister', '+201011112222', 1),
('EC002', 'RV001', 'John Smith', 'Father', '+201011112223', 0),
('EC003', 'RV001', 'Mary Smith', 'Mother', '+201011112224', 0),
('EC004', 'RV002', 'Youssef Bedair', 'Spouse', '+201022223333', 1),
('EC005', 'RV002', 'Fatima Garcia', 'Mother', '+201022223334', 0),
('EC006', 'RV002', 'Amir Bedair', 'Son', '+201022223335', 0),
('EC007', 'RV003', 'Mona Rashad', 'Guardian', '+201033334444', 1),
('EC008', 'RV003', 'Khalid Rashad', 'Uncle', '+201033334445', 0),
('EC009', 'RV003', 'Laila Ali', 'Neighbor', '+201033334446', 0),
('EC010', 'RV001', 'Dr. Ahmed', 'Doctor', '+201033333333', 0),
('EC011', 'RV002', 'Alexandria EMS', 'Emergency', '123', 0),
('EC012', 'RV003', 'Red Crescent', 'Emergency', '123', 0),
('EC013', 'RV001', 'Robert Smith', 'Brother', '+201011112225', 0),
('EC014', 'RV002', 'Elena G.', 'Caregiver', '+201022223336', 0),
('EC015', 'RV003', 'Hassan Ali', 'Grandfather', '+201033334447', 0),
('ed11ea55-8912-4ad4-9155-98068bb02252', 'a63df65d-0743-40a3-9038-deb1bbca4ccd', 'Jane Smith', '', '+20-100-555-0002', 1);

-- --------------------------------------------------------

--
-- Table structure for table `emotion_tracking`
--

CREATE TABLE `emotion_tracking` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `date` date NOT NULL,
  `avg_sentiment` decimal(3,2) DEFAULT NULL,
  `primary_emotion` varchar(50) DEFAULT NULL,
  `distress_detected` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `health_conditions`
--

CREATE TABLE `health_conditions` (
  `id` char(36) NOT NULL,
  `name` varchar(100) NOT NULL,
  `icd10_code` varchar(20) DEFAULT NULL,
  `description` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `health_conditions`
--

INSERT INTO `health_conditions` (`id`, `name`, `icd10_code`, `description`, `created_at`) VALUES
('cc1d86e1-3849-11f1-a697-00155dd53540', 'Paraplegia', 'G82.2', NULL, '2026-04-14 21:35:03'),
('cc1d91df-3849-11f1-a697-00155dd53540', 'Quadriplegia', 'G82.5', NULL, '2026-04-14 21:35:03'),
('cc1d9293-3849-11f1-a697-00155dd53540', 'Cerebral Palsy', 'G80', NULL, '2026-04-14 21:35:03'),
('cc1d92ec-3849-11f1-a697-00155dd53540', 'Multiple Sclerosis', 'G35', NULL, '2026-04-14 21:35:03'),
('cc1d9341-3849-11f1-a697-00155dd53540', 'Muscular Dystrophy', 'G71.0', NULL, '2026-04-14 21:35:03'),
('cc1d9396-3849-11f1-a697-00155dd53540', 'Spina Bifida', 'Q05', NULL, '2026-04-14 21:35:03'),
('cc1d93ed-3849-11f1-a697-00155dd53540', 'Amyotrophic Lateral Sclerosis (ALS)', 'G12.21', NULL, '2026-04-14 21:35:03'),
('cc1d9442-3849-11f1-a697-00155dd53540', 'Parkinsons Disease', 'G20', NULL, '2026-04-14 21:35:03'),
('cc1d9499-3849-11f1-a697-00155dd53540', 'Stroke Recovery', 'I69', NULL, '2026-04-14 21:35:03'),
('cc1d953d-3849-11f1-a697-00155dd53540', 'Traumatic Brain Injury', 'S06', NULL, '2026-04-14 21:35:03'),
('cc1d9592-3849-11f1-a697-00155dd53540', 'Rheumatoid Arthritis', 'M06', NULL, '2026-04-14 21:35:03'),
('cc1d969b-3849-11f1-a697-00155dd53540', 'Visual Impairment', 'H54', NULL, '2026-04-14 21:35:03'),
('cc1d974f-3849-11f1-a697-00155dd53540', 'Hearing Impairment', 'H90', NULL, '2026-04-14 21:35:03'),
('cc1d97a4-3849-11f1-a697-00155dd53540', 'Amputation', 'Z89', NULL, '2026-04-14 21:35:03'),
('cc1d97f8-3849-11f1-a697-00155dd53540', 'Autism Spectrum Disorder', 'F84.0', NULL, '2026-04-14 21:35:03');

-- --------------------------------------------------------

--
-- Table structure for table `identity_documents`
--

CREATE TABLE `identity_documents` (
  `id` char(36) NOT NULL,
  `caregiver_id` char(36) NOT NULL,
  `id_type_id` char(36) DEFAULT NULL,
  `document_url` text NOT NULL,
  `is_verified` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `id_types`
--

CREATE TABLE `id_types` (
  `id` char(36) NOT NULL,
  `name` varchar(50) NOT NULL,
  `country_id` char(36) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `main_disability`
--

CREATE TABLE `main_disability` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `condition_name` varchar(150) NOT NULL,
  `onset_date` date DEFAULT NULL,
  `is_congenital` tinyint(1) DEFAULT 0,
  `notes` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `measurement_devices`
--

CREATE TABLE `measurement_devices` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `device_name` varchar(100) DEFAULT NULL,
  `device_type` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `medical_license_documents`
--

CREATE TABLE `medical_license_documents` (
  `id` char(36) NOT NULL,
  `doctor_id` char(36) NOT NULL,
  `document_url` text NOT NULL,
  `is_verified` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `medical_notes`
--

CREATE TABLE `medical_notes` (
  `id` char(36) NOT NULL,
  `doctor_id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `note_type_id` char(36) DEFAULT NULL,
  `note_content` text NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `medical_notes`
--

INSERT INTO `medical_notes` (`id`, `doctor_id`, `rover_id`, `note_type_id`, `note_content`, `created_at`) VALUES
('MN01', 'DOC001', 'RV001', 'NT01', 'Patient showing early signs of hypertension.', '2026-04-14 20:34:57'),
('MN02', 'DOC002', 'RV001', 'NT02', 'Recovery from physical therapy is on track.', '2026-04-14 20:34:57'),
('MN03', 'DOC003', 'RV002', 'NT10', 'Neurological assessment complete; no tremors detected.', '2026-04-14 20:34:57');

-- --------------------------------------------------------

--
-- Table structure for table `medical_note_types`
--

CREATE TABLE `medical_note_types` (
  `id` char(36) NOT NULL,
  `note_type` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `medical_note_types`
--

INSERT INTO `medical_note_types` (`id`, `note_type`) VALUES
('NT09', 'Allergy Warning'),
('NT01', 'Diagnosis'),
('NT07', 'Discharge Summary'),
('NT14', 'Emergency Report'),
('NT11', 'Follow-up'),
('NT10', 'Initial Assessment'),
('NT04', 'Lab Results'),
('NT12', 'Mental Health Eval'),
('NT03', 'Prescription Note'),
('NT02', 'Progress Note'),
('NT06', 'Referral'),
('NT15', 'Routine Check'),
('NT05', 'Surgery Summary'),
('NT08', 'Treatment Plan'),
('NT13', 'Vitals Analysis');

-- --------------------------------------------------------

--
-- Table structure for table `medication_catalog`
--

CREATE TABLE `medication_catalog` (
  `med_id` bigint(20) UNSIGNED NOT NULL,
  `brand_name` varchar(255) NOT NULL,
  `generic_name` varchar(255) NOT NULL,
  `manufacturer` varchar(255) DEFAULT NULL,
  `therapeutic_class` varchar(255) DEFAULT NULL,
  `is_controlled_substance` tinyint(1) DEFAULT 0,
  `product_external_id` varchar(50) DEFAULT NULL,
  `product_type` varchar(100) DEFAULT NULL,
  `route` varchar(100) DEFAULT NULL,
  `marketing_category` varchar(100) DEFAULT NULL,
  `substance_name` text DEFAULT NULL,
  `active_strength` decimal(10,4) DEFAULT NULL,
  `strength_unit` varchar(50) DEFAULT NULL,
  `pharm_classes` text DEFAULT NULL,
  `dea_schedule` varchar(10) DEFAULT NULL,
  `dosage_form` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `medication_catalog`
--

INSERT INTO `medication_catalog` (`med_id`, `brand_name`, `generic_name`, `manufacturer`, `therapeutic_class`, `is_controlled_substance`, `product_external_id`, `product_type`, `route`, `marketing_category`, `substance_name`, `active_strength`, `strength_unit`, `pharm_classes`, `dea_schedule`, `dosage_form`) VALUES
(1, 'Lipitor', 'Atorvastatin', 'Pfizer', 'Antihyperlipidemic', 0, '00071-0155-23', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ATORVASTATIN CALCIUM', 10.0000, 'mg', 'HMG-CoA Reductase Inhibitor', NULL, 'TABLET'),
(2, 'Glucophage', 'Metformin', 'Bristol-Myers Squibb', 'Antidiabetic', 0, '00087-6060-05', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'METFORMIN HYDROCHLORIDE', 500.0000, 'mg', 'Biguanide', NULL, 'TABLET'),
(3, 'Synthroid', 'Levothyroxine', 'AbbVie', 'Thyroid Hormone', 0, '00074-4341-13', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LEVOTHYROXINE SODIUM', 0.0500, 'mg', 'Thyroid Hormone Replacement', NULL, 'TABLET'),
(4, 'Zestril', 'Lisinopril', 'AstraZeneca', 'Antihypertensive', 0, '00310-0130-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LISINOPRIL', 10.0000, 'mg', 'Angiotensin Converting Enzyme Inhibitor', NULL, 'TABLET'),
(5, 'Norvasc', 'Amlodipine', 'Pfizer', 'Antihypertensive', 0, '00069-1520-68', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'AMLODIPINE BESYLATE', 5.0000, 'mg', 'Calcium Channel Blocker', NULL, 'TABLET'),
(6, 'Lopressor', 'Metoprolol', 'Novartis', 'Beta-Blocker', 0, '00078-0457-05', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'METOPROLOL TARTRATE', 50.0000, 'mg', 'Beta-1 Adrenergic Blocker', NULL, 'TABLET'),
(7, 'ProAir HFA', 'Albuterol', 'Teva', 'Bronchodilator', 0, '00591-2325-04', 'HUMAN PRESCRIPTION DRUG', 'RESPIRATORY (INHALATION)', 'NDA', 'ALBUTEROL SULFATE', 0.0900, 'mg/actuation', 'Beta-2 Adrenergic Agonist', NULL, 'AEROSOL, METERED'),
(8, 'Cozaar', 'Losartan', 'Merck', 'Antihypertensive', 0, '00006-0951-54', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LOSARTAN POTASSIUM', 50.0000, 'mg', 'Angiotensin II Receptor Antagonist', NULL, 'TABLET'),
(9, 'Neurontin', 'Gabapentin', 'Pfizer', 'Anticonvulsant', 0, '00071-0801-24', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'GABAPENTIN', 300.0000, 'mg', 'Gamma-Aminobutyric Acid Analog', NULL, 'CAPSULE'),
(10, 'Prilosec', 'Omeprazole', 'Procter & Gamble', 'Gastrointestinal', 0, '00006-0397-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'OMEPRAZOLE', 20.0000, 'mg', 'Proton Pump Inhibitor', NULL, 'CAPSULE, DELAYED RELEASE'),
(11, 'Zoloft', 'Sertraline', 'Roerig', 'Antidepressant', 0, '00049-4960-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'SERTRALINE HYDROCHLORIDE', 50.0000, 'mg', 'Selective Serotonin Reuptake Inhibitor', NULL, 'TABLET'),
(12, 'Crestor', 'Rosuvastatin', 'AstraZeneca', 'Antihyperlipidemic', 0, '00310-0751-90', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ROSUVASTATIN CALCIUM', 10.0000, 'mg', 'HMG-CoA Reductase Inhibitor', NULL, 'TABLET'),
(13, 'Protonix', 'Pantoprazole', 'Wyeth', 'Gastrointestinal', 0, '00008-0841-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'PANTOPRAZOLE SODIUM', 40.0000, 'mg', 'Proton Pump Inhibitor', NULL, 'TABLET, DELAYED RELEASE'),
(14, 'Lexapro', 'Escitalopram', 'Forest Labs', 'Antidepressant', 0, '00456-2010-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ESCITALOPRAM OXALATE', 10.0000, 'mg', 'Selective Serotonin Reuptake Inhibitor', NULL, 'TABLET'),
(15, 'Microzide', 'Hydrochlorothiazide', 'Watson', 'Diuretic', 0, '00525-0601-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'HYDROCHLOROTHIAZIDE', 12.5000, 'mg', 'Thiazide Diuretic', NULL, 'CAPSULE'),
(16, 'Wellbutrin', 'Bupropion', 'GSK', 'Antidepressant', 0, '00173-0135-55', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'BUPROPION HYDROCHLORIDE', 75.0000, 'mg', 'Aminoketone', NULL, 'TABLET'),
(17, 'Prozac', 'Fluoxetine', 'Eli Lilly', 'Antidepressant', 0, '00002-3104-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'FLUOXETINE HYDROCHLORIDE', 20.0000, 'mg', 'Selective Serotonin Reuptake Inhibitor', NULL, 'CAPSULE'),
(18, 'Ozempic', 'Semaglutide', 'Novo Nordisk', 'Antidiabetic', 0, '00169-4132-12', 'HUMAN PRESCRIPTION DRUG', 'SUBCUTANEOUS', 'NDA', 'SEMAGLUTIDE', 2.0000, 'mg/mL', 'GLP-1 Receptor Agonist', NULL, 'INJECTION, SOLUTION'),
(19, 'Singulair', 'Montelukast', 'Organon', 'Respiratory', 0, '00006-0841-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'MONTELUKAST SODIUM', 10.0000, 'mg', 'Leukotriene Receptor Antagonist', NULL, 'TABLET'),
(20, 'Desyrel', 'Trazodone', 'Apothecon', 'Antidepressant', 0, '00003-0605-11', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'TRAZODONE HYDROCHLORIDE', 50.0000, 'mg', 'Serotonin Modulator', NULL, 'TABLET'),
(21, 'Zocor', 'Simvastatin', 'Merck', 'Antihyperlipidemic', 0, '00006-0735-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'SIMVASTATIN', 20.0000, 'mg', 'HMG-CoA Reductase Inhibitor', NULL, 'TABLET'),
(22, 'Amoxil', 'Amoxicillin', 'GSK', 'Antibiotic', 0, '00029-6007-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'AMOXICILLIN', 500.0000, 'mg', 'Penicillin-class Antibacterial', NULL, 'CAPSULE'),
(23, 'Flomax', 'Tamsulosin', 'Boehringer Ingelheim', 'Urological', 0, '00025-0058-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'TAMSULOSIN HYDROCHLORIDE', 0.4000, 'mg', 'Alpha-1 Adrenergic Antagonist', NULL, 'CAPSULE, EXTENDED RELEASE'),
(24, 'Vicodin', 'Hydrocodone/APAP', 'AbbVie', 'Analgesic', 1, '00074-3041-13', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'HYDROCODONE BITARTRATE / ACETAMINOPHEN', 5.0000, 'mg', 'Opioid Agonist', 'C-II', 'TABLET'),
(25, 'Flonase', 'Fluticasone', 'GSK', 'Respiratory', 0, '00173-0453-01', 'HUMAN PRESCRIPTION DRUG', 'NASAL', 'NDA', 'FLUTICASONE PROPIONATE', 0.0500, 'mg/actuation', 'Corticosteroid', NULL, 'SPRAY, METERED'),
(26, 'Mobic', 'Meloxicam', 'Boehringer Ingelheim', 'Analgesic', 0, '00597-0029-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'MELOXICAM', 7.5000, 'mg', 'Nonsteroidal Anti-inflammatory Drug', NULL, 'TABLET'),
(27, 'Eliquis', 'Apixaban', 'BMS', 'Anticoagulant', 0, '00003-0893-21', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'APIXABAN', 5.0000, 'mg', 'Factor Xa Inhibitor', NULL, 'TABLET'),
(28, 'Lasix', 'Furosemide', 'Sanofi-Aventis', 'Diuretic', 0, '00039-0060-13', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'FUROSEMIDE', 40.0000, 'mg', 'Loop Diuretic', NULL, 'TABLET'),
(29, 'Lantus', 'Insulin Glargine', 'Sanofi', 'Antidiabetic', 0, '00088-2220-33', 'HUMAN PRESCRIPTION DRUG', 'SUBCUTANEOUS', 'NDA', 'INSULIN GLARGINE', 100.0000, 'UNIT/mL', 'Long-acting Insulin', NULL, 'INJECTION, SOLUTION'),
(30, 'Cymbalta', 'Duloxetine', 'Eli Lilly', 'Antidepressant', 0, '00002-3235-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'DULOXETINE HYDROCHLORIDE', 30.0000, 'mg', 'Serotonin Norepinephrine Reuptake Inhibitor', NULL, 'CAPSULE, DELAYED RELEASE'),
(31, 'Advil', 'Ibuprofen', 'Wyeth', 'Analgesic', 0, '00573-0150-20', 'HUMAN OTC DRUG', 'ORAL', 'OTC MONOGRAPH FINAL', 'IBUPROFEN', 200.0000, 'mg', 'Nonsteroidal Anti-inflammatory Drug', NULL, 'TABLET'),
(32, 'Pepcid', 'Famotidine', 'Merck', 'Gastrointestinal', 0, '00006-3528-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'FAMOTIDINE', 20.0000, 'mg', 'H2 Receptor Antagonist', NULL, 'TABLET'),
(33, 'Jardiance', 'Empagliflozin', 'Boehringer Ingelheim', 'Antidiabetic', 0, '00597-0152-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'EMPAGLIFLOZIN', 10.0000, 'mg', 'SGLT2 Inhibitor', NULL, 'TABLET'),
(34, 'Coreg', 'Carvedilol', 'GSK', 'Antihypertensive', 0, '00007-4139-13', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'CARVEDILOL', 6.2500, 'mg', 'Alpha/Beta Adrenergic Blocker', NULL, 'TABLET'),
(35, 'Ultram', 'Tramadol', 'Janssen', 'Analgesic', 1, '50458-0653-60', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'TRAMADOL HYDROCHLORIDE', 50.0000, 'mg', 'Opioid Agonist', 'C-IV', 'TABLET'),
(36, 'Xanax', 'Alprazolam', 'Pharmacia & Upjohn', 'Antianxiety', 1, '00009-0029-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ALPRAZOLAM', 0.2500, 'mg', 'Benzodiazepine', 'C-IV', 'TABLET'),
(37, 'Deltasone', 'Prednisone', 'Pharmacia & Upjohn', 'Corticosteroid', 0, '00009-0147-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'PREDNISONE', 5.0000, 'mg', 'Glucocorticoid', NULL, 'TABLET'),
(38, 'Vistaril', 'Hydroxyzine', 'Pfizer', 'Antihistamine', 0, '00069-0541-66', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'HYDROXYZINE PAMOATE', 25.0000, 'mg', 'H1 Receptor Antagonist', NULL, 'CAPSULE'),
(39, 'BuSpar', 'Buspirone', 'BMS', 'Antianxiety', 0, '00087-0818-41', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'BUSPIRONE HYDROCHLORIDE', 5.0000, 'mg', 'Azapirone Anxiolytic', NULL, 'TABLET'),
(40, 'Plavix', 'Clopidogrel', 'Sanofi-Aventis', 'Antiplatelet', 0, '00024-1950-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'CLOPIDOGREL BISULFATE', 75.0000, 'mg', 'P2Y12 Platelet Inhibitor', NULL, 'TABLET'),
(41, 'Glucotrol', 'Glipizide', 'Pfizer', 'Antidiabetic', 0, '00049-4110-66', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'GLIPIZIDE', 5.0000, 'mg', 'Sulfonylurea', NULL, 'TABLET'),
(42, 'Celexa', 'Citalopram', 'Forest Labs', 'Antidepressant', 0, '00456-4010-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'CITALOPRAM HYDROBROMIDE', 10.0000, 'mg', 'Selective Serotonin Reuptake Inhibitor', NULL, 'TABLET'),
(43, 'Klor-Con', 'Potassium Chloride', 'Upsher-Smith', 'Electrolyte', 0, '00245-0036-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'POTASSIUM CHLORIDE', 8.0000, 'mEq', 'Electrolyte Supplement', NULL, 'TABLET, EXTENDED RELEASE'),
(44, 'Zyloprim', 'Allopurinol', 'Prometheus', 'Antigout', 0, '00656-0301-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ALLOPURINOL', 100.0000, 'mg', 'Xanthine Oxidase Inhibitor', NULL, 'TABLET'),
(45, 'Ecotrin', 'Aspirin', 'Medtech', 'Analgesic', 0, '00135-0158-01', 'HUMAN OTC DRUG', 'ORAL', 'OTC MONOGRAPH FINAL', 'ASPIRIN', 325.0000, 'mg', 'Salicylate', NULL, 'TABLET, ENTERIC COATED'),
(46, 'Flexeril', 'Cyclobenzaprine', 'Merck', 'Muscle Relaxant', 0, '00006-0039-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'CYCLOBENZAPRINE HYDROCHLORIDE', 5.0000, 'mg', 'Skeletal Muscle Relaxant', NULL, 'TABLET'),
(47, 'Drisdol', 'Ergocalciferol', 'Sanofi-Aventis', 'Vitamin', 0, '00024-0391-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ERGOCALCIFEROL', 50000.0000, 'UNIT', 'Vitamin D Analog', NULL, 'CAPSULE'),
(48, 'OxyContin', 'Oxycodone', 'Purdue Pharma', 'Analgesic', 1, '00074-3041-13', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'OXYCODONE HYDROCHLORIDE', 10.0000, 'mg', 'Opioid Agonist', 'C-II', 'TABLET, EXTENDED RELEASE'),
(49, 'Concerta', 'Methylphenidate', 'Janssen', 'CNS Stimulant', 1, '50458-0585-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'METHYLPHENIDATE HYDROCHLORIDE', 18.0000, 'mg', 'Central Nervous System Stimulant', 'C-II', 'TABLET, EXTENDED RELEASE'),
(50, 'Effexor XR', 'Venlafaxine', 'Wyeth', 'Antidepressant', 0, '00008-0833-21', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'VENLAFAXINE HYDROCHLORIDE', 75.0000, 'mg', 'Serotonin Norepinephrine Reuptake Inhibitor', NULL, 'CAPSULE, EXTENDED RELEASE'),
(51, 'Aldactone', 'Spironolactone', 'Pfizer', 'Diuretic', 0, '00025-1001-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'SPIRONOLACTONE', 25.0000, 'mg', 'Aldosterone Receptor Antagonist', NULL, 'TABLET'),
(52, 'Zofran', 'Ondansetron', 'GSK', 'Antiemetic', 0, '00173-0446-00', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ONDANSETRON HYDROCHLORIDE', 4.0000, 'mg', '5-HT3 Receptor Antagonist', NULL, 'TABLET'),
(53, 'Ambien', 'Zolpidem', 'Sanofi-Aventis', 'Sedative', 1, '00024-5401-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ZOLPIDEM TARTRATE', 5.0000, 'mg', 'Non-benzodiazepine Hypnotic', 'C-IV', 'TABLET'),
(54, 'Zyrtec', 'Cetirizine', 'McNeil', 'Antihistamine', 0, '00501-2001-01', 'HUMAN OTC DRUG', 'ORAL', 'NDA', 'CETIRIZINE HYDROCHLORIDE', 10.0000, 'mg', 'H1 Receptor Antagonist', NULL, 'TABLET'),
(55, 'Vivelle-Dot', 'Estradiol', 'Novartis', 'Estrogen', 0, '00078-0346-42', 'HUMAN PRESCRIPTION DRUG', 'TRANSDERMAL', 'NDA', 'ESTRADIOL', 0.0375, 'mg/24hr', 'Estrogen Receptor Agonist', NULL, 'PATCH'),
(56, 'Pravachol', 'Pravastatin', 'BMS', 'Antihyperlipidemic', 0, '00003-0154-05', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'PRAVASTATIN SODIUM', 20.0000, 'mg', 'HMG-CoA Reductase Inhibitor', NULL, 'TABLET'),
(57, 'Zestoretic', 'Lisinopril/HCTZ', 'AstraZeneca', 'Antihypertensive', 0, '00310-0145-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LISINOPRIL / HYDROCHLOROTHIAZIDE', 20.0000, 'mg', 'ACE Inhibitor/Thiazide Combo', NULL, 'TABLET'),
(58, 'Lamictal', 'Lamotrigine', 'GSK', 'Anticonvulsant', 0, '00173-0633-54', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LAMOTRIGINE', 100.0000, 'mg', 'Phenyltriazine Antiepileptic', NULL, 'TABLET'),
(59, 'Seroquel', 'Quetiapine', 'AstraZeneca', 'Antipsychotic', 0, '00310-0271-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'QUETIAPINE FUMARATE', 25.0000, 'mg', 'Atypical Antipsychotic', NULL, 'TABLET'),
(60, 'Advair Diskus', 'Fluticasone/Salmeterol', 'GSK', 'Respiratory', 0, '00173-0695-00', 'HUMAN PRESCRIPTION DRUG', 'RESPIRATORY (INHALATION)', 'NDA', 'FLUTICASONE PROPIONATE / SALMETEROL XINAFOATE', 100.0000, 'mcg', 'Corticosteroid/Beta-2 Agonist', NULL, 'POWDER, METERED'),
(61, 'Klonopin', 'Clonazepam', 'Roche', 'Anticonvulsant', 1, '00004-0001-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'CLONAZEPAM', 0.5000, 'mg', 'Benzodiazepine', 'C-IV', 'TABLET'),
(62, 'Trulicity', 'Dulaglutide', 'Eli Lilly', 'Antidiabetic', 0, '00002-1433-80', 'HUMAN PRESCRIPTION DRUG', 'SUBCUTANEOUS', 'NDA', 'DULAGLUTIDE', 0.7500, 'mg/0.5mL', 'GLP-1 Receptor Agonist', NULL, 'INJECTION, SOLUTION'),
(63, 'Zithromax', 'Azithromycin', 'Pfizer', 'Antibiotic', 0, '00069-3050-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'AZITHROMYCIN ANHYDROUS', 250.0000, 'mg', 'Macrolide Antibacterial', NULL, 'TABLET'),
(64, 'Hyzaar', 'Losartan/HCTZ', 'Merck', 'Antihypertensive', 0, '00006-0717-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LOSARTAN POTASSIUM / HYDROCHLOROTHIAZIDE', 50.0000, 'mg', 'ARB/Thiazide Combo', NULL, 'TABLET'),
(65, 'Augmentin', 'Amoxicillin/Clavulanate', 'GSK', 'Antibiotic', 0, '00029-6086-12', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'AMOXICILLIN / CLAVULANATE POTASSIUM', 875.0000, 'mg', 'Penicillin/Beta-lactamase Inhibitor', NULL, 'TABLET'),
(66, 'Xalatan', 'Latanoprost', 'Pfizer', 'Ophthalmic', 0, '00069-0430-01', 'HUMAN PRESCRIPTION DRUG', 'OPHTHALMIC', 'NDA', 'LATANOPROST', 0.0500, 'mg/mL', 'Prostaglandin F2-alpha Analog', NULL, 'SOLUTION/ DROPS'),
(67, 'Vitamin D3', 'Cholecalciferol', 'Various', 'Vitamin', 0, '00000-0000-00', 'HUMAN OTC DRUG', 'ORAL', 'OTC MONOGRAPH NOT FINAL', 'CHOLECALCIFEROL', 2000.0000, 'UNIT', 'Vitamin D Supplement', NULL, 'CAPSULE'),
(68, 'Inderal', 'Propranolol', 'Wyeth-Ayerst', 'Antihypertensive', 0, '00046-0421-81', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'PROPRANOLOL HYDROCHLORIDE', 10.0000, 'mg', 'Non-selective Beta Blocker', NULL, 'TABLET'),
(69, 'Zetia', 'Ezetimibe', 'Merck/Schering-Plough', 'Antihyperlipidemic', 0, '00006-0414-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'EZETIMIBE', 10.0000, 'mg', 'Cholesterol Absorption Inhibitor', NULL, 'TABLET'),
(70, 'Topamax', 'Topiramate', 'Janssen', 'Anticonvulsant', 0, '50458-0639-60', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'TOPIRAMATE', 25.0000, 'mg', 'Sulfamate-substituted Monosaccharide', NULL, 'TABLET'),
(71, 'Paxil', 'Paroxetine', 'GSK', 'Antidepressant', 0, '00029-3210-13', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'PAROXETINE HYDROCHLORIDE', 20.0000, 'mg', 'Selective Serotonin Reuptake Inhibitor', NULL, 'TABLET'),
(72, 'Voltaren', 'Diclofenac', 'Novartis', 'Analgesic', 0, '00078-0431-05', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'DICLOFENAC SODIUM', 50.0000, 'mg', 'Nonsteroidal Anti-inflammatory Drug', NULL, 'TABLET, DELAYED RELEASE'),
(73, 'Symbicort', 'Budesonide/Formoterol', 'AstraZeneca', 'Respiratory', 0, '00186-0370-20', 'HUMAN PRESCRIPTION DRUG', 'RESPIRATORY (INHALATION)', 'NDA', 'BUDESONIDE / FORMOTEROL FUMARATE DIHYDRATE', 80.0000, 'mcg', 'Corticosteroid/Beta-2 Agonist', NULL, 'AEROSOL, METERED'),
(74, 'Tenormin', 'Atenolol', 'AstraZeneca', 'Antihypertensive', 0, '00310-0105-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ATENOLOL', 50.0000, 'mg', 'Beta-1 Adrenergic Blocker', NULL, 'TABLET'),
(75, 'Vyvanse', 'Lisdexamfetamine', 'Shire', 'CNS Stimulant', 1, '59417-0101-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LISDEXAMFETAMINE DIMESYLATE', 30.0000, 'mg', 'Central Nervous System Stimulant', 'C-II', 'CAPSULE'),
(76, 'Vibramycin', 'Doxycycline', 'Pfizer', 'Antibiotic', 0, '00069-0941-66', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'DOXYCYCLINE HYCLATE', 100.0000, 'mg', 'Tetracycline-class Antibacterial', NULL, 'CAPSULE'),
(77, 'Lyrica', 'Pregabalin', 'Pfizer', 'Anticonvulsant', 1, '00071-1012-68', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'PREGABALIN', 75.0000, 'mg', 'Gamma-Aminobutyric Acid Analog', 'C-V', 'CAPSULE'),
(78, 'Amaryl', 'Glimepiride', 'Sanofi-Aventis', 'Antidiabetic', 0, '00039-0221-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'GLIMEPIRIDE', 1.0000, 'mg', 'Sulfonylurea', NULL, 'TABLET'),
(79, 'Zanaflex', 'Tizanidine', 'Acanas', 'Muscle Relaxant', 0, '00703-4631-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'TIZANIDINE HYDROCHLORIDE', 2.0000, 'mg', 'Alpha-2 Adrenergic Agonist', NULL, 'TABLET'),
(80, 'Catapres', 'Clonidine', 'Boehringer Ingelheim', 'Antihypertensive', 0, '00597-0006-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'CLONIDINE HYDROCHLORIDE', 0.1000, 'mg', 'Alpha-2 Adrenergic Agonist', NULL, 'TABLET'),
(81, 'Tricor', 'Fenofibrate', 'Abbott', 'Antihyperlipidemic', 0, '00074-3424-90', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'FENOFIBRATE', 145.0000, 'mg', 'Fibric Acid Derivative', NULL, 'TABLET'),
(82, 'Humalog', 'Insulin Lispro', 'Eli Lilly', 'Antidiabetic', 0, '00002-7510-01', 'HUMAN PRESCRIPTION DRUG', 'SUBCUTANEOUS', 'NDA', 'INSULIN LISPRO', 100.0000, 'UNIT/mL', 'Rapid-acting Insulin', NULL, 'INJECTION, SOLUTION'),
(83, 'Diovan', 'Valsartan', 'Novartis', 'Antihypertensive', 0, '00078-0358-34', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'VALSARTAN', 80.0000, 'mg', 'Angiotensin II Receptor Antagonist', NULL, 'TABLET'),
(84, 'Keflex', 'Cephalexin', 'Advancis', 'Antibiotic', 0, '00143-3145-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'CEPHALEXIN', 500.0000, 'mg', 'Cephalosporin-class Antibacterial', NULL, 'CAPSULE'),
(85, 'Lioresal', 'Baclofen', 'Novartis', 'Muscle Relaxant', 0, '00078-0660-05', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'BACLOFEN', 10.0000, 'mg', 'GABA-B Agonist', NULL, 'TABLET'),
(86, 'Xarelto', 'Rivaroxaban', 'Janssen', 'Anticoagulant', 0, '50458-0577-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'RIVAROXABAN', 20.0000, 'mg', 'Factor Xa Inhibitor', NULL, 'TABLET'),
(87, 'Feosol', 'Ferrous Sulfate', 'Meda', 'Hematinic', 0, '00000-0000-00', 'HUMAN OTC DRUG', 'ORAL', 'OTC MONOGRAPH FINAL', 'FERROUS SULFATE', 325.0000, 'mg', 'Iron Supplement', NULL, 'TABLET'),
(88, 'Elavil', 'Amitriptyline', 'AstraZeneca', 'Antidepressant', 0, '00310-0121-10', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'AMITRIPTYLINE HYDROCHLORIDE', 25.0000, 'mg', 'Tricyclic Antidepressant', NULL, 'TABLET'),
(89, 'Proscar', 'Finasteride', 'Merck', 'Urological', 0, '00006-0071-31', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'FINASTERIDE', 5.0000, 'mg', '5-alpha Reductase Inhibitor', NULL, 'TABLET'),
(90, 'Farxiga', 'Dapagliflozin', 'AstraZeneca', 'Antidiabetic', 0, '00310-6170-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'DAPAGLIFLOZIN', 10.0000, 'mg', 'SGLT2 Inhibitor', NULL, 'TABLET'),
(91, 'Percocet', 'Oxycodone/APAP', 'Endo', 'Analgesic', 1, '63481-0623-70', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'OXYCODONE HYDROCHLORIDE / ACETAMINOPHEN', 5.0000, 'mg', 'Opioid Agonist', 'C-II', 'TABLET'),
(92, 'Folvite', 'Folic Acid', 'Lederle', 'Vitamin', 0, '00005-0000-00', 'HUMAN OTC DRUG', 'ORAL', 'OTC MONOGRAPH FINAL', 'FOLIC ACID', 1.0000, 'mg', 'B-complex Vitamin', NULL, 'TABLET'),
(93, 'Abilify', 'Aripiprazole', 'Otsuka', 'Antipsychotic', 0, '59148-0006-13', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'ARIPIPRAZOLE', 5.0000, 'mg', 'Atypical Antipsychotic', NULL, 'TABLET'),
(94, 'Benicar', 'Olmesartan', 'Daiichi Sankyo', 'Antihypertensive', 0, '65597-0101-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'OLMESARTAN MEDOXOMIL', 20.0000, 'mg', 'Angiotensin II Receptor Antagonist', NULL, 'TABLET'),
(95, 'Valtrex', 'Valacyclovir', 'GSK', 'Antiviral', 0, '00173-0565-04', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'VALACYCLOVIR HYDROCHLORIDE', 500.0000, 'mg', 'Nucleoside Analog DNA Polymerase Inhibitor', NULL, 'TABLET'),
(96, 'Remeron', 'Mirtazapine', 'Organon', 'Antidepressant', 0, '00052-0104-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'MIRTAZAPINE', 15.0000, 'mg', 'Tetracyclic Antidepressant', NULL, 'TABLET'),
(97, 'Ativan', 'Lorazepam', 'Wyeth', 'Antianxiety', 1, '00008-0064-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'LORAZEPAM', 1.0000, 'mg', 'Benzodiazepine', 'C-IV', 'TABLET'),
(98, 'Suboxone', 'Buprenorphine/Naloxone', 'Indivior', 'Opioid Dependence', 1, '12496-1202-03', 'HUMAN PRESCRIPTION DRUG', 'SUBLINGUAL', 'NDA', 'BUPRENORPHINE HYDROCHLORIDE / NALOXONE HYDROCHLORIDE', 2.0000, 'mg', 'Partial Opioid Agonist', 'C-III', 'FILM'),
(99, 'Wellbutrin XL', 'Bupropion XL', 'Valeant', 'Antidepressant', 0, '00187-0731-30', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'BUPROPION HYDROCHLORIDE', 300.0000, 'mg', 'Aminoketone', NULL, 'TABLET, EXTENDED RELEASE'),
(100, 'Valium', 'Diazepam', 'Roche', 'Antianxiety', 1, '00004-0002-01', 'HUMAN PRESCRIPTION DRUG', 'ORAL', 'NDA', 'DIAZEPAM', 5.0000, 'mg', 'Benzodiazepine', 'C-IV', 'TABLET');

-- --------------------------------------------------------

--
-- Table structure for table `notifications`
--

CREATE TABLE `notifications` (
  `id` char(36) NOT NULL,
  `recipient_id` char(36) NOT NULL,
  `notification_type_id` char(36) DEFAULT NULL,
  `title` varchar(150) DEFAULT NULL,
  `message` text DEFAULT NULL,
  `is_read` tinyint(1) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `notification_types`
--

CREATE TABLE `notification_types` (
  `id` char(36) NOT NULL,
  `type_name` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `password_reset_tokens`
--

CREATE TABLE `password_reset_tokens` (
  `id` char(36) NOT NULL,
  `user_id` char(36) NOT NULL,
  `token_hash` varchar(255) NOT NULL,
  `expires_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `used_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `relationship_types`
--

CREATE TABLE `relationship_types` (
  `id` char(36) NOT NULL,
  `relationship` varchar(50) NOT NULL,
  `description` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `relationship_types`
--

INSERT INTO `relationship_types` (`id`, `relationship`, `description`) VALUES
('R001', 'parent', NULL),
('R002', 'spouse', NULL),
('R003', 'sibling', NULL),
('R004', 'child', NULL),
('R005', 'professional_caregiver', NULL);

-- --------------------------------------------------------

--
-- Table structure for table `rovers`
--

CREATE TABLE `rovers` (
  `id` char(36) NOT NULL,
  `user_id` char(36) NOT NULL,
  `date_of_birth` date DEFAULT NULL,
  `gender` enum('male','female','other') DEFAULT NULL,
  `address_id` char(36) DEFAULT NULL,
  `primary_caregiver_id` char(36) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `rovers`
--

INSERT INTO `rovers` (`id`, `user_id`, `date_of_birth`, `gender`, `address_id`, `primary_caregiver_id`) VALUES
('a63df65d-0743-40a3-9038-deb1bbca4ccd', '4cf57fae-76f6-43b9-aee1-e5f4fb3f4eb3', '1980-08-20', 'male', NULL, NULL),
('bf3fed8e-d9d8-4f45-837a-f5d028e3a0d8', 'd92c5c53-9ccf-4510-b769-b342a6486e60', '1995-01-15', 'male', NULL, NULL),
('RV001', 'U001', '1995-05-15', 'male', 'AD001', 'CG001'),
('RV002', 'U004', '1988-11-20', 'female', 'AD002', 'CG002'),
('RV003', 'U007', '2000-01-10', 'male', 'AD003', 'CG003');

-- --------------------------------------------------------

--
-- Table structure for table `rover_allergies`
--

CREATE TABLE `rover_allergies` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `allergy_id` char(36) NOT NULL,
  `severity` enum('mild','moderate','severe') DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT 1
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `rover_health_conditions`
--

CREATE TABLE `rover_health_conditions` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `condition_id` char(36) NOT NULL,
  `severity` enum('mild','moderate','severe') DEFAULT 'moderate',
  `notes` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `rover_medications`
--

CREATE TABLE `rover_medications` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `medication_id` char(36) NOT NULL,
  `dosage` varchar(100) DEFAULT NULL,
  `frequency` varchar(100) DEFAULT NULL,
  `start_date` date DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT 1
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `secure_messages`
--

CREATE TABLE `secure_messages` (
  `id` char(36) NOT NULL,
  `sender_id` char(36) NOT NULL,
  `recipient_id` char(36) NOT NULL,
  `message_body` text DEFAULT NULL,
  `is_read` tinyint(1) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `sessions`
--

CREATE TABLE `sessions` (
  `id` char(36) NOT NULL,
  `user_id` char(36) NOT NULL,
  `access_token` text NOT NULL,
  `refresh_token` text NOT NULL,
  `device_info_id` char(36) DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL,
  `expires_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `is_active` tinyint(1) DEFAULT 1
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `sessions`
--

INSERT INTO `sessions` (`id`, `user_id`, `access_token`, `refresh_token`, `device_info_id`, `ip_address`, `expires_at`, `is_active`) VALUES
('32471d88-3bab-4e33-baa1-f35b0d9c3249', '4cf57fae-76f6-43b9-aee1-e5f4fb3f4eb3', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0Y2Y1N2ZhZS03NmY2LTQzYjktYWVlMS1lNWY0ZmIzZjRlYjMiLCJyb2xlcyI6WyJyb3ZlciJdLCJ0eXBlIjoiYWNjZXNzIiwiaWF0IjoxNzc2MjAyNDQ4LCJleHAiOjE3NzYyMDMzNDgsImp0aSI6IjRjNTQxMjNkLTM4MGYtNDdjNC1hZWVkLWY3N2Y4ZTI3NzAxYiJ9.hjE5QJT-PlStkOith24KyRGW8m_pX0YhUZ7BSUigUWY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0Y2Y1N2ZhZS03NmY2LTQzYjktYWVlMS1lNWY0ZmIzZjRlYjMiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc3NjIwMjQ0OCwiZXhwIjoxNzc4Nzk0NDQ4LCJqdGkiOiJjNDU0YWFlYS0zMGI3LTQ4MTUtYmM1YS01NDQzOTcwZGIyYmIifQ.xDKVQU4uOgyDTvdu863K-WpdCuEsCitE3RiLxhcTJYc', NULL, '127.0.0.1', '2026-04-14 21:34:08', 1),
('6c46eec7-6355-45f8-87cf-44caad4d13da', 'd92c5c53-9ccf-4510-b769-b342a6486e60', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkOTJjNWM1My05Y2NmLTQ1MTAtYjc2OS1iMzQyYTY0ODZlNjAiLCJyb2xlcyI6WyJyb3ZlciJdLCJ0eXBlIjoiYWNjZXNzIiwiaWF0IjoxNzc2MjAyMTkxLCJleHAiOjE3NzYyMDMwOTEsImp0aSI6IjZkZGFiM2I5LTVmZjctNDgzYy05NTBhLTYwYzUxNDNiZDllMyJ9.mMNs1AB6M8abxP2Y162efohMv4ruL9ZsUKEky1q-iNo', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkOTJjNWM1My05Y2NmLTQ1MTAtYjc2OS1iMzQyYTY0ODZlNjAiLCJ0eXBlIjoicmVmcmVzaCIsImlhdCI6MTc3NjIwMjE5MSwiZXhwIjoxNzc4Nzk0MTkxLCJqdGkiOiI2NzAxM2FjZC1hMWUxLTRhYTctODQyMi0zMGU3ZTg0YjI1YTUifQ.VhOPPQhnUyIq_C0OC8wlja8cvB6aBVYW3kH4ccx-vt0', NULL, '127.0.0.1', '2026-04-14 21:29:51', 1);

-- --------------------------------------------------------

--
-- Table structure for table `specializations`
--

CREATE TABLE `specializations` (
  `id` char(36) NOT NULL,
  `name` varchar(100) NOT NULL,
  `description` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `specializations`
--

INSERT INTO `specializations` (`id`, `name`, `description`) VALUES
('SP01', 'Cardiology', 'Heart and blood vessel specialist'),
('SP02', 'Neurology', 'Brain and nervous system specialist'),
('SP03', 'Pediatrics', 'Medical care for infants and children'),
('SP04', 'Psychiatry', 'Mental health and behavioral specialist'),
('SP05', 'Orthopedics', 'Musculoskeletal system specialist'),
('SP06', 'Dermatology', 'Skin, hair, and nail specialist'),
('SP07', 'Endocrinology', 'Hormone and gland specialist'),
('SP08', 'General Practice', 'Primary healthcare provider'),
('SP09', 'Geriatrics', 'Elderly care specialist'),
('SP10', 'Oncology', 'Cancer treatment specialist'),
('SP11', 'Ophthalmology', 'Eye and vision specialist'),
('SP12', 'Gastroenterology', 'Digestive system specialist'),
('SP13', 'Pulmonology', 'Respiratory system specialist'),
('SP14', 'Radiology', 'Medical imaging specialist'),
('SP15', 'Rehabilitation', 'Physical therapy and recovery');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` char(36) NOT NULL,
  `email` varchar(255) NOT NULL,
  `hashed_password` varchar(255) DEFAULT NULL,
  `google_id` varchar(255) DEFAULT NULL,
  `first_name` varchar(100) NOT NULL,
  `last_name` varchar(100) NOT NULL,
  `profile_picture_url` text DEFAULT NULL,
  `is_email_verified` tinyint(1) DEFAULT 0,
  `email_verified_at` timestamp NULL DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `email`, `hashed_password`, `google_id`, `first_name`, `last_name`, `profile_picture_url`, `is_email_verified`, `email_verified_at`, `is_active`, `created_at`, `updated_at`) VALUES
('4cf57fae-76f6-43b9-aee1-e5f4fb3f4eb3', 'john.smith@novacare.demo', '$2b$12$sTaXkz1SF7aBkNr3Yfi6HOXYOGELyTF12SxxXBNyQ0QaH0Nrb7aby', NULL, 'John', 'Smith', NULL, 0, NULL, 1, '2026-04-14 19:34:07', '2026-04-14 19:34:07'),
('b9e300cb-6395-4abc-943e-60371d563eb4', 'testuser@novacare.demo', '$2b$12$4.NFkDdCpqfawYC/V8w78e.Tk5YjYc9HDIzSuf5dkIx7ljOFYNpFe', NULL, 'Test', 'User', NULL, 0, NULL, 1, '2026-04-14 19:17:08', '2026-04-14 19:17:08'),
('d92c5c53-9ccf-4510-b769-b342a6486e60', 'test_final@novacare.demo', '$2b$12$yLbJf/xvSOSU2PEo5xyvSubhNa6ifmdpxddhZQ8DT6rwSeNxd0RnC', NULL, 'Test', 'User', NULL, 0, NULL, 1, '2026-04-14 19:29:51', '2026-04-14 19:29:51'),
('f63ba7cd-561e-4833-b017-a7a3acd2f7e3', 'basant@novacare.test', '$2b$12$dmar67B0ZiuvTKNuYGIZ6.boZRbA7nVcWl3FMAcFuJPJ6W832Qjye', NULL, 'Basant', 'Admin', NULL, 0, NULL, 1, '2026-04-14 19:22:03', '2026-04-14 19:22:03'),
('U001', 'alex.smith@email.com', NULL, NULL, 'Alex', 'Smith', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U002', 'sarah.jones@email.com', NULL, NULL, 'Sarah', 'Jones', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U003', 'ahmed.doctor@clinic.eg', NULL, NULL, 'Ahmed', 'Shalaby', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U004', 'maria.g@provider.net', NULL, NULL, 'Maria', 'Garcia', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U005', 'youssef.b@care.com', NULL, NULL, 'Youssef', 'Bedair', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U006', 'linda.v@rehab.com', NULL, NULL, 'Linda', 'Vazquez', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U007', 'omar.k@proton.me', NULL, NULL, 'Omar', 'Khalil', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U008', 'mona.r@family.org', NULL, NULL, 'Mona', 'Rashad', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U009', 'zaki.neuro@medical.eg', NULL, NULL, 'Zaki', 'Mansour', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U010', 'laila.n@gmail.com', NULL, NULL, 'Laila', 'Nour', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U011', 'basant.dev@aiu.edu.eg', NULL, NULL, 'Basant', 'Awad', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U012', 'nadira.s@aiu.edu.eg', NULL, NULL, 'Nadira', 'Sami', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U013', 'nourine.y@aiu.edu.eg', NULL, NULL, 'Nourine', 'Yasser', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U014', 'mohamed.m@aiu.edu.eg', NULL, NULL, 'Mohamed', 'Mosaad', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07'),
('U015', 'ramez.a@aiu.edu.eg', NULL, NULL, 'Ramez', 'Asaad', NULL, 1, NULL, 1, '2026-04-14 18:50:07', '2026-04-14 18:50:07');

-- --------------------------------------------------------

--
-- Table structure for table `user_roles`
--

CREATE TABLE `user_roles` (
  `id` char(36) NOT NULL,
  `user_id` char(36) NOT NULL,
  `role` enum('rover','caregiver','doctor') NOT NULL,
  `is_active` tinyint(1) DEFAULT 1
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `user_roles`
--

INSERT INTO `user_roles` (`id`, `user_id`, `role`, `is_active`) VALUES
('585cbcef-a247-42f4-8faf-1431a4846cbc', '4cf57fae-76f6-43b9-aee1-e5f4fb3f4eb3', 'rover', 1),
('9db51fa5-0d93-4f4e-a91e-adab62494cec', 'f63ba7cd-561e-4833-b017-a7a3acd2f7e3', 'rover', 1),
('bb5d4110-b45c-4270-b589-de710dd5a14d', 'd92c5c53-9ccf-4510-b769-b342a6486e60', 'rover', 1),
('RL001', 'U001', 'rover', 1),
('RL002', 'U002', 'caregiver', 1),
('RL003', 'U003', 'doctor', 1),
('RL004', 'U004', 'rover', 1),
('RL005', 'U005', 'caregiver', 1),
('RL006', 'U006', 'doctor', 1),
('RL007', 'U007', 'rover', 1),
('RL008', 'U008', 'caregiver', 1),
('RL009', 'U009', 'doctor', 1),
('RL010', 'U010', 'rover', 1),
('RL011', 'U011', 'caregiver', 1),
('RL012', 'U012', 'doctor', 1),
('RL013', 'U013', 'rover', 1),
('RL014', 'U014', 'caregiver', 1),
('RL015', 'U015', 'doctor', 1),
('UR-001', 'U001', 'rover', 1);

-- --------------------------------------------------------

--
-- Table structure for table `verification_statuses`
--

CREATE TABLE `verification_statuses` (
  `id` char(36) NOT NULL,
  `status_name` varchar(50) NOT NULL,
  `display_label` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `verification_statuses`
--

INSERT INTO `verification_statuses` (`id`, `status_name`, `display_label`) VALUES
('V001', 'pending', 'Verification Pending'),
('V002', 'verified', 'Identity Verified'),
('V003', 'rejected', 'Verification Rejected');

-- --------------------------------------------------------

--
-- Table structure for table `vital_signs`
--

CREATE TABLE `vital_signs` (
  `id` char(36) NOT NULL,
  `rover_id` char(36) NOT NULL,
  `heart_rate` int(11) DEFAULT NULL,
  `spo2` decimal(5,2) DEFAULT NULL,
  `temperature` decimal(5,2) DEFAULT NULL,
  `measured_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `measurement_device_id` char(36) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `vital_signs`
--

INSERT INTO `vital_signs` (`id`, `rover_id`, `heart_rate`, `spo2`, `temperature`, `measured_at`, `measurement_device_id`) VALUES
('VT001', 'RV001', 72, 98.50, 36.60, '2026-04-14 18:50:07', NULL),
('VT002', 'RV001', 75, 98.00, 36.70, '2026-04-14 17:50:07', NULL),
('VT003', 'RV001', 80, 97.50, 36.80, '2026-04-14 16:50:07', NULL),
('VT004', 'RV002', 68, 99.00, 36.50, '2026-04-14 18:50:07', NULL),
('VT005', 'RV002', 70, 99.00, 36.50, '2026-04-14 17:50:07', NULL),
('VT006', 'RV002', 72, 98.50, 36.60, '2026-04-14 16:50:07', NULL),
('VT007', 'RV003', 90, 96.00, 37.10, '2026-04-14 18:50:07', NULL),
('VT008', 'RV003', 92, 95.50, 37.20, '2026-04-14 17:50:07', NULL),
('VT009', 'RV003', 88, 96.50, 37.00, '2026-04-14 16:50:07', NULL),
('VT010', 'RV001', 74, 98.20, 36.60, '2026-04-14 15:50:07', NULL),
('VT011', 'RV001', 71, 98.80, 36.50, '2026-04-14 14:50:07', NULL),
('VT012', 'RV002', 69, 99.10, 36.40, '2026-04-14 15:50:07', NULL),
('VT013', 'RV002', 71, 98.90, 36.50, '2026-04-14 14:50:07', NULL),
('VT014', 'RV003', 85, 97.00, 36.90, '2026-04-14 15:50:07', NULL),
('VT015', 'RV003', 82, 97.50, 36.80, '2026-04-14 14:50:07', NULL);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `action_statuses`
--
ALTER TABLE `action_statuses`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `status_name` (`status_name`);

--
-- Indexes for table `action_types`
--
ALTER TABLE `action_types`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `action_name` (`action_name`);

--
-- Indexes for table `addresses`
--
ALTER TABLE `addresses`
  ADD PRIMARY KEY (`id`),
  ADD KEY `country_id` (`country_id`);

--
-- Indexes for table `ai_interactions`
--
ALTER TABLE `ai_interactions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`);

--
-- Indexes for table `allergies`
--
ALTER TABLE `allergies`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indexes for table `appointments`
--
ALTER TABLE `appointments`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`),
  ADD KEY `doctor_id` (`doctor_id`),
  ADD KEY `appointment_type_id` (`appointment_type_id`),
  ADD KEY `status_id` (`status_id`);

--
-- Indexes for table `appointment_statuses`
--
ALTER TABLE `appointment_statuses`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `status_name` (`status_name`);

--
-- Indexes for table `appointment_types`
--
ALTER TABLE `appointment_types`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `type_name` (`type_name`);

--
-- Indexes for table `approval_statuses`
--
ALTER TABLE `approval_statuses`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `status_name` (`status_name`);

--
-- Indexes for table `audit_logs`
--
ALTER TABLE `audit_logs`
  ADD PRIMARY KEY (`id`),
  ADD KEY `actor_id` (`actor_id`),
  ADD KEY `action_type_id` (`action_type_id`),
  ADD KEY `action_status_id` (`action_status_id`);

--
-- Indexes for table `caregivers`
--
ALTER TABLE `caregivers`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`),
  ADD KEY `address_id` (`address_id`),
  ADD KEY `verification_status_id` (`verification_status_id`);

--
-- Indexes for table `caregiver_rover_assignments`
--
ALTER TABLE `caregiver_rover_assignments`
  ADD PRIMARY KEY (`id`),
  ADD KEY `caregiver_id` (`caregiver_id`),
  ADD KEY `rover_id` (`rover_id`),
  ADD KEY `relationship_type_id` (`relationship_type_id`);

--
-- Indexes for table `clinic_organizations`
--
ALTER TABLE `clinic_organizations`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD KEY `address_id` (`address_id`);

--
-- Indexes for table `countries`
--
ALTER TABLE `countries`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD UNIQUE KEY `iso_code` (`iso_code`);

--
-- Indexes for table `device_info`
--
ALTER TABLE `device_info`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `doctors`
--
ALTER TABLE `doctors`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`),
  ADD KEY `verification_status_id` (`verification_status_id`);

--
-- Indexes for table `email_verification_tokens`
--
ALTER TABLE `email_verification_tokens`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `emergency_contacts`
--
ALTER TABLE `emergency_contacts`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`);

--
-- Indexes for table `emotion_tracking`
--
ALTER TABLE `emotion_tracking`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`);

--
-- Indexes for table `health_conditions`
--
ALTER TABLE `health_conditions`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`),
  ADD UNIQUE KEY `icd10_code` (`icd10_code`);

--
-- Indexes for table `identity_documents`
--
ALTER TABLE `identity_documents`
  ADD PRIMARY KEY (`id`),
  ADD KEY `caregiver_id` (`caregiver_id`);

--
-- Indexes for table `id_types`
--
ALTER TABLE `id_types`
  ADD PRIMARY KEY (`id`),
  ADD KEY `country_id` (`country_id`);

--
-- Indexes for table `main_disability`
--
ALTER TABLE `main_disability`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`);

--
-- Indexes for table `measurement_devices`
--
ALTER TABLE `measurement_devices`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`);

--
-- Indexes for table `medical_license_documents`
--
ALTER TABLE `medical_license_documents`
  ADD PRIMARY KEY (`id`),
  ADD KEY `doctor_id` (`doctor_id`);

--
-- Indexes for table `medical_notes`
--
ALTER TABLE `medical_notes`
  ADD PRIMARY KEY (`id`),
  ADD KEY `doctor_id` (`doctor_id`),
  ADD KEY `rover_id` (`rover_id`),
  ADD KEY `note_type_id` (`note_type_id`);

--
-- Indexes for table `medical_note_types`
--
ALTER TABLE `medical_note_types`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `note_type` (`note_type`);

--
-- Indexes for table `medication_catalog`
--
ALTER TABLE `medication_catalog`
  ADD PRIMARY KEY (`med_id`);

--
-- Indexes for table `notifications`
--
ALTER TABLE `notifications`
  ADD PRIMARY KEY (`id`),
  ADD KEY `recipient_id` (`recipient_id`),
  ADD KEY `notification_type_id` (`notification_type_id`);

--
-- Indexes for table `notification_types`
--
ALTER TABLE `notification_types`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `type_name` (`type_name`);

--
-- Indexes for table `password_reset_tokens`
--
ALTER TABLE `password_reset_tokens`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `relationship_types`
--
ALTER TABLE `relationship_types`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `relationship` (`relationship`);

--
-- Indexes for table `rovers`
--
ALTER TABLE `rovers`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`),
  ADD KEY `address_id` (`address_id`),
  ADD KEY `primary_caregiver_id` (`primary_caregiver_id`);

--
-- Indexes for table `rover_allergies`
--
ALTER TABLE `rover_allergies`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`),
  ADD KEY `allergy_id` (`allergy_id`);

--
-- Indexes for table `rover_health_conditions`
--
ALTER TABLE `rover_health_conditions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`),
  ADD KEY `condition_id` (`condition_id`);

--
-- Indexes for table `rover_medications`
--
ALTER TABLE `rover_medications`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`);

--
-- Indexes for table `secure_messages`
--
ALTER TABLE `secure_messages`
  ADD PRIMARY KEY (`id`),
  ADD KEY `sender_id` (`sender_id`),
  ADD KEY `recipient_id` (`recipient_id`);

--
-- Indexes for table `sessions`
--
ALTER TABLE `sessions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `device_info_id` (`device_info_id`);

--
-- Indexes for table `specializations`
--
ALTER TABLE `specializations`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `user_roles`
--
ALTER TABLE `user_roles`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `verification_statuses`
--
ALTER TABLE `verification_statuses`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `status_name` (`status_name`);

--
-- Indexes for table `vital_signs`
--
ALTER TABLE `vital_signs`
  ADD PRIMARY KEY (`id`),
  ADD KEY `rover_id` (`rover_id`),
  ADD KEY `measurement_device_id` (`measurement_device_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `medication_catalog`
--
ALTER TABLE `medication_catalog`
  MODIFY `med_id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=101;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `addresses`
--
ALTER TABLE `addresses`
  ADD CONSTRAINT `addresses_ibfk_1` FOREIGN KEY (`country_id`) REFERENCES `countries` (`id`);

--
-- Constraints for table `ai_interactions`
--
ALTER TABLE `ai_interactions`
  ADD CONSTRAINT `ai_interactions_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`);

--
-- Constraints for table `appointments`
--
ALTER TABLE `appointments`
  ADD CONSTRAINT `appointments_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`),
  ADD CONSTRAINT `appointments_ibfk_2` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`),
  ADD CONSTRAINT `appointments_ibfk_3` FOREIGN KEY (`appointment_type_id`) REFERENCES `appointment_types` (`id`),
  ADD CONSTRAINT `appointments_ibfk_4` FOREIGN KEY (`status_id`) REFERENCES `appointment_statuses` (`id`);

--
-- Constraints for table `audit_logs`
--
ALTER TABLE `audit_logs`
  ADD CONSTRAINT `audit_logs_ibfk_1` FOREIGN KEY (`actor_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `audit_logs_ibfk_2` FOREIGN KEY (`action_type_id`) REFERENCES `action_types` (`id`),
  ADD CONSTRAINT `audit_logs_ibfk_3` FOREIGN KEY (`action_status_id`) REFERENCES `action_statuses` (`id`);

--
-- Constraints for table `caregivers`
--
ALTER TABLE `caregivers`
  ADD CONSTRAINT `caregivers_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `caregivers_ibfk_2` FOREIGN KEY (`address_id`) REFERENCES `addresses` (`id`),
  ADD CONSTRAINT `caregivers_ibfk_3` FOREIGN KEY (`verification_status_id`) REFERENCES `verification_statuses` (`id`);

--
-- Constraints for table `caregiver_rover_assignments`
--
ALTER TABLE `caregiver_rover_assignments`
  ADD CONSTRAINT `caregiver_rover_assignments_ibfk_1` FOREIGN KEY (`caregiver_id`) REFERENCES `caregivers` (`id`),
  ADD CONSTRAINT `caregiver_rover_assignments_ibfk_2` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`),
  ADD CONSTRAINT `caregiver_rover_assignments_ibfk_3` FOREIGN KEY (`relationship_type_id`) REFERENCES `relationship_types` (`id`);

--
-- Constraints for table `clinic_organizations`
--
ALTER TABLE `clinic_organizations`
  ADD CONSTRAINT `clinic_organizations_ibfk_1` FOREIGN KEY (`address_id`) REFERENCES `addresses` (`id`);

--
-- Constraints for table `device_info`
--
ALTER TABLE `device_info`
  ADD CONSTRAINT `device_info_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `doctors`
--
ALTER TABLE `doctors`
  ADD CONSTRAINT `doctors_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `doctors_ibfk_2` FOREIGN KEY (`verification_status_id`) REFERENCES `verification_statuses` (`id`);

--
-- Constraints for table `email_verification_tokens`
--
ALTER TABLE `email_verification_tokens`
  ADD CONSTRAINT `email_verification_tokens_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `emergency_contacts`
--
ALTER TABLE `emergency_contacts`
  ADD CONSTRAINT `emergency_contacts_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`);

--
-- Constraints for table `emotion_tracking`
--
ALTER TABLE `emotion_tracking`
  ADD CONSTRAINT `emotion_tracking_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`);

--
-- Constraints for table `identity_documents`
--
ALTER TABLE `identity_documents`
  ADD CONSTRAINT `identity_documents_ibfk_1` FOREIGN KEY (`caregiver_id`) REFERENCES `caregivers` (`id`);

--
-- Constraints for table `id_types`
--
ALTER TABLE `id_types`
  ADD CONSTRAINT `id_types_ibfk_1` FOREIGN KEY (`country_id`) REFERENCES `countries` (`id`);

--
-- Constraints for table `main_disability`
--
ALTER TABLE `main_disability`
  ADD CONSTRAINT `main_disability_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `measurement_devices`
--
ALTER TABLE `measurement_devices`
  ADD CONSTRAINT `measurement_devices_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`);

--
-- Constraints for table `medical_license_documents`
--
ALTER TABLE `medical_license_documents`
  ADD CONSTRAINT `medical_license_documents_ibfk_1` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`);

--
-- Constraints for table `medical_notes`
--
ALTER TABLE `medical_notes`
  ADD CONSTRAINT `medical_notes_ibfk_1` FOREIGN KEY (`doctor_id`) REFERENCES `doctors` (`id`),
  ADD CONSTRAINT `medical_notes_ibfk_2` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`),
  ADD CONSTRAINT `medical_notes_ibfk_3` FOREIGN KEY (`note_type_id`) REFERENCES `medical_note_types` (`id`);

--
-- Constraints for table `notifications`
--
ALTER TABLE `notifications`
  ADD CONSTRAINT `notifications_ibfk_1` FOREIGN KEY (`recipient_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `notifications_ibfk_2` FOREIGN KEY (`notification_type_id`) REFERENCES `notification_types` (`id`);

--
-- Constraints for table `password_reset_tokens`
--
ALTER TABLE `password_reset_tokens`
  ADD CONSTRAINT `password_reset_tokens_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `rovers`
--
ALTER TABLE `rovers`
  ADD CONSTRAINT `rovers_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `rovers_ibfk_2` FOREIGN KEY (`address_id`) REFERENCES `addresses` (`id`),
  ADD CONSTRAINT `rovers_ibfk_3` FOREIGN KEY (`primary_caregiver_id`) REFERENCES `caregivers` (`id`);

--
-- Constraints for table `rover_allergies`
--
ALTER TABLE `rover_allergies`
  ADD CONSTRAINT `rover_allergies_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`),
  ADD CONSTRAINT `rover_allergies_ibfk_2` FOREIGN KEY (`allergy_id`) REFERENCES `allergies` (`id`);

--
-- Constraints for table `rover_health_conditions`
--
ALTER TABLE `rover_health_conditions`
  ADD CONSTRAINT `rover_health_conditions_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `rover_health_conditions_ibfk_2` FOREIGN KEY (`condition_id`) REFERENCES `health_conditions` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `rover_medications`
--
ALTER TABLE `rover_medications`
  ADD CONSTRAINT `rover_medications_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`);

--
-- Constraints for table `secure_messages`
--
ALTER TABLE `secure_messages`
  ADD CONSTRAINT `secure_messages_ibfk_1` FOREIGN KEY (`sender_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `secure_messages_ibfk_2` FOREIGN KEY (`recipient_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `sessions`
--
ALTER TABLE `sessions`
  ADD CONSTRAINT `sessions_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `sessions_ibfk_2` FOREIGN KEY (`device_info_id`) REFERENCES `device_info` (`id`);

--
-- Constraints for table `user_roles`
--
ALTER TABLE `user_roles`
  ADD CONSTRAINT `user_roles_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `vital_signs`
--
ALTER TABLE `vital_signs`
  ADD CONSTRAINT `vital_signs_ibfk_1` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`),
  ADD CONSTRAINT `vital_signs_ibfk_2` FOREIGN KEY (`measurement_device_id`) REFERENCES `measurement_devices` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
