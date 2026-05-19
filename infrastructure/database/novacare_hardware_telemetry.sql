-- ============================================================================
-- NovaCare IoT & Hardware Telemetry Database Extension Schema
-- Dialect: MariaDB / MySQL (InnoDB, UTF-8)
-- ============================================================================

-- 1. Hardware Devices Table (Static Identity Registry)
CREATE TABLE IF NOT EXISTS `hardware_devices` (
  `id` CHAR(36) NOT NULL,
  `rover_id` CHAR(36) NOT NULL,
  `device_name` VARCHAR(100) NOT NULL,
  `device_type` VARCHAR(100) NOT NULL,
  `firmware_version` VARCHAR(50) DEFAULT NULL,
  `serial_number` VARCHAR(100) DEFAULT NULL,
  `status` ENUM('online', 'offline', 'maintenance') DEFAULT 'offline',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_hd_rover` (`rover_id`),
  CONSTRAINT `hd_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 2. Rover Telemetry Table (Time-series System logs)
CREATE TABLE IF NOT EXISTS `rover_telemetry` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `rover_id` CHAR(36) NOT NULL,
  `battery_percentage` DECIMAL(5,2) DEFAULT NULL,
  `cpu_usage` DECIMAL(5,2) DEFAULT NULL,
  `ram_usage` DECIMAL(5,2) DEFAULT NULL,
  `temperature` DECIMAL(5,2) DEFAULT NULL,
  `wifi_signal_strength` INT DEFAULT NULL,
  `network_latency_ms` INT DEFAULT NULL,
  `is_charging` TINYINT(1) NOT NULL DEFAULT 0,
  `emergency_mode` TINYINT(1) NOT NULL DEFAULT 0,
  `recorded_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_rt_rover_time` (`rover_id`, `recorded_at`),
  CONSTRAINT `rt_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 3. Sensor Readings Table (Time-series environment telemetry)
CREATE TABLE IF NOT EXISTS `sensor_readings` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `rover_id` CHAR(36) NOT NULL,
  `hardware_device_id` CHAR(36) DEFAULT NULL,
  `sensor_type` VARCHAR(100) NOT NULL,
  `sensor_value` DECIMAL(10,3) DEFAULT NULL,
  `unit` VARCHAR(20) DEFAULT NULL,
  `recorded_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_sr_rover_device` (`rover_id`, `hardware_device_id`),
  KEY `idx_sr_time` (`recorded_at`),
  CONSTRAINT `sr_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE,
  CONSTRAINT `sr_device_fk` FOREIGN KEY (`hardware_device_id`) REFERENCES `hardware_devices` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 4. Rover Locations Table (Spatial Breadcrumbs)
CREATE TABLE IF NOT EXISTS `rover_locations` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `rover_id` CHAR(36) NOT NULL,
  `pos_x` DECIMAL(10,2) DEFAULT NULL,
  `pos_y` DECIMAL(10,2) DEFAULT NULL,
  `pos_z` DECIMAL(10,2) DEFAULT NULL,
  `room_name` VARCHAR(100) DEFAULT NULL,
  `floor_number` INT DEFAULT NULL,
  `navigation_status` ENUM('idle', 'moving', 'charging', 'blocked', 'error') DEFAULT 'idle',
  `recorded_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_rl_rover_time` (`rover_id`, `recorded_at`),
  CONSTRAINT `rl_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 5. Rover Commands Table (Audit Trail of Commands Sent)
CREATE TABLE IF NOT EXISTS `rover_commands` (
  `id` CHAR(36) NOT NULL,
  `rover_id` CHAR(36) NOT NULL,
  `command_type` VARCHAR(100) NOT NULL,
  `command_payload` LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`command_payload`)),
  `command_status` ENUM('pending', 'sent', 'executing', 'completed', 'failed') DEFAULT 'pending',
  `created_by` CHAR(36) DEFAULT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `executed_at` TIMESTAMP NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_rc_rover` (`rover_id`),
  KEY `idx_rc_status` (`command_status`),
  CONSTRAINT `rc_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE,
  CONSTRAINT `rc_user_fk` FOREIGN KEY (`created_by`) REFERENCES `users` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 6. Rover Errors Table (Diagnostic Fault Registry)
CREATE TABLE IF NOT EXISTS `rover_errors` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `rover_id` CHAR(36) NOT NULL,
  `error_code` VARCHAR(100) DEFAULT NULL,
  `error_message` TEXT DEFAULT NULL,
  `severity` ENUM('low', 'medium', 'high', 'critical') DEFAULT 'low',
  `resolved` TINYINT(1) NOT NULL DEFAULT 0,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_re_rover_time` (`rover_id`, `created_at`),
  CONSTRAINT `re_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- 7. Camera Events Table (AI/Visual incident triggers)
CREATE TABLE IF NOT EXISTS `camera_events` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `rover_id` CHAR(36) NOT NULL,
  `event_type` VARCHAR(100) DEFAULT NULL,
  `confidence_score` DECIMAL(5,2) DEFAULT NULL,
  `image_path` TEXT DEFAULT NULL,
  `video_path` TEXT DEFAULT NULL,
  `detected_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_ce_rover_time` (`rover_id`, `detected_at`),
  CONSTRAINT `ce_rover_fk` FOREIGN KEY (`rover_id`) REFERENCES `rovers` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
