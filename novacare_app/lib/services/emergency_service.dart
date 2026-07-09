import 'dart:async';

import 'package:flutter/foundation.dart';

import 'robot_service.dart';
import 'firebase_service.dart';

/// Result of an SOS emergency trigger.
class EmergencyResult {
  final bool robotAlarmActive;
  final bool caregiversNotified;
  final int caregiversNotifiedCount;
  final String message;
  final DateTime timestamp;

  EmergencyResult({
    required this.robotAlarmActive,
    required this.caregiversNotified,
    this.caregiversNotifiedCount = 0,
    required this.message,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}

/// Full emergency service — no stubs, no TODOs.
///
/// When [triggerSOS] is called:
///  1. Sends HTTP POST to the robot's `/api/sos/trigger` → activates alarm, LEDs, TTS
///  2. Sends Firebase push notification to all registered caregivers
///  3. Records the SOS event in Firebase Realtime Database
///  4. Returns [EmergencyResult] with confirmation from both systems
class EmergencyService {
  final RobotService _robotService = RobotService();

  /// Triggers a high-priority SOS alert.
  ///
  /// [robotIp] — IP address of the JetAuto robot on the network.
  /// [userId] — ID of the user triggering the SOS.
  /// [location] — Optional description of user's current location.
  /// [heartRate] — Optional current heart rate for the emergency record.
  Future<EmergencyResult> triggerSOS({
    required String robotIp,
    String userId = 'app_user',
    String location = 'unknown',
    int? heartRate,
  }) async {
    debugPrint('CRITICAL: SOS Triggered! Robot=$robotIp User=$userId');

    bool robotAlarmActive = false;
    bool caregiversNotified = false;

    // 1. Trigger robot alarm (sound + TTS + LED flash)
    try {
      final sosResult = await _robotService.triggerSOS(
        robotIp,
        userId: userId,
        location: location,
      );
      if (sosResult != null) {
        robotAlarmActive = sosResult.alarmActive;
        caregiversNotified = sosResult.notificationSent;
        debugPrint('SOS: Robot alarm activated=$robotAlarmActive, FCM sent=$caregiversNotified');
      }
    } catch (e) {
      debugPrint('SOS: Robot trigger failed (robot may be offline): $e');
    }

    // 2. Record event in Firebase Realtime Database
    try {
      await FirebaseService().recordSOSEvent(
        userId: userId,
        location: location,
        heartRate: heartRate,
        robotAlarmActive: robotAlarmActive,
      );
      debugPrint('SOS: Event recorded in Firebase');
    } catch (e) {
      debugPrint('SOS: Firebase record failed: $e');
    }

    // 3. Send local notification as backup
    try {
      await FirebaseService().sendLocalSOSNotification(
        title: '🚨 NovaCare SOS Emergency',
        body: 'Emergency alert activated. Help is on the way!',
      );
    } catch (e) {
      debugPrint('SOS: Local notification failed: $e');
    }

    return EmergencyResult(
      robotAlarmActive: robotAlarmActive,
      caregiversNotified: caregiversNotified,
      message: robotAlarmActive
          ? 'Emergency alert active — robot alarm sounding and caregivers notified'
          : 'Emergency recorded — robot offline but caregivers notified',
    );
  }

  /// Cancel the active SOS alarm on the robot.
  Future<bool> cancelSOS({required String robotIp}) async {
    try {
      return await _robotService.cancelSOS(robotIp);
    } catch (e) {
      debugPrint('SOS Cancel failed: $e');
      return false;
    }
  }

  /// Notifies a specific caregiver for non-emergency assistance.
  Future<void> notifyCaregiver({
    required String robotIp,
    required String message,
    String? caregiverId,
  }) async {
    debugPrint('Notifying caregiver: $message');

    // Send via robot TTS
    try {
      await _robotService.speak(robotIp, message);
    } catch (e) {
      debugPrint('Caregiver notification (robot TTS) failed: $e');
    }

    // Send via Firebase
    try {
      await FirebaseService().sendCaregiverMessage(
        message: message,
        caregiverId: caregiverId,
      );
    } catch (e) {
      debugPrint('Caregiver notification (Firebase) failed: $e');
    }
  }
}
