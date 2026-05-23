import 'dart:async';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../providers/alert_provider.dart';
import '../providers/auth_provider.dart';
import '../providers/rover_provider.dart';
import 'notification_service.dart';

/// Handles the full SOS flow:
/// 1. Fires a full-screen alarm notification with sound (local, works on-device).
/// 2. Adds an SOS alert to AlertProvider so the caregiver sees it in their dashboard.
/// 3. Opens the SMS app pre-filled with an emergency message to the emergency contact.
/// 4. Opens the email client as an additional fallback.
///
/// Firebase FCM push to a caregiver's separate device requires a Firebase project;
/// the infrastructure (firebase_messaging in pubspec) is already in place — add
/// your google-services.json / GoogleService-Info.plist to enable it.
class EmergencyService {
  static const _teamEmail = 'novacare.emergency@gmail.com';
  static const _teamPhone = '+201234567890'; // replace with your team's number

  Future<bool> triggerSOS({
    required BuildContext context,
    required AuthProvider auth,
    required AlertProvider alertProvider,
    RoverProvider? rover,
  }) async {
    final patientName = auth.name.isNotEmpty ? auth.name : 'Patient';

    // 1 — Alarm notification (full-screen, plays sound, works while app is backgrounded).
    await NotificationService().showSosAlarm(patientName: patientName);

    // 2 — Add SOS alert to in-app feed (caregiver sees it if on same device / same session).
    alertProvider.addSosAlert(patientName);

    // 3 — Try robot siren via RoverProvider if connected.
    unawaited(rover?.sendEmergency() ?? Future.value());

    // 4 — Open SMS app with pre-filled emergency message.
    final ecPhone = auth.ecPhone.isNotEmpty ? auth.ecPhone : _teamPhone;
    final smsBody = Uri.encodeComponent(
      'NOVACARE SOS — $patientName has pressed the emergency button and needs immediate help. '
      'Time: ${DateTime.now().toString().substring(0, 16)}',
    );
    final smsUri = Uri.parse('sms:$ecPhone?body=$smsBody');
    if (await canLaunchUrl(smsUri)) {
      await launchUrl(smsUri);
    }

    return true;
  }

  /// Extra fallback: open email client to team address.
  Future<void> sendEmergencyEmail({
    required String patientName,
    String? location,
  }) async {
    final subject = Uri.encodeComponent('NOVACARE SOS — $patientName');
    final body = Uri.encodeComponent(
      'Emergency alert from NovaCare app.\n\n'
      'Patient: $patientName\n'
      'Location: ${location ?? 'Unknown'}\n'
      'Time: ${DateTime.now().toString().substring(0, 16)}\n\n'
      'Please respond immediately.',
    );
    final emailUri = Uri.parse('mailto:$_teamEmail?subject=$subject&body=$body');
    if (await canLaunchUrl(emailUri)) {
      await launchUrl(emailUri);
    }
  }

  /// Legacy method kept for backward compatibility with SosScreen.
  Future<bool> triggerSOSLegacy() async {
    print('CRITICAL: SOS Triggered (legacy)!');
    await Future.delayed(const Duration(seconds: 2));
    return true;
  }

  Future<void> notifyCaregiver(String message) async {
    print('DEBUG: Notifying caregiver: $message');
    await Future.delayed(const Duration(seconds: 1));
  }
}
