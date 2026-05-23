import 'package:flutter/painting.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/timezone.dart' as tz;
import 'package:timezone/data/latest.dart' as tz;

class NotificationService {
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;
  NotificationService._internal();

  final FlutterLocalNotificationsPlugin _plugin = FlutterLocalNotificationsPlugin();

  // Notification channel IDs
  static const _channelReminders = 'reminders_channel';
  static const _channelSos       = 'sos_alarm_channel';

  Future<void> init() async {
    tz.initializeTimeZones();

    const androidSettings = AndroidInitializationSettings('@mipmap/launcher_icon');
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    await _plugin.initialize(
      const InitializationSettings(android: androidSettings, iOS: iosSettings),
    );

    // Create the SOS alarm channel with max importance up-front (Android).
    final androidPlugin = _plugin
        .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>();
    await androidPlugin?.createNotificationChannel(
      const AndroidNotificationChannel(
        _channelSos,
        'SOS Alarm',
        description: 'Full-screen alarm when a patient triggers SOS',
        importance: Importance.max,
        playSound: true,
        enableVibration: true,
        enableLights: true,
        ledColor: const Color(0xFFD8473D),
      ),
    );
  }

  /// Show a full-screen SOS alarm notification immediately.
  /// [patientName] is shown in the body so the caregiver knows who triggered it.
  Future<void> showSosAlarm({required String patientName}) async {
    await _plugin.show(
      999, // fixed id — a new SOS overwrites the previous one
      '🆘 SOS Emergency — $patientName',
      'Your patient needs immediate help. Tap to open the app.',
      NotificationDetails(
        android: AndroidNotificationDetails(
          _channelSos,
          'SOS Alarm',
          channelDescription: 'Full-screen alarm when a patient triggers SOS',
          importance: Importance.max,
          priority: Priority.max,
          fullScreenIntent: true,
          category: AndroidNotificationCategory.alarm,
          playSound: true,
          enableVibration: true,
          color: const Color(0xFFD8473D),
          icon: '@mipmap/launcher_icon',
          ongoing: false,
        ),
        iOS: const DarwinNotificationDetails(
          presentAlert: true,
          presentBadge: true,
          presentSound: true,
          sound: 'default',
          interruptionLevel: InterruptionLevel.critical,
        ),
      ),
    );
  }

  Future<void> scheduleNotification({
    required int id,
    required String title,
    required String body,
    required DateTime scheduledDate,
    String? sound,
  }) async {
    await _plugin.zonedSchedule(
      id,
      title,
      body,
      tz.TZDateTime.from(scheduledDate, tz.local),
      NotificationDetails(
        android: AndroidNotificationDetails(
          _channelReminders,
          'Reminders',
          channelDescription: 'Medication and daily reminder notifications',
          importance: Importance.max,
          priority: Priority.high,
          sound: sound != null ? RawResourceAndroidNotificationSound(sound.split('.').first) : null,
          playSound: true,
        ),
        iOS: DarwinNotificationDetails(sound: sound, presentSound: true),
      ),
      androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      matchDateTimeComponents: DateTimeComponents.time,
    );
  }

  Future<void> cancelNotification(int id) async => _plugin.cancel(id);
  Future<void> cancelAll() async => _plugin.cancelAll();
}
