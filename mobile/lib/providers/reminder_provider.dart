import 'package:flutter/material.dart';
import '../services/notification_service.dart';
import '../services/voice_service.dart';

class ReminderModel {
  final String id;
  final String title;
  final String reason;
  final TimeOfDay time;
  final String? sound;
  bool isActive;

  ReminderModel({
    required this.id,
    required this.title,
    this.reason = '',
    required this.time,
    this.sound,
    this.isActive = true,
  });

  int get notificationId => id.hashCode;
}

class ReminderProvider extends ChangeNotifier {
  final List<ReminderModel> _reminders = [];
  final NotificationService _notificationService = NotificationService();

  List<ReminderModel> get reminders => List.unmodifiable(_reminders);

  Future<void> addReminder(ReminderModel reminder) async {
    _reminders.add(reminder);
    if (reminder.isActive) await _schedule(reminder);
    notifyListeners();
    VoiceService().speak('Reminder added: ${reminder.title}');
  }

  Future<void> toggleReminder(String id) async {
    final i = _reminders.indexWhere((r) => r.id == id);
    if (i == -1) return;
    _reminders[i].isActive = !_reminders[i].isActive;
    if (_reminders[i].isActive) {
      await _schedule(_reminders[i]);
      VoiceService().speak('Reminder ${_reminders[i].title} turned on');
    } else {
      await _notificationService.cancelNotification(_reminders[i].notificationId);
      VoiceService().speak('Reminder ${_reminders[i].title} turned off');
    }
    notifyListeners();
  }

  Future<void> deleteReminder(String id) async {
    final i = _reminders.indexWhere((r) => r.id == id);
    if (i == -1) return;
    final title = _reminders[i].title;
    await _notificationService.cancelNotification(_reminders[i].notificationId);
    _reminders.removeAt(i);
    notifyListeners();
    VoiceService().speak('Reminder $title deleted');
  }

  Future<void> _schedule(ReminderModel reminder) async {
    final now = DateTime.now();
    var scheduled = DateTime(now.year, now.month, now.day, reminder.time.hour, reminder.time.minute);
    if (scheduled.isBefore(now)) scheduled = scheduled.add(const Duration(days: 1));

    await _notificationService.scheduleNotification(
      id: reminder.notificationId,
      title: reminder.title,
      body: reminder.reason.isNotEmpty ? reminder.reason : 'Time for your reminder!',
      scheduledDate: scheduled,
      sound: reminder.sound,
    );
  }
}
