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

  Map<String, dynamic> toJson() => {
    'id': id,
    'title': title,
    'reason': reason,
    'time_hour': time.hour,
    'time_minute': time.minute,
    'sound': sound,
    'isActive': isActive,
  };
}

class ReminderProvider extends ChangeNotifier {
  final List<ReminderModel> _reminders = [];
  final NotificationService _notificationService = NotificationService();

  List<ReminderModel> get reminders => List.unmodifiable(_reminders);

  Future<void> addReminder(ReminderModel reminder) async {
    _reminders.add(reminder);
    if (reminder.isActive) {
      await _schedule(reminder);
    }
    notifyListeners();
    VoiceService().speak("Reminder added: ${reminder.title}");
  }

  Future<void> toggleReminder(String id) async {
    final index = _reminders.indexWhere((r) => r.id == id);
    if (index != -1) {
      _reminders[index].isActive = !_reminders[index].isActive;
      if (_reminders[index].isActive) {
        await _schedule(_reminders[index]);
        VoiceService().speak("Reminder ${reminders[index].title} turned on");
      } else {
        await _notificationService.cancelNotification(_reminders[index].notificationId);
        VoiceService().speak("Reminder ${reminders[index].title} turned off");
      }
      notifyListeners();
    }
  }

  Future<void> deleteReminder(String id) async {
    final index = _reminders.indexWhere((r) => r.id == id);
    if (index != -1) {
      String title = _reminders[index].title;
      await _notificationService.cancelNotification(_reminders[index].notificationId);
      _reminders.removeAt(index);
      notifyListeners();
      VoiceService().speak("Reminder $title deleted");
    }
  }

  Future<void> _schedule(ReminderModel reminder) async {
    final now = DateTime.now();
    DateTime scheduledDate = DateTime(
      now.year,
      now.month,
      now.day,
      reminder.time.hour,
      reminder.time.minute,
    );

    if (scheduledDate.isBefore(now)) {
      scheduledDate = scheduledDate.add(const Duration(days: 1));
    }

    await _notificationService.scheduleNotification(
      id: reminder.notificationId,
      title: reminder.title,
      body: reminder.reason.isNotEmpty ? reminder.reason : 'Time for your reminder!',
      scheduledDate: scheduledDate,
      sound: reminder.sound,
    );
  }
}
