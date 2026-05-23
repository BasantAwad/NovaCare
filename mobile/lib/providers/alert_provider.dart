import 'package:flutter/material.dart';

enum AlertSeverity { info, warning, error }

class AlertModel {
  final String id;
  final String title;
  final String body;
  final DateTime timestamp;
  final IconData icon;
  final Color color;
  final AlertSeverity severity;
  bool isRead;

  AlertModel({
    required this.id,
    required this.title,
    required this.body,
    required this.timestamp,
    required this.icon,
    required this.color,
    this.severity = AlertSeverity.info,
    this.isRead = false,
  });
}

class AlertProvider extends ChangeNotifier {
  final List<AlertModel> _alerts = [
    AlertModel(
      id: '1',
      title: 'Reminder',
      body: 'Time to take your afternoon medication.',
      timestamp: DateTime.now().subtract(const Duration(minutes: 10)),
      icon: Icons.medication_rounded,
      color: Colors.purple,
    ),
    AlertModel(
      id: '2',
      title: 'Rover Low Battery',
      body: 'Rover battery is below 15%. Moving to dock.',
      timestamp: DateTime.now().subtract(const Duration(hours: 1)),
      icon: Icons.battery_alert_rounded,
      color: Colors.orange,
      severity: AlertSeverity.warning,
    ),
  ];

  List<AlertModel> get alerts    => List.unmodifiable(_alerts);
  int get unreadCount            => _alerts.where((a) => !a.isRead).length;

  void addAlert(AlertModel alert) {
    _alerts.insert(0, alert);
    notifyListeners();
  }

  void dismissAlert(String id) {
    _alerts.removeWhere((a) => a.id == id);
    notifyListeners();
  }

  void markAsRead(String id) {
    final i = _alerts.indexWhere((a) => a.id == id);
    if (i != -1 && !_alerts[i].isRead) {
      _alerts[i].isRead = true;
      notifyListeners();
    }
  }

  void markAllAsRead() {
    for (final a in _alerts) { a.isRead = true; }
    notifyListeners();
  }

  void clearAll() {
    _alerts.clear();
    notifyListeners();
  }

  /// Adds a critical SOS alert — call this when the patient presses the SOS button.
  void addSosAlert(String patientName) {
    addAlert(AlertModel(
      id:        'sos_${DateTime.now().millisecondsSinceEpoch}',
      title:     'SOS — $patientName',
      body:      'Emergency alert triggered. Patient needs immediate help.',
      timestamp: DateTime.now(),
      icon:      Icons.emergency_rounded,
      color:     Colors.red,
      severity:  AlertSeverity.error,
    ));
  }
}
