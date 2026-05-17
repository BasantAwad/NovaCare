import 'package:flutter/material.dart';

/// Manages app-wide settings: theme, language, and user profile.
class SettingsProvider extends ChangeNotifier {
  // ─── Theme ──────────────────────────────────────────────────────
  ThemeMode _themeMode = ThemeMode.system;
  bool _isHighContrast = false;

  ThemeMode get themeMode => _isHighContrast ? ThemeMode.dark : _themeMode;
  bool get isHighContrast => _isHighContrast;

  void setThemeMode(ThemeMode mode) {
    _themeMode = mode;
    _isHighContrast = false;
    notifyListeners();
  }

  void enableHighContrast() {
    _isHighContrast = true;
    _themeMode = ThemeMode.dark;
    notifyListeners();
  }

  void toggleHighContrast() {
    _isHighContrast = !_isHighContrast;
    _themeMode = _isHighContrast ? ThemeMode.dark : ThemeMode.light;
    notifyListeners();
  }

  String get themeLabel {
    if (_isHighContrast) return 'High Contrast';
    switch (_themeMode) {
      case ThemeMode.light:
        return 'Light';
      case ThemeMode.dark:
        return 'Dark';
      case ThemeMode.system:
        return 'System';
    }
  }

  // ─── Language / Locale ──────────────────────────────────────────
  Locale _locale = const Locale('en', '');
  final List<Locale> _supportedLocales = [
    const Locale('en', ''),
    const Locale('ar', ''),
    const Locale('es', ''),
    const Locale('fr', ''),
  ];

  Locale get locale => _locale;
  List<Locale> get supportedLocales => _supportedLocales;

  bool get isArabic => _locale.languageCode == 'ar';

  void setLocale(Locale locale) {
    _locale = locale;
    notifyListeners();
  }

  String getLanguageName(Locale l) {
    switch (l.languageCode) {
      case 'en': return 'English';
      case 'ar': return 'Arabic';
      case 'es': return 'Spanish';
      case 'fr': return 'French';
      default: return l.languageCode;
    }
  }

  void toggleLanguage() {
    _locale = _locale.languageCode == 'en'
        ? const Locale('ar', '')
        : const Locale('en', '');
    notifyListeners();
  }

  // ─── User Profile ──────────────────────────────────────────────
  String _userName = 'NovaCare User';
  String _userId = 'NC-USER-001';
  String _email = 'user@novacare.com';
  String _emergencyContact = '+1 234 567 890';
  String _disabilityType = 'None';
  String? _profileImagePath;
  bool _voiceFeedbackEnabled = false;
  String _robotIp = '10.34.19.247';

  String get userName => _userName;
  String get userId => _userId;
  String get email => _email;
  String get emergencyContact => _emergencyContact;
  String get disabilityType => _disabilityType;
  String? get profileImagePath => _profileImagePath;
  bool get voiceFeedbackEnabled => _voiceFeedbackEnabled;
  String get robotIp => _robotIp;

  void updateProfile({
    String? name,
    String? id,
    String? email,
    String? emergencyContact,
    String? disability,
    String? profileImagePath,
    bool? voiceFeedback,
    String? robotIp,
  }) {
    if (name != null) _userName = name;
    if (id != null) _userId = id;
    if (email != null) _email = email;
    if (emergencyContact != null) _emergencyContact = emergencyContact;
    if (disability != null) _disabilityType = disability;
    if (profileImagePath != null) _profileImagePath = profileImagePath;
    if (voiceFeedback != null) _voiceFeedbackEnabled = voiceFeedback;
    if (robotIp != null) _robotIp = robotIp;
    notifyListeners();
  }

  // ─── Permissions ────────────────────────────────────────────────
  bool _cameraPermission = false;
  bool _locationPermission = false;
  bool _microphonePermission = false;

  bool get cameraPermission => _cameraPermission;
  bool get locationPermission => _locationPermission;
  bool get microphonePermission => _microphonePermission;

  void setCameraPermission(bool value) {
    _cameraPermission = value;
    notifyListeners();
  }

  void setLocationPermission(bool value) {
    _locationPermission = value;
    notifyListeners();
  }

  void setMicrophonePermission(bool value) {
    _microphonePermission = value;
    notifyListeners();
  }
}
