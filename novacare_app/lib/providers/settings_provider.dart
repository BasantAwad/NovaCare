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
  Locale get locale => _locale;

  bool get isArabic => _locale.languageCode == 'ar';

  void setLocale(Locale locale) {
    _locale = locale;
    notifyListeners();
  }

  void toggleLanguage() {
    _locale = _locale.languageCode == 'en'
        ? const Locale('ar', '')
        : const Locale('en', '');
    notifyListeners();
  }

  // ─── User Profile ──────────────────────────────────────────────
  String _userName = 'User';
  String _userId = '';
  String _disabilityType = 'None';
  bool _voiceFeedbackEnabled = false;
  bool _largeTextEnabled = false;
  String _profileImagePath = '';

  String get userName => _userName;
  String get userId => _userId;
  String get disabilityType => _disabilityType;
  bool get voiceFeedbackEnabled => _voiceFeedbackEnabled;
  bool get largeTextEnabled => _largeTextEnabled;
  String get profileImagePath => _profileImagePath;

  void updateProfile({
    String? name,
    String? id,
    String? disability,
    bool? voiceFeedback,
    bool? largeTextEnabled,
    String? profileImagePath,
  }) {
    if (name != null) _userName = name;
    if (id != null) _userId = id;
    if (disability != null) _disabilityType = disability;
    if (voiceFeedback != null) _voiceFeedbackEnabled = voiceFeedback;
    if (largeTextEnabled != null) _largeTextEnabled = largeTextEnabled;
    if (profileImagePath != null) _profileImagePath = profileImagePath;
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
