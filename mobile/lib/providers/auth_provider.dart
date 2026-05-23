import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum UserRole { patient, caregiver }

class AuthProvider extends ChangeNotifier {
  static const _keyLoggedIn = 'auth_logged_in';
  static const _keyRole     = 'auth_role';
  static const _keyName     = 'auth_name';
  static const _keyEmail    = 'auth_email';
  static const _keyEcPhone  = 'auth_ec_phone';
  static const _keyEcName   = 'auth_ec_name';
  static const _keyPassword = 'auth_password';

  bool     _isLoggedIn = false;
  UserRole _role       = UserRole.patient;
  String   _name       = '';
  String   _email      = '';
  String   _password   = '';
  String   _ecPhone    = '';
  String   _ecName     = '';

  bool     get isLoggedIn  => _isLoggedIn;
  UserRole get role        => _role;
  String   get name        => _name;
  String   get email       => _email;
  String   get ecPhone     => _ecPhone;
  String   get ecName      => _ecName;
  bool     get isCaregiver => _role == UserRole.caregiver;
  bool     get isPatient   => _role == UserRole.patient;

  // Call once at startup to rehydrate from disk.
  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    _isLoggedIn = prefs.getBool(_keyLoggedIn) ?? false;
    _role    = (prefs.getString(_keyRole) ?? 'patient') == 'caregiver'
        ? UserRole.caregiver
        : UserRole.patient;
    _name     = prefs.getString(_keyName)     ?? '';
    _email    = prefs.getString(_keyEmail)    ?? '';
    _password = prefs.getString(_keyPassword) ?? '';
    _ecPhone  = prefs.getString(_keyEcPhone)  ?? '';
    _ecName   = prefs.getString(_keyEcName)   ?? '';
    notifyListeners();
  }

  /// Register a new account (local — no backend yet).
  Future<void> signup({
    required String name,
    required String email,
    required String password,
    required UserRole role,
    String ecPhone = '',
    String ecName  = '',
  }) async {
    if (name.trim().isEmpty)     throw 'Name is required';
    if (email.trim().isEmpty)    throw 'Email is required';
    if (password.length < 6)     throw 'Password must be at least 6 characters';
    _name       = name.trim();
    _email      = email.trim();
    _password   = password;
    _role       = role;
    _ecPhone    = ecPhone.trim();
    _ecName     = ecName.trim();
    _isLoggedIn = true;
    await _persist();
    notifyListeners();
  }

  /// Sign in with email + password.
  Future<void> login({
    required String email,
    required String password,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final savedEmail    = prefs.getString(_keyEmail)    ?? '';
    final savedPassword = prefs.getString(_keyPassword) ?? '';

    if (email.trim().isEmpty)    throw 'Email is required';
    if (password.isEmpty)        throw 'Password is required';
    if (savedEmail.isEmpty)      throw 'No account found. Please sign up first.';
    if (email.trim() != savedEmail) throw 'Email not found. Did you mean $savedEmail?';
    if (password != savedPassword)  throw 'Incorrect password.';

    _isLoggedIn = true;
    notifyListeners();
  }

  Future<void> logout() async {
    _isLoggedIn = false;
    await _persist();
    notifyListeners();
  }

  Future<void> _persist() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyLoggedIn,    _isLoggedIn);
    await prefs.setString(_keyRole,      _role == UserRole.caregiver ? 'caregiver' : 'patient');
    await prefs.setString(_keyName,      _name);
    await prefs.setString(_keyEmail,     _email);
    await prefs.setString(_keyPassword,  _password);
    await prefs.setString(_keyEcPhone,   _ecPhone);
    await prefs.setString(_keyEcName,    _ecName);
  }
}
