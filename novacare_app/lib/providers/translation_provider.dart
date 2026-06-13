import 'package:flutter/material.dart';

class TranslationProvider extends ChangeNotifier {
  Locale _locale = const Locale('en');
  Locale get locale => _locale;

  final Map<String, Map<String, String>> _localizedValues = {
    'en': {
      'home': 'Home',
      'reminders': 'Reminders',
      'alerts': 'Alerts',
      'settings': 'Settings',
      'good_morning': 'Good Morning,',
      'vital_stats': 'Vital Statistics',
      'emergency': 'Emergency & Assistance',
      'summon': 'Summon Robot',
      'controls': 'Rover Controls',
      'heart_rate': 'Heart Rate',
      'battery': 'Robot Battery',
      'location': 'Robot Location',
      'temperature': 'Temperature',
      'profile': 'User Profile',
      'language': 'Language',
      'voice_feedback': 'Voice Feedback',
      'disability': 'Disability Type',
    },
    'ar': {
      'home': 'Ø§Ù„Ø±Ø¦ÙŠØ³ÙŠØ©',
      'reminders': 'Ø§Ù„ØªØ°ÙƒÙŠØ±Ø§Øª',
      'alerts': 'Ø§Ù„ØªÙ†Ø¨ÙŠÙ‡Ø§Øª',
      'settings': 'Ø§Ù„Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª',
      'good_morning': 'ØµØ¨Ø§Ø­ Ø§Ù„Ø®ÙŠØ±ØŒ',
      'vital_stats': 'Ø§Ù„Ø¥Ø­ØµØ§Ø¦ÙŠØ§Øª Ø§Ù„Ø­ÙŠÙˆÙŠØ©',
      'emergency': 'Ø§Ù„Ø·ÙˆØ§Ø±Ø¦ ÙˆØ§Ù„Ù…Ø³Ø§Ø¹Ø¯Ø©',
      'summon': 'Ø§Ø³ØªØ¯Ø¹Ø§Ø¡ Ø§Ù„Ø±ÙˆØ¨ÙˆØª',
      'controls': 'Ø§Ù„ØªØ­ÙƒÙ… Ø¨Ø§Ù„Ø±ÙˆØ¨ÙˆØª',
      'heart_rate': 'Ù†Ø¨Ø¶ Ø§Ù„Ù‚Ù„Ø¨',
      'battery': 'Ø¨Ø·Ø§Ø±ÙŠØ© Ø§Ù„Ø±ÙˆØ¨ÙˆØª',
      'location': 'Ù…ÙˆÙ‚Ø¹ Ø§Ù„Ø±ÙˆØ¨ÙˆØª',
      'temperature': 'Ø¯Ø±Ø¬Ø© Ø§Ù„Ø­Ø±Ø§Ø±Ø©',
      'profile': 'Ù…Ù„Ù Ø§Ù„Ù…Ø³ØªØ®Ø¯Ù…',
      'language': 'Ø§Ù„Ù„ØºØ©',
      'voice_feedback': 'Ø§Ù„ØªØ¹Ù„ÙŠÙ‚ Ø§Ù„ØµÙˆØªÙŠ',
      'disability': 'Ù†ÙˆØ¹ Ø§Ù„Ø¥Ø¹Ø§Ù‚Ø©',
    },
    'es': {
      'home': 'Inicio',
      'reminders': 'Recordatorios',
      'alerts': 'Alertas',
      'settings': 'Ajustes',
      'good_morning': 'Buenos dÃ­as,',
      'vital_stats': 'EstadÃ­sticas vitales',
      'emergency': 'Emergencia y Asistencia',
      'summon': 'Llamar Robot',
      'controls': 'Controles',
      'heart_rate': 'Ritmo cardÃ­aco',
      'battery': 'BaterÃ­a del robot',
      'location': 'UbicaciÃ³n',
      'temperature': 'Temperatura',
      'profile': 'Perfil',
      'language': 'Idioma',
      'voice_feedback': 'Voz',
      'disability': 'Discapacidad',
    },
    'fr': {
      'home': 'Accueil',
      'reminders': 'Rappels',
      'alerts': 'Alertes',
      'settings': 'ParamÃ¨tres',
      'good_morning': 'Bonjour,',
      'vital_stats': 'Statistiques vitales',
      'emergency': 'Urgence et Assistance',
      'summon': 'Appeler le robot',
      'controls': 'Commandes',
      'heart_rate': 'Rythme cardiaque',
      'battery': 'Batterie',
      'location': 'Localisation',
      'temperature': 'TempÃ©rature',
      'profile': 'Profil',
      'language': 'Langue',
      'voice_feedback': 'Retour vocal',
      'disability': 'Handicap',
    },
  };

  void setLocale(Locale locale) {
    _locale = locale;
    notifyListeners();
  }

  String translate(String key) {
    return _localizedValues[_locale.languageCode]?[key] ?? key;
  }
}
