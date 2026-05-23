import 'package:flutter/material.dart';

class TranslationProvider extends ChangeNotifier {
  Locale _locale = const Locale('en');
  Locale get locale => _locale;

  final Map<String, Map<String, String>> _strings = {
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
      'home': 'الرئيسية',
      'reminders': 'التذكيرات',
      'alerts': 'التنبيهات',
      'settings': 'الإعدادات',
      'good_morning': 'صباح الخير،',
      'vital_stats': 'الإحصائيات الحيوية',
      'emergency': 'الطوارئ والمساعدة',
      'summon': 'استدعاء الروبوت',
      'controls': 'التحكم بالروبوت',
      'heart_rate': 'نبض القلب',
      'battery': 'بطارية الروبوت',
      'location': 'موقع الروبوت',
      'temperature': 'درجة الحرارة',
      'profile': 'ملف المستخدم',
      'language': 'اللغة',
      'voice_feedback': 'التعليق الصوتي',
      'disability': 'نوع الإعاقة',
    },
    'es': {
      'home': 'Inicio',      'reminders': 'Recordatorios', 'alerts': 'Alertas',
      'settings': 'Ajustes', 'good_morning': 'Buenos días,', 'vital_stats': 'Estadísticas vitales',
      'emergency': 'Emergencia y Asistencia', 'summon': 'Llamar Robot',
      'controls': 'Controles', 'heart_rate': 'Ritmo cardíaco', 'battery': 'Batería del robot',
      'location': 'Ubicación', 'temperature': 'Temperatura', 'profile': 'Perfil',
      'language': 'Idioma', 'voice_feedback': 'Voz', 'disability': 'Discapacidad',
    },
    'fr': {
      'home': 'Accueil',     'reminders': 'Rappels',    'alerts': 'Alertes',
      'settings': 'Paramètres', 'good_morning': 'Bonjour,', 'vital_stats': 'Statistiques vitales',
      'emergency': 'Urgence et Assistance', 'summon': 'Appeler le robot',
      'controls': 'Commandes', 'heart_rate': 'Rythme cardiaque', 'battery': 'Batterie',
      'location': 'Localisation', 'temperature': 'Température', 'profile': 'Profil',
      'language': 'Langue', 'voice_feedback': 'Retour vocal', 'disability': 'Handicap',
    },
  };

  void setLocale(Locale locale) {
    _locale = locale;
    notifyListeners();
  }

  String translate(String key) =>
      _strings[_locale.languageCode]?[key] ?? key;
}
