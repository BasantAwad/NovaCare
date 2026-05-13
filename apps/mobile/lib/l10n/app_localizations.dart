import 'package:flutter/material.dart';

/// Simple localization system supporting English and Arabic.
class AppLocalizations {
  final Locale locale;
  AppLocalizations(this.locale);

  static AppLocalizations of(BuildContext context) {
    return Localizations.of<AppLocalizations>(context, AppLocalizations) ??
        AppLocalizations(const Locale('en', ''));
  }

  static const List<LocalizationsDelegate> localizationsDelegates = [
    _AppLocalizationsDelegate(),
    // Add Flutter's built-in delegates
  ];

  bool get isArabic => locale.languageCode == 'ar';

  static final Map<String, Map<String, String>> _localizedValues = {
    'en': {
      'app_title': 'NovaCare',
      'app_subtitle': 'Your Assistive Companion',
      'dashboard': 'Dashboard',
      'settings': 'Settings',
      'sos_emergency': 'SOS Emergency',
      'sos_desc': 'Trigger alarm & notify guardian',
      'medication': 'Medication',
      'medication_desc': 'Request medicine delivery',
      'home_dock': 'Home / Dock',
      'home_dock_desc': 'Return rover to charging station',
      'follow_me': 'Follow Me',
      'follow_me_desc': 'Rover tracks your location',
      'battery': 'Battery',
      'heart_rate': 'Heart Rate',
      'location': 'Location',
      'temperature': 'Temperature',
      'rover_status': 'Rover Status',
      'online': 'Online',
      'offline': 'Offline',
      'connected': 'Connected',
      'disconnected': 'Disconnected',
      'connecting': 'Connecting...',
      'bpm': 'BPM',
      'user_account': 'User Account',
      'profile_management': 'Profile Management',
      'user_name': 'User Name',
      'user_id': 'Student / User ID',
      'disability_type': 'Primary Disability Type',
      'voice_feedback': 'Voice Feedback',
      'language': 'Language',
      'english': 'English',
      'arabic': 'العربية',
      'app_theme': 'App Theme',
      'light_mode': 'Light Mode',
      'dark_mode': 'Dark Mode',
      'high_contrast': 'High Contrast',
      'privacy_security': 'Privacy & Security',
      'data_encryption': 'Data Encryption',
      'encryption_desc': 'All health and location data is encrypted using AES-256 encryption both in transit and at rest.',
      'permissions': 'Permissions',
      'camera_access': 'Camera Access',
      'location_tracking': 'Location Tracking',
      'microphone_use': 'Microphone Use',
      'connectivity': 'Connectivity',
      'bluetooth_ble': 'Bluetooth (BLE)',
      'wifi_cloud': 'Wi-Fi / Cloud',
      'scan_devices': 'Scan for Devices',
      'signal_strength': 'Signal Strength',
      'idle': 'Idle',
      'following': 'Following User',
      'navigating_home': 'Navigating Home',
      'delivering': 'Delivering Medicine',
      'emergency_mode': 'Emergency Mode',
      'cancel': 'Cancel',
      'confirm': 'Confirm',
      'save': 'Save',
      'are_you_sure': 'Are you sure?',
      'sos_confirm': 'This will trigger the rover alarm and notify your guardian immediately.',
      'speed': 'Speed',
      'none_selected': 'None',
      'visual_impairment': 'Visual Impairment',
      'motor_disability': 'Motor Disability',
      'hearing_impairment': 'Hearing Impairment',
      'cognitive_disability': 'Cognitive Disability',
      'about': 'About NovaCare',
      'version': 'Version 1.0.0',
    },
    'ar': {
      'app_title': 'نوفا كير',
      'app_subtitle': 'مساعدك الذكي',
      'dashboard': 'لوحة التحكم',
      'settings': 'الإعدادات',
      'sos_emergency': 'طوارئ SOS',
      'sos_desc': 'تشغيل الإنذار وإبلاغ الوصي',
      'medication': 'الدواء',
      'medication_desc': 'طلب توصيل الدواء',
      'home_dock': 'العودة / الشحن',
      'home_dock_desc': 'إرجاع الروبوت لمحطة الشحن',
      'follow_me': 'تابعني',
      'follow_me_desc': 'الروبوت يتتبع موقعك',
      'battery': 'البطارية',
      'heart_rate': 'نبض القلب',
      'location': 'الموقع',
      'temperature': 'درجة الحرارة',
      'rover_status': 'حالة الروبوت',
      'online': 'متصل',
      'offline': 'غير متصل',
      'connected': 'متصل',
      'disconnected': 'غير متصل',
      'connecting': 'جاري الاتصال...',
      'bpm': 'نبضة/د',
      'user_account': 'حساب المستخدم',
      'profile_management': 'إدارة الملف الشخصي',
      'user_name': 'اسم المستخدم',
      'user_id': 'رقم الطالب / المستخدم',
      'disability_type': 'نوع الإعاقة الأساسية',
      'voice_feedback': 'التغذية الصوتية',
      'language': 'اللغة',
      'english': 'English',
      'arabic': 'العربية',
      'app_theme': 'مظهر التطبيق',
      'light_mode': 'الوضع الفاتح',
      'dark_mode': 'الوضع الداكن',
      'high_contrast': 'تباين عالي',
      'privacy_security': 'الخصوصية والأمان',
      'data_encryption': 'تشفير البيانات',
      'encryption_desc': 'جميع البيانات الصحية وبيانات الموقع مشفرة باستخدام تشفير AES-256 أثناء النقل وفي حالة السكون.',
      'permissions': 'الأذونات',
      'camera_access': 'الوصول للكاميرا',
      'location_tracking': 'تتبع الموقع',
      'microphone_use': 'استخدام الميكروفون',
      'connectivity': 'الاتصال',
      'bluetooth_ble': 'بلوتوث (BLE)',
      'wifi_cloud': 'واي فاي / سحابي',
      'scan_devices': 'البحث عن أجهزة',
      'signal_strength': 'قوة الإشارة',
      'idle': 'خامل',
      'following': 'يتابع المستخدم',
      'navigating_home': 'يعود للمحطة',
      'delivering': 'يوصل الدواء',
      'emergency_mode': 'وضع الطوارئ',
      'cancel': 'إلغاء',
      'confirm': 'تأكيد',
      'save': 'حفظ',
      'are_you_sure': 'هل أنت متأكد؟',
      'sos_confirm': 'سيتم تشغيل إنذار الروبوت وإبلاغ الوصي فوراً.',
      'speed': 'السرعة',
      'none_selected': 'لا يوجد',
      'visual_impairment': 'إعاقة بصرية',
      'motor_disability': 'إعاقة حركية',
      'hearing_impairment': 'إعاقة سمعية',
      'cognitive_disability': 'إعاقة ذهنية',
      'about': 'عن نوفا كير',
      'version': 'الإصدار ١.٠.٠',
    },
  };

  String translate(String key) {
    return _localizedValues[locale.languageCode]?[key] ??
        _localizedValues['en']?[key] ??
        key;
  }
}

class _AppLocalizationsDelegate extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  bool isSupported(Locale locale) => ['en', 'ar'].contains(locale.languageCode);

  @override
  Future<AppLocalizations> load(Locale locale) async {
    return AppLocalizations(locale);
  }

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}
