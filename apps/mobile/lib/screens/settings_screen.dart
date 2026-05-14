import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/settings_provider.dart';
import '../providers/translation_provider.dart';
import '../services/voice_service.dart';
import 'profile_screen.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final settings = context.watch<SettingsProvider>();
    final translation = context.watch<TranslationProvider>();
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(translation.translate('settings')),
      ),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          _buildSectionHeader(context, 'Account'),
          ListTile(
            leading: Hero(
              tag: 'profile_pic',
              child: CircleAvatar(
                backgroundColor: theme.colorScheme.primary.withOpacity(0.1),
                backgroundImage: settings.profileImagePath != null
                    ? FileImage(File(settings.profileImagePath!))
                    : null,
                child: settings.profileImagePath == null
                    ? Icon(Icons.person, color: theme.colorScheme.primary)
                    : null,
              ),
            ),
            title: Text(settings.userName, style: const TextStyle(fontWeight: FontWeight.bold)),
            subtitle: Text(settings.userId),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const ProfileScreen()),
            ),
          ),

          _buildSectionHeader(context, 'Accessibility'),
          SwitchListTile(
            title: const Text('High Contrast Mode'),
            subtitle: const Text('Improve visibility for easier reading'),
            value: settings.isHighContrast,
            onChanged: (val) {
              settings.toggleHighContrast();
              VoiceService().speak(val ? "High contrast enabled" : "High contrast disabled");
            },
          ),

          _buildSectionHeader(context, 'Preferences'),
          ListTile(
            title: Text(translation.translate('language')),
            subtitle: Text(settings.getLanguageName(translation.locale)),
            trailing: const Icon(Icons.language),
            onTap: () => _showLanguagePicker(context, settings, translation),
          ),
          ListTile(
            title: Text(translation.translate('voice_feedback')),
            subtitle: const Text('Enable robotic voice responses'),
            trailing: Switch(
              value: settings.voiceFeedbackEnabled,
              onChanged: (val) {
                settings.updateProfile(voiceFeedback: val);
                VoiceService().setEnabled(val);
                VoiceService().speak(val ? "Voice feedback enabled" : "Voice feedback disabled");
              },
            ),
          ),

          _buildSectionHeader(context, 'Robot Configuration'),
          ListTile(
            title: const Text('Robot ID'),
            subtitle: const Text('SERBOT-NC-001'),
            trailing: const Icon(Icons.qr_code_scanner),
          ),

          const SizedBox(height: 40),
          Center(
            child: Text(
              'NovaCare Assistant v1.0.0',
              style: theme.textTheme.bodySmall,
            ),
          ),
        ],
      ),
    );
  }

  void _showLanguagePicker(BuildContext context, SettingsProvider settings, TranslationProvider translation) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(translation.translate('language')),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: settings.supportedLocales.map((locale) {
            return RadioListTile<Locale>(
              title: Text(settings.getLanguageName(locale)),
              value: locale,
              groupValue: translation.locale,
              onChanged: (val) {
                if (val != null) {
                  translation.setLocale(val);
                  VoiceService().speak("Language changed to ${settings.getLanguageName(val)}");
                }
                Navigator.pop(context);
              },
            );
          }).toList(),
        ),
      ),
    );
  }

  Widget _buildSectionHeader(BuildContext context, String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: Text(
        title.toUpperCase(),
        style: Theme.of(context).textTheme.labelLarge?.copyWith(
          color: Colors.grey,
          letterSpacing: 1.2,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}
