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
            leading: CircleAvatar(
              backgroundColor: theme.colorScheme.primary.withOpacity(0.1),
              backgroundImage: settings.profileImagePath != null
                  ? FileImage(File(settings.profileImagePath!))
                  : null,
              child: settings.profileImagePath == null
                  ? Icon(Icons.person, color: theme.colorScheme.primary)
                  : null,
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
            title: const Text('Robot IP Address'),
            subtitle: Text(settings.robotIp),
            trailing: const Icon(Icons.edit_rounded),
            onTap: () => _showIpInputDialog(context, settings),
          ),
          const ListTile(
            title: Text('Robot ID'),
            subtitle: Text('SERBOT-NC-001'),
            trailing: Icon(Icons.qr_code_scanner),
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

  void _showIpInputDialog(BuildContext context, SettingsProvider settings) {
    final controller = TextEditingController(text: settings.robotIp);
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Robot IP Address'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Enter the IP address of the rover (Jetson Nano) running the orchestrator runtime.'),
            const SizedBox(height: 16),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                labelText: 'Host IP / Name',
                hintText: 'e.g. 192.168.1.100',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.text,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              if (controller.text.trim().isNotEmpty) {
                settings.updateProfile(robotIp: controller.text.trim());
              }
              Navigator.pop(context);
            },
            child: const Text('Save'),
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
