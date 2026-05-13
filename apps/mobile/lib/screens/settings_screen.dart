import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/settings_provider.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final settings = context.watch<SettingsProvider>();
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          _buildSectionHeader(context, 'Accessibility'),
          SwitchListTile(
            title: const Text('High Contrast Mode'),
            subtitle: const Text('Improve visibility for easier reading'),
            value: settings.isHighContrast,
            onChanged: (val) => settings.toggleHighContrast(),
          ),
          _buildSectionHeader(context, 'Preferences'),
          ListTile(
            title: const Text('App Language'),
            subtitle: Text(settings.locale.languageCode == 'en' ? 'English' : 'Arabic'),
            trailing: const Icon(Icons.language),
            onTap: () {
              // TODO: Show language picker
            },
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

  Widget _buildSectionHeader(BuildContext context, String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: Text(
        title.toUpperCase(),
        style: Theme.of(context).textTheme.labelLarge?.copyWith(
          color: Colors.grey,
          letterSpacing: 1.2,
        ),
      ),
    );
  }
}
