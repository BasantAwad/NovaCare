import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/settings_provider.dart';
import '../providers/ble_provider.dart';
import '../theme/app_colors.dart';
import '../l10n/app_localizations.dart';

/// Settings screen with User Account, Language, Theme, Privacy, and Connectivity sections.
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final settings = context.watch<SettingsProvider>();
    final ble = context.watch<BleProvider>();

    return Scaffold(
      appBar: AppBar(
        title: Text(l10n.translate('settings')),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_rounded),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: ListView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.symmetric(vertical: 8),
        children: [
          // ═══════════════════════════════════════════════════════
          //  USER ACCOUNT
          // ═══════════════════════════════════════════════════════
          _SectionHeader(
            icon: Icons.person_rounded,
            title: l10n.translate('user_account'),
          ),
          _SettingsCard(
            children: [
              _ProfileTile(settings: settings, l10n: l10n),
              const Divider(height: 1),
              _DisabilityTile(settings: settings, l10n: l10n),
              const Divider(height: 1),
              _SwitchTile(
                icon: Icons.record_voice_over_rounded,
                title: l10n.translate('voice_feedback'),
                subtitle: 'Enhanced audio feedback for visually impaired users',
                value: settings.voiceFeedbackEnabled,
                onChanged: (v) =>
                    settings.updateProfile(voiceFeedback: v),
              ),
            ],
          ),

          // ═══════════════════════════════════════════════════════
          //  LANGUAGE
          // ═══════════════════════════════════════════════════════
          _SectionHeader(
            icon: Icons.translate_rounded,
            title: l10n.translate('language'),
          ),
          _SettingsCard(
            children: [
              _RadioTile<String>(
                icon: Icons.language_rounded,
                title: l10n.translate('english'),
                subtitle: 'English',
                value: 'en',
                groupValue: settings.locale.languageCode,
                onChanged: (_) => settings.setLocale(const Locale('en', '')),
              ),
              const Divider(height: 1),
              _RadioTile<String>(
                icon: Icons.language_rounded,
                title: l10n.translate('arabic'),
                subtitle: 'العربية (مصري)',
                value: 'ar',
                groupValue: settings.locale.languageCode,
                onChanged: (_) => settings.setLocale(const Locale('ar', '')),
              ),
            ],
          ),

          // ═══════════════════════════════════════════════════════
          //  APP THEME
          // ═══════════════════════════════════════════════════════
          _SectionHeader(
            icon: Icons.palette_rounded,
            title: l10n.translate('app_theme'),
          ),
          _SettingsCard(
            children: [
              _ThemeTile(
                icon: Icons.light_mode_rounded,
                title: l10n.translate('light_mode'),
                isSelected: !settings.isHighContrast &&
                    settings.themeMode == ThemeMode.light,
                onTap: () => settings.setThemeMode(ThemeMode.light),
                color: const Color(0xFFF59E0B),
              ),
              const Divider(height: 1),
              _ThemeTile(
                icon: Icons.dark_mode_rounded,
                title: l10n.translate('dark_mode'),
                isSelected: !settings.isHighContrast &&
                    settings.themeMode == ThemeMode.dark,
                onTap: () => settings.setThemeMode(ThemeMode.dark),
                color: const Color(0xFF7C3AED),
              ),
              const Divider(height: 1),
              _ThemeTile(
                icon: Icons.contrast_rounded,
                title: l10n.translate('high_contrast'),
                isSelected: settings.isHighContrast,
                onTap: () => settings.enableHighContrast(),
                color: const Color(0xFF00FF88),
              ),
            ],
          ),

          // ═══════════════════════════════════════════════════════
          //  PRIVACY & SECURITY
          // ═══════════════════════════════════════════════════════
          _SectionHeader(
            icon: Icons.shield_rounded,
            title: l10n.translate('privacy_security'),
          ),
          _SettingsCard(
            children: [
              // Encryption info
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: AppColors.successGreen.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(
                    Icons.lock_rounded,
                    color: AppColors.successGreen,
                    size: 22,
                  ),
                ),
                title: Text(
                  l10n.translate('data_encryption'),
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                subtitle: Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(
                    l10n.translate('encryption_desc'),
                    style: TextStyle(
                      fontSize: 13,
                      color: theme.colorScheme.onSurface.withOpacity(0.6),
                      height: 1.4,
                    ),
                  ),
                ),
                isThreeLine: true,
              ),
              const Divider(height: 1),
              // Permission toggles
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
                child: Text(
                  l10n.translate('permissions'),
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: theme.colorScheme.primary,
                    letterSpacing: 0.5,
                  ),
                ),
              ),
              _SwitchTile(
                icon: Icons.camera_alt_rounded,
                title: l10n.translate('camera_access'),
                value: settings.cameraPermission,
                onChanged: settings.setCameraPermission,
              ),
              const Divider(height: 1),
              _SwitchTile(
                icon: Icons.location_on_rounded,
                title: l10n.translate('location_tracking'),
                value: settings.locationPermission,
                onChanged: settings.setLocationPermission,
              ),
              const Divider(height: 1),
              _SwitchTile(
                icon: Icons.mic_rounded,
                title: l10n.translate('microphone_use'),
                value: settings.microphonePermission,
                onChanged: settings.setMicrophonePermission,
              ),
            ],
          ),

          // ═══════════════════════════════════════════════════════
          //  CONNECTIVITY
          // ═══════════════════════════════════════════════════════
          _SectionHeader(
            icon: Icons.bluetooth_rounded,
            title: l10n.translate('connectivity'),
          ),
          _SettingsCard(
            children: [
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: (ble.isConnected ? AppColors.primaryBlue : AppColors.neutralGray400)
                        .withOpacity(0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(
                    Icons.bluetooth_connected_rounded,
                    color: ble.isConnected ? AppColors.primaryBlue : AppColors.neutralGray400,
                    size: 22,
                  ),
                ),
                title: Text(
                  l10n.translate('bluetooth_ble'),
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                subtitle: Text(
                  ble.isConnected
                      ? '${ble.connectedDeviceName} (${ble.signalStrength})'
                      : l10n.translate('disconnected'),
                ),
                trailing: ble.status == BleConnectionStatus.scanning
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : TextButton(
                        onPressed: () => ble.startScan(),
                        child: Text(l10n.translate('scan_devices')),
                      ),
              ),
              if (ble.discoveredDevices.isNotEmpty) ...[
                const Divider(height: 1),
                ...ble.discoveredDevices.map((device) => ListTile(
                      leading: const Icon(Icons.devices_rounded, size: 20),
                      title: Text(device['name'] ?? 'Unknown'),
                      subtitle: Text(device['id'] ?? ''),
                      trailing: ble.connectedDeviceId == device['id']
                          ? Chip(
                              label: Text(l10n.translate('connected')),
                              backgroundColor:
                                  AppColors.successGreen.withOpacity(0.1),
                              labelStyle: const TextStyle(
                                color: AppColors.successGreen,
                                fontSize: 12,
                              ),
                            )
                          : OutlinedButton(
                              onPressed: () => ble.connectToDevice(
                                device['id']!,
                                device['name']!,
                              ),
                              child: const Text('Connect'),
                            ),
                      dense: true,
                    )),
              ],
            ],
          ),

          // ─── About ──────────────────────────────────────────
          const SizedBox(height: 24),
          Center(
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    gradient: AppColors.primaryGradient,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: const Icon(
                    Icons.health_and_safety_rounded,
                    color: Colors.white,
                    size: 32,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  l10n.translate('about'),
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    color: theme.colorScheme.onSurface,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  l10n.translate('version'),
                  style: TextStyle(
                    fontSize: 13,
                    color: theme.colorScheme.onSurface.withOpacity(0.5),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 40),
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════
//  HELPER WIDGETS
// ═══════════════════════════════════════════════════════════════════

class _SectionHeader extends StatelessWidget {
  final IconData icon;
  final String title;

  const _SectionHeader({required this.icon, required this.title});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
      child: Row(
        children: [
          Icon(icon, size: 18, color: theme.colorScheme.primary),
          const SizedBox(width: 8),
          Text(
            title,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: theme.colorScheme.primary,
              letterSpacing: 0.5,
            ),
          ),
        ],
      ),
    );
  }
}

class _SettingsCard extends StatelessWidget {
  final List<Widget> children;

  const _SettingsCard({required this.children});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Card(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: children,
        ),
      ),
    );
  }
}

class _SwitchTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? subtitle;
  final bool value;
  final ValueChanged<bool> onChanged;

  const _SwitchTile({
    required this.icon,
    required this.title,
    this.subtitle,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SwitchListTile.adaptive(
      secondary: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: (value ? theme.colorScheme.primary : AppColors.neutralGray400)
              .withOpacity(0.1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(
          icon,
          size: 20,
          color: value ? theme.colorScheme.primary : AppColors.neutralGray400,
        ),
      ),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
      subtitle: subtitle != null
          ? Text(subtitle!, style: const TextStyle(fontSize: 13))
          : null,
      value: value,
      onChanged: onChanged,
    );
  }
}

class _RadioTile<T> extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? subtitle;
  final T value;
  final T groupValue;
  final ValueChanged<T?> onChanged;

  const _RadioTile({
    required this.icon,
    required this.title,
    this.subtitle,
    required this.value,
    required this.groupValue,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final isSelected = value == groupValue;
    final theme = Theme.of(context);

    return RadioListTile<T>(
      secondary: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: (isSelected ? theme.colorScheme.primary : AppColors.neutralGray400)
              .withOpacity(0.1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(
          icon,
          size: 20,
          color: isSelected ? theme.colorScheme.primary : AppColors.neutralGray400,
        ),
      ),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
      subtitle: subtitle != null
          ? Text(subtitle!, style: const TextStyle(fontSize: 13))
          : null,
      value: value,
      groupValue: groupValue,
      onChanged: onChanged,
    );
  }
}

class _ThemeTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final bool isSelected;
  final VoidCallback onTap;
  final Color color;

  const _ThemeTile({
    required this.icon,
    required this.title,
    required this.isSelected,
    required this.onTap,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(icon, size: 20, color: color),
      ),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
      trailing: isSelected
          ? Container(
              padding: const EdgeInsets.all(2),
              decoration: BoxDecoration(
                color: theme.colorScheme.primary,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.check, color: Colors.white, size: 16),
            )
          : null,
      onTap: onTap,
    );
  }
}

class _ProfileTile extends StatelessWidget {
  final SettingsProvider settings;
  final AppLocalizations l10n;

  const _ProfileTile({required this.settings, required this.l10n});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return ListTile(
      leading: Container(
        width: 48,
        height: 48,
        decoration: BoxDecoration(
          gradient: AppColors.primaryGradient,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Center(
          child: Text(
            settings.userName.isNotEmpty
                ? settings.userName[0].toUpperCase()
                : 'U',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ),
      title: Text(
        settings.userName,
        style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16),
      ),
      subtitle: Text(
        settings.userId.isNotEmpty
            ? 'ID: ${settings.userId}'
            : l10n.translate('profile_management'),
        style: TextStyle(
          fontSize: 13,
          color: theme.colorScheme.onSurface.withOpacity(0.6),
        ),
      ),
      trailing: Icon(
        Icons.edit_rounded,
        size: 20,
        color: theme.colorScheme.primary,
      ),
      onTap: () => _showEditProfileDialog(context, settings, l10n),
    );
  }

  void _showEditProfileDialog(
    BuildContext context,
    SettingsProvider settings,
    AppLocalizations l10n,
  ) {
    final nameController = TextEditingController(text: settings.userName);
    final idController = TextEditingController(text: settings.userId);

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Text(l10n.translate('profile_management')),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameController,
              decoration: InputDecoration(
                labelText: l10n.translate('user_name'),
                prefixIcon: const Icon(Icons.person_rounded),
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: idController,
              decoration: InputDecoration(
                labelText: l10n.translate('user_id'),
                prefixIcon: const Icon(Icons.badge_rounded),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text(l10n.translate('cancel')),
          ),
          ElevatedButton(
            onPressed: () {
              settings.updateProfile(
                name: nameController.text,
                id: idController.text,
              );
              Navigator.of(ctx).pop();
            },
            child: Text(l10n.translate('save')),
          ),
        ],
      ),
    );
  }
}

class _DisabilityTile extends StatelessWidget {
  final SettingsProvider settings;
  final AppLocalizations l10n;

  const _DisabilityTile({required this.settings, required this.l10n});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final options = [
      l10n.translate('none_selected'),
      l10n.translate('visual_impairment'),
      l10n.translate('motor_disability'),
      l10n.translate('hearing_impairment'),
      l10n.translate('cognitive_disability'),
    ];

    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: theme.colorScheme.primary.withOpacity(0.1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(
          Icons.accessibility_new_rounded,
          size: 20,
          color: theme.colorScheme.primary,
        ),
      ),
      title: Text(
        l10n.translate('disability_type'),
        style: const TextStyle(fontWeight: FontWeight.w500),
      ),
      subtitle: Text(
        settings.disabilityType,
        style: const TextStyle(fontSize: 13),
      ),
      trailing: Icon(
        Icons.chevron_right_rounded,
        color: theme.colorScheme.onSurface.withOpacity(0.4),
      ),
      onTap: () {
        showModalBottomSheet(
          context: context,
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          builder: (ctx) => SafeArea(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const SizedBox(height: 12),
                Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: theme.colorScheme.onSurface.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  l10n.translate('disability_type'),
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 8),
                ...options.map((option) => ListTile(
                      leading: Radio<String>(
                        value: option,
                        groupValue: settings.disabilityType,
                        onChanged: (v) {
                          settings.updateProfile(disability: v);
                          Navigator.of(ctx).pop();
                        },
                      ),
                      title: Text(option),
                      onTap: () {
                        settings.updateProfile(disability: option);
                        Navigator.of(ctx).pop();
                      },
                    )),
                const SizedBox(height: 16),
              ],
            ),
          ),
        );
      },
    );
  }
}
