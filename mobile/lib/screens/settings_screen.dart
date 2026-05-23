import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';

import '../providers/auth_provider.dart';
import '../providers/settings_provider.dart';
import 'auth/login_screen.dart';
import '../providers/ble_provider.dart';
import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../l10n/app_localizations.dart';
import '../widgets/nova_logo.dart';
import '../widgets/nc_primitives.dart';

/// SettingsScreen — SKILL §4.6.
///
/// Sections: Profile · Account · Accessibility · Preferences · Robot.
/// Logic preserved from previous implementation
/// (provider-backed user profile, locale, theme, permissions, BLE).
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final settings = context.watch<SettingsProvider>();
    final ble = context.watch<BleProvider>();
    final rover = context.watch<RoverProvider>();

    return Scaffold(
      backgroundColor: AppColors.canvas,
      body: Column(
        children: [
          NcAppBar(
            leading: const NovaLogo(),
            title: Text(l10n.translate('settings'), style: AppText.appBarTitle()),
            battery: rover.batteryLevel,
          ),
          Expanded(
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsetsDirectional.fromSTEB(20, 8, 20, 40),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(l10n.translate('settings'), style: AppText.display1()),
                  const SizedBox(height: 16),

                  // ─── Profile card ──────────────────────────────
                  _ProfileCard(
                    settings: settings,
                    l10n: l10n,
                    onEditProfile: () => _editProfile(context, settings, l10n),
                  ),

                  // ─── Account ───────────────────────────────────
                  const NcSectionHead(title: 'Account'),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.supervisor_account_rounded),
                        title: 'Guardian access',
                        subtitle: '2 caregivers',
                        trailing: const Icon(
                          Icons.chevron_right_rounded,
                          color: AppColors.inkMuted,
                        ),
                        onTap: () {},
                      ),
                      NcRow(
                        icon: const Icon(Icons.badge_rounded),
                        title: l10n.translate('user_id'),
                        subtitle: settings.userId.isEmpty
                            ? 'Not set'
                            : settings.userId,
                        trailing: const Icon(
                          Icons.chevron_right_rounded,
                          color: AppColors.inkMuted,
                        ),
                        onTap: () => _editProfile(context, settings, l10n),
                      ),
                      NcRow(
                        icon: const Icon(Icons.logout_rounded, color: AppColors.danger),
                        iconBg: AppColors.danger2,
                        title: 'Sign out',
                        subtitle: 'Return to the login screen',
                        trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.inkMuted),
                        onTap: () => _confirmSignOut(context),
                      ),
                    ],
                  ),

                  // ─── Accessibility ─────────────────────────────
                  const NcSectionHead(title: 'Accessibility'),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.contrast_rounded),
                        title: l10n.translate('high_contrast'),
                        subtitle: 'Bold borders, ink-on-paper palette',
                        trailing: NcSwitch(
                          value: settings.isHighContrast,
                          onChanged: (v) {
                            if (v) {
                              settings.enableHighContrast();
                            } else {
                              settings.setThemeMode(ThemeMode.light);
                            }
                          },
                          semanticLabel: l10n.translate('high_contrast'),
                        ),
                      ),
                      NcRow(
                        icon: const Icon(Icons.text_increase_rounded),
                        title: 'Larger text',
                        subtitle: 'Boost readable text size',
                        trailing: NcSwitch(
                          value: settings.largeTextEnabled,
                          onChanged: (v) =>
                              settings.updateProfile(largeTextEnabled: v),
                          semanticLabel: 'Larger text',
                        ),
                      ),
                      NcRow(
                        icon: const Icon(Icons.record_voice_over_rounded),
                        title: l10n.translate('voice_feedback'),
                        subtitle: 'Spoken UI for low-vision users',
                        trailing: NcSwitch(
                          value: settings.voiceFeedbackEnabled,
                          onChanged: (v) =>
                              settings.updateProfile(voiceFeedback: v),
                          semanticLabel: l10n.translate('voice_feedback'),
                        ),
                      ),
                      NcRow(
                        icon: const Icon(Icons.accessibility_new_rounded),
                        title: l10n.translate('disability_type'),
                        subtitle: settings.disabilityType,
                        trailing: const Icon(
                          Icons.chevron_right_rounded,
                          color: AppColors.inkMuted,
                        ),
                        onTap: () => _pickDisability(context, settings, l10n),
                      ),
                    ],
                  ),

                  // ─── Preferences ───────────────────────────────
                  const NcSectionHead(title: 'Preferences'),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.translate_rounded),
                        title: l10n.translate('language'),
                        subtitle: settings.isArabic
                            ? l10n.translate('arabic')
                            : l10n.translate('english'),
                        trailing: _LangChip(
                          isArabic: settings.isArabic,
                          onTap: settings.toggleLanguage,
                        ),
                      ),
                      NcRow(
                        icon: Icon(
                          settings.themeMode == ThemeMode.dark
                              ? Icons.dark_mode_rounded
                              : Icons.light_mode_rounded,
                        ),
                        title: l10n.translate('app_theme'),
                        subtitle: settings.themeLabel,
                        trailing: const Icon(
                          Icons.chevron_right_rounded,
                          color: AppColors.inkMuted,
                        ),
                        onTap: () => _pickTheme(context, settings, l10n),
                      ),
                    ],
                  ),

                  // ─── Privacy ───────────────────────────────────
                  const NcSectionHead(title: 'Privacy & permissions'),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.camera_alt_rounded),
                        title: l10n.translate('camera_access'),
                        trailing: NcSwitch(
                          value: settings.cameraPermission,
                          onChanged: settings.setCameraPermission,
                          semanticLabel: l10n.translate('camera_access'),
                        ),
                      ),
                      NcRow(
                        icon: const Icon(Icons.mic_rounded),
                        title: l10n.translate('microphone_use'),
                        trailing: NcSwitch(
                          value: settings.microphonePermission,
                          onChanged: settings.setMicrophonePermission,
                          semanticLabel: l10n.translate('microphone_use'),
                        ),
                      ),
                      NcRow(
                        icon: const Icon(Icons.location_on_rounded),
                        title: l10n.translate('location_tracking'),
                        trailing: NcSwitch(
                          value: settings.locationPermission,
                          onChanged: settings.setLocationPermission,
                          semanticLabel: l10n.translate('location_tracking'),
                        ),
                      ),
                    ],
                  ),

                  // ─── Robot ─────────────────────────────────────
                  const NcSectionHead(title: 'Robot'),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.smart_toy_rounded),
                        title: ble.connectedDeviceName ?? 'SERBOT-NC-001',
                        subtitle: ble.isConnected
                            ? '${l10n.translate('connected')} · ${ble.signalStrength}'
                            : l10n.translate('disconnected'),
                        trailing: ble.status == BleConnectionStatus.scanning
                            ? const SizedBox(
                                width: 22,
                                height: 22,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              )
                            : NcChip(
                                label: ble.isConnected ? 'Connected' : 'Pair',
                                style: ble.isConnected
                                    ? NcChipStyle.success
                                    : NcChipStyle.normal,
                              ),
                        onTap: () => _openBleSheet(context, ble, l10n),
                      ),
                      NcRow(
                        icon: const Icon(Icons.qr_code_2_rounded),
                        title: 'Pair with QR code',
                        subtitle: 'Scan the QR sticker on the rover',
                        trailing: const Icon(
                          Icons.chevron_right_rounded,
                          color: AppColors.inkMuted,
                        ),
                        onTap: () {
                          // TODO(feature): QR scanner screen
                          // (mobile_scanner package); on success, hand the
                          // device ID to BleProvider.connectToDevice().
                        },
                      ),
                    ],
                  ),

                  // ─── Footer ────────────────────────────────────
                  const SizedBox(height: 32),
                  Center(
                    child: Text(
                      'NovaCare · ${l10n.translate('version')}',
                      style: AppText.caption(),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ──────────────────────────────────────────────────────────────────
  void _editProfile(
    BuildContext context,
    SettingsProvider settings,
    AppLocalizations l10n,
  ) {
    final nameController = TextEditingController(text: settings.userName);
    final idController = TextEditingController(text: settings.userId);

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.paper,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(Radii.lg),
        ),
        title: Text(
          l10n.translate('profile_management'),
          style: AppText.display3(),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            GestureDetector(
              onTap: () async {
                final ImagePicker picker = ImagePicker();
                final XFile? image = await picker.pickImage(source: ImageSource.gallery);
                if (image != null) {
                  settings.updateProfile(profileImagePath: image.path);
                  if (ctx.mounted) Navigator.of(ctx).pop();
                }
              },
              child: Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: AppColors.accent3,
                  shape: BoxShape.circle,
                  image: (settings.profileImagePath?.isNotEmpty ?? false)
                      ? DecorationImage(
                          image: FileImage(File(settings.profileImagePath!)),
                          fit: BoxFit.cover,
                        )
                      : null,
                ),
                child: (settings.profileImagePath?.isEmpty ?? true)
                    ? const Icon(Icons.camera_alt, color: AppColors.inkTeal)
                    : null,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: nameController,
              decoration: InputDecoration(
                labelText: l10n.translate('user_name'),
                prefixIcon: const Icon(Icons.person_rounded),
              ),
            ),
            const SizedBox(height: 12),
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

  void _pickDisability(
    BuildContext context,
    SettingsProvider settings,
    AppLocalizations l10n,
  ) {
    final options = [
      l10n.translate('none_selected'),
      l10n.translate('visual_impairment'),
      l10n.translate('motor_disability'),
      l10n.translate('hearing_impairment'),
      l10n.translate('cognitive_disability'),
    ];

    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.paper,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
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
                color: AppColors.inkLight,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                l10n.translate('disability_type'),
                style: AppText.display3(),
              ),
            ),
            ...options.map(
              (o) => NcRow(
                icon: Icon(
                  o == settings.disabilityType
                      ? Icons.radio_button_checked_rounded
                      : Icons.radio_button_unchecked_rounded,
                ),
                title: o,
                onTap: () {
                  settings.updateProfile(disability: o);
                  Navigator.of(ctx).pop();
                },
              ),
            ),
            const SizedBox(height: 12),
          ],
        ),
      ),
    );
  }

  void _openBleSheet(
    BuildContext context,
    BleProvider ble,
    AppLocalizations l10n,
  ) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.paper,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) => _BleSheet(),
    );
    // Kick off a scan as soon as the sheet opens.
    if (ble.status != BleConnectionStatus.scanning) ble.startScan();
  }

  void _pickTheme(
    BuildContext context,
    SettingsProvider settings,
    AppLocalizations l10n,
  ) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.paper,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
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
                color: AppColors.inkLight,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Text(l10n.translate('app_theme'), style: AppText.display3()),
            ),
            NcRow(
              icon: const Icon(Icons.light_mode_rounded),
              title: l10n.translate('light_mode'),
              onTap: () {
                settings.setThemeMode(ThemeMode.light);
                Navigator.pop(ctx);
              },
            ),
            NcRow(
              icon: const Icon(Icons.dark_mode_rounded),
              title: l10n.translate('dark_mode'),
              onTap: () {
                settings.setThemeMode(ThemeMode.dark);
                Navigator.pop(ctx);
              },
            ),
            NcRow(
              icon: const Icon(Icons.contrast_rounded),
              title: l10n.translate('high_contrast'),
              onTap: () {
                settings.enableHighContrast();
                Navigator.pop(ctx);
              },
            ),
            const SizedBox(height: 12),
          ],
        ),
      ),
    );
  }

  void _confirmSignOut(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Radii.lg)),
        backgroundColor: AppColors.paper,
        title: Row(
          children: [
            const Icon(Icons.logout_rounded, color: AppColors.danger),
            const SizedBox(width: 10),
            Text('Sign out?', style: AppText.display3()),
          ],
        ),
        content: Text('You will be returned to the login screen.', style: AppText.body()),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('CANCEL'),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.danger,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Radii.sm)),
            ),
            onPressed: () async {
              Navigator.pop(ctx);
              await context.read<AuthProvider>().logout();
              if (context.mounted) {
                Navigator.of(context).pushAndRemoveUntil(
                  MaterialPageRoute(builder: (_) => const LoginScreen()),
                  (_) => false,
                );
              }
            },
            child: const Text('SIGN OUT'),
          ),
        ],
      ),
    );
  }
}


// ════════════════════════════════════════════════════════════════════
//  Profile card
// ════════════════════════════════════════════════════════════════════
class _ProfileCard extends StatelessWidget {
  final SettingsProvider settings;
  final AppLocalizations l10n;
  final VoidCallback onEditProfile;
  const _ProfileCard({
    required this.settings,
    required this.l10n,
    required this.onEditProfile,
  });

  @override
  Widget build(BuildContext context) {
    final initials = settings.userName.isEmpty
        ? 'U'
        : settings.userName
            .trim()
            .split(RegExp(r'\s+'))
            .take(2)
            .map((s) => s.isNotEmpty ? s[0] : '')
            .join()
            .toUpperCase();

    return GestureDetector(
      onTap: onEditProfile,
      child: Container(
        padding: const EdgeInsetsDirectional.all(16),
        decoration: BoxDecoration(
          color: AppColors.paper,
          borderRadius: BorderRadius.circular(Radii.lg),
          border: Border.all(color: AppColors.line),
        ),
        child: Row(
          children: [
          Container(
            width: 52,
            height: 52,
            decoration: BoxDecoration(
              color: AppColors.accent3,
              shape: BoxShape.circle,
              border: Border.all(color: AppColors.accent2),
              image: (settings.profileImagePath?.isNotEmpty ?? false)
                  ? DecorationImage(
                      image: FileImage(File(settings.profileImagePath!)),
                      fit: BoxFit.cover,
                    )
                  : null,
            ),
            alignment: Alignment.center,
            child: (settings.profileImagePath?.isEmpty ?? true)
                ? Text(
                    initials,
                    style: AppText.display3(color: AppColors.inkTeal)
                        .copyWith(fontWeight: FontWeight.w800),
                  )
                : null,
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  settings.userName.isEmpty
                      ? 'Amira Hassan'
                      : settings.userName,
                  style: AppText.display3(),
                ),
                const SizedBox(height: 2),
                Text(
                  'Tap to edit profile',
                  style: AppText.caption(),
                ),
              ],
            ),
          ),
          Icon(
            Icons.edit_rounded,
            color: AppColors.brandTeal,
            size: 20,
          ),
        ],
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  BLE device picker bottom sheet
//  Re-implements the discovered-devices list from the previous Settings
//  screen, restyled with NovaCare tokens.
// ════════════════════════════════════════════════════════════════════
class _BleSheet extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final ble = context.watch<BleProvider>();
    final l10n = AppLocalizations.of(context);

    return SafeArea(
      child: Padding(
        padding: const EdgeInsetsDirectional.fromSTEB(20, 12, 20, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: AppColors.inkLight,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: Text(
                    l10n.translate('bluetooth_ble'),
                    style: AppText.display3(),
                  ),
                ),
                if (ble.status == BleConnectionStatus.scanning)
                  const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                else
                  GestureDetector(
                    onTap: ble.startScan,
                    child: NcChip(
                      label: l10n.translate('scan_devices'),
                      style: NcChipStyle.info,
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            if (ble.discoveredDevices.isEmpty &&
                ble.status != BleConnectionStatus.scanning)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 24),
                child: Center(
                  child: Text(
                    'No devices found yet.',
                    style: AppText.body(color: AppColors.inkMuted),
                  ),
                ),
              )
            else
              NcGroup(
                children: [
                  for (final d in ble.discoveredDevices)
                    NcRow(
                      icon: const Icon(Icons.smart_toy_rounded),
                      title: d['name'] ?? 'Unknown',
                      subtitle: d['id'] ?? '',
                      trailing: ble.connectedDeviceId == d['id']
                          ? const NcChip(
                              label: 'Connected',
                              style: NcChipStyle.success,
                            )
                          : GestureDetector(
                              onTap: () => ble.connectToDevice(
                                d['id']!,
                                d['name']!,
                              ),
                              child: const NcChip(
                                label: 'Connect',
                                style: NcChipStyle.info,
                              ),
                            ),
                    ),
                ],
              ),
            const SizedBox(height: 16),
            // TODO(feature): "Pair via QR" entry that opens mobile_scanner
            // and feeds the scanned device ID to BleProvider.connectToDevice.
          ],
        ),
      ),
    );
  }
}

class _LangChip extends StatelessWidget {
  final bool isArabic;
  final VoidCallback onTap;
  const _LangChip({required this.isArabic, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: AppColors.canvas2,
          borderRadius: BorderRadius.circular(Radii.pill),
          border: Border.all(color: AppColors.line),
        ),
        child: Text(
          isArabic ? 'AR' : 'EN',
          style: AppText.caption(color: AppColors.inkNavy)
              .copyWith(fontWeight: FontWeight.w700, fontSize: 12),
        ),
      ),
    );
  }
}
