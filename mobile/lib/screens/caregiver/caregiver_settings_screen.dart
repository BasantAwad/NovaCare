import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/settings_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_text_styles.dart';
import '../../widgets/nc_primitives.dart';
import '../../widgets/nova_logo.dart';
import '../auth/login_screen.dart';

class CaregiverSettingsScreen extends StatelessWidget {
  const CaregiverSettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final auth     = context.watch<AuthProvider>();
    final settings = context.watch<SettingsProvider>();

    return Scaffold(
      backgroundColor: AppColors.canvas,
      body: Column(
        children: [
          NcAppBar(
            leading: const NovaLogo(),
            title: Text('Settings', style: AppText.appBarTitle()),
          ),
          Expanded(
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsetsDirectional.fromSTEB(20, 8, 20, 40),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Settings', style: AppText.display1()),
                  const SizedBox(height: 20),

                  // ─── Account card ──────────────────────────────────
                  _AccountCard(auth: auth),
                  const SizedBox(height: 8),

                  // ─── Account actions ───────────────────────────────
                  const NcSectionHead(title: 'Account'),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.badge_rounded),
                        title: 'Role',
                        subtitle: auth.isCaregiver ? 'Caregiver' : 'Patient',
                        trailing: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: AppColors.brandAquaSoft,
                            borderRadius: BorderRadius.circular(Radii.pill),
                          ),
                          child: Text(
                            auth.isCaregiver ? 'Caregiver' : 'Patient',
                            style: AppText.caption(color: AppColors.brandTeal)
                                .copyWith(fontWeight: FontWeight.w700),
                          ),
                        ),
                        onTap: () {},
                      ),
                      NcRow(
                        icon: const Icon(Icons.email_outlined),
                        title: 'Email',
                        subtitle: auth.email.isNotEmpty ? auth.email : 'Not set',
                        trailing: const Icon(Icons.chevron_right_rounded, color: AppColors.inkMuted),
                        onTap: () {},
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

                  // ─── Accessibility ─────────────────────────────────
                  const NcSectionHead(title: 'Accessibility'),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.contrast_rounded),
                        title: 'High contrast',
                        subtitle: 'Bold borders, ink-on-paper palette',
                        trailing: NcSwitch(
                          value: settings.isHighContrast,
                          onChanged: (v) => v
                              ? settings.enableHighContrast()
                              : settings.setThemeMode(ThemeMode.light),
                          semanticLabel: 'High contrast',
                        ),
                      ),
                      NcRow(
                        icon: const Icon(Icons.text_increase_rounded),
                        title: 'Larger text',
                        subtitle: 'Boost readable text size',
                        trailing: NcSwitch(
                          value: settings.largeTextEnabled,
                          onChanged: (v) => settings.updateProfile(largeTextEnabled: v),
                          semanticLabel: 'Larger text',
                        ),
                      ),
                    ],
                  ),

                  // ─── Appearance ────────────────────────────────────
                  const NcSectionHead(title: 'Appearance'),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.light_mode_rounded),
                        title: 'Light',
                        trailing: settings.themeMode == ThemeMode.light && !settings.isHighContrast
                            ? const Icon(Icons.check_rounded, color: AppColors.brandTeal)
                            : const Icon(Icons.chevron_right_rounded, color: AppColors.inkMuted),
                        onTap: () => settings.setThemeMode(ThemeMode.light),
                      ),
                      NcRow(
                        icon: const Icon(Icons.dark_mode_rounded),
                        title: 'Dark',
                        trailing: settings.themeMode == ThemeMode.dark && !settings.isHighContrast
                            ? const Icon(Icons.check_rounded, color: AppColors.brandTeal)
                            : const Icon(Icons.chevron_right_rounded, color: AppColors.inkMuted),
                        onTap: () => settings.setThemeMode(ThemeMode.dark),
                      ),
                      NcRow(
                        icon: const Icon(Icons.devices_rounded),
                        title: 'System default',
                        trailing: settings.themeMode == ThemeMode.system && !settings.isHighContrast
                            ? const Icon(Icons.check_rounded, color: AppColors.brandTeal)
                            : const Icon(Icons.chevron_right_rounded, color: AppColors.inkMuted),
                        onTap: () => settings.setThemeMode(ThemeMode.system),
                      ),
                    ],
                  ),

                  const SizedBox(height: 32),

                  // ─── Big sign-out button ───────────────────────────
                  GestureDetector(
                    onTap: () => _confirmSignOut(context),
                    child: Container(
                      height: 54,
                      decoration: BoxDecoration(
                        color: AppColors.danger2,
                        borderRadius: BorderRadius.circular(Radii.sm),
                        border: Border.all(color: AppColors.danger.withValues(alpha: 0.3)),
                      ),
                      alignment: Alignment.center,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.logout_rounded, color: AppColors.danger, size: 20),
                          const SizedBox(width: 10),
                          Text('Sign out', style: AppText.bodyStrong(color: AppColors.danger)),
                        ],
                      ),
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
//  Account card at the top
// ════════════════════════════════════════════════════════════════════
class _AccountCard extends StatelessWidget {
  final AuthProvider auth;
  const _AccountCard({required this.auth});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(Radii.md),
        border: Border.all(color: AppColors.line),
        boxShadow: Elevations.e1,
      ),
      child: Row(
        children: [
          Container(
            width: 52, height: 52,
            decoration: const BoxDecoration(
              color: AppColors.brandAquaSoft,
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.medical_services_outlined, color: AppColors.brandTeal, size: 26),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  auth.name.isNotEmpty ? auth.name : 'Caregiver',
                  style: AppText.bodyStrong(),
                ),
                Text(
                  auth.email.isNotEmpty ? auth.email : '—',
                  style: AppText.caption(),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.brandAquaSoft,
              borderRadius: BorderRadius.circular(Radii.pill),
            ),
            child: Text(
              'Caregiver',
              style: AppText.caption(color: AppColors.brandTeal).copyWith(fontWeight: FontWeight.w700),
            ),
          ),
        ],
      ),
    );
  }
}
