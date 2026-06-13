import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import 'package:url_launcher/url_launcher.dart';
import '../live_feed_screen.dart';

import '../../providers/alert_provider.dart';
import '../../providers/auth_provider.dart';
import '../../providers/rover_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_text_styles.dart';
import '../../widgets/nc_primitives.dart';
import '../../widgets/nova_logo.dart';

/// Caregiver dashboard — shows live SOS alerts and patient status.
class CaregiverHomeScreen extends StatelessWidget {
  const CaregiverHomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final auth    = context.watch<AuthProvider>();
    final alerts  = context.watch<AlertProvider>();
    final rover   = context.watch<RoverProvider>();

    final sosAlerts  = alerts.alerts.where((a) => a.severity == AlertSeverity.error).toList();
    final otherAlerts = alerts.alerts.where((a) => a.severity != AlertSeverity.error).toList();
    final unread     = alerts.unreadCount;

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Column(
        children: [
          NcAppBar(
            leading: const NovaLogo(),
            title: Text('Caregiver', style: AppText.appBarTitle()),
            battery: rover.batteryLevel,
            status: rover.isConnected ? NcConnectionStatus.online : NcConnectionStatus.offline,
            statusLabel: rover.isConnected ? 'Live' : 'Offline',
          ),
          Expanded(
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsetsDirectional.fromSTEB(20, 8, 20, 40),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ─── Greeting ───────────────────────────────────────
                  Text('Hello, ${auth.name}', style: AppText.display1()),
                  const SizedBox(height: 4),
                  Text(
                    unread > 0
                        ? '$unread unread alert${unread > 1 ? 's' : ''} — check below.'
                        : 'All patients are safe. No active alerts.',
                    style: AppText.body(color: AppColors.inkMuted),
                  ),
                  const SizedBox(height: 20),

                  // ─── SOS alerts ────────────────────────────────────
                  if (sosAlerts.isNotEmpty) ...[
                    _SectionHead(
                      title: 'SOS Alerts',
                      count: sosAlerts.length,
                      countColor: AppColors.danger,
                    ),
                    const SizedBox(height: 8),
                    ...sosAlerts.map((a) => Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: _SosAlertCard(
                        alert: a,
                        onAcknowledge: () {
                          HapticFeedback.mediumImpact();
                          alerts.markAsRead(a.id);
                        },
                        onDismiss: () {
                          HapticFeedback.lightImpact();
                          alerts.dismissAlert(a.id);
                        },
                        onCall: () => _callPatient(context, auth),
                      ),
                    )),
                    const SizedBox(height: 16),
                  ] else ...[
                    _AllClearCard(),
                    const SizedBox(height: 16),
                  ],

                  // ─── Patient overview ──────────────────────────────
                  const NcSectionHead(title: 'Patient overview'),
                  _PatientCard(
                    rover: rover,
                    patientName: 'Your Patient',
                    onCall: () => _callPatient(context, auth),
                  ),
                  const SizedBox(height: 16),

                  // ─── Other alerts ──────────────────────────────────
                  if (otherAlerts.isNotEmpty) ...[
                    NcSectionHead(
                      title: 'Recent activity',
                      action: GestureDetector(
                        onTap: () => alerts.markAllAsRead(),
                        child: Text('Read all', style: AppText.caption(color: AppColors.brandTeal)
                            .copyWith(fontWeight: FontWeight.w700)),
                      ),
                    ),
                    NcGroup(
                      children: otherAlerts.take(5).map((a) {
                        final time = DateFormat('h:mm a').format(a.timestamp);
                        return NcRow(
                          icon: Icon(a.icon, color: a.color),
                          iconBg: a.color.withOpacity(0.12),
                          title: a.title,
                          subtitle: '$time  •  ${a.body}',
                          trailing: a.isRead
                              ? const Icon(Icons.chevron_right_rounded, color: AppColors.inkMuted)
                              : Container(
                                  width: 8, height: 8,
                                  decoration: const BoxDecoration(
                                    color: AppColors.accent, shape: BoxShape.circle,
                                  ),
                                ),
                          onTap: () => alerts.markAsRead(a.id),
                        );
                      }).toList(),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _callPatient(BuildContext context, AuthProvider auth) async {
    final phone = auth.ecPhone.trim();
    if (phone.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No emergency contact phone configured.')),
      );
      return;
    }
    final uri = Uri.parse('tel:$phone');
    if (await canLaunchUrl(uri)) await launchUrl(uri);
  }
}

// ════════════════════════════════════════════════════════════════════
//  SOS Alert Card — prominent red card with action buttons
// ════════════════════════════════════════════════════════════════════
class _SosAlertCard extends StatelessWidget {
  final AlertModel alert;
  final VoidCallback onAcknowledge;
  final VoidCallback onDismiss;
  final VoidCallback onCall;

  const _SosAlertCard({
    required this.alert,
    required this.onAcknowledge,
    required this.onDismiss,
    required this.onCall,
  });

  @override
  Widget build(BuildContext context) {
    final time = DateFormat('h:mm a · MMM d').format(alert.timestamp);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: alert.isRead ? Theme.of(context).colorScheme.surface : AppColors.danger2,
        borderRadius: BorderRadius.circular(Radii.md),
        border: Border.all(
          color: alert.isRead ? AppColors.line : AppColors.danger.withOpacity(0.4),
          width: alert.isRead ? 1 : 1.5,
        ),
        boxShadow: alert.isRead ? [] : [
          BoxShadow(color: AppColors.danger.withOpacity(0.10), blurRadius: 16, offset: const Offset(0, 4)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 36, height: 36,
                decoration: BoxDecoration(
                  color: AppColors.danger.withOpacity(0.15),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.emergency_rounded, color: AppColors.danger, size: 20),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(alert.title, style: AppText.bodyStrong(color: AppColors.danger)),
                    Text(time, style: AppText.caption()),
                  ],
                ),
              ),
              if (!alert.isRead)
                Container(
                  width: 10, height: 10,
                  decoration: const BoxDecoration(color: AppColors.danger, shape: BoxShape.circle),
                ),
            ],
          ),
          const SizedBox(height: 10),
          Text(alert.body, style: AppText.body()),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: _ActionBtn(
                  label: 'Call patient',
                  icon: Icons.phone_rounded,
                  color: AppColors.success,
                  onTap: onCall,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _ActionBtn(
                  label: alert.isRead ? 'Acknowledged' : 'Acknowledge',
                  icon: Icons.check_circle_outline_rounded,
                  color: alert.isRead ? AppColors.inkMuted : AppColors.brandTeal,
                  onTap: alert.isRead ? () {} : onAcknowledge,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ActionBtn extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _ActionBtn({required this.label, required this.icon, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 40,
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(Radii.sm),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 16, color: color),
            const SizedBox(width: 6),
            Text(label, style: AppText.caption(color: color).copyWith(fontWeight: FontWeight.w700)),
          ],
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  All-clear card
// ════════════════════════════════════════════════════════════════════
class _AllClearCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
      decoration: BoxDecoration(
        color: AppColors.success2,
        borderRadius: BorderRadius.circular(Radii.md),
        border: Border.all(color: AppColors.brandLeaf.withOpacity(0.4)),
      ),
      child: Row(
        children: [
          const Icon(Icons.check_circle_rounded, color: AppColors.success, size: 28),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('All clear', style: AppText.bodyStrong(color: AppColors.success)),
                Text('No SOS alerts from your patient.', style: AppText.caption()),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  Patient overview card
// ════════════════════════════════════════════════════════════════════
class _PatientCard extends StatelessWidget {
  final RoverProvider rover;
  final String patientName;
  final VoidCallback onCall;

  const _PatientCard({required this.rover, required this.patientName, required this.onCall});

  @override
  Widget build(BuildContext context) {
    final battColor = AppColors.batteryColor(rover.batteryLevel);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(Radii.md),
        border: Border.all(color: Theme.of(context).dividerColor),
        boxShadow: Elevations.e1,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 44, height: 44,
                decoration: BoxDecoration(
                  color: AppColors.brandAquaSoft,
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.accessibility_new_rounded, color: AppColors.brandTeal, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(patientName, style: AppText.bodyStrong()),
                    Row(
                      children: [
                        Container(
                          width: 8, height: 8,
                          decoration: BoxDecoration(
                            color: rover.isConnected ? AppColors.success : AppColors.inkLight,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 4),
                        Text(
                          rover.isConnected ? 'Robot online' : 'Robot offline',
                          style: AppText.caption(color: rover.isConnected ? AppColors.success : AppColors.inkMuted),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              GestureDetector(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => const LiveFeedScreen()),
                  );
                },
                child: Container(
                  width: 36, height: 36,
                  decoration: BoxDecoration(
                    color: AppColors.brandTeal.withOpacity(0.15),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.videocam_rounded, color: AppColors.brandTeal, size: 18),
                ),
              ),
              const SizedBox(width: 8),
              GestureDetector(
                onTap: onCall,
                child: Container(
                  width: 36, height: 36,
                  decoration: BoxDecoration(
                    color: AppColors.success2,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.phone_rounded, color: AppColors.success, size: 18),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          const Divider(height: 1, color: AppColors.line),
          const SizedBox(height: 12),
          Row(
            children: [
              _StatChip(
                label: 'Heart rate',
                value: '${rover.heartRate} bpm',
                icon: Icons.favorite_rounded,
                color: AppColors.danger,
              ),
              const SizedBox(width: 10),
              _StatChip(
                label: 'Robot battery',
                value: '${rover.batteryLevel}%',
                icon: Icons.battery_full_rounded,
                color: battColor,
              ),
              const SizedBox(width: 10),
              _StatChip(
                label: 'Location',
                value: rover.roverLocation,
                icon: Icons.location_on_rounded,
                color: AppColors.brandTeal,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatChip extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;

  const _StatChip({required this.label, required this.value, required this.icon, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(Radii.sm),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 14, color: color),
            const SizedBox(height: 4),
            Text(value, style: AppText.caption(color: AppColors.inkNavy).copyWith(fontWeight: FontWeight.w700)),
            Text(label, style: AppText.caption(color: AppColors.inkMuted).copyWith(fontSize: 10)),
          ],
        ),
      ),
    );
  }
}

class _SectionHead extends StatelessWidget {
  final String title;
  final int count;
  final Color countColor;

  const _SectionHead({required this.title, required this.count, required this.countColor});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(title.toUpperCase(), style: AppText.eyebrow()),
        const SizedBox(width: 8),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
          decoration: BoxDecoration(
            color: countColor.withOpacity(0.12),
            borderRadius: BorderRadius.circular(Radii.pill),
          ),
          child: Text('$count', style: AppText.caption(color: countColor).copyWith(fontWeight: FontWeight.w800)),
        ),
      ],
    );
  }
}
