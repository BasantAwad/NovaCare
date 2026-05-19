import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';

import '../providers/alert_provider.dart';
import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../widgets/nova_logo.dart';
import '../widgets/nc_primitives.dart';

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final alertProvider = context.watch<AlertProvider>();
    final rover         = context.watch<RoverProvider>();
    final alerts        = alertProvider.alerts;
    final isEmpty       = alerts.isEmpty;
    final unread        = alertProvider.unreadCount;

    return Scaffold(
      backgroundColor: AppColors.canvas,
      body: Column(
        children: [
          NcAppBar(
            leading: const NovaLogo(),
            title: Text('Alerts', style: AppText.appBarTitle()),
            battery: rover.batteryLevel,
            trailing: isEmpty
                ? null
                : [
                    GestureDetector(
                      onTap: () => alertProvider.markAllAsRead(),
                      child: const NcChip(label: 'Read all', style: NcChipStyle.normal),
                    ),
                    const SizedBox(width: 8),
                    GestureDetector(
                      onTap: () => _showClearAllDialog(context, alertProvider),
                      child: const NcChip(label: 'Clear all', style: NcChipStyle.normal),
                    ),
                  ],
          ),
          Expanded(
            child: isEmpty
                ? _EmptyState()
                : SingleChildScrollView(
                    padding: const EdgeInsetsDirectional.fromSTEB(20, 8, 20, 40),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Activity', style: AppText.display1()),
                        const SizedBox(height: 4),
                        Text('$unread unread', style: AppText.body(color: AppColors.inkMuted)),
                        const SizedBox(height: 16),
                        NcGroup(
                          children: alerts
                              .map((a) => _AlertRow(
                                    alert: a,
                                    onTap: () => alertProvider.markAsRead(a.id),
                                    onDismiss: () => alertProvider.dismissAlert(a.id),
                                  ))
                              .toList(),
                        ),
                      ],
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  void _showClearAllDialog(BuildContext context, AlertProvider provider) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Radii.lg)),
        backgroundColor: AppColors.paper,
        title: Text('Clear all alerts?', style: AppText.display3()),
        content: Text('This cannot be undone.', style: AppText.body()),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('CANCEL')),
          TextButton(
            onPressed: () { provider.clearAll(); Navigator.pop(ctx); },
            child: const Text('CLEAR ALL', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.notifications_off_rounded, size: 48, color: AppColors.inkMuted),
          const SizedBox(height: 12),
          Text('All clear', style: AppText.display2()),
          const SizedBox(height: 4),
          Text(
            'No alerts right now — you and SERBOT are fine.',
            style: AppText.body(color: AppColors.inkMuted),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

class _AlertRow extends StatelessWidget {
  final AlertModel alert;
  final VoidCallback onTap;
  final VoidCallback onDismiss;

  const _AlertRow({required this.alert, required this.onTap, required this.onDismiss});

  @override
  Widget build(BuildContext context) {
    final (iconBg, iconColor) = switch (alert.severity) {
      AlertSeverity.error   => (AppColors.danger2,  AppColors.danger),
      AlertSeverity.warning => (AppColors.accent3,  const Color(0xFF8A6913)),
      AlertSeverity.info    => (AppColors.info2,    AppColors.info),
    };

    final timeStr = DateFormat('h:mm a').format(alert.timestamp);
    final isToday = DateTime.now().day == alert.timestamp.day;
    final dateStr = isToday ? 'Today' : DateFormat('MMM d').format(alert.timestamp);

    return Dismissible(
      key: Key(alert.id),
      direction: DismissDirection.endToStart,
      onDismissed: (_) => onDismiss(),
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 20),
        decoration: BoxDecoration(color: Colors.redAccent, borderRadius: BorderRadius.circular(Radii.md)),
        child: const Icon(Icons.delete_outline_rounded, color: Colors.white),
      ),
      child: Container(
        color: alert.isRead ? null : AppColors.accent3,
        child: NcRow(
          icon: Icon(alert.icon, color: iconColor),
          iconBg: iconBg,
          title: alert.title,
          subtitle: '$dateStr, $timeStr  •  ${alert.body}',
          trailing: alert.isRead
              ? const Icon(Icons.chevron_right_rounded, color: AppColors.inkMuted)
              : Container(
                  width: 8, height: 8,
                  decoration: const BoxDecoration(color: AppColors.accent, shape: BoxShape.circle),
                ),
          onTap: onTap,
        ),
      ),
    );
  }
}
