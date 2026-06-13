import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../widgets/nova_logo.dart';
import '../widgets/nc_primitives.dart';

// TODO(backend): real alerts feed from Firebase RTDB `/alerts/{userId}`.
//   - Stream into an AlertsProvider (StreamProvider/StateNotifier)
//   - Severity drives row color (high/med/low)
//   - Tapping an alert opens detail sheet with map + timestamp + actions
//     (acknowledge, call caregiver, dismiss)
class AlertsScreen extends StatefulWidget {
  const AlertsScreen({super.key});

  @override
  State<AlertsScreen> createState() => _AlertsScreenState();
}

class _AlertsScreenState extends State<AlertsScreen> {
  final List<_Alert> today = [
    _Alert(
      sev: _Sev.high,
      title: 'Heart-rate spike detected',
      meta: '12:42 · Kitchen',
      unread: true,
    ),
  ];
  
  final List<_Alert> earlier = [
    _Alert(sev: _Sev.med, title: 'Fall risk — slow motion', meta: 'Yesterday · 19:08'),
    _Alert(sev: _Sev.low, title: 'Battery below 30%', meta: 'Yesterday · 16:22'),
    _Alert(sev: _Sev.low, title: 'Door left open', meta: '2 days ago · 22:10'),
    _Alert(sev: _Sev.med, title: 'Medication missed', meta: '3 days ago · 13:05'),
  ];

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();

    final isEmpty = today.isEmpty && earlier.isEmpty;
    final unread = today.where((a) => a.unread).length +
        earlier.where((a) => a.unread).length;

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
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
                      onTap: () {
                        setState(() {
                          today.clear();
                          earlier.clear();
                        });
                      },
                      child: const NcChip(
                        label: 'Clear all',
                        style: NcChipStyle.normal,
                      ),
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
                        Text(
                          '$unread unread',
                          style: AppText.body(color: AppColors.inkMuted),
                        ),
                        if (today.isNotEmpty) ...[
                          const NcSectionHead(title: 'Today'),
                          NcGroup(
                            children: [for (final a in today) _AlertRow(alert: a)],
                          ),
                        ],
                        if (earlier.isNotEmpty) ...[
                          const NcSectionHead(title: 'Earlier'),
                          NcGroup(
                            children: [
                              for (final a in earlier) _AlertRow(alert: a)
                            ],
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
}

class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.notifications_off_rounded,
            size: 48,
            color: AppColors.inkMuted,
          ),
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

// ─── Internal types ─────────────────────────────────────────────────
enum _Sev { high, med, low }

class _Alert {
  final _Sev sev;
  final String title;
  final String meta;
  final bool unread;
  _Alert({
    required this.sev,
    required this.title,
    required this.meta,
    this.unread = false,
  });
}

class _AlertRow extends StatelessWidget {
  final _Alert alert;
  const _AlertRow({required this.alert});

  @override
  Widget build(BuildContext context) {
    final (iconBg, iconColor, icon) = switch (alert.sev) {
      _Sev.high => (AppColors.danger2, AppColors.danger, Icons.priority_high_rounded),
      _Sev.med => (AppColors.accent3, const Color(0xFF8A6913), Icons.warning_amber_rounded),
      _Sev.low => (AppColors.info2, AppColors.info, Icons.info_outline_rounded),
    };

    return Container(
      color: alert.unread ? AppColors.accent3 : null,
      child: NcRow(
        icon: Icon(icon, color: iconColor),
        iconBg: iconBg,
        title: alert.title,
        subtitle: alert.meta,
        trailing: alert.unread
            ? Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  color: AppColors.accent,
                  shape: BoxShape.circle,
                ),
              )
            : const Icon(
                Icons.chevron_right_rounded,
                color: AppColors.inkMuted,
              ),
        onTap: () {
          // TODO(feature): alert detail sheet (map, timestamp, ack/dismiss).
        },
      ),
    );
  }
}
