import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../l10n/app_localizations.dart';

/// Displays the rover's current status (online/offline, mode) as a compact bar.
class StatusBarWidget extends StatelessWidget {
  const StatusBarWidget({super.key});

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();
    final l10n = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    final modeLabel = _getModeLabel(rover.currentMode, l10n);
    final modeColor = _getModeColor(rover.currentMode);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        gradient: rover.isRoverOnline
            ? LinearGradient(
                begin: Alignment.centerLeft,
                end: Alignment.centerRight,
                colors: isDark
                    ? [const Color(0xFF064E3B), const Color(0xFF0F172A)]
                    : [const Color(0xFFECFDF5), const Color(0xFFF0FDF4)],
              )
            : LinearGradient(
                begin: Alignment.centerLeft,
                end: Alignment.centerRight,
                colors: isDark
                    ? [const Color(0xFF3F0000), const Color(0xFF0F172A)]
                    : [const Color(0xFFFEF2F2), const Color(0xFFFFF1F2)],
              ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: rover.isRoverOnline
              ? AppColors.successGreen.withOpacity(0.3)
              : AppColors.batteryLow.withOpacity(0.3),
          width: 1,
        ),
      ),
      child: Row(
        children: [
          // Status dot
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: rover.isRoverOnline
                  ? AppColors.onlineStatus
                  : AppColors.offlineStatus,
              shape: BoxShape.circle,
              boxShadow: rover.isRoverOnline
                  ? [
                      BoxShadow(
                        color: AppColors.onlineStatus.withOpacity(0.4),
                        blurRadius: 8,
                        spreadRadius: 2,
                      ),
                    ]
                  : null,
            ),
          ),
          const SizedBox(width: 12),

          // Status text
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${l10n.translate('rover_status')}: ${rover.isRoverOnline ? l10n.translate('online') : l10n.translate('offline')}',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: theme.colorScheme.onSurface,
                  ),
                ),
                if (rover.currentMode != RoverMode.idle) ...[
                  const SizedBox(height: 2),
                  Text(
                    modeLabel,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      color: modeColor,
                    ),
                  ),
                ],
              ],
            ),
          ),

          // Mode badge
          if (rover.currentMode != RoverMode.idle)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: modeColor.withOpacity(0.15),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: modeColor.withOpacity(0.3)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    _getModeIcon(rover.currentMode),
                    size: 14,
                    color: modeColor,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    'Active',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: modeColor,
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  String _getModeLabel(RoverMode mode, AppLocalizations l10n) {
    switch (mode) {
      case RoverMode.idle:
        return l10n.translate('idle');
      case RoverMode.followingUser:
        return l10n.translate('following');
      case RoverMode.navigatingHome:
        return l10n.translate('navigating_home');
      case RoverMode.deliveringMedicine:
        return l10n.translate('delivering');
      case RoverMode.emergency:
        return l10n.translate('emergency_mode');
    }
  }

  Color _getModeColor(RoverMode mode) {
    switch (mode) {
      case RoverMode.idle:
        return AppColors.neutralGray400;
      case RoverMode.followingUser:
        return AppColors.followBlue;
      case RoverMode.navigatingHome:
        return AppColors.homeTeal;
      case RoverMode.deliveringMedicine:
        return AppColors.medicationPurple;
      case RoverMode.emergency:
        return AppColors.sosRed;
    }
  }

  IconData _getModeIcon(RoverMode mode) {
    switch (mode) {
      case RoverMode.idle:
        return Icons.pause_circle_rounded;
      case RoverMode.followingUser:
        return Icons.directions_walk_rounded;
      case RoverMode.navigatingHome:
        return Icons.home_rounded;
      case RoverMode.deliveringMedicine:
        return Icons.medication_rounded;
      case RoverMode.emergency:
        return Icons.emergency_rounded;
    }
  }
}
