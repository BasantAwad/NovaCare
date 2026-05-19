import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../l10n/app_localizations.dart';

/// Compact rover status strip (online state + current mode).
///
/// Refactored to NovaCare design tokens. Logic preserved from the previous
/// implementation — still driven by [RoverProvider.isRoverOnline] /
/// [currentMode].
class StatusBarWidget extends StatelessWidget {
  const StatusBarWidget({super.key});

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();
    final l10n = AppLocalizations.of(context);

    final online = rover.isRoverOnline;
    final modeLabel = _modeLabel(rover.currentMode, l10n);
    final modeColor = _modeColor(rover.currentMode);
    final showMode = rover.currentMode != RoverMode.idle;

    return Container(
      padding: const EdgeInsetsDirectional.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: online ? AppColors.success2 : AppColors.danger2,
        borderRadius: BorderRadius.circular(Radii.md),
        border: Border.all(
          color: online
              ? AppColors.success.withOpacity(0.3)
              : AppColors.danger.withOpacity(0.3),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: online ? AppColors.success : AppColors.danger,
              boxShadow: online
                  ? [
                      BoxShadow(
                        color: AppColors.success.withOpacity(0.4),
                        blurRadius: 8,
                        spreadRadius: 2,
                      ),
                    ]
                  : null,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${l10n.translate('rover_status')}: '
                  '${online ? l10n.translate('online') : l10n.translate('offline')}',
                  style: AppText.bodyStrong(),
                ),
                if (showMode) ...[
                  const SizedBox(height: 2),
                  Text(modeLabel, style: AppText.caption(color: modeColor)),
                ],
              ],
            ),
          ),
          if (showMode)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: modeColor.withOpacity(0.15),
                borderRadius: BorderRadius.circular(Radii.pill),
                border: Border.all(color: modeColor.withOpacity(0.3)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(_modeIcon(rover.currentMode), size: 14, color: modeColor),
                  const SizedBox(width: 4),
                  Text(
                    'Active',
                    style: AppText.caption(color: modeColor)
                        .copyWith(fontWeight: FontWeight.w700, fontSize: 11),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  String _modeLabel(RoverMode m, AppLocalizations l10n) => switch (m) {
        RoverMode.idle => l10n.translate('idle'),
        RoverMode.followingUser => l10n.translate('following'),
        RoverMode.navigatingHome => l10n.translate('navigating_home'),
        RoverMode.deliveringMedicine => l10n.translate('delivering'),
        RoverMode.emergency => l10n.translate('emergency_mode'),
      };

  Color _modeColor(RoverMode m) => switch (m) {
        RoverMode.idle => AppColors.inkMuted,
        RoverMode.followingUser => AppColors.brandTeal,
        RoverMode.navigatingHome => AppColors.brandLeaf,
        RoverMode.deliveringMedicine => AppColors.accent,
        RoverMode.emergency => AppColors.danger,
      };

  IconData _modeIcon(RoverMode m) => switch (m) {
        RoverMode.idle => Icons.pause_circle_rounded,
        RoverMode.followingUser => Icons.directions_walk_rounded,
        RoverMode.navigatingHome => Icons.home_rounded,
        RoverMode.deliveringMedicine => Icons.medication_rounded,
        RoverMode.emergency => Icons.emergency_rounded,
      };
}
