import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// Compact rover status strip (online / offline indicator).
class StatusBarWidget extends StatelessWidget {
  const StatusBarWidget({super.key});

  @override
  Widget build(BuildContext context) {
    final rover  = context.watch<RoverProvider>();
    final online = rover.isRoverOnline;

    return Container(
      padding: const EdgeInsetsDirectional.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: online ? AppColors.success2 : AppColors.danger2,
        borderRadius: BorderRadius.circular(Radii.md),
        border: Border.all(
          color: online
              ? AppColors.success.withValues(alpha: 0.3)
              : AppColors.danger.withValues(alpha: 0.3),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 10, height: 10,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: online ? AppColors.success : AppColors.danger,
              boxShadow: online
                  ? [BoxShadow(color: AppColors.success.withValues(alpha: 0.4), blurRadius: 8, spreadRadius: 2)]
                  : null,
            ),
          ),
          const SizedBox(width: 12),
          Text(
            online ? 'Robot online' : 'Robot offline',
            style: AppText.bodyStrong(color: online ? AppColors.success : AppColors.danger),
          ),
        ],
      ),
    );
  }
}
