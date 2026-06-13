import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// 5-tab nav bar (Home Â· Reminders Â· Companion Â· Alerts Â· Settings).
/// Companion tab is a raised yellow bubble per SKILL Â§3.4.
enum NcTab { home, reminders, companion, alerts, settings }

class NcBottomNav extends StatelessWidget {
  final NcTab active;
  final ValueChanged<NcTab> onChange;

  const NcBottomNav({
    super.key,
    required this.active,
    required this.onChange,
  });

  static const _tabs = [
    (NcTab.home, Icons.home_rounded, 'Home'),
    (NcTab.reminders, Icons.notifications_active_rounded, 'Reminders'),
    (NcTab.companion, Icons.chat_bubble_rounded, 'Companion'),
    (NcTab.alerts, Icons.error_rounded, 'Alerts'),
    (NcTab.settings, Icons.tune_rounded, 'Settings'),
  ];

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).padding.bottom;

    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: const Border(top: BorderSide(color: AppColors.line)),
        boxShadow: Elevations.e2,
      ),
      padding: EdgeInsets.only(top: 10, bottom: bottomInset > 0 ? bottomInset : 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          for (final t in _tabs)
            Expanded(
              child: t.$1 == NcTab.companion
                  ? _CompanionTab(
                      active: active == t.$1,
                      onTap: () => _handleTap(t.$1),
                    )
                  : _Tab(
                      icon: t.$2,
                      label: t.$3,
                      active: active == t.$1,
                      onTap: () => _handleTap(t.$1),
                    ),
            ),
        ],
      ),
    );
  }

  void _handleTap(NcTab t) {
    HapticFeedback.selectionClick();
    onChange(t);
  }
}

class _Tab extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool active;
  final VoidCallback onTap;

  const _Tab({
    required this.icon,
    required this.label,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = active ? AppColors.inkNavy : AppColors.inkMuted;
    return InkResponse(
      onTap: onTap,
      highlightShape: BoxShape.rectangle,
      child: ConstrainedBox(
        constraints: const BoxConstraints(minHeight: 72),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 24, color: color),
            const SizedBox(height: 4),
            Text(
              label,
              style: AppText.caption(color: color).copyWith(
                fontSize: 10.5,
                fontWeight: active ? FontWeight.w700 : FontWeight.w500,
              ),
            ),
            const SizedBox(height: 4),
            AnimatedContainer(
              duration: const Duration(milliseconds: 220),
              width: active ? 18 : 0,
              height: 3,
              decoration: BoxDecoration(
                color: AppColors.accent,
                borderRadius: BorderRadius.circular(Radii.pill),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CompanionTab extends StatelessWidget {
  final bool active;
  final VoidCallback onTap;
  const _CompanionTab({required this.active, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkResponse(
      onTap: onTap,
      highlightShape: BoxShape.rectangle,
      child: ConstrainedBox(
        constraints: const BoxConstraints(minHeight: 72),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Transform.translate(
              offset: const Offset(0, -18),
              child: Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  color: AppColors.accent,
                  shape: BoxShape.circle,
                  border: Border.all(color: Theme.of(context).colorScheme.surface, width: 4),
                  boxShadow: const [
                    BoxShadow(
                      color: Color(0x33F0B82B),
                      blurRadius: 16,
                      offset: Offset(0, 6),
                    ),
                  ],
                ),
                child: const Icon(
                  Icons.chat_bubble_rounded,
                  color: AppColors.inkNavy,
                  size: 28,
                ),
              ),
            ),
            Transform.translate(
              offset: const Offset(0, -12),
              child: Text(
                'Companion',
                style: AppText.caption(
                  color: active ? AppColors.inkNavy : AppColors.inkMuted,
                ).copyWith(
                  fontSize: 10.5,
                  fontWeight: active ? FontWeight.w700 : FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
