import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

// ════════════════════════════════════════════════════════════════════
//  Small reusable primitives — see SKILL §3.
//  Grouped in one file for discoverability. Larger components
//  (tile, btn_card, dpad, feed, bottom_nav) live in their own files.
// ════════════════════════════════════════════════════════════════════

// ─── 3.3 StatusPill / BatteryPill ──────────────────────────────────

enum NcConnectionStatus { online, offline, weak }

/// Pill-shaped status indicator with an optional pulsing dot.
class NcStatusPill extends StatelessWidget {
  final NcConnectionStatus status;
  final String label;
  final bool dark;
  const NcStatusPill({
    super.key,
    required this.status,
    required this.label,
    this.dark = false,
  });

  @override
  Widget build(BuildContext context) {
    final dotColor = switch (status) {
      NcConnectionStatus.online => AppColors.success,
      NcConnectionStatus.offline => AppColors.danger,
      NcConnectionStatus.weak => AppColors.accent,
    };

    return Container(
      height: 28,
      padding: const EdgeInsets.symmetric(horizontal: 10),
      decoration: BoxDecoration(
        color: dark ? AppColors.roverDarkCard : Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(Radii.pill),
        border: Border.all(
          color: dark ? AppColors.roverDarkBorder : Theme.of(context).dividerColor,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _PulseDot(color: dotColor),
          const SizedBox(width: 6),
          Text(
            label,
            style: AppText.caption(
              color: dark ? AppColors.roverDarkText : AppColors.inkNavy,
            ).copyWith(fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

class _PulseDot extends StatefulWidget {
  final Color color;
  const _PulseDot({required this.color});

  @override
  State<_PulseDot> createState() => _PulseDotState();
}

class _PulseDotState extends State<_PulseDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(seconds: 2))
        ..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 14,
      height: 14,
      child: Stack(
        alignment: Alignment.center,
        children: [
          AnimatedBuilder(
            animation: _c,
            builder: (_, __) {
              final t = _c.value;
              return Opacity(
                opacity: (0.5 - t * 0.5).clamp(0.0, 0.5),
                child: Transform.scale(
                  scale: 1 + t * 1.4,
                  child: Container(
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: widget.color.withOpacity(0.45),
                    ),
                  ),
                ),
              );
            },
          ),
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: widget.color,
            ),
          ),
        ],
      ),
    );
  }
}

/// Battery pill (icon + percent text). Same chrome as [NcStatusPill].
class NcBatteryPill extends StatelessWidget {
  final int level;
  final bool dark;
  const NcBatteryPill({super.key, required this.level, this.dark = false});

  @override
  Widget build(BuildContext context) {
    final color = AppColors.batteryColor(level);
    return Container(
      height: 28,
      padding: const EdgeInsets.symmetric(horizontal: 10),
      decoration: BoxDecoration(
        color: dark ? AppColors.roverDarkCard : Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(Radii.pill),
        border: Border.all(
          color: dark ? AppColors.roverDarkBorder : AppColors.line,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(_iconFor(level), size: 14, color: color),
          const SizedBox(width: 4),
          Text(
            '$level%',
            style: AppText.caption(
              color: dark ? AppColors.roverDarkText : AppColors.inkNavy,
            ).copyWith(fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }

  IconData _iconFor(int n) {
    if (n >= 90) return Icons.battery_full_rounded;
    if (n >= 60) return Icons.battery_5_bar_rounded;
    if (n >= 35) return Icons.battery_3_bar_rounded;
    if (n >= 15) return Icons.battery_2_bar_rounded;
    return Icons.battery_alert_rounded;
  }
}

// ─── 3.3 NcAppBar ──────────────────────────────────────────────────

/// Custom app bar — not a Material AppBar, lays out
/// `[leading + title] [spacer] [statusPill] [batteryPill] [trailing]`.
class NcAppBar extends StatelessWidget implements PreferredSizeWidget {
  final Widget? leading;
  final Widget? title;
  final NcConnectionStatus? status;
  final String? statusLabel;
  final int? battery;
  final List<Widget>? trailing;
  final bool dark;

  const NcAppBar({
    super.key,
    this.leading,
    this.title,
    this.status,
    this.statusLabel,
    this.battery,
    this.trailing,
    this.dark = false,
  });

  @override
  Size get preferredSize => const Size.fromHeight(64);

  @override
  Widget build(BuildContext context) {
    final padding = MediaQuery.of(context).padding.top;
    return Container(
      color: dark ? AppColors.roverDarkBg : Theme.of(context).scaffoldBackgroundColor,
      padding: EdgeInsetsDirectional.only(
        start: 20,
        end: 20,
        top: padding + 8,
        bottom: 8,
      ),
      child: Row(
        children: [
          if (leading != null) ...[
            leading!,
            const SizedBox(width: 12),
          ],
          if (title != null) Flexible(child: title!),
          const Spacer(),
          if (status != null) ...[
            NcStatusPill(
              status: status!,
              label: statusLabel ?? _statusText(status!),
              dark: dark,
            ),
            const SizedBox(width: 8),
          ],
          if (battery != null) ...[
            NcBatteryPill(level: battery!, dark: dark),
            const SizedBox(width: 8),
          ],
          if (trailing != null) ...trailing!,
        ],
      ),
    );
  }

  String _statusText(NcConnectionStatus s) => switch (s) {
        NcConnectionStatus.online => 'Online',
        NcConnectionStatus.offline => 'Offline',
        NcConnectionStatus.weak => 'Weak',
      };
}

// ─── 3.5 NcSwitch ──────────────────────────────────────────────────

/// 48Ã—28 pill switch with spring-eased thumb and haptic on toggle.
/// RTL-aware: thumb travels in the layout direction.
class NcSwitch extends StatelessWidget {
  final bool value;
  final ValueChanged<bool> onChanged;
  final String? semanticLabel;
  final bool dark;

  const NcSwitch({
    super.key,
    required this.value,
    required this.onChanged,
    this.semanticLabel,
    this.dark = false,
  });

  @override
  Widget build(BuildContext context) {
    final trackColor = value
        ? AppColors.brandTeal
        : (dark ? const Color(0x29FFFFFF) : AppColors.inkLight);

    return Semantics(
      label: semanticLabel,
      toggled: value,
      button: true,
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () {
          HapticFeedback.selectionClick();
          onChanged(!value);
        },
        child: ConstrainedBox(
          constraints: const BoxConstraints(minWidth: 72, minHeight: 56),
          child: Center(
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 280),
              curve: const Cubic(0.4, 1.5, 0.5, 1),
              width: 48,
              height: 28,
              padding: const EdgeInsets.all(2),
              decoration: BoxDecoration(
                color: trackColor,
                borderRadius: BorderRadius.circular(Radii.pill),
              ),
              child: AnimatedAlign(
                duration: const Duration(milliseconds: 280),
                curve: const Cubic(0.4, 1.5, 0.5, 1),
                alignment: value
                    ? AlignmentDirectional.centerEnd
                    : AlignmentDirectional.centerStart,
                child: Container(
                  width: 24,
                  height: 24,
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Color(0x1A000000),
                        blurRadius: 2,
                        offset: Offset(0, 1),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─── 3.9 NcSectionHead ─────────────────────────────────────────────

class NcSectionHead extends StatelessWidget {
  final String title;
  final Widget? action;
  const NcSectionHead({super.key, required this.title, this.action});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsetsDirectional.only(top: 24, bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Expanded(
            child: Text(
              title.toUpperCase(),
              style: AppText.sectionHead(),
            ),
          ),
          if (action != null) action!,
        ],
      ),
    );
  }
}

// ─── 3.10 NcGroup ──────────────────────────────────────────────────

/// White card with line border, rLG corners, children separated by 1dp dividers.
class NcGroup extends StatelessWidget {
  final List<Widget> children;
  final EdgeInsetsGeometry? padding;
  const NcGroup({super.key, required this.children, this.padding});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(Radii.lg),
        border: Border.all(color: Theme.of(context).dividerColor),
      ),
      padding: padding,
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          for (int i = 0; i < children.length; i++) ...[
            if (i > 0) const Divider(height: 1, color: AppColors.line),
            children[i],
          ],
        ],
      ),
    );
  }
}

// ─── 3.8 NcRow ─────────────────────────────────────────────────────

class NcRow extends StatelessWidget {
  final Widget icon;
  final String title;
  final String? subtitle;
  final Widget? trailing;
  final VoidCallback? onTap;
  final Color? iconBg;

  const NcRow({
    super.key,
    required this.icon,
    required this.title,
    this.subtitle,
    this.trailing,
    this.onTap,
    this.iconBg,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: ConstrainedBox(
        constraints: const BoxConstraints(minHeight: 72),
        child: Padding(
          padding: const EdgeInsetsDirectional.symmetric(
            horizontal: 16,
            vertical: 14,
          ),
          child: Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: iconBg ?? Theme.of(context).colorScheme.primaryContainer,
                  borderRadius: BorderRadius.circular(12),
                ),
                alignment: Alignment.center,
                child: IconTheme(
                  data: IconThemeData(color: Theme.of(context).colorScheme.onPrimaryContainer, size: 20),
                  child: icon,
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(title, style: AppText.display3()),
                    if (subtitle != null) ...[
                      const SizedBox(height: 2),
                      Text(
                        subtitle!,
                        style: AppText.caption(),
                      ),
                    ],
                  ],
                ),
              ),
              if (trailing != null) ...[
                const SizedBox(width: 8),
                trailing!,
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ─── 3.11 NcChip ───────────────────────────────────────────────────

enum NcChipStyle { normal, success, warn, danger, info, beta }

class NcChip extends StatelessWidget {
  final String label;
  final NcChipStyle style;
  const NcChip({
    super.key,
    required this.label,
    this.style = NcChipStyle.normal,
  });

  @override
  Widget build(BuildContext context) {
    final (bg, fg) = switch (style) {
      NcChipStyle.normal => (Theme.of(context).colorScheme.surfaceContainerHighest, Theme.of(context).colorScheme.onSurface),
      NcChipStyle.success => (AppColors.success2, AppColors.success),
      NcChipStyle.warn => (AppColors.accent3, const Color(0xFF8A6913)),
      NcChipStyle.danger => (AppColors.danger2, AppColors.danger),
      NcChipStyle.info => (AppColors.info2, AppColors.info),
      NcChipStyle.beta => (Theme.of(context).colorScheme.surfaceContainerHighest, Theme.of(context).colorScheme.onSurfaceVariant),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 3),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(Radii.pill),
      ),
      child: Text(
        label,
        style: AppText.caption(color: fg)
            .copyWith(fontWeight: FontWeight.w700, fontSize: 11),
      ),
    );
  }
}

// ─── 3.14 NcSeg (segmented control) ────────────────────────────────

class NcSeg extends StatelessWidget {
  final List<String> labels;
  final int selected;
  final ValueChanged<int> onSelect;
  const NcSeg({
    super.key,
    required this.labels,
    required this.selected,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(Radii.pill),
        border: Border.all(color: Theme.of(context).dividerColor),
      ),
      child: Row(
        children: [
          for (int i = 0; i < labels.length; i++)
            Expanded(
              child: GestureDetector(
                onTap: () => onSelect(i),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  height: 40,
                  decoration: BoxDecoration(
                    color: i == selected ? Theme.of(context).colorScheme.surface : Colors.transparent,
                    borderRadius: BorderRadius.circular(Radii.pill),
                    boxShadow: i == selected ? Elevations.e1 : null,
                  ),
                  alignment: Alignment.center,
                  child: Text(
                    labels[i],
                    style: AppText.bodyStrong(
                      color: i == selected
                          ? AppColors.inkNavy
                          : AppColors.inkMuted,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
