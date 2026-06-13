import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// NcBtnCard variants — see SKILL §3.7.
enum NcBtnCardVariant { normal, sos, brand, accent }

/// Large action card: [icon] [title + subtitle] [chevron / trailing].
/// minHeight 72, rLG, padding 16. Press = scale(0.99) 100ms.
class NcBtnCard extends StatefulWidget {
  final Widget icon;
  final String title;
  final String subtitle;
  final VoidCallback? onTap;
  final NcBtnCardVariant variant;
  final bool active;
  final bool loading;
  final Widget? trailing;

  const NcBtnCard({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
    this.variant = NcBtnCardVariant.normal,
    this.active = false,
    this.loading = false,
    this.trailing,
  });

  @override
  State<NcBtnCard> createState() => _NcBtnCardState();
}

class _NcBtnCardState extends State<NcBtnCard>
    with SingleTickerProviderStateMixin {
  bool _pressed = false;
  late final AnimationController _sosRing = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2400),
  );

  @override
  void initState() {
    super.initState();
    if (widget.variant == NcBtnCardVariant.sos) _sosRing.repeat();
  }

  @override
  void didUpdateWidget(covariant NcBtnCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.variant == NcBtnCardVariant.sos && !_sosRing.isAnimating) {
      _sosRing.repeat();
    } else if (widget.variant != NcBtnCardVariant.sos && _sosRing.isAnimating) {
      _sosRing.stop();
    }
  }

  @override
  void dispose() {
    _sosRing.dispose();
    super.dispose();
  }

  ({Color bg, Color border, Color iconBg, Color iconColor, Color titleColor})
      _palette() {
    switch (widget.variant) {
      case NcBtnCardVariant.normal:
        return (
          bg: Theme.of(context).colorScheme.surface,
          border: Theme.of(context).dividerColor,
          iconBg: Theme.of(context).colorScheme.primaryContainer,
          iconColor: Theme.of(context).colorScheme.onPrimaryContainer,
          titleColor: Theme.of(context).colorScheme.onSurface,
        );
      case NcBtnCardVariant.brand:
        return (
          bg: AppColors.brandTeal,
          border: AppColors.brandTeal,
          iconBg: Colors.white.withOpacity(0.2),
          iconColor: Colors.white,
          titleColor: Colors.white,
        );
      case NcBtnCardVariant.accent:
        return (
          bg: AppColors.accent3,
          border: AppColors.accent2,
          iconBg: AppColors.accent,
          iconColor: AppColors.inkNavy,
          titleColor: AppColors.inkNavy,
        );
      case NcBtnCardVariant.sos:
        return (
          bg: AppColors.brandTeal,
          border: AppColors.brandTeal,
          iconBg: AppColors.danger,
          iconColor: Colors.white,
          titleColor: Colors.white,
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    final p = _palette();
    final subtitleColor = (widget.variant == NcBtnCardVariant.sos || widget.variant == NcBtnCardVariant.brand)
        ? Colors.white.withOpacity(0.7)
        : AppColors.inkMuted;

    final card = AnimatedScale(
      scale: _pressed ? 0.99 : 1.0,
      duration: const Duration(milliseconds: 100),
      child: Container(
        constraints: const BoxConstraints(minHeight: 72),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: p.bg,
          borderRadius: BorderRadius.circular(Radii.lg),
          border: Border.all(color: p.border),
          boxShadow: widget.variant == NcBtnCardVariant.sos
              ? null
              : Elevations.e1,
        ),
        child: Row(
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: p.iconBg,
                borderRadius: BorderRadius.circular(14),
              ),
              alignment: Alignment.center,
              child: IconTheme(
                data: IconThemeData(color: p.iconColor, size: 22),
                child: widget.icon,
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    widget.title,
                    style: AppText.display3(color: p.titleColor),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    widget.subtitle,
                    style: AppText.caption(color: subtitleColor),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            if (widget.loading)
              SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(
                  strokeWidth: 2.4,
                  valueColor: AlwaysStoppedAnimation<Color>(p.iconColor),
                ),
              )
            else if (widget.trailing != null)
              widget.trailing!
            else if (widget.active)
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: p.iconColor,
                  borderRadius: BorderRadius.circular(Radii.pill),
                ),
                child: const Text(
                  'ON',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w800,
                    fontSize: 11,
                  ),
                ),
              )
            else
              Icon(
                Icons.chevron_right_rounded,
                color: widget.variant == NcBtnCardVariant.sos
                    ? Colors.white54
                    : AppColors.inkMuted,
              ),
          ],
        ),
      ),
    );

    return Semantics(
      button: true,
      label: '${widget.title}. ${widget.subtitle}',
      child: GestureDetector(
        onTapDown: (_) => setState(() => _pressed = true),
        onTapCancel: () => setState(() => _pressed = false),
        onTapUp: (_) => setState(() => _pressed = false),
        onTap: widget.onTap == null
            ? null
            : () {
                HapticFeedback.selectionClick();
                widget.onTap!();
              },
        child: widget.variant == NcBtnCardVariant.sos
            ? Stack(
                children: [
                  AnimatedBuilder(
                    animation: _sosRing,
                    builder: (_, __) => Container(
                      margin: const EdgeInsets.all(0),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(Radii.lg),
                        boxShadow: [
                          BoxShadow(
                            color: AppColors.danger
                                .withOpacity(0.4 * (1 - _sosRing.value)),
                            blurRadius: 24 * _sosRing.value,
                            spreadRadius: 6 * _sosRing.value,
                          ),
                        ],
                      ),
                      child: card,
                    ),
                  ),
                ],
              )
            : card,
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  Legacy compat shim
//  Maps the old ActionButtonWidget API to NcBtnCard so existing screens
//  still compile. TODO(refactor): migrate call sites to NcBtnCard.
// ════════════════════════════════════════════════════════════════════
class ActionButtonWidget extends StatelessWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final Color color;
  final Color backgroundColor;
  final VoidCallback onPressed;
  final bool isLarge;
  final bool isEmergency;
  final bool isActive;
  final bool isLoading;

  const ActionButtonWidget({
    super.key,
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.color,
    required this.backgroundColor,
    required this.onPressed,
    this.isLarge = false,
    this.isEmergency = false,
    this.isActive = false,
    this.isLoading = false,
  });

  @override
  Widget build(BuildContext context) {
    return NcBtnCard(
      icon: Icon(icon),
      title: label,
      subtitle: subtitle,
      onTap: onPressed,
      active: isActive,
      loading: isLoading,
      variant:
          isEmergency ? NcBtnCardVariant.sos : NcBtnCardVariant.normal,
    );
  }
}
