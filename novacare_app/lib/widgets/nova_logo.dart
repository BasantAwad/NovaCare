import 'package:flutter/material.dart';

import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// Rounded-corner image of the NovaCare NC logo.
///
/// Looks for `assets/images/logo.jpeg`. If the asset cannot be found we draw a
/// stylized "NC" fallback so the UI still renders during early development.
///
/// Corner radius is computed as `size * 0.28` per SKILL §3.1.
class NovaLogo extends StatelessWidget {
  final double size;
  const NovaLogo({super.key, this.size = 36});

  @override
  Widget build(BuildContext context) {
    final radius = size * 0.28;

    return ClipRRect(
      borderRadius: BorderRadius.circular(radius),
      child: Image.asset(
        'assets/images/NovaLogo.jpeg',
        width: size,
        height: size,
        fit: BoxFit.cover,
        errorBuilder: (_, __, ___) => _LogoFallback(size: size, radius: radius),
      ),
    );
  }
}

class _LogoFallback extends StatelessWidget {
  final double size;
  final double radius;
  const _LogoFallback({required this.size, required this.radius});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(radius),
        border: Border.all(color: AppColors.line),
      ),
      alignment: Alignment.center,
      child: RichText(
        text: TextSpan(
          style: AppText.display3(color: AppColors.inkTeal)
              .copyWith(fontSize: size * 0.42),
          children: const [
            TextSpan(text: 'N'),
            TextSpan(text: 'C', style: TextStyle(color: AppColors.accent)),
          ],
        ),
      ),
    );
  }
}

/// Logo + "NovaCare" wordmark lockup (SKILL §3.2).
/// Compact by default — no subtitle. Pass [showSubtitle: true] only on
/// branded surfaces like the splash screen.
class NovaWordmark extends StatelessWidget {
  final bool showSubtitle;
  final double logoSize;
  final double titleSize;
  const NovaWordmark({
    super.key,
    this.showSubtitle = false,
    this.logoSize = 32,
    this.titleSize = 22,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        NovaLogo(size: logoSize),
        const SizedBox(width: 8),
        Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            RichText(
              text: TextSpan(
                style: AppText.display2().copyWith(
                  fontSize: titleSize,
                  fontWeight: FontWeight.w800,
                  height: 1.0,
                ),
                children: const [
                  TextSpan(text: 'Nova', style: TextStyle(color: AppColors.inkTeal)),
                  TextSpan(text: 'Care', style: TextStyle(color: AppColors.accent)),
                ],
              ),
            ),
            if (showSubtitle)
              Padding(
                padding: const EdgeInsets.only(top: 2),
                child: Text(
                  'ASSISTIVE ROBOTICS',
                  style: AppText.eyebrow().copyWith(fontSize: 9.5),
                ),
              ),
          ],
        ),
      ],
    );
  }
}
