import 'package:flutter/material.dart';

import 'package:provider/provider.dart';

import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../widgets/nova_logo.dart';
import '../providers/auth_provider.dart';
import 'auth/auth_wrapper.dart';

/// Branded splash screen — warm cream canvas, breathing NovaCare logo,
/// transitions into the main tabbed shell after 2.5s.
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late final AnimationController _intro = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 900),
  )..forward();

  late final AnimationController _breathe = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 3600),
  )..repeat(reverse: true);

  @override
  void initState() {
    super.initState();
    
    // Wait for minimum splash time AND auth load
    Future.wait([
      Future.delayed(const Duration(milliseconds: 2500)),
      context.read<AuthProvider>().load(),
    ]).then((_) => _goHome());
  }

  void _goHome() {
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      PageRouteBuilder(
        transitionDuration: const Duration(milliseconds: 500),
        pageBuilder: (_, a, __) => FadeTransition(
          opacity: a,
          child: const AuthWrapper(),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _intro.dispose();
    _breathe.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: SafeArea(
        child: AnimatedBuilder(
          animation: Listenable.merge([_intro, _breathe]),
          builder: (context, _) {
            final introT = Curves.easeOutCubic.transform(_intro.value);
            final breatheT = Curves.easeInOut.transform(_breathe.value);

            return Column(
              children: [
                const Spacer(flex: 3),
                Opacity(
                  opacity: introT,
                  child: Transform.translate(
                    offset: Offset(0, 16 * (1 - introT)),
                    child: Transform.scale(
                      scale: 0.96 + 0.04 * breatheT,
                      child: Container(
                        width: 168,
                        height: 168,
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.surface,
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: AppColors.line,
                            width: 2,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: AppColors.accent
                                  .withOpacity(0.18 + 0.12 * breatheT),
                              blurRadius: 40 + 20 * breatheT,
                              spreadRadius: 4 + 4 * breatheT,
                            ),
                          ],
                        ),
                        padding: const EdgeInsets.all(18),
                        child: const NovaLogo(size: 130),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 28),
                Opacity(
                  opacity: introT,
                  child: Column(
                    children: [
                      RichText(
                        text: TextSpan(
                          style: AppText.display1().copyWith(fontSize: 38),
                          children: const [
                            TextSpan(
                              text: 'Nova',
                              style: TextStyle(color: AppColors.inkTeal),
                            ),
                            TextSpan(
                              text: 'Care',
                              style: TextStyle(color: AppColors.accent),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'ASSISTIVE ROBOTICS',
                        style: AppText.eyebrow().copyWith(fontSize: 11),
                      ),
                    ],
                  ),
                ),
                const Spacer(flex: 4),
                Opacity(
                  opacity: introT,
                  child: const _LoadingDots(),
                ),
                const SizedBox(height: 48),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _LoadingDots extends StatefulWidget {
  const _LoadingDots();

  @override
  State<_LoadingDots> createState() => _LoadingDotsState();
}

class _LoadingDotsState extends State<_LoadingDots>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1400),
  )..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _c,
      builder: (_, __) {
        return Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(3, (i) {
            final phase = (_c.value + i * 0.18) % 1.0;
            final opacity = (0.4 + 0.6 * (1 - (phase * 2 - 1).abs())).clamp(0.0, 1.0);
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: Opacity(
                opacity: opacity,
                child: Container(
                  width: 8,
                  height: 8,
                  decoration: const BoxDecoration(
                    color: AppColors.inkTeal,
                    shape: BoxShape.circle,
                  ),
                ),
              ),
            );
          }),
        );
      },
    );
  }
}
