import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

class SummonScreen extends StatefulWidget {
  const SummonScreen({super.key});

  @override
  State<SummonScreen> createState() => _SummonScreenState();
}

class _SummonScreenState extends State<SummonScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 4),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Text('Summon Robot',
            style: AppText.appBarTitle().copyWith(color: Colors.white)),
        centerTitle: false,
      ),
      body: SafeArea(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Spacer(),
            // Radar Animation
            Center(
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // Outer radar circle
                  Container(
                    width: 250,
                    height: 250,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: Colors.white.withOpacity(0.2),
                        width: 2,
                      ),
                    ),
                  ),
                  // Rotating dot
                  RotationTransition(
                    turns: _controller,
                    child: Container(
                      width: 250,
                      height: 250,
                      alignment: Alignment.centerRight,
                      child: Container(
                        width: 16,
                        height: 16,
                        decoration: const BoxDecoration(
                          color: Colors.white,
                          shape: BoxShape.circle,
                        ),
                      ),
                    ),
                  ),
                  // Center pin icon
                  Container(
                    width: 80,
                    height: 80,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.3),
                          blurRadius: 10,
                          spreadRadius: 2,
                        ),
                      ],
                    ),
                    child: Icon(
                      Icons.person_pin_circle,
                      size: 50,
                      color: Theme.of(context).scaffoldBackgroundColor,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 60),
            Text(
              'Robot is finding\nyou...',
              textAlign: TextAlign.center,
              style: AppText.display1().copyWith(
                color: Colors.white,
                fontSize: 32,
              ),
            ),
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40),
              child: Text(
                'SERBot is navigating to your current location using its LiDAR and tracking system.',
                textAlign: TextAlign.center,
                style: AppText.body().copyWith(color: Colors.white70),
              ),
            ),
            const Spacer(),
            // Cancel button
            Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: TextButton(
                onPressed: () => Navigator.of(context).pop(),
                style: TextButton.styleFrom(
                  backgroundColor: Colors.white.withOpacity(0.1),
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20),
                  ),
                ),
                child: Text(
                  'Cancel Summon',
                  style: AppText.body().copyWith(color: AppColors.danger),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
