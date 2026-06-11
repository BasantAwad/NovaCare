import 'package:flutter/material.dart';

/// Semantic color accessors that respect the current theme.
/// Use these throughout the app for consistent coloring.
class AppColors {
  AppColors._();

  // ─── UI Colors ──────────────────────────────────────────────
  static const Color primaryBlue = Color(0xFF2563EB);
  static const Color warningAmber = Color(0xFFF59E0B);
  static const Color successGreen = Color(0xFF22C55E);
  static const Color neutralGray400 = Color(0xFF94A3B8);

  // ─── Status Colors (theme-independent) ──────────────────────────
  static const Color batteryGood = Color(0xFF22C55E);
  static const Color batteryMedium = Color(0xFFF59E0B);
  static const Color batteryLow = Color(0xFFEF4444);

  static const Color heartRateNormal = Color(0xFF22C55E);
  static const Color heartRateElevated = Color(0xFFF59E0B);
  static const Color heartRateCritical = Color(0xFFEF4444);

  static const Color onlineStatus = Color(0xFF22C55E);
  static const Color offlineStatus = Color(0xFF94A3B8);

  // ─── Action Button Colors ───────────────────────────────────────
  static const Color sosRed = Color(0xFFDC2626);
  static const Color sosBg = Color(0xFFFEE2E2);
  static const Color sosBgDark = Color(0xFF450A0A);

  static const Color medicationPurple = Color(0xFF7C3AED);
  static const Color medicationBg = Color(0xFFF3E8FF);
  static const Color medicationBgDark = Color(0xFF2E1065);

  static const Color homeTeal = Color(0xFF0D9488);
  static const Color homeBg = Color(0xFFCCFBF1);
  static const Color homeBgDark = Color(0xFF042F2E);

  static const Color followBlue = Color(0xFF2563EB);
  static const Color followBg = Color(0xFFDBEAFE);
  static const Color followBgDark = Color(0xFF172554);

  // ─── Gradients ──────────────────────────────────────────────────
  static const LinearGradient primaryGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFF2563EB), Color(0xFF7C3AED)],
  );

  static const LinearGradient emergencyGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFFDC2626), Color(0xFFEF4444)],
  );

  static const LinearGradient tealGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Color(0xFF0D9488), Color(0xFF14B8A6)],
  );

  static const LinearGradient darkOverlay = LinearGradient(
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
    colors: [Color(0x00000000), Color(0x80000000)],
  );

  /// Returns a color representing the battery level.
  static Color batteryColor(int level) {
    if (level > 50) return batteryGood;
    if (level > 20) return batteryMedium;
    return batteryLow;
  }

  /// Returns a color representing heart rate.
  static Color heartRateColor(int bpm) {
    if (bpm < 100) return heartRateNormal;
    if (bpm < 120) return heartRateElevated;
    return heartRateCritical;
  }

  // ─── Dark theme (rover controls / joystick) ──────────────────
  static const Color roverDarkBg = Color(0xFF0B1E27);
  static const Color roverDarkCard = Color(0x0AFFFFFF);
  static const Color roverDarkBorder = Color(0x1AFFFFFF);
  static const Color roverDarkText = Color(0xE6FFFFFF);
  static const Color roverDarkMuted = Color(0x80FFFFFF);

  // ─── Brand accents (for joystick gradient) ──────────────────
  static const Color brandTeal = Color(0xFF2C5F7E);
  static const Color brandLeaf = Color(0xFF6FAF93);
  static const Color accent = Color(0xFFF0B82B);
}
