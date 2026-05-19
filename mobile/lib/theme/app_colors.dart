import 'package:flutter/material.dart';

/// NovaCare design tokens.
/// Palette derives from the NovaCare logo (teal N + yellow C on warm cream).
/// Keep in sync with docs/NOVACARE_SKILL.md §1.1.
class AppColors {
  AppColors._();

  // ─── Brand ink (navy → muted) ───────────────────────────────────
  static const Color inkNavy = Color(0xFF1A3845);
  static const Color inkTeal = Color(0xFF2C5F7E);
  static const Color inkMid = Color(0xFF4A8095);
  static const Color inkMuted = Color(0xFF7A9AA8);
  static const Color inkLight = Color(0xFFB6C8D1);
  static const Color inkHairline = Color(0xFFDCE5EA);

  // ─── Surfaces (warm cream) ──────────────────────────────────────
  static const Color paper = Color(0xFFFFFFFF);
  static const Color canvas = Color(0xFFF6F2E8);
  static const Color canvas2 = Color(0xFFFBF7EE);
  static const Color line = Color(0xFFE5DCC4);
  static const Color line2 = Color(0xFFD4C8AB);

  // ─── Brand accents ──────────────────────────────────────────────
  static const Color brandTeal = Color(0xFF2C5F7E);
  static const Color brandAqua = Color(0xFFB5D7DC);
  static const Color brandAquaSoft = Color(0xFFDCEDF1);
  static const Color brandLeaf = Color(0xFF6FAF93);
  static const Color brandLeafSoft = Color(0xFFDDECE0);

  // ─── Semantic ───────────────────────────────────────────────────
  static const Color accent = Color(0xFFF0B82B); // logo yellow — CTA
  static const Color accent2 = Color(0xFFFAE0A1);
  static const Color accent3 = Color(0xFFFFF4D2);

  static const Color danger = Color(0xFFD8473D);
  static const Color danger2 = Color(0xFFFBE3DF);
  static const Color success = Color(0xFF2E8F65);
  static const Color success2 = Color(0xFFDDEEDF);
  static const Color info = Color(0xFF2C6E8F);
  static const Color info2 = Color(0xFFDCECF2);

  // ─── Dark theme (rover controls only) ───────────────────────────
  static const Color roverDarkBg = Color(0xFF0B1E27);
  static const Color roverDarkCard = Color(0x0AFFFFFF); // rgba(255,255,255,0.04)
  static const Color roverDarkBorder = Color(0x1AFFFFFF);
  static const Color roverDarkText = Color(0xE6FFFFFF);
  static const Color roverDarkMuted = Color(0x80FFFFFF);

  // ─── High-contrast overrides ────────────────────────────────────
  static const Color hcLine = Color(0xFF0F2329);

  // ─── Derived helpers ────────────────────────────────────────────
  /// Pick a battery indicator color by level.
  static Color batteryColor(int level) {
    if (level > 50) return success;
    if (level > 20) return accent;
    return danger;
  }

  /// Pick a heart-rate indicator color.
  static Color heartRateColor(int bpm) {
    if (bpm < 100) return success;
    if (bpm < 120) return accent;
    return danger;
  }
}

/// Radius scale — see SKILL §1.3.
class Radii {
  Radii._();
  static const double xs = 10.0;
  static const double sm = 14.0;
  static const double md = 18.0;
  static const double lg = 24.0;
  static const double xl = 32.0;
  static const double pill = 999.0;
}

/// Spacing scale — see SKILL §1.4.
class Spacing {
  Spacing._();
  static const double s1 = 4;
  static const double s2 = 8;
  static const double s3 = 12;
  static const double s4 = 16;
  static const double s5 = 20;
  static const double s6 = 24;
  static const double s7 = 32;
  static const double s8 = 40;
}

/// Elevation presets — see SKILL §1.6.
class Elevations {
  Elevations._();
  static const List<BoxShadow> e1 = [
    BoxShadow(color: Color(0x0A1A3845), blurRadius: 8, offset: Offset(0, 2)),
  ];
  static const List<BoxShadow> e2 = [
    BoxShadow(color: Color(0x0D1A3845), blurRadius: 24, offset: Offset(0, 12)),
  ];
  static const List<BoxShadow> e3 = [
    BoxShadow(color: Color(0x0F1A3845), blurRadius: 40, offset: Offset(0, 24)),
  ];
}
