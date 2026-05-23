import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import 'app_colors.dart';
import 'app_text_styles.dart';

/// NovaCare ThemeData factory.
///
/// The app uses ONE primary light theme (warm cream + teal + yellow). The
/// rover-controls screen wraps itself in its own [roverDarkTheme]. A
/// [highContrastTheme] override is also provided for the accessibility
/// switch in Settings.
///
/// See docs/NOVACARE_SKILL.md §1, §7, §8.
class AppTheme {
  AppTheme._();

  // ─── Light (default everywhere except RoverControls) ────────────
  static ThemeData lightTheme() {
    final base = ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: AppColors.canvas,
      colorScheme: const ColorScheme.light(
        primary: AppColors.brandTeal,
        onPrimary: Colors.white,
        secondary: AppColors.accent,
        onSecondary: AppColors.inkNavy,
        error: AppColors.danger,
        onError: Colors.white,
        surface: AppColors.paper,
        onSurface: AppColors.inkNavy,
      ),
      textTheme: GoogleFonts.dmSansTextTheme().apply(
        bodyColor: AppColors.inkNavy,
        displayColor: AppColors.inkNavy,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.canvas,
        foregroundColor: AppColors.inkNavy,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: AppText.appBarTitle(),
        iconTheme: const IconThemeData(color: AppColors.inkNavy),
      ),
      cardTheme: CardThemeData(
        color: AppColors.paper,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(24),
          side: const BorderSide(color: AppColors.line, width: 1),
        ),
      ),
      dividerTheme: const DividerThemeData(
        color: AppColors.line,
        thickness: 1,
        space: 1,
      ),
      switchTheme: SwitchThemeData(
        trackColor: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) return AppColors.brandTeal;
          return AppColors.inkLight;
        }),
        thumbColor: const WidgetStatePropertyAll(Colors.white),
        trackOutlineColor: const WidgetStatePropertyAll(Colors.transparent),
      ),
      iconTheme: const IconThemeData(color: AppColors.inkNavy, size: 22),
      splashFactory: InkRipple.splashFactory,
    );

    return base;
  }

  // ─── Dark (Rover Controls only) ─────────────────────────────────
  static ThemeData roverDarkTheme() {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.roverDarkBg,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.accent,
        onPrimary: AppColors.inkNavy,
        secondary: AppColors.brandTeal,
        onSecondary: Colors.white,
        error: AppColors.danger,
        onError: Colors.white,
        surface: AppColors.roverDarkBg,
        onSurface: Colors.white,
      ),
      textTheme: GoogleFonts.dmSansTextTheme().apply(
        bodyColor: AppColors.roverDarkText,
        displayColor: AppColors.roverDarkText,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.roverDarkBg,
        foregroundColor: AppColors.roverDarkText,
        elevation: 0,
        titleTextStyle: AppText.appBarTitle(color: AppColors.roverDarkText),
        iconTheme: const IconThemeData(color: AppColors.roverDarkText),
      ),
    );
  }

  // ─── High-contrast override ─────────────────────────────────────
  /// Layered on top of [lightTheme] when the user enables HC mode.
  static ThemeData highContrastTheme() {
    final light = lightTheme();
    return light.copyWith(
      scaffoldBackgroundColor: AppColors.paper,
      colorScheme: light.colorScheme.copyWith(
        surface: AppColors.paper,
        onSurface: AppColors.inkNavy,
      ),
      dividerTheme: const DividerThemeData(
        color: AppColors.hcLine,
        thickness: 1.5,
        space: 1.5,
      ),
      cardTheme: CardThemeData(
        color: AppColors.paper,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(24),
          side: const BorderSide(color: AppColors.inkNavy, width: 1.5),
        ),
      ),
    );
  }
}
