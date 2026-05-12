import 'package:flutter/material.dart';

/// NovaCare Design System
/// Provides Light, Dark, and High-Contrast themes for maximum accessibility.
class AppTheme {
  AppTheme._();

  // ─── Brand Colors ───────────────────────────────────────────────
  static const Color primaryBlue = Color(0xFF2563EB);
  static const Color primaryBlueLight = Color(0xFF60A5FA);
  static const Color primaryBlueDark = Color(0xFF1D4ED8);

  static const Color accentTeal = Color(0xFF14B8A6);
  static const Color accentTealLight = Color(0xFF5EEAD4);

  static const Color emergencyRed = Color(0xFFDC2626);
  static const Color emergencyRedDark = Color(0xFFB91C1C);

  static const Color warningAmber = Color(0xFFF59E0B);
  static const Color successGreen = Color(0xFF22C55E);

  // ─── Neutral Palette ────────────────────────────────────────────
  static const Color neutralWhite = Color(0xFFF8FAFC);
  static const Color neutralGray50 = Color(0xFFF1F5F9);
  static const Color neutralGray100 = Color(0xFFE2E8F0);
  static const Color neutralGray200 = Color(0xFFCBD5E1);
  static const Color neutralGray400 = Color(0xFF94A3B8);
  static const Color neutralGray600 = Color(0xFF475569);
  static const Color neutralGray800 = Color(0xFF1E293B);
  static const Color neutralGray900 = Color(0xFF0F172A);

  // ─── Dark Surface Colors ────────────────────────────────────────
  static const Color darkSurface = Color(0xFF111827);
  static const Color darkCard = Color(0xFF1F2937);
  static const Color darkCardBorder = Color(0xFF374151);

  // ─── High Contrast Colors ──────────────────────────────────────
  static const Color hcBackground = Color(0xFF000000);
  static const Color hcSurface = Color(0xFF1A1A1A);
  static const Color hcText = Color(0xFFFFFFFF);
  static const Color hcAccent = Color(0xFF00FF88);
  static const Color hcBorder = Color(0xFFFFFFFF);

  // ─── Shared Values ─────────────────────────────────────────────
  static const double borderRadiusSm = 12.0;
  static const double borderRadiusMd = 16.0;
  static const double borderRadiusLg = 24.0;
  static const double buttonHeight = 72.0;
  static const double bigButtonSize = 140.0;

  static final _baseFontFamily = 'Inter';

  // ═══════════════════════════════════════════════════════════════
  //  LIGHT THEME
  // ═══════════════════════════════════════════════════════════════
  static ThemeData lightTheme() {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      fontFamily: _baseFontFamily,
      scaffoldBackgroundColor: neutralWhite,
      colorScheme: const ColorScheme.light(
        primary: primaryBlue,
        secondary: accentTeal,
        error: emergencyRed,
        surface: Colors.white,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onError: Colors.white,
        onSurface: neutralGray900,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.white,
        foregroundColor: neutralGray900,
        elevation: 0,
        scrolledUnderElevation: 2,
        centerTitle: true,
        titleTextStyle: TextStyle(
          fontFamily: 'Inter',
          fontSize: 20,
          fontWeight: FontWeight.w700,
          color: neutralGray900,
        ),
      ),
      cardTheme: CardThemeData(
        color: Colors.white,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(borderRadiusMd),
          side: BorderSide(color: neutralGray100, width: 1),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryBlue,
          foregroundColor: Colors.white,
          minimumSize: const Size.fromHeight(buttonHeight),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(borderRadiusMd),
          ),
          textStyle: const TextStyle(
            fontFamily: 'Inter',
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
          elevation: 0,
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: neutralGray50,
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(borderRadiusSm),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(borderRadiusSm),
          borderSide: BorderSide(color: neutralGray100, width: 1),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(borderRadiusSm),
          borderSide: const BorderSide(color: primaryBlue, width: 2),
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: Colors.white,
        selectedItemColor: primaryBlue,
        unselectedItemColor: neutralGray400,
        type: BottomNavigationBarType.fixed,
        elevation: 8,
      ),
      dividerTheme: const DividerThemeData(
        color: neutralGray100,
        thickness: 1,
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  DARK THEME
  // ═══════════════════════════════════════════════════════════════
  static ThemeData darkTheme() {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      fontFamily: _baseFontFamily,
      scaffoldBackgroundColor: darkSurface,
      colorScheme: const ColorScheme.dark(
        primary: primaryBlueLight,
        secondary: accentTealLight,
        error: emergencyRed,
        surface: darkCard,
        onPrimary: neutralGray900,
        onSecondary: neutralGray900,
        onError: Colors.white,
        onSurface: neutralWhite,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: darkSurface,
        foregroundColor: neutralWhite,
        elevation: 0,
        scrolledUnderElevation: 2,
        centerTitle: true,
        titleTextStyle: TextStyle(
          fontFamily: 'Inter',
          fontSize: 20,
          fontWeight: FontWeight.w700,
          color: neutralWhite,
        ),
      ),
      cardTheme: CardThemeData(
        color: darkCard,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(borderRadiusMd),
          side: BorderSide(color: darkCardBorder, width: 1),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryBlueLight,
          foregroundColor: neutralGray900,
          minimumSize: const Size.fromHeight(buttonHeight),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(borderRadiusMd),
          ),
          textStyle: const TextStyle(
            fontFamily: 'Inter',
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
          elevation: 0,
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: darkCard,
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(borderRadiusSm),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(borderRadiusSm),
          borderSide: BorderSide(color: darkCardBorder, width: 1),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(borderRadiusSm),
          borderSide: const BorderSide(color: primaryBlueLight, width: 2),
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: darkCard,
        selectedItemColor: primaryBlueLight,
        unselectedItemColor: neutralGray400,
        type: BottomNavigationBarType.fixed,
        elevation: 8,
      ),
      dividerTheme: const DividerThemeData(
        color: darkCardBorder,
        thickness: 1,
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  HIGH CONTRAST THEME (Accessibility)
  // ═══════════════════════════════════════════════════════════════
  static ThemeData highContrastTheme() {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      fontFamily: _baseFontFamily,
      scaffoldBackgroundColor: hcBackground,
      colorScheme: const ColorScheme.dark(
        primary: hcAccent,
        secondary: Color(0xFF00BFFF),
        error: Color(0xFFFF4444),
        surface: hcSurface,
        onPrimary: hcBackground,
        onSecondary: hcBackground,
        onError: hcText,
        onSurface: hcText,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: hcBackground,
        foregroundColor: hcText,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(
          fontFamily: 'Inter',
          fontSize: 22,
          fontWeight: FontWeight.w700,
          color: hcText,
        ),
      ),
      cardTheme: CardThemeData(
        color: hcSurface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(borderRadiusMd),
          side: const BorderSide(color: hcBorder, width: 2),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: hcAccent,
          foregroundColor: hcBackground,
          minimumSize: const Size.fromHeight(buttonHeight),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(borderRadiusMd),
            side: const BorderSide(color: hcBorder, width: 2),
          ),
          textStyle: const TextStyle(
            fontFamily: 'Inter',
            fontSize: 20,
            fontWeight: FontWeight.w700,
          ),
          elevation: 0,
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: hcSurface,
        selectedItemColor: hcAccent,
        unselectedItemColor: Colors.white70,
        type: BottomNavigationBarType.fixed,
        elevation: 8,
      ),
      dividerTheme: const DividerThemeData(
        color: hcBorder,
        thickness: 2,
      ),
    );
  }
}
