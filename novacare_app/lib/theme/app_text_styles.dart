import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';



/// Typography helpers — see SKILL §1.2.
/// Display family: Baloo 2. Body family: DM Sans. Mono: JetBrains Mono.
/// All sourced via google_fonts to avoid bundling .ttf files.
class AppText {
  AppText._();

  // ─── Numeric tabular features ───────────────────────────────────
  static const _tabular = [FontFeature.tabularFigures()];

  static TextStyle display1({Color? color}) =>
      GoogleFonts.baloo2(
        fontWeight: FontWeight.w800,
        fontSize: 30,
        letterSpacing: -0.6,
        color: color,
        height: 1.1,
      );

  static TextStyle display2({Color? color}) =>
      GoogleFonts.baloo2(
        fontWeight: FontWeight.w700,
        fontSize: 22,
        letterSpacing: -0.22,
        color: color,
        height: 1.15,
      );

  static TextStyle display3({Color? color}) =>
      GoogleFonts.baloo2(
        fontWeight: FontWeight.w700,
        fontSize: 17,
        color: color,
      );

  static TextStyle appBarTitle({Color? color}) =>
      GoogleFonts.baloo2(
        fontWeight: FontWeight.w700,
        fontSize: 19,
        letterSpacing: -0.19,
        color: color,
      );

  static TextStyle tileValue({Color? color}) =>
      GoogleFonts.baloo2(
        fontWeight: FontWeight.w800,
        fontSize: 30,
        color: color,
        fontFeatures: _tabular,
        height: 1.0,
      );

  static TextStyle tileUnit({Color? color}) =>
      GoogleFonts.baloo2(
        fontWeight: FontWeight.w600,
        fontSize: 16,
        color: color,
      );

  static TextStyle body({Color? color}) =>
      GoogleFonts.dmSans(
        fontWeight: FontWeight.w400,
        fontSize: 14,
        color: color,
        height: 1.5,
      );

  static TextStyle bodyStrong({Color? color}) =>
      GoogleFonts.dmSans(
        fontWeight: FontWeight.w600,
        fontSize: 15,
        color: color,
        height: 1.45,
      );

  static TextStyle caption({Color? color}) =>
      GoogleFonts.dmSans(
        fontWeight: FontWeight.w500,
        fontSize: 12.5,
        color: color,
      );

  /// Uppercase eyebrow / section-head label.
  static TextStyle eyebrow({Color? color}) =>
      GoogleFonts.baloo2(
        fontWeight: FontWeight.w700,
        fontSize: 11,
        letterSpacing: 1.32,
        color: color,
      );

  /// Slightly larger section-head used in SKILL §3.9.
  static TextStyle sectionHead({Color? color}) =>
      GoogleFonts.baloo2(
        fontWeight: FontWeight.w700,
        fontSize: 13,
        letterSpacing: 1.3,
        color: color,
      );

  static TextStyle mono({Color color = Colors.white70}) =>
      GoogleFonts.jetBrainsMono(
        fontWeight: FontWeight.w400,
        fontSize: 10,
        color: color,
      );
}
