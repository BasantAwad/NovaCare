import 'package:flutter/material.dart';

import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// NcTile — telemetry card (SKILL §3.6).
///
/// Replaces the legacy [TelemetryCardWidget]. Re-exported with the old name
/// so existing imports keep working during refactor; prefer `NcTile` in new
/// code.
class NcTile extends StatelessWidget {
  final String label;
  final Widget value;
  final Widget? footer;
  final Widget? decoration;

  const NcTile({
    super.key,
    required this.label,
    required this.value,
    this.footer,
    this.decoration,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: const BoxConstraints(minHeight: 148),
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(Radii.lg),
        border: Border.all(color: AppColors.line),
        boxShadow: Elevations.e1,
      ),
      padding: const EdgeInsetsDirectional.fromSTEB(16, 16, 16, 14),
      child: Stack(
        children: [
          if (decoration != null)
            PositionedDirectional(top: 6, end: 6, child: decoration!),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label.toUpperCase(),
                style: AppText.eyebrow().copyWith(fontSize: 11.5),
              ),
              const SizedBox(height: 8),
              Align(alignment: AlignmentDirectional.centerStart, child: value),
              if (footer != null) ...[
                const SizedBox(height: 10),
                footer!,
              ],
            ],
          ),
        ],
      ),
    );
  }
}

/// Convenience: value + unit, baseline-aligned (e.g. "76" "bpm").
/// Auto-shrinks to fit the tile width so long labels like "Living Room"
/// never overflow.
class NcTileValue extends StatelessWidget {
  final String value;
  final String? unit;
  const NcTileValue({super.key, required this.value, this.unit});

  @override
  Widget build(BuildContext context) {
    return FittedBox(
      fit: BoxFit.scaleDown,
      alignment: AlignmentDirectional.centerStart,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(value, style: AppText.tileValue(), maxLines: 1),
          if (unit != null) ...[
            const SizedBox(width: 4),
            Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Text(unit!, style: AppText.tileUnit(), maxLines: 1),
            ),
          ],
        ],
      ),
    );
  }
}

/// Thin animated ECG-like waveform — placeholder for live heart-rate
/// telemetry on the HeartRate tile (SKILL §3.6).
class NcEcgWaveform extends StatefulWidget {
  final Color color;
  final double height;
  const NcEcgWaveform({
    super.key,
    this.color = AppColors.danger,
    this.height = 28,
  });

  @override
  State<NcEcgWaveform> createState() => _NcEcgWaveformState();
}

class _NcEcgWaveformState extends State<NcEcgWaveform>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2200),
  )..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: widget.height,
      child: AnimatedBuilder(
        animation: _c,
        builder: (_, __) => CustomPaint(
          painter: _EcgPainter(progress: _c.value, color: widget.color),
          size: Size.infinite,
        ),
      ),
    );
  }
}

class _EcgPainter extends CustomPainter {
  final double progress;
  final Color color;
  _EcgPainter({required this.progress, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 1.6
      ..style = PaintingStyle.stroke
      ..strokeJoin = StrokeJoin.round;

    final mid = size.height / 2;
    final path = Path();
    final shift = progress * size.width;

    for (double x = -size.width; x <= size.width * 2; x += 2) {
      // Periodic spike pattern — flat with a sharp QRS every ~70px.
      final cycle = ((x + shift) % 70);
      double y = mid;
      if (cycle > 26 && cycle < 30) y = mid - size.height * 0.45;
      if (cycle >= 30 && cycle < 34) y = mid + size.height * 0.35;
      if (cycle >= 34 && cycle < 36) y = mid - size.height * 0.15;
      if (x == -size.width) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.save();
    canvas.clipRect(Offset.zero & size);
    canvas.drawPath(path, paint);
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _EcgPainter old) =>
      old.progress != progress || old.color != color;
}

/// Thin animated battery fill bar — SKILL §3.6 Battery variant.
class NcBattFillBar extends StatefulWidget {
  final double percent;
  final Color color;
  const NcBattFillBar({
    super.key,
    required this.percent,
    this.color = AppColors.brandTeal,
  });

  @override
  State<NcBattFillBar> createState() => _NcBattFillBarState();
}

class _NcBattFillBarState extends State<NcBattFillBar>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1400),
  )..forward();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _c,
      builder: (_, __) => Container(
        height: 6,
        decoration: BoxDecoration(
          color: AppColors.canvas,
          borderRadius: BorderRadius.circular(3),
        ),
        clipBehavior: Clip.antiAlias,
        child: FractionallySizedBox(
          alignment: AlignmentDirectional.centerStart,
          widthFactor: (widget.percent.clamp(0, 1)) *
              Curves.easeOutCubic.transform(_c.value),
          child: Container(
            decoration: BoxDecoration(
              color: widget.color,
              borderRadius: BorderRadius.circular(3),
            ),
          ),
        ),
      ),
    );
  }
}

/// Gradient temperature bar with animated marker (SKILL §3.6 Temperature).
class NcTempBar extends StatefulWidget {
  final double tempC;
  final double minC;
  final double maxC;
  const NcTempBar({
    super.key,
    required this.tempC,
    this.minC = 15,
    this.maxC = 32,
  });

  @override
  State<NcTempBar> createState() => _NcTempBarState();
}

class _NcTempBarState extends State<NcTempBar>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1400),
  )..forward();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final pos = ((widget.tempC - widget.minC) / (widget.maxC - widget.minC))
        .clamp(0.0, 1.0);
    return AnimatedBuilder(
      animation: _c,
      builder: (_, __) {
        final t = Curves.easeOutCubic.transform(_c.value) * pos;
        return LayoutBuilder(
          builder: (_, c) {
            final width = c.maxWidth;
            return SizedBox(
              height: 12,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  Positioned.fill(
                    top: 4,
                    bottom: 4,
                    child: Container(
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [
                            Color(0xFF3F8FE6),
                            Color(0xFF6FAF93),
                            Color(0xFFF0B82B),
                            Color(0xFFD8473D),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  PositionedDirectional(
                    start: (width - 10) * t,
                    top: 0,
                    child: Container(
                      width: 10,
                      height: 10,
                      decoration: BoxDecoration(
                        color: AppColors.paper,
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: AppColors.inkNavy,
                          width: 1.5,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}

/// Radar-style pulsing dot for the Location tile (SKILL §3.6).
class NcRadarDot extends StatefulWidget {
  const NcRadarDot({super.key});

  @override
  State<NcRadarDot> createState() => _NcRadarDotState();
}

class _NcRadarDotState extends State<NcRadarDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2400),
  )..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 18,
      height: 18,
      child: Stack(
        alignment: Alignment.center,
        children: [
          for (final phase in const [0.0, 0.5])
            AnimatedBuilder(
              animation: _c,
              builder: (_, __) {
                final t = (_c.value + phase) % 1.0;
                return Opacity(
                  opacity: (0.8 - t * 0.8).clamp(0.0, 0.8),
                  child: Transform.scale(
                    scale: 0.4 + t * 1.4,
                    child: Container(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: AppColors.brandTeal.withOpacity(0.35),
                      ),
                    ),
                  ),
                );
              },
            ),
          Container(
            width: 6,
            height: 6,
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              color: AppColors.brandTeal,
            ),
          ),
        ],
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  Legacy compat shim
//  Keeps the old TelemetryCardWidget API alive so existing call sites
//  compile while we migrate. TODO(refactor): remove once all screens
//  use NcTile directly.
// ════════════════════════════════════════════════════════════════════
class TelemetryCardWidget extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final String? subtitle;
  final Color color;
  final double? progress;

  const TelemetryCardWidget({
    super.key,
    required this.icon,
    required this.label,
    required this.value,
    this.subtitle,
    required this.color,
    this.progress,
  });

  @override
  Widget build(BuildContext context) {
    return NcTile(
      label: label,
      value: NcTileValue(value: value, unit: subtitle),
      decoration: Icon(icon, size: 18, color: color),
      footer: progress != null
          ? NcBattFillBar(percent: progress!.clamp(0, 1), color: color)
          : null,
    );
  }
}
