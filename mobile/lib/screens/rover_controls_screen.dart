import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../providers/ble_provider.dart';
import '../providers/settings_provider.dart';
import '../services/robot_service.dart';
import '../services/voice_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../widgets/nc_primitives.dart';

/// RoverControlsScreen — SKILL §4.2.
///
/// Full-screen dark theme overlay (hides the bottom nav). Wraps itself in a
/// dark [Theme] override so AppBar/text inherit the correct colors without
/// affecting the rest of the app.
class RoverControlsScreen extends StatefulWidget {
  const RoverControlsScreen({super.key});

  @override
  State<RoverControlsScreen> createState() => _RoverControlsScreenState();
}

class _RoverControlsScreenState extends State<RoverControlsScreen> {
  String? _activeDir;
  bool _autoAvoid = true;
  bool _autonomous = false;

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();
    final ble = context.watch<BleProvider>();
    final connected = rover.isConnected || ble.isConnected;

    return Theme(
      data: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: AppColors.roverDarkBg,
      ),
      child: Scaffold(
        backgroundColor: AppColors.roverDarkBg,
        body: Column(
          children: [
            NcAppBar(
              dark: true,
              leading: _GhostBtn(
                icon: Icons.arrow_back_ios_new_rounded,
                onTap: () => Navigator.of(context).pop(),
              ),
              title: Text(
                'Remote control',
                style: AppText.appBarTitle(color: AppColors.roverDarkText),
              ),
              status: connected
                  ? NcConnectionStatus.online
                  : NcConnectionStatus.offline,
              statusLabel: connected ? 'Connected' : 'Offline',
            ),

            // ─── Telemetry strip ───────────────────────────────
            Padding(
              padding: const EdgeInsetsDirectional.fromSTEB(20, 8, 20, 8),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                decoration: BoxDecoration(
                  color: AppColors.roverDarkCard,
                  borderRadius: BorderRadius.circular(Radii.md),
                  border: Border.all(color: AppColors.roverDarkBorder),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: _MetricBlock(
                        label: 'Battery',
                        value: '${rover.batteryLevel}%',
                      ),
                    ),
                    Container(
                      width: 1,
                      height: 32,
                      color: AppColors.roverDarkBorder,
                    ),
                    Expanded(
                      child: _MetricBlock(
                        label: 'Speed',
                        value: '${rover.roverSpeed.toStringAsFixed(1)} m/s',
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // ─── D-Pad ─────────────────────────────────────────
            Expanded(
              child: Center(
                child: _DPad(
                  activeDir: _activeDir,
                  autonomous: _autonomous,
                  onPress: (dir) async {
                    HapticFeedback.selectionClick();
                    setState(() => _activeDir = dir);
                    final rover    = context.read<RoverProvider>();
                    final settings = context.read<SettingsProvider>();
                    final movement = switch (dir) {
                      'up'    => RobotMovement.forward,
                      'down'  => RobotMovement.backward,
                      'left'  => RobotMovement.left,
                      'right' => RobotMovement.right,
                      _       => RobotMovement.stop,
                    };
                    VoiceService().speak('Moving ${dir == 'up' ? 'forward' : dir == 'down' ? 'backward' : dir}');
                    await rover.moveRover(movement, settings.robotIp);
                    if (mounted) setState(() => _activeDir = null);
                  },
                  onToggleAutonomous: () async {
                    HapticFeedback.mediumImpact();
                    setState(() => _autonomous = !_autonomous);
                    // Follow-me mode toggle — Firebase/MQTT integration point.
                  },
                ),
              ),
            ),

            // ─── Footer controls ───────────────────────────────
            Padding(
              padding: EdgeInsetsDirectional.only(
                start: 20,
                end: 20,
                bottom: 24 + MediaQuery.of(context).padding.bottom,
              ),
              child: Column(
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: _GhostBtnLarge(
                          icon: Icons.home_rounded,
                          label: 'Dock',
                          onTap: () async {
                            HapticFeedback.mediumImpact();
                            VoiceService().speak('Moving to charging dock');
                            await rover.goHome(context.read<SettingsProvider>().robotIp);
                          },
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _GhostBtnLarge(
                          icon: Icons.stop_rounded,
                          label: 'Stop',
                          danger: true,
                          onTap: () async {
                            HapticFeedback.heavyImpact();
                            VoiceService().speak('Stopping');
                            await rover.cancelCurrentMode(context.read<SettingsProvider>().robotIp);
                          },
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsetsDirectional.symmetric(
                      horizontal: 16,
                      vertical: 12,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.roverDarkCard,
                      borderRadius: BorderRadius.circular(Radii.md),
                      border: Border.all(color: AppColors.roverDarkBorder),
                    ),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.shield_moon_rounded,
                          color: AppColors.accent,
                          size: 20,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Avoid obstacles',
                                style: AppText.bodyStrong(
                                  color: AppColors.roverDarkText,
                                ),
                              ),
                              Text(
                                'LiDAR halts motion under 0.5 m',
                                style: AppText.caption(
                                  color: AppColors.roverDarkMuted,
                                ),
                              ),
                            ],
                          ),
                        ),
                        NcSwitch(
                          value: _autoAvoid,
                          dark: true,
                          onChanged: (v) {
                            setState(() => _autoAvoid = v);
                            // TODO(backend): toggle LiDAR safety guard on
                            // robot side via BLE.
                          },
                          semanticLabel: 'Avoid obstacles',
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  D-Pad
// ════════════════════════════════════════════════════════════════════
class _DPad extends StatelessWidget {
  final String? activeDir;
  final bool autonomous;
  final void Function(String dir) onPress;
  final VoidCallback onToggleAutonomous;

  const _DPad({
    required this.activeDir,
    required this.autonomous,
    required this.onPress,
    required this.onToggleAutonomous,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 296,
      height: 296,
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Outer ring placeholder — TODO(feature): animated SVG gradient
          // (teal → leaf → accent) spinning 16s linear infinite.
          Container(
            width: 296,
            height: 296,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: const SweepGradient(
                colors: [
                  AppColors.brandTeal,
                  AppColors.brandLeaf,
                  AppColors.accent,
                  AppColors.brandTeal,
                ],
              ),
            ),
            padding: const EdgeInsets.all(2),
            child: Container(
              decoration: const BoxDecoration(
                color: AppColors.roverDarkBg,
                shape: BoxShape.circle,
              ),
            ),
          ),

          // Cardinal direction buttons
          PositionedDirectional(
            top: 16,
            child: _DpadBtn(
              icon: Icons.keyboard_arrow_up_rounded,
              active: activeDir == 'up',
              onTap: () => onPress('up'),
            ),
          ),
          PositionedDirectional(
            bottom: 16,
            child: _DpadBtn(
              icon: Icons.keyboard_arrow_down_rounded,
              active: activeDir == 'down',
              onTap: () => onPress('down'),
            ),
          ),
          PositionedDirectional(
            start: 16,
            child: _DpadBtn(
              icon: Icons.keyboard_arrow_left_rounded,
              active: activeDir == 'left',
              onTap: () => onPress('left'),
            ),
          ),
          PositionedDirectional(
            end: 16,
            child: _DpadBtn(
              icon: Icons.keyboard_arrow_right_rounded,
              active: activeDir == 'right',
              onTap: () => onPress('right'),
            ),
          ),

          // Center "Auto · Go" button
          GestureDetector(
            onTap: onToggleAutonomous,
            child: Container(
              width: 104,
              height: 104,
              decoration: BoxDecoration(
                color: AppColors.accent,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: AppColors.accent.withOpacity(0.4),
                    blurRadius: 24,
                    spreadRadius: 4,
                  ),
                ],
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    autonomous ? Icons.pause_rounded : Icons.play_arrow_rounded,
                    color: AppColors.inkNavy,
                    size: 36,
                  ),
                  Text(
                    autonomous ? 'Running' : 'Auto · Go',
                    style: AppText.caption(color: AppColors.inkNavy)
                        .copyWith(fontWeight: FontWeight.w800, fontSize: 11),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _DpadBtn extends StatelessWidget {
  final IconData icon;
  final bool active;
  final VoidCallback onTap;
  const _DpadBtn({
    required this.icon,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        width: 72,
        height: 72,
        decoration: BoxDecoration(
          color: active ? AppColors.accent.withOpacity(0.16) : AppColors.roverDarkCard,
          shape: BoxShape.circle,
          border: Border.all(color: AppColors.roverDarkBorder),
        ),
        child: Icon(
          icon,
          size: 32,
          color: active ? AppColors.accent : AppColors.roverDarkText,
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════
class _MetricBlock extends StatelessWidget {
  final String label;
  final String value;
  const _MetricBlock({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label.toUpperCase(),
          style: AppText.eyebrow(color: AppColors.roverDarkMuted),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: AppText.tileValue(color: AppColors.roverDarkText)
              .copyWith(fontSize: 20),
        ),
      ],
    );
  }
}

class _GhostBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  const _GhostBtn({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: AppColors.roverDarkCard,
          shape: BoxShape.circle,
          border: Border.all(color: AppColors.roverDarkBorder),
        ),
        child: Icon(icon, color: AppColors.roverDarkText, size: 18),
      ),
    );
  }
}

class _GhostBtnLarge extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final bool danger;
  const _GhostBtnLarge({
    required this.icon,
    required this.label,
    required this.onTap,
    this.danger = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = danger ? AppColors.danger : AppColors.roverDarkText;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 64,
        decoration: BoxDecoration(
          color: danger
              ? AppColors.danger.withOpacity(0.16)
              : AppColors.roverDarkCard,
          borderRadius: BorderRadius.circular(Radii.md),
          border: Border.all(
            color: danger
                ? AppColors.danger.withOpacity(0.45)
                : AppColors.roverDarkBorder,
          ),
        ),
        alignment: Alignment.center,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: color, size: 22),
            const SizedBox(width: 8),
            Text(
              label,
              style: AppText.bodyStrong(color: color),
            ),
          ],
        ),
      ),
    );
  }
}
