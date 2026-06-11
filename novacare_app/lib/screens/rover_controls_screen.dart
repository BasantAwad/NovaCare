import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../providers/ble_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../widgets/nc_primitives.dart';
import '../widgets/virtual_joystick.dart';

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
  bool _autoAvoid = true;
  bool _autonomous = false;

  @override
  void initState() {
    super.initState();

    // Auto-connect to the local TCP test rover server if the BLE provider
    // already has an endpoint configured.
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final ble = context.read<BleProvider>();
      if (!ble.isTcpConnected) {
        await ble.connectToTcp();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();
    final ble = context.watch<BleProvider>();
    final connected = rover.isConnected || ble.isAnyConnected;

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

            // ─── Joystick Control ──────────────────────────────
            Expanded(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    VirtualJoystick(
                      onDirectionChanged: (dir) async {
                        try {
                          if (dir == null) {
                            await ble.sendCommand('STOP');
                          } else {
                            final rawDegree = int.parse(dir);
                            final tcpDegree = (rawDegree + 270) % 360;
                            await ble.sendCommand('MOVE:$tcpDegree');
                          }
                        } catch (_) {
                          // swallow communication errors
                        }
                      },
                    ),
                    const SizedBox(height: 24),
                    // Autonomous toggle button
                    GestureDetector(
                      onTap: () async {
                        HapticFeedback.mediumImpact();
                        setState(() => _autonomous = !_autonomous);
                        try {
                          await ble.sendCommand(_autonomous ? 'AUTONOMOUS:ON' : 'AUTONOMOUS:OFF');
                        } catch (_) {}
                      },
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                        decoration: BoxDecoration(
                          color: _autonomous ? AppColors.accent : AppColors.roverDarkCard,
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: _autonomous ? AppColors.accent : AppColors.roverDarkBorder,
                          ),
                          boxShadow: _autonomous
                              ? [
                                  BoxShadow(
                                    color: AppColors.accent.withOpacity(0.3),
                                    blurRadius: 12,
                                    spreadRadius: 1,
                                  )
                                ]
                              : [],
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              _autonomous ? Icons.pause_rounded : Icons.play_arrow_rounded,
                              color: _autonomous ? AppColors.inkNavy : AppColors.roverDarkText,
                              size: 20,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              _autonomous ? 'Running Autonomously' : 'Start Auto Mode',
                              style: AppText.bodyStrong(
                                color: _autonomous ? AppColors.inkNavy : AppColors.roverDarkText,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
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
                            // send a DOCK command if possible and update state
                            await ble.sendCommand('DOCK');
                            await rover.goHome();
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
                            // send immediate STOP / EMERGENCY to rover if connected
                            await ble.sendCommand('STOP');
                            rover.cancelCurrentMode();
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
                          onChanged: (v) async {
                            setState(() => _autoAvoid = v);
                            final ble = context.read<BleProvider>();
                            await ble.sendCommand(v
                                ? 'OBSTACLE_AVOIDANCE:ON'
                                : 'OBSTACLE_AVOIDANCE:OFF');
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

// DPad removed in favor of VirtualJoystick

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
