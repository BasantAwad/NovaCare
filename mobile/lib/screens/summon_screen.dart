import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/summon_provider.dart';
import '../providers/ble_provider.dart';
import '../providers/rover_provider.dart';
import '../providers/settings_provider.dart';
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
  bool _summonStarted = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(seconds: 4))..repeat();

    WidgetsBinding.instance.addPostFrameCallback((_) => _startSummonSequence());
  }

  Future<void> _startSummonSequence() async {
    if (_summonStarted) return;
    _summonStarted = true;

    final summon   = context.read<SummonProvider>();
    final ble      = context.read<BleProvider>();
    final settings = context.read<SettingsProvider>();

    // Connect WebSocket (port 9999) if not already connected.
    if (!summon.isConnected) {
      await summon.connectToRobot(robotHost: settings.robotIp);
    }

    final bleMac = ble.connectedDeviceId ?? 'AA:BB:CC:DD:EE:FF';
    await summon.startSummon(userId: settings.userId, bleMac: bleMac);

    // Feed live BLE RSSI → robot's Kalman-filtered navigator.
    ble.addListener(_onBleUpdate);
  }

  void _onBleUpdate() {
    if (!mounted) return;
    final ble    = context.read<BleProvider>();
    final summon = context.read<SummonProvider>();
    if (summon.isActive) summon.sendRssiUpdate(ble.rssi);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Color _rssiColor(int rssi) {
    if (rssi >= -50) return Colors.green;
    if (rssi >= -70) return Colors.orange;
    if (rssi >= -85) return Colors.deepOrange;
    return Colors.red;
  }

  String _phaseIcon(SummonPhase phase) {
    switch (phase) {
      case SummonPhase.arrived:     return '✅';
      case SummonPhase.failed:      return '❌';
      case SummonPhase.navigating:
      case SummonPhase.wallFollowing:
      case SummonPhase.recovering:  return '🤖';
      default:                      return '📡';
    }
  }

  @override
  Widget build(BuildContext context) {
    final summon = context.watch<SummonProvider>();
    final theme  = Theme.of(context);

    return Scaffold(
      backgroundColor: AppColors.canvas,
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

            // ── Radar / status animation ─────────────────────────
            Center(
              child: Stack(
                alignment: Alignment.center,
                children: [
                  Container(
                    width: 250, height: 250,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(color: Colors.white.withOpacity(0.2), width: 2),
                    ),
                  ),
                  if (summon.isActive)
                    RotationTransition(
                      turns: _controller,
                      child: Container(
                        width: 250, height: 250,
                        alignment: Alignment.centerRight,
                        child: Container(
                          width: 16, height: 16,
                          decoration: BoxDecoration(
                            color: _rssiColor(summon.rssiCurrent),
                            shape: BoxShape.circle,
                          ),
                        ),
                      ),
                    ),
                  Container(
                    width: 80, height: 80,
                    decoration: BoxDecoration(
                      color: summon.phase == SummonPhase.failed
                          ? Colors.red.shade900
                          : Colors.white,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(color: Colors.black.withOpacity(0.3), blurRadius: 10, spreadRadius: 2),
                      ],
                    ),
                    child: Center(
                      child: Text(
                        _phaseIcon(summon.phase),
                        style: const TextStyle(fontSize: 32),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 32),

            // ── Status message ──────────────────────────────────
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40),
              child: Text(
                summon.statusMessage.isNotEmpty
                    ? summon.statusMessage
                    : 'Robot is finding you...',
                textAlign: TextAlign.center,
                style: AppText.display1().copyWith(color: Colors.white, fontSize: 28),
              ),
            ),

            const SizedBox(height: 16),

            // ── RSSI + perf telemetry ───────────────────────────
            if (summon.isActive || summon.isTerminal) ...[
              Text(
                '${summon.signalStrengthLabel}  •  ${summon.rssiCurrent} dBm  •  ${summon.rssiTrend}',
                style: AppText.body().copyWith(
                  color: _rssiColor(summon.rssiCurrent),
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              if (summon.obstacleDetected)
                Text('⚠ Obstacle detected',
                    style: AppText.caption().copyWith(color: Colors.orangeAccent)),
              if (summon.isWallFollowing)
                Text('↩ Wall-following mode',
                    style: AppText.caption().copyWith(color: Colors.lightBlueAccent)),
              if (summon.recoveryAttempts > 0)
                Text('Recovery attempts: ${summon.recoveryAttempts}',
                    style: AppText.caption().copyWith(color: Colors.redAccent)),

              // Jetson Nano perf telemetry
              if (summon.cpuUsage > 0) ...[
                const SizedBox(height: 8),
                Text(
                  'CPU ${summon.cpuUsage.toStringAsFixed(0)}%  RAM ${summon.ramUsage.toStringAsFixed(0)}%'
                  '  ${summon.cameraFps.toStringAsFixed(1)} FPS  ${summon.loopLatencyMs.toStringAsFixed(0)}ms',
                  style: AppText.caption().copyWith(color: Colors.white54),
                  textAlign: TextAlign.center,
                ),
              ],
            ],

            if (summon.phase == SummonPhase.failed && summon.failureReason.isNotEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 8),
                child: Text(summon.failureReason,
                    style: AppText.body().copyWith(color: Colors.redAccent),
                    textAlign: TextAlign.center),
              ),

            if (!summon.isActive && !summon.isTerminal)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40),
                child: Text(
                  'SERBot navigates to you using RSSI-guided path planning (Kalman + 5-tap avg).',
                  textAlign: TextAlign.center,
                  style: AppText.body().copyWith(color: Colors.white70),
                ),
              ),

            const Spacer(),

            // ── Cancel / Back button ────────────────────────────
            Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: TextButton(
                onPressed: () async {
                  if (summon.isActive) {
                    await summon.cancelSummon(userId: context.read<SettingsProvider>().userId);
                    await context.read<RoverProvider>().cancelCurrentMode(
                        context.read<SettingsProvider>().robotIp);
                    context.read<BleProvider>().removeListener(_onBleUpdate);
                  }
                  summon.resetToIdle();
                  if (mounted) Navigator.of(context).pop();
                },
                style: TextButton.styleFrom(
                  backgroundColor: Colors.white.withOpacity(0.1),
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                ),
                child: Text(
                  summon.isActive ? 'Cancel Summon' : 'Back',
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
