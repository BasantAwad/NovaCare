import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../providers/settings_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../l10n/app_localizations.dart';
import '../widgets/nova_logo.dart';
import '../widgets/nc_primitives.dart';
import '../widgets/nc_bottom_nav.dart';
import '../widgets/telemetry_card_widget.dart'
    show NcTile, NcTileValue, NcBattFillBar, NcTempBar, NcRadarDot, NcEcgWaveform;
import '../widgets/action_button_widget.dart' show NcBtnCard, NcBtnCardVariant;
import '../widgets/play_sound_button.dart';
import '../widgets/audio_share_widget.dart';
import 'rover_controls_screen.dart';
import 'summon_screen.dart';
import 'sos_screen.dart';
import 'main_navigation.dart';

/// HomeScreen — top of the tab stack.
///
/// Driven by [RoverProvider] (mock telemetry for now; real BLE/Firebase
/// streams wire in via TODOs below). Matches the layout in SKILL §4.1.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    // BLE telemetry → RoverProvider is wired in BleProvider.onTelemetry (set in main.dart).
    // No simulated updates needed.
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final rover = context.watch<RoverProvider>();

    return Scaffold(
      backgroundColor: AppColors.canvas,
      body: Column(
        children: [
          NcAppBar(
            leading: const NovaWordmark(),
            status: rover.isRoverOnline
                ? NcConnectionStatus.online
                : NcConnectionStatus.offline,
            statusLabel: rover.isRoverOnline ? 'Live' : 'Offline',
            battery: rover.batteryLevel,
          ),
          Expanded(
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsetsDirectional.fromSTEB(20, 8, 20, 40),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ─── Hero greeting ─────────────────────────────
                  Text(_greeting(), style: AppText.display1()),
                  const SizedBox(height: 4),
                  Text(
                    'SERBOT-NC-001 is nearby and ready.',
                    style: AppText.body(color: AppColors.inkMuted),
                  ),

                  // ─── Telemetry grid ────────────────────────────
                  NcSectionHead(
                    title: 'Telemetry',
                    action: GestureDetector(
                      onTap: () {
                        // TODO(feature): full telemetry history screen with
                        // charts (HR trend, battery curve, temp log).
                      },
                      child: Text(
                        'View all',
                        style: AppText.caption(color: AppColors.brandTeal)
                            .copyWith(fontWeight: FontWeight.w700),
                      ),
                    ),
                  ),
                  _telemetryGrid(rover, l10n),

                  // ─── Quick actions ─────────────────────────────
                  const NcSectionHead(title: 'Quick actions'),
                  NcBtnCard(
                    variant: NcBtnCardVariant.sos,
                    icon: const Icon(Icons.emergency_rounded),
                    title: l10n.translate('sos_emergency'),
                    subtitle: 'Notify caregivers & sound alarm',
                    loading: rover.isProcessing,
                    onTap: () => _confirmSos(context, rover, l10n),
                  ),
                  const SizedBox(height: 10),
                  NcBtnCard(
                    variant: NcBtnCardVariant.brand,
                    icon: const Icon(Icons.front_hand_rounded),
                    title: 'Summon robot',
                    subtitle: 'Call SERBOT-NC-001 to you',
                    loading: false,
                    onTap: () async {
                      HapticFeedback.mediumImpact();
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const SummonScreen(),
                          fullscreenDialog: true,
                        ),
                      );
                    },
                  ),
                  const SizedBox(height: 10),
                  NcBtnCard(
                    variant: NcBtnCardVariant.brand,
                    icon: const Icon(Icons.videogame_asset_rounded),
                    title: 'Rover controls',
                    subtitle: 'D-pad, live feed, dock & avoid',
                    onTap: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const RoverControlsScreen(),
                          fullscreenDialog: true,
                        ),
                      );
                    },
                  ),

                  // ─── Robot Sound & Audio ───────────────────────
                  const NcSectionHead(title: 'Robot Audio'),
                  Row(
                    children: const [
                      PlaySoundButton(frequency: 880, duration: 0.4, label: 'Beep Robot'),
                      SizedBox(width: 12),
                      AudioShareWidget(),
                    ],
                  ),

                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ──────────────────────────────────────────────────────────────────
  //  Telemetry grid
  // ──────────────────────────────────────────────────────────────────
  Widget _telemetryGrid(RoverProvider rover, AppLocalizations l10n) {
    final battColor = AppColors.batteryColor(rover.batteryLevel);
    // Rough runtime estimate: assume ~7h on a full charge.
    final minutesLeft = (rover.batteryLevel / 100 * 7 * 60).round();
    final battEta =
        '${minutesLeft ~/ 60}h ${(minutesLeft % 60).toString().padLeft(2, '0')}m left';

    return Column(
      children: [
        SizedBox(
          height: 184,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: NcTile(
                  label: l10n.translate('heart_rate'),
                  decoration: const Icon(
                    Icons.favorite_rounded,
                    color: AppColors.danger,
                    size: 18,
                  ),
                  value: NcTileValue(
                    value: '${rover.heartRate}',
                    unit: 'bpm',
                  ),
                  footer: const NcEcgWaveform(),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: NcTile(
                  label: 'Robot battery',
                  value: NcTileValue(value: '${rover.batteryLevel}', unit: '%'),
                  decoration: Icon(
                    Icons.battery_full_rounded,
                    color: battColor,
                    size: 18,
                  ),
                  footer: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(battEta, style: AppText.caption()),
                      const SizedBox(height: 6),
                      NcBattFillBar(
                        percent: rover.batteryLevel / 100,
                        color: battColor,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        SizedBox(
          height: 184,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: NcTile(
                  label: l10n.translate('location'),
                  decoration: const NcRadarDot(),
                  value: NcTileValue(value: rover.roverLocation),
                  footer: Text(
                    _zoneLabel(rover.roverLocation),
                    style: AppText.caption(),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: NcTile(
                  label: l10n.translate('temperature'),
                  decoration: Icon(
                    Icons.thermostat_rounded,
                    color: _tempColor(rover.temperature),
                    size: 18,
                  ),
                  value: NcTileValue(
                    value: rover.temperature.toStringAsFixed(1),
                    unit: '°C',
                  ),
                  footer: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(_tempLabel(rover.temperature), style: AppText.caption()),
                      const SizedBox(height: 6),
                      NcTempBar(tempC: rover.temperature),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // TODO(backend): derive zone + comfort labels from real sensor metadata
  // instead of these heuristics.
  String _zoneLabel(String room) {
    switch (room.toLowerCase()) {
      case 'kitchen':
        return 'Zone 02 · West';
      case 'living room':
        return 'Zone 01 · Main';
      case 'bedroom':
        return 'Zone 03 · East';
      default:
        return 'Zone — · Home';
    }
  }

  String _tempLabel(double c) {
    if (c < 18) return 'Cool';
    if (c < 25) return 'Comfortable';
    if (c < 29) return 'Warm';
    return 'Hot';
  }

  Color _tempColor(double c) {
    if (c < 18) return AppColors.info;
    if (c < 25) return AppColors.success;
    if (c < 29) return AppColors.accent;
    return AppColors.danger;
  }

  // ──────────────────────────────────────────────────────────────────
  String _greeting() {
    final h = DateTime.now().hour;
    if (h < 12) return 'Good morning';
    if (h < 18) return 'Good afternoon';
    return 'Good evening';
  }

  void _confirmSos(BuildContext context, RoverProvider rover, AppLocalizations l10n) {
    HapticFeedback.heavyImpact();
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(Radii.lg),
        ),
        backgroundColor: AppColors.paper,
        title: Row(
          children: [
            const Icon(Icons.emergency_rounded, color: AppColors.danger),
            const SizedBox(width: 10),
            Text(l10n.translate('sos_emergency'), style: AppText.display3()),
          ],
        ),
        content: Text(l10n.translate('sos_confirm'), style: AppText.body()),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text(l10n.translate('cancel')),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.danger,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(Radii.sm),
              ),
            ),
            onPressed: () async {
              Navigator.of(ctx).pop();
              await rover.sendEmergency();
              if (context.mounted) {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const SosScreen(), fullscreenDialog: true),
                );
              }
            },
            child: Text(l10n.translate('confirm')),
          ),
        ],
      ),
    );
  }

  void _toast(BuildContext context, String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg, style: AppText.bodyStrong(color: Colors.white)),
        behavior: SnackBarBehavior.floating,
        backgroundColor: AppColors.success,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Radii.md)),
        margin: const EdgeInsets.all(16),
        duration: const Duration(seconds: 2),
      ),
    );
  }
}
