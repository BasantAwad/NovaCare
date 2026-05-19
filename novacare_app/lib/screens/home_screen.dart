import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../l10n/app_localizations.dart';
import '../widgets/nova_logo.dart';
import '../widgets/nc_primitives.dart';
import '../widgets/nc_bottom_nav.dart';
import '../widgets/telemetry_card_widget.dart'
    show NcTile, NcTileValue, NcBattFillBar, NcTempBar, NcRadarDot, NcEcgWaveform;
import '../widgets/action_button_widget.dart' show NcBtnCard, NcBtnCardVariant;
import 'rover_controls_screen.dart';
import 'summon_screen.dart';
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
    // TODO(backend): replace startSimulatedUpdates() with a real BLE
    // notification subscription via BleService.subscribeTelemetry().
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<RoverProvider>().startSimulatedUpdates();
    });
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
                    loading: rover.isProcessingCommand &&
                        rover.currentMode == RoverMode.emergency,
                    onTap: () => _confirmSos(context, rover, l10n),
                  ),
                  const SizedBox(height: 10),
                  NcBtnCard(
                    variant: NcBtnCardVariant.brand,
                    icon: const Icon(Icons.front_hand_rounded),
                    title: 'Summon robot',
                    subtitle: 'Call SERBOT-NC-001 to you',
                    loading: rover.isProcessingCommand &&
                        rover.currentMode == RoverMode.followingUser,
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

                  // ─── Autonomous ────────────────────────────────
                  NcSectionHead(
                    title: 'Autonomous',
                    action: const NcChip(label: 'Beta', style: NcChipStyle.beta),
                  ),
                  NcGroup(
                    children: [
                      NcRow(
                        icon: const Icon(Icons.directions_walk_rounded),
                        title: l10n.translate('follow_me'),
                        subtitle: l10n.translate('follow_me_desc'),
                        trailing: NcSwitch(
                          value: rover.currentMode == RoverMode.followingUser,
                          onChanged: (_) async {
                            HapticFeedback.mediumImpact();
                            await rover.toggleFollowMe();
                            if (mounted) _toast(context, rover);
                          },
                          semanticLabel: l10n.translate('follow_me'),
                        ),
                      ),
                      NcRow(
                        icon: const Icon(Icons.home_rounded),
                        title: l10n.translate('home_dock'),
                        subtitle: l10n.translate('home_dock_desc'),
                        trailing: NcSwitch(
                          value: rover.currentMode == RoverMode.navigatingHome,
                          onChanged: (_) async {
                            HapticFeedback.mediumImpact();
                            // TODO(backend): toggle AprilTag dock-detection
                            // ROS node via BLE write.
                            await rover.goHome();
                            if (mounted) _toast(context, rover);
                          },
                          semanticLabel: l10n.translate('home_dock'),
                        ),
                      ),
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
              // TODO(backend): write SOS to Firebase /sos, send FCM push to
              // caregivers, and trigger robot siren via BLE.
              await rover.sendEmergency();
              if (context.mounted) _toast(context, rover);
            },
            child: Text(l10n.translate('confirm')),
          ),
        ],
      ),
    );
  }

  void _toast(BuildContext context, RoverProvider rover) {
    final msg = rover.lastCommandStatus;
    if (msg == null) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg, style: AppText.bodyStrong(color: Colors.white)),
        behavior: SnackBarBehavior.floating,
        backgroundColor: AppColors.success,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(Radii.md),
        ),
        margin: const EdgeInsets.all(16),
        duration: const Duration(seconds: 2),
      ),
    );
  }
}
