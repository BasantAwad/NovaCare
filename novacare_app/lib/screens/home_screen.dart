import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../l10n/app_localizations.dart';
import '../widgets/status_bar_widget.dart';
import '../widgets/action_button_widget.dart';
import '../widgets/telemetry_card_widget.dart';
import '../widgets/connection_indicator.dart';
import 'settings_screen.dart';

/// Main dashboard screen with big action buttons and rover telemetry.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  late AnimationController _staggerController;

  @override
  void initState() {
    super.initState();
    _staggerController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    )..forward();

    // Start simulated telemetry updates
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<RoverProvider>().startSimulatedUpdates();
    });
  }

  @override
  void dispose() {
    _staggerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final rover = context.watch<RoverProvider>();
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      body: SafeArea(
        child: CustomScrollView(
          physics: const BouncingScrollPhysics(),
          slivers: [
            // ─── App Bar ─────────────────────────────────────
            SliverAppBar(
              floating: true,
              snap: true,
              backgroundColor: theme.scaffoldBackgroundColor,
              title: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      gradient: AppColors.primaryGradient,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(
                      Icons.health_and_safety_rounded,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        l10n.translate('app_title'),
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          color: theme.colorScheme.onSurface,
                        ),
                      ),
                      Text(
                        l10n.translate('app_subtitle'),
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w400,
                          color: theme.colorScheme.onSurface.withOpacity(0.6),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              actions: [
                // Connection indicator
                const ConnectionIndicator(),
                const SizedBox(width: 4),
                // Settings gear icon
                IconButton(
                  icon: Icon(
                    Icons.settings_rounded,
                    color: theme.colorScheme.onSurface.withOpacity(0.7),
                  ),
                  tooltip: l10n.translate('settings'),
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => const SettingsScreen(),
                      ),
                    );
                  },
                ),
                const SizedBox(width: 8),
              ],
            ),

            // ─── Rover Status Bar ────────────────────────────
            SliverToBoxAdapter(
              child: _buildStaggeredChild(
                index: 0,
                child: const Padding(
                  padding: EdgeInsets.fromLTRB(16, 8, 16, 4),
                  child: StatusBarWidget(),
                ),
              ),
            ),

            // ─── Telemetry Cards ─────────────────────────────
            SliverToBoxAdapter(
              child: _buildStaggeredChild(
                index: 1,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Row(
                    children: [
                      Expanded(
                        child: TelemetryCardWidget(
                          icon: Icons.battery_charging_full_rounded,
                          label: l10n.translate('battery'),
                          value: '${rover.batteryLevel}%',
                          color: AppColors.batteryColor(rover.batteryLevel),
                          progress: rover.batteryLevel / 100,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: TelemetryCardWidget(
                          icon: Icons.favorite_rounded,
                          label: l10n.translate('heart_rate'),
                          value: '${rover.heartRate}',
                          subtitle: l10n.translate('bpm'),
                          color: AppColors.heartRateColor(rover.heartRate),
                          progress: rover.heartRate / 160,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            SliverToBoxAdapter(
              child: _buildStaggeredChild(
                index: 2,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                  child: Row(
                    children: [
                      Expanded(
                        child: TelemetryCardWidget(
                          icon: Icons.location_on_rounded,
                          label: l10n.translate('location'),
                          value: rover.roverLocation,
                          color: theme.colorScheme.primary,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: TelemetryCardWidget(
                          icon: Icons.thermostat_rounded,
                          label: l10n.translate('temperature'),
                          value: '${rover.temperature.toStringAsFixed(1)}°C',
                          color: AppColors.warningAmber,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // ─── Section Title ───────────────────────────────
            SliverToBoxAdapter(
              child: _buildStaggeredChild(
                index: 3,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
                  child: Text(
                    'Quick Actions',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: theme.colorScheme.onSurface,
                    ),
                  ),
                ),
              ),
            ),

            // ─── SOS Emergency Button (PROMINENT) ────────────
            SliverToBoxAdapter(
              child: _buildStaggeredChild(
                index: 4,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                  child: ActionButtonWidget(
                    icon: Icons.emergency_rounded,
                    label: l10n.translate('sos_emergency'),
                    subtitle: l10n.translate('sos_desc'),
                    color: AppColors.sosRed,
                    backgroundColor: isDark ? AppColors.sosBgDark : AppColors.sosBg,
                    isLarge: true,
                    isEmergency: true,
                    isLoading: rover.isProcessingCommand &&
                        rover.currentMode == RoverMode.emergency,
                    onPressed: () => _showSOSConfirmation(context, rover, l10n),
                  ),
                ),
              ),
            ),

            // ─── Action Button Grid ──────────────────────────
            SliverToBoxAdapter(
              child: _buildStaggeredChild(
                index: 5,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                  child: Row(
                    children: [
                      // Medication Button
                      Expanded(
                        child: ActionButtonWidget(
                          icon: Icons.medication_rounded,
                          label: l10n.translate('medication'),
                          subtitle: l10n.translate('medication_desc'),
                          color: AppColors.medicationPurple,
                          backgroundColor: isDark
                              ? AppColors.medicationBgDark
                              : AppColors.medicationBg,
                          isLoading: rover.isProcessingCommand &&
                              rover.currentMode == RoverMode.deliveringMedicine,
                          onPressed: () async {
                            HapticFeedback.mediumImpact();
                            await rover.requestMedication();
                            _showCommandStatus(context, rover);
                          },
                        ),
                      ),
                      const SizedBox(width: 12),
                      // Home / Dock Button
                      Expanded(
                        child: ActionButtonWidget(
                          icon: Icons.home_rounded,
                          label: l10n.translate('home_dock'),
                          subtitle: l10n.translate('home_dock_desc'),
                          color: AppColors.homeTeal,
                          backgroundColor: isDark
                              ? AppColors.homeBgDark
                              : AppColors.homeBg,
                          isLoading: rover.isProcessingCommand &&
                              rover.currentMode == RoverMode.navigatingHome,
                          onPressed: () async {
                            HapticFeedback.mediumImpact();
                            await rover.goHome();
                            _showCommandStatus(context, rover);
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            // ─── Follow Me Button ────────────────────────────
            SliverToBoxAdapter(
              child: _buildStaggeredChild(
                index: 6,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 4, 16, 20),
                  child: ActionButtonWidget(
                    icon: Icons.directions_walk_rounded,
                    label: l10n.translate('follow_me'),
                    subtitle: l10n.translate('follow_me_desc'),
                    color: AppColors.followBlue,
                    backgroundColor: isDark
                        ? AppColors.followBgDark
                        : AppColors.followBg,
                    isActive: rover.currentMode == RoverMode.followingUser,
                    isLoading: rover.isProcessingCommand &&
                        rover.currentMode == RoverMode.followingUser,
                    onPressed: () async {
                      HapticFeedback.mediumImpact();
                      await rover.toggleFollowMe();
                      _showCommandStatus(context, rover);
                    },
                  ),
                ),
              ),
            ),

            // Bottom padding
            const SliverToBoxAdapter(
              child: SizedBox(height: 40),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStaggeredChild({required int index, required Widget child}) {
    final begin = (index * 0.1).clamp(0.0, 0.7);
    final end = (begin + 0.4).clamp(0.0, 1.0);

    return AnimatedBuilder(
      animation: _staggerController,
      builder: (context, _) {
        final value = Curves.easeOutCubic.transform(
          (((_staggerController.value - begin) / (end - begin))
              .clamp(0.0, 1.0)),
        );
        return Opacity(
          opacity: value,
          child: Transform.translate(
            offset: Offset(0, 30 * (1 - value)),
            child: child,
          ),
        );
      },
    );
  }

  void _showSOSConfirmation(
    BuildContext context,
    RoverProvider rover,
    AppLocalizations l10n,
  ) {
    HapticFeedback.heavyImpact();

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Row(
          children: [
            const Icon(Icons.emergency_rounded, color: AppColors.sosRed, size: 28),
            const SizedBox(width: 12),
            Text(l10n.translate('sos_emergency')),
          ],
        ),
        content: Text(
          l10n.translate('sos_confirm'),
          style: const TextStyle(fontSize: 16, height: 1.5),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text(l10n.translate('cancel')),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.sosRed,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            onPressed: () async {
              Navigator.of(ctx).pop();
              await rover.sendEmergency();
              if (context.mounted) _showCommandStatus(context, rover);
            },
            child: Text(l10n.translate('confirm')),
          ),
        ],
      ),
    );
  }

  void _showCommandStatus(BuildContext context, RoverProvider rover) {
    if (rover.lastCommandStatus != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(
            children: [
              const Icon(Icons.check_circle_rounded, color: Colors.white, size: 20),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  rover.lastCommandStatus!,
                  style: const TextStyle(fontWeight: FontWeight.w500),
                ),
              ),
            ],
          ),
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          margin: const EdgeInsets.all(16),
          backgroundColor: AppColors.successGreen,
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }
}
