import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';
import '../providers/settings_provider.dart';
import '../providers/translation_provider.dart';
import '../widgets/status_card.dart';
import '../widgets/quick_action_button.dart';
import '../widgets/play_sound_button.dart';
import '../widgets/audio_share_widget.dart';
import 'sos_screen.dart';
import 'rover_control_screen.dart';
import 'rover_summon_screen.dart';
import 'live_feed_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rover = context.watch<RoverProvider>();
    final settings = context.watch<SettingsProvider>();
    final translation = context.watch<TranslationProvider>();

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            return SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Responsive Header
                  _buildHeader(context, settings, translation),

                  const SizedBox(height: 24),

                  // Robot Connection Status Card
                  _buildConnectionCard(context, rover),

                  const SizedBox(height: 24),

                  // Vital Statistics
                  Text(
                    translation.translate('vital_stats'),
                    style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),

                  // Flexible Grid
                  GridView.count(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    crossAxisCount: constraints.maxWidth > 600 ? 4 : 2,
                    mainAxisSpacing: 12,
                    crossAxisSpacing: 12,
                    childAspectRatio: 1.1,
                    children: [
                      StatusCard(
                        icon: Icons.favorite,
                        label: translation.translate('heart_rate'),
                        value: '${rover.heartRate} BPM',
                        color: Colors.redAccent,
                      ),
                      StatusCard(
                        icon: Icons.battery_charging_full,
                        label: translation.translate('battery'),
                        value: '${rover.batteryLevel}%',
                        color: Colors.green,
                      ),
                      StatusCard(
                        icon: Icons.location_on,
                        label: translation.translate('location'),
                        value: rover.roverLocation,
                        color: Colors.blueAccent,
                      ),
                      StatusCard(
                        icon: Icons.thermostat,
                        label: translation.translate('temperature'),
                        value: '${rover.temperature}°C',
                        color: Colors.orangeAccent,
                      ),
                    ],
                  ),

                  const SizedBox(height: 24),

                  // Emergency Section
                  Text(
                    translation.translate('emergency'),
                    style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  QuickActionButton(
                    icon: Icons.warning_rounded,
                    label: 'SOS EMERGENCY',
                    subtitle: 'Trigger alarm & notify caregivers',
                    color: theme.colorScheme.error,
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const SosScreen()),
                    ),
                  ),

                  const SizedBox(height: 12),

                  QuickActionButton(
                    icon: Icons.smart_toy_rounded,
                    label: translation.translate('summon').toUpperCase(),
                    subtitle: 'Call the assistant to your location',
                    color: theme.colorScheme.primary,
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const RoverSummonScreen()),
                    ),
                  ),

                  const SizedBox(height: 12),

                  QuickActionButton(
                    icon: Icons.gamepad_rounded,
                    label: translation.translate('controls').toUpperCase(),
                    subtitle: 'Manual movement & docking',
                    color: Colors.indigo,
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const RoverControlScreen()),
                    ),
                  ),

                  const SizedBox(height: 12),

                  QuickActionButton(
                    icon: Icons.videocam_rounded,
                    label: 'VIEW LIVE FEED',
                    subtitle: 'Watch the robot\'s camera in real-time',
                    color: Colors.teal,
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const LiveFeedScreen()),
                    ),
                  ),

                  const SizedBox(height: 12),

                  // Play sound + upload audio quick actions
                  Row(
                    children: const [
                      PlaySoundButton(frequency: 880, duration: 0.4, label: 'Beep'),
                      SizedBox(width: 12),
                      AudioShareWidget(),
                    ],
                  ),

                  const SizedBox(height: 24),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context, SettingsProvider settings, TranslationProvider translation) {
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                translation.translate('good_morning'),
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              Text(
                settings.userName,
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
        const SizedBox(width: 12),
        GestureDetector(
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const SettingsScreen()),
          ),
          child: CircleAvatar(
            radius: 28,
            backgroundColor: Theme.of(context).colorScheme.primary.withOpacity(0.1),
            backgroundImage: settings.profileImagePath != null
                ? FileImage(File(settings.profileImagePath!))
                : null,
            child: settings.profileImagePath == null
                ? Icon(Icons.person, color: Theme.of(context).colorScheme.primary, size: 30)
                : null,
          ),
        ),
      ],
    );
  }

  Widget _buildConnectionCard(BuildContext context, RoverProvider rover) {
    final isOnline = rover.isRoverOnline;
    final color = isOnline ? Colors.green : Colors.red;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            child: Icon(
              isOnline ? Icons.check_circle : Icons.error,
              color: Colors.white,
              size: 20,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isOnline ? 'Rover is Online' : 'Rover is Offline',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                Text(
                  isOnline ? 'Active and monitoring' : 'Check connection status',
                  style: TextStyle(color: Colors.grey.shade600, fontSize: 13),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
