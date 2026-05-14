import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';
import '../providers/translation_provider.dart';
import '../services/voice_service.dart';

class RoverControlScreen extends StatelessWidget {
  const RoverControlScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rover = context.watch<RoverProvider>();
    final translation = context.watch<TranslationProvider>();

    return Scaffold(
      appBar: AppBar(
        title: Text(translation.translate('controls')),
        centerTitle: true,
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          return Column(
            children: [
              // Telemetry Header
              _buildTelemetryHeader(context, rover),

              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Directional Pad (Arrows)
                      _buildDPad(context, rover, theme),

                      const SizedBox(height: 48),

                      // Action Buttons
                      Row(
                        children: [
                          Expanded(
                            child: _buildActionButton(
                              context,
                              Icons.home_rounded,
                              'Dock',
                              Colors.blueGrey,
                              () {
                                VoiceService().speak("Moving to charging dock");
                                rover.goHome();
                              },
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: _buildActionButton(
                              context,
                              Icons.stop_circle_rounded,
                              'Stop',
                              Colors.red,
                              () {
                                VoiceService().speak("Stopping all movement");
                                rover.cancelCurrentMode();
                              },
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              // Footer Info
              Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  'Manual control is limited for safety. The robot will automatically avoid obstacles.',
                  style: theme.textTheme.bodySmall?.copyWith(fontStyle: FontStyle.italic),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildTelemetryHeader(BuildContext context, RoverProvider rover) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 24),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.primary.withOpacity(0.05),
        border: Border(bottom: BorderSide(color: Colors.grey.shade200)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _telemetryItem(context, Icons.battery_std, '${rover.batteryLevel}%', 'Power'),
          _telemetryItem(context, Icons.speed, '${rover.roverSpeed} m/s', 'Speed'),
          _telemetryItem(context, Icons.radar, 'Active', 'Obstacles'),
        ],
      ),
    );
  }

  Widget _telemetryItem(BuildContext context, IconData icon, String value, String label) {
    final theme = Theme.of(context);
    return Column(
      children: [
        Icon(icon, color: theme.colorScheme.primary, size: 20),
        const SizedBox(height: 4),
        Text(value, style: theme.textTheme.labelLarge?.copyWith(fontWeight: FontWeight.bold)),
        Text(label, style: theme.textTheme.bodySmall),
      ],
    );
  }

  Widget _buildDPad(BuildContext context, RoverProvider rover, ThemeData theme) {
    double buttonSize = 90;

    return Column(
      children: [
        // Up Arrow
        _dPadButton(theme, Icons.keyboard_arrow_up_rounded, "Forward", () {
          VoiceService().speak("Moving forward");
        }),

        const SizedBox(height: 12),

        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Left Arrow
            _dPadButton(theme, Icons.keyboard_arrow_left_rounded, "Left", () {
              VoiceService().speak("Turning left");
            }),

            SizedBox(width: buttonSize + 12), // Gap in center

            // Right Arrow
            _dPadButton(theme, Icons.keyboard_arrow_right_rounded, "Right", () {
              VoiceService().speak("Turning right");
            }),
          ],
        ),

        const SizedBox(height: 12),

        // Down Arrow
        _dPadButton(theme, Icons.keyboard_arrow_down_rounded, "Backward", () {
          VoiceService().speak("Moving backward");
        }),
      ],
    );
  }

  Widget _dPadButton(ThemeData theme, IconData icon, String label, VoidCallback onPressed) {
    return InkWell(
      onTap: onPressed,
      borderRadius: BorderRadius.circular(24),
      child: Container(
        width: 90,
        height: 90,
        decoration: BoxDecoration(
          color: theme.colorScheme.primary.withOpacity(0.1),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: theme.colorScheme.primary.withOpacity(0.3), width: 2),
        ),
        child: Icon(icon, size: 56, color: theme.colorScheme.primary),
      ),
    );
  }

  Widget _buildActionButton(BuildContext context, IconData icon, String label, Color color, VoidCallback onPressed) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 24, color: Colors.white),
      label: Text(label, style: const TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.bold)),
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        padding: const EdgeInsets.symmetric(vertical: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    );
  }
}
