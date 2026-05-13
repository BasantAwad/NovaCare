import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';

class RoverControlScreen extends StatelessWidget {
  const RoverControlScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rover = context.watch<RoverProvider>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Rover Controls'),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // Telemetry Header
          _buildTelemetryHeader(context, rover),

          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Directional Pad
                  _buildDPad(context, rover, theme),

                  const SizedBox(height: 60),

                  // Action Grid
                  GridView.count(
                    shrinkWrap: true,
                    crossAxisCount: 2,
                    mainAxisSpacing: 16,
                    crossAxisSpacing: 16,
                    childAspectRatio: 2.5,
                    children: [
                      _buildActionButton(
                        context,
                        Icons.home_rounded,
                        'Dock',
                        Colors.blueGrey,
                        () => rover.goHome(),
                      ),
                      _buildActionButton(
                        context,
                        Icons.stop_circle_rounded,
                        'Stop All',
                        Colors.red,
                        () => rover.cancelCurrentMode(),
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
      ),
    );
  }

  Widget _buildTelemetryHeader(BuildContext context, RoverProvider rover) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 32),
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
        Icon(icon, color: theme.colorScheme.primary, size: 24),
        const SizedBox(height: 4),
        Text(value, style: theme.textTheme.labelLarge?.copyWith(fontWeight: FontWeight.bold)),
        Text(label, style: theme.textTheme.bodySmall),
      ],
    );
  }

  Widget _buildDPad(BuildContext context, RoverProvider rover, ThemeData theme) {
    return Column(
      children: [
        _dPadButton(theme, Icons.arrow_upward_rounded, () => print('Forward')),
        const SizedBox(height: 16),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _dPadButton(theme, Icons.arrow_back_rounded, () => print('Left')),
            const SizedBox(height: 80, width: 80), // Center Gap
            _dPadButton(theme, Icons.arrow_forward_rounded, () => print('Right')),
          ],
        ),
        const SizedBox(height: 16),
        _dPadButton(theme, Icons.arrow_downward_rounded, () => print('Backward')),
      ],
    );
  }

  Widget _dPadButton(ThemeData theme, IconData icon, VoidCallback onPressed) {
    return GestureDetector(
      onTapDown: (_) => onPressed(),
      child: Container(
        width: 80,
        height: 80,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.grey.shade300, width: 2),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Icon(icon, size: 40, color: theme.colorScheme.primary),
      ),
    );
  }

  Widget _buildActionButton(BuildContext context, IconData icon, String label, Color color, VoidCallback onPressed) {
    return ElevatedButton.icon(
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        padding: EdgeInsets.zero,
        minimumSize: Size.zero,
      ),
      onPressed: onPressed,
      icon: Icon(icon, size: 20),
      label: Text(label, style: const TextStyle(fontSize: 14)),
    );
  }
}
