import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';
import '../providers/settings_provider.dart';
import '../widgets/status_card.dart';
import '../widgets/quick_action_button.dart';
import 'sos_screen.dart';
import 'rover_control_screen.dart';

import 'rover_summon_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rover = context.watch<RoverProvider>();

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Good Morning,', style: theme.textTheme.bodyLarge),
                      Text('NovaCare User', style: theme.textTheme.displayLarge),
                    ],
                  ),
                  CircleAvatar(
                    radius: 28,
                    backgroundColor: theme.colorScheme.primary.withOpacity(0.1),
                    child: Icon(Icons.person_outline, color: theme.colorScheme.primary, size: 30),
                  ),
                ],
              ),

              const SizedBox(height: 32),

              // Robot Connection Status Card
              _buildConnectionCard(context, rover),

              const SizedBox(height: 32),

              // Quick Stats Grid
              Text('Vital Statistics', style: theme.textTheme.headlineMedium),
              const SizedBox(height: 16),
              GridView.count(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                crossAxisCount: 2,
                mainAxisSpacing: 16,
                crossAxisSpacing: 16,
                childAspectRatio: 1.4,
                children: [
                  StatusCard(
                    icon: Icons.favorite,
                    label: 'Heart Rate',
                    value: '${rover.heartRate} BPM',
                    color: Colors.redAccent,
                  ),
                  StatusCard(
                    icon: Icons.battery_charging_full,
                    label: 'Robot Battery',
                    value: '${rover.batteryLevel}%',
                    color: Colors.green,
                  ),
                  StatusCard(
                    icon: Icons.location_on,
                    label: 'Robot Location',
                    value: rover.roverLocation,
                    color: Colors.blueAccent,
                  ),
                  StatusCard(
                    icon: Icons.thermostat,
                    label: 'Temperature',
                    value: '${rover.temperature}°C',
                    color: Colors.orangeAccent,
                  ),
                ],
              ),

              const SizedBox(height: 32),

              // Emergency Section
              Text('Emergency & Assistance', style: theme.textTheme.headlineMedium),
              const SizedBox(height: 16),
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

              const SizedBox(height: 16),

              // Robot Summon Section
              QuickActionButton(
                icon: Icons.smart_toy_rounded,
                label: 'SUMMON ROBOT',
                subtitle: 'Call the assistant to your location',
                color: theme.colorScheme.primary,
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const RoverSummonScreen()),
                ),
              ),

              const SizedBox(height: 16),

              // Remote Controls Entry
              QuickActionButton(
                icon: Icons.gamepad_rounded,
                label: 'ROVER CONTROLS',
                subtitle: 'Manual movement & docking',
                color: Colors.indigo,
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const RoverControlScreen()),
                ),
              ),

              const SizedBox(height: 80), // Padding for bottom
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildConnectionCard(BuildContext context, RoverProvider rover) {
    final theme = Theme.of(context);
    final isOnline = rover.isRoverOnline;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: isOnline ? Colors.green.shade50 : Colors.red.shade50,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: isOnline ? Colors.green.shade200 : Colors.red.shade200,
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: isOnline ? Colors.green : Colors.red,
              shape: BoxShape.circle,
            ),
            child: Icon(
              isOnline ? Icons.check_circle : Icons.error,
              color: Colors.white,
            ),
          ),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isOnline ? 'Rover is Online' : 'Rover is Offline',
                style: theme.textTheme.labelLarge?.copyWith(
                  color: isOnline ? Colors.green.shade800 : Colors.red.shade800,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                isOnline ? 'Active and monitoring' : 'Check connection status',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: isOnline ? Colors.green.shade700 : Colors.red.shade700,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
