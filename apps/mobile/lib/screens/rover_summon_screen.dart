import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';
import '../providers/summon_provider.dart';
import '../providers/ble_provider.dart';
import '../providers/settings_provider.dart';

class RoverSummonScreen extends StatefulWidget {
  const RoverSummonScreen({super.key});

  @override
  State<RoverSummonScreen> createState() => _RoverSummonScreenState();
}

class _RoverSummonScreenState extends State<RoverSummonScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  bool _summonStarted = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat();
    
    // Start summon after build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _startSummonSequence();
    });
  }

  void _startSummonSequence() async {
    if (_summonStarted) return;
    _summonStarted = true;
    
    final summonProvider = context.read<SummonProvider>();
    final bleProvider = context.read<BleProvider>();
    final settingsProvider = context.read<SettingsProvider>();
    
    // We assume the user has a user ID and BLE MAC from settings/auth
    // For now we use hardcoded or from BleProvider
    final bleMac = bleProvider.connectedDeviceId ?? 'AA:BB:CC:DD:EE:FF';
    
    // Connect WS if not connected
    if (!summonProvider.isConnected) {
      await summonProvider.connectToRobot(robotHost: settingsProvider.robotIp);
    }
    
    summonProvider.startSummon(userId: 'user_123', bleMac: bleMac);
    
    // Start listening to BLE RSSI changes to feed into SummonProvider
    bleProvider.addListener(_onBleUpdate);
  }
  
  void _onBleUpdate() {
    if (!mounted) return;
    final bleProvider = context.read<BleProvider>();
    final summonProvider = context.read<SummonProvider>();
    
    if (summonProvider.isActive) {
      summonProvider.sendRssiUpdate(bleProvider.rssi);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rover = context.watch<RoverProvider>();
    final summon = context.watch<SummonProvider>();

    return Scaffold(
      appBar: AppBar(title: const Text('Summon Robot')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Stack(
                alignment: Alignment.center,
                children: [
                  if (summon.isActive)
                    RotationTransition(
                      turns: _controller,
                      child: Container(
                        width: 250,
                        height: 250,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: theme.colorScheme.primary.withOpacity(0.2),
                            width: 2,
                          ),
                        ),
                        child: Align(
                          alignment: Alignment.topCenter,
                          child: Container(
                            width: 20,
                            height: 20,
                            decoration: BoxDecoration(
                              color: theme.colorScheme.primary,
                              shape: BoxShape.circle,
                            ),
                          ),
                        ),
                      ),
                    ),
                  Icon(
                    summon.phase == SummonPhase.arrived ? Icons.check_circle : Icons.person_pin_circle_rounded,
                    size: 80, 
                    color: summon.phase == SummonPhase.failed ? theme.colorScheme.error : theme.colorScheme.primary
                  ),
                ],
              ),
              const SizedBox(height: 40),
              Text(
                summon.statusMessage.isNotEmpty ? summon.statusMessage : 'Preparing to summon...',
                style: theme.textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              if (summon.isActive) ...[
                Text(
                  'Signal Strength: ${summon.signalStrengthLabel} (${summon.rssiCurrent} dBm)',
                  style: theme.textTheme.bodyLarge?.copyWith(
                    color: _getRssiColor(summon.rssiCurrent, theme),
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Trend: ${summon.rssiTrend}',
                  style: theme.textTheme.bodyMedium?.copyWith(color: Colors.grey),
                ),
              ],
              if (summon.phase == SummonPhase.failed) ...[
                Text(
                  summon.failureReason,
                  style: theme.textTheme.bodyLarge?.copyWith(color: theme.colorScheme.error),
                  textAlign: TextAlign.center,
                ),
              ],
              const SizedBox(height: 60),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: summon.isActive 
                      ? theme.colorScheme.error.withOpacity(0.1) 
                      : theme.colorScheme.primary.withOpacity(0.1),
                  foregroundColor: summon.isActive 
                      ? theme.colorScheme.error 
                      : theme.colorScheme.primary,
                ),
                onPressed: () {
                  if (summon.isActive) {
                    summon.cancelSummon(userId: 'user_123');
                  }
                  rover.cancelCurrentMode(context.read<SettingsProvider>().robotIp);
                  // Stop listening
                  context.read<BleProvider>().removeListener(_onBleUpdate);
                  summon.resetToIdle();
                  Navigator.pop(context);
                },
                child: Text(summon.isActive ? 'Cancel Summon' : 'Back'),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Color _getRssiColor(int rssi, ThemeData theme) {
    if (rssi >= -50) return Colors.green;
    if (rssi >= -70) return Colors.orange;
    if (rssi >= -85) return Colors.deepOrange;
    return theme.colorScheme.error;
  }
}
