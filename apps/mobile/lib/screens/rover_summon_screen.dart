import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';

class RoverSummonScreen extends StatefulWidget {
  const RoverSummonScreen({super.key});

  @override
  State<RoverSummonScreen> createState() => _RoverSummonScreenState();
}

class _RoverSummonScreenState extends State<RoverSummonScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat();
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
                  Icon(Icons.person_pin_circle_rounded, size: 80, color: theme.colorScheme.primary),
                ],
              ),
              const SizedBox(height: 60),
              Text(
                'Robot is finding you...',
                style: theme.textTheme.displaySmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Text(
                'SERBot is navigating to your current location using its LiDAR and tracking system.',
                style: theme.textTheme.bodyLarge?.copyWith(color: Colors.grey),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 80),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: theme.colorScheme.error.withOpacity(0.1),
                  foregroundColor: theme.colorScheme.error,
                ),
                onPressed: () {
                  rover.cancelCurrentMode();
                  Navigator.pop(context);
                },
                child: const Text('Cancel Summon'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
