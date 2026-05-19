import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';

class SosScreen extends StatefulWidget {
  const SosScreen({super.key});

  @override
  State<SosScreen> createState() => _SosScreenState();
}

class _SosScreenState extends State<SosScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  bool _isAlertSent = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(seconds: 1))
      ..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _triggerSos() async {
    setState(() => _isAlertSent = true);
    await context.read<RoverProvider>().sendEmergency();
    await Future.delayed(const Duration(seconds: 2));
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Caregivers have been notified!')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: _isAlertSent ? Colors.white : const Color(0xFF7F1D1D),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: IconThemeData(color: _isAlertSent ? Colors.black : Colors.white),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (!_isAlertSent) ...[
                FadeTransition(
                  opacity: _controller,
                  child: Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.3),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.warning_amber_rounded, color: Colors.white, size: 80),
                  ),
                ),
                const SizedBox(height: 40),
                Text(
                  'Emergency Alert',
                  style: theme.textTheme.displayLarge?.copyWith(color: Colors.white),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                Text(
                  'Press the button below to notify your caregivers immediately.',
                  style: theme.textTheme.bodyLarge?.copyWith(color: Colors.white70),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 60),
                GestureDetector(
                  onTap: _triggerSos,
                  child: Container(
                    width: 200,
                    height: 200,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(color: Colors.black.withOpacity(0.3), blurRadius: 30, spreadRadius: 5),
                      ],
                    ),
                    child: Center(
                      child: Text(
                        'SOS',
                        style: theme.textTheme.displayLarge?.copyWith(
                          color: Colors.red.shade900,
                          fontSize: 48,
                        ),
                      ),
                    ),
                  ),
                ),
              ] else ...[
                const Icon(Icons.check_circle_rounded, color: Colors.green, size: 100),
                const SizedBox(height: 32),
                Text('Help is on the way!', style: theme.textTheme.displayMedium, textAlign: TextAlign.center),
                const SizedBox(height: 16),
                Text(
                  'Your emergency contacts and the robot have been alerted.',
                  style: theme.textTheme.bodyLarge,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 48),
                ElevatedButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Return to Home'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
