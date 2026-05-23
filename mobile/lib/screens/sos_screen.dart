import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../providers/alert_provider.dart';
import '../providers/auth_provider.dart';
import '../providers/rover_provider.dart';
import '../services/emergency_service.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// Full-screen SOS screen.
///
/// When the patient presses the button:
///   1. Alarm notification fires immediately (sound + full-screen intent).
///   2. SOS alert is added to AlertProvider → caregiver dashboard.
///   3. SMS app opens with a pre-filled emergency message to emergency contact.
class SosScreen extends StatefulWidget {
  const SosScreen({super.key});

  @override
  State<SosScreen> createState() => _SosScreenState();
}

class _SosScreenState extends State<SosScreen> with SingleTickerProviderStateMixin {
  late AnimationController _pulse;
  bool _alertSent  = false;
  bool _loading    = false;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(vsync: this, duration: const Duration(milliseconds: 900))
      ..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  Future<void> _triggerSos() async {
    if (_loading || _alertSent) return;
    HapticFeedback.heavyImpact();
    setState(() => _loading = true);

    final auth         = context.read<AuthProvider>();
    final alertProvider = context.read<AlertProvider>();
    final rover        = context.read<RoverProvider>();

    await EmergencyService().triggerSOS(
      context:       context,
      auth:          auth,
      alertProvider: alertProvider,
      rover:         rover,
    );

    if (mounted) setState(() { _alertSent = true; _loading = false; });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _alertSent ? Colors.white : const Color(0xFF7F1D1D),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: Icon(
            Icons.arrow_back_ios_new_rounded,
            color: _alertSent ? AppColors.inkNavy : Colors.white,
            size: 18,
          ),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: _alertSent ? _SentState(context: context) : _ReadyState(
            pulse:   _pulse,
            loading: _loading,
            onPress: _triggerSos,
          ),
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  Ready state — pulsing SOS button
// ════════════════════════════════════════════════════════════════════
class _ReadyState extends StatelessWidget {
  final AnimationController pulse;
  final bool loading;
  final VoidCallback onPress;

  const _ReadyState({required this.pulse, required this.loading, required this.onPress});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        FadeTransition(
          opacity: pulse,
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
          style: AppText.display1(color: Colors.white),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 12),
        Text(
          'Press the button to notify your caregiver immediately.\n'
          'An alarm will sound and a text message will be sent.',
          style: AppText.body(color: Colors.white70),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 56),
        GestureDetector(
          onTap: loading ? null : onPress,
          child: AnimatedBuilder(
            animation: pulse,
            builder: (_, __) => Container(
              width: 200,
              height: 200,
              decoration: BoxDecoration(
                color: Colors.white,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.red.withOpacity(0.3 + 0.3 * pulse.value),
                    blurRadius: 40 + 20 * pulse.value,
                    spreadRadius: 8 + 8 * pulse.value,
                  ),
                ],
              ),
              child: Center(
                child: loading
                    ? const CircularProgressIndicator(color: Colors.red, strokeWidth: 3)
                    : Text(
                        'SOS',
                        style: AppText.display1(color: Colors.red).copyWith(fontSize: 52),
                      ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 32),
        Text(
          'Hold tight — press once to alert your caregiver.',
          style: AppText.caption(color: Colors.white60),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}

// ════════════════════════════════════════════════════════════════════
//  Sent state — confirmation + extra options
// ════════════════════════════════════════════════════════════════════
class _SentState extends StatelessWidget {
  final BuildContext context;
  const _SentState({required this.context});

  @override
  Widget build(BuildContext ctx) {
    final auth = ctx.read<AuthProvider>();
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.check_circle_rounded, color: Colors.green, size: 100),
        const SizedBox(height: 28),
        Text(
          'Help is on the way!',
          style: AppText.display2(),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 12),
        Text(
          auth.ecName.isNotEmpty
              ? 'Alert sent to ${auth.ecName}. Stay calm — they will reach you shortly.'
              : 'Your emergency contact and caregiver have been alerted. Stay calm.',
          style: AppText.body(color: AppColors.inkMuted),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 32),
        // Extra: send email to team as well
        OutlinedButton.icon(
          icon: const Icon(Icons.email_outlined),
          label: const Text('Also send email alert'),
          style: OutlinedButton.styleFrom(
            foregroundColor: AppColors.brandTeal,
            side: const BorderSide(color: AppColors.brandTeal),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Radii.sm)),
          ),
          onPressed: () => EmergencyService().sendEmergencyEmail(
            patientName: auth.name.isNotEmpty ? auth.name : 'Patient',
          ),
        ),
        const SizedBox(height: 16),
        ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.brandTeal,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Radii.sm)),
            minimumSize: const Size(double.infinity, 50),
          ),
          onPressed: () => Navigator.pop(ctx),
          child: const Text('Return to Home'),
        ),
      ],
    );
  }
}
