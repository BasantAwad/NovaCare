import 'package:flutter/material.dart';
import '../services/summon_service.dart';

/// A small button widget that triggers a play_sound request on the robot.
class PlaySoundButton extends StatefulWidget {
  final int frequency;
  final double duration;
  final String label;

  const PlaySoundButton({
    Key? key,
    this.frequency = 440,
    this.duration = 0.5,
    this.label = 'Play Sound',
  }) : super(key: key);

  @override
  _PlaySoundButtonState createState() => _PlaySoundButtonState();
}

class _PlaySoundButtonState extends State<PlaySoundButton> {
  bool _loading = false;
  String _status = '';

  Future<void> _onPressed() async {
    setState(() {
      _loading = true;
      _status = '';
    });

    final ok = await SummonService().sendPlaySound(
      frequency: widget.frequency,
      duration: widget.duration,
    );

    setState(() {
      _loading = false;
      _status = ok ? 'Sent' : 'Failed';
    });

    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _status = '');
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        ElevatedButton(
          onPressed: _loading ? null : _onPressed,
          child: _loading ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)) : Text(widget.label),
        ),
        if (_status.isNotEmpty) Padding(
          padding: const EdgeInsets.only(top: 6.0),
          child: Text(_status, style: const TextStyle(fontSize: 12)),
        ),
      ],
    );
  }
}
