import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../services/summon_service.dart';

/// Pick an audio file and send it as base64 to the robot speaker.
class AudioShareWidget extends StatefulWidget {
  const AudioShareWidget({super.key});

  @override
  State<AudioShareWidget> createState() => _AudioShareWidgetState();
}

class _AudioShareWidgetState extends State<AudioShareWidget> {
  String _status = '';

  Future<void> _pickAndSend() async {
    setState(() => _status = 'Picking file...');
    final result = await FilePicker.platform.pickFiles(type: FileType.audio);
    if (result == null || result.files.isEmpty) {
      setState(() => _status = 'Cancelled');
      return;
    }
    final path = result.files.first.path;
    if (path == null) { setState(() => _status = 'No path'); return; }

    try {
      setState(() => _status = 'Reading...');
      final bytes = await File(path).readAsBytes();
      setState(() => _status = 'Sending...');
      final ok = await SummonService().sendPlayAudio(
        name: result.files.first.name,
        audioBase64: base64Encode(bytes),
      );
      setState(() => _status = ok ? 'Sent!' : 'Failed');
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
    Future.delayed(const Duration(seconds: 2), () { if (mounted) setState(() => _status = ''); });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        ElevatedButton.icon(
          onPressed: _pickAndSend,
          icon: const Icon(Icons.upload_file),
          label: const Text('Send Audio to Robot'),
        ),
        if (_status.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(_status, style: const TextStyle(fontSize: 12)),
          ),
      ],
    );
  }
}
