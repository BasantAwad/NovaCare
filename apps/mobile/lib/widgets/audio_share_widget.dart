import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../services/summon_service.dart';

class AudioShareWidget extends StatefulWidget {
  const AudioShareWidget({super.key});

  @override
  State<AudioShareWidget> createState() => _AudioShareWidgetState();
}

class _AudioShareWidgetState extends State<AudioShareWidget> {
  String _status = '';

  Future<void> _pickAndSend() async {
    setState(() => _status = 'Picking file...');
    final result = await FilePicker.pickFiles(type: FileType.audio);
    if (result == null || result.files.isEmpty) {
      setState(() => _status = 'No file selected');
      return;
    }
    final file = File(result.files.first.path!);
    await _sendFile(file);
  }

  Future<void> _sendFile(File file) async {
    try {
      setState(() => _status = 'Reading file...');
      final bytes = await file.readAsBytes();
      final b64 = base64Encode(bytes);
      setState(() => _status = 'Sending...');
      final ok = await SummonService().sendPlayAudio(
        name: file.uri.pathSegments.last,
        audioBase64: b64,
        mime: 'audio/mpeg',
      );

      setState(() => _status = ok ? 'Sent' : 'Failed');
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
          label: const Text('Send Audio File'),
        ),
        if (_status.isNotEmpty) Padding(
          padding: const EdgeInsets.only(top: 8.0),
          child: Text(_status, style: const TextStyle(fontSize: 12)),
        ),
      ],
    );
  }
}
