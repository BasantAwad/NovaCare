import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import '../providers/settings_provider.dart';
import '../services/voice_service.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  late TextEditingController _nameCtrl;
  late TextEditingController _idCtrl;
  late TextEditingController _emailCtrl;
  late TextEditingController _emergencyCtrl;
  late TextEditingController _otherDisabilityCtrl;
  late TextEditingController _robotIpCtrl;
  String? _selectedDisability;
  String? _profileImagePath;

  static const _disabilityOptions = ['None', 'Mobility', 'Visual', 'Hearing', 'Cognitive', 'Other'];

  @override
  void initState() {
    super.initState();
    final s = context.read<SettingsProvider>();
    _nameCtrl       = TextEditingController(text: s.userName);
    _idCtrl         = TextEditingController(text: s.userId);
    _emailCtrl      = TextEditingController(text: s.email);
    _emergencyCtrl  = TextEditingController(text: s.emergencyContact);
    _robotIpCtrl    = TextEditingController(text: s.robotIp);
    _profileImagePath = s.profileImagePath;

    if (_disabilityOptions.contains(s.disabilityType)) {
      _selectedDisability = s.disabilityType;
      _otherDisabilityCtrl = TextEditingController();
    } else {
      _selectedDisability = 'Other';
      _otherDisabilityCtrl = TextEditingController(text: s.disabilityType);
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _idCtrl.dispose();
    _emailCtrl.dispose();
    _emergencyCtrl.dispose();
    _otherDisabilityCtrl.dispose();
    _robotIpCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final img = await showModalBottomSheet<XFile?>(
      context: context,
      builder: (_) => SafeArea(
        child: Wrap(children: [
          ListTile(
            leading: const Icon(Icons.photo_library),
            title: const Text('Gallery'),
            onTap: () async { Navigator.pop(context, await picker.pickImage(source: ImageSource.gallery)); },
          ),
          ListTile(
            leading: const Icon(Icons.camera_alt),
            title: const Text('Camera'),
            onTap: () async { Navigator.pop(context, await picker.pickImage(source: ImageSource.camera)); },
          ),
        ]),
      ),
    );
    if (img != null) setState(() => _profileImagePath = img.path);
  }

  void _save() {
    final disability = _selectedDisability == 'Other'
        ? _otherDisabilityCtrl.text
        : (_selectedDisability ?? 'None');

    context.read<SettingsProvider>().updateProfile(
      name: _nameCtrl.text,
      id: _idCtrl.text,
      email: _emailCtrl.text,
      emergencyContact: _emergencyCtrl.text,
      disability: disability,
      profileImagePath: _profileImagePath,
      robotIp: _robotIpCtrl.text.trim(),
    );

    VoiceService().speak('Profile updated successfully');
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Profile updated')),
    );
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('User Profile'),
        actions: [IconButton(onPressed: _save, icon: const Icon(Icons.check_rounded))],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            // Avatar
            GestureDetector(
              onTap: _pickImage,
              child: Stack(
                children: [
                  CircleAvatar(
                    radius: 60,
                    backgroundColor: theme.colorScheme.primary.withOpacity(0.1),
                    backgroundImage: _profileImagePath != null ? FileImage(File(_profileImagePath!)) : null,
                    child: _profileImagePath == null
                        ? Icon(Icons.person, size: 80, color: theme.colorScheme.primary)
                        : null,
                  ),
                  Positioned(
                    bottom: 0, right: 0,
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(color: theme.colorScheme.primary, shape: BoxShape.circle),
                      child: const Icon(Icons.edit, color: Colors.white, size: 20),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),
            _field('Full Name',         _nameCtrl,      Icons.person_outline),
            _field('User ID',           _idCtrl,        Icons.badge_outlined),
            _field('Email',             _emailCtrl,     Icons.email_outlined),
            _field('Emergency Contact', _emergencyCtrl, Icons.contact_phone_outlined),
            _field('Robot IP',          _robotIpCtrl,   Icons.router_outlined,
                hint: '192.168.137.150', keyboard: TextInputType.number),
            // Disability dropdown
            Container(
              margin: const EdgeInsets.only(bottom: 20),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(12),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _selectedDisability,
                  isExpanded: true,
                  hint: const Text('Select Disability Type'),
                  items: _disabilityOptions.map((v) =>
                      DropdownMenuItem(value: v, child: Text(v))).toList(),
                  onChanged: (v) => setState(() => _selectedDisability = v),
                ),
              ),
            ),
            if (_selectedDisability == 'Other')
              _field('Specify Disability', _otherDisabilityCtrl, Icons.edit_note_rounded),

            const SizedBox(height: 16),
            SwitchListTile(
              title: const Text('Voice Feedback'),
              subtitle: const Text('Robot speaks status updates'),
              value: context.watch<SettingsProvider>().voiceFeedbackEnabled,
              onChanged: (val) {
                context.read<SettingsProvider>().updateProfile(voiceFeedback: val);
                VoiceService().setEnabled(val);
                VoiceService().speak(val ? 'Voice feedback enabled' : 'Voice feedback disabled');
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _field(
    String label,
    TextEditingController ctrl,
    IconData icon, {
    String? hint,
    TextInputType? keyboard,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: TextField(
        controller: ctrl,
        keyboardType: keyboard,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          prefixIcon: Icon(icon),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        ),
      ),
    );
  }
}
