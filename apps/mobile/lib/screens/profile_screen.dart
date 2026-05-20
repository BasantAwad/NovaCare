import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import '../providers/settings_provider.dart';
import '../providers/translation_provider.dart';
import '../services/voice_service.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  late TextEditingController _nameController;
  late TextEditingController _idController;
  late TextEditingController _emailController;
  late TextEditingController _emergencyController;
  late TextEditingController _otherDisabilityController;
  String? _selectedDisability;
  String? _profileImagePath;

  final List<String> _disabilityOptions = [
    'None',
    'Mobility',
    'Visual',
    'Hearing',
    'Cognitive',
    'Other'
  ];

  @override
  void initState() {
    super.initState();
    final settings = context.read<SettingsProvider>();
    _nameController = TextEditingController(text: settings.userName);
    _idController = TextEditingController(text: settings.userId);
    _emailController = TextEditingController(text: settings.email);
    _emergencyController = TextEditingController(text: settings.emergencyContact);

    if (_disabilityOptions.contains(settings.disabilityType)) {
      _selectedDisability = settings.disabilityType;
      _otherDisabilityController = TextEditingController();
    } else {
      _selectedDisability = 'Other';
      _otherDisabilityController = TextEditingController(text: settings.disabilityType);
    }

    _profileImagePath = settings.profileImagePath;
  }

  @override
  void dispose() {
    _nameController.dispose();
    _idController.dispose();
    _emailController.dispose();
    _emergencyController.dispose();
    _otherDisabilityController.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await showModalBottomSheet<XFile?>(
      context: context,
      builder: (context) => SafeArea(
        child: Wrap(
          children: [
            ListTile(
              leading: const Icon(Icons.photo_library),
              title: const Text('Gallery'),
              onTap: () async {
                final img = await picker.pickImage(source: ImageSource.gallery);
                Navigator.pop(context, img);
              },
            ),
            ListTile(
              leading: const Icon(Icons.camera_alt),
              title: const Text('Camera'),
              onTap: () async {
                final img = await picker.pickImage(source: ImageSource.camera);
                Navigator.pop(context, img);
              },
            ),
          ],
        ),
      ),
    );

    if (image != null) {
      setState(() {
        _profileImagePath = image.path;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final translation = context.watch<TranslationProvider>();

    return Scaffold(
      appBar: AppBar(
        title: Text(translation.translate('profile')),
        actions: [
          IconButton(
            onPressed: _saveProfile,
            icon: const Icon(Icons.check_rounded),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            GestureDetector(
              onTap: _pickImage,
              child: Stack(
                children: [
                  CircleAvatar(
                    radius: 60,
                    backgroundColor: Theme.of(context).colorScheme.primary.withOpacity(0.1),
                    backgroundImage: _profileImagePath != null ? FileImage(File(_profileImagePath!)) : null,
                    child: _profileImagePath == null
                      ? Icon(Icons.person, size: 80, color: Theme.of(context).colorScheme.primary)
                      : null,
                  ),
                  Positioned(
                    bottom: 0,
                    right: 0,
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.primary,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.edit, color: Colors.white, size: 20),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),
            _buildTextField('Full Name', _nameController, Icons.person_outline),
            _buildTextField('User ID', _idController, Icons.badge_outlined),
            _buildTextField('Email', _emailController, Icons.email_outlined),
            _buildTextField('Emergency Contact', _emergencyController, Icons.contact_phone_outlined),

            // Disability Dropdown
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
                  items: _disabilityOptions.map((String value) {
                    return DropdownMenuItem<String>(
                      value: value,
                      child: Text(value),
                    );
                  }).toList(),
                  onChanged: (newValue) {
                    setState(() {
                      _selectedDisability = newValue;
                    });
                  },
                ),
              ),
            ),

            if (_selectedDisability == 'Other')
              _buildTextField('Specify Disability', _otherDisabilityController, Icons.edit_note_rounded),

            const SizedBox(height: 16),
            SwitchListTile(
              title: Text(translation.translate('voice_feedback')),
              subtitle: const Text('Robot speaks status updates'),
              value: context.watch<SettingsProvider>().voiceFeedbackEnabled,
              onChanged: (val) {
                context.read<SettingsProvider>().updateProfile(voiceFeedback: val);
                VoiceService().setEnabled(val);
                VoiceService().speak(val ? "Voice feedback enabled" : "Voice feedback disabled");
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextField(String label, TextEditingController controller, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        ),
      ),
    );
  }

  void _saveProfile() {
    String finalDisability = _selectedDisability == 'Other'
        ? _otherDisabilityController.text
        : (_selectedDisability ?? 'None');

    context.read<SettingsProvider>().updateProfile(
      name: _nameController.text,
      id: _idController.text,
      email: _emailController.text,
      emergencyContact: _emergencyController.text,
      disability: finalDisability,
      profileImagePath: _profileImagePath,
    );

    VoiceService().speak("Profile updated successfully");
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Profile updated successfully')),
    );
    Navigator.pop(context);
  }
}
