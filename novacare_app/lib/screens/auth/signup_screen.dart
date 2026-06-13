import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_text_styles.dart';
import '../caregiver/caregiver_navigation.dart';
import '../main_navigation.dart';
import 'auth_widgets.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({super.key});

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  final _formKey      = GlobalKey<FormState>();
  final _nameCtrl     = TextEditingController();
  final _emailCtrl    = TextEditingController();
  final _passwordCtrl = TextEditingController();
  final _confirmCtrl  = TextEditingController();
  final _ecPhoneCtrl  = TextEditingController();
  final _ecNameCtrl   = TextEditingController();
  UserRole _role      = UserRole.patient;
  bool _obscure       = true;
  bool _loading       = false;
  String? _error;

  @override
  void dispose() {
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    _confirmCtrl.dispose();
    _ecPhoneCtrl.dispose();
    _ecNameCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() { _loading = true; _error = null; });
    HapticFeedback.mediumImpact();
    try {
      await context.read<AuthProvider>().signup(
        name:     _nameCtrl.text,
        email:    _emailCtrl.text,
        password: _passwordCtrl.text,
        role:     _role,
        ecPhone:  _ecPhoneCtrl.text,
        ecName:   _ecNameCtrl.text,
      );
      if (!mounted) return;
      _goHome();
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _goHome() {
    final auth = context.read<AuthProvider>();
    final dest = auth.isCaregiver
        ? const CaregiverNavigation()
        : const MainNavigation();
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => dest),
      (_) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, color: AppColors.inkNavy, size: 18),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Text('Create account', style: AppText.appBarTitle()),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 12),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [

                // ─── Role selector ────────────────────────────────────
                Text('I am a…', style: AppText.eyebrow()),
                const SizedBox(height: 10),
                RoleChooser(selected: _role, onSelect: (r) => setState(() => _role = r)),
                const SizedBox(height: 24),

                // ─── Basic info ───────────────────────────────────────
                Text('Your details', style: AppText.eyebrow()),
                const SizedBox(height: 10),
                AuthField(
                  controller: _nameCtrl,
                  label: 'Full name',
                  icon: Icons.person_outline_rounded,
                  validator: (v) => (v == null || v.trim().isEmpty) ? 'Required' : null,
                ),
                const SizedBox(height: 14),
                AuthField(
                  controller: _emailCtrl,
                  label: 'Email address',
                  icon: Icons.email_outlined,
                  keyboardType: TextInputType.emailAddress,
                  validator: (v) {
                    if (v == null || v.trim().isEmpty) return 'Required';
                    if (!v.contains('@')) return 'Enter a valid email';
                    return null;
                  },
                ),
                const SizedBox(height: 14),

                // ─── Password ─────────────────────────────────────────
                TextFormField(
                  controller:  _passwordCtrl,
                  obscureText: _obscure,
                  style:       AppText.body(),
                  validator: (v) {
                    if (v == null || v.isEmpty) return 'Required';
                    if (v.length < 6) return 'At least 6 characters';
                    return null;
                  },
                  decoration: _pwDecoration('Password', suffixToggle: true),
                ),
                const SizedBox(height: 14),
                TextFormField(
                  controller:  _confirmCtrl,
                  obscureText: _obscure,
                  style:       AppText.body(),
                  validator: (v) {
                    if (v == null || v.isEmpty) return 'Required';
                    if (v != _passwordCtrl.text) return 'Passwords do not match';
                    return null;
                  },
                  decoration: _pwDecoration('Confirm password'),
                ),

                // ─── Emergency contact (patients only) ────────────────
                if (_role == UserRole.patient) ...[
                  const SizedBox(height: 24),
                  Row(
                    children: [
                      Text('Emergency contact', style: AppText.eyebrow()),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: AppColors.danger2,
                          borderRadius: BorderRadius.circular(Radii.pill),
                        ),
                        child: Text('SOS alerts sent here', style: AppText.caption(color: AppColors.danger)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  AuthField(
                    controller: _ecNameCtrl,
                    label: 'Contact name (caregiver / family)',
                    icon: Icons.people_outline_rounded,
                    validator: (v) => (v == null || v.trim().isEmpty) ? 'Required for SOS' : null,
                  ),
                  const SizedBox(height: 14),
                  AuthField(
                    controller: _ecPhoneCtrl,
                    label: 'Phone number (+ country code)',
                    icon: Icons.phone_outlined,
                    keyboardType: TextInputType.phone,
                    validator: (v) => (v == null || v.trim().isEmpty) ? 'Required for SOS' : null,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'When you press SOS, a text message will be sent to this number immediately.',
                    style: AppText.caption(color: AppColors.inkMuted),
                  ),
                ],

                // ─── Caregiver note ────────────────────────────────────
                if (_role == UserRole.caregiver) ...[
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: AppColors.brandAquaSoft,
                      borderRadius: BorderRadius.circular(Radii.md),
                      border: Border.all(color: AppColors.brandAqua),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.info_outline_rounded, color: AppColors.brandTeal, size: 18),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            'As a caregiver you will receive real-time SOS alerts from your patients in your dashboard.',
                            style: AppText.caption(color: AppColors.inkTeal),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],

                if (_error != null) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: AppColors.danger2,
                      borderRadius: BorderRadius.circular(Radii.sm),
                    ),
                    child: Text(_error!, style: AppText.caption(color: AppColors.danger)),
                  ),
                ],

                const SizedBox(height: 28),
                AuthPrimaryButton(label: 'Create account', loading: _loading, onTap: _submit),
                const SizedBox(height: 32),
              ],
            ),
          ),
        ),
      ),
    );
  }

  InputDecoration _pwDecoration(String label, {bool suffixToggle = false}) {
    return InputDecoration(
      labelText:  label,
      labelStyle: AppText.body(color: AppColors.inkMuted),
      prefixIcon: const Icon(Icons.lock_outline_rounded, size: 20, color: AppColors.inkMuted),
      suffixIcon: suffixToggle
          ? IconButton(
              icon: Icon(
                _obscure ? Icons.visibility_outlined : Icons.visibility_off_outlined,
                size: 20, color: AppColors.inkMuted,
              ),
              onPressed: () => setState(() => _obscure = !_obscure),
            )
          : null,
      filled: true, fillColor: Theme.of(context).colorScheme.surface,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      border:        OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm), borderSide: const BorderSide(color: AppColors.line2)),
      enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm), borderSide: const BorderSide(color: AppColors.line2)),
      focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm), borderSide: const BorderSide(color: AppColors.brandTeal, width: 1.5)),
      errorBorder:   OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm), borderSide: const BorderSide(color: AppColors.danger)),
    );
  }
}
