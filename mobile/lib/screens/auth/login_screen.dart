import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_text_styles.dart';
import '../../widgets/nova_logo.dart';
import '../caregiver/caregiver_navigation.dart';
import '../main_navigation.dart';
import 'auth_widgets.dart';
import 'signup_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey      = GlobalKey<FormState>();
  final _emailCtrl    = TextEditingController();
  final _passwordCtrl = TextEditingController();
  bool _obscure       = true;
  bool _loading       = false;
  String? _error;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() { _loading = true; _error = null; });
    HapticFeedback.mediumImpact();
    try {
      await context.read<AuthProvider>().login(
        email:    _emailCtrl.text,
        password: _passwordCtrl.text,
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
      backgroundColor: AppColors.canvas,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 32),
              // ─── Brand ─────────────────────────────────────────────
              Center(
                child: Container(
                  width: 88, height: 88,
                  decoration: BoxDecoration(
                    color: AppColors.paper,
                    shape: BoxShape.circle,
                    boxShadow: Elevations.e2,
                  ),
                  padding: const EdgeInsets.all(12),
                  child: const NovaLogo(size: 64),
                ),
              ),
              const SizedBox(height: 20),
              Center(
                child: RichText(
                  text: TextSpan(
                    style: AppText.display2(),
                    children: const [
                      TextSpan(text: 'Nova', style: TextStyle(color: AppColors.inkTeal)),
                      TextSpan(text: 'Care', style: TextStyle(color: AppColors.accent)),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 6),
              Center(child: Text('Sign in to continue', style: AppText.body(color: AppColors.inkMuted))),
              const SizedBox(height: 40),

              Form(
                key: _formKey,
                child: Column(
                  children: [
                    AuthField(
                      controller: _emailCtrl,
                      label: 'Email address',
                      icon: Icons.email_outlined,
                      keyboardType: TextInputType.emailAddress,
                      validator: (v) {
                        if (v == null || v.trim().isEmpty) return 'Email is required';
                        if (!v.contains('@')) return 'Enter a valid email';
                        return null;
                      },
                    ),
                    const SizedBox(height: 14),
                    // Password field (custom, not AuthField so we can add show/hide)
                    TextFormField(
                      controller:  _passwordCtrl,
                      obscureText: _obscure,
                      style:       AppText.body(),
                      validator: (v) => (v == null || v.isEmpty) ? 'Password is required' : null,
                      decoration: InputDecoration(
                        labelText:  'Password',
                        labelStyle: AppText.body(color: AppColors.inkMuted),
                        prefixIcon: const Icon(Icons.lock_outline_rounded, size: 20, color: AppColors.inkMuted),
                        suffixIcon: IconButton(
                          icon: Icon(
                            _obscure ? Icons.visibility_outlined : Icons.visibility_off_outlined,
                            size: 20, color: AppColors.inkMuted,
                          ),
                          onPressed: () => setState(() => _obscure = !_obscure),
                        ),
                        filled: true, fillColor: AppColors.paper,
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                        border:        OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm), borderSide: const BorderSide(color: AppColors.line2)),
                        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm), borderSide: const BorderSide(color: AppColors.line2)),
                        focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm), borderSide: const BorderSide(color: AppColors.brandTeal, width: 1.5)),
                        errorBorder:   OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm), borderSide: const BorderSide(color: AppColors.danger)),
                      ),
                    ),
                  ],
                ),
              ),

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
              AuthPrimaryButton(label: 'Sign in', loading: _loading, onTap: _submit),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text("Don't have an account? ", style: AppText.body(color: AppColors.inkMuted)),
                  GestureDetector(
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute(builder: (_) => const SignupScreen()),
                    ),
                    child: Text('Sign up', style: AppText.bodyStrong(color: AppColors.brandTeal)),
                  ),
                ],
              ),
              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}
