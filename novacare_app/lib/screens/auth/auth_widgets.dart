import 'package:flutter/material.dart';

import '../../providers/auth_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_text_styles.dart';

/// Shared widgets used by both LoginScreen and SignupScreen.

class RoleChooser extends StatelessWidget {
  final UserRole selected;
  final ValueChanged<UserRole> onSelect;
  const RoleChooser({super.key, required this.selected, required this.onSelect});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(child: RoleCard(
          role: UserRole.patient,
          label: 'Patient',
          subtitle: 'I use the assistive robot',
          icon: Icons.accessibility_new_rounded,
          selected: selected == UserRole.patient,
          onTap: () => onSelect(UserRole.patient),
        )),
        const SizedBox(width: 12),
        Expanded(child: RoleCard(
          role: UserRole.caregiver,
          label: 'Caregiver',
          subtitle: 'I monitor a patient',
          icon: Icons.medical_services_outlined,
          selected: selected == UserRole.caregiver,
          onTap: () => onSelect(UserRole.caregiver),
        )),
      ],
    );
  }
}

class RoleCard extends StatelessWidget {
  final UserRole role;
  final String label;
  final String subtitle;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;

  const RoleCard({
    super.key,
    required this.role,
    required this.label,
    required this.subtitle,
    required this.icon,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final border = selected
        ? Border.all(color: AppColors.brandTeal, width: 2)
        : Border.all(color: AppColors.line2);
    final bg        = selected ? AppColors.brandAquaSoft : Theme.of(context).colorScheme.surface;
    final iconColor = selected ? AppColors.brandTeal : AppColors.inkMuted;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 16),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(Radii.md),
          border: border,
          boxShadow: selected ? Elevations.e1 : [],
        ),
        child: Column(
          children: [
            Icon(icon, size: 30, color: iconColor),
            const SizedBox(height: 8),
            Text(label, style: AppText.bodyStrong()),
            const SizedBox(height: 2),
            Text(subtitle, style: AppText.caption(), textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}

class AuthField extends StatelessWidget {
  final TextEditingController controller;
  final String label;
  final IconData icon;
  final TextInputType? keyboardType;
  final String? Function(String?)? validator;

  const AuthField({
    super.key,
    required this.controller,
    required this.label,
    required this.icon,
    this.keyboardType,
    this.validator,
  });

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      controller:   controller,
      keyboardType: keyboardType,
      validator:    validator,
      style:        AppText.body(),
      decoration: InputDecoration(
        labelText:  label,
        labelStyle: AppText.body(color: AppColors.inkMuted),
        prefixIcon: Icon(icon, size: 20, color: AppColors.inkMuted),
        filled:     true,
        fillColor:  Theme.of(context).colorScheme.surface,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Radii.sm),
          borderSide: const BorderSide(color: AppColors.line2),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Radii.sm),
          borderSide: const BorderSide(color: AppColors.line2),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Radii.sm),
          borderSide: const BorderSide(color: AppColors.brandTeal, width: 1.5),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Radii.sm),
          borderSide: const BorderSide(color: AppColors.danger),
        ),
      ),
    );
  }
}

class AuthPrimaryButton extends StatelessWidget {
  final String label;
  final bool loading;
  final VoidCallback onTap;

  const AuthPrimaryButton({
    super.key,
    required this.label,
    required this.loading,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: loading ? null : onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        height: 54,
        decoration: BoxDecoration(
          color: loading ? AppColors.inkLight : AppColors.brandTeal,
          borderRadius: BorderRadius.circular(Radii.sm),
          boxShadow: loading ? [] : Elevations.e1,
        ),
        alignment: Alignment.center,
        child: loading
            ? const SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2.5, color: Colors.white))
            : Text(label, style: AppText.bodyStrong(color: Colors.white)),
      ),
    );
  }
}
