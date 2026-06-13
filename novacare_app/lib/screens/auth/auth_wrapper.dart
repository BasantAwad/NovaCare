import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../main_navigation.dart';
import 'login_screen.dart';
import '../caregiver/caregiver_navigation.dart';

class AuthWrapper extends StatelessWidget {
  const AuthWrapper({super.key});

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();

    if (!auth.isLoggedIn) {
      return const LoginScreen();
    } else if (auth.isCaregiver) {
      return const CaregiverNavigation();
    } else {
      return const MainNavigation();
    }
  }
}
