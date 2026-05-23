import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/alert_provider.dart';
import '../../theme/app_colors.dart';
import '../../theme/app_text_styles.dart';
import 'caregiver_home_screen.dart';
import 'caregiver_settings_screen.dart';

/// Caregiver tab shell — replaces MainNavigation for users with role=caregiver.
class CaregiverNavigation extends StatefulWidget {
  const CaregiverNavigation({super.key});

  @override
  State<CaregiverNavigation> createState() => _CaregiverNavigationState();
}

class _CaregiverNavigationState extends State<CaregiverNavigation> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    final unread = context.watch<AlertProvider>().unreadCount;

    return Scaffold(
      backgroundColor: AppColors.canvas,
      body: IndexedStack(
        index: _tab,
        children: const [
          CaregiverHomeScreen(),
          CaregiverSettingsScreen(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        backgroundColor: AppColors.paper,
        indicatorColor: AppColors.brandAquaSoft,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        destinations: [
          NavigationDestination(
            icon: Badge(
              isLabelVisible: unread > 0,
              label: Text('$unread', style: AppText.caption(color: Colors.white).copyWith(fontSize: 10)),
              child: const Icon(Icons.dashboard_outlined),
            ),
            selectedIcon: Badge(
              isLabelVisible: unread > 0,
              label: Text('$unread', style: AppText.caption(color: Colors.white).copyWith(fontSize: 10)),
              child: const Icon(Icons.dashboard_rounded),
            ),
            label: 'Dashboard',
          ),
          const NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings_rounded),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
