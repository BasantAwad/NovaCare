import 'package:flutter/material.dart';

import '../theme/app_colors.dart';
import '../widgets/nc_bottom_nav.dart';
import 'home_screen.dart';
import 'reminders_screen.dart';
import 'companion_screen.dart';
import 'alerts_screen.dart';
import 'settings_screen.dart';

/// Top-level tabbed shell.
///
/// Uses an [IndexedStack] so each tab keeps its scroll position when the
/// user switches away and back. The Rover-controls screen is NOT a tab —
/// it gets pushed full-screen on top of HomeScreen.
class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  /// Imperatively switch tabs from anywhere in the widget tree.
  static void switchTab(BuildContext context, NcTab tab) {
    final state = context.findAncestorStateOfType<_MainNavigationState>();
    state?._setTab(tab);
  }

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  NcTab _active = NcTab.home;

  static const _order = [
    NcTab.home,
    NcTab.reminders,
    NcTab.companion,
    NcTab.alerts,
    NcTab.settings,
  ];

  void _setTab(NcTab tab) => setState(() => _active = tab);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: IndexedStack(
        index: _order.indexOf(_active),
        children: const [
          HomeScreen(),
          RemindersScreen(),
          CompanionScreen(),
          AlertsScreen(),
          SettingsScreen(),
        ],
      ),
      bottomNavigationBar: NcBottomNav(
        active: _active,
        onChange: _setTab,
      ),
    );
  }
}
