import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'providers/rover_provider.dart';
import 'providers/settings_provider.dart';
import 'providers/ble_provider.dart';
import 'providers/alert_provider.dart';
import 'providers/reminder_provider.dart';
import 'providers/translation_provider.dart';
import 'providers/summon_provider.dart';
import 'services/notification_service.dart';
import 'services/voice_service.dart';
import 'theme/app_theme.dart';
import 'screens/splash_screen.dart';
import 'screens/home_screen.dart';
import 'screens/reminders_screen.dart';
import 'screens/notifications_screen.dart';
import 'screens/settings_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  await NotificationService().init();
  await VoiceService().init();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  runApp(const NovaCareApp());
}

class NovaCareApp extends StatelessWidget {
  const NovaCareApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => SettingsProvider()),
        ChangeNotifierProvider(create: (_) => RoverProvider()),
        ChangeNotifierProvider(create: (_) => BleProvider()),
        ChangeNotifierProvider(create: (_) => AlertProvider()),
        ChangeNotifierProvider(create: (_) => ReminderProvider()),
        ChangeNotifierProvider(create: (_) => TranslationProvider()),
        ChangeNotifierProvider(create: (_) => SummonProvider()),
      ],
      child: Consumer2<SettingsProvider, TranslationProvider>(
        builder: (context, settings, translation, _) {
          // Sync voice enabled state
          VoiceService().setEnabled(settings.voiceFeedbackEnabled);

          return MaterialApp(
            title: 'NovaCare Assistant',
            debugShowCheckedModeBanner: false,
            locale: translation.locale,
            theme: AppTheme.lightTheme(),
            darkTheme: AppTheme.darkTheme(),
            themeMode: settings.themeMode,
            home: const MainNavigationHolder(),
          );
        },
      ),
    );
  }
}

class MainNavigationHolder extends StatefulWidget {
  const MainNavigationHolder({super.key});

  @override
  State<MainNavigationHolder> createState() => _MainNavigationHolderState();
}

class _MainNavigationHolderState extends State<MainNavigationHolder> {
  int _currentIndex = 0;

  @override
  void initState() {
    super.initState();
    // Auto-connect to robot on startup using saved settings
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final settings = context.read<SettingsProvider>();
      final summon = context.read<SummonProvider>();
      final ip = settings.robotIp;
      await summon.connectToRobot(robotHost: ip, port: 9999);
    });
  }

  final List<Widget> _screens = [
    const HomeScreen(),
    const RemindersScreen(),
    const NotificationsScreen(),
    const SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    final translation = context.watch<TranslationProvider>();

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() => _currentIndex = index);
          final labels = [
            translation.translate('home'),
            translation.translate('reminders'),
            translation.translate('alerts'),
            translation.translate('settings')
          ];
          VoiceService().speak(labels[index]);
        },
        type: BottomNavigationBarType.fixed,
        selectedItemColor: AppTheme.accentPink,
        unselectedItemColor: Colors.grey,
        showSelectedLabels: true,
        showUnselectedLabels: true,
        items: [
          BottomNavigationBarItem(
            icon: const Icon(Icons.home_rounded),
            label: translation.translate('home'),
          ),
          BottomNavigationBarItem(
            icon: const Icon(Icons.alarm_rounded),
            label: translation.translate('reminders'),
          ),
          BottomNavigationBarItem(
            icon: Consumer<AlertProvider>(
              builder: (context, alertProvider, child) {
                final count = alertProvider.unreadCount;
                if (count == 0) return const Icon(Icons.notifications_rounded);
                return Badge(
                  label: Text(count.toString()),
                  child: const Icon(Icons.notifications_rounded),
                );
              },
            ),
            label: translation.translate('alerts'),
          ),
          BottomNavigationBarItem(
            icon: const Icon(Icons.settings_rounded),
            label: translation.translate('settings'),
          ),
        ],
      ),
    );
  }
}
