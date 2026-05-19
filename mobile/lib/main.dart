import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'providers/rover_provider.dart';
import 'providers/settings_provider.dart';
import 'providers/ble_provider.dart';
import 'providers/alert_provider.dart';
import 'providers/reminder_provider.dart';
import 'providers/summon_provider.dart';
import 'providers/translation_provider.dart';
import 'services/notification_service.dart';
import 'services/voice_service.dart';
import 'theme/app_theme.dart';
import 'screens/splash_screen.dart';
import 'l10n/app_localizations.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await NotificationService().init();
  await VoiceService().init();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );

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
        ChangeNotifierProvider(create: (ctx) {
          final ble = BleProvider();
          // Wire BLE ESP32 telemetry → RoverProvider without a circular dependency.
          ble.onTelemetry = (data) {
            // ctx is the MultiProvider context — RoverProvider is already registered above.
            try {
              ctx.read<RoverProvider>().updateFromTelemetry(data);
            } catch (_) {}
          };
          return ble;
        }),
        ChangeNotifierProvider(create: (_) => AlertProvider()),
        ChangeNotifierProvider(create: (_) => ReminderProvider()),
        ChangeNotifierProvider(create: (_) => SummonProvider()),
        ChangeNotifierProvider(create: (_) => TranslationProvider()),
      ],
      child: Consumer2<SettingsProvider, TranslationProvider>(
        builder: (context, settings, translation, _) {
          // Keep VoiceService in sync with user preference.
          VoiceService().setEnabled(settings.voiceFeedbackEnabled);

          // Auto-connect Summon WebSocket after first frame.
          WidgetsBinding.instance.addPostFrameCallback((_) async {
            final summon = context.read<SummonProvider>();
            if (!summon.isConnected) {
              await summon.connectToRobot(robotHost: settings.robotIp);
            }
          });

          return MaterialApp(
            title: 'NovaCare',
            debugShowCheckedModeBanner: false,
            theme: settings.isHighContrast
                ? AppTheme.highContrastTheme()
                : AppTheme.lightTheme(),
            darkTheme: settings.isHighContrast
                ? AppTheme.highContrastTheme()
                : AppTheme.roverDarkTheme(),
            themeMode: settings.themeMode,
            locale: settings.locale,
            supportedLocales: const [
              Locale('en', ''),
              Locale('ar', ''),
              Locale('es', ''),
              Locale('fr', ''),
            ],
            localizationsDelegates: AppLocalizations.localizationsDelegates,
            builder: (context, child) {
              final scale = settings.largeTextEnabled ? 1.5 : 1.0;
              return MediaQuery(
                data: MediaQuery.of(context).copyWith(textScaler: TextScaler.linear(scale)),
                child: child!,
              );
            },
            home: const SplashScreen(),
          );
        },
      ),
    );
  }
}
