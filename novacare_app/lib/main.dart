import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'providers/rover_provider.dart';
import 'providers/settings_provider.dart';
import 'providers/ble_provider.dart';
import 'theme/app_theme.dart';
import 'screens/splash_screen.dart';
import 'l10n/app_localizations.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock orientation to portrait for accessibility
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Initialize Firebase (uncomment when firebase_options.dart is generated)
  // await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

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
      ],
      child: Consumer<SettingsProvider>(
        builder: (context, settings, _) {
          return MaterialApp(
            title: 'NovaCare Assistant',
            debugShowCheckedModeBanner: false,
            themeMode: settings.themeMode,
            theme: AppTheme.lightTheme(),
            darkTheme: settings.isHighContrast
                ? AppTheme.highContrastTheme()
                : AppTheme.darkTheme(),
            locale: settings.locale,
            supportedLocales: const [
              Locale('en', ''),
              Locale('ar', ''),
            ],
            localizationsDelegates: AppLocalizations.localizationsDelegates,
            home: const SplashScreen(),
          );
        },
      ),
    );
  }
}
