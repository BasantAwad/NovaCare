# NovaCare Mobile App

A Flutter-based mobile application for the NovaCare assistive rover project. This app serves as a remote control and monitoring dashboard for individuals with disabilities.

## Features

### 🎮 Core Interface (Big Button Layout)
- **🆘 SOS / Emergency** — Prominent red button that triggers the rover's local alarm and dispatches a notification to the guardian dashboard
- **💊 Medication Request** — Request the rover to navigate and deliver medication
- **🏠 Home / Dock** — Command the rover to return to its charging station
- **🚶 Follow Me** — Toggle mode where the rover tracks the user's phone GPS

### 📊 Real-time Monitoring
- Battery level with visual indicator
- Heart rate from on-board sensors
- Current rover location
- Body temperature reading
- Connection status (BLE / Cloud)

### ⚙️ Settings
- **User Profile** — Name, ID, disability type, voice feedback toggle
- **Language** — English and Arabic (Egyptian dialect support)
- **Theme** — Light Mode, Dark Mode, High-Contrast Mode
- **Privacy & Security** — Encryption info, granular permission controls
- **Connectivity** — BLE device scanning and connection management

### 🔗 Technical Integration
- **BLE (Bluetooth Low Energy)** — Short-range control via ESP32
- **Firebase Realtime Database** — Long-range commands and telemetry sync
- **Provider** — State management architecture

## Project Structure

```
novacare_app/
├── lib/
│   ├── main.dart                    # App entry point
│   ├── l10n/
│   │   └── app_localizations.dart   # English & Arabic translations
│   ├── providers/
│   │   ├── ble_provider.dart        # BLE connection state
│   │   ├── rover_provider.dart      # Rover telemetry & commands
│   │   └── settings_provider.dart   # Theme, language, profile
│   ├── screens/
│   │   ├── splash_screen.dart       # Animated splash
│   │   ├── home_screen.dart         # Main dashboard
│   │   └── settings_screen.dart     # Settings subscreen
│   ├── services/
│   │   ├── ble_service.dart         # flutter_blue_plus wrapper
│   │   └── firebase_service.dart    # Firebase RTDB wrapper
│   ├── theme/
│   │   ├── app_theme.dart           # Light/Dark/HC themes
│   │   └── app_colors.dart          # Semantic color system
│   └── widgets/
│       ├── action_button_widget.dart # Big accessible buttons
│       ├── connection_indicator.dart # App bar status chip
│       ├── status_bar_widget.dart    # Rover status banner
│       └── telemetry_card_widget.dart# Sensor data cards
├── android/
│   └── app/src/main/AndroidManifest.xml
├── ios/
│   └── Runner/Info.plist
└── pubspec.yaml
```

## Getting Started

### Prerequisites
- Flutter SDK >= 3.2.0
- Android Studio or Xcode
- A physical device (BLE doesn't work on emulators)

### Setup

```bash
# Navigate to the app directory
cd novacare_app

# Install dependencies
flutter pub get

# Run on connected device
flutter run

# Build APK
flutter build apk --release
```

### Firebase Setup
1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Run `flutterfire configure` to generate `firebase_options.dart`
3. Uncomment the Firebase initialization in `main.dart`

### ESP32 BLE Setup
The app expects the ESP32 to advertise the following UUIDs:
- **Service UUID**: `4fafc201-1fb5-459e-8fcc-c5c9c331914b`
- **Command Characteristic**: `beb5483e-36e1-4688-b7f5-ea07361b26a8`
- **Telemetry Characteristic**: `beb5483e-36e1-4688-b7f5-ea07361b26a9`

Telemetry format: `BAT:85|HR:72|LOC:Living Room|TEMP:36.5|SPD:0.5`

## Accessibility
- High-contrast theme with WCAG AAA compliance
- Large touch targets (72px minimum height)
- Voice feedback option for visually impaired users
- Portrait-locked orientation
- Haptic feedback on all actions
- RTL support for Arabic language

## Architecture

The app uses the **Provider** pattern for state management:
- `SettingsProvider` — App-wide settings (theme, locale, profile)
- `RoverProvider` — Rover state, telemetry, and command dispatch
- `BleProvider` — Bluetooth connection and device management

Communication channels:
- **BLE** → Short-range, low-latency commands via ESP32
- **Firebase** → Long-range commands, telemetry storage, guardian notifications
