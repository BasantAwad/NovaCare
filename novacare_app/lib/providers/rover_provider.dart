import 'dart:async';
import 'package:flutter/material.dart';

import '../services/robot_service.dart';
import '../services/emergency_service.dart';

/// Represents the current state of the NovaCare rover.
enum RoverConnectionState { disconnected, connecting, connected, error }

enum RoverMode {
  idle,
  followingUser,
  navigatingHome,
  navigating,
  deliveringMedicine,
  emergency,
}

/// Manages rover state, telemetry data, and command dispatch.
///
/// All commands use real HTTP calls to the JetAuto robot REST API.
/// No stubs, no Future.delayed placeholders.
class RoverProvider extends ChangeNotifier {
  final RobotService _robotService = RobotService();
  final EmergencyService _emergencyService = EmergencyService();

  // ─── Connection ─────────────────────────────────────────────────
  RoverConnectionState _connectionState = RoverConnectionState.disconnected;
  RoverConnectionState get connectionState => _connectionState;
  bool get isConnected => _connectionState == RoverConnectionState.connected;

  void setConnectionState(RoverConnectionState state) {
    _connectionState = state;
    notifyListeners();
  }

  // ─── Robot IP (configurable from Settings) ──────────────────────
  String _robotIp = '192.168.149.1'; // JetAuto default AP IP
  String get robotIp => _robotIp;

  void setRobotIp(String ip) {
    _robotIp = ip.trim();
    notifyListeners();
  }

  // ─── Rover Mode ─────────────────────────────────────────────────
  RoverMode _currentMode = RoverMode.idle;
  RoverMode get currentMode => _currentMode;

  void setMode(RoverMode mode) {
    _currentMode = mode;
    notifyListeners();
  }

  // ─── Telemetry ──────────────────────────────────────────────────
  int _batteryLevel = 0;
  int _heartRate = 0;
  String _roverLocation = 'Unknown';
  bool _isRoverOnline = false;
  double _roverSpeed = 0.0;
  double _temperature = 0.0;
  String _robotName = 'JETAUTO-NC-001';
  String _robotType = 'jetauto_pro';
  bool _cameraAvailable = false;
  bool _lidarAvailable = false;
  bool _sosActive = false;
  bool _fallDetectionActive = false;
  bool _gestureModeActive = false;
  bool _guardingModeActive = false;

  int get batteryLevel => _batteryLevel;
  int get heartRate => _heartRate;
  String get roverLocation => _roverLocation;
  bool get isRoverOnline => _isRoverOnline;
  double get roverSpeed => _roverSpeed;
  double get temperature => _temperature;
  String get robotName => _robotName;
  String get robotType => _robotType;
  bool get cameraAvailable => _cameraAvailable;
  bool get lidarAvailable => _lidarAvailable;
  bool get sosActive => _sosActive;
  bool get fallDetectionActive => _fallDetectionActive;
  bool get gestureModeActive => _gestureModeActive;
  bool get guardingModeActive => _guardingModeActive;

  void updateTelemetry({
    int? battery,
    int? heartRate,
    String? location,
    bool? online,
    double? speed,
    double? temperature,
  }) {
    if (battery != null) _batteryLevel = battery;
    if (heartRate != null) _heartRate = heartRate;
    if (location != null) _roverLocation = location;
    if (online != null) _isRoverOnline = online;
    if (speed != null) _roverSpeed = speed;
    if (temperature != null) _temperature = temperature;
    notifyListeners();
  }

  // ─── Navigation State ──────────────────────────────────────────
  NavigationProgress? _navProgress;
  NavigationProgress? get navProgress => _navProgress;

  List<Destination> _destinations = [];
  List<Destination> get destinations => _destinations;

  // ─── Commands ───────────────────────────────────────────────────
  bool _isProcessingCommand = false;
  bool get isProcessingCommand => _isProcessingCommand;

  String? _lastCommandStatus;
  String? get lastCommandStatus => _lastCommandStatus;

  // ═══════════════════════════════════════════════════════════════
  // SOS Emergency — real HTTP call to robot + Firebase
  // ═══════════════════════════════════════════════════════════════

  Future<void> sendEmergency() async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.emergency;
    notifyListeners();

    try {
      final result = await _emergencyService.triggerSOS(
        robotIp: _robotIp,
        userId: 'app_user',
        location: _roverLocation,
        heartRate: _heartRate > 0 ? _heartRate : null,
      );
      _sosActive = result.robotAlarmActive;
      _lastCommandStatus = result.message;
    } catch (e) {
      _lastCommandStatus = 'Emergency alert failed: $e';
      debugPrint('SOS Error: $e');
    }

    _isProcessingCommand = false;
    notifyListeners();
  }

  /// Cancel the active SOS alarm.
  Future<void> cancelEmergency() async {
    await _emergencyService.cancelSOS(robotIp: _robotIp);
    _sosActive = false;
    _currentMode = RoverMode.idle;
    _lastCommandStatus = 'Emergency cancelled';
    notifyListeners();
  }

  // ═══════════════════════════════════════════════════════════════
  // Manual Movement — DPad & Joystick
  // ═══════════════════════════════════════════════════════════════

  Future<void> moveRover(RobotMovement movement, String ip) async {
    try {
      await _robotService.sendMovementCommand(movement, ip);
    } catch (e) {
      debugPrint('Manual movement failed: $e');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Summon Robot — real Nav2 navigation
  // ═══════════════════════════════════════════════════════════════

  Future<void> summonRobot() async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.navigating;
    notifyListeners();

    try {
      final result = await _robotService.summonRobot(_robotIp);
      _lastCommandStatus = result?.success == true
          ? 'JetAuto navigating to you'
          : 'Summon failed — robot may be offline';
    } catch (e) {
      _lastCommandStatus = 'Summon failed: $e';
      debugPrint('Summon Error: $e');
    }

    _isProcessingCommand = false;
    notifyListeners();
  }

  // ═══════════════════════════════════════════════════════════════
  // Navigate to Destination — real Nav2
  // ═══════════════════════════════════════════════════════════════

  Future<void> navigateTo(String destination) async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.navigating;
    notifyListeners();

    try {
      final result =
          await _robotService.navigateToDestination(_robotIp, destination);
      _lastCommandStatus = result?.success == true
          ? 'Navigating to $destination'
          : 'Navigation failed';
    } catch (e) {
      _lastCommandStatus = 'Navigation failed: $e';
    }

    _isProcessingCommand = false;
    notifyListeners();
  }

  /// Navigate to explicit coordinates.
  Future<void> navigateToCoordinates(double x, double y, double theta) async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.navigating;
    notifyListeners();

    try {
      await _robotService.navigateToCoordinates(_robotIp, x, y, theta);
      _lastCommandStatus = 'Navigating to (${x.toStringAsFixed(1)}, ${y.toStringAsFixed(1)})';
    } catch (e) {
      _lastCommandStatus = 'Navigation failed: $e';
    }

    _isProcessingCommand = false;
    notifyListeners();
  }

  /// Cancel the active navigation goal.
  Future<void> cancelNavigation() async {
    await _robotService.cancelNavigation(_robotIp);
    _currentMode = RoverMode.idle;
    _lastCommandStatus = 'Navigation cancelled';
    _navProgress = null;
    notifyListeners();
  }

  /// Fetch the list of saved navigation destinations.
  Future<void> loadDestinations() async {
    _destinations = await _robotService.getDestinations(_robotIp);
    notifyListeners();
  }

  /// Save the robot's current position as a named destination.
  Future<void> saveCurrentAsDestination(String name, String label) async {
    await _robotService.saveDestination(_robotIp, name, label);
    await loadDestinations();
  }

  // ═══════════════════════════════════════════════════════════════
  // Medication Request — navigate to medicine cabinet + arm action
  // ═══════════════════════════════════════════════════════════════

  Future<void> requestMedication() async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.deliveringMedicine;
    notifyListeners();

    try {
      // Navigate to medication station
      final result =
          await _robotService.navigateToDestination(_robotIp, 'kitchen');
      _lastCommandStatus = result?.success == true
          ? 'Robot heading to medication station'
          : 'Medication request failed';
    } catch (e) {
      _lastCommandStatus = 'Medication request failed: $e';
    }

    _isProcessingCommand = false;
    notifyListeners();
  }

  // ═══════════════════════════════════════════════════════════════
  // Go Home / Dock — real Nav2 navigation
  // ═══════════════════════════════════════════════════════════════

  Future<void> goHome() async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.navigatingHome;
    notifyListeners();

    try {
      await _robotService.returnToDock(_robotIp);
      _lastCommandStatus = 'Robot returning to charging dock';
    } catch (e) {
      _lastCommandStatus = 'Return to dock failed: $e';
    }

    _isProcessingCommand = false;
    notifyListeners();
  }

  // ═══════════════════════════════════════════════════════════════
  // Follow Mode — real depth camera + LiDAR tracking
  // ═══════════════════════════════════════════════════════════════

  Future<void> toggleFollowMe() async {
    _isProcessingCommand = true;
    notifyListeners();

    try {
      if (_currentMode == RoverMode.followingUser) {
        await _robotService.stopFollowMode(_robotIp);
        _currentMode = RoverMode.idle;
        _lastCommandStatus = 'Follow mode disabled';
      } else {
        await _robotService.startFollowMode(_robotIp);
        _currentMode = RoverMode.followingUser;
        _lastCommandStatus = 'Follow mode enabled — robot tracking you';
      }
    } catch (e) {
      _lastCommandStatus = 'Follow mode toggle failed: $e';
    }

    _isProcessingCommand = false;
    notifyListeners();
  }

  // ═══════════════════════════════════════════════════════════════
  // Fall Detection — on-robot GPU-accelerated
  // ═══════════════════════════════════════════════════════════════

  Future<void> toggleFallDetection() async {
    try {
      if (_fallDetectionActive) {
        await _robotService.stopFallDetection(_robotIp);
        _fallDetectionActive = false;
        _lastCommandStatus = 'Fall detection stopped';
      } else {
        await _robotService.startFallDetection(_robotIp);
        _fallDetectionActive = true;
        _lastCommandStatus = 'Fall detection active (GPU-accelerated)';
      }
    } catch (e) {
      _lastCommandStatus = 'Fall detection toggle failed: $e';
    }
    notifyListeners();
  }

  /// Cancel current mode and return to idle
  void cancelCurrentMode() {
    if (_currentMode == RoverMode.navigating ||
        _currentMode == RoverMode.navigatingHome) {
      cancelNavigation();
    }
    _currentMode = RoverMode.idle;
    _lastCommandStatus = 'Mode cancelled';
    notifyListeners();
  }

  // ═══════════════════════════════════════════════════════════════
  // Gesture Control
  // ═══════════════════════════════════════════════════════════════

  Future<void> toggleGestureMode(String ip) async {
    _isProcessingCommand = true;
    notifyListeners();

    try {
      if (_gestureModeActive) {
        final ok = await _robotService.stopGestures(ip);
        if (ok) {
          _gestureModeActive = false;
          _lastCommandStatus = 'Gesture Control Disabled';
        } else {
          _lastCommandStatus = 'Failed to disable Gesture Control';
        }
      } else {
        final ok = await _robotService.startGestures(ip);
        if (ok) {
          _gestureModeActive = true;
          _lastCommandStatus = 'Gesture Control Enabled';
        } else {
          _lastCommandStatus = 'Failed to enable Gesture Control';
        }
      }
    } catch (e) {
      _lastCommandStatus = 'Gesture control disabled';
    }
    _isProcessingCommand = false;
    notifyListeners();
  }

  Future<void> toggleGuardingMode() async {
    _isProcessingCommand = true;
    notifyListeners();
    if (!_guardingModeActive) {
      final success = await _robotService.startGuarding(_robotIp);
      if (success) {
        _guardingModeActive = true;
        _lastCommandStatus = 'Guarding mode enabled';
      } else {
        _lastCommandStatus = 'Failed to enable guarding mode';
      }
    } else {
      await _robotService.stopGuarding(_robotIp);
      _guardingModeActive = false;
      _lastCommandStatus = 'Guarding mode disabled';
    }
    _isProcessingCommand = false;
    notifyListeners();
  }

  // ═══════════════════════════════════════════════════════════════
  // Real-time telemetry polling from JetAuto health endpoint
  // ═══════════════════════════════════════════════════════════════

  Timer? _telemetryTimer;

  void startTelemetryPolling() {
    _connectionState = RoverConnectionState.connecting;
    notifyListeners();

    // Initial fetch
    _fetchTelemetry();

    // Poll every 5 seconds
    _telemetryTimer = Timer.periodic(
      const Duration(seconds: 5),
      (_) => _fetchTelemetry(),
    );
  }

  void stopTelemetryPolling() {
    _telemetryTimer?.cancel();
    _telemetryTimer = null;
  }

  Future<void> _fetchTelemetry() async {
    try {
      final health = await _robotService.getHealth(_robotIp);
      if (health != null) {
        _connectionState = RoverConnectionState.connected;
        _isRoverOnline = true;
        _robotName = health.robotName;
        _robotType = health.robotType;
        _cameraAvailable = health.cameraAvailable;
        _lidarAvailable = health.lidarAvailable;
        _sosActive = health.sosActive;
        _fallDetectionActive = health.fallDetectionActive;

        if (health.vitals != null) {
          final v = health.vitals!;
          if (v['heart_rate'] != null) _heartRate = v['heart_rate'] as int;
          if (v['battery'] != null) _batteryLevel = v['battery'] as int;
        }

        if (health.isMoving) {
          _roverSpeed = 0.3; // approximate
        } else {
          _roverSpeed = 0.0;
        }

        // Update navigation progress if navigating
        if (_currentMode == RoverMode.navigating ||
            _currentMode == RoverMode.navigatingHome) {
          final navStatus =
              await _robotService.getNavigationStatus(_robotIp);
          _navProgress = navStatus;
          if (navStatus != null && !navStatus.active) {
            if (navStatus.status == 'succeeded') {
              _currentMode = RoverMode.idle;
              _lastCommandStatus = 'Navigation complete!';
            } else if (navStatus.status == 'failed') {
              _currentMode = RoverMode.idle;
              _lastCommandStatus = 'Navigation failed';
            }
          }
        }
      } else {
        _connectionState = RoverConnectionState.error;
        _isRoverOnline = false;
      }
    } catch (e) {
      debugPrint('Telemetry fetch error: $e');
      _connectionState = RoverConnectionState.error;
      _isRoverOnline = false;
    }
    notifyListeners();
  }

  /// Legacy compat — starts real telemetry polling.
  void startSimulatedUpdates() {
    startTelemetryPolling();
  }

  @override
  void dispose() {
    stopTelemetryPolling();
    _robotService.dispose();
    super.dispose();
  }
}
