import 'package:flutter/material.dart';

/// Represents the current state of the NovaCare rover.
enum RoverConnectionState { disconnected, connecting, connected, error }

enum RoverMode { idle, followingUser, navigatingHome, deliveringMedicine, emergency }

/// Manages rover state, telemetry data, and command dispatch.
class RoverProvider extends ChangeNotifier {
  // ─── Connection ─────────────────────────────────────────────────
  RoverConnectionState _connectionState = RoverConnectionState.disconnected;
  RoverConnectionState get connectionState => _connectionState;

  bool get isConnected => _connectionState == RoverConnectionState.connected;

  void setConnectionState(RoverConnectionState state) {
    _connectionState = state;
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
  int _batteryLevel = 85;
  int _heartRate = 72;
  String _roverLocation = 'Living Room';
  bool _isRoverOnline = true;
  double _roverSpeed = 0.0;
  double _temperature = 36.5;

  int get batteryLevel => _batteryLevel;
  int get heartRate => _heartRate;
  String get roverLocation => _roverLocation;
  bool get isRoverOnline => _isRoverOnline;
  double get roverSpeed => _roverSpeed;
  double get temperature => _temperature;

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

  // ─── Commands ───────────────────────────────────────────────────
  bool _isProcessingCommand = false;
  bool get isProcessingCommand => _isProcessingCommand;

  String? _lastCommandStatus;
  String? get lastCommandStatus => _lastCommandStatus;

  /// Send SOS emergency command
  Future<void> sendEmergency() async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.emergency;
    notifyListeners();

    // TODO: Send via BLE / Firebase
    await Future.delayed(const Duration(milliseconds: 500));

    _lastCommandStatus = 'Emergency alert sent!';
    _isProcessingCommand = false;
    notifyListeners();
  }

  /// Summon the rover to the user's current location.
  /// TODO(backend): publish a ROS goal pose via MQTT/Firebase to the robot.
  Future<void> summonRobot() async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.followingUser;
    notifyListeners();

    await Future.delayed(const Duration(milliseconds: 500));

    _lastCommandStatus = 'Summoning SERBOT-NC-001 to you';
    _isProcessingCommand = false;
    notifyListeners();
  }

  /// Request medication delivery
  Future<void> requestMedication() async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.deliveringMedicine;
    notifyListeners();

    await Future.delayed(const Duration(milliseconds: 500));

    _lastCommandStatus = 'Medication request sent to rover';
    _isProcessingCommand = false;
    notifyListeners();
  }

  /// Command rover to return home / dock
  Future<void> goHome() async {
    _isProcessingCommand = true;
    _currentMode = RoverMode.navigatingHome;
    notifyListeners();

    await Future.delayed(const Duration(milliseconds: 500));

    _lastCommandStatus = 'Rover returning to dock';
    _isProcessingCommand = false;
    notifyListeners();
  }

  /// Toggle follow-me mode
  Future<void> toggleFollowMe() async {
    _isProcessingCommand = true;
    notifyListeners();

    await Future.delayed(const Duration(milliseconds: 500));

    if (_currentMode == RoverMode.followingUser) {
      _currentMode = RoverMode.idle;
      _lastCommandStatus = 'Follow mode disabled';
    } else {
      _currentMode = RoverMode.followingUser;
      _lastCommandStatus = 'Follow mode enabled';
    }

    _isProcessingCommand = false;
    notifyListeners();
  }

  /// Cancel current mode and return to idle
  void cancelCurrentMode() {
    _currentMode = RoverMode.idle;
    _lastCommandStatus = 'Mode cancelled';
    notifyListeners();
  }

  // ─── Simulated Realtime Updates (for demo) ─────────────────────
  void startSimulatedUpdates() {
    _connectionState = RoverConnectionState.connected;
    _isRoverOnline = true;
    notifyListeners();

    // Simulate periodic telemetry updates
    Future.doWhile(() async {
      await Future.delayed(const Duration(seconds: 5));
      if (_connectionState != RoverConnectionState.connected) return false;

      // Slight random variations for demo
      _heartRate = 68 + (DateTime.now().second % 10);
      _batteryLevel = (_batteryLevel > 5) ? _batteryLevel : 85;
      _temperature = 36.2 + (DateTime.now().second % 8) * 0.1;
      notifyListeners();
      return true;
    });
  }
}
