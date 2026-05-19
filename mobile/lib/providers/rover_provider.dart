import 'package:flutter/material.dart';
import '../services/robot_service.dart';
import '../services/emergency_service.dart';

class RoverProvider extends ChangeNotifier {
  final RobotService _robotService       = RobotService();
  final EmergencyService _emergencyService = EmergencyService();

  RobotStatus _status        = RobotStatus.online;
  int         _batteryLevel  = 85;
  int         _heartRate     = 76;
  String      _roverLocation = 'Living Room';
  double      _roverSpeed    = 0.0;
  double      _temperature   = 36.8;
  bool        _isProcessing  = false;

  RobotStatus get status        => _status;
  int         get batteryLevel  => _batteryLevel;
  int         get heartRate     => _heartRate;
  String      get roverLocation => _roverLocation;
  double      get roverSpeed    => _roverSpeed;
  double      get temperature   => _temperature;
  bool        get isProcessing  => _isProcessing;
  bool        get isRoverOnline => _status != RobotStatus.offline;

  // Keep for backward-compat with existing screens
  bool        get isConnected   => isRoverOnline;

  RoverProvider() {
    _robotService.statusStream.listen((s) { _status = s; notifyListeners(); });
    _robotService.batteryStream.listen((b) { _batteryLevel = b; notifyListeners(); });
  }

  /// Feed parsed BLE telemetry (BAT|HR|LOC|TEMP|SPD) directly into provider.
  void updateFromTelemetry(Map<String, dynamic> data) {
    if (data['battery']     != null) _batteryLevel  = data['battery']     as int;
    if (data['heartRate']   != null) _heartRate     = data['heartRate']   as int;
    if (data['location']    != null) _roverLocation = data['location']    as String;
    if (data['temperature'] != null) _temperature   = (data['temperature'] as num).toDouble();
    if (data['speed']       != null) _roverSpeed    = (data['speed']       as num).toDouble();
    notifyListeners();
  }

  /// Real HTTP POST to port 9000 via RobotService.
  Future<void> moveRover(RobotMovement direction, String robotIp) async {
    _roverSpeed = direction == RobotMovement.stop ? 0.0 : 0.5;
    await _robotService.sendMovementCommand(direction, robotIp);
    notifyListeners();
  }

  Future<void> goHome(String robotIp) async {
    _isProcessing = true;
    notifyListeners();
    await _robotService.returnToDock(robotIp);
    _isProcessing = false;
    notifyListeners();
  }

  Future<void> cancelCurrentMode(String robotIp) async {
    _roverSpeed = 0.0;
    await _robotService.sendMovementCommand(RobotMovement.stop, robotIp);
    notifyListeners();
  }

  Future<void> sendEmergency() async {
    _isProcessing = true;
    notifyListeners();
    await _emergencyService.triggerSOS();
    _isProcessing = false;
    notifyListeners();
  }

  @override
  void dispose() {
    _robotService.dispose();
    super.dispose();
  }
}
