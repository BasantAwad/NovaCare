import 'package:flutter/material.dart';
import '../services/robot_service.dart';
import '../services/emergency_service.dart';

class RoverProvider extends ChangeNotifier {
  final RobotService _robotService = RobotService();
  final EmergencyService _emergencyService = EmergencyService();

  // ─── State ──────────────────────────────────────────────────────
  RobotStatus _status = RobotStatus.online;
  int _batteryLevel = 85;
  int _heartRate = 76;
  String _roverLocation = 'Living Room';
  double _roverSpeed = 0.0;
  double _temperature = 36.8;
  bool _isProcessing = false;

  // ─── Getters ────────────────────────────────────────────────────
  RobotStatus get status => _status;
  int get batteryLevel => _batteryLevel;
  int get heartRate => _heartRate;
  String get roverLocation => _roverLocation;
  double get roverSpeed => _roverSpeed;
  double get temperature => _temperature;
  bool get isProcessing => _isProcessing;
  bool get isRoverOnline => _status != RobotStatus.offline;

  RoverProvider() {
    _init();
  }

  void _init() {
    _robotService.statusStream.listen((status) {
      _status = status;
      notifyListeners();
    });

    _robotService.batteryStream.listen((battery) {
      _batteryLevel = battery;
      notifyListeners();
    });
  }

  // ─── Actions ────────────────────────────────────────────────────

  Future<void> sendEmergency() async {
    _isProcessing = true;
    notifyListeners();
    
    await _emergencyService.triggerSOS();
    
    _isProcessing = false;
    notifyListeners();
  }

  Future<void> summonRobot() async {
    _isProcessing = true;
    notifyListeners();
    
    await _robotService.summonRobot();
    
    _isProcessing = false;
    notifyListeners();
  }

  Future<void> goHome(String robotIp) async {
    _isProcessing = true;
    notifyListeners();
    
    await _robotService.returnToDock(robotIp);
    
    _isProcessing = false;
    notifyListeners();
  }

  Future<void> moveRover(RobotMovement direction, String robotIp) async {
    _roverSpeed = direction == RobotMovement.stop ? 0.0 : 0.5;
    await _robotService.sendMovementCommand(direction, robotIp);
    notifyListeners();
  }

  Future<void> cancelCurrentMode(String robotIp) async {
    _roverSpeed = 0.0;
    await _robotService.sendMovementCommand(RobotMovement.stop, robotIp);
    notifyListeners();
  }

  /// Move rover by a 360° angle (from the virtual joystick).
  Future<void> moveRoverByAngle(String angleDeg, String robotIp) async {
    _roverSpeed = 0.5;
    await _robotService.sendAngleCommand(angleDeg, robotIp);
    notifyListeners();
  }

  @override
  void dispose() {
    _robotService.dispose();
    super.dispose();
  }
}

