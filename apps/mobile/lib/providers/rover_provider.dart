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

  Future<void> goHome() async {
    _isProcessing = true;
    notifyListeners();
    
    await _robotService.returnToDock();
    
    _isProcessing = false;
    notifyListeners();
  }

  Future<void> cancelCurrentMode() async {
    await _robotService.sendMovementCommand(RobotMovement.stop);
    notifyListeners();
  }

  @override
  void dispose() {
    _robotService.dispose();
    super.dispose();
  }
}

