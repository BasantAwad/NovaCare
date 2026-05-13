import 'dart:async';

enum RobotMovement {
  forward,
  backward,
  left,
  right,
  stop,
  home,
  dock,
  comeToMe
}

enum RobotStatus {
  online,
  offline,
  charging,
  moving,
  error
}

class RobotService {
  final _statusController = StreamController<RobotStatus>.broadcast();
  final _batteryController = StreamController<int>.broadcast();

  Stream<RobotStatus> get statusStream => _statusController.stream;
  Stream<int> get batteryStream => _batteryController.stream;

  RobotService() {
    // Simulate initial status
    _statusController.add(RobotStatus.online);
    _batteryController.add(85);
  }

  /// Sends a movement command to the robot
  Future<void> sendMovementCommand(RobotMovement command) async {
    print('DEBUG: Sending robot command: ${command.name}');
    // TODO: Implement WebSocket/MQTT communication with SERBot
    await Future.delayed(const Duration(milliseconds: 200));
  }

  /// Requests the robot to come to the user's location (summon)
  Future<void> summonRobot() async {
    print('DEBUG: Requesting Robot Summon');
    // TODO: Implement navigation goal request to Jetson Nano
    await Future.delayed(const Duration(seconds: 1));
  }

  /// Requests the robot to return to the charging dock
  Future<void> returnToDock() async {
    print('DEBUG: Requesting Return to Dock');
    await Future.delayed(const Duration(seconds: 1));
  }

  void dispose() {
    _statusController.close();
    _batteryController.close();
  }
}
