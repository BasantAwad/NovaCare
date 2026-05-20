import 'dart:async';
import 'dart:convert';
import 'dart:io';

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

  static const String _apiKey = 'novacare-secure-key-2026';

  RobotService() {
    // Simulate initial status
    _statusController.add(RobotStatus.online);
    _batteryController.add(85);
  }

  /// Sends a movement command to the robot
  Future<void> sendMovementCommand(RobotMovement command, String robotIp) async {
    print('DEBUG: Sending robot command: ${command.name} to $robotIp');
    
    final client = HttpClient();
    client.connectionTimeout = const Duration(seconds: 3);

    try {
      if (command == RobotMovement.stop) {
        final uri = Uri.parse('http://$robotIp:9000/api/move/stop');
        final request = await client.postUrl(uri);
        request.headers.set('X-API-Key', _apiKey);
        request.headers.set('Content-Type', 'application/json');
        
        final response = await request.close();
        final responseBody = await response.transform(utf8.decoder).join();
        if (response.statusCode == 200) {
          print('DEBUG: Stop command sent successfully: $responseBody');
        } else {
          print('DEBUG: Stop failed with status ${response.statusCode}: $responseBody');
        }
      } else {
        final uri = Uri.parse('http://$robotIp:9000/api/move');
        final request = await client.postUrl(uri);
        request.headers.set('X-API-Key', _apiKey);
        request.headers.set('Content-Type', 'application/json');
        
        String direction;
        switch (command) {
          case RobotMovement.forward:
            direction = 'forward';
            break;
          case RobotMovement.backward:
            direction = 'backward';
            break;
          case RobotMovement.left:
            direction = 'turn_left';
            break;
          case RobotMovement.right:
            direction = 'turn_right';
            break;
          default:
            direction = 'stop';
        }

        final body = jsonEncode({
          'direction': direction,
          'speed': 35, // default safe movement speed for Dpad control
          'duration': 0.5 // short bursts of half a second for responsive manual control!
        });

        request.write(body);
        final response = await request.close();
        final responseBody = await response.transform(utf8.decoder).join();
        if (response.statusCode == 200) {
          print('DEBUG: Move command $direction sent successfully: $responseBody');
        } else {
          print('DEBUG: Move failed with status ${response.statusCode}: $responseBody');
        }
      }
    } catch (e) {
      print('DEBUG: Failed to send movement command - $e');
    } finally {
      client.close();
    }
  }

  /// Requests the robot to return to the charging dock (stops everything)
  Future<void> returnToDock(String robotIp) async {
    print('DEBUG: Requesting Return to Dock');
    await sendMovementCommand(RobotMovement.stop, robotIp);
  }

  /// Requests the robot to come to the user's location (summon)
  Future<void> summonRobot() async {
    print('DEBUG: Requesting Robot Summon (legacy REST call stub)');
    await Future.delayed(const Duration(seconds: 1));
  }

  void dispose() {
    _statusController.close();
    _batteryController.close();
  }
}
