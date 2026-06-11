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

  /// Sends a 360° angle-based movement command to the robot.
  /// [angleDeg] is a string like "0"–"359" representing the heading.
  Future<void> sendAngleCommand(String angleDeg, String robotIp) async {
    print('DEBUG: Sending angle command: $angleDeg° to $robotIp');

    final client = HttpClient();
    client.connectionTimeout = const Duration(seconds: 3);

    try {
      final uri = Uri.parse('http://$robotIp:9000/api/move');
      final request = await client.postUrl(uri);
      request.headers.set('X-API-Key', _apiKey);
      request.headers.set('Content-Type', 'application/json');

      final body = jsonEncode({
        'direction': angleDeg,
        'speed': 35,
        'duration': 0.5,
      });

      request.write(body);
      final response = await request.close();
      final responseBody = await response.transform(utf8.decoder).join();
      if (response.statusCode == 200) {
        print('DEBUG: Angle move $angleDeg° sent successfully: $responseBody');
      } else {
        print('DEBUG: Angle move failed with status ${response.statusCode}: $responseBody');
      }
    } catch (e) {
      print('DEBUG: Failed to send angle command - $e');
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

  // ─── Camera Session Management ──────────────────────────────────

  /// Checks camera availability on the robot.
  Future<Map<String, dynamic>?> getCameraStatus(String robotIp) async {
    print('DEBUG: Checking camera status at $robotIp');

    final client = HttpClient();
    client.connectionTimeout = const Duration(seconds: 5);

    try {
      final uri = Uri.parse('http://$robotIp:9000/api/camera/status');
      final request = await client.getUrl(uri);
      request.headers.set('X-API-Key', _apiKey);

      final response = await request.close();
      final responseBody = await response.transform(utf8.decoder).join();

      if (response.statusCode == 200) {
        print('DEBUG: Camera status: $responseBody');
        return jsonDecode(responseBody) as Map<String, dynamic>;
      } else {
        print('DEBUG: Camera status failed: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('DEBUG: Camera status error - $e');
      return null;
    } finally {
      client.close();
    }
  }

  /// Starts a camera streaming session on the robot.
  /// Returns the response map containing 'stream_url' on success.
  Future<Map<String, dynamic>?> startCameraSession(String robotIp) async {
    print('DEBUG: Starting camera session at $robotIp');

    final client = HttpClient();
    client.connectionTimeout = const Duration(seconds: 5);

    try {
      final uri = Uri.parse('http://$robotIp:9000/api/camera/session/start');
      final request = await client.postUrl(uri);
      request.headers.set('X-API-Key', _apiKey);
      request.headers.set('Content-Type', 'application/json');

      final response = await request.close();
      final responseBody = await response.transform(utf8.decoder).join();

      if (response.statusCode == 200) {
        print('DEBUG: Camera session started: $responseBody');
        return jsonDecode(responseBody) as Map<String, dynamic>;
      } else if (response.statusCode == 503) {
        print('DEBUG: Camera not available (503)');
        return jsonDecode(responseBody) as Map<String, dynamic>;
      } else {
        print('DEBUG: Camera session start failed: ${response.statusCode}');
        return {'error': 'Failed to start camera (${response.statusCode})'};
      }
    } catch (e) {
      print('DEBUG: Camera session start error - $e');
      return null;
    } finally {
      client.close();
    }
  }

  /// Stops the camera streaming session on the robot.
  Future<void> stopCameraSession(String robotIp) async {
    print('DEBUG: Stopping camera session at $robotIp');

    final client = HttpClient();
    client.connectionTimeout = const Duration(seconds: 3);

    try {
      final uri = Uri.parse('http://$robotIp:9000/api/camera/session/stop');
      final request = await client.postUrl(uri);
      request.headers.set('X-API-Key', _apiKey);
      request.headers.set('Content-Type', 'application/json');

      final response = await request.close();
      final responseBody = await response.transform(utf8.decoder).join();
      print('DEBUG: Camera session stopped: $responseBody');
    } catch (e) {
      print('DEBUG: Camera session stop error - $e');
    } finally {
      client.close();
    }
  }

  void dispose() {
    _statusController.close();
    _batteryController.close();
  }
}
