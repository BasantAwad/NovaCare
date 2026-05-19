import 'dart:async';
import 'dart:convert';
import 'dart:io';

enum RobotMovement { forward, backward, left, right, stop, home, dock, comeToMe }

enum RobotStatus { online, offline, charging, moving, error }

class RobotService {
  final _statusController = StreamController<RobotStatus>.broadcast();
  final _batteryController = StreamController<int>.broadcast();

  Stream<RobotStatus> get statusStream => _statusController.stream;
  Stream<int> get batteryStream => _batteryController.stream;

  static const String _apiKey = 'novacare-secure-key-2026';

  RobotService() {
    _statusController.add(RobotStatus.online);
    _batteryController.add(85);
  }

  Future<void> sendMovementCommand(RobotMovement command, String robotIp) async {
    final client = HttpClient();
    client.connectionTimeout = const Duration(seconds: 3);

    try {
      if (command == RobotMovement.stop) {
        final uri = Uri.parse('http://$robotIp:9000/api/move/stop');
        final request = await client.postUrl(uri);
        request.headers.set('X-API-Key', _apiKey);
        request.headers.set('Content-Type', 'application/json');
        final response = await request.close();
        await response.transform(utf8.decoder).join();
      } else {
        final uri = Uri.parse('http://$robotIp:9000/api/move');
        final request = await client.postUrl(uri);
        request.headers.set('X-API-Key', _apiKey);
        request.headers.set('Content-Type', 'application/json');

        String direction;
        switch (command) {
          case RobotMovement.forward:  direction = 'forward';    break;
          case RobotMovement.backward: direction = 'backward';   break;
          case RobotMovement.left:     direction = 'turn_left';  break;
          case RobotMovement.right:    direction = 'turn_right'; break;
          default:                     direction = 'stop';
        }

        final bytes = utf8.encode(jsonEncode({
          'direction': direction,
          'speed': 35,
          'duration': 0.5,
        }));
        request.headers.contentLength = bytes.length;
        request.add(bytes);
        final response = await request.close();
        await response.transform(utf8.decoder).join();
      }
    } catch (e) {
      print('RobotService: command failed - $e');
    } finally {
      client.close();
    }
  }

  Future<void> returnToDock(String robotIp) async {
    await sendMovementCommand(RobotMovement.stop, robotIp);
  }

  Future<void> summonRobot() async {
    await Future.delayed(const Duration(seconds: 1));
  }

  void dispose() {
    _statusController.close();
    _batteryController.close();
  }
}
