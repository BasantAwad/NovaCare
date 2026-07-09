import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// Robot movement directions for D-pad control.
enum RobotMovement {
  forward,
  backward,
  left,
  right,
  turnLeft,
  turnRight,
  stop,
  home,
  dock,
  comeToMe,
}

/// Robot connectivity and operational status.
enum RobotStatus {
  online,
  offline,
  charging,
  moving,
  navigating,
  following,
  error,
}

/// Navigation status from Nav2.
enum NavStatus { idle, navigating, succeeded, failed, cancelled }

/// Result of a navigation command.
class NavigationResult {
  final bool success;
  final String message;
  final String? destination;
  final double? x, y, theta;

  NavigationResult({
    required this.success,
    required this.message,
    this.destination,
    this.x,
    this.y,
    this.theta,
  });

  factory NavigationResult.fromJson(Map<String, dynamic> json) {
    return NavigationResult(
      success: json['status'] == 'navigating',
      message: json['message'] ?? '',
      destination: json['destination'],
      x: (json['x'] as num?)?.toDouble(),
      y: (json['y'] as num?)?.toDouble(),
      theta: (json['theta'] as num?)?.toDouble(),
    );
  }
}

/// Real-time navigation progress from Nav2.
class NavigationProgress {
  final bool active;
  final String status;
  final double? currentX, currentY;
  final double? distanceRemaining;
  final int? estimatedTimeRemaining;

  NavigationProgress({
    required this.active,
    required this.status,
    this.currentX,
    this.currentY,
    this.distanceRemaining,
    this.estimatedTimeRemaining,
  });

  factory NavigationProgress.fromJson(Map<String, dynamic> json) {
    final nav = json['navigation'] as Map<String, dynamic>? ?? {};
    final feedback = nav['feedback'] as Map<String, dynamic>?;
    return NavigationProgress(
      active: nav['active'] ?? false,
      status: nav['status'] ?? 'idle',
      currentX: (feedback?['current_x'] as num?)?.toDouble(),
      currentY: (feedback?['current_y'] as num?)?.toDouble(),
      distanceRemaining:
          (feedback?['distance_remaining'] as num?)?.toDouble(),
      estimatedTimeRemaining:
          feedback?['estimated_time_remaining'] as int?,
    );
  }
}

/// Named navigation destination from robot config.
class Destination {
  final String name;
  final String label;
  final double x, y, theta;

  Destination({
    required this.name,
    required this.label,
    required this.x,
    required this.y,
    required this.theta,
  });

  factory Destination.fromJson(String key, Map<String, dynamic> json) {
    return Destination(
      name: key,
      label: json['label'] ?? key,
      x: (json['x'] as num?)?.toDouble() ?? 0,
      y: (json['y'] as num?)?.toDouble() ?? 0,
      theta: (json['theta'] as num?)?.toDouble() ?? 0,
    );
  }
}

/// SOS trigger result.
class SOSResult {
  final bool alarmActive;
  final bool notificationSent;
  final String message;

  SOSResult({
    required this.alarmActive,
    required this.notificationSent,
    required this.message,
  });

  factory SOSResult.fromJson(Map<String, dynamic> json) {
    return SOSResult(
      alarmActive: json['alarm_active'] ?? false,
      notificationSent: json['notification_sent'] ?? false,
      message: json['message'] ?? 'SOS triggered',
    );
  }
}

/// Robot health and telemetry snapshot.
class RobotHealth {
  final bool cameraAvailable;
  final bool motionAvailable;
  final bool lidarAvailable;
  final bool isMoving;
  final bool sosActive;
  final bool fallDetectionActive;
  final String robotName;
  final String robotType;
  final Map<String, dynamic>? vitals;
  final Map<String, dynamic>? pose;

  RobotHealth({
    required this.cameraAvailable,
    required this.motionAvailable,
    required this.lidarAvailable,
    required this.isMoving,
    required this.sosActive,
    required this.fallDetectionActive,
    required this.robotName,
    required this.robotType,
    this.vitals,
    this.pose,
  });

  factory RobotHealth.fromJson(Map<String, dynamic> json) {
    final hw = json['hardware'] as Map<String, dynamic>? ?? {};
    final robot = json['robot'] as Map<String, dynamic>? ?? {};
    return RobotHealth(
      cameraAvailable: hw['camera'] ?? false,
      motionAvailable: hw['motion'] ?? false,
      lidarAvailable: hw['lidar'] ?? false,
      isMoving: hw['moving'] ?? false,
      sosActive: json['sos_active'] ?? false,
      fallDetectionActive:
          (json['fall_detection'] as Map?)?['active'] ?? false,
      robotName: robot['name'] ?? 'Unknown',
      robotType: robot['type'] ?? 'unknown',
      vitals: json['vitals'] as Map<String, dynamic>?,
      pose: json['pose'] as Map<String, dynamic>?,
    );
  }
}

/// Service for communicating with the NovaCare JetAuto robot REST API.
class RobotService {
  final _statusController = StreamController<RobotStatus>.broadcast();
  final _batteryController = StreamController<int>.broadcast();

  Stream<RobotStatus> get statusStream => _statusController.stream;
  Stream<int> get batteryStream => _batteryController.stream;

  static const String _apiKey = 'novacare-secure-key-2026';
  static const Duration _timeout = Duration(seconds: 5);

  RobotService() {
    _statusController.add(RobotStatus.online);
    _batteryController.add(85);
  }

  // ═══════════════════════════════════════════════════════════════════
  // HTTP Helper
  // ═══════════════════════════════════════════════════════════════════

  Future<Map<String, dynamic>?> _get(String robotIp, String path) async {
    final client = HttpClient();
    client.connectionTimeout = _timeout;
    try {
      final uri = Uri.parse('http://$robotIp:9000$path');
      final request = await client.getUrl(uri);
      request.headers.set('X-API-Key', _apiKey);
      final response = await request.close();
      final body = await response.transform(utf8.decoder).join();
      if (response.statusCode == 200) {
        return jsonDecode(body) as Map<String, dynamic>;
      }
      print('DEBUG: GET $path failed: ${response.statusCode} $body');
      return null;
    } catch (e) {
      print('DEBUG: GET $path error: $e');
      return null;
    } finally {
      client.close();
    }
  }

  Future<Map<String, dynamic>?> _post(
    String robotIp,
    String path, [
    Map<String, dynamic>? body,
  ]) async {
    final client = HttpClient();
    client.connectionTimeout = _timeout;
    try {
      final uri = Uri.parse('http://$robotIp:9000$path');
      final request = await client.postUrl(uri);
      request.headers.set('X-API-Key', _apiKey);
      request.headers.set('Content-Type', 'application/json');
      if (body != null) {
        request.write(jsonEncode(body));
      }
      final response = await request.close();
      final responseBody = await response.transform(utf8.decoder).join();
      if (response.statusCode == 200) {
        return jsonDecode(responseBody) as Map<String, dynamic>;
      }
      print('DEBUG: POST $path failed: ${response.statusCode} $responseBody');
      return null;
    } catch (e) {
      print('DEBUG: POST $path error: $e');
      return null;
    } finally {
      client.close();
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Movement Commands
  // ═══════════════════════════════════════════════════════════════════

  /// Sends a movement command to the JetAuto robot.
  Future<void> sendMovementCommand(
      RobotMovement command, String robotIp) async {
    if (command == RobotMovement.stop) {
      await _post(robotIp, '/api/move/stop');
      return;
    }

    String direction;
    switch (command) {
      case RobotMovement.forward:
        direction = 'forward';
        break;
      case RobotMovement.backward:
        direction = 'backward';
        break;
      case RobotMovement.left:
        direction = 'left';
        break;
      case RobotMovement.right:
        direction = 'right';
        break;
      case RobotMovement.turnLeft:
        direction = 'turn_left';
        break;
      case RobotMovement.turnRight:
        direction = 'turn_right';
        break;
      default:
        direction = 'stop';
    }

    await _post(robotIp, '/api/move', {
      'direction': direction,
      'speed': 35,
      'duration': 0.5,
    });
  }

  /// Requests the robot to return to charging dock via Nav2 navigation.
  Future<void> returnToDock(String robotIp) async {
    await navigateToDestination(robotIp, 'dock');
  }

  /// Requests the robot to come to the user's location via Nav2 navigation.
  Future<NavigationResult?> summonRobot(String robotIp,
      {double? x, double? y, double? theta}) async {
    if (x != null && y != null) {
      return navigateToCoordinates(robotIp, x, y, theta ?? 0);
    }
    // If no coordinates provided, navigate to a default user location
    return navigateToDestination(robotIp, 'living');
  }

  // ═══════════════════════════════════════════════════════════════════
  // Autonomous Navigation (Nav2)
  // ═══════════════════════════════════════════════════════════════════

  /// Navigate to a named destination using Nav2 autonomous navigation.
  Future<NavigationResult?> navigateToDestination(
      String robotIp, String destination) async {
    final result = await _post(robotIp, '/api/navigate', {
      'destination': destination,
    });
    if (result != null) {
      _statusController.add(RobotStatus.navigating);
      return NavigationResult.fromJson(result);
    }
    return null;
  }

  /// Navigate to specific coordinates using Nav2.
  Future<NavigationResult?> navigateToCoordinates(
      String robotIp, double x, double y, double theta) async {
    final result = await _post(robotIp, '/api/navigate', {
      'x': x,
      'y': y,
      'theta': theta,
    });
    if (result != null) {
      _statusController.add(RobotStatus.navigating);
      return NavigationResult.fromJson(result);
    }
    return null;
  }

  /// Get real-time navigation status from Nav2.
  Future<NavigationProgress?> getNavigationStatus(String robotIp) async {
    final result = await _get(robotIp, '/api/navigation/status');
    if (result != null) {
      return NavigationProgress.fromJson(result);
    }
    return null;
  }

  /// Cancel the active Nav2 navigation goal.
  Future<bool> cancelNavigation(String robotIp) async {
    final result = await _post(robotIp, '/api/navigation/cancel');
    if (result != null) {
      _statusController.add(RobotStatus.online);
      return true;
    }
    return false;
  }

  /// Get the list of saved navigation destinations.
  Future<List<Destination>> getDestinations(String robotIp) async {
    final result = await _get(robotIp, '/api/map/destinations');
    if (result != null && result['destinations'] != null) {
      final dests = result['destinations'] as Map<String, dynamic>;
      return dests.entries
          .map((e) => Destination.fromJson(e.key, e.value))
          .toList();
    }
    return [];
  }

  /// Save a new named destination at the robot's current position.
  Future<bool> saveDestination(
      String robotIp, String name, String label) async {
    final result = await _post(robotIp, '/api/map/destinations', {
      'name': name,
      'label': label,
    });
    return result != null;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Follow Mode
  // ═══════════════════════════════════════════════════════════════════

  /// Start person-following mode using depth camera + LiDAR tracking.
  Future<bool> startFollowMode(String robotIp) async {
    final result = await _post(robotIp, '/api/follow/start');
    if (result != null) {
      _statusController.add(RobotStatus.following);
      return true;
    }
    return false;
  }

  /// Stop person-following mode.
  Future<bool> stopFollowMode(String robotIp) async {
    final result = await _post(robotIp, '/api/follow/stop');
    if (result != null) {
      _statusController.add(RobotStatus.online);
      return true;
    }
    return false;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Camera
  // ═══════════════════════════════════════════════════════════════════

  /// Start a camera session.
  Future<Map<String, dynamic>?> startCameraSession(String robotIp) async {
    return await _post(robotIp, '/api/camera/session/start');
  }

  /// Stop a camera session.
  Future<void> stopCameraSession(String robotIp) async {
    await _post(robotIp, '/api/camera/session/stop');
  }

  /// Get a single camera frame as base64 JPEG.
  Future<String?> getCameraFrame(String robotIp) async {
    final result = await _get(robotIp, '/api/camera/frame');
    return result?['image'] as String?;
  }

  /// Get the live MJPEG stream URL.
  String getStreamUrl(String robotIp) {
    return 'http://$robotIp:9000/api/camera/stream?api_key=$_apiKey';
  }

  // ═══════════════════════════════════════════════════════════════════
  // SOS / Emergency
  // ═══════════════════════════════════════════════════════════════════

  /// Trigger the full SOS emergency sequence on the robot.
  ///
  /// This activates the robot alarm, announces emergency via TTS,
  /// and sends push notifications to all registered caregivers.
  Future<SOSResult?> triggerSOS(String robotIp,
      {String userId = 'app_user', String location = 'unknown'}) async {
    final result = await _post(robotIp, '/api/sos/trigger', {
      'user_id': userId,
      'location': location,
    });
    if (result != null) {
      return SOSResult.fromJson(result);
    }
    return null;
  }

  /// Cancel the SOS alarm on the robot.
  Future<bool> cancelSOS(String robotIp) async {
    final result = await _post(robotIp, '/api/sos/cancel');
    return result != null;
  }

  // ═══════════════════════════════════════════════════════════════════
  // SLAM Map
  // ═══════════════════════════════════════════════════════════════════

  /// Get current SLAM map as PNG image URL.
  String getMapImageUrl(String robotIp) {
    return 'http://$robotIp:9000/api/map/current?api_key=$_apiKey';
  }

  /// Save the current SLAM map.
  Future<bool> saveMap(String robotIp, {String name = 'home_map'}) async {
    final result = await _post(robotIp, '/api/map/save', {'name': name});
    return result != null;
  }

  /// Load a saved SLAM map.
  Future<bool> loadMap(String robotIp, {String name = 'home_map'}) async {
    final result = await _post(robotIp, '/api/map/load', {'name': name});
    return result != null;
  }

  // ═══════════════════════════════════════════════════════════════════
  // LiDAR
  // ═══════════════════════════════════════════════════════════════════

  /// Check if obstacle is ahead.
  Future<bool> isObstacleAhead(String robotIp) async {
    final result = await _get(robotIp, '/api/lidar/obstacle');
    return result?['obstacle_ahead'] == true;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Fall Detection
  // ═══════════════════════════════════════════════════════════════════

  /// Start on-robot fall detection.
  Future<bool> startFallDetection(String robotIp) async {
    final result = await _post(robotIp, '/api/fall-detection/start');
    return result != null;
  }

  /// Stop on-robot fall detection.
  Future<bool> stopFallDetection(String robotIp) async {
    final result = await _post(robotIp, '/api/fall-detection/stop');
    return result != null;
  }

  /// Get fall detection status and last event.
  Future<Map<String, dynamic>?> getFallDetectionStatus(
      String robotIp) async {
    return await _get(robotIp, '/api/fall-detection/status');
  }

  // ═══════════════════════════════════════════════════════════════════
  // Health & Telemetry
  // ═══════════════════════════════════════════════════════════════════

  /// Get robot health, telemetry, vitals, and navigation state.
  Future<RobotHealth?> getHealth(String robotIp) async {
    final result = await _get(robotIp, '/health');
    if (result != null) {
      return RobotHealth.fromJson(result);
    }
    return null;
  }

  /// Get current vitals from smart watch.
  Future<Map<String, dynamic>?> getVitals(String robotIp) async {
    return await _get(robotIp, '/api/vitals/current');
  }

  // ═══════════════════════════════════════════════════════════════════
  // Audio
  // ═══════════════════════════════════════════════════════════════════

  /// Speak text on the robot speaker.
  Future<bool> speak(String robotIp, String text, {String lang = 'en'}) async {
    final result = await _post(robotIp, '/api/tts/speak', {
      'text': text,
      'lang': lang,
    });
    return result != null;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Gestures
  // ═══════════════════════════════════════════════════════════════════

  Future<bool> startGestures(String robotIp) async {
    final result = await _post(robotIp, '/api/gestures/start', {});
    return result != null;
  }

  Future<bool> stopGestures(String robotIp) async {
    final result = await _post(robotIp, '/api/gestures/stop', {});
    return result != null;
  }

  Future<bool?> getGestureResponse(String robotIp) async {
    final result = await _get(robotIp, '/api/gestures/response');
    if (result != null && result.containsKey('binary_response')) {
      return result['binary_response'] as bool?;
    }
    return null;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Lidar Guarding
  // ═══════════════════════════════════════════════════════════════════

  Future<bool> startGuarding(String robotIp) async {
    final result = await _post(robotIp, '/api/guarding/start', {});
    return result != null;
  }

  Future<bool> stopGuarding(String robotIp) async {
    final result = await _post(robotIp, '/api/guarding/stop', {});
    return result != null;
  }

  void dispose() {
    _statusController.close();
    _batteryController.close();
  }
}
