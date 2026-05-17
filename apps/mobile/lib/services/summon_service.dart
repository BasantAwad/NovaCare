import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';

// BLE fallback
import 'ble_service.dart';

/// WebSocket-based service for the Summon Robot feature.
///
/// Connects to the robot's existing WebSocket server (port 9999)
/// and handles the summon protocol:
///   - Sends summon_request, summon_cancel, rssi_update
///   - Receives summon_status, summon_arrived, summon_failed, summon_ack
class SummonService {
  static final SummonService _instance = SummonService._internal();
  factory SummonService() => _instance;
  SummonService._internal();

  // Default robot WebSocket URL (configurable)
  String _wsUrl = 'ws://192.168.1.100:9999';

  // Connection state
  WebSocket? _channel; // Real dart:io WebSocket
  bool _connected = false;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  static const int _maxReconnectAttempts = 10;
  static const Duration _reconnectDelay = Duration(seconds: 3);

  // Stream controllers for incoming messages
  final _statusController = StreamController<Map<String, dynamic>>.broadcast();
  final _arrivedController = StreamController<Map<String, dynamic>>.broadcast();
  final _failedController = StreamController<Map<String, dynamic>>.broadcast();
  final _ackController = StreamController<Map<String, dynamic>>.broadcast();
  final _connectionController = StreamController<bool>.broadcast();

  // Public streams
  Stream<Map<String, dynamic>> get statusStream => _statusController.stream;
  Stream<Map<String, dynamic>> get arrivedStream => _arrivedController.stream;
  Stream<Map<String, dynamic>> get failedStream => _failedController.stream;
  Stream<Map<String, dynamic>> get ackStream => _ackController.stream;
  Stream<bool> get connectionStream => _connectionController.stream;

  bool get isConnected => _connected;

  /// Configure the WebSocket URL
  void configure({required String robotHost, int port = 9999}) {
    _wsUrl = 'ws://$robotHost:$port';
    debugPrint('SummonService: configured URL=$_wsUrl');
  }

  /// Connect with automatic fallback: try WebSocket, then BLE scan/connect.
  /// Optionally pass a known BLE device id to skip scanning.
  Future<bool> connectWithFallback({String? bleDeviceId}) async {
    // Try WebSocket first
    final wsOk = await connect();
    if (wsOk) return true;

    // Try BLE fallback
    try {
      final ble = BleService();

      if (bleDeviceId != null && bleDeviceId.isNotEmpty) {
        final ok = await ble.connect(bleDeviceId);
        return ok;
      }

      // Scan for devices and connect to the first match
      final devices = await ble.scanForDevices(timeout: const Duration(seconds: 4));
      if (devices.isEmpty) return false;
      final first = devices.first;
      final id = first['id'] ?? '';
      if (id.isEmpty) return false;

      final ok = await ble.connect(id);
      return ok;
    } catch (e) {
      debugPrint('SummonService: BLE fallback failed - $e');
      return false;
    }
  }

  /// Connect to the robot's WebSocket server
  Future<bool> connect() async {
    if (_connected) return true;

    try {
      final ws = await _createWebSocket(_wsUrl);
      if (ws == null) return false;

      _channel = ws;
      _connected = true;
      _reconnectAttempts = 0;
      _connectionController.add(true);
      debugPrint('SummonService: connected to $_wsUrl');

      // Listen for messages
      _listenForMessages();
      return true;
    } catch (e) {
      debugPrint('SummonService: connection failed - $e');
      _connected = false;
      _connectionController.add(false);
      _scheduleReconnect();
      return false;
    }
  }

  /// Disconnect from the WebSocket server
  Future<void> disconnect() async {
    _reconnectTimer?.cancel();
    _reconnectTimer = null;
    _reconnectAttempts = 0;

    if (_channel != null) {
      try {
        await _channel!.close();
      } catch (_) {}
      _channel = null;
    }

    _connected = false;
    _connectionController.add(false);
    debugPrint('SummonService: disconnected');
  }

  // ─── Outgoing Messages ─────────────────────────────────────────

  /// Send a summon request to the robot
  Future<bool> sendSummonRequest({
    required String userId,
    required String bleMac,
    String bleServiceUuid = '',
    int priority = 5,
  }) async {
    return _sendMessage({
      'type': 'summon_request',
      'payload': {
        'user_id': userId,
        'ble_mac': bleMac,
        'ble_service_uuid': bleServiceUuid,
        'priority': priority,
        'timestamp': DateTime.now().toIso8601String(),
      },
    });
  }

  /// Cancel an active summon
  Future<bool> sendSummonCancel({
    required String userId,
    String reason = 'user_cancelled',
  }) async {
    return _sendMessage({
      'type': 'summon_cancel',
      'payload': {
        'user_id': userId,
        'reason': reason,
        'timestamp': DateTime.now().toIso8601String(),
      },
    });
  }

  /// Send an RSSI update from the phone
  Future<bool> sendRssiUpdate({required int rssi, String bleMac = ''}) async {
    return _sendMessage({
      'type': 'rssi_update',
      'payload': {
        'rssi': rssi,
        'ble_mac': bleMac,
        'timestamp': DateTime.now().toIso8601String(),
      },
    });
  }

  // ─── Internal ──────────────────────────────────────────────────

  Future<bool> _sendMessage(Map<String, dynamic> message) async {
    // Prefer WebSocket if connected
    if (_connected && _channel != null) {
      try {
        final raw = jsonEncode(message);
        _channel!.add(raw);
        debugPrint('SummonService: sent ${message['type']} via WS');
        return true;
      } catch (e) {
        debugPrint('SummonService: send error (WS) - $e');
        // fallthrough to BLE
      }
    }

    // BLE fallback: write a simple command
    try {
      final ble = BleService();
      final payload = jsonEncode(message);
      final ok = await ble.writeCommand(payload);
      if (ok) {
        debugPrint('SummonService: sent ${message['type']} via BLE');
        return true;
      }
    } catch (e) {
      debugPrint('SummonService: BLE send error - $e');
    }

    debugPrint('SummonService: cannot send - no connection available');
    return false;
  }

  /// Send a play_sound request (tries WS, then BLE fallback)
  Future<bool> sendPlaySound({int frequency = 440, double duration = 0.5}) async {
    final msg = {
      'type': 'play_sound',
      'payload': {'frequency': frequency, 'duration': duration}
    };
    return _sendMessage(msg);
  }

  /// Send raw audio (base64) to robot to play
  Future<bool> sendPlayAudio({required String name, required String audioBase64, String mime = 'audio/mpeg'}) async {
    final msg = {
      'type': 'play_audio',
      'payload': {'name': name, 'mime': mime, 'audio_base64': audioBase64}
    };
    return _sendMessage(msg);
  }

  void _listenForMessages() {
    _channel!.listen(
      (data) {
        try {
          final message = jsonDecode(data as String) as Map<String, dynamic>;
          _routeMessage(message);
        } catch (e) {
          debugPrint('SummonService: parse error - $e');
        }
      },
      onDone: () {
        debugPrint('SummonService: connection closed');
        _connected = false;
        _connectionController.add(false);
        _scheduleReconnect();
      },
      onError: (e) {
        debugPrint('SummonService: stream error - $e');
        _connected = false;
        _connectionController.add(false);
        _scheduleReconnect();
      },
    );
  }

  void _routeMessage(Map<String, dynamic> message) {
    final type = message['type'] as String?;
    final payload = message['payload'] as Map<String, dynamic>? ??
        message['data'] as Map<String, dynamic>? ??
        {};

    switch (type) {
      case 'summon_status':
        _statusController.add(payload);
        break;
      case 'summon_arrived':
        _arrivedController.add(payload);
        break;
      case 'summon_failed':
        _failedController.add(payload);
        break;
      case 'summon_ack':
        _ackController.add(payload);
        break;
      default:
        // Ignore non-summon messages (state_update, emotion, etc.)
        break;
    }
  }

  void _scheduleReconnect() {
    if (_reconnectAttempts >= _maxReconnectAttempts) {
      debugPrint('SummonService: max reconnect attempts reached');
      return;
    }

    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(_reconnectDelay, () {
      _reconnectAttempts++;
      debugPrint('SummonService: reconnecting (attempt $_reconnectAttempts)...');
      connect();
    });
  }

  /// Create a WebSocket connection.
  Future<WebSocket?> _createWebSocket(String url) async {
    try {
      return await WebSocket.connect(url).timeout(const Duration(seconds: 5));
    } catch (e) {
      debugPrint('SummonService: WebSocket connect error - $e');
      return null;
    }
  }

  void dispose() {
    disconnect();
    _statusController.close();
    _arrivedController.close();
    _failedController.close();
    _ackController.close();
    _connectionController.close();
  }
}
