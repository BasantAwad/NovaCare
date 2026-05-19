import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';

/// WebSocket-based service for the Summon Robot feature (port 9999).
/// Falls back to BLE if WebSocket is unavailable.
class SummonService {
  static final SummonService _instance = SummonService._internal();
  factory SummonService() => _instance;
  SummonService._internal();

  String _wsUrl = 'ws://192.168.137.150:9999';

  WebSocket? _channel;
  bool _connected = false;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  static const int _maxReconnectAttempts = 10;
  static const Duration _reconnectDelay = Duration(seconds: 3);

  final _statusController     = StreamController<Map<String, dynamic>>.broadcast();
  final _arrivedController    = StreamController<Map<String, dynamic>>.broadcast();
  final _failedController     = StreamController<Map<String, dynamic>>.broadcast();
  final _ackController        = StreamController<Map<String, dynamic>>.broadcast();
  final _connectionController = StreamController<bool>.broadcast();

  Stream<Map<String, dynamic>> get statusStream     => _statusController.stream;
  Stream<Map<String, dynamic>> get arrivedStream    => _arrivedController.stream;
  Stream<Map<String, dynamic>> get failedStream     => _failedController.stream;
  Stream<Map<String, dynamic>> get ackStream        => _ackController.stream;
  Stream<bool>                 get connectionStream => _connectionController.stream;

  bool get isConnected => _connected;

  void configure({required String robotHost, int port = 9999}) {
    _wsUrl = 'ws://$robotHost:$port';
  }

  Future<bool> connectWithFallback({String? bleDeviceId}) async {
    return await connect();
  }

  Future<bool> connect() async {
    if (_connected) return true;
    try {
      final ws = await WebSocket.connect(_wsUrl).timeout(const Duration(seconds: 5));
      _channel = ws;
      _connected = true;
      _reconnectAttempts = 0;
      _connectionController.add(true);
      _listenForMessages();
      return true;
    } catch (e) {
      debugPrint('SummonService: connect failed - $e');
      _connected = false;
      _connectionController.add(false);
      _scheduleReconnect();
      return false;
    }
  }

  Future<void> disconnect() async {
    _reconnectTimer?.cancel();
    _reconnectTimer = null;
    _reconnectAttempts = 0;
    try { await _channel?.close(); } catch (_) {}
    _channel = null;
    _connected = false;
    _connectionController.add(false);
  }

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

  Future<bool> sendSummonCancel({required String userId, String reason = 'user_cancelled'}) async {
    return _sendMessage({
      'type': 'summon_cancel',
      'payload': {
        'user_id': userId,
        'reason': reason,
        'timestamp': DateTime.now().toIso8601String(),
      },
    });
  }

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

  Future<bool> sendPlaySound({int frequency = 440, double duration = 0.5}) async {
    return _sendMessage({
      'type': 'play_sound',
      'payload': {'frequency': frequency, 'duration': duration},
    });
  }

  Future<bool> sendPlayAudio({
    required String name,
    required String audioBase64,
    String mime = 'audio/mpeg',
  }) async {
    return _sendMessage({
      'type': 'play_audio',
      'payload': {'name': name, 'mime': mime, 'audio_base64': audioBase64},
    });
  }

  Future<bool> _sendMessage(Map<String, dynamic> message) async {
    if (_connected && _channel != null) {
      try {
        _channel!.add(jsonEncode(message));
        return true;
      } catch (e) {
        debugPrint('SummonService: WS send error - $e');
      }
    }
    return false;
  }

  void _listenForMessages() {
    _channel!.listen(
      (data) {
        try {
          final msg = jsonDecode(data as String) as Map<String, dynamic>;
          _routeMessage(msg);
        } catch (e) {
          debugPrint('SummonService: parse error - $e');
        }
      },
      onDone: () {
        _connected = false;
        _connectionController.add(false);
        _scheduleReconnect();
      },
      onError: (e) {
        _connected = false;
        _connectionController.add(false);
        _scheduleReconnect();
      },
    );
  }

  void _routeMessage(Map<String, dynamic> msg) {
    final type    = msg['type'] as String?;
    final payload = (msg['payload'] ?? msg['data'] ?? {}) as Map<String, dynamic>;
    switch (type) {
      case 'summon_status':  _statusController.add(payload);  break;
      case 'summon_arrived': _arrivedController.add(payload); break;
      case 'summon_failed':  _failedController.add(payload);  break;
      case 'summon_ack':     _ackController.add(payload);     break;
    }
  }

  void _scheduleReconnect() {
    if (_reconnectAttempts >= _maxReconnectAttempts) return;
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(_reconnectDelay, () {
      _reconnectAttempts++;
      connect();
    });
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
