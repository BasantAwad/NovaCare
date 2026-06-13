import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';


import 'package:flutter/material.dart';

/// TCP connection state for a remote NovaCare rover command server.
enum TcpConnectionStatus { disconnected, connecting, connected, error }

class TcpProvider extends ChangeNotifier {
  TcpConnectionStatus _status = TcpConnectionStatus.disconnected;
  String _host = '192.168.8.50';
  int _port = 5555;
  String? _lastResponse;
  String? _errorMessage;

  Socket? _socket;
  StreamSubscription<Uint8List>? _socketSubscription;

  TcpConnectionStatus get status => _status;
  bool get isConnected => _status == TcpConnectionStatus.connected;
  String get host => _host;
  int get port => _port;
  String get endpoint => '$_host:$_port';
  String? get lastResponse => _lastResponse;
  String? get errorMessage => _errorMessage;

  void updateEndpoint(String host, int port) {
    _host = host.trim();
    _port = port;
    notifyListeners();
  }

  Future<void> connect({String? host, int? port}) async {
    if (host != null || port != null) {
      _host = host?.trim() ?? _host;
      _port = port ?? _port;
    }

    if (_status == TcpConnectionStatus.connected && _socket != null) {
      return;
    }

    _status = TcpConnectionStatus.connecting;
    _errorMessage = null;
    _lastResponse = null;
    notifyListeners();

    try {
      _socket = await Socket.connect(_host, _port, timeout: const Duration(seconds: 5));
      _status = TcpConnectionStatus.connected;
      notifyListeners();

      _socketSubscription = _socket!.listen(
        (data) {
          _lastResponse = utf8.decode(data);
          notifyListeners();
        },
        onDone: () async {
          _status = TcpConnectionStatus.disconnected;
          notifyListeners();
          await disconnect();
        },
        onError: (error) {
          _errorMessage = error.toString();
          _status = TcpConnectionStatus.error;
          notifyListeners();
        },
        cancelOnError: true,
      );
    } catch (error) {
      _errorMessage = error.toString();
      _status = TcpConnectionStatus.error;
      notifyListeners();
    }
  }

  Future<void> disconnect() async {
    try {
      await _socketSubscription?.cancel();
      _socketSubscription = null;
      await _socket?.close();
      _socket = null;
    } catch (_) {
      try {
        _socket?.destroy();
      } catch (_) {}
      _socket = null;
    }

    _status = TcpConnectionStatus.disconnected;
    notifyListeners();
  }

  Future<bool> sendCommand(String command) async {
    if (!isConnected || _socket == null) {
      return false;
    }

    try {
      final bytes = utf8.encode('$command\n');
      _socket!.add(bytes);
      await _socket!.flush();
      return true;
    } catch (error) {
      _errorMessage = error.toString();
      _status = TcpConnectionStatus.error;
      notifyListeners();
      return false;
    }
  }

  @override
  void dispose() {
    _socketSubscription?.cancel();
    _socket?.destroy();
    super.dispose();
  }
}
