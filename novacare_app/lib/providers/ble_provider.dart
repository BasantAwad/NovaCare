import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

/// BLE connection state for the ESP32 rover controller.
enum BleConnectionStatus { idle, scanning, connecting, connected, disconnected, error }

/// TCP connection state for the rover command server.
enum TcpConnectionStatus { disconnected, connecting, connected, error }

/// Manages Bluetooth Low Energy communication with the rover's ESP32 module.
/// Also provides a TCP transport fallback for the same command protocol.
class BleProvider extends ChangeNotifier {
  BleConnectionStatus _status = BleConnectionStatus.idle;
  BleConnectionStatus get status => _status;

  String? _connectedDeviceName;
  String? get connectedDeviceName => _connectedDeviceName;

  String? _connectedDeviceId;
  String? get connectedDeviceId => _connectedDeviceId;

  int _rssi = 0;
  int get rssi => _rssi;

  List<Map<String, String>> _discoveredDevices = [];
  List<Map<String, String>> get discoveredDevices => _discoveredDevices;

  bool get isConnected => _status == BleConnectionStatus.connected;

  // Note: flutter_blue_plus exposes static APIs on `FlutterBluePlus`.

  // Internal maps and state
  final Map<String, BluetoothDevice> _deviceMap = {};
  StreamSubscription<List<ScanResult>>? _scanSub;
  StreamSubscription<BluetoothConnectionState>? _deviceStateSub;
  BluetoothDevice? _connectedDevice;
  BluetoothCharacteristic? _writeCharacteristic;
  BluetoothCharacteristic? _notifyCharacteristic;
  StreamSubscription<List<int>>? _notifySub;

  // TCP transport state
  TcpConnectionStatus _tcpStatus = TcpConnectionStatus.disconnected;
  TcpConnectionStatus get tcpStatus => _tcpStatus;
  bool get isTcpConnected => _tcpStatus == TcpConnectionStatus.connected;
  bool get isAnyConnected => isConnected || isTcpConnected;

  String _tcpHost = '10.34.19.247'; // default matches TcpProvider; override via Settings > Rover server
  int _tcpPort = 5555;
  String get tcpEndpoint => '$_tcpHost:$_tcpPort';
  String? _tcpErrorMessage;
  String? get tcpErrorMessage => _tcpErrorMessage;
  String? _tcpLastResponse;
  String? get tcpLastResponse => _tcpLastResponse;

  Socket? _tcpSocket;
  StreamSubscription<Uint8List>? _tcpSubscription;

  // UUIDs -- replace these with the rover's real service/characteristic UUIDs
  static const String SERVICE_UUID = '0000ffe0-0000-1000-8000-00805f9b34fb';
  static const String WRITE_CHAR_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb';
  static const String NOTIFY_CHAR_UUID = WRITE_CHAR_UUID;

  /// Start scanning for nearby BLE devices
  Future<void> startScan({Duration timeout = const Duration(seconds: 4)}) async {
    _status = BleConnectionStatus.scanning;
    _discoveredDevices = [];
    _deviceMap.clear();
    notifyListeners();

    try {
      // Ensure previous scan is stopped
      await FlutterBluePlus.stopScan();

      _scanSub = FlutterBluePlus.scanResults.listen((results) {
        final List<Map<String, String>> found = [];
        for (final r in results) {
          final id = r.device.id.id;
          final name = (r.device.name.isNotEmpty) ? r.device.name : r.advertisementData.localName ?? 'Unknown';
          _deviceMap[id] = r.device;
          found.add({'name': name, 'id': id});
        }
        _discoveredDevices = found;
        _status = BleConnectionStatus.idle;
        notifyListeners();
      });

      await FlutterBluePlus.startScan(timeout: timeout);
      // stopScan will be called automatically after timeout by the plugin,
      // but ensure we call stopScan if needed.
      Future.delayed(timeout, () async {
          try {
          await FlutterBluePlus.stopScan();
        } catch (_) {}
        _scanSub?.cancel();
        _scanSub = null;
      });
    } catch (e) {
      debugPrint('BleProvider.startScan error: $e');
      _status = BleConnectionStatus.error;
      notifyListeners();
    }
  }

  /// Connect to a specific BLE device
  Future<void> connectToDevice(String deviceId, String deviceName) async {
    _status = BleConnectionStatus.connecting;
    notifyListeners();

    final device = _deviceMap[deviceId];
    if (device == null) {
      debugPrint('BleProvider: device $deviceId not found in scanned list');
      _status = BleConnectionStatus.error;
      notifyListeners();
      return;
    }

    try {
      _connectedDevice = device;
      _connectedDeviceName = deviceName;
      _connectedDeviceId = deviceId;

      // Listen to connection state changes
      _deviceStateSub?.cancel();
      _deviceStateSub = device.connectionState.listen((s) async {
        debugPrint('BleProvider: device state changed: $s');
        if (s == BluetoothConnectionState.connected) {
          _status = BleConnectionStatus.connected;
          notifyListeners();
        } else if (s == BluetoothConnectionState.disconnected) {
          _status = BleConnectionStatus.disconnected;
          notifyListeners();
        }
      });

      // Connect
      await device.connect(autoConnect: false, timeout: const Duration(seconds: 10));

      // Discover services
      final services = await device.discoverServices();
      // Debug: list all services & characteristics so we can capture UUIDs
      for (final s in services) {
        debugPrint('BleProvider: Service ${s.uuid}');
        for (final c in s.characteristics) {
          debugPrint('BleProvider:  Characteristic ${c.uuid} props=' +
              '{read:${c.properties.read}, write:${c.properties.write}, notify:${c.properties.notify}, writeWithoutResponse:${c.properties.writeWithoutResponse}}');
        }

        if (s.uuid.toString().toLowerCase() == SERVICE_UUID) {
          for (final c in s.characteristics) {
            final cu = c.uuid.toString().toLowerCase();
            if (cu == WRITE_CHAR_UUID) {
              _writeCharacteristic = c;
            }
            if (cu == NOTIFY_CHAR_UUID) {
              _notifyCharacteristic = c;
            }
          }
        }
      }

      // If no specific service matched, try heuristics: pick first writable characteristic
      if (_writeCharacteristic == null) {
        for (final s in services) {
          for (final c in s.characteristics) {
            if (c.properties.write || c.properties.writeWithoutResponse) {
              _writeCharacteristic = c;
              break;
            }
          }
          if (_writeCharacteristic != null) break;
        }
      }

      // Subscribe to notifications if available
      if (_notifyCharacteristic != null) {
        try {
          await _notifyCharacteristic!.setNotifyValue(true);
          _notifySub = _notifyCharacteristic!.value.listen((data) {
            final payload = utf8.decode(data);
            debugPrint('BleProvider: notification: $payload');
            // TODO: parse telemetry frames and call RoverProvider.updateTelemetry
          });
        } catch (e) {
          debugPrint('BleProvider: failed to subscribe to notifyChar: $e');
        }
      }

      // Read initial RSSI if supported
      try {
        final r = await device.readRssi();
        _rssi = r;
        notifyListeners();
      } catch (_) {}

      _status = BleConnectionStatus.connected;
      notifyListeners();
      debugPrint('BleProvider: connected to $deviceName ($deviceId)');
    } catch (e) {
      debugPrint('BleProvider.connectToDevice error: $e');
      _status = BleConnectionStatus.error;
      notifyListeners();
    }
  }

  /// Disconnect from current device
  Future<void> disconnect() async {
    try {
      _notifySub?.cancel();
      _notifySub = null;
      _notifyCharacteristic = null;
      _writeCharacteristic = null;

      if (_connectedDevice != null) {
        try {
          await _connectedDevice!.disconnect();
        } catch (_) {}
      }
    } catch (e) {
      debugPrint('BleProvider.disconnect error: $e');
    }

    _connectedDevice = null;
    _connectedDeviceId = null;
    _connectedDeviceName = null;
    _rssi = 0;
    _status = BleConnectionStatus.disconnected;
    notifyListeners();
  }

  /// Update TCP endpoint used by the local rover command server.
  void updateTcpEndpoint(String host, int port) {
    _tcpHost = host.trim();
    _tcpPort = port;
    notifyListeners();
  }

  /// Connect to the rover command server via TCP.
  Future<void> connectToTcp({String? host, int? port}) async {
    if (host != null) _tcpHost = host.trim();
    if (port != null) _tcpPort = port;

    if (_tcpStatus == TcpConnectionStatus.connected && _tcpSocket != null) {
      return;
    }

    _tcpStatus = TcpConnectionStatus.connecting;
    _tcpErrorMessage = null;
    _tcpLastResponse = null;
    notifyListeners();

    try {
      _tcpSocket = await Socket.connect(_tcpHost, _tcpPort, timeout: const Duration(seconds: 5));
      _tcpStatus = TcpConnectionStatus.connected;
      notifyListeners();

      _tcpSubscription = _tcpSocket!.listen(
        (data) {
          _tcpLastResponse = utf8.decode(data);
          notifyListeners();
        },
        onDone: () async {
          _tcpStatus = TcpConnectionStatus.disconnected;
          notifyListeners();
          await disconnectTcp();
        },
        onError: (error) {
          _tcpErrorMessage = error.toString();
          _tcpStatus = TcpConnectionStatus.error;
          notifyListeners();
        },
        cancelOnError: true,
      );

      // By default, tell the rover server this connection is intended for input-only
      // sensor acquisition and command control. The rover may use this to enable
      // its onboard camera/microphone/ultrasonic sensors for data forwarding.
      await sendTcpCommand('INPUT_MODE:SERBOT_ONLY');
      await sendTcpCommand('SENSOR_SET:camera,microphone,ultrasonic');
    } catch (error) {
      _tcpErrorMessage = error.toString();
      _tcpStatus = TcpConnectionStatus.error;
      notifyListeners();
    }
  }

  /// Disconnect from current TCP rover server.
  Future<void> disconnectTcp() async {
    try {
      await _tcpSubscription?.cancel();
      _tcpSubscription = null;
      await _tcpSocket?.close();
      _tcpSocket = null;
    } catch (_) {
      try {
        _tcpSocket?.destroy();
      } catch (_) {}
      _tcpSocket = null;
    }

    _tcpStatus = TcpConnectionStatus.disconnected;
    notifyListeners();
  }

  /// Send a command through the active transport.
  /// Prefer TCP when available; fall back to BLE.
  Future<bool> sendCommand(String command) async {
    debugPrint('BleProvider: sendCommand -> $command');
    if (isTcpConnected) {
      return await sendTcpCommand(command);
    }
    return await sendBleCommand(command);
  }

  /// Send a command via BLE characteristic.
  Future<bool> sendBleCommand(String command) async {
    if (!isConnected) return false;

    try {
      if (_writeCharacteristic == null) {
        debugPrint('BleProvider: No writable characteristic available');
        return false;
      }

      final bytes = utf8.encode(command);
      final withoutResponse = !_writeCharacteristic!.properties.write;
      await _writeCharacteristic!.write(bytes, withoutResponse: withoutResponse);
      debugPrint('BLE Command sent: $command');
      return true;
    } catch (e) {
      debugPrint('BleProvider.sendBleCommand error: $e');
      return false;
    }
  }

  /// Send a command over TCP to the rover server.
  Future<bool> sendTcpCommand(String command) async {
    if (!isTcpConnected || _tcpSocket == null) return false;

    try {
      final bytes = utf8.encode('$command\n');
      _tcpSocket!.add(bytes);
      await _tcpSocket!.flush();
      debugPrint('TCP Command sent: $command');
      return true;
    } catch (e) {
      debugPrint('BleProvider.sendTcpCommand error: $e');
      _tcpErrorMessage = e.toString();
      _tcpStatus = TcpConnectionStatus.error;
      notifyListeners();
      return false;
    }
  }

  /// Update RSSI signal strength
  void updateRssi(int value) {
    _rssi = value;
    notifyListeners();
  }

  String get signalStrength {
    if (_rssi >= -50) return 'Excellent';
    if (_rssi >= -70) return 'Good';
    if (_rssi >= -85) return 'Fair';
    return 'Weak';
  }
}
