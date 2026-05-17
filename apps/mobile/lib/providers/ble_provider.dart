import 'dart:async';
import 'package:flutter/material.dart';
import '../services/ble_service.dart';

/// BLE connection state for the ESP32 rover controller.
enum BleConnectionStatus { idle, scanning, connecting, connected, disconnected, error }

/// Manages Bluetooth Low Energy communication with the rover's ESP32 module.
class BleProvider extends ChangeNotifier {
  final BleService _bleService = BleService();

  BleConnectionStatus _status = BleConnectionStatus.idle;
  BleConnectionStatus get status => _status;

  String? _connectedDeviceName;
  String? get connectedDeviceName => _connectedDeviceName;

  String? _connectedDeviceId;
  String? get connectedDeviceId => _connectedDeviceId;

  int _rssi = -100;
  int get rssi => _rssi;

  List<Map<String, String>> _discoveredDevices = [];
  List<Map<String, String>> get discoveredDevices => _discoveredDevices;

  bool get isConnected => _status == BleConnectionStatus.connected;

  StreamSubscription<int>? _rssiSubscription;

  /// Start scanning for nearby BLE devices
  Future<void> startScan() async {
    _status = BleConnectionStatus.scanning;
    _discoveredDevices = [];
    notifyListeners();

    _discoveredDevices = await _bleService.scanForDevices();

    _status = BleConnectionStatus.idle;
    notifyListeners();
  }

  /// Connect to a specific BLE device
  Future<void> connectToDevice(String deviceId, String deviceName) async {
    _status = BleConnectionStatus.connecting;
    notifyListeners();

    final success = await _bleService.connect(deviceId);

    if (success) {
      _connectedDeviceId = deviceId;
      _connectedDeviceName = deviceName;
      _status = BleConnectionStatus.connected;
      _rssi = -100; // Reset
      
      // Start streaming RSSI
      _rssiSubscription?.cancel();
      _rssiSubscription = _bleService.streamRssi(deviceId).listen((newRssi) {
        _rssi = newRssi;
        notifyListeners();
      });
      
    } else {
      _status = BleConnectionStatus.error;
    }
    
    notifyListeners();
  }

  /// Disconnect from current device
  Future<void> disconnect() async {
    if (_connectedDeviceId != null) {
      await _bleService.disconnect(_connectedDeviceId!);
    }
    _rssiSubscription?.cancel();
    _rssiSubscription = null;
    
    _connectedDeviceId = null;
    _connectedDeviceName = null;
    _rssi = -100;
    _status = BleConnectionStatus.disconnected;
    notifyListeners();
  }

  /// Send a command via BLE characteristic
  Future<bool> sendCommand(String command) async {
    if (_status != BleConnectionStatus.connected) return false;
    return await _bleService.writeCommand(command);
  }

  /// Update RSSI signal strength manually
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

  @override
  void dispose() {
    _rssiSubscription?.cancel();
    super.dispose();
  }
}
