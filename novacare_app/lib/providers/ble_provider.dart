import 'package:flutter/material.dart';

/// BLE connection state for the ESP32 rover controller.
enum BleConnectionStatus { idle, scanning, connecting, connected, disconnected, error }

/// Manages Bluetooth Low Energy communication with the rover's ESP32 module.
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

  /// Start scanning for nearby BLE devices
  Future<void> startScan() async {
    _status = BleConnectionStatus.scanning;
    _discoveredDevices = [];
    notifyListeners();

    // TODO: Replace with actual flutter_blue_plus scanning
    await Future.delayed(const Duration(seconds: 2));

    // Simulated discovered devices
    _discoveredDevices = [
      {'name': 'NovaCare-Rover-01', 'id': 'AA:BB:CC:DD:EE:01'},
      {'name': 'NovaCare-Rover-02', 'id': 'AA:BB:CC:DD:EE:02'},
    ];

    _status = BleConnectionStatus.idle;
    notifyListeners();
  }

  /// Connect to a specific BLE device
  Future<void> connectToDevice(String deviceId, String deviceName) async {
    _status = BleConnectionStatus.connecting;
    notifyListeners();

    // TODO: Replace with actual BLE connection logic
    await Future.delayed(const Duration(seconds: 1));

    _connectedDeviceId = deviceId;
    _connectedDeviceName = deviceName;
    _rssi = -55;
    _status = BleConnectionStatus.connected;
    notifyListeners();
  }

  /// Disconnect from current device
  Future<void> disconnect() async {
    // TODO: Replace with actual BLE disconnection
    _connectedDeviceId = null;
    _connectedDeviceName = null;
    _rssi = 0;
    _status = BleConnectionStatus.disconnected;
    notifyListeners();
  }

  /// Send a command via BLE characteristic
  Future<bool> sendCommand(String command) async {
    if (_status != BleConnectionStatus.connected) return false;

    // TODO: Replace with actual BLE write to characteristic
    await Future.delayed(const Duration(milliseconds: 200));
    debugPrint('BLE Command sent: $command');
    return true;
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
