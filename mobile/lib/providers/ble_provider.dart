import 'package:flutter/material.dart';

enum BleConnectionStatus { idle, scanning, connecting, connected, disconnected, error }

/// BLE disabled — all communication is over WiFi.
class BleProvider extends ChangeNotifier {
  BleConnectionStatus get status => BleConnectionStatus.idle;
  String? get connectedDeviceName => null;
  String? get connectedDeviceId => null;
  int get rssi => -100;
  List<Map<String, String>> get discoveredDevices => [];
  bool get isConnected => false;
  String get signalStrength => 'N/A';

  void Function(Map<String, dynamic>)? onTelemetry;

  Future<void> startScan() async {}
  Future<void> connectToDevice(String deviceId, String deviceName) async {}
  Future<void> disconnect() async {}
  Future<bool> sendCommand(String command) async => false;
  void updateRssi(int value) {}
}
