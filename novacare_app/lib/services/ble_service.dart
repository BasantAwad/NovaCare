import 'package:flutter/foundation.dart';

/// Service layer for BLE communication with the rover's ESP32 module.
/// Wraps flutter_blue_plus for clean integration with providers.
class BleService {
  static final BleService _instance = BleService._internal();
  factory BleService() => _instance;
  BleService._internal();

  // ESP32 NovaCare Service UUID
  static const String serviceUuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b';
  // Command Characteristic UUID
  static const String commandCharUuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8';
  // Telemetry Characteristic UUID (Notify)
  static const String telemetryCharUuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a9';

  bool _isScanning = false;
  bool get isScanning => _isScanning;

  /// Start BLE scan for NovaCare rover devices
  Future<List<Map<String, String>>> scanForDevices({
    Duration timeout = const Duration(seconds: 4),
  }) async {
    _isScanning = true;

    // TODO: Replace with actual flutter_blue_plus scan
    // await FlutterBluePlus.startScan(
    //   withServices: [Guid(serviceUuid)],
    //   timeout: timeout,
    // );
    //
    // final results = await FlutterBluePlus.scanResults.first;
    // return results.map((r) => {
    //   'name': r.device.platformName,
    //   'id': r.device.remoteId.str,
    //   'rssi': r.rssi.toString(),
    // }).toList();

    await Future.delayed(timeout);
    _isScanning = false;

    // Simulated results
    return [
      {'name': 'NovaCare-Rover-01', 'id': 'AA:BB:CC:DD:EE:01', 'rssi': '-55'},
    ];
  }

  /// Connect to a specific device
  Future<bool> connect(String deviceId) async {
    try {
      // TODO: Replace with actual BLE connection
      // final device = BluetoothDevice.fromId(deviceId);
      // await device.connect(autoConnect: false, timeout: const Duration(seconds: 10));
      // await device.discoverServices();

      debugPrint('BleService: Connected to $deviceId');
      return true;
    } catch (e) {
      debugPrint('BleService: Connection failed - $e');
      return false;
    }
  }

  /// Disconnect from current device
  Future<void> disconnect(String deviceId) async {
    // TODO: Replace with actual BLE disconnection
    // final device = BluetoothDevice.fromId(deviceId);
    // await device.disconnect();
    debugPrint('BleService: Disconnected from $deviceId');
  }

  /// Write a command to the ESP32 command characteristic
  Future<bool> writeCommand(String command) async {
    try {
      // TODO: Replace with actual characteristic write
      // final services = await device.discoverServices();
      // final service = services.firstWhere((s) => s.uuid.toString() == serviceUuid);
      // final char = service.characteristics.firstWhere((c) => c.uuid.toString() == commandCharUuid);
      // await char.write(utf8.encode(command));

      debugPrint('BleService: Wrote command - $command');
      return true;
    } catch (e) {
      debugPrint('BleService: Write failed - $e');
      return false;
    }
  }

  /// Subscribe to telemetry notifications from ESP32
  Stream<List<int>>? subscribeTelemetry() {
    // TODO: Replace with actual characteristic notification subscription
    // final services = await device.discoverServices();
    // final service = services.firstWhere((s) => s.uuid.toString() == serviceUuid);
    // final char = service.characteristics.firstWhere((c) => c.uuid.toString() == telemetryCharUuid);
    // await char.setNotifyValue(true);
    // return char.onValueReceived;

    debugPrint('BleService: Subscribed to telemetry');
    return null;
  }

  /// Parse telemetry bytes from ESP32
  /// Expected format: "BAT:85|HR:72|LOC:Living Room|TEMP:36.5"
  static Map<String, dynamic> parseTelemetry(List<int> bytes) {
    final data = String.fromCharCodes(bytes);
    final parts = data.split('|');
    final result = <String, dynamic>{};

    for (final part in parts) {
      final keyValue = part.split(':');
      if (keyValue.length == 2) {
        final key = keyValue[0].trim();
        final value = keyValue[1].trim();

        switch (key) {
          case 'BAT':
            result['battery'] = int.tryParse(value) ?? 0;
            break;
          case 'HR':
            result['heartRate'] = int.tryParse(value) ?? 0;
            break;
          case 'LOC':
            result['location'] = value;
            break;
          case 'TEMP':
            result['temperature'] = double.tryParse(value) ?? 0.0;
            break;
          case 'SPD':
            result['speed'] = double.tryParse(value) ?? 0.0;
            break;
        }
      }
    }

    return result;
  }
}
