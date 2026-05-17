import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

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

  BluetoothDevice? _connectedDevice;
  StreamSubscription<List<ScanResult>>? _scanSubscription;

  /// Start BLE scan for NovaCare rover devices
  Future<List<Map<String, String>>> scanForDevices({
    Duration timeout = const Duration(seconds: 4),
  }) async {
    _isScanning = true;

    try {
      if (await FlutterBluePlus.isSupported == false) {
        debugPrint('BleService: Bluetooth not supported');
        return [];
      }
      
      await FlutterBluePlus.startScan(
        withServices: [Guid(serviceUuid)],
        timeout: timeout,
      );

      final results = await FlutterBluePlus.scanResults.firstWhere(
        (results) => results.isNotEmpty,
        orElse: () => [],
      );

      return results.map((r) => {
        'name': r.device.platformName.isEmpty ? 'Unknown Device' : r.device.platformName,
        'id': r.device.remoteId.str,
        'rssi': r.rssi.toString(),
      }).toList();
    } catch (e) {
      debugPrint('BleService: Scan error - $e');
      return [];
    } finally {
      _isScanning = false;
    }
  }

  /// Start continuous scan to monitor RSSI of a specific device
  /// Returns a stream of RSSI values
  Stream<int> streamRssi(String deviceId) {
    // We use a stream controller to emit RSSI updates
    final controller = StreamController<int>();

    FlutterBluePlus.startScan(
      withServices: [Guid(serviceUuid)],
      continuousUpdates: true,
      continuousDivisor: 1, // Get updates as fast as possible
    );

    _scanSubscription = FlutterBluePlus.scanResults.listen((results) {
      for (ScanResult r in results) {
        if (r.device.remoteId.str == deviceId) {
          controller.add(r.rssi);
        }
      }
    });

    controller.onCancel = () {
      _scanSubscription?.cancel();
      FlutterBluePlus.stopScan();
    };

    return controller.stream;
  }

  /// Connect to a specific device
  Future<bool> connect(String deviceId) async {
    try {
      final device = BluetoothDevice.fromId(deviceId);
      await device.connect(autoConnect: false, timeout: const Duration(seconds: 10));
      _connectedDevice = device;
      
      debugPrint('BleService: Connected to $deviceId');
      return true;
    } catch (e) {
      debugPrint('BleService: Connection failed - $e');
      return false;
    }
  }

  /// Disconnect from current device
  Future<void> disconnect(String deviceId) async {
    if (_connectedDevice != null && _connectedDevice!.remoteId.str == deviceId) {
      await _connectedDevice!.disconnect();
      _connectedDevice = null;
    } else {
      final device = BluetoothDevice.fromId(deviceId);
      await device.disconnect();
    }
    debugPrint('BleService: Disconnected from $deviceId');
  }

  /// Write a command to the ESP32 command characteristic
  Future<bool> writeCommand(String command) async {
    if (_connectedDevice == null) return false;
    
    try {
      final services = await _connectedDevice!.discoverServices();
      final service = services.firstWhere((s) => s.uuid.toString() == serviceUuid);
      final char = service.characteristics.firstWhere((c) => c.uuid.toString() == commandCharUuid);
      
      await char.write(utf8.encode(command));

      debugPrint('BleService: Wrote command - $command');
      return true;
    } catch (e) {
      debugPrint('BleService: Write failed - $e');
      return false;
    }
  }

  /// Subscribe to telemetry notifications from ESP32
  Future<Stream<List<int>>?> subscribeTelemetry() async {
    if (_connectedDevice == null) return null;
    
    try {
      final services = await _connectedDevice!.discoverServices();
      final service = services.firstWhere((s) => s.uuid.toString() == serviceUuid);
      final char = service.characteristics.firstWhere((c) => c.uuid.toString() == telemetryCharUuid);
      
      await char.setNotifyValue(true);
      debugPrint('BleService: Subscribed to telemetry');
      return char.onValueReceived;
    } catch (e) {
      debugPrint('BleService: Subscribe failed - $e');
      return null;
    }
  }

  /// Parse telemetry bytes from ESP32
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
