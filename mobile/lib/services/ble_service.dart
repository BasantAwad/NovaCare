/// BLE disabled — all communication is over WiFi (WebSocket + HTTP).
class BleService {
  static final BleService _instance = BleService._internal();
  factory BleService() => _instance;
  BleService._internal();

  bool get isScanning => false;

  Future<List<Map<String, String>>> scanForDevices({Duration? timeout}) async => [];
  Stream<int> streamRssi(String deviceId) => const Stream.empty();
  Future<bool> connect(String deviceId) async => false;
  Future<void> disconnect(String deviceId) async {}
  Future<bool> writeCommand(String command) async => false;
  Future<Stream<List<int>>?> subscribeTelemetry() async => null;

  static Map<String, dynamic> parseTelemetry(List<int> bytes) => {};
}
