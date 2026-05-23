import 'dart:async';
import 'package:flutter/material.dart';
import '../services/summon_service.dart';

/// Mirrors the robot-side SummonState machine (11 phases).
enum SummonPhase {
  idle,
  requested,
  initializing,
  scanningRssi,
  navigating,
  wallFollowing,
  recovering,
  arriving,
  arrived,
  failed,
  cancelled,
}

class SummonProvider extends ChangeNotifier {
  final SummonService _service = SummonService();

  SummonPhase _phase           = SummonPhase.idle;
  int         _rssiCurrent     = -100;
  String      _rssiTrend       = 'unknown';
  double      _distanceEstimate = -1;
  bool        _obstacleDetected = false;
  bool        _isWallFollowing  = false;
  double      _elapsedSeconds  = 0;
  int         _recoveryAttempts = 0;
  String      _statusMessage   = '';
  bool        _isConnected     = false;
  String      _failureReason   = '';

  // Real-time performance telemetry from Jetson Nano
  double _cpuUsage        = 0;
  double _ramUsage        = 0;
  double _cameraFps       = 0;
  double _loopLatencyMs   = 0;
  String _runtimeMode     = '';

  StreamSubscription? _statusSub;
  StreamSubscription? _arrivedSub;
  StreamSubscription? _failedSub;
  StreamSubscription? _connectionSub;

  SummonPhase get phase            => _phase;
  int         get rssiCurrent      => _rssiCurrent;
  String      get rssiTrend        => _rssiTrend;
  double      get distanceEstimate => _distanceEstimate;
  bool        get obstacleDetected => _obstacleDetected;
  bool        get isWallFollowing  => _isWallFollowing;
  double      get elapsedSeconds   => _elapsedSeconds;
  int         get recoveryAttempts => _recoveryAttempts;
  String      get statusMessage    => _statusMessage;
  bool        get isConnected      => _isConnected;
  String      get failureReason    => _failureReason;
  double      get cpuUsage         => _cpuUsage;
  double      get ramUsage         => _ramUsage;
  double      get cameraFps        => _cameraFps;
  double      get loopLatencyMs    => _loopLatencyMs;
  String      get runtimeMode      => _runtimeMode;

  bool get isActive =>
      _phase != SummonPhase.idle &&
      _phase != SummonPhase.arrived &&
      _phase != SummonPhase.failed &&
      _phase != SummonPhase.cancelled;

  bool get isTerminal =>
      _phase == SummonPhase.arrived ||
      _phase == SummonPhase.failed ||
      _phase == SummonPhase.cancelled;

  String get signalStrengthLabel {
    if (_rssiCurrent >= -40) return 'Very Close';
    if (_rssiCurrent >= -55) return 'Close';
    if (_rssiCurrent >= -70) return 'Medium';
    if (_rssiCurrent >= -85) return 'Far';
    return 'Very Far';
  }

  SummonProvider() {
    _statusSub     = _service.statusStream.listen(_onStatus);
    _arrivedSub    = _service.arrivedStream.listen(_onArrived);
    _failedSub     = _service.failedStream.listen(_onFailed);
    _connectionSub = _service.connectionStream.listen(_onConnection);
  }

  Future<bool> connectToRobot({required String robotHost, int port = 9999}) async {
    _service.configure(robotHost: robotHost, port: port);
    final connected = await _service.connectWithFallback();
    _isConnected = connected;
    notifyListeners();
    return connected;
  }

  Future<bool> startSummon({required String userId, required String bleMac}) async {
    if (isActive) return false;

    _phase            = SummonPhase.requested;
    _failureReason    = '';
    _statusMessage    = 'Sending summon request...';
    _elapsedSeconds   = 0;
    _recoveryAttempts = 0;
    notifyListeners();

    return await _service.sendSummonRequest(userId: userId, bleMac: bleMac);
  }

  Future<bool> cancelSummon({required String userId}) async {
    if (!isActive) return false;
    _phase         = SummonPhase.cancelled;
    _statusMessage = 'Cancelling...';
    notifyListeners();
    return await _service.sendSummonCancel(userId: userId);
  }

  void sendRssiUpdate(int rssi) {
    _service.sendRssiUpdate(rssi: rssi);
  }

  void resetToIdle() {
    _phase            = SummonPhase.idle;
    _statusMessage    = '';
    _failureReason    = '';
    _rssiCurrent      = -100;
    _distanceEstimate = -1;
    notifyListeners();
  }

  void _onStatus(Map<String, dynamic> data) {
    _phase            = _parsePhase(data['state'] as String? ?? 'idle');
    _rssiCurrent      = (data['rssi_current']     as num?)?.toInt()    ?? -100;
    _rssiTrend        = data['rssi_trend']         as String?           ?? 'unknown';
    _distanceEstimate = (data['distance_estimate_m'] as num?)?.toDouble() ?? -1;
    _obstacleDetected = data['obstacle_detected']  as bool?             ?? false;
    _isWallFollowing  = data['is_wall_following']  as bool?             ?? false;
    _elapsedSeconds   = (data['elapsed_seconds']   as num?)?.toDouble() ?? 0;
    _recoveryAttempts = (data['recovery_attempts'] as num?)?.toInt()    ?? 0;
    _statusMessage    = data['message']            as String?           ?? '';

    // Real-time Jetson Nano performance telemetry
    _cpuUsage       = (data['cpu_percent']      as num?)?.toDouble() ?? 0;
    _ramUsage       = (data['ram_percent']      as num?)?.toDouble() ?? 0;
    _cameraFps      = (data['camera_fps']       as num?)?.toDouble() ?? 0;
    _loopLatencyMs  = (data['loop_latency_ms']  as num?)?.toDouble() ?? 0;
    _runtimeMode    = data['runtime_mode']      as String?           ?? '';

    notifyListeners();
  }

  void _onArrived(Map<String, dynamic> data) {
    _phase         = SummonPhase.arrived;
    _statusMessage = 'Robot has arrived!';
    notifyListeners();
  }

  void _onFailed(Map<String, dynamic> data) {
    _phase         = SummonPhase.failed;
    _failureReason = data['reason'] as String? ?? 'Unknown error';
    _statusMessage = 'Summon failed';
    notifyListeners();
  }

  void _onConnection(bool connected) {
    _isConnected = connected;
    notifyListeners();
  }

  SummonPhase _parsePhase(String state) {
    switch (state) {
      case 'idle':          return SummonPhase.idle;
      case 'requested':     return SummonPhase.requested;
      case 'initializing':  return SummonPhase.initializing;
      case 'scanning_rssi': return SummonPhase.scanningRssi;
      case 'navigating':    return SummonPhase.navigating;
      case 'wall_following':return SummonPhase.wallFollowing;
      case 'recovering':    return SummonPhase.recovering;
      case 'arriving':      return SummonPhase.arriving;
      case 'arrived':       return SummonPhase.arrived;
      case 'failed':        return SummonPhase.failed;
      case 'cancelled':     return SummonPhase.cancelled;
      default:              return SummonPhase.idle;
    }
  }

  @override
  void dispose() {
    _statusSub?.cancel();
    _arrivedSub?.cancel();
    _failedSub?.cancel();
    _connectionSub?.cancel();
    _service.dispose();
    super.dispose();
  }
}
