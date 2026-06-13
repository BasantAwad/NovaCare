import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/settings_provider.dart';
import '../services/robot_service.dart';

/// Full-screen live camera feed from the SerBot.
///
/// Lifecycle:
///   - initState  → POST /api/camera/session/start
///   - dispose    → POST /api/camera/session/stop
///
/// The MJPEG stream is loaded via Image.network pointing directly at
/// the robot's /api/camera/stream endpoint with api_key query param.
class LiveFeedScreen extends StatefulWidget {
  const LiveFeedScreen({super.key});

  @override
  State<LiveFeedScreen> createState() => _LiveFeedScreenState();
}

enum _FeedState { connecting, live, error, offline }

class _LiveFeedScreenState extends State<LiveFeedScreen> {
  final RobotService _robotService = RobotService();
  _FeedState _feedState = _FeedState.connecting;
  String? _streamUrl;
  String _errorMessage = '';
  String _robotIp = '192.168.8.50';

  // Unique key to force Image.network reload on retry
  int _imageKey = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final settings = context.read<SettingsProvider>();
      _robotIp = settings.robotIp;
      _startSession();
    });
  }

  @override
  void dispose() {
    _stopSession();
    super.dispose();
  }

  Future<void> _startSession() async {
    setState(() {
      _feedState = _FeedState.connecting;
      _errorMessage = '';
    });

    try {
      final result = await _robotService.startCameraSession(_robotIp);
      if (result != null && result.containsKey('stream_url')) {
        setState(() {
          // Build direct stream URL using the robot IP and API key
          _streamUrl =
              'http://$_robotIp:9000/api/camera/stream?api_key=novacare-secure-key-2026';
          _feedState = _FeedState.live;
          _imageKey++;
        });
      } else if (result != null && result.containsKey('error')) {
        setState(() {
          _feedState = _FeedState.offline;
          _errorMessage = result['error'] ?? 'Camera not available';
        });
      } else {
        setState(() {
          _feedState = _FeedState.error;
          _errorMessage = 'Unexpected response from robot';
        });
      }
    } catch (e) {
      setState(() {
        _feedState = _FeedState.error;
        _errorMessage = 'Cannot reach robot at $_robotIp';
      });
    }
  }

  Future<void> _stopSession() async {
    try {
      await _robotService.stopCameraSession(_robotIp);
    } catch (_) {
      // Best-effort — don't block dispose
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded, color: Colors.white, size: 28),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: _feedState == _FeedState.live
            ? _buildLiveBadge()
            : const Text(
                'Live Feed',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
        centerTitle: true,
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    switch (_feedState) {
      case _FeedState.connecting:
        return _buildConnectingState();
      case _FeedState.live:
        return _buildLiveState();
      case _FeedState.error:
        return _buildErrorState();
      case _FeedState.offline:
        return _buildOfflineState();
    }
  }

  // ─── Connecting State ────────────────────────────────────────────

  Widget _buildConnectingState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Pulsing camera icon
          TweenAnimationBuilder<double>(
            tween: Tween(begin: 0.6, end: 1.0),
            duration: const Duration(milliseconds: 800),
            builder: (context, value, child) {
              return Opacity(
                opacity: value,
                child: Transform.scale(
                  scale: value,
                  child: Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.teal.withOpacity(0.15),
                      border: Border.all(color: Colors.teal.withOpacity(0.3), width: 2),
                    ),
                    child: const Icon(
                      Icons.videocam_rounded,
                      color: Colors.teal,
                      size: 48,
                    ),
                  ),
                ),
              );
            },
            onEnd: () {
              // Restart animation loop
              if (mounted && _feedState == _FeedState.connecting) {
                setState(() {});
              }
            },
          ),
          const SizedBox(height: 24),
          const Text(
            'Connecting to camera...',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 18,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            _robotIp,
            style: const TextStyle(
              color: Colors.white38,
              fontSize: 14,
              fontFamily: 'monospace',
            ),
          ),
          const SizedBox(height: 24),
          const SizedBox(
            width: 32,
            height: 32,
            child: CircularProgressIndicator(
              color: Colors.teal,
              strokeWidth: 3,
            ),
          ),
        ],
      ),
    );
  }

  // ─── Live State ──────────────────────────────────────────────────

  Widget _buildLiveState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Stream container with rounded corners
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(8, 80, 8, 16),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Container(
                  decoration: BoxDecoration(
                    color: Colors.grey.shade900,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: Colors.teal.withOpacity(0.3),
                      width: 1.5,
                    ),
                  ),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      // MJPEG Stream — Flutter's Image.network handles
                      // multipart/x-mixed-replace MJPEG streams natively
                      Image.network(
                        _streamUrl!,
                        key: ValueKey(_imageKey),
                        fit: BoxFit.contain,
                        gaplessPlayback: true,
                        loadingBuilder: (context, child, loadingProgress) {
                          if (loadingProgress == null) return child;
                          return Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                CircularProgressIndicator(
                                  color: Colors.teal,
                                  value: loadingProgress.expectedTotalBytes != null
                                      ? loadingProgress.cumulativeBytesLoaded /
                                          loadingProgress.expectedTotalBytes!
                                      : null,
                                ),
                                const SizedBox(height: 16),
                                const Text(
                                  'Loading stream...',
                                  style: TextStyle(color: Colors.white54),
                                ),
                              ],
                            ),
                          );
                        },
                        errorBuilder: (context, error, stackTrace) {
                          return _buildStreamError();
                        },
                      ),

                      // Resolution badge (bottom-left)
                      Positioned(
                        bottom: 12,
                        left: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: const Text(
                            '320×240 • 10fps',
                            style: TextStyle(
                              color: Colors.white60,
                              fontSize: 11,
                              fontFamily: 'monospace',
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),

          // Bottom controls
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Refresh button
                _buildControlButton(
                  icon: Icons.refresh_rounded,
                  label: 'Refresh',
                  onTap: () {
                    setState(() {
                      _imageKey++;
                    });
                  },
                ),
                const SizedBox(width: 24),
                // Disconnect button
                _buildControlButton(
                  icon: Icons.stop_circle_outlined,
                  label: 'Disconnect',
                  color: Colors.red.shade400,
                  onTap: () => Navigator.of(context).pop(),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStreamError() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.signal_wifi_off_rounded, color: Colors.red.shade300, size: 48),
          const SizedBox(height: 12),
          const Text(
            'Stream interrupted',
            style: TextStyle(color: Colors.white70, fontSize: 16),
          ),
          const SizedBox(height: 16),
          TextButton.icon(
            onPressed: () {
              setState(() {
                _imageKey++;
              });
            },
            icon: const Icon(Icons.refresh, color: Colors.teal),
            label: const Text('Retry', style: TextStyle(color: Colors.teal)),
          ),
        ],
      ),
    );
  }

  // ─── Error State ─────────────────────────────────────────────────

  Widget _buildErrorState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.red.withOpacity(0.1),
              ),
              child: Icon(
                Icons.signal_wifi_connected_no_internet_4_rounded,
                color: Colors.red.shade300,
                size: 56,
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Connection Failed',
              style: TextStyle(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              _errorMessage,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white54, fontSize: 15),
            ),
            const SizedBox(height: 8),
            Text(
              'Robot IP: $_robotIp',
              style: const TextStyle(
                color: Colors.white38,
                fontSize: 13,
                fontFamily: 'monospace',
              ),
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: _startSession,
              icon: const Icon(Icons.refresh_rounded),
              label: const Text('Retry Connection'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.teal,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ─── Camera Offline State ────────────────────────────────────────

  Widget _buildOfflineState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.orange.withOpacity(0.1),
              ),
              child: const Icon(
                Icons.videocam_off_rounded,
                color: Colors.orange,
                size: 56,
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Camera Offline',
              style: TextStyle(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              _errorMessage.isNotEmpty ? _errorMessage : 'The robot camera is not available.',
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white54, fontSize: 15),
            ),
            const SizedBox(height: 32),
            OutlinedButton.icon(
              onPressed: _startSession,
              icon: const Icon(Icons.refresh_rounded, color: Colors.teal),
              label: const Text('Try Again', style: TextStyle(color: Colors.teal)),
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: Colors.teal),
                padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ─── Shared Widgets ──────────────────────────────────────────────

  Widget _buildLiveBadge() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            color: Colors.red,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: Colors.red.withOpacity(0.6),
                blurRadius: 6,
                spreadRadius: 2,
              ),
            ],
          ),
        ),
        const SizedBox(width: 8),
        const Text(
          'LIVE',
          style: TextStyle(
            color: Colors.red,
            fontWeight: FontWeight.bold,
            fontSize: 16,
            letterSpacing: 1.5,
          ),
        ),
      ],
    );
  }

  Widget _buildControlButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    Color color = Colors.white70,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color.withOpacity(0.1),
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: Icon(icon, color: color, size: 24),
          ),
          const SizedBox(height: 6),
          Text(
            label,
            style: TextStyle(color: color, fontSize: 12),
          ),
        ],
      ),
    );
  }
}
