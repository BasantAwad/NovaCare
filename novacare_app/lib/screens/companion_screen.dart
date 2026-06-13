import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';

import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../widgets/nova_logo.dart';
import '../widgets/nc_primitives.dart';

/// CompanionScreen — SKILL §4.3.
///
/// Currently a UI stub. The real behavior wires up in three layers — each
/// noted with TODO(backend) below:
///   1. Text mode → stream tokens from the LLM endpoint
///   2. Voice mode → Whisper STT (or on-device) → LLM → TTS playback
///   3. Sign mode → MediaPipe Hands → ASL/ArSL gloss → LLM
class CompanionScreen extends StatefulWidget {
  const CompanionScreen({super.key});

  @override
  State<CompanionScreen> createState() => _CompanionScreenState();
}

class _CompanionScreenState extends State<CompanionScreen> {
  int _mode = 0; // 0=text, 1=voice, 2=sign
  final TextEditingController _draft = TextEditingController();

  // Seed conversation. TODO(backend): replace with real chat history stream.
  final List<_Msg> _messages = [
    _Msg('Hi Amira — how are you feeling this afternoon?', isBot: true),
    _Msg('A little tired. Could you remind me about my medication?', isBot: false),
    _Msg('Of course. Your next dose is Metformin at 1:00 PM.', isBot: true),
  ];

  @override
  void dispose() {
    _draft.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Column(
        children: [
          NcAppBar(
            leading: const NovaLogo(),
            title: Text('Companion', style: AppText.appBarTitle()),
            status: NcConnectionStatus.online,
            statusLabel: 'Online',
            battery: rover.batteryLevel,
          ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsetsDirectional.fromSTEB(20, 8, 20, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  NcSeg(
                    labels: const ['Text', 'Voice', 'Sign'],
                    selected: _mode,
                    onSelect: (i) => setState(() => _mode = i),
                  ),
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsetsDirectional.all(12),
                    decoration: BoxDecoration(
                      color: AppColors.brandAquaSoft,
                      borderRadius: BorderRadius.circular(Radii.md),
                    ),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.shield_rounded,
                          size: 18,
                          color: AppColors.brandTeal,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Conversations are private and end-to-end encrypted.',
                            style: AppText.caption(color: AppColors.inkTeal),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  for (final m in _messages) _Bubble(msg: m),
                  const SizedBox(height: 8),
                  const _TypingDots(),
                  if (_mode == 2) ...[
                    const SizedBox(height: 16),
                    _SignPanel(),
                  ],
                  if (_mode == 1) ...[
                    const SizedBox(height: 16),
                    _VoicePanel(),
                  ],
                ],
              ),
            ),
          ),
          _Composer(
            controller: _draft,
            onSend: (text) {
              if (text.trim().isEmpty) return;
              setState(() {
                _messages.add(_Msg(text, isBot: false));
                _draft.clear();
              });
              // TODO(backend): stream LLM reply (OpenAI / on-device LLaMA);
              // for voice mode, pipe response into TTS.
            },
          ),
        ],
      ),
    );
  }
}

// ─── Message model ──────────────────────────────────────────────────
class _Msg {
  final String text;
  final bool isBot;
  _Msg(this.text, {required this.isBot});
}

class _Bubble extends StatelessWidget {
  final _Msg msg;
  const _Bubble({required this.msg});

  @override
  Widget build(BuildContext context) {
    final isBot = msg.isBot;
    final radius = BorderRadiusDirectional.only(
      topStart: const Radius.circular(18),
      topEnd: const Radius.circular(18),
      bottomStart: Radius.circular(isBot ? 6 : 18),
      bottomEnd: Radius.circular(isBot ? 18 : 6),
    );

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment:
            isBot ? MainAxisAlignment.start : MainAxisAlignment.end,
        children: [
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: isBot ? Theme.of(context).colorScheme.surface : AppColors.brandTeal,
                border: isBot ? Border.all(color: AppColors.line) : null,
                borderRadius: radius.resolve(Directionality.of(context)),
              ),
              child: Text(
                msg.text,
                style: AppText.body(
                  color: isBot ? AppColors.inkNavy : Colors.white,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _TypingDots extends StatefulWidget {
  const _TypingDots();

  @override
  State<_TypingDots> createState() => _TypingDotsState();
}

class _TypingDotsState extends State<_TypingDots>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1400),
  )..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _c,
      builder: (_, __) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(3, (i) {
            final phase = (_c.value + i * 0.15) % 1.0;
            final t = (phase * 2 - 1).abs();
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 3),
              child: Transform.translate(
                offset: Offset(0, -3 * (1 - t)),
                child: Opacity(
                  opacity: 0.4 + 0.6 * (1 - t),
                  child: Container(
                    width: 6,
                    height: 6,
                    decoration: const BoxDecoration(
                      color: AppColors.inkMuted,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
              ),
            );
          }),
        );
      },
    );
  }
}

class _SignPanel extends StatefulWidget {
  @override
  State<_SignPanel> createState() => _SignPanelState();
}

class _SignPanelState extends State<_SignPanel> {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  int _currentCameraIndex = 0;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) return;
      
      // Try to find the front camera first
      _currentCameraIndex = _cameras.indexWhere((c) => c.lensDirection == CameraLensDirection.front);
      if (_currentCameraIndex == -1) _currentCameraIndex = 0;

      await _setupController();
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<void> _setupController() async {
    if (_cameras.isEmpty) return;
    
    final oldController = _controller;
    if (oldController != null) {
      _controller = null;
      await oldController.dispose();
    }

    final newController = CameraController(
      _cameras[_currentCameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await newController.initialize();
      if (mounted) {
        setState(() {
          _controller = newController;
          _isInitialized = true;
        });
      }
    } catch (e) {
      debugPrint('Error setting up camera controller: $e');
    }
  }

  Future<void> _flipCamera() async {
    if (_cameras.length < 2) return;
    
    setState(() {
      _isInitialized = false;
    });
    
    _currentCameraIndex = (_currentCameraIndex + 1) % _cameras.length;
    await _setupController();
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 240,
      decoration: BoxDecoration(
        color: const Color(0xFF061821),
        borderRadius: BorderRadius.circular(Radii.md),
        border: Border.all(color: AppColors.line),
      ),
      clipBehavior: Clip.hardEdge,
      child: !_isInitialized || _controller == null
          ? const Center(child: CircularProgressIndicator(color: AppColors.brandTeal))
          : Stack(
              fit: StackFit.expand,
              children: [
                FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: _controller!.value.previewSize?.height ?? 1,
                    height: _controller!.value.previewSize?.width ?? 1,
                    child: CameraPreview(_controller!),
                  ),
                ),
                if (_cameras.length > 1)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: IconButton(
                      icon: const Icon(Icons.flip_camera_ios_rounded, color: Colors.white),
                      onPressed: _flipCamera,
                      style: IconButton.styleFrom(
                        backgroundColor: Colors.black45,
                      ),
                    ),
                  ),
              ],
            ),
    );
  }
}

class _VoicePanel extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: AppColors.accent3,
        borderRadius: BorderRadius.circular(Radii.md),
        border: Border.all(color: AppColors.accent2),
      ),
      child: Center(
        child: Column(
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: const BoxDecoration(
                color: AppColors.accent,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.mic_rounded,
                color: AppColors.inkNavy,
                size: 32,
              ),
            ),
            const SizedBox(height: 12),
            // TODO(feature): Whisper STT capture; show live waveform.
            Text('Tap and hold to speak', style: AppText.bodyStrong()),
          ],
        ),
      ),
    );
  }
}

class _Composer extends StatelessWidget {
  final TextEditingController controller;
  final ValueChanged<String> onSend;
  const _Composer({required this.controller, required this.onSend});

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).padding.bottom;
    return Container(
      padding: EdgeInsetsDirectional.only(
        start: 14,
        end: 14,
        top: 10,
        bottom: 10 + bottomInset,
      ),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: const Border(top: BorderSide(color: AppColors.line)),
      ),
      child: Row(
        children: [
          _IconBtn(
            icon: Icons.front_hand_rounded,
            onTap: () {
              // TODO(feature): one-tap quick phrase ("I need help", "Water").
            },
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: Theme.of(context).scaffoldBackgroundColor,
                border: Border.all(color: AppColors.line),
                borderRadius: BorderRadius.circular(Radii.pill),
              ),
              padding: const EdgeInsetsDirectional.symmetric(
                horizontal: 16,
                vertical: 10,
              ),
              child: TextField(
                controller: controller,
                decoration: const InputDecoration.collapsed(
                  hintText: 'Type a message…',
                ),
                style: AppText.body(),
                onSubmitted: onSend,
              ),
            ),
          ),
          const SizedBox(width: 8),
          _IconBtn(
            icon: Icons.send_rounded,
            primary: true,
            onTap: () => onSend(controller.text),
          ),
        ],
      ),
    );
  }
}

class _IconBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final bool primary;
  const _IconBtn({
    required this.icon,
    required this.onTap,
    this.primary = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 48,
        height: 48,
        decoration: BoxDecoration(
          color: primary ? AppColors.brandTeal : Theme.of(context).colorScheme.surface,
          shape: BoxShape.circle,
          border: Border.all(color: AppColors.line),
        ),
        child: Icon(
          icon,
          color: primary ? Colors.white : AppColors.inkNavy,
          size: 20,
        ),
      ),
    );
  }
}
