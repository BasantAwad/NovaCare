import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';
import '../providers/translation_provider.dart';
import '../providers/settings_provider.dart';
import '../services/voice_service.dart';
import '../services/robot_service.dart';

class RoverControlScreen extends StatelessWidget {
  const RoverControlScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rover = context.watch<RoverProvider>();
    final translation = context.watch<TranslationProvider>();
    final settings = context.watch<SettingsProvider>();

    return Scaffold(
      appBar: AppBar(
        title: Text(translation.translate('controls')),
        centerTitle: true,
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          return Column(
            children: [
              // Telemetry Header
              _buildTelemetryHeader(context, rover),

              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Directional Pad (Arrows)
                      // _buildDPad(context, rover, theme, settings), // Kept original code intact per request
                      
                      // New Joystick UI
                      RoverJoystick(rover: rover, settings: settings),

                      const SizedBox(height: 48),

                      // Action Buttons
                      Row(
                        children: [
                          Expanded(
                            child: _buildActionButton(
                              context,
                              Icons.home_rounded,
                              'Dock',
                              Colors.blueGrey,
                              () {
                                VoiceService().speak("Moving to charging dock");
                                rover.goHome(settings.robotIp);
                              },
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: _buildActionButton(
                              context,
                              Icons.stop_circle_rounded,
                              'Stop',
                              Colors.red,
                              () {
                                VoiceService().speak("Stopping all movement");
                                rover.cancelCurrentMode(settings.robotIp);
                              },
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              // Footer Info
              Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  'Manual control is limited for safety. The robot will automatically avoid obstacles.',
                  style: theme.textTheme.bodySmall?.copyWith(fontStyle: FontStyle.italic),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildTelemetryHeader(BuildContext context, RoverProvider rover) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 24),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.primary.withOpacity(0.05),
        border: Border(bottom: BorderSide(color: Colors.grey.shade200)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _telemetryItem(context, Icons.battery_std, '${rover.batteryLevel}%', 'Power'),
          _telemetryItem(context, Icons.speed, '${rover.roverSpeed} m/s', 'Speed'),
          _telemetryItem(context, Icons.radar, 'Active', 'Obstacles'),
        ],
      ),
    );
  }

  Widget _telemetryItem(BuildContext context, IconData icon, String value, String label) {
    final theme = Theme.of(context);
    return Column(
      children: [
        Icon(icon, color: theme.colorScheme.primary, size: 20),
        const SizedBox(height: 4),
        Text(value, style: theme.textTheme.labelLarge?.copyWith(fontWeight: FontWeight.bold)),
        Text(label, style: theme.textTheme.bodySmall),
      ],
    );
  }

  // ignore: unused_element
  Widget _buildDPad(BuildContext context, RoverProvider rover, ThemeData theme, SettingsProvider settings) {
    double buttonSize = 90;

    return Column(
      children: [
        // Up Arrow
        _dPadButton(theme, Icons.keyboard_arrow_up_rounded, "Forward", () {
          VoiceService().speak("Moving forward");
          rover.moveRover(RobotMovement.forward, settings.robotIp);
        }),

        const SizedBox(height: 12),

        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Left Arrow
            _dPadButton(theme, Icons.keyboard_arrow_left_rounded, "Left", () {
              VoiceService().speak("Turning left");
              rover.moveRover(RobotMovement.left, settings.robotIp);
            }),

            SizedBox(width: buttonSize + 12), // Gap in center

            // Right Arrow
            _dPadButton(theme, Icons.keyboard_arrow_right_rounded, "Right", () {
              VoiceService().speak("Turning right");
              rover.moveRover(RobotMovement.right, settings.robotIp);
            }),
          ],
        ),

        const SizedBox(height: 12),

        // Down Arrow
        _dPadButton(theme, Icons.keyboard_arrow_down_rounded, "Backward", () {
          VoiceService().speak("Moving backward");
          rover.moveRover(RobotMovement.backward, settings.robotIp);
        }),
      ],
    );
  }

  Widget _dPadButton(ThemeData theme, IconData icon, String label, VoidCallback onPressed) {
    return InkWell(
      onTap: onPressed,
      borderRadius: BorderRadius.circular(24),
      child: Container(
        width: 90,
        height: 90,
        decoration: BoxDecoration(
          color: theme.colorScheme.primary.withOpacity(0.1),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: theme.colorScheme.primary.withOpacity(0.3), width: 2),
        ),
        child: Icon(icon, size: 56, color: theme.colorScheme.primary),
      ),
    );
  }

  Widget _buildActionButton(BuildContext context, IconData icon, String label, Color color, VoidCallback onPressed) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 24, color: Colors.white),
      label: Text(label, style: const TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.bold)),
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        padding: const EdgeInsets.symmetric(vertical: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    );
  }
}

class RoverJoystick extends StatefulWidget {
  final RoverProvider rover;
  final SettingsProvider settings;
  const RoverJoystick({super.key, required this.rover, required this.settings});

  @override
  State<RoverJoystick> createState() => _RoverJoystickState();
}

class _RoverJoystickState extends State<RoverJoystick> {
  Offset _position = Offset.zero;
  double _angle = 0.0;
  double _distance = 0.0;
  final double _joystickRadius = 110.0;
  final double _knobRadius = 40.0;
  DateTime _lastCommandTime = DateTime.now();

  void _updatePosition(Offset localPosition) {
    Offset center = Offset(_joystickRadius, _joystickRadius);
    Offset delta = localPosition - center;
    double dist = delta.distance;
    
    if (dist > _joystickRadius - _knobRadius) {
      delta = Offset.fromDirection(delta.direction, _joystickRadius - _knobRadius);
    }
    
    setState(() {
      _position = delta;
      _distance = (delta.distance / (_joystickRadius - _knobRadius)).clamp(0.0, 1.0);
      _angle = delta.direction * 180 / 3.141592653589793;
      // Adjust angle so 0 is Up (Forward)
      _angle += 90;
      if (_angle < 0) _angle += 360;
      if (_angle >= 360) _angle -= 360;
    });

    _sendCommand();
  }

  void _resetPosition() {
    setState(() {
      _position = Offset.zero;
      _distance = 0.0;
    });
  }

  void _sendCommand() {
    if (DateTime.now().difference(_lastCommandTime).inMilliseconds < 300) return;

    if (_distance > 0.2) {
      RobotMovement movement;
      if (_angle >= 315 || _angle < 45) {
        movement = RobotMovement.forward; // up
      } else if (_angle >= 45 && _angle < 135) {
        movement = RobotMovement.right; // right
      } else if (_angle >= 135 && _angle < 225) {
        movement = RobotMovement.backward; // down
      } else {
        movement = RobotMovement.left; // left
      }
      
      try {
        (widget.rover as dynamic).moveRover(movement, widget.settings.robotIp);
      } catch (e) {
        debugPrint("Move command error: $e");
      }
      _lastCommandTime = DateTime.now();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.primary.withOpacity(0.1),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                'Angle: ${_angle.toStringAsFixed(0)}°',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
            ),
            const SizedBox(width: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.primary.withOpacity(0.1),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                'Power: ${(_distance * 100).toStringAsFixed(0)}%',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 32),
        GestureDetector(
          onPanStart: (details) => _updatePosition(details.localPosition),
          onPanUpdate: (details) => _updatePosition(details.localPosition),
          onPanEnd: (details) => _resetPosition(),
          child: Container(
            width: _joystickRadius * 2,
            height: _joystickRadius * 2,
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 15,
                  spreadRadius: 5,
                ),
              ],
              border: Border.all(
                color: Theme.of(context).colorScheme.primary.withOpacity(0.15),
                width: 4,
              ),
            ),
            child: Center(
              child: Transform.translate(
                offset: _position,
                child: Container(
                  width: _knobRadius * 2,
                  height: _knobRadius * 2,
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.primary,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                        blurRadius: 10,
                        spreadRadius: 2,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.control_camera_rounded,
                    color: Colors.white,
                    size: 36,
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

