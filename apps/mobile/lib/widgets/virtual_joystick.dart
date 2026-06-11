import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../theme/app_colors.dart';

class VirtualJoystick extends StatefulWidget {
  final ValueChanged<String?> onDirectionChanged;
  final double size;

  const VirtualJoystick({
    super.key,
    required this.onDirectionChanged,
    this.size = 220.0,
  });

  @override
  State<VirtualJoystick> createState() => _VirtualJoystickState();
}

class _VirtualJoystickState extends State<VirtualJoystick> with SingleTickerProviderStateMixin {
  Offset _joystickOffset = Offset.zero;
  String? _currentDirection;
  String? _indicatorDirection;
  late AnimationController _resetController;
  late Animation<Offset> _resetAnimation;

  @override
  void initState() {
    super.initState();
    _resetController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),
    );
    _resetAnimation = Tween<Offset>(begin: Offset.zero, end: Offset.zero).animate(
      CurvedAnimation(parent: _resetController, curve: Curves.easeOut),
    );

    _resetController.addListener(() {
      setState(() {
        _joystickOffset = _resetAnimation.value;
      });
    });
  }

  @override
  void dispose() {
    _resetController.dispose();
    super.dispose();
  }

  void _updateJoystick(Offset localPosition, double maxRadius) {
    // Calculate relative offset from center of joystick
    final center = Offset(widget.size / 2, widget.size / 2);
    Offset offset = localPosition - center;

    // Constrain offset to maximum radius
    final distance = offset.distance;
    if (distance > maxRadius) {
      offset = offset / distance * maxRadius;
    }

    setState(() {
      _joystickOffset = offset;
    });

    // Determine direction from offset
    final deadzone = maxRadius * 0.25;
    if (offset.distance < deadzone) {
      _setDirection(null, null);
    } else {
      // Calculate angle in radians (-pi to pi)
      final angle = offset.direction;
      // Convert to degrees (0 to 360) where 0 is East/Right, 90 is South/Down, 180 is West/Left, 270 is North/Up
      final deg = (angle * 180 / math.pi + 360) % 360;
      
      // Determine which visual direction indicator to light up based on quadrants
      String visualDirection;
      if (deg >= 225 && deg < 315) {
        visualDirection = 'up';
      } else if (deg >= 45 && deg < 135) {
        visualDirection = 'down';
      } else if (deg >= 135 && deg < 225) {
        visualDirection = 'left';
      } else {
        visualDirection = 'right';
      }
      
      _setDirection(deg.round().toString(), visualDirection);
    }
  }

  void _setDirection(String? direction, String? visualDirection) {
    if (_currentDirection != direction || _indicatorDirection != visualDirection) {
      setState(() {
        _currentDirection = direction;
        _indicatorDirection = visualDirection;
      });
      HapticFeedback.selectionClick();
      widget.onDirectionChanged(direction);
    }
  }

  void _resetJoystick() {
    _resetAnimation = Tween<Offset>(
      begin: _joystickOffset,
      end: Offset.zero,
    ).animate(
      CurvedAnimation(parent: _resetController, curve: Curves.easeOut),
    );
    _resetController.forward(from: 0.0);
    _setDirection(null, null);
  }

  @override
  Widget build(BuildContext context) {
    final maxRadius = widget.size / 2 - 32.0;

    return GestureDetector(
      onPanStart: (details) {
        _resetController.stop();
        _updateJoystick(details.localPosition, maxRadius);
      },
      onPanUpdate: (details) {
        _updateJoystick(details.localPosition, maxRadius);
      },
      onPanEnd: (_) {
        _resetJoystick();
      },
      child: SizedBox(
        width: widget.size,
        height: widget.size,
        child: Stack(
          alignment: Alignment.center,
          children: [
            // Outer track container with a gradient border
            Container(
              width: widget.size,
              height: widget.size,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: SweepGradient(
                  colors: [
                    AppColors.brandTeal,
                    AppColors.brandLeaf,
                    AppColors.accent,
                    AppColors.brandTeal,
                  ],
                ),
              ),
              padding: const EdgeInsets.all(2.0),
              child: Container(
                decoration: BoxDecoration(
                  color: AppColors.roverDarkBg,
                  shape: BoxShape.circle,
                  border: Border.all(color: AppColors.roverDarkBorder.withOpacity(0.5)),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.3),
                      blurRadius: 16,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    // Directional Indicator arrows in the background
                    _buildDirectionIndicator('up', Alignment.topCenter, Icons.keyboard_arrow_up_rounded),
                    _buildDirectionIndicator('down', Alignment.bottomCenter, Icons.keyboard_arrow_down_rounded),
                    _buildDirectionIndicator('left', Alignment.centerLeft, Icons.keyboard_arrow_left_rounded),
                    _buildDirectionIndicator('right', Alignment.centerRight, Icons.keyboard_arrow_right_rounded),
                    
                    // Subtle inner guiding circle
                    Container(
                      width: widget.size * 0.5,
                      height: widget.size * 0.5,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: AppColors.roverDarkBorder.withOpacity(0.15),
                          width: 1.5,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // The moving joystick knob
            Transform.translate(
              offset: _joystickOffset,
              child: Container(
                width: 72.0,
                height: 72.0,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    colors: [
                      AppColors.brandTeal,
                      AppColors.accent,
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.accent.withOpacity(0.4),
                      blurRadius: 14,
                      spreadRadius: 1,
                      offset: const Offset(0, 4),
                    ),
                    BoxShadow(
                      color: Colors.black.withOpacity(0.5),
                      blurRadius: 6,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Center(
                  child: Container(
                    width: 66.0,
                    height: 66.0,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: AppColors.roverDarkBg.withOpacity(0.9),
                    ),
                    child: Center(
                      child: Icon(
                        Icons.drag_indicator_rounded,
                        color: _currentDirection != null ? AppColors.accent : AppColors.roverDarkText.withOpacity(0.7),
                        size: 26,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDirectionIndicator(String direction, Alignment alignment, IconData icon) {
    final isActive = _indicatorDirection == direction;
    return Align(
      alignment: alignment,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 150),
          opacity: isActive ? 1.0 : 0.25,
          child: Icon(
            icon,
            color: isActive ? AppColors.accent : AppColors.roverDarkText,
            size: 28,
          ),
        ),
      ),
    );
  }
}
