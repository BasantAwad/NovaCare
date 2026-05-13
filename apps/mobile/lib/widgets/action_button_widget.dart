import 'package:flutter/material.dart';

/// Big accessible action button with icon, label, optional subtitle,
/// active state glow, and emergency styling.
class ActionButtonWidget extends StatefulWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final Color color;
  final Color backgroundColor;
  final VoidCallback onPressed;
  final bool isLarge;
  final bool isEmergency;
  final bool isActive;
  final bool isLoading;

  const ActionButtonWidget({
    super.key,
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.color,
    required this.backgroundColor,
    required this.onPressed,
    this.isLarge = false,
    this.isEmergency = false,
    this.isActive = false,
    this.isLoading = false,
  });

  @override
  State<ActionButtonWidget> createState() => _ActionButtonWidgetState();
}

class _ActionButtonWidgetState extends State<ActionButtonWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.04).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    if (widget.isEmergency) {
      _pulseController.repeat(reverse: true);
    }
  }

  @override
  void didUpdateWidget(ActionButtonWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isActive && !_pulseController.isAnimating) {
      _pulseController.repeat(reverse: true);
    } else if (!widget.isActive && !widget.isEmergency) {
      _pulseController.stop();
      _pulseController.reset();
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return AnimatedBuilder(
      animation: _pulseAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: (widget.isEmergency || widget.isActive)
              ? _pulseAnimation.value
              : 1.0,
          child: child,
        );
      },
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: widget.isLoading ? null : widget.onPressed,
          borderRadius: BorderRadius.circular(20),
          splashColor: widget.color.withOpacity(0.1),
          highlightColor: widget.color.withOpacity(0.05),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            padding: EdgeInsets.all(widget.isLarge ? 20 : 16),
            decoration: BoxDecoration(
              color: widget.backgroundColor,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: widget.isActive
                    ? widget.color.withOpacity(0.5)
                    : widget.color.withOpacity(0.15),
                width: widget.isActive ? 2 : 1,
              ),
              boxShadow: widget.isActive || widget.isEmergency
                  ? [
                      BoxShadow(
                        color: widget.color.withOpacity(0.2),
                        blurRadius: 16,
                        spreadRadius: 2,
                      ),
                    ]
                  : null,
            ),
            child: widget.isLoading
                ? _buildLoadingState()
                : _buildContent(theme),
          ),
        ),
      ),
    );
  }

  Widget _buildContent(ThemeData theme) {
    return Row(
      children: [
        // Icon container
        Container(
          width: widget.isLarge ? 60 : 48,
          height: widget.isLarge ? 60 : 48,
          decoration: BoxDecoration(
            color: widget.color.withOpacity(0.15),
            borderRadius: BorderRadius.circular(widget.isLarge ? 18 : 14),
          ),
          child: Icon(
            widget.icon,
            color: widget.color,
            size: widget.isLarge ? 32 : 24,
          ),
        ),
        const SizedBox(width: 14),

        // Text
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                widget.label,
                style: TextStyle(
                  fontSize: widget.isLarge ? 20 : 16,
                  fontWeight: FontWeight.w700,
                  color: widget.color,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                widget.subtitle,
                style: TextStyle(
                  fontSize: widget.isLarge ? 14 : 12,
                  fontWeight: FontWeight.w400,
                  color: widget.color.withOpacity(0.7),
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),

        // Arrow / Active indicator
        if (widget.isActive)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: widget.color,
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Text(
              'ON',
              style: TextStyle(
                color: Colors.white,
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          )
        else
          Icon(
            Icons.chevron_right_rounded,
            color: widget.color.withOpacity(0.4),
            size: 24,
          ),
      ],
    );
  }

  Widget _buildLoadingState() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(
            strokeWidth: 2.5,
            valueColor: AlwaysStoppedAnimation<Color>(widget.color),
          ),
        ),
        const SizedBox(width: 12),
        Text(
          'Sending...',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: widget.color,
          ),
        ),
      ],
    );
  }
}
