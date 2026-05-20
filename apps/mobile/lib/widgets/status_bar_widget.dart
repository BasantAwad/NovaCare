import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/rover_provider.dart';
import '../services/robot_service.dart';

class StatusBarWidget extends StatelessWidget {
  const StatusBarWidget({super.key});

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();
    final theme = Theme.of(context);
    
    final statusColor = _getStatusColor(rover.status, theme);
    final statusLabel = _getStatusLabel(rover.status);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: statusColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: statusColor.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: statusColor,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Robot Status: $statusLabel',
                  style: theme.textTheme.labelLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: statusColor,
                  ),
                ),
                Text(
                  rover.roverLocation,
                  style: theme.textTheme.bodySmall?.copyWith(color: statusColor.withOpacity(0.8)),
                ),
              ],
            ),
          ),
          if (rover.status == RobotStatus.moving)
            const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
        ],
      ),
    );
  }

  Color _getStatusColor(RobotStatus status, ThemeData theme) {
    switch (status) {
      case RobotStatus.online:
        return Colors.green;
      case RobotStatus.offline:
        return Colors.red;
      case RobotStatus.charging:
        return Colors.blue;
      case RobotStatus.moving:
        return Colors.orange;
      case RobotStatus.error:
        return Colors.red;
    }
  }

  String _getStatusLabel(RobotStatus status) {
    switch (status) {
      case RobotStatus.online:
        return 'Online';
      case RobotStatus.offline:
        return 'Offline';
      case RobotStatus.charging:
        return 'Charging';
      case RobotStatus.moving:
        return 'In Motion';
      case RobotStatus.error:
        return 'System Error';
    }
  }
}

