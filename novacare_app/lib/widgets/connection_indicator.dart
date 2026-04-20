import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../providers/ble_provider.dart';
import '../theme/app_colors.dart';

/// Small connection status indicator shown in the app bar.
class ConnectionIndicator extends StatelessWidget {
  const ConnectionIndicator({super.key});

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();
    final ble = context.watch<BleProvider>();
    final theme = Theme.of(context);

    final isConnected = rover.isConnected || ble.isConnected;
    final label = ble.isConnected ? 'BLE' : (rover.isConnected ? 'Cloud' : '');

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: isConnected
            ? AppColors.successGreen.withOpacity(0.1)
            : AppColors.batteryLow.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: isConnected
              ? AppColors.successGreen.withOpacity(0.3)
              : AppColors.batteryLow.withOpacity(0.3),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(
              color: isConnected ? AppColors.successGreen : AppColors.batteryLow,
              shape: BoxShape.circle,
              boxShadow: isConnected
                  ? [
                      BoxShadow(
                        color: AppColors.successGreen.withOpacity(0.4),
                        blurRadius: 6,
                        spreadRadius: 1,
                      ),
                    ]
                  : null,
            ),
          ),
          if (label.isNotEmpty) ...[
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: isConnected
                    ? AppColors.successGreen
                    : AppColors.batteryLow,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
