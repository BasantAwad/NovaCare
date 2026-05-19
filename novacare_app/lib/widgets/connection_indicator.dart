import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/ble_provider.dart';
import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';

/// Tiny BLE/Cloud connection chip for the AppBar trailing slot.
class ConnectionIndicator extends StatelessWidget {
  const ConnectionIndicator({super.key});

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();
    final ble = context.watch<BleProvider>();

    final isConnected = rover.isConnected || ble.isConnected;
    final label = ble.isConnected ? 'BLE' : (rover.isConnected ? 'Cloud' : '');
    final color = isConnected ? AppColors.success : AppColors.danger;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(Radii.pill),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(shape: BoxShape.circle, color: color),
          ),
          if (label.isNotEmpty) ...[
            const SizedBox(width: 6),
            Text(
              label,
              style: AppText.caption(color: color)
                  .copyWith(fontWeight: FontWeight.w700, fontSize: 11),
            ),
          ],
        ],
      ),
    );
  }
}
