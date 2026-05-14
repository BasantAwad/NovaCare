import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import '../providers/alert_provider.dart';

class NotificationsScreen extends StatelessWidget {
  const NotificationsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final alertProvider = context.watch<AlertProvider>();
    final alerts = alertProvider.alerts;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Alerts & Notifications'),
        actions: [
          if (alerts.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.done_all_rounded),
              tooltip: 'Mark all as read',
              onPressed: () => alertProvider.markAllAsRead(),
            ),
          if (alerts.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_sweep_rounded),
              tooltip: 'Clear all',
              onPressed: () => _showClearAllDialog(context, alertProvider),
            ),
        ],
      ),
      body: alerts.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.notifications_none_rounded, size: 64, color: Colors.grey.shade300),
                  const SizedBox(height: 16),
                  Text('No alerts at the moment', style: TextStyle(color: Colors.grey.shade500)),
                ],
              ),
            )
          : ListView.separated(
              padding: const EdgeInsets.all(24),
              itemCount: alerts.length,
              separatorBuilder: (_, __) => const SizedBox(height: 16),
              itemBuilder: (context, index) {
                final alert = alerts[index];
                return Dismissible(
                  key: Key(alert.id),
                  direction: DismissDirection.endToStart,
                  background: Container(
                    alignment: Alignment.centerRight,
                    padding: const EdgeInsets.only(right: 20),
                    decoration: BoxDecoration(
                      color: Colors.redAccent,
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: const Icon(Icons.delete_outline_rounded, color: Colors.white),
                  ),
                  onDismissed: (direction) {
                    alertProvider.dismissAlert(alert.id);
                  },
                  child: InkWell(
                    onTap: () => alertProvider.markAsRead(alert.id),
                    borderRadius: BorderRadius.circular(18),
                    child: _buildNotificationCard(context, alert),
                  ),
                );
              },
            ),
    );
  }

  void _showClearAllDialog(BuildContext context, AlertProvider provider) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear all alerts?'),
        content: const Text('This action cannot be undone.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('CANCEL')),
          TextButton(
            onPressed: () {
              provider.clearAll();
              Navigator.pop(context);
            },
            child: const Text('CLEAR ALL', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  Widget _buildNotificationCard(BuildContext context, AlertModel alert) {
    final timeStr = DateFormat('h:mm a').format(alert.timestamp);
    final isToday = DateTime.now().day == alert.timestamp.day;
    final dateDisplay = isToday ? 'Today' : DateFormat('MMM d').format(alert.timestamp);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: alert.isRead ? Colors.white : Colors.blue.shade50.withOpacity(0.3),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: alert.isRead ? Colors.grey.shade100 : Colors.blue.shade100,
          width: alert.isRead ? 1 : 1.5,
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: alert.color.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(alert.icon, color: alert.color, size: 24),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Text(
                        alert.title,
                        style: TextStyle(
                          fontWeight: alert.isRead ? FontWeight.w500 : FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ),
                    Text(
                      '$dateDisplay, $timeStr',
                      style: TextStyle(color: Colors.grey.shade500, fontSize: 11),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  alert.body,
                  style: TextStyle(
                    color: Colors.grey.shade700,
                    fontSize: 14,
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
          if (!alert.isRead)
            Container(
              margin: const EdgeInsets.only(left: 8, top: 4),
              width: 8,
              height: 8,
              decoration: const BoxDecoration(color: Colors.blue, shape: BoxShape.circle),
            ),
        ],
      ),
    );
  }
}
