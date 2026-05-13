import 'package:flutter/material.dart';

class RemindersScreen extends StatelessWidget {
  const RemindersScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Medication & Reminders'),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {},
        child: const Icon(Icons.add),
      ),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          _buildReminderItem(
            'Morning Pill',
            '8:00 AM',
            ['Daily', '2 pills'],
            true,
          ),
          const SizedBox(height: 16),
          _buildReminderItem(
            'Vitamin D',
            '12:30 PM',
            ['Daily', '1 drop'],
            false,
          ),
          const SizedBox(height: 16),
          _buildReminderItem(
            'Check Blood Pressure',
            '6:00 PM',
            ['Weekly'],
            false,
          ),
        ],
      ),
    );
  }

  Widget _buildReminderItem(String title, String time, List<String> tags, bool isActive) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: isActive ? Colors.indigo.shade50 : Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: isActive ? Colors.indigo.shade200 : Colors.grey.shade200,
        ),
      ),
      child: Row(
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                time,
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: isActive ? Colors.indigo : Colors.black87,
                ),
              ),
              Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500)),
              const SizedBox(height: 8),
              Row(
                children: tags.map((t) => Container(
                  margin: const EdgeInsets.only(right: 8),
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.5),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.grey.shade300),
                  ),
                  child: Text(t, style: const TextStyle(fontSize: 12)),
                )).toList(),
              ),
            ],
          ),
          const Spacer(),
          Switch(value: isActive, onChanged: (v) {}),
        ],
      ),
    );
  }
}
