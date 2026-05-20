import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:audioplayers/audioplayers.dart';
import '../providers/reminder_provider.dart';
import '../providers/translation_provider.dart';
import '../services/voice_service.dart';

class RemindersScreen extends StatelessWidget {
  const RemindersScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final reminderProvider = context.watch<ReminderProvider>();
    final reminders = reminderProvider.reminders;
    final translation = context.watch<TranslationProvider>();

    return Scaffold(
      appBar: AppBar(
        title: Text(translation.translate('reminders')),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _showAddReminderDialog(context, reminderProvider),
        child: const Icon(Icons.add),
      ),
      body: reminders.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.alarm_add_rounded, size: 64, color: Colors.grey.shade300),
                  const SizedBox(height: 16),
                  Text('No reminders set', style: TextStyle(color: Colors.grey.shade500)),
                ],
              ),
            )
          : ListView.separated(
              padding: const EdgeInsets.all(24),
              itemCount: reminders.length + 1,
              separatorBuilder: (_, __) => const SizedBox(height: 16),
              itemBuilder: (context, index) {
                if (index == reminders.length) {
                  return _buildMedicationPlaceholder();
                }
                final reminder = reminders[index];
                return _buildReminderItem(context, reminder, reminderProvider);
              },
            ),
    );
  }

  Widget _buildMedicationPlaceholder() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.grey.shade300, style: BorderStyle.none),
      ),
      child: const Column(
        children: [
          Row(
            children: [
              Icon(Icons.medication_rounded, color: Colors.grey),
              SizedBox(width: 12),
              Text(
                'Medication Reminders',
                style: TextStyle(color: Colors.grey, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          SizedBox(height: 8),
          Text(
            'Medication schedules from database will appear here.',
            style: TextStyle(color: Colors.grey, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildReminderItem(BuildContext context, ReminderModel reminder, ReminderProvider provider) {
    final timeStr = reminder.time.format(context);

    return Dismissible(
      key: Key(reminder.id),
      direction: DismissDirection.endToStart,
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 20),
        decoration: BoxDecoration(
          color: Colors.redAccent,
          borderRadius: BorderRadius.circular(20),
        ),
        child: const Icon(Icons.delete_outline_rounded, color: Colors.white),
      ),
      onDismissed: (_) => provider.deleteReminder(reminder.id),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: reminder.isActive ? Colors.indigo.shade50 : Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: reminder.isActive ? Colors.indigo.shade200 : Colors.grey.shade200,
          ),
        ),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    timeStr,
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: reminder.isActive ? Colors.indigo : Colors.black87,
                    ),
                  ),
                  Text(
                    reminder.title,
                    style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
                  ),
                  if (reminder.reason.isNotEmpty)
                    Text(
                      reminder.reason,
                      style: TextStyle(color: Colors.grey.shade600, fontSize: 14),
                    ),
                  if (reminder.sound != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 4),
                      child: Row(
                        children: [
                          const Icon(Icons.music_note, size: 14, color: Colors.indigo),
                          const SizedBox(width: 4),
                          Text(reminder.sound!, style: const TextStyle(fontSize: 12, color: Colors.indigo)),
                        ],
                      ),
                    ),
                ],
              ),
            ),
            Switch(
              value: reminder.isActive,
              onChanged: (v) => provider.toggleReminder(reminder.id),
              activeColor: Colors.indigo,
            ),
          ],
        ),
      ),
    );
  }

  void _showAddReminderDialog(BuildContext context, ReminderProvider provider) async {
    final titleController = TextEditingController();
    final reasonController = TextEditingController();
    TimeOfDay selectedTime = TimeOfDay.now();
    String? selectedSound;
    final AudioPlayer player = AudioPlayer();

    final sounds = ['Default', 'Bells', 'Digital', 'Soft', 'Alert'];

    await showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => AlertDialog(
          title: const Text('New Reminder'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: titleController,
                  decoration: const InputDecoration(labelText: 'Alarm Name'),
                ),
                TextField(
                  controller: reasonController,
                  decoration: const InputDecoration(labelText: 'Reason'),
                ),
                const SizedBox(height: 20),
                ListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Time'),
                  trailing: Text(selectedTime.format(context), style: const TextStyle(fontWeight: FontWeight.bold)),
                  onTap: () async {
                    final time = await showTimePicker(context: context, initialTime: selectedTime);
                    if (time != null) setState(() => selectedTime = time);
                  },
                ),
                const Divider(),
                const Text('Choose Sound', style: TextStyle(fontWeight: FontWeight.bold)),
                Wrap(
                  spacing: 8,
                  children: sounds.map((s) => ChoiceChip(
                    label: Text(s),
                    selected: (selectedSound ?? 'Default') == s,
                    onSelected: (selected) {
                      setState(() => selectedSound = s);
                      // In a real app, play a preview here
                      VoiceService().speak("Previewing $s sound");
                    },
                  )).toList(),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('CANCEL')),
            ElevatedButton(
              onPressed: () {
                if (titleController.text.isNotEmpty) {
                  provider.addReminder(ReminderModel(
                    id: DateTime.now().toIso8601String(),
                    title: titleController.text,
                    reason: reasonController.text,
                    time: selectedTime,
                    sound: selectedSound == 'Default' ? null : selectedSound,
                  ));
                  Navigator.pop(context);
                }
              },
              child: const Text('SAVE'),
            ),
          ],
        ),
      ),
    );
  }
}
