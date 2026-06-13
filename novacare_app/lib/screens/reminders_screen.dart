import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/rover_provider.dart';
import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../widgets/nova_logo.dart';
import '../widgets/nc_primitives.dart';

// TODO(feature): real reminder model + persistence.
//   - Define `Reminder { id, time, title, ringtone, on, repeatRule }`
//   - Store locally in SQLite (drift) and sync to Firebase per user
//   - Background scheduler (flutter_local_notifications + workmanager)
//     fires the ringtone + sends a BLE "speak now" frame to the rover
//   - Medication schedule fed from /reminders/medication; status chip
//     reflects taken/due/later state from a confirmation event
class RemindersScreen extends StatefulWidget {
  const RemindersScreen({super.key});

  @override
  State<RemindersScreen> createState() => _RemindersScreenState();
}

class _RemindersScreenState extends State<RemindersScreen> {
  final List<_Reminder> daily = [
    _Reminder('07:30', 'Morning check-in', 'Default tone', true),
    _Reminder('09:00', 'Stretch + walk', 'Chime', true),
    _Reminder('13:00', 'Hydration nudge', 'Soft bell', false),
    _Reminder('19:30', 'Evening call', 'Family ringtone', true),
  ];

  final List<_Med> meds = [
    _Med('Amlodipine', '08:00', _MedStatus.taken),
    _Med('Metformin', '13:00', _MedStatus.due),
    _Med('Atorvastatin', '21:00', _MedStatus.later),
  ];

  @override
  Widget build(BuildContext context) {
    final rover = context.watch<RoverProvider>();

    final dueCount = meds.where((m) => m.status == _MedStatus.due).length;
    final activeCount = daily.where((d) => d.on).length;

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Column(
        children: [
          NcAppBar(
            leading: const NovaLogo(),
            title: Text('Reminders', style: AppText.appBarTitle()),
            battery: rover.batteryLevel,
          ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsetsDirectional.fromSTEB(20, 8, 20, 40),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Today', style: AppText.display1()),
                  const SizedBox(height: 4),
                  Text(
                    '$activeCount active · $dueCount medication due',
                    style: AppText.body(color: AppColors.inkMuted),
                  ),
                  NcSectionHead(
                    title: 'Daily',
                    action: Text(
                      'Edit',
                      style: AppText.caption(color: AppColors.brandTeal)
                          .copyWith(fontWeight: FontWeight.w700),
                    ),
                  ),
                  NcGroup(
                    children: [
                      for (int i = 0; i < daily.length; i++)
                        _DailyRow(
                          reminder: daily[i],
                          onToggle: (val) {
                            setState(() {
                              daily[i] = _Reminder(
                                daily[i].time,
                                daily[i].title,
                                daily[i].ringtone,
                                val,
                              );
                            });
                          },
                          onEdit: () async {
                            final edited = await _showReminderDialog(reminder: daily[i]);
                            if (edited != null) {
                              setState(() {
                                daily[i] = edited;
                              });
                            }
                          },
                        ),
                    ],
                  ),
                  const NcSectionHead(title: 'Medication'),
                  NcGroup(
                    children: [
                      for (final m in meds) _MedRow(med: m),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: AppColors.brandTeal,
        foregroundColor: Colors.white,
        elevation: 4,
        onPressed: () async {
          final newReminder = await _showReminderDialog();
          if (newReminder != null) {
            setState(() {
              daily.add(newReminder);
            });
          }
        },
        child: const Icon(Icons.add_rounded),
      ),
    );
  }

  Future<_Reminder?> _showReminderDialog({_Reminder? reminder}) async {
    final isEditing = reminder != null;
    final titleController = TextEditingController(text: isEditing ? reminder.title : '');
    TimeOfDay selectedTime = isEditing 
        ? TimeOfDay(
            hour: int.parse(reminder.time.split(':')[0]), 
            minute: int.parse(reminder.time.split(':')[1])
          )
        : TimeOfDay.now();

    return showDialog<_Reminder>(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setStateDialog) {
            return AlertDialog(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Radii.lg)),
              backgroundColor: Theme.of(context).colorScheme.surface,
              title: Text(isEditing ? 'Edit Reminder' : 'Add Reminder', style: AppText.display3()),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  TextField(
                    controller: titleController,
                    decoration: InputDecoration(
                      labelText: 'Title',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(Radii.sm)),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('Time: ${selectedTime.format(context)}', style: AppText.body()),
                      TextButton(
                        onPressed: () async {
                          final TimeOfDay? time = await showTimePicker(
                            context: context,
                            initialTime: selectedTime,
                          );
                          if (time != null) {
                            setStateDialog(() => selectedTime = time);
                          }
                        },
                        child: Text('Select Time', style: AppText.bodyStrong(color: AppColors.brandTeal)),
                      ),
                    ],
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: Text('Cancel', style: AppText.body(color: AppColors.inkMuted)),
                ),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.brandTeal,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Radii.sm)),
                  ),
                  onPressed: () {
                    final timeString = '${selectedTime.hour.toString().padLeft(2, '0')}:${selectedTime.minute.toString().padLeft(2, '0')}';
                    Navigator.of(context).pop(_Reminder(
                      timeString,
                      titleController.text.trim().isEmpty ? 'New Reminder' : titleController.text.trim(),
                      isEditing ? reminder.ringtone : 'Default tone',
                      isEditing ? reminder.on : true,
                    ));
                  },
                  child: const Text('Save'),
                ),
              ],
            );
          }
        );
      },
    );
  }
}

// ─── Internal placeholder types ─────────────────────────────────────
class _Reminder {
  final String time;
  final String title;
  final String ringtone;
  final bool on;
  _Reminder(this.time, this.title, this.ringtone, this.on);
}

class _DailyRow extends StatelessWidget {
  final _Reminder reminder;
  final ValueChanged<bool> onToggle;
  final VoidCallback? onEdit;
  const _DailyRow({
    required this.reminder,
    required this.onToggle,
    this.onEdit,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsetsDirectional.symmetric(
        horizontal: 16,
        vertical: 14,
      ),
      child: Row(
        children: [
          // Time always rendered LTR (clock format).
          Directionality(
            textDirection: TextDirection.ltr,
            child: Container(
              constraints: const BoxConstraints(minWidth: 70),
              child: FittedBox(
                fit: BoxFit.scaleDown,
                alignment: AlignmentDirectional.centerStart,
                child: Text(
                  reminder.time,
                  style: AppText.tileValue().copyWith(fontSize: 24),
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(reminder.title, style: AppText.display3()),
                const SizedBox(height: 2),
                Text(reminder.ringtone, style: AppText.caption()),
              ],
            ),
          ),
          NcSwitch(
            value: reminder.on,
            onChanged: onToggle,
            semanticLabel: reminder.title,
          ),
          if (onEdit != null) ...[
            const SizedBox(width: 8),
            IconButton(
              icon: const Icon(Icons.edit_rounded, size: 20),
              color: AppColors.brandTeal,
              onPressed: onEdit,
            ),
          ],
        ],
      ),
    );
  }
}

enum _MedStatus { taken, due, later }

class _Med {
  final String name;
  final String time;
  final _MedStatus status;
  _Med(this.name, this.time, this.status);
}

class _MedRow extends StatelessWidget {
  final _Med med;
  const _MedRow({required this.med});

  @override
  Widget build(BuildContext context) {
    final (chipLabel, chipStyle) = switch (med.status) {
      _MedStatus.taken => ('Taken', NcChipStyle.success),
      _MedStatus.due => ('Due', NcChipStyle.warn),
      _MedStatus.later => ('Later', NcChipStyle.normal),
    };

    return NcRow(
      icon: const Icon(Icons.medication_rounded),
      iconBg: AppColors.accent3,
      title: med.name,
      subtitle: med.time,
      trailing: NcChip(label: chipLabel, style: chipStyle),
      onTap: () {
        // TODO(feature): med detail sheet (dosage, last taken,
        // confirm-now action that writes to Firebase).
      },
    );
  }
}
