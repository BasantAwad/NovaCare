class EmergencyService {
  Future<bool> triggerSOS() async {
    print('CRITICAL: SOS Triggered!');
    // TODO: Send push notification to all caregivers via Firebase
    // TODO: Trigger robot siren/audio via SummonService.sendPlaySound
    await Future.delayed(const Duration(seconds: 2));
    return true;
  }

  Future<void> notifyCaregiver(String message) async {
    print('DEBUG: Notifying caregiver: $message');
    // TODO: Integrate with Firebase Cloud Messaging
    await Future.delayed(const Duration(seconds: 1));
  }
}
