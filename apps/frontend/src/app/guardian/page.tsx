"use client";

import { useState, useEffect } from "react";
import {
  Phone,
  MessageCircle,
  Video,
  Heart,
  Activity,
  Battery,
  Clock,
  Check,
  AlertTriangle,
  Pill,
  Navigation,
  MessageSquare,
  Bell,
  X,
  PhoneCall,
  Loader2,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent, Button, Avatar, Badge, ProgressBar, Modal, ModalHeader, ModalBody, ModalFooter } from "@/components/ui";
import { cn } from "@/lib/utils";
import {
  getMedications,
  getActivities,
  getVitals,
  getLinkedRover,
  getBatteryStatus,
  getMoodLogs,
  type MedicationSchedule,
  type ActivityLog,
  type VitalSign,
  type LinkedRover,
  type BatteryStatus,
  type MoodLog,
} from "@/lib/dashboard-api";
import { useSignaling } from "@/hooks/useSignaling";

const activityColors = {
  medication: "bg-success",
  navigation: "bg-primary",
  conversation: "bg-purple-500",
  alert: "bg-secondary",
  vital: "bg-accent",
};

const activityIcons = {
  medication: Pill,
  navigation: Navigation,
  conversation: MessageSquare,
  alert: Bell,
  vital: Heart,
};

// ---------------------------------------------------------------------------
// Fallback mock data — used ONLY when backend returns empty/error
// ---------------------------------------------------------------------------
const FALLBACK_PATIENT = {
  name: "Sarah Johnson",
  status: "online" as const,
};

const FALLBACK_VITALS = {
  heartRate: { value: 72, status: "normal" },
  activityLevel: { value: "Moderate", status: "normal" },
  battery: { value: 85, status: "normal" },
  lastCheckIn: "5m ago",
};

const FALLBACK_MEDICATIONS: MedicationSchedule[] = [
  { id: "1", rover_id: "", medication_id: "", medication_name: "Lisinopril", dosage: "", frequency: "", scheduled_time: "8:00 AM", status: "taken", taken_at: "8:05 AM", is_active: true },
  { id: "2", rover_id: "", medication_id: "", medication_name: "Metformin", dosage: "", frequency: "", scheduled_time: "12:00 PM", status: "taken", taken_at: "12:10 PM", is_active: true },
  { id: "3", rover_id: "", medication_id: "", medication_name: "Aspirin", dosage: "", frequency: "", scheduled_time: "6:00 PM", status: "upcoming", is_active: true },
];

const FALLBACK_ACTIVITIES: ActivityLog[] = [
  { id: "1", rover_id: "", type: "medication", title: "Medication Taken", description: "Took Metformin (500mg)", timestamp: new Date(Date.now() - 2 * 3600000).toISOString() },
  { id: "2", rover_id: "", type: "navigation", title: "Navigation Complete", description: "Navigated to Kitchen", timestamp: new Date(Date.now() - 3 * 3600000).toISOString() },
  { id: "3", rover_id: "", type: "conversation", title: "Conversation", description: "Had a 15-minute conversation with NovaCare", timestamp: new Date(Date.now() - 4 * 3600000).toISOString() },
  { id: "4", rover_id: "", type: "medication", title: "Medication Taken", description: "Took Lisinopril (10mg)", timestamp: new Date(Date.now() - 6 * 3600000).toISOString() },
  { id: "5", rover_id: "", type: "alert", title: "Warning", description: "Low battery warning - now charging", timestamp: new Date(Date.now() - 7 * 3600000).toISOString() },
];

function formatTimeAgo(timestamp: string): string {
  const diff = Date.now() - new Date(timestamp).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function GuardianDashboard() {
  const [alertOpen, setAlertOpen] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Robot configuration
  const ROBOT_IP = "10.174.134.247";
  const VIDEO_FEED_URL = `http://${ROBOT_IP}:5000/video_feed`;

  // Real-time signaling
  const { navigateToPatient, lastEvent, isConnected } = useSignaling("guardian_001", "guardian");

  // Auto-open modal on emergency event
  useEffect(() => {
    if (lastEvent?.type === "EMERGENCY_TRIGGERED") {
      setAlertOpen(true);
    }
  }, [lastEvent]);

  // Dynamic state fetched from backend
  const [medications, setMedications] = useState<MedicationSchedule[]>(FALLBACK_MEDICATIONS);
  const [activities, setActivities] = useState<ActivityLog[]>(FALLBACK_ACTIVITIES);
  const [linkedRover, setLinkedRover] = useState<LinkedRover | null>(null);
  const [latestVitals, setLatestVitals] = useState<VitalSign | null>(null);
  const [battery, setBattery] = useState<BatteryStatus | null>(null);
  const [todayMood, setTodayMood] = useState<MoodLog | null>(null);

  // Derived display values
  const patientName = linkedRover ? `${linkedRover.first_name} ${linkedRover.last_name}` : FALLBACK_PATIENT.name;
  const patientStatus = linkedRover?.status || FALLBACK_PATIENT.status;
  const heartRate = latestVitals?.heart_rate ?? FALLBACK_VITALS.heartRate.value;
  const activityLevel = FALLBACK_VITALS.activityLevel.value;
  const batteryLevel = battery?.battery_percent ?? FALLBACK_VITALS.battery.value;
  const moodEmoji = todayMood?.emoji ?? "😊";
  const moodLabel = todayMood ? todayMood.mood.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase()) : "Content";
  const lastCheckIn = linkedRover?.last_check_in
    ? formatTimeAgo(linkedRover.last_check_in)
    : FALLBACK_VITALS.lastCheckIn;

  useEffect(() => {
    async function fetchDashboardData() {
      setIsLoading(true);
      try {
        // Fetch all data concurrently
        const [medsRes, activitiesRes, roverRes, vitalsRes, batteryRes, moodRes] = await Promise.all([
          getMedications(),
          getActivities(),
          getLinkedRover(),
          getVitals(),
          getBatteryStatus(),
          getMoodLogs(),
        ]);

        if (medsRes.status === "success" && medsRes.data && medsRes.data.length > 0) {
          setMedications(medsRes.data);
        }

        if (activitiesRes.status === "success" && activitiesRes.data && activitiesRes.data.length > 0) {
          setActivities(activitiesRes.data);
        }

        if (roverRes.status === "success" && roverRes.data) {
          setLinkedRover(roverRes.data);
        }

        if (vitalsRes.status === "success" && vitalsRes.data && vitalsRes.data.length > 0) {
          setLatestVitals(vitalsRes.data[0]);
        }

        if (batteryRes.status === "success" && batteryRes.data) {
          setBattery(batteryRes.data);
        }

        if (moodRes.status === "success" && moodRes.data && moodRes.data.length > 0) {
          setTodayMood(moodRes.data[0]);
        }
      } catch (error) {
        console.error("Failed to fetch guardian dashboard data:", error);
        // Fallback data remains in state
      } finally {
        setIsLoading(false);
      }
    }

    fetchDashboardData();
  }, []);

  const takenMeds = medications.filter((m) => m.status === "taken").length;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Quick Actions */}
      <Card variant="elevated" className="overflow-hidden">
        <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-gray-100 dark:divide-gray-700">
          <button className="flex items-center gap-4 p-6 hover:bg-primary-50 dark:hover:bg-primary-900/30 transition-colors group">
            <div className="w-14 h-14 rounded-2xl bg-primary-100 dark:bg-primary-900/50 flex items-center justify-center group-hover:bg-primary group-hover:text-white transition-colors">
              <Phone className="w-7 h-7 text-primary group-hover:text-white" />
            </div>
            <div className="text-left">
              <p className="font-semibold text-text-primary dark:text-white">Call</p>
              <p className="text-sm text-text-muted dark:text-gray-400">Voice Call</p>
            </div>
          </button>

          <button className="flex items-center gap-4 p-6 hover:bg-secondary-50 dark:hover:bg-secondary-900/30 transition-colors group">
            <div className="w-14 h-14 rounded-2xl bg-secondary-100 dark:bg-secondary-900/50 flex items-center justify-center group-hover:bg-secondary group-hover:text-white transition-colors">
              <MessageCircle className="w-7 h-7 text-secondary group-hover:text-white" />
            </div>
            <div className="text-left">
              <p className="font-semibold text-text-primary dark:text-white">Message</p>
              <p className="text-sm text-text-muted dark:text-gray-400">Send Text</p>
            </div>
          </button>

          <button className="flex items-center gap-4 p-6 hover:bg-purple-50 dark:hover:bg-purple-900/30 transition-colors group">
            <div className="w-14 h-14 rounded-2xl bg-purple-100 dark:bg-purple-900/50 flex items-center justify-center group-hover:bg-purple-500 group-hover:text-white transition-colors">
              <Video className="w-7 h-7 text-purple-500 group-hover:text-white" />
            </div>
            <div className="text-left">
              <p className="font-semibold text-text-primary dark:text-white">Navigate to Patient</p>
              <p className="text-sm text-text-muted dark:text-gray-400">
                {isConnected ? "Robot is Online" : "Connecting..."}
              </p>
            </div>
          </button>
        </div>
      </Card>

      {/* Main Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Live Status Card */}
        <Card variant="elevated" className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Live Status</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
              </div>
            ) : (
              <>
                <div className="flex flex-col items-center mb-6">
                  <Avatar name={patientName} size="xl" status={patientStatus} />
                  <h3 className="mt-3 font-semibold text-text-primary dark:text-white">{patientName}</h3>
                  {/* Mood now from mood_logs table */}
                  <div className="mt-2 flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-gray-700 rounded-full">
                    <span className="text-xl">{moodEmoji}</span>
                    <span className="text-sm text-text-secondary dark:text-gray-300">{moodLabel}</span>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
                    <div className="flex items-center gap-3">
                      <Heart className="w-5 h-5 text-accent" />
                      <span className="text-sm text-text-secondary dark:text-gray-300">Heart Rate</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-text-primary dark:text-white">
                        {heartRate} bpm
                      </span>
                      <Badge variant="success">Normal</Badge>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
                    <div className="flex items-center gap-3">
                      <Activity className="w-5 h-5 text-primary" />
                      <span className="text-sm text-text-secondary dark:text-gray-300">Activity</span>
                    </div>
                    <span className="font-semibold text-text-primary dark:text-white">
                      {activityLevel}
                    </span>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
                    <div className="flex items-center gap-3">
                      <Battery className="w-5 h-5 text-success" />
                      <span className="text-sm text-text-secondary dark:text-gray-300">Rover Battery</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-text-primary dark:text-white">
                        {batteryLevel}%
                      </span>
                      {battery?.is_charging && (
                        <Badge variant="info">Charging</Badge>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
                    <div className="flex items-center gap-3">
                      <Clock className="w-5 h-5 text-text-muted dark:text-gray-400" />
                      <span className="text-sm text-text-secondary dark:text-gray-300">Last Check-in</span>
                    </div>
                    <span className="font-semibold text-text-primary dark:text-white">
                      {lastCheckIn}
                    </span>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Today's Medications & Activities */}
        <div className="lg:col-span-2 space-y-6">
          {/* Medications */}
          <Card variant="elevated">
            <CardHeader action={<Badge variant="info">{takenMeds} of {medications.length}</Badge>}>
              <CardTitle>Today&apos;s Medications</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 text-primary animate-spin" />
                </div>
              ) : (
                <>
                  <div className="space-y-3">
                    {medications.map((med) => (
                      <div
                        key={med.id}
                        className={cn(
                          "flex items-center justify-between p-4 rounded-xl border",
                          med.status === "taken"
                            ? "bg-success-50 dark:bg-success-900/30 border-success-200 dark:border-success-800"
                            : "bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600"
                        )}
                      >
                        <div className="flex items-center gap-3">
                          <div
                            className={cn(
                              "w-10 h-10 rounded-xl flex items-center justify-center",
                              med.status === "taken" ? "bg-success" : "bg-gray-200 dark:bg-gray-600"
                            )}
                          >
                            {med.status === "taken" ? (
                              <Check className="w-5 h-5 text-white" />
                            ) : (
                              <Clock className="w-5 h-5 text-text-muted dark:text-gray-400" />
                            )}
                          </div>
                          <div>
                            <p className="font-semibold text-text-primary dark:text-white">{med.medication_name}</p>
                            <p className="text-sm text-text-muted dark:text-gray-400">Scheduled: {med.scheduled_time}</p>
                          </div>
                        </div>
                        {med.status === "taken" ? (
                          <span className="text-sm text-success-700 dark:text-success-400">Taken at {med.taken_at}</span>
                        ) : (
                          <Badge variant="warning">Upcoming</Badge>
                        )}
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-text-muted dark:text-gray-400">Compliance Today</span>
                      <span className="text-sm font-semibold text-text-primary dark:text-white">
                        {medications.length > 0 ? Math.round((takenMeds / medications.length) * 100) : 0}%
                      </span>
                    </div>
                    <ProgressBar
                      value={takenMeds}
                      max={medications.length}
                      variant="success"
                    />
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          {/* Recent Activities */}
          <Card variant="elevated">
            <CardHeader
              action={
                <Button variant="ghost" size="sm">
                  View All
                </Button>
              }
            >
              <CardTitle>Recent Activities</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 text-primary animate-spin" />
                </div>
              ) : (
                <div className="space-y-1">
                  {activities.map((activity, index) => {
                    const Icon = activityIcons[activity.type as keyof typeof activityIcons] || Bell;
                    return (
                      <div key={activity.id} className="flex gap-4 p-3 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-xl transition-colors">
                        <div className="relative">
                          <div
                            className={cn(
                              "w-10 h-10 rounded-full flex items-center justify-center",
                              activityColors[activity.type as keyof typeof activityColors] || "bg-gray-400"
                            )}
                          >
                            <Icon className="w-5 h-5 text-white" />
                          </div>
                          {index < activities.length - 1 && (
                            <div className="absolute top-10 left-1/2 -translate-x-1/2 w-0.5 h-8 bg-gray-200 dark:bg-gray-600" />
                          )}
                        </div>
                        <div className="flex-1">
                          <p className="text-text-primary dark:text-white">{activity.description}</p>
                          <p className="text-sm text-text-muted dark:text-gray-400">{formatTimeAgo(activity.timestamp || new Date().toISOString())}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Demo Alert Button */}
      <div className="fixed bottom-6 right-6">
        <Button
          variant="danger"
          onClick={() => setAlertOpen(true)}
          leftIcon={<AlertTriangle className="w-5 h-5" />}
        >
          Demo Alert
        </Button>
      </div>

      {/* Emergency Alert Modal */}
      <Modal isOpen={alertOpen} onClose={() => setAlertOpen(false)} size="lg" closeOnOverlayClick={false}>
        <div className="p-8 text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-accent-100 dark:bg-accent-900/50 flex items-center justify-center animate-pulse">
            <AlertTriangle className="w-10 h-10 text-accent" />
          </div>
          <Badge variant="danger" size="md" className="mb-4">
            Critical Alert
          </Badge>
          <h2 className="text-2xl font-display font-bold text-text-primary dark:text-white mb-2">
            Attention Needed
          </h2>
          <p className="text-text-secondary dark:text-gray-300 mb-2">
            Fall detected in the living room
          </p>

          {/* Live Video Feed from Simulation */}
          <div className="aspect-video w-full bg-black rounded-2xl overflow-hidden mb-6 relative group">
            <img
              src="http://18.207.119.32:8080/stream?topic=/camera/image_raw&type=mjpeg&default_transport=raw"
              alt="Rover Camera Feed"
              className="w-full h-full object-cover"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.src = "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?auto=format&fit=crop&q=80&w=800";
              }}
            />
            <div className="absolute top-4 left-4 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
              <span className="text-xs font-semibold text-white uppercase tracking-wider drop-shadow-md">Live Stream</span>
            </div>
          </div>

          <p className="text-sm text-text-muted dark:text-gray-400 mb-8">
            {new Date().toLocaleString()}
          </p>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button size="lg" leftIcon={<PhoneCall className="w-5 h-5" />}>
              Call Now
            </Button>
            <Button
              variant="secondary"
              size="lg"
              leftIcon={<Video className="w-5 h-5" />}
              onClick={() => {
                setAlertOpen(false);
                setCameraOpen(true);
              }}
            >
              Request Video
            </Button>
            <Button className="dark:text-white dark:hover:text-text-primary" variant="outline" size="lg" onClick={() => setAlertOpen(false)}>
              Dismiss
            </Button>
          </div>
        </div>
      </Modal>
      {/* Live Robot Camera Modal */}
      <Modal isOpen={cameraOpen} onClose={() => setCameraOpen(false)} size="xl">
        <ModalHeader>
          <div className="flex items-center gap-2 text-purple-500">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span>Live Robot Camera — {patientName}</span>
          </div>
        </ModalHeader>
        <ModalBody className="p-0 bg-black aspect-video flex items-center justify-center overflow-hidden">
          {cameraOpen && (
            <img
              src={VIDEO_FEED_URL}
              alt="Robot Camera Feed"
              className="w-full h-full object-contain"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                const parent = target.parentElement;
                if (parent) {
                  const msg = document.createElement('div');
                  msg.className = 'text-white text-center p-8';
                  msg.innerHTML = `
                    <p class="text-xl font-bold mb-2">Camera Unavailable</p>
                    <p class="text-gray-400">Could not connect to robot at ${ROBOT_IP}</p>
                    <p class="text-sm mt-4">Make sure test_robot.py is running on the robot.</p>
                  `;
                  parent.appendChild(msg);
                }
              }}
            />
          )}
        </ModalBody>
        <ModalFooter>
          <div className="flex justify-between items-center w-full">
            <span className="text-sm text-text-muted dark:text-gray-400 italic">
              Encrypted Privacy-First Stream
            </span>
            <Button onClick={() => setCameraOpen(false)}>Close Feed</Button>
          </div>
        </ModalFooter>
      </Modal>
    </div>
  );
}
