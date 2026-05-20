"use client";

import { useState, useEffect } from "react";
import {
  Heart,
  Activity,
  Thermometer,
  Moon,
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Phone,
  Video,
  Plus,
  Check,
  Clock,
  AlertTriangle,
  Pill,
  FileText,
  ChevronRight,
  Loader2,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Avatar, ProgressBar, Modal, ModalHeader, ModalBody, ModalFooter, Input } from "@/components/ui";
import { cn } from "@/lib/utils";
import {
  getVitals,
  getMedications,
  getActivities,
  getSleepLogs,
  type VitalSign,
  type MedicationSchedule,
  type ActivityLog,
  type SleepLog,
} from "@/lib/dashboard-api";

// ---------------------------------------------------------------------------
// Types for display
// ---------------------------------------------------------------------------
interface VitalDisplay {
  id: string;
  label: string;
  value: number;
  unit: string;
  status: string;
  trend: string;
  icon: typeof Heart;
  color: string;
  range: string;
}

interface MedicationDisplay {
  id: string;
  name: string;
  dosage: string;
  time: string;
  status: string;
  takenAt?: string;
}

interface AlertDisplay {
  id: string;
  type: string;
  description: string;
  time: string;
  priority: string;
}

// ---------------------------------------------------------------------------
// Fallback mock data — used ONLY when backend returns empty/error
// ---------------------------------------------------------------------------
const FALLBACK_VITALS: VitalDisplay[] = [
  { id: "heartRate", label: "Heart Rate", value: 72, unit: "bpm", status: "normal", trend: "stable", icon: Heart, color: "accent", range: "60-100 bpm" },
  { id: "bloodOxygen", label: "Blood Oxygen", value: 98, unit: "%", status: "normal", trend: "up", icon: Activity, color: "primary", range: "95-100%" },
  { id: "temperature", label: "Temperature", value: 36.6, unit: "°C", status: "normal", trend: "stable", icon: Thermometer, color: "secondary", range: "36-37.5°C" },
  { id: "sleepQuality", label: "Sleep Last Night", value: 7.5, unit: "hrs", status: "normal", trend: "up", icon: Moon, color: "purple", range: "7-9 hours" },
];

const FALLBACK_MEDICATIONS: MedicationDisplay[] = [
  { id: "1", name: "Lisinopril", dosage: "10mg", time: "8:00 AM", status: "taken", takenAt: "8:05 AM" },
  { id: "2", name: "Metformin", dosage: "500mg", time: "12:00 PM", status: "taken", takenAt: "12:10 PM" },
  { id: "3", name: "Aspirin", dosage: "81mg", time: "6:00 PM", status: "upcoming" },
  { id: "4", name: "Metformin", dosage: "500mg", time: "8:00 PM", status: "upcoming" },
];

const FALLBACK_ACTIVITIES: AlertDisplay[] = [
  { id: "1", type: "vital", description: "Blood pressure slightly elevated", time: "1h ago", priority: "medium" },
  { id: "2", type: "medication", description: "Metformin taken on time", time: "2h ago", priority: "low" },
  { id: "3", type: "alert", description: "Missed medication reminder sent", time: "Yesterday", priority: "high" },
];

function mapApiToVitals(apiVitals: VitalSign[], sleepData?: SleepLog | null): VitalDisplay[] {
  if (!apiVitals || apiVitals.length === 0) return FALLBACK_VITALS;
  const latest = apiVitals[0];
  const prev = apiVitals.length > 1 ? apiVitals[1] : null;

  function trend(curr?: number, p?: number) {
    if (curr === undefined || p === undefined) return "stable";
    if (curr > p) return "up";
    if (curr < p) return "down";
    return "stable";
  }

  const vitals: VitalDisplay[] = [];
  if (latest.heart_rate != null) {
    vitals.push({ id: "heartRate", label: "Heart Rate", value: latest.heart_rate, unit: "bpm", status: "normal", trend: trend(latest.heart_rate, prev?.heart_rate), icon: Heart, color: "accent", range: "60-100 bpm" });
  }

  const spo2 = latest.spo2 ?? latest.blood_oxygen;
  const prevSpo2 = prev?.spo2 ?? prev?.blood_oxygen;
  if (spo2 != null) {
    vitals.push({ id: "bloodOxygen", label: "Blood Oxygen", value: Number(spo2), unit: "%", status: "normal", trend: trend(Number(spo2), prevSpo2 != null ? Number(prevSpo2) : undefined), icon: Activity, color: "primary", range: "95-100%" });
  }
  if (latest.temperature != null) {
    vitals.push({ id: "temperature", label: "Temperature", value: Number(latest.temperature), unit: "°C", status: "normal", trend: trend(Number(latest.temperature), prev?.temperature != null ? Number(prev.temperature) : undefined), icon: Thermometer, color: "secondary", range: "36-37.5°C" });
  }
  
  if (sleepData) {
    vitals.push({ id: "sleepQuality", label: "Sleep Last Night", value: Number(sleepData.duration_hours), unit: "hrs", status: "normal", trend: "stable", icon: Moon, color: "purple", range: "7-9 hours" });
  } else {
    vitals.push({ id: "sleepQuality", label: "Sleep Last Night", value: 7.5, unit: "hrs", status: "normal", trend: "up", icon: Moon, color: "purple", range: "7-9 hours" });
  }

  return vitals.length > 0 ? vitals : FALLBACK_VITALS;
}

function mapApiToMedications(apiMeds: MedicationSchedule[]): MedicationDisplay[] {
  return apiMeds.map((m) => ({
    id: m.id,
    name: m.medication_name || "Unknown",
    dosage: m.dosage || "",
    time: m.scheduled_time || "",
    status: m.status,
    takenAt: m.taken_at || undefined,
  }));
}

function mapApiToAlerts(apiActivities: ActivityLog[]): AlertDisplay[] {
  return apiActivities.slice(0, 5).map((a) => ({
    id: a.id,
    type: a.type,
    description: a.description,
    time: formatTimeAgo(a.timestamp),
    priority: a.priority || "low",
  }));
}

function formatTimeAgo(timestamp: string): string {
  const diff = Date.now() - new Date(timestamp).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return "Yesterday";
}

const trendIcons = {
  up: TrendingUp,
  down: TrendingDown,
  stable: Minus,
};

const colorClasses: Record<string, { bg: string; text: string; icon: string }> = {
  accent: { bg: "bg-accent-100 dark:bg-accent-900/30", text: "text-accent", icon: "bg-accent" },
  primary: { bg: "bg-primary-100 dark:bg-primary-900/30", text: "text-primary", icon: "bg-primary" },
  secondary: { bg: "bg-secondary-100 dark:bg-secondary-900/30", text: "text-secondary", icon: "bg-secondary" },
  purple: { bg: "bg-purple-100 dark:bg-purple-900/30", text: "text-purple-500", icon: "bg-purple-500" },
};

export default function MedicalDashboard() {
  const [scheduleModalOpen, setScheduleModalOpen] = useState(false);
  const [addMedModalOpen, setAddMedModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const [vitals, setVitals] = useState<VitalDisplay[]>(FALLBACK_VITALS);
  const [medications, setMedications] = useState<MedicationDisplay[]>(FALLBACK_MEDICATIONS);
  const [recentActivities, setRecentActivities] = useState<AlertDisplay[]>(FALLBACK_ACTIVITIES);

  useEffect(() => {
    async function fetchData() {
      setIsLoading(true);
      try {
        const [vitalsRes, medsRes, activitiesRes, sleepRes] = await Promise.all([
          getVitals(),
          getMedications(),
          getActivities(),
          getSleepLogs(),
        ]);

        const apiVitals = vitalsRes.status === "success" && vitalsRes.data ? vitalsRes.data : [];
        const latestSleep = (sleepRes.status === "success" && sleepRes.data?.length) ? sleepRes.data[0] : null;

        if (apiVitals.length > 0) {
          setVitals(mapApiToVitals(apiVitals, latestSleep));
        }

        if (medsRes.status === "success" && medsRes.data && medsRes.data.length > 0) {
          setMedications(mapApiToMedications(medsRes.data));
        }

        if (activitiesRes.status === "success" && activitiesRes.data && activitiesRes.data.length > 0) {
          setRecentActivities(mapApiToAlerts(activitiesRes.data));
        }
      } catch (error) {
        console.error("Failed to fetch medical dashboard data:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, []);

  const takenCount = medications.filter((m) => m.status === "taken").length;
  const compliancePercent = medications.length > 0 ? Math.round((takenCount / medications.length) * 100) : 0;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Vital Signs Grid */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-10 h-10 text-primary animate-spin" />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {vitals.map((vital) => {
              const TrendIcon = trendIcons[vital.trend as keyof typeof trendIcons] || Minus;
              const colors = colorClasses[vital.color] || colorClasses.primary;
              
              return (
                <Card key={vital.id} variant="elevated" className="hover:shadow-soft transition-shadow cursor-pointer">
                  <CardContent>
                    <div className="flex items-start justify-between mb-3">
                      <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center", colors.icon)}>
                        <vital.icon className="w-5 h-5 text-white" />
                      </div>
                      <div className={cn("flex items-center gap-1 text-sm", colors.text)}>
                        <TrendIcon className="w-4 h-4" />
                      </div>
                    </div>
                    <p className="text-2xl font-bold text-text-primary dark:text-white">
                      {vital.value}
                      <span className="text-sm font-normal text-text-muted dark:text-gray-400 ml-1">{vital.unit}</span>
                    </p>
                    <p className="text-sm text-text-muted dark:text-gray-400 mt-1">{vital.label}</p>
                    <div className="flex items-center justify-between mt-3">
                      <Badge variant="success">Normal</Badge>
                      <span className="text-xs text-text-muted dark:text-gray-400">{vital.range}</span>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Main Grid */}
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Medications */}
            <Card variant="elevated" className="lg:col-span-2">
              <CardHeader
                action={
                  <Button size="sm" leftIcon={<Plus className="w-4 h-4" />} onClick={() => setAddMedModalOpen(true)}>
                    Add Medication
                  </Button>
                }
              >
                <CardTitle>Today&apos;s Medications</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {medications.map((med) => (
                    <div
                      key={med.id}
                      className={cn(
                        "flex items-center justify-between p-4 rounded-xl border",
                        med.status === "taken"
                          ? "bg-success-50 dark:bg-success-900/30 border-success-200 dark:border-success-800"
                          : "bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-700"
                      )}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={cn(
                            "w-10 h-10 rounded-xl flex items-center justify-center",
                            med.status === "taken" ? "bg-success" : "bg-gray-300 dark:bg-gray-600"
                          )}
                        >
                          {med.status === "taken" ? (
                            <Check className="w-5 h-5 text-white" />
                          ) : (
                            <Clock className="w-5 h-5 text-white" />
                          )}
                        </div>
                        <div>
                          <p className="font-semibold text-text-primary dark:text-white">
                            {med.name} <span className="font-normal text-text-muted dark:text-gray-400">({med.dosage})</span>
                          </p>
                          <p className="text-sm text-text-muted dark:text-gray-400">Scheduled: {med.time}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {med.status === "taken" ? (
                          <span className="text-sm text-success-700 dark:text-success-400">Taken at {med.takenAt}</span>
                        ) : (
                          <Badge variant="warning">Upcoming</Badge>
                        )}
                        <Button variant="ghost" size="sm">
                          <ChevronRight className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-text-muted dark:text-gray-400">Today&apos;s Compliance</span>
                    <span className="text-sm font-semibold text-text-primary dark:text-white">{compliancePercent}%</span>
                  </div>
                  <ProgressBar value={compliancePercent} variant="primary" />
                </div>
              </CardContent>
            </Card>

            {/* Actions & Alerts */}
            <div className="space-y-6">
              {/* Quick Actions */}
              <Card variant="elevated">
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button className="w-full justify-start text-text-primary dark:text-white dark:hover:text-text-primary" variant="outline" leftIcon={<Calendar className="w-5 h-5" />} onClick={() => setScheduleModalOpen(true)}>
                    Schedule Appointment
                  </Button>
                  <Button className="w-full justify-start text-text-primary dark:text-white dark:hover:text-text-primary" variant="outline" leftIcon={<Phone className="w-5 h-5" />}>
                    Call Caregiver
                  </Button>
                  <Button className="w-full justify-start text-text-primary dark:text-white dark:hover:text-text-primary" variant="outline" leftIcon={<Video className="w-5 h-5" />}>
                    Video Consultation
                  </Button>
                  <Button className="w-full justify-start text-text-primary dark:text-white dark:hover:text-text-primary" variant="outline" leftIcon={<FileText className="w-5 h-5" />}>
                    Generate Report
                  </Button>
                </CardContent>
              </Card>

              {/* Recent Alerts */}
              <Card variant="elevated">
                <CardHeader>
                  <CardTitle>Recent Alerts</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {recentActivities.map((activity) => (
                    <div
                      key={activity.id}
                      className={cn(
                        "p-3 rounded-xl border",
                        activity.priority === "high"
                          ? "bg-accent-50 dark:bg-accent-900/30 border-accent-200 dark:border-accent-800"
                          : activity.priority === "medium"
                          ? "bg-secondary-50 dark:bg-secondary-900/30 border-secondary-200 dark:border-secondary-800"
                          : "bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-700"
                      )}
                    >
                      <div className="flex items-start gap-3">
                        {activity.priority === "high" && (
                          <AlertTriangle className="w-5 h-5 text-accent flex-shrink-0" />
                        )}
                        <div className="flex-1">
                          <p className="text-sm font-medium text-text-primary dark:text-white">{activity.description}</p>
                          <p className="text-xs text-text-muted dark:text-gray-400 mt-1">{activity.time}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </div>
        </>
      )}

      {/* Schedule Appointment Modal */}
      <Modal isOpen={scheduleModalOpen} onClose={() => setScheduleModalOpen(false)} size="lg">
        <ModalHeader>Schedule Appointment</ModalHeader>
        <ModalBody className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <button className="p-4 rounded-xl border-2 border-primary bg-primary-50 dark:bg-primary-900/30 text-center">
              <Phone className="w-6 h-6 text-primary mx-auto mb-2" />
              <span className="text-sm font-medium text-primary">Phone Call</span>
            </button>
            <button className="p-4 rounded-xl border-2 border-gray-200 dark:border-gray-700 hover:border-primary-300 text-center transition-colors">
              <Video className="w-6 h-6 text-text-muted dark:text-gray-400 mx-auto mb-2" />
              <span className="text-sm font-medium text-text-secondary dark:text-gray-300">Video Call</span>
            </button>
            <button className="p-4 rounded-xl border-2 border-gray-200 dark:border-gray-700 hover:border-primary-300 text-center transition-colors">
              <Calendar className="w-6 h-6 text-text-muted dark:text-gray-400 mx-auto mb-2" />
              <span className="text-sm font-medium text-text-secondary dark:text-gray-300">In-Person</span>
            </button>
          </div>
          <Input label="Date" type="date" />
          <Input label="Time" type="time" />
          <div>
            <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Notes</label>
            <textarea
              className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-text-primary dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none resize-none placeholder:text-text-muted dark:placeholder:text-gray-400"
              rows={3}
              placeholder="Reason for appointment..."
            />
          </div>
        </ModalBody>
        <ModalFooter>
          <Button className="text-text-primary dark:text-white dark:hover:text-text-primary" variant="outline" onClick={() => setScheduleModalOpen(false)}>Cancel</Button>
          <Button onClick={() => setScheduleModalOpen(false)}>Schedule</Button>
        </ModalFooter>
      </Modal>

      {/* Add Medication Modal */}
      <Modal isOpen={addMedModalOpen} onClose={() => setAddMedModalOpen(false)} size="lg">
        <ModalHeader>Add Medication</ModalHeader>
        <ModalBody className="space-y-4">
          <Input label="Medication Name" placeholder="e.g., Lisinopril" />
          <div className="grid grid-cols-2 gap-4">
            <Input label="Dosage" placeholder="e.g., 10mg" />
            <Input label="Frequency" placeholder="e.g., Once daily" />
          </div>
          <Input label="Time" type="time" />
          <div>
            <label className="block text-sm font-medium text-text-secondary dark:text-gray-300 mb-2">Special Instructions</label>
            <textarea
              className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-text-primary dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none resize-none placeholder:text-text-muted dark:placeholder:text-gray-400"
              rows={3}
              placeholder="Take with food, avoid grapefruit, etc..."
            />
          </div>
        </ModalBody>
        <ModalFooter>
          <Button variant="outline" onClick={() => setAddMedModalOpen(false)}>Cancel</Button>
          <Button onClick={() => setAddMedModalOpen(false)}>Add Medication</Button>
        </ModalFooter>
      </Modal>
    </div>
  );
}
