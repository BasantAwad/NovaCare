/**
 * NovaCare — Dashboard API Client
 *
 * Communicates with the Flask auth-backend dashboard endpoints at
 * NEXT_PUBLIC_AUTH_API_URL (defaults to http://localhost:5001).
 *
 * Mirrors the pattern used by auth-api.ts.
 */

import { getAccessToken } from "./auth-api";
import { getDynamicUrl } from "./utils";

const AUTH_API = process.env.NEXT_PUBLIC_AUTH_API_URL || "http://localhost:5001";
const NOVABOT_API = process.env.NEXT_PUBLIC_NOVABOT_API_URL || "http://localhost:5000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VitalSign {
  id: string;
  rover_id: string;
  heart_rate?: number;
  spo2?: number;         // DB column name
  blood_oxygen?: number; // alias used by frontend
  temperature?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  measured_at: string;
}

export interface MedicationSchedule {
  id: string;
  rover_id: string;
  medication_id: string;
  medication_name: string;
  generic_name?: string;
  dosage: string;
  frequency: string;
  scheduled_time: string;
  scheduled_date?: string;
  instructions?: string;
  status: "taken" | "upcoming" | "missed" | "due";
  taken_at?: string;
  prescribed_by?: string;
  start_date?: string;
  end_date?: string;
  is_active: boolean;
}

export interface ActivityLog {
  id: string;
  rover_id: string;
  type: "medication" | "navigation" | "conversation" | "alert" | "vital" | "system";
  title: string;
  description: string;
  timestamp: string;
  priority?: "low" | "medium" | "high";
}

export interface LinkedRover {
  rover_id: string;
  user_id: string;
  first_name: string;
  last_name: string;
  status: "online" | "offline" | "resting";
  last_check_in?: string;
}

export interface MedicationComplianceStats {
  total_doses: number;
  taken_doses: number;
  missed_doses: number;
  upcoming_doses: number;
}

export interface SleepLog {
  id: string;
  rover_id: string;
  date: string;
  bed_time?: string;
  wake_time?: string;
  duration_hours: number;
  quality?: "poor" | "fair" | "good" | "excellent";
  deep_sleep_minutes?: number;
  light_sleep_minutes?: number;
  rem_sleep_minutes?: number;
  awakenings?: number;
  notes?: string;
}

export interface HydrationLog {
  id: string;
  rover_id: string;
  date: string;
  glasses: number;
  total_ml?: number;
  goal_glasses: number;
}

export interface WeightLog {
  id: string;
  rover_id: string;
  date: string;
  weight_kg: number;
  weight_lbs: number;
  target_weight_kg?: number;
  bmi?: number;
}

export interface BatteryStatus {
  id: string;
  rover_id: string;
  battery_percent: number;
  is_charging: boolean;
  estimated_remaining_minutes?: number;
  recorded_at: string;
}

export interface MoodLog {
  id: string;
  rover_id: string;
  date: string;
  mood: "very_sad" | "sad" | "neutral" | "happy" | "very_happy";
  energy_level?: "very_low" | "low" | "moderate" | "high" | "very_high";
  anxiety_level?: number;
  notes?: string;
  emoji?: string;
}

interface ApiResponse<T = unknown> {
  status: "success" | "error";
  data?: T;
  error?: string;
}

// ---------------------------------------------------------------------------
// Fetch wrapper (mirrors auth-api.ts pattern)
// ---------------------------------------------------------------------------

async function dashboardFetch<T = unknown>(
  path: string,
  options: RequestInit = {},
): Promise<ApiResponse<T>> {
  const url = `${getDynamicUrl(AUTH_API)}${path}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  const token = getAccessToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    const res = await fetch(url, { ...options, headers });
    const json = await res.json();
    return json as ApiResponse<T>;
  } catch {
    return { status: "error", error: "Could not connect to dashboard service." };
  }
}

// ---------------------------------------------------------------------------
// Dashboard API Functions
// ---------------------------------------------------------------------------

/** Fetch vital signs for the authenticated rover */
export async function getVitals(): Promise<ApiResponse<VitalSign[]>> {
  return dashboardFetch<VitalSign[]>("/api/dashboard/vitals");
}

/** Fetch user profile and role details */
export async function getProfile(): Promise<ApiResponse<Record<string, unknown>>> {
  return dashboardFetch<Record<string, unknown>>("/api/dashboard/profile");
}

/** Fetch medication schedules for the authenticated user's linked rover */
export async function getMedications(): Promise<ApiResponse<MedicationSchedule[]>> {
  return dashboardFetch<MedicationSchedule[]>("/api/dashboard/medications");
}

/** Fetch activity logs for the authenticated user's linked rover */
export async function getActivities(): Promise<ApiResponse<ActivityLog[]>> {
  return dashboardFetch<ActivityLog[]>("/api/dashboard/activities");
}

/** For caregivers: fetch linked rover's profile and status */
export async function getLinkedRover(): Promise<ApiResponse<LinkedRover>> {
  return dashboardFetch<LinkedRover>("/api/dashboard/linked-rover");
}

/** Fetch medication compliance stats */
export async function getMedicationStats(): Promise<ApiResponse<MedicationComplianceStats>> {
  return dashboardFetch<MedicationComplianceStats>("/api/dashboard/medication-stats");
}

/** Mark a medication as taken */
export async function markMedicationTaken(
  medicationScheduleId: string,
): Promise<ApiResponse<{ taken_at: string }>> {
  return dashboardFetch<{ taken_at: string }>("/api/dashboard/medications/take", {
    method: "POST",
    body: JSON.stringify({ schedule_id: medicationScheduleId }),
  });
}

/** Fetch medical notes */
export async function getMedicalNotes(): Promise<ApiResponse<Record<string, unknown>[]>> {
  return dashboardFetch<Record<string, unknown>[]>("/api/dashboard/notes");
}

/** Fetch sleep logs for the rover */
export async function getSleepLogs(): Promise<ApiResponse<SleepLog[]>> {
  return dashboardFetch<SleepLog[]>("/api/dashboard/sleep");
}

/** Fetch hydration logs for the rover */
export async function getHydrationLogs(): Promise<ApiResponse<HydrationLog[]>> {
  return dashboardFetch<HydrationLog[]>("/api/dashboard/hydration");
}

/** Log a glass of water for today */
export async function logHydration(): Promise<ApiResponse<{ glasses: number }>> {
  return dashboardFetch<{ glasses: number }>("/api/dashboard/hydration/log", {
    method: "POST",
  });
}

/** Fetch weight logs for the rover */
export async function getWeightLogs(): Promise<ApiResponse<WeightLog[]>> {
  return dashboardFetch<WeightLog[]>("/api/dashboard/weight");
}

/** Fetch latest rover battery status */
export async function getBatteryStatus(): Promise<ApiResponse<BatteryStatus>> {
  return dashboardFetch<BatteryStatus>("/api/dashboard/battery");
}

/** Fetch mood logs for the rover */
export async function getMoodLogs(): Promise<ApiResponse<MoodLog[]>> {
  return dashboardFetch<MoodLog[]>("/api/dashboard/mood");
}

/** Get navigation status from the LLM backend */
export async function getNavigationStatus(): Promise<{
  status: "success" | "error";
  data?: {
    destination: string | null;
    status: 'navigating' | 'idle' | 'completed' | 'cancelled';
    progress: number;
    follow_mode: boolean;
  };
}> {
  try {
    const res = await fetch(`${getDynamicUrl(NOVABOT_API)}/api/navigation`);
    if (!res.ok) throw new Error("Failed to fetch navigation status");
    return await res.json();
  } catch (error) {
    console.error("getNavigationStatus error:", error);
    return { status: "error" };
  }
}

/** Update navigation status on the LLM backend */
export async function updateNavigation(
  destination: string | null,
  status: 'navigating' | 'idle' | 'completed' | 'cancelled',
  followMode: boolean
): Promise<{ status: "success" | "error" }> {
  try {
    const res = await fetch(`${getDynamicUrl(NOVABOT_API)}/api/navigation`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        destination,
        status,
        follow_mode: followMode,
      }),
    });
    if (!res.ok) throw new Error("Failed to update navigation");
    return await res.json();
  } catch (error) {
    console.error("updateNavigation error:", error);
    return { status: "error" };
  }
}
