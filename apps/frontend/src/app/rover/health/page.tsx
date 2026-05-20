"use client";

import { useState, useEffect } from "react";
import { Heart, Activity, Thermometer, Moon, Droplets, Scale, ArrowLeft, TrendingUp, TrendingDown, Minus, Loader2 } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { getVitals, getSleepLogs, getHydrationLogs, getWeightLogs, logHydration, type VitalSign, type SleepLog, type HydrationLog, type WeightLog } from "@/lib/dashboard-api";

interface VitalDisplay {
  id: string;
  name: string;
  value: number;
  unit: string;
  status: "normal" | "low" | "high";
  trend: "up" | "down" | "stable";
  icon: typeof Heart;
  color: string;
  normalRange: string;
}

// ---------------------------------------------------------------------------
// Fallback mock data — used ONLY when backend returns empty/error
// ---------------------------------------------------------------------------
const FALLBACK_VITALS: VitalDisplay[] = [
  {
    id: "heartRate",
    name: "Heart Rate",
    value: 72,
    unit: "bpm",
    status: "normal",
    trend: "stable",
    icon: Heart,
    color: "bg-accent",
    normalRange: "60-100",
  },
  {
    id: "oxygen",
    name: "Blood Oxygen",
    value: 98,
    unit: "%",
    status: "normal",
    trend: "up",
    icon: Activity,
    color: "bg-primary",
    normalRange: "95-100",
  },
  {
    id: "temperature",
    name: "Temperature",
    value: 98.6,
    unit: "°F",
    status: "normal",
    trend: "stable",
    icon: Thermometer,
    color: "bg-secondary",
    normalRange: "97-99",
  },
  {
    id: "sleep",
    name: "Sleep Last Night",
    // TODO: Sleep quality/hours has no direct DB counterpart in vital_signs — add dedicated column or table
    value: 7.5,
    unit: "hours",
    status: "normal",
    trend: "up",
    icon: Moon,
    color: "bg-indigo-500",
    normalRange: "7-9",
  },
  {
    id: "hydration",
    name: "Hydration",
    // TODO: Hydration tracking has no direct DB counterpart — add dedicated tracking table
    value: 6,
    unit: "glasses",
    status: "low",
    trend: "stable",
    icon: Droplets,
    color: "bg-cyan-500",
    normalRange: "8+",
  },
  {
    id: "weight",
    name: "Weight",
    // TODO: Weight tracking has no direct DB counterpart in vital_signs — add dedicated column
    value: 145,
    unit: "lbs",
    status: "normal",
    trend: "down",
    icon: Scale,
    color: "bg-purple-500",
    normalRange: "Target: 140",
  },
];

function mapApiToVitals(
  apiVitals: VitalSign[],
  sleepData?: SleepLog | null,
  hydrationData?: HydrationLog | null,
  weightData?: WeightLog | null
): VitalDisplay[] {
  if (!apiVitals || apiVitals.length === 0) return FALLBACK_VITALS;

  const latest = apiVitals[0];
  const previous = apiVitals.length > 1 ? apiVitals[1] : null;

  function determineTrend(current: number | undefined, prev: number | undefined): "up" | "down" | "stable" {
    if (current === undefined || prev === undefined) return "stable";
    if (current > prev) return "up";
    if (current < prev) return "down";
    return "stable";
  }

  function determineStatus(value: number, low: number, high: number): "normal" | "low" | "high" {
    if (value < low) return "low";
    if (value > high) return "high";
    return "normal";
  }

  const vitals: VitalDisplay[] = [];

  // Heart Rate
  if (latest.heart_rate !== undefined && latest.heart_rate !== null) {
    vitals.push({
      id: "heartRate", name: "Heart Rate", value: latest.heart_rate, unit: "bpm",
      status: determineStatus(latest.heart_rate, 60, 100),
      trend: determineTrend(latest.heart_rate, previous?.heart_rate),
      icon: Heart, color: "bg-accent", normalRange: "60-100",
    });
  }

  // Blood Oxygen (DB column is spo2)
  const spo2 = latest.spo2 ?? latest.blood_oxygen;
  const prevSpo2 = previous?.spo2 ?? previous?.blood_oxygen;
  if (spo2 !== undefined && spo2 !== null) {
    vitals.push({
      id: "oxygen", name: "Blood Oxygen", value: Number(spo2), unit: "%",
      status: determineStatus(Number(spo2), 95, 101),
      trend: determineTrend(Number(spo2), prevSpo2 != null ? Number(prevSpo2) : undefined),
      icon: Activity, color: "bg-primary", normalRange: "95-100",
    });
  }

  // Temperature
  if (latest.temperature !== undefined && latest.temperature !== null) {
    vitals.push({
      id: "temperature", name: "Temperature", value: Number(latest.temperature), unit: "°C",
      status: determineStatus(Number(latest.temperature), 36, 37.5),
      trend: determineTrend(Number(latest.temperature), previous?.temperature != null ? Number(previous.temperature) : undefined),
      icon: Thermometer, color: "bg-secondary", normalRange: "36-37.5",
    });
  }

  // Sleep — from sleep_logs table
  if (sleepData) {
    vitals.push({
      id: "sleep", name: "Sleep Last Night",
      value: Number(sleepData.duration_hours), unit: "hours",
      status: determineStatus(Number(sleepData.duration_hours), 7, 9.5),
      trend: "stable", icon: Moon, color: "bg-indigo-500", normalRange: "7-9",
    });
  } else {
    vitals.push({
      id: "sleep", name: "Sleep Last Night", value: 7.5, unit: "hours",
      status: "normal", trend: "up", icon: Moon, color: "bg-indigo-500", normalRange: "7-9",
    });
  }

  // Hydration — from hydration_logs table
  if (hydrationData) {
    vitals.push({
      id: "hydration", name: "Hydration",
      value: hydrationData.glasses, unit: "glasses",
      status: hydrationData.glasses >= hydrationData.goal_glasses ? "normal" : "low",
      trend: "stable", icon: Droplets, color: "bg-cyan-500",
      normalRange: `${hydrationData.goal_glasses}+`,
    });
  } else {
    vitals.push({
      id: "hydration", name: "Hydration", value: 6, unit: "glasses",
      status: "low", trend: "stable", icon: Droplets, color: "bg-cyan-500", normalRange: "8+",
    });
  }

  // Weight — from weight_logs table
  if (weightData) {
    vitals.push({
      id: "weight", name: "Weight",
      value: Number(weightData.weight_lbs), unit: "lbs",
      status: "normal",
      trend: weightData.target_weight_kg && Number(weightData.weight_kg) > Number(weightData.target_weight_kg) ? "down" : "stable",
      icon: Scale, color: "bg-purple-500",
      normalRange: weightData.target_weight_kg ? `Target: ${Math.round(Number(weightData.target_weight_kg) * 2.2)} lbs` : "",
    });
  } else {
    vitals.push({
      id: "weight", name: "Weight", value: 145, unit: "lbs",
      status: "normal", trend: "down", icon: Scale, color: "bg-purple-500", normalRange: "Target: 140",
    });
  }

  return vitals.length > 0 ? vitals : FALLBACK_VITALS;
}

const trendIcons = {
  up: TrendingUp,
  down: TrendingDown,
  stable: Minus,
};

const trendColors = {
  up: "text-success",
  down: "text-accent",
  stable: "text-text-muted",
};

export default function HealthPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [vitals, setVitals] = useState<VitalDisplay[]>(FALLBACK_VITALS);
  const [hydrationGlasses, setHydrationGlasses] = useState(6);
  const [hydrationGoal, setHydrationGoal] = useState(8);
  const [isLoggingWater, setIsLoggingWater] = useState(false);

  useEffect(() => {
    async function fetchAll() {
      setIsLoading(true);
      try {
        const [vitalsRes, sleepRes, hydrationRes, weightRes] = await Promise.all([
          getVitals(),
          getSleepLogs(),
          getHydrationLogs(),
          getWeightLogs(),
        ]);

        const apiVitals = vitalsRes.status === "success" ? vitalsRes.data : null;
        const latestSleep = (sleepRes.status === "success" && sleepRes.data?.length) ? sleepRes.data[0] : null;
        const todayHydration = (hydrationRes.status === "success" && hydrationRes.data?.length) ? hydrationRes.data[0] : null;
        const latestWeight = (weightRes.status === "success" && weightRes.data?.length) ? weightRes.data[0] : null;

        if (todayHydration) {
          setHydrationGlasses(todayHydration.glasses);
          setHydrationGoal(todayHydration.goal_glasses);
        }

        if (apiVitals) {
          setVitals(mapApiToVitals(apiVitals, latestSleep, todayHydration, latestWeight));
        }
      } catch (error) {
        console.error("Failed to fetch health data:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchAll();
  }, []);

  const abnormalCount = vitals.filter((v) => v.status !== "normal").length;

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link
          href="/rover"
          className="rover-btn w-14 h-14 rounded-2xl bg-gray-100 dark:bg-gray-800 flex items-center justify-center hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          <ArrowLeft className="w-6 h-6 text-text-secondary dark:text-gray-400" />
        </Link>
        <div>
          <h1 className="text-3xl font-display font-bold text-text-primary dark:text-white">Health Check</h1>
          <p className="text-text-muted dark:text-gray-400">Your current vital signs</p>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-10 h-10 text-primary animate-spin" />
        </div>
      ) : (
        <>
          {/* Overall Status */}
          <div className={cn(
            "border-2 rounded-3xl p-8 text-center",
            abnormalCount === 0
              ? "bg-success-50 dark:bg-success-900/30 border-success"
              : "bg-secondary-50 dark:bg-secondary-900/30 border-secondary"
          )}>
            <div className={cn(
              "w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center",
              abnormalCount === 0 ? "bg-success" : "bg-secondary"
            )}>
              <Heart className="w-10 h-10 text-white" />
            </div>
            <h2 className={cn(
              "text-3xl font-bold mb-2",
              abnormalCount === 0 ? "text-success" : "text-secondary"
            )}>
              {abnormalCount === 0 ? "You're Doing Great! 💪" : "Some Vitals Need Attention ⚠️"}
            </h2>
            <p className="text-lg text-text-secondary dark:text-gray-300">
              {abnormalCount === 0
                ? "Most of your vitals are within normal range"
                : `${abnormalCount} vital(s) outside normal range`}
            </p>
          </div>

          {/* Vitals Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {vitals.map((vital) => {
              const TrendIcon = trendIcons[vital.trend as keyof typeof trendIcons];
              const trendColor = trendColors[vital.trend as keyof typeof trendColors];

              return (
                <div
                  key={vital.id}
                  className={cn(
                    "rover-card bg-white dark:bg-gray-800 rounded-3xl p-6 border-2 transition-all",
                    vital.status === "normal" ? "border-gray-100 dark:border-gray-700" : "border-accent bg-accent-50 dark:bg-accent-900/30"
                  )}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center", vital.color)}>
                      <vital.icon className="w-7 h-7 text-white" />
                    </div>
                    <div className={cn("flex items-center gap-1", trendColor)}>
                      <TrendIcon className="w-5 h-5" />
                    </div>
                  </div>

                  <div className="mb-2">
                    <span className="text-4xl font-bold text-text-primary dark:text-white">{vital.value}</span>
                    <span className="text-xl text-text-muted dark:text-gray-400 ml-1">{vital.unit}</span>
                  </div>

                  <h3 className="text-lg font-medium text-text-secondary dark:text-gray-300 mb-1">{vital.name}</h3>
                  
                  <div className="flex items-center justify-between">
                    <span className={cn(
                      "text-sm font-medium px-3 py-1 rounded-full",
                      vital.status === "normal" 
                        ? "bg-success-100 dark:bg-success-900/50 text-success" 
                        : "bg-accent-100 dark:bg-accent-900/50 text-accent"
                    )}>
                      {vital.status === "normal" ? "Normal" : vital.status === "low" ? "Low" : "High"}
                    </span>
                    <span className="text-sm text-text-muted dark:text-gray-400">{vital.normalRange}</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Hydration Reminder */}
          <div className="bg-cyan-50 dark:bg-cyan-900/30 border-2 border-cyan-200 dark:border-cyan-800 rounded-3xl p-6 flex items-center gap-4">
            <div className="w-16 h-16 bg-cyan-500 rounded-2xl flex items-center justify-center">
              <Droplets className="w-8 h-8 text-white" />
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-cyan-700 dark:text-cyan-400">Stay Hydrated! 💧</h3>
              <p className="text-cyan-600 dark:text-cyan-300">
                You&apos;ve had {hydrationGlasses} glasses today.
                {hydrationGlasses < hydrationGoal
                  ? ` Try to drink ${hydrationGoal - hydrationGlasses} more!`
                  : " Great job — goal reached! 🎉"}
              </p>
            </div>
            <button
              className="rover-btn px-6 py-3 bg-cyan-500 text-white rounded-2xl font-semibold hover:bg-cyan-600 transition-colors disabled:opacity-50"
              disabled={isLoggingWater}
              onClick={async () => {
                setIsLoggingWater(true);
                try {
                  const res = await logHydration();
                  if (res.status === "success" && res.data) {
                    setHydrationGlasses(res.data.glasses);
                  }
                } catch (err) {
                  console.error("Failed to log water:", err);
                } finally {
                  setIsLoggingWater(false);
                }
              }}
            >
              {isLoggingWater ? "Logging..." : "Log Water"}
            </button>
          </div>
        </>
      )}

      {/* Last Updated */}
      <p className="text-center text-text-muted dark:text-gray-400 text-lg">
        Last updated: {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </p>
    </div>
  );
}
