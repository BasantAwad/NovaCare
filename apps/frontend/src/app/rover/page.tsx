"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { MessageCircle, Pill, Navigation, AlertTriangle, Heart, Music, Smile, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import EmotionDetectionModal from "@/components/EmotionDetectionModal";
import { useAuth } from "@/context/AuthContext";
import { getVitals, getMedications, getBatteryStatus, type VitalSign, type MedicationSchedule, type BatteryStatus } from "@/lib/dashboard-api";

/** Per-tile accent: darker on light UI (readable on bright gradient stops), lighter on dark UI. */
const mainFeatures = [
  {
    href: "/rover/talk",
    icon: MessageCircle,
    label: "Talk to Nova",
    description: "Chat or speak with your AI assistant",
    color: "from-primary to-primary-600",
    copy: {
      icon: "text-teal-50 dark:text-cyan-200",
      title: "text-teal-50 dark:text-cyan-50",
      desc: "text-teal-100/95 dark:text-teal-100/90",
    },
  },
  {
    href: "/rover/medications",
    icon: Pill,
    label: "Medications",
    description: "View schedule and reminders",
    color: "from-purple-400 to-purple-600",
    textColor: "text-success",
    // Badge will be set dynamically
    badgeKey: "medsDue",
    copy: {
      icon: "text-violet-900 dark:text-fuchsia-200",
      title: "text-violet-950 dark:text-violet-50",
      desc: "text-purple-900/90 dark:text-purple-100/90",
    },
  },
  {
    href: "/rover/navigate",
    icon: Navigation,
    label: "Navigate",
    description: "Go somewhere or follow me",
    color: "from-secondary to-secondary-600",
    copy: {
      icon: "text-amber-950 dark:text-amber-200",
      title: "text-orange-950 dark:text-amber-50",
      desc: "text-orange-900/90 dark:text-orange-50/95",
    },
  },
  {
    href: "/rover/emergency",
    icon: AlertTriangle,
    label: "Emergency",
    description: "Get help immediately",
    color: "from-accent to-accent-600",
    copy: {
      icon: "text-rose-950 dark:text-rose-200",
      title: "text-rose-950 dark:text-rose-50",
      desc: "text-rose-900/90 dark:text-pink-100/90",
    },
  },
  {
    href: "/rover/health",
    icon: Heart,
    label: "Health Check",
    description: "View your vital signs",
    color: "from-success to-success-600",
    copy: {
      icon: "text-emerald-950 dark:text-lime-200",
      title: "text-emerald-950 dark:text-emerald-50",
      desc: "text-green-900/90 dark:text-green-100/90",
    },
  },
  {
    href: "/rover/entertainment",
    icon: Music,
    label: "Entertainment",
    description: "Music, videos, and games",
    color: "from-indigo-400 to-indigo-600",
    copy: {
      icon: "text-indigo-950 dark:text-sky-200",
      title: "text-indigo-950 dark:text-indigo-50",
      desc: "text-indigo-900/90 dark:text-blue-100/90",
    },
  },
];

export default function RoverHomePage() {
  const [isEmotionModalOpen, setIsEmotionModalOpen] = useState(false);
  const { user } = useAuth();
  const userName = user?.first_name || "Friend";

  // Dynamic state from backend
  const [heartRate, setHeartRate] = useState<number>(72);
  const [medsToday, setMedsToday] = useState<number>(2);
  const [battery, setBattery] = useState<BatteryStatus | null>(null);
  const roverStatus = battery
    ? `${battery.battery_percent}% ${battery.is_charging ? "⚡" : "🔋"}`
    : "Ready";

  useEffect(() => {
    async function fetchQuickStats() {
      try {
        const [vitalsRes, medsRes, batteryRes] = await Promise.all([
          getVitals(),
          getMedications(),
          getBatteryStatus(),
        ]);

        if (vitalsRes.status === "success" && vitalsRes.data && vitalsRes.data.length > 0) {
          const latest = vitalsRes.data[0];
          if (latest.heart_rate) {
            setHeartRate(latest.heart_rate);
          }
        }

        if (medsRes.status === "success" && medsRes.data) {
          const todayMeds = medsRes.data.filter(
            (m: MedicationSchedule) => m.status === "due" || m.status === "upcoming"
          );
          setMedsToday(todayMeds.length);
        }

        if (batteryRes.status === "success" && batteryRes.data) {
          setBattery(batteryRes.data);
        }
      } catch (error) {
        console.error("Failed to fetch rover quick stats:", error);
      }
    }

    fetchQuickStats();
    const interval = setInterval(fetchQuickStats, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in">
      {/* Emotion Detection Modal */}
      <EmotionDetectionModal
        isOpen={isEmotionModalOpen}
        onClose={() => setIsEmotionModalOpen(false)}
      />
      {/* Greeting */}
      <div className="text-center">
        <h1 className="text-4xl font-display font-bold text-text-primary dark:text-white mb-2">
          Hello, {userName}! 👋
        </h1>
        <p className="text-xl text-text-muted dark:text-gray-400">How can I help you today?</p>
      </div>

      {/* Main Feature Grid - 2x3 Large Touch Targets */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
        {mainFeatures.map((feature) => (
          <Link
            key={feature.href}
            href={feature.href}
            className={cn(
              "rover-card relative overflow-hidden rounded-3xl p-8 transition-all transform hover:scale-[1.02] active:scale-[0.98]",
              "bg-gradient-to-br",
              feature.color,
              feature.href === "/rover/emergency" && "col-span-1 row-span-1"
            )}
            style={{ minHeight: "200px" }}
          >
            {/* Badge — dynamic for medications */}
            {feature.badgeKey === "medsDue" && medsToday > 0 && (
              <span className="absolute top-4 right-4 px-3 py-1 bg-white/20 backdrop-blur-sm rounded-full text-sm font-medium text-white">
                {medsToday} Due
              </span>
            )}

            <div className="flex h-full flex-col justify-between drop-shadow-[0_1px_2px_rgba(0,0,0,0.25)] dark:drop-shadow-[0_1px_3px_rgba(0,0,0,0.5)]">
              <div
                className={cn(
                  "mb-4 flex h-16 w-16 items-center justify-center rounded-2xl",
                  "bg-white/20 backdrop-blur-sm"
                )}
              >
                <feature.icon className={cn("h-8 w-8", feature.copy.icon)} />
              </div>
              <div>
                <h2 className={cn("mb-2 text-2xl font-bold", feature.copy.title)}>{feature.label}</h2>
                <p className={cn("text-base leading-snug", feature.copy.desc)}>{feature.description}</p>
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Quick Status Bar */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-soft border border-gray-100 dark:border-gray-700 text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Heart className="w-6 h-6 text-accent" />
            <span className="text-3xl font-bold text-text-primary dark:text-white">{heartRate}</span>
          </div>
          <p className="text-text-muted dark:text-gray-400">Heart Rate</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-soft border border-gray-100 dark:border-gray-700 text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Pill className="w-6 h-6 text-purple-500" />
            <span className="text-3xl font-bold text-text-primary dark:text-white">{medsToday}</span>
          </div>
          <p className="text-text-muted dark:text-gray-400">Medications Today</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-soft border border-gray-100 dark:border-gray-700 text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Navigation className="w-6 h-6 text-secondary" />
            {/* TODO: Rover hardware status has no direct DB counterpart — integrate with rover hardware API */}
            <span className="text-3xl font-bold text-success">{roverStatus}</span>
          </div>
          <p className="text-text-muted dark:text-gray-400">Rover Status</p>
        </div>
        {/* Detect Emotion Button */}
        <button
          onClick={() => setIsEmotionModalOpen(true)}
          className="bg-gradient-to-br from-pink-500 to-purple-600 rounded-2xl p-6 shadow-soft text-center transition-all transform hover:scale-[1.02] active:scale-[0.98] cursor-pointer"
        >
          <div className="flex items-center justify-center gap-2 mb-2">
            <Smile className="w-6 h-6 text-white" />
          </div>
          <p className="text-white font-semibold">Detect Emotion</p>
        </button>
      </div>

      {/* Voice Activation Hint */}
      <div className="bg-primary-50 dark:bg-primary-900/30 rounded-2xl p-6 text-center border border-primary-100 dark:border-primary-800">
        <p className="text-lg text-primary">
          💡 <span className="font-medium">Tip:</span> You can say "Hey Nova" to talk to me anytime!
        </p>
      </div>
    </div>
  );
}
