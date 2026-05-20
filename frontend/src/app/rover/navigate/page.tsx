"use client";

import { useState, useEffect, useCallback } from "react";
import { Navigation, MapPin, Home, Utensils, Bath, Sofa, Bed, Coffee, ArrowLeft, Play, Pause, X, UserCheck, AlertTriangle, Loader2 } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  checkRobotHealth,
  moveRobot,
  stopRobot,
  navigateRobot,
  startFollow,
  stopFollow,
  type RobotHealth,
} from "@/lib/robot-api";
import { getNavigationStatus, updateNavigation } from "@/lib/dashboard-api";

interface Destination {
  id: string;
  name: string;
  icon: any;
  color: string;
  distance: string;
}

const destinations: Destination[] = [
  { id: "kitchen", name: "Kitchen", icon: Utensils, color: "bg-secondary", distance: "15m" },
  { id: "bathroom", name: "Bathroom", icon: Bath, color: "bg-primary", distance: "8m" },
  { id: "living", name: "Living Room", icon: Sofa, color: "bg-purple-500", distance: "10m" },
  { id: "bedroom", name: "Bedroom", icon: Bed, color: "bg-indigo-500", distance: "12m" },
  { id: "dining", name: "Dining Room", icon: Coffee, color: "bg-success", distance: "18m" },
  { id: "entrance", name: "Front Door", icon: Home, color: "bg-accent", distance: "20m" },
];

export default function NavigatePage() {
  const [selectedDest, setSelectedDest] = useState<Destination | null>(null);
  const [isNavigating, setIsNavigating] = useState(false);
  const [followMode, setFollowMode] = useState(false);
  const [robotConnected, setRobotConnected] = useState<boolean | null>(null);
  const [robotHealth, setRobotHealth] = useState<RobotHealth | null>(null);
  const [navStatus, setNavStatus] = useState<string>("");
  const [navETA, setNavETA] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  // Check robot health on mount
  useEffect(() => {
    const check = async () => {
      try {
        const health = await checkRobotHealth();
        setRobotHealth(health);
        setRobotConnected(health.status === "healthy");
      } catch {
        setRobotConnected(false);
      }
    };
    check();
    // Re-check every 5 seconds
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  const startNavigation = useCallback(async () => {
    if (!selectedDest) return;
    setIsLoading(true);
    setNavStatus("Starting navigation...");

    try {
      const result = await navigateRobot({ destination: selectedDest.id });
      setIsNavigating(true);
      setNavStatus(`Navigating to ${selectedDest.name}...`);
      setNavETA(result.estimated_duration_s);
    } catch (err) {
      console.error("[Navigate] Error:", err);
      setNavStatus("Failed to start navigation. Check robot connection.");
    } finally {
      setIsLoading(false);
      setProgress(10);
      try {
        await updateNavigation(selectedDest.id, 'navigating', false);
      } catch (e) {
        console.error(e);
      }
    }
  }, [selectedDest]);

  const stopNavigation = useCallback(async () => {
    setIsLoading(true);
    try {
      await stopRobot();
      setIsNavigating(false);
      setSelectedDest(null);
      setProgress(0);
      try {
        await updateNavigation(null, 'idle', false);
      } catch (e) {
        console.error(e);
      }
      setNavStatus("Navigation stopped.");
    } catch (err) {
      console.error("[Navigate] Stop error:", err);
      setNavStatus("Failed to stop. Try again.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const toggleFollowMode = useCallback(async () => {
    setIsLoading(true);
    try {
      if (followMode) {
        await stopFollow();
        setFollowMode(false);
        setNavStatus("Follow mode stopped.");
      } else {
        await startFollow();
        setFollowMode(true);
        setIsNavigating(false);
        setSelectedDest(null);
        setNavStatus("Following you...");
      }
    } catch (err) {
      console.error("[Navigate] Follow error:", err);
      setNavStatus("Failed to toggle follow mode. Check robot connection.");
    } finally {
      setIsLoading(false);
    }
  }, [followMode]);

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
        <div className="flex-1">
          <h1 className="text-3xl font-display font-bold text-text-primary dark:text-white">Navigation</h1>
          <p className="text-text-muted dark:text-gray-400">Choose a destination or follow me mode</p>
        </div>
        {/* Robot Connection Status */}
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "w-3 h-3 rounded-full",
              robotConnected === null
                ? "bg-gray-400 animate-pulse"
                : robotConnected
                  ? "bg-green-500"
                  : "bg-red-500"
            )}
          />
          <span className="text-sm text-text-muted dark:text-gray-400">
            {robotConnected === null ? "Checking..." : robotConnected ? "Robot Ready" : "Robot Offline"}
          </span>
        </div>
      </div>

      {/* Robot offline warning */}
      {robotConnected === false && (
        <div className="p-4 bg-orange-50 dark:bg-orange-900/30 border border-orange-200 dark:border-orange-800 rounded-2xl flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-orange-500 flex-shrink-0" />
          <span className="text-orange-700 dark:text-orange-300">
            Robot service is not connected. Navigation requires the robot to be online (port 9000).
          </span>
        </div>
      )}

      {/* Hardware Status (when connected) */}
      {robotHealth && robotConnected && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: "Motors", ok: robotHealth.hardware.motion, icon: "🚗" },
            { label: "Camera", ok: robotHealth.hardware.camera, icon: "📷" },
            { label: "LiDAR", ok: robotHealth.hardware.lidar, icon: "📡" },
            { label: "Moving", ok: robotHealth.hardware.moving, icon: "🔄" },
          ].map(({ label, ok, icon }) => (
            <div
              key={label}
              className={cn(
                "p-3 rounded-2xl text-center text-sm font-medium",
                ok
                  ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-400"
              )}
            >
              <span className="text-lg">{icon}</span>
              <p>{label}: {ok ? "✓" : "—"}</p>
            </div>
          ))}
        </div>
      )}

      {/* Follow Me Mode */}
      <button
        onClick={toggleFollowMode}
        disabled={!robotConnected || isLoading}
        className={cn(
          "w-full py-6 px-8 rounded-3xl flex items-center justify-center gap-4 transition-all",
          !robotConnected || isLoading
            ? "bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed"
            : followMode
              ? "bg-success text-white animate-pulse"
              : "bg-success-50 dark:bg-success-900/30 text-success border-2 border-success hover:bg-success-100 dark:hover:bg-success-900/50"
        )}
      >
        {isLoading ? (
          <Loader2 className="w-10 h-10 animate-spin" />
        ) : (
          <UserCheck className="w-10 h-10" />
        )}
        <div className="text-left">
          <h2 className="text-2xl font-bold">{followMode ? "Following You..." : "Follow Me Mode"}</h2>
          <p className={followMode ? "text-white/80" : "text-success"}>
            {followMode ? "Tap to stop following" : "I'll follow you wherever you go"}
          </p>
        </div>
      </button>

      {/* Navigation Status */}
      {navStatus && (
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-2xl text-blue-700 dark:text-blue-300 text-center font-medium">
          {navStatus}
        </div>
      )}

      {/* Active Navigation */}
      {isNavigating && selectedDest && (
        <div className="bg-primary-50 dark:bg-primary-900/30 border-2 border-primary rounded-3xl p-8 animate-scale-in">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className={cn("w-16 h-16 rounded-2xl flex items-center justify-center", selectedDest.color)}>
                <selectedDest.icon className="w-8 h-8 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-primary">Navigating to {selectedDest.name}</h2>
                <p className="text-text-secondary dark:text-gray-300 text-lg">
                  ETA: ~{Math.ceil(navETA)}s · {selectedDest.distance}
                </p>
              </div>
            </div>
            <button
              onClick={stopNavigation}
              disabled={isLoading}
              className="rover-btn w-16 h-16 bg-accent text-white rounded-2xl flex items-center justify-center hover:bg-accent-600 transition-colors"
            >
              {isLoading ? <Loader2 className="w-8 h-8 animate-spin" /> : <X className="w-8 h-8" />}
            </button>
          </div>

          {/* Navigation Progress */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-text-muted dark:text-gray-400">Progress</span>
              <span className="font-semibold text-primary">In Progress</span>
            </div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full w-[30%] transition-all duration-500 animate-pulse" />
            </div>
            <p className="text-center text-text-muted dark:text-gray-400 mt-4 text-lg">
              🤖 Navigating safely... Obstacle avoidance active
            </p>
          </div>
        </div>
      )}

      {/* Destination Selection */}
      {!isNavigating && !followMode && (
        <>
          <h2 className="text-xl font-semibold text-text-primary dark:text-white">Select Destination</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {destinations.map((dest) => (
              <button
                key={dest.id}
                onClick={() => setSelectedDest(dest)}
                disabled={!robotConnected}
                className={cn(
                  "rover-card p-6 rounded-3xl flex flex-col items-center gap-4 transition-all",
                  !robotConnected
                    ? "opacity-50 cursor-not-allowed"
                    : selectedDest?.id === dest.id
                      ? "ring-4 ring-primary bg-primary-50 dark:bg-primary-900/30"
                      : "bg-white dark:bg-gray-800 border-2 border-gray-100 dark:border-gray-700 hover:border-primary"
                )}
              >
                <div className={cn("w-16 h-16 rounded-2xl flex items-center justify-center", dest.color)}>
                  <dest.icon className="w-8 h-8 text-white" />
                </div>
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-text-primary dark:text-white">{dest.name}</h3>
                  <p className="text-text-muted dark:text-gray-400">{dest.distance}</p>
                </div>
              </button>
            ))}
          </div>

          {/* Custom Location */}
          <button className="w-full py-5 px-6 rounded-2xl border-2 border-dashed border-gray-300 dark:border-gray-600 text-text-secondary dark:text-gray-400 hover:border-primary hover:text-primary transition-colors flex items-center justify-center gap-3">
            <MapPin className="w-6 h-6" />
            <span className="text-lg font-medium">Enter Custom Location</span>
          </button>

          {/* Start Navigation Button */}
          {selectedDest && (
            <button
              onClick={startNavigation}
              disabled={!robotConnected || isLoading}
              className={cn(
                "w-full py-6 px-8 rounded-3xl flex items-center justify-center gap-4 text-2xl font-bold transition-colors",
                !robotConnected || isLoading
                  ? "bg-gray-300 dark:bg-gray-600 text-gray-500 cursor-not-allowed"
                  : "bg-primary text-white hover:bg-primary-600"
              )}
            >
              {isLoading ? (
                <Loader2 className="w-8 h-8 animate-spin" />
              ) : (
                <Play className="w-8 h-8" />
              )}
              Go to {selectedDest.name}
            </button>
          )}
        </>
      )}

      {/* Safety Information */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-2xl p-6 flex items-start gap-4">
        <div className="w-12 h-12 bg-primary-100 dark:bg-primary-900/30 rounded-xl flex items-center justify-center flex-shrink-0">
          <Navigation className="w-6 h-6 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-text-primary dark:text-white mb-1">Safety First</h3>
          <p className="text-text-muted dark:text-gray-400">
            The robot uses LiDAR obstacle avoidance to navigate safely. It will stop automatically
            if an obstacle is detected. You can stop navigation at any time.
          </p>
        </div>
      </div>
    </div>
  );
}
