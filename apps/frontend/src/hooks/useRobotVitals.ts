/**
 * useRobotVitals — Custom hook for fetching and polling real-time vitals
 * 
 * Fetches heart rate and other metrics from the rover's smart watch
 * integration, with automatic polling and fallback to dashboard API.
 */

"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { getHeartRate, getRobotVitals, type RobotVitals } from "@/lib/robot-vitals-api";
import { getVitals, type VitalSign } from "@/lib/dashboard-api";

export interface UseRobotVitalsOptions {
  pollInterval?: number; // milliseconds (default: 2000)
  retryCount?: number;   // times to retry on error (default: 3)
  fallbackToDashboard?: boolean; // use dashboard API as fallback (default: true)
}

export interface UseRobotVitalsReturn {
  vitals: RobotVitals | null;
  isLoading: boolean;
  isError: boolean;
  error: string | null;
  source: "watch" | "dashboard" | "none";
  refetch: () => Promise<void>;
}

export function useRobotVitals(
  options: UseRobotVitalsOptions = {}
): UseRobotVitalsReturn {
  const {
    pollInterval = 2000,
    retryCount = 3,
    fallbackToDashboard = true,
  } = options;

  const [vitals, setVitals] = useState<RobotVitals | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<"watch" | "dashboard" | "none">("none");

  const abortControllerRef = useRef<AbortController | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timer | null>(null);

  const fetchVitals = useCallback(async () => {
    setIsLoading(true);
    setIsError(false);
    setError(null);

    try {
      // Try robot service first
      const robotVitals = await getRobotVitals();

      if (
        robotVitals.status === "success" &&
        robotVitals.heart_rate !== undefined
      ) {
        setVitals(robotVitals);
        setSource("watch");
        return;
      }

      // Fallback to dashboard API if enabled
      if (fallbackToDashboard) {
        try {
          const dashboardData = await getVitals();

          if (
            dashboardData.status === "success" &&
            dashboardData.data &&
            dashboardData.data.length > 0
          ) {
            const latest = dashboardData.data[0];
            setVitals({
              status: "success",
              heart_rate: latest.heart_rate,
              timestamp: latest.measured_at,
            });
            setSource("dashboard");
            return;
          }
        } catch (dashboardErr) {
          console.warn("Dashboard fallback failed:", dashboardErr);
        }
      }

      // No data available
      setSource("none");
      setError("No vital data available");
    } catch (err) {
      setIsError(true);
      setError(
        err instanceof Error ? err.message : "Failed to fetch vitals"
      );
      console.error("Vitals fetch error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [fallbackToDashboard]);

  // Initial fetch
  useEffect(() => {
    fetchVitals();
  }, [fetchVitals]);

  // Polling
  useEffect(() => {
    pollIntervalRef.current = setInterval(fetchVitals, pollInterval);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [fetchVitals, pollInterval]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  return {
    vitals,
    isLoading,
    isError,
    error,
    source,
    refetch: fetchVitals,
  };
}
