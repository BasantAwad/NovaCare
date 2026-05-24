/**
 * NovaCare — Robot Vitals API Client
 * 
 * Communicates with the robot service at NEXT_PUBLIC_ROBOT_API_URL
 * to fetch real-time vital signs from the smart watch.
 */

const ROBOT_API = process.env.NEXT_PUBLIC_ROBOT_API_URL || "http://localhost:9000";

export interface RobotVitals {
  heart_rate?: number;
  steps?: number;
  battery?: number;
  timestamp?: string;
  status: "success" | "unavailable" | "error";
  message?: string;
}

interface ApiResponse<T = unknown> {
  status: "success" | "unavailable" | "error";
  data?: T;
  message?: string;
  error?: string;
}

async function robotFetch<T = unknown>(
  path: string,
  options: RequestInit = {},
): Promise<RobotVitals | ApiResponse<T>> {
  try {
    const url = `${ROBOT_API}${path}`;
    
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": process.env.NEXT_PUBLIC_ROBOT_API_KEY || "novacare-secure-key-2026",
        ...(options.headers as Record<string, string>),
      },
      ...options,
    });

    if (!response.ok) {
      return {
        status: response.status === 503 ? "unavailable" : "error",
        message: `HTTP ${response.status}`,
      } as RobotVitals;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Robot API error:", error);
    return {
      status: "error",
      message: error instanceof Error ? error.message : "Unknown error",
    } as RobotVitals;
  }
}

/**
 * Get latest heart rate from the robot's smart watch
 */
export async function getHeartRate(): Promise<RobotVitals> {
  const result = await robotFetch("/api/vitals/heart-rate");
  return result as RobotVitals;
}

/**
 * Get all current vitals (heart rate, steps, battery) from the robot's smart watch
 */
export async function getRobotVitals(): Promise<RobotVitals> {
  const result = await robotFetch("/api/vitals/current");
  return result as RobotVitals;
}

/**
 * Get robot service health status including vitals
 */
export async function getRobotHealth(): Promise<{
  status: string;
  service: string;
  hardware: Record<string, boolean>;
  vitals?: RobotVitals;
}> {
  const result = await robotFetch("/health");
  return result as any;
}
