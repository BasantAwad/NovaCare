/**
 * NovaCare — Robot API Client
 *
 * TypeScript client for the Robot REST Service (port 9000).
 * Provides methods for camera, movement, audio, and LiDAR control.
 */

import { getDynamicUrl } from "./utils";

const ROBOT_API = process.env.NEXT_PUBLIC_ROBOT_API_URL || "http://localhost:9000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RobotHealth {
  status: string;
  service: string;
  hardware: {
    camera: boolean;
    motion: boolean;
    tts: boolean;
    stt: boolean;
    lidar: boolean;
    moving: boolean;
  };
}

export interface MoveRequest {
  direction: "forward" | "backward" | "left" | "right" | "turn_left" | "turn_right";
  speed?: number;
  duration?: number;
}

export interface MoveResponse {
  status: string;
  direction?: string;
  speed?: number;
  duration?: number;
  message?: string;
}

export interface NavigateRequest {
  destination: string;
}

export interface NavigateResponse {
  status: string;
  destination: string;
  estimated_duration_s: number;
}

export interface TTSRequest {
  text: string;
  lang?: string;
}

export interface STTResponse {
  status: string;
  text: string | null;
}

export interface CameraFrameResponse {
  image: string;  // base64 JPEG
  status: string;
}

export interface LidarScanResponse {
  points: Array<{ angle: number; distance_mm: number }>;
  count: number;
}

export interface ObstacleResponse {
  obstacle_ahead: boolean;
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function robotFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${getDynamicUrl(ROBOT_API)}${path}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  const res = await fetch(url, { ...options, headers });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Robot API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Robot API Functions
// ---------------------------------------------------------------------------

/** Check robot service health and hardware status */
export async function checkRobotHealth(): Promise<RobotHealth> {
  return robotFetch<RobotHealth>("/health");
}

// ---- Camera ----

/** Get a single camera frame as base64 JPEG */
export async function getCameraFrame(): Promise<CameraFrameResponse> {
  return robotFetch<CameraFrameResponse>("/api/camera/frame");
}

/** Get the MJPEG stream URL (use directly in <img> src) */
export function getCameraStreamUrl(): string {
  return `${ROBOT_API}/api/camera/stream`;
}

// ---- Movement ----

/** Move the robot in a direction */
export async function moveRobot(req: MoveRequest): Promise<MoveResponse> {
  return robotFetch<MoveResponse>("/api/move", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/** Stop all robot movement */
export async function stopRobot(): Promise<MoveResponse> {
  return robotFetch<MoveResponse>("/api/move/stop", { method: "POST" });
}

/** Navigate to a predefined destination */
export async function navigateRobot(req: NavigateRequest): Promise<NavigateResponse> {
  return robotFetch<NavigateResponse>("/api/navigate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/** Start follow-user mode */
export async function startFollow(): Promise<{ status: string }> {
  return robotFetch<{ status: string }>("/api/follow/start", { method: "POST" });
}

/** Stop follow-user mode */
export async function stopFollow(): Promise<{ status: string }> {
  return robotFetch<{ status: string }>("/api/follow/stop", { method: "POST" });
}

// ---- Audio ----

/** Speak text through the robot's speaker */
export async function robotSpeak(req: TTSRequest): Promise<{ status: string; text: string }> {
  return robotFetch<{ status: string; text: string }>("/api/tts/speak", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/** Listen for speech via the robot's microphone */
export async function robotListen(
  timeout?: number,
  phraseTimeout?: number,
): Promise<STTResponse> {
  return robotFetch<STTResponse>("/api/stt/listen", {
    method: "POST",
    body: JSON.stringify({ timeout, phrase_timeout: phraseTimeout }),
  });
}

/** Check STT/TTS availability */
export async function checkAudioStatus(): Promise<{
  stt_available: boolean;
  tts_available: boolean;
}> {
  return robotFetch<{ stt_available: boolean; tts_available: boolean }>("/api/stt/status");
}

// ---- LiDAR ----

/** Get full LiDAR scan data */
export async function getLidarScan(): Promise<LidarScanResponse> {
  return robotFetch<LidarScanResponse>("/api/lidar/scan");
}

/** Check if there's an obstacle ahead */
export async function checkObstacle(): Promise<ObstacleResponse> {
  return robotFetch<ObstacleResponse>("/api/lidar/obstacle");
}
