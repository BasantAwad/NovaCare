/**
 * NovaCare — Emotion Detection API Client
 */

const EMOTION_API_URL = process.env.NEXT_PUBLIC_NOVABOT_API_URL || "http://localhost:5000";

export interface EmotionResult {
  status: "success" | "error";
  emotion: string;
  confidence: number;
  face_detected: boolean;
  all_scores?: Record<string, number>;
  error?: string;
}

/**
 * Detect emotion from a base64 encoded image frame.
 */
export async function detectEmotion(imageBase64: string): Promise<EmotionResult> {
  try {
    const res = await fetch(`${EMOTION_API_URL}/api/emotion/detect`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageBase64 }),
    });

    if (!res.ok) {
      throw new Error("Emotion detection failed");
    }

    return await res.json();
  } catch (error) {
    console.error("[Emotion API] Detect error:", error);
    return { status: "error", emotion: "neutral", confidence: 0, face_detected: false, error: "Detection failed" };
  }
}

/**
 * Check if the emotion detection service is healthy.
 */
export async function checkEmotionHealth(): Promise<{ status: "available" | "unavailable" }> {
  try {
    const res = await fetch(`${EMOTION_API_URL}/api/health`, {
      method: "GET",
      signal: AbortSignal.timeout(3000),
    });
    return { status: res.ok ? "available" : "unavailable" };
  } catch {
    return { status: "unavailable" };
  }
}

/**
 * Get the emoji representation for a given emotion.
 */
export function getEmotionEmoji(emotion: string): string {
  const emojis: Record<string, string> = {
    happy: "😊",
    sad: "😢",
    angry: "😠",
    surprised: "😮",
    neutral: "😐",
    fear: "😨",
    disgust: "🤢",
  };
  return emojis[emotion.toLowerCase()] || "😐";
}

/**
 * Get the color representation for a given emotion.
 */
export function getEmotionColor(emotion: string): string {
  const colors: Record<string, string> = {
    happy: "#10B981", // Success Green
    sad: "#3B82F6",    // Info Blue
    angry: "#EF4444",   // Danger Red
    surprised: "#F59E0B", // Warning Orange
    neutral: "#6B7280",   // Muted Gray
    fear: "#8B5CF6",    // Purple
    disgust: "#059669",  // Emerald
  };
  return colors[emotion.toLowerCase()] || "#6B7280";
}
