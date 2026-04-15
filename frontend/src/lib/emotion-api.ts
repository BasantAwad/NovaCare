/**
 * NovaCare - Emotion Detection API Client
 * 
 * Communicates with the llm-backend server (usually on port 5000)
 * for facial emotion recognition.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_LLM_API_URL || "http://localhost:5000";

export interface EmotionResult {
  status: "success" | "error";
  emotion: string;
  confidence: number;
  face_detected: boolean;
  all_scores?: Record<string, number>;
  error?: string;
}

export interface EmotionHealth {
  status: "available" | "unavailable";
  message?: string;
  device?: string;
  labels?: string[];
}

/**
 * Detect emotion from a base64 encoded image
 */
export async function detectEmotion(imageBase64: string): Promise<EmotionResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/emotion/detect`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageBase64 }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return {
        status: "error",
        emotion: "unknown",
        confidence: 0,
        face_detected: false,
        error: errorData.error || `HTTP Error ${response.status}`,
      };
    }

    return await response.json();
  } catch (error) {
    console.error("Emotion detection failed:", error);
    return {
      status: "error",
      emotion: "unknown",
      confidence: 0,
      face_detected: false,
      error: error instanceof Error ? error.message : "Network error",
    };
  }
}

/**
 * Check if the emotion detection service is available
 */
export async function checkEmotionHealth(): Promise<EmotionHealth> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/emotion/health`);
    if (!response.ok) {
      return { status: "unavailable" };
    }
    return await response.json();
  } catch (error) {
    console.error("Emotion health check failed:", error);
    return { status: "unavailable", message: String(error) };
  }
}

/**
 * Helper to get an emoji for a specific emotion
 */
export function getEmotionEmoji(emotion: string): string {
  const normalized = emotion.toLowerCase().trim();
  switch (normalized) {
    case "happy":
    case "joy":
      return "😊";
    case "sad":
    case "sadness":
      return "😢";
    case "angry":
    case "anger":
      return "😠";
    case "surprise":
    case "surprised":
      return "😲";
    case "fear":
    case "fearful":
      return "😨";
    case "disgust":
    case "disgusted":
      return "🤢";
    case "neutral":
    default:
      return "😐";
  }
}

/**
 * Helper to get a color code for a specific emotion
 */
export function getEmotionColor(emotion: string): string {
  const normalized = emotion.toLowerCase().trim();
  switch (normalized) {
    case "happy":
    case "joy":
      return "#ef4444"; // red/pink tones for happiness in many UI libs, or use gold: #f59e0b
    case "sad":
    case "sadness":
      return "#3b82f6"; // blue
    case "angry":
    case "anger":
      return "#dc2626"; // deep red
    case "surprise":
    case "surprised":
      return "#8b5cf6"; // purple
    case "fear":
    case "fearful":
      return "#f97316"; // orange
    case "disgust":
    case "disgusted":
      return "#22c55e"; // green
    case "neutral":
    default:
      return "#6b7280"; // gray
  }
}
