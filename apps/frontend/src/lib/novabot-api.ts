/**
 * NovaCare — NovaBot LLM API Client
 */

import { getDynamicUrl } from "./utils";

const NOVABOT_API = process.env.NEXT_PUBLIC_NOVABOT_API_URL || "http://localhost:5000";

export interface ChatResponse {
  response: string;
  actions: Array<{ name: string; parameters?: any }>;
}

/**
 * Options for sending a message to the NovaBot API.
 */
export interface SendMessageOptions extends RequestInit {
  emotion?: string;
  emotionConfidence?: number;
}

/**
 * Send a message to the NovaBot LLM and get a response.
 */
export async function sendMessage(text: string, options?: SendMessageOptions): Promise<ChatResponse> {
  try {
    const payload: any = { message: text };
    if (options?.emotion) {
      payload.emotion = options.emotion;
      if (options.emotionConfidence !== undefined) {
        payload.emotion_confidence = options.emotionConfidence;
      }
    }

    // Extract standard fetch options
    const fetchOptions: RequestInit = { ...options };
    delete (fetchOptions as any).emotion;
    delete (fetchOptions as any).emotionConfidence;

    const res = await fetch(`${getDynamicUrl(NOVABOT_API)}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      ...fetchOptions,
    });

    if (!res.ok) {
      let errorText = "";
      try {
        errorText = await res.text();
      } catch (e) {
        errorText = "Unknown error text";
      }
      throw new Error(`Failed to communicate with NovaBot (Status ${res.status}): ${errorText}`);
    }

    const data = await res.json();
    return {
      response: data.response || "I'm sorry, I couldn't understand that.",
      actions: data.actions || [],
    };
  } catch (error) {
    console.error("[NovaBot API] Error:", error);
    throw error;
  }
}

/**
 * Check if the NovaBot server is healthy.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${getDynamicUrl(NOVABOT_API)}/health`, {
      method: "GET",
      // Increase timeout to 15s because local LLM inference can block the single-threaded Flask dev server for several seconds
      signal: AbortSignal.timeout(15000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Clear the chat history on the server.
 */
export async function clearHistory(): Promise<void> {
  try {
    await fetch(`${getDynamicUrl(NOVABOT_API)}/api/clear`, {
      method: "POST",
    });
  } catch (error) {
    console.error("[NovaBot API] Error clearing history:", error);
  }
}
