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
 * Send a message to the NovaBot LLM and get a response.
 */
export async function sendMessage(text: string, options?: RequestInit): Promise<ChatResponse> {
  try {
    const res = await fetch(`${getDynamicUrl(NOVABOT_API)}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: text }),
      ...options,
    });

    if (!res.ok) {
      throw new Error("Failed to communicate with NovaBot");
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
      // Set a short timeout for health check
      signal: AbortSignal.timeout(3000),
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
