/**
 * NovaCare — NovaBot LLM API Client
 */

const NOVABOT_API = process.env.NEXT_PUBLIC_NOVABOT_API_URL || "http://localhost:5000";

/**
 * Send a message to the NovaBot LLM and get a response.
 */
export async function sendMessage(text: string): Promise<string> {
  try {
    const res = await fetch(`${NOVABOT_API}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: text }),
    });

    if (!res.ok) {
      throw new Error("Failed to communicate with NovaBot");
    }

    const data = await res.json();
    return data.response || "I'm sorry, I couldn't understand that.";
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
    const res = await fetch(`${NOVABOT_API}/api/health`, {
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
    await fetch(`${NOVABOT_API}/api/chat/clear`, {
      method: "POST",
    });
  } catch (error) {
    console.error("[NovaBot API] Error clearing history:", error);
  }
}
