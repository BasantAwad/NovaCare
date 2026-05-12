/**
 * NovaCare — ASL (American Sign Language) API Client
 */

const ASL_API_URL = process.env.NEXT_PUBLIC_ASL_API_URL || "http://localhost:8001";

export class ASLAPIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ASLAPIError";
  }
}

export const aslAPI = {
  /**
   * Check if the ASL API server is healthy.
   */
  async checkHealth(): Promise<boolean> {
    try {
      const res = await fetch(`${ASL_API_URL}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(3000),
      });
      return res.ok;
    } catch {
      throw new ASLAPIError("ASL API is not reachable");
    }
  },

  /**
   * Predict a sign from a base64 encoded image frame.
   */
  async predict(frameBase64: string): Promise<any> {
    try {
      const res = await fetch(`${ASL_API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: frameBase64 }),
      });

      if (!res.ok) {
        throw new Error("Prediction failed");
      }

      return await res.json();
    } catch (error) {
      console.error("[ASL API] Predict error:", error);
      throw error;
    }
  },

  /**
   * Add a letter to the server-side text accumulator.
   */
  async addLetter(letter: string): Promise<any> {
    try {
      const res = await fetch(`${ASL_API_URL}/accumulator/add`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ letter }),
      });

      if (!res.ok) {
        throw new Error("Failed to add letter");
      }

      return await res.json();
    } catch (error) {
      console.error("[ASL API] Add letter error:", error);
      throw error;
    }
  },

  /**
   * Clear the server-side text accumulator.
   */
  async clearAccumulator(): Promise<void> {
    try {
      const res = await fetch(`${ASL_API_URL}/accumulator/clear`, {
        method: "POST",
      });

      if (!res.ok) {
        throw new Error("Failed to clear accumulator");
      }
    } catch (error) {
      console.error("[ASL API] Clear accumulator error:", error);
      throw error;
    }
  },
};
