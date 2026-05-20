import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Utility function to merge tailwind classes safely.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Dynamic URL helper to adjust backend endpoints to the host's actual IP
 * when accessed on external devices (like mobile phones or tablets).
 */
export function getDynamicUrl(defaultUrl: string): string {
  if (typeof window === "undefined") return defaultUrl;
  try {
    const hostname = window.location.hostname;
    if (hostname && hostname !== "localhost" && hostname !== "127.0.0.1") {
      const url = new URL(defaultUrl);
      url.hostname = hostname;
      return url.toString().replace(/\/$/, "");
    }
  } catch (e) {
    console.error("Error generating dynamic URL:", e);
  }
  return defaultUrl;
}
