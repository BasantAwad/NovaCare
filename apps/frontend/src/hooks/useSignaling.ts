import { useState, useCallback } from "react";

export function useSignaling(id: string, role: string) {
  // Mock signaling hook for frontend build
  const [isConnected, setIsConnected] = useState(true);
  const [lastEvent, setLastEvent] = useState<{ type: string; payload?: any } | null>(null);

  const navigateToPatient = useCallback(() => {
    console.log(`[Signaling ${role}] Navigate to patient triggered`);
  }, [role]);

  return { navigateToPatient, lastEvent, isConnected };
}
