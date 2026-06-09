import { useEffect } from 'react';
import { commandRegistry } from '../commands';

/**
 * Hook to register a custom function as a voice command.
 * 
 * @param commandId Unique ID for the command (e.g., 'SUBMIT_LOGIN')
 * @param action The function to execute when the command is triggered
 * @param description A description of what the command does
 */
export function useVoiceAction(commandId: string, action: () => void, description: string) {
  useEffect(() => {
    commandRegistry.register(commandId, action, description);

    return () => {
      commandRegistry.unregister(commandId);
    };
  }, [commandId, action, description]);
}
