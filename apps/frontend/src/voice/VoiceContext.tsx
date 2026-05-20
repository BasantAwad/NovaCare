'use client';

/**
 * NovaCare Voice Assistant — VoiceContext
 *
 * Key improvements over previous version:
 * 1. Role-aware navigation — detects user role from current URL
 * 2. Route validation — NEVER navigates to an unregistered path
 * 3. Uses routeRegistry (single source of truth) for all navigation
 * 4. Emits debug events consumed by VoiceDebugPanel
 * 5. No hardcoded paths
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useRef,
  ReactNode,
  useCallback,
} from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { speechService } from './speechService';
import { parseIntent, generateResponse } from './intentParser';
import { commandRegistry } from './commands';
import { isValidRoute, UserRole } from './routeRegistry';
import { sendMessage } from '../lib/novabot-api';

// ─────────────────────────────────────────────────────────────────────────────
// Debug event (consumed by VoiceDebugPanel)
// ─────────────────────────────────────────────────────────────────────────────

export interface NavDebugEvent {
  timestamp: string;
  transcript: string;
  detectedIntent: string;
  resolvedRoute: string | null;
  routeValid: boolean;
  navigationResult: 'success' | 'blocked' | 'skipped' | 'browser-history';
  reason: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context shape
// ─────────────────────────────────────────────────────────────────────────────

interface VoiceContextType {
  isActive: boolean;
  isListening: boolean;
  isSpeaking: boolean;
  transcript: string;
  interimTranscript: string;
  response: string;
  currentRole: UserRole;
  debugLog: NavDebugEvent[];
  toggleAssistant: () => void;
  processTextCommand: (text: string) => void;
  clearDebugLog: () => void;
}

const VoiceContext = createContext<VoiceContextType | undefined>(undefined);

// ─────────────────────────────────────────────────────────────────────────────
// Role detection from URL pathname
// ─────────────────────────────────────────────────────────────────────────────

function detectRole(pathname: string): UserRole {
  if (pathname.startsWith('/admin')) return 'admin';
  if (pathname.startsWith('/medical')) return 'medical';
  if (pathname.startsWith('/guardian')) return 'guardian';
  if (pathname.startsWith('/rover')) return 'rover';
  return 'unknown';
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────────────────────

export function VoiceProvider({ children }: { children: ReactNode }) {
  const [isActive, setIsActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [debugLog, setDebugLog] = useState<NavDebugEvent[]>([]);

  const router = useRouter();
  const pathname = usePathname();

  // Derive role reactively from the current URL
  const currentRole: UserRole = detectRole(pathname ?? '');

  const continuousListeningRef = useRef(true);
  const isActiveRef = useRef(isActive);
  const isSpeakingRef = useRef(isSpeaking);
  const roleRef = useRef(currentRole);

  useEffect(() => { isActiveRef.current = isActive; }, [isActive]);
  useEffect(() => { isSpeakingRef.current = isSpeaking; }, [isSpeaking]);
  useEffect(() => { roleRef.current = currentRole; }, [currentRole]);

  // ── Add debug event helper ───────────────────────────────────────────────
  const addDebugEvent = useCallback((event: NavDebugEvent) => {
    setDebugLog((prev) => [event, ...prev].slice(0, 50)); // keep last 50
  }, []);

  // ── Core command processor ───────────────────────────────────────────────
  const processCommand = useCallback(async (text: string) => {
    speechService?.stopListening();
    setIsListening(false);
    continuousListeningRef.current = false;

    const role = roleRef.current;
    const intent = parseIntent(text, role);

    let aiResponse = generateResponse(intent, role);
    let debugEvent: NavDebugEvent | null = null;

    // ── Handle navigation ────────────────────────────────────────────────
    if (intent.type === 'NAVIGATE') {
      const { resolution } = intent;
      const alias = resolution.matchedAlias;

      // Browser history specials (back/forward) — always safe
      if (alias === '__back__') {
        router.back();
        debugEvent = {
          timestamp: new Date().toISOString(),
          transcript: text,
          detectedIntent: 'NAVIGATE',
          resolvedRoute: '__back__',
          routeValid: true,
          navigationResult: 'browser-history',
          reason: 'browser back triggered',
        };
      } else if (alias === '__forward__') {
        router.forward();
        debugEvent = {
          timestamp: new Date().toISOString(),
          transcript: text,
          detectedIntent: 'NAVIGATE',
          resolvedRoute: '__forward__',
          routeValid: true,
          navigationResult: 'browser-history',
          reason: 'browser forward triggered',
        };
      } else if (resolution.route) {
        const targetPath = resolution.route.path;

        // ── ROUTE VALIDATION before navigation ────────────────────────
        if (isValidRoute(targetPath)) {
          router.push(targetPath);
          debugEvent = {
            timestamp: new Date().toISOString(),
            transcript: text,
            detectedIntent: 'NAVIGATE',
            resolvedRoute: targetPath,
            routeValid: true,
            navigationResult: 'success',
            reason: resolution.reason,
          };
        } else {
          // Should never happen since resolveRoute only returns registered paths,
          // but guard defensively.
          aiResponse = `I found "${resolution.route.label}" but couldn't verify the route. Navigation blocked.`;
          debugEvent = {
            timestamp: new Date().toISOString(),
            transcript: text,
            detectedIntent: 'NAVIGATE',
            resolvedRoute: targetPath,
            routeValid: false,
            navigationResult: 'blocked',
            reason: 'route failed validation check',
          };
        }
      } else {
        // No route found — respond with suggestions, do NOT navigate
        debugEvent = {
          timestamp: new Date().toISOString(),
          transcript: text,
          detectedIntent: 'NAVIGATE',
          resolvedRoute: null,
          routeValid: false,
          navigationResult: 'blocked',
          reason: resolution.reason,
        };
      }

      if (debugEvent) addDebugEvent(debugEvent);

    } else if (intent.type === 'SCROLL') {
      commandRegistry.execute(`SCROLL_${intent.direction.toUpperCase()}`);

    } else if (intent.type === 'CLICK') {
      const el = document.querySelector(
        `[data-voice-action="${intent.target}"]`,
      ) as HTMLElement | null;
      if (el) el.click();

    } else if (intent.type === 'SEARCH') {
      const searchInput = document.querySelector(
        'input[type="search"], input[data-voice-search]',
      ) as HTMLInputElement | null;
      if (searchInput) {
        searchInput.value = intent.query;
        searchInput.dispatchEvent(new Event('input', { bubbles: true }));
      }

    } else if (intent.type === 'CONVERSATION' || intent.type === 'UNKNOWN') {
      // Try LLM for conversational responses
      setResponse('Nova is thinking...');
      try {
        const llmResponse = await sendMessage(text);
        aiResponse = llmResponse;
      } catch (error) {
        console.warn('[VoiceContext] LLM offline, using fallback:', error);
      }
    }

    setResponse(aiResponse);
    setIsSpeaking(true);
    setInterimTranscript('');

    speechService?.speak(aiResponse, () => {
      setIsSpeaking(false);
      continuousListeningRef.current = true;
      if (isActiveRef.current) {
        speechService?.startListening();
      }
    });
  }, [router, addDebugEvent]);

  // ── Speech service event wiring ──────────────────────────────────────────
  useEffect(() => {
    if (!speechService) return;

    // Register scroll commands (only these — no route commands)
    commandRegistry.register('SCROLL_DOWN', () => window.scrollBy({ top: 500, behavior: 'smooth' }), 'Scroll down');
    commandRegistry.register('SCROLL_UP', () => window.scrollBy({ top: -500, behavior: 'smooth' }), 'Scroll up');

    speechService.onStart(() => setIsListening(true));

    // Interim results — stream partial transcript in real time
    speechService.onInterim((partial: string) => {
      setInterimTranscript(partial);
    });

    // Safety: ensure isSpeaking resets if TTS stops for any reason
    speechService.onSpeakEnd(() => {
      setIsSpeaking(false);
    });

    speechService.onEnd(() => {
      setIsListening(false);
      setInterimTranscript('');
      if (isActiveRef.current && !isSpeakingRef.current && continuousListeningRef.current) {
        setTimeout(() => speechService?.startListening(), 100);
      }
    });

    speechService.onResult((text: string) => {
      const lower = text.toLowerCase();

      // Wake word detection
      if (!isActiveRef.current && (lower.includes('hey nova') || lower.includes('hi nova'))) {
        setIsActive(true);
        setTranscript('Hey Nova');
        setResponse("I'm listening...");
        setIsSpeaking(true);
        speechService?.speak('Hi, how can I help you?', () => {
          setIsSpeaking(false);
          speechService?.startListening();
        });

        const remainder = text
          .replace(/hey nova/i, '')
          .replace(/hi nova/i, '')
          .trim();
        if (!remainder) return;
        setTranscript(remainder);
        processCommand(remainder);
        return;
      }

      if (isActiveRef.current) {
        setTranscript(text);
        processCommand(text);
      }
    });

    return () => {
      speechService?.stopListening();
    };
    // processCommand is stable (useCallback with stable deps)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Activate / deactivate listening ──────────────────────────────────────
  useEffect(() => {
    if (isActive) {
      speechService?.startListening();
    } else {
      speechService?.stopListening();
      speechService?.stopSpeaking();
    }
  }, [isActive]);

  // ── Public API ────────────────────────────────────────────────────────────
  const processTextCommand = useCallback((text: string) => {
    setTranscript(text);
    processCommand(text);
  }, [processCommand]);

  const toggleAssistant = useCallback(() => {
    setIsActive((prev) => {
      if (!prev) {
        setResponse('How can I help you?');
        setTranscript('');
        speechService?.startListening();
      } else {
        continuousListeningRef.current = true;
      }
      return !prev;
    });
  }, []);

  const clearDebugLog = useCallback(() => setDebugLog([]), []);

  return (
    <VoiceContext.Provider
      value={{
        isActive,
        isListening,
        isSpeaking,
        transcript,
        interimTranscript,
        response,
        currentRole,
        debugLog,
        toggleAssistant,
        processTextCommand,
        clearDebugLog,
      }}
    >
      {children}
    </VoiceContext.Provider>
  );
}

export const useVoice = () => {
  const context = useContext(VoiceContext);
  if (context === undefined) {
    throw new Error('useVoice must be used within a VoiceProvider');
  }
  return context;
};
