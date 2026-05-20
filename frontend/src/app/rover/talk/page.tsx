"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Mic, MicOff, Send, Volume2, VolumeX, Hand, ArrowLeft, Loader2, WifiOff, RefreshCw,
  Play, Pause, Compass, CheckCircle2, Phone, PhoneOff, Calendar, AlertOctagon, Music, X, Trash2
} from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { sendMessage as sendToNovaBot, checkHealth, clearHistory } from "@/lib/novabot-api";
import { STTService, TTSService } from "@/lib/speech";
import { robotSpeak, robotListen, checkRobotHealth } from "@/lib/robot-api";
import ASLRecognitionModal from "@/components/ASLRecognitionModal";
import { getMedications, markMedicationTaken, getNavigationStatus, updateNavigation } from "@/lib/dashboard-api";

interface Message {
  id: number;
  type: "user" | "nova";
  content: string;
  timestamp: Date;
  action?: { name: string; parameters?: any };
}

// ==========================================
// 1. Music Player Tool Widget
// ==========================================
// Shared global audio player to prevent duplicate playback and allow global control
let globalAudio: HTMLAudioElement | null = null;
let globalAudioListeners: Set<(isPlaying: boolean, progress: number) => void> = new Set();
let globalAudioProgressInterval: any = null;

const playGlobalAudio = (url: string) => {
  if (typeof window === "undefined") return;
  if (globalAudio) {
    globalAudio.pause();
    clearInterval(globalAudioProgressInterval);
  }
  globalAudio = new Audio(url);
  globalAudio.loop = true;
  globalAudio.play().then(() => {
    notifyListeners(true, 0);
  }).catch(err => {
    console.log("Global audio playback failed:", err);
  });

  // Start progress interval
  globalAudioProgressInterval = setInterval(() => {
    if (globalAudio) {
      const prog = (globalAudio.currentTime / (globalAudio.duration || 300)) * 100;
      notifyListeners(!globalAudio.paused, prog);
    }
  }, 500);
};

const pauseGlobalAudio = () => {
  if (globalAudio) {
    globalAudio.pause();
    notifyListeners(false, (globalAudio.currentTime / (globalAudio.duration || 300)) * 100);
  }
};

const notifyListeners = (isPlaying: boolean, progress: number) => {
  globalAudioListeners.forEach(listener => listener(isPlaying, progress));
};

// ==========================================
// 1. Music Player Tool Widget
// ==========================================
function MusicWidget({ mood, track }: { mood?: string; track?: string }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  const trackTitle = track || (mood === 'upbeat' ? "Morning Jazz" : mood === 'focus' ? "Classical Favorites" : "Relaxing Piano");
  const trackArtist = mood === 'upbeat' ? "Smooth Jazz" : mood === 'focus' ? "Orchestra" : "Ambient Sounds";
  const audioUrl = mood === 'upbeat'
    ? "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3"
    : mood === 'focus'
      ? "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3"
      : "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3";

  useEffect(() => {
    const listener = (playing: boolean, prog: number) => {
      setIsPlaying(playing);
      setProgress(prog);
    };
    globalAudioListeners.add(listener);

    // Auto-play immediately if not already playing this song
    if (!globalAudio || globalAudio.src !== audioUrl) {
      playGlobalAudio(audioUrl);
    } else {
      setIsPlaying(!globalAudio.paused);
      setProgress((globalAudio.currentTime / (globalAudio.duration || 300)) * 100);
    }

    return () => {
      globalAudioListeners.delete(listener);
    };
  }, [audioUrl]);

  const togglePlay = () => {
    if (!globalAudio) return;
    if (isPlaying) {
      pauseGlobalAudio();
    } else {
      if (globalAudio.src !== audioUrl) {
        playGlobalAudio(audioUrl);
      } else {
        globalAudio.play().then(() => {
          setIsPlaying(true);
        }).catch(err => console.log("Play failed:", err));
      }
    }
  };

  return (
    <div className="bg-gradient-to-r from-purple-500/10 to-indigo-500/10 border border-purple-200 dark:border-purple-800 rounded-3xl p-5 mt-2 flex flex-col gap-4 shadow-soft">
      <div className="flex items-center gap-4">
        <div className={cn(
          "w-16 h-16 rounded-2xl flex items-center justify-center bg-purple-500 shadow-md transition-all duration-700",
          isPlaying && "animate-spin [animation-duration:10s]"
        )}>
          <Music className="w-8 h-8 text-white" />
        </div>
        <div className="flex-1">
          <h4 className="text-lg font-bold text-text-primary dark:text-white">{trackTitle}</h4>
          <p className="text-sm text-text-muted dark:text-gray-400">{trackArtist}</p>
        </div>
        <button
          onClick={togglePlay}
          className="w-12 h-12 rounded-2xl bg-purple-500 hover:bg-purple-600 text-white flex items-center justify-center transition-colors shadow-sm"
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
        </button>
      </div>

      <div className="space-y-1">
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div className="h-full bg-purple-500 transition-all duration-300" style={{ width: `${progress}%` }} />
        </div>
        <div className="flex justify-between text-xs text-text-muted dark:text-gray-400">
          <span>{Math.floor(progress * 0.1)}s</span>
          <span>Playing</span>
        </div>
      </div>

      {isPlaying && (
        <div className="flex justify-center gap-1 items-end h-4 py-1">
          {[1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1].map((h, i) => (
            <div
              key={i}
              className="w-1 bg-purple-500 rounded-full animate-bounce"
              style={{
                height: `${h * 20}%`,
                animationDelay: `${i * 0.1}s`,
                animationDuration: '0.8s'
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ==========================================
// 2. Navigation Simulation Tool Widget
// ==========================================
function NavigationWidget({ destination }: { destination?: string }) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'navigating' | 'completed' | 'cancelled'>('navigating');
  const [activeDest, setActiveDest] = useState(destination || "Kitchen");

  const destName = activeDest.charAt(0).toUpperCase() + activeDest.slice(1);

  // Sync state on mount and update database
  useEffect(() => {
    const initNav = async () => {
      if (destination) {
        await updateNavigation(destination, 'navigating', false);
      }
      pollStatus();
    };
    initNav();
  }, [destination]);

  const pollStatus = async () => {
    try {
      const response = await getNavigationStatus();
      if (response.status === "success" && response.data) {
        const { destination: curDest, status: curStatus, progress: curProgress } = response.data;
        if (curDest) setActiveDest(curDest);

        if (curStatus === 'navigating') {
          setStatus('navigating');
          setProgress(curProgress);
        } else if (curProgress === 100) {
          setStatus('completed');
          setProgress(100);
        } else if (curStatus === 'idle') {
          setStatus('cancelled');
          setProgress(0);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (status !== 'navigating') return;

    const interval = setInterval(() => {
      pollStatus();
    }, 1500);

    return () => clearInterval(interval);
  }, [status]);

  const handleCancel = async () => {
    setStatus('cancelled');
    setProgress(0);
    try {
      await updateNavigation(null, 'idle', false);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className={cn(
      "border rounded-3xl p-5 mt-2 flex flex-col gap-4 shadow-soft transition-all duration-500",
      status === 'completed'
        ? "bg-green-500/10 border-green-200 dark:border-green-800"
        : status === 'cancelled'
          ? "bg-red-500/10 border-red-200 dark:border-red-800"
          : "bg-primary-50/50 dark:bg-primary-900/10 border-primary-200 dark:border-primary-800"
    )}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={cn(
            "w-14 h-14 rounded-2xl flex items-center justify-center shadow-sm text-white",
            status === 'completed' ? "bg-green-500" : status === 'cancelled' ? "bg-red-500" : "bg-primary animate-pulse"
          )}>
            {status === 'completed' ? <CheckCircle2 className="w-7 h-7" /> : status === 'cancelled' ? <X className="w-7 h-7" /> : <Compass className="w-7 h-7" />}
          </div>
          <div>
            <h4 className="text-lg font-bold text-text-primary dark:text-white">
              {status === 'completed' ? `Arrived at ${destName}` : status === 'cancelled' ? "Navigation Stopped" : `Going to ${destName}`}
            </h4>
            <p className="text-sm text-text-muted dark:text-gray-400">
              {status === 'completed' ? "Successfully completed" : status === 'cancelled' ? "Operation aborted" : "Rover in motion..."}
            </p>
          </div>
        </div>
        {status === 'navigating' && (
          <button
            onClick={handleCancel}
            className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-xl text-sm font-semibold transition-colors"
          >
            Cancel
          </button>
        )}
      </div>

      {status === 'navigating' && (
        <div className="space-y-1">
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div className="h-full bg-primary transition-all duration-300" style={{ width: `${progress}%` }} />
          </div>
          <div className="flex justify-between text-xs text-text-muted dark:text-gray-400">
            <span>Estimated distance: {Math.max(1, Math.round((100 - progress) / 10))}m</span>
            <span>{progress}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ==========================================
// 3. Guardian Calling Tool Widget
// ==========================================
function CallWidget() {
  const [callState, setCallState] = useState<'dialing' | 'connected' | 'ended'>('dialing');
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (callState === 'dialing') {
      const timer = setTimeout(() => {
        setCallState('connected');
      }, 3000);
      return () => clearTimeout(timer);
    } else if (callState === 'connected') {
      const interval = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [callState]);

  const formatTime = (totalSec: number) => {
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className={cn(
      "border rounded-3xl p-5 mt-2 flex flex-col items-center gap-4 shadow-soft transition-all duration-500",
      callState === 'ended'
        ? "bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700"
        : "bg-blue-500/10 border-blue-200 dark:border-blue-800"
    )}>
      {callState !== 'ended' ? (
        <>
          <div className="relative">
            <div className="w-20 h-20 bg-blue-500 rounded-full flex items-center justify-center text-white shadow-lg z-10 relative">
              <Phone className="w-10 h-10 animate-bounce" />
            </div>
            {callState === 'dialing' && (
              <div className="absolute inset-0 bg-blue-400 rounded-full animate-ping opacity-75 -z-0" />
            )}
          </div>

          <div className="text-center">
            <h4 className="text-xl font-bold text-text-primary dark:text-white">Guardian</h4>
            <p className="text-sm text-text-muted dark:text-gray-400 mt-1">
              {callState === 'dialing' ? "Calling..." : `Connected • ${formatTime(seconds)}`}
            </p>
          </div>

          <button
            onClick={() => setCallState('ended')}
            className="w-14 h-14 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center shadow-lg transition-colors mt-2"
          >
            <PhoneOff className="w-6 h-6" />
          </button>
        </>
      ) : (
        <div className="flex items-center gap-3 py-2 w-full justify-center">
          <PhoneOff className="w-6 h-6 text-text-muted" />
          <span className="text-lg font-semibold text-text-muted dark:text-gray-400">Call Ended</span>
        </div>
      )}
    </div>
  );
}

// ==========================================
// 4. Medications Checklist Tool Widget
// ==========================================
function MedicationsWidget() {
  const [meds, setMeds] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchMeds = async () => {
    try {
      const response = await getMedications();
      if (response.status === "success") {
        setMeds(response.data.map((m: any) => ({
          id: m.id,
          name: m.medication_name || m.name,
          time: m.scheduled_time,
          dosage: m.dosage,
          checked: m.status === 'taken'
        })));
      }
    } catch (e) {
      console.error("Error fetching medications for widget:", e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMeds();
  }, []);

  const toggleMed = async (id: string | number) => {
    // Optimistic toggle
    setMeds(prev => prev.map(m => m.id === id ? { ...m, checked: !m.checked } : m));
    try {
      await markMedicationTaken(String(id));
      fetchMeds();
    } catch (e) {
      console.error("Failed to mark taken:", e);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-gradient-to-r from-teal-500/10 to-emerald-500/10 border border-teal-200 dark:border-teal-800 rounded-3xl p-5 mt-2 flex items-center justify-center min-h-[120px]">
        <Loader2 className="w-6 h-6 text-teal-600 animate-spin" />
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-r from-teal-500/10 to-emerald-500/10 border border-teal-200 dark:border-teal-800 rounded-3xl p-5 mt-2 flex flex-col gap-4 shadow-soft">
      <div className="flex items-center gap-3">
        <Calendar className="w-6 h-6 text-teal-600 dark:text-teal-400" />
        <h4 className="text-lg font-bold text-text-primary dark:text-white">Daily Medication Checklist</h4>
      </div>

      <div className="flex flex-col gap-3">
        {meds.map((med) => (
          <button
            key={med.id}
            onClick={() => toggleMed(med.id)}
            className={cn(
              "w-full p-4 rounded-2xl flex items-center gap-4 text-left transition-all border",
              med.checked
                ? "bg-emerald-500/20 border-emerald-300 dark:border-emerald-800"
                : "bg-white dark:bg-gray-800 border-gray-100 dark:border-gray-700 hover:border-teal-400"
            )}
          >
            <div className={cn(
              "w-8 h-8 rounded-xl flex items-center justify-center border-2 transition-all",
              med.checked
                ? "bg-emerald-500 border-emerald-500 text-white"
                : "border-gray-300 dark:border-gray-600 text-transparent"
            )}>
              {med.checked && <CheckCircle2 className="w-5 h-5" />}
            </div>
            <div className="flex-1">
              <span className={cn("text-lg font-semibold block text-text-primary dark:text-white", med.checked && "line-through opacity-60")}>
                {med.name}
              </span>
              <span className="text-sm text-text-muted dark:text-gray-400 block">{med.dosage}</span>
            </div>
            <span className="text-lg font-bold text-teal-600 dark:text-teal-400">{med.time}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ==========================================
// 5. Emergency Warning Alarm Tool Widget
// ==========================================
function EmergencyWidget() {
  const [seconds, setSeconds] = useState(10);
  const [status, setStatus] = useState<'active' | 'cancelled' | 'triggered'>('active');

  useEffect(() => {
    if (status !== 'active') return;

    if (seconds <= 0) {
      setStatus('triggered');
      return;
    }

    const timer = setTimeout(() => {
      setSeconds(seconds - 1);
    }, 1000);

    return () => clearTimeout(timer);
  }, [seconds, status]);

  return (
    <div className={cn(
      "border-2 rounded-3xl p-6 mt-2 flex flex-col items-center gap-4 text-center transition-all duration-500",
      status === 'active'
        ? "bg-red-500/20 border-red-500 animate-pulse shadow-[0_0_30px_rgba(239,68,68,0.4)]"
        : status === 'triggered'
          ? "bg-red-600 border-red-700 text-white shadow-[0_0_40px_rgba(220,38,38,0.6)]"
          : "bg-green-500/10 border-green-500 text-green-700 dark:text-green-300"
    )}>
      {status === 'active' && (
        <>
          <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center text-white animate-bounce shadow-md">
            <AlertOctagon className="w-10 h-10" />
          </div>
          <div>
            <h4 className="text-xl font-bold text-red-600 dark:text-red-400">EMERGENCY ALERT TRIGGERED</h4>
            <p className="text-base text-text-secondary dark:text-gray-300 mt-2">
              An emergency state has been identified. Notifying emergency services and your guardian in:
            </p>
            <div className="text-5xl font-extrabold text-red-600 dark:text-red-400 mt-3 font-mono">
              {seconds}s
            </div>
          </div>
          <button
            onClick={() => setStatus('cancelled')}
            className="w-full py-4 bg-white dark:bg-gray-800 text-red-600 border-2 border-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-2xl text-lg font-bold transition-all mt-2"
          >
            Cancel Alert
          </button>
        </>
      )}

      {status === 'triggered' && (
        <>
          <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center text-red-600 animate-ping">
            <AlertOctagon className="w-12 h-12" />
          </div>
          <div>
            <h4 className="text-2xl font-black tracking-wide text-white">DISPATCHING HELP</h4>
            <p className="text-white/80 mt-2 text-lg">
              Emergency contacts and local services have been dispatched. Sirens are activated.
              Please stay calm and follow instructions.
            </p>
          </div>
          <button
            onClick={() => setStatus('cancelled')}
            className="w-full py-4 bg-white text-red-600 rounded-2xl text-lg font-bold hover:bg-gray-100 transition-colors shadow-lg"
          >
            False Alarm? Cancel Now
          </button>
        </>
      )}

      {status === 'cancelled' && (
        <div className="flex flex-col items-center gap-2 py-4">
          <CheckCircle2 className="w-14 h-14 text-green-500" />
          <h4 className="text-xl font-bold">Emergency Cancelled</h4>
          <p className="text-sm text-text-muted dark:text-gray-400 text-center max-w-sm mt-1">
            System has returned to normal monitoring state. Notification was canceled.
          </p>
        </div>
      )}
    </div>
  );
}

const suggestedQuestions = [
  "What time is my next medication?",
  "How am I feeling today?",
  "Call my guardian",
  "Play some music",
  "Take me to the kitchen",
];

export default function TalkPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      type: "nova",
      content: "Hi! I'm Nova, your AI assistant. How can I help you today? You can type, speak, or use sign language.",
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isTTSEnabled, setIsTTSEnabled] = useState(true);
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [isCheckingConnection, setIsCheckingConnection] = useState(false);
  const [isASLModalOpen, setIsASLModalOpen] = useState(false);
  const [robotAvailable, setRobotAvailable] = useState(false);
  const [useRobotAudio, setUseRobotAudio] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const sttRef = useRef<STTService | null>(null);
  const ttsRef = useRef<TTSService | null>(null);
  const messageIdRef = useRef(1);
  const abortControllerRef = useRef<AbortController | null>(null);

  // 1. Load persistent chat history from localStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("novabot_chat_messages");
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const restored = parsed.map((m: any) => ({
            ...m,
            timestamp: new Date(m.timestamp)
          }));
          setMessages(restored);

          const maxId = restored.reduce((max: number, m: any) => Math.max(max, m.id), 1);
          messageIdRef.current = maxId;
        } catch (e) {
          console.error("Failed to parse saved chat messages", e);
        }
      }
    }
  }, []);

  // 2. Save chat history to localStorage on updates
  useEffect(() => {
    if (typeof window !== "undefined" && messages.length > 0) {
      if (messages.length > 1 || messages[0].content !== "Hi! I'm Nova, your AI assistant. How can I help you today? You can type, speak, or use sign language.") {
        localStorage.setItem("novabot_chat_messages", JSON.stringify(messages));
      }
    }
  }, [messages]);

  // Initialize STT and TTS
  useEffect(() => {
    if (typeof window !== 'undefined') {
      sttRef.current = new STTService({ lang: 'en-US', continuous: false });
      ttsRef.current = new TTSService({ rate: 0.9, pitch: 1.0, volume: 1.0 });

      // Set up STT callbacks
      sttRef.current.onTranscript((text) => {
        setInputText(text);
        setIsListening(false);
      });

      sttRef.current.onEnd(() => {
        setIsListening(false);
      });

      sttRef.current.onError((error, message) => {
        console.error('[STT Error]', error, message);
        setIsListening(false);
      });
    }

    return () => {
      sttRef.current?.stop();
      ttsRef.current?.stop();
    };
  }, []);

  // Check API health on mount (NovaBot + Robot)
  useEffect(() => {
    const checkConnection = async () => {
      setIsCheckingConnection(true);
      const healthy = await checkHealth();
      setIsConnected(healthy);
      setIsCheckingConnection(false);

      // Also check robot service
      try {
        const robotHealth = await checkRobotHealth();
        const robotOk = robotHealth.status === "healthy";
        setRobotAvailable(robotOk);
        // Auto-enable robot audio if robot has TTS+STT
        if (robotOk && robotHealth.hardware.tts) {
          setUseRobotAudio(true);
        }
      } catch {
        setRobotAvailable(false);
      }
    };
    checkConnection();
  }, []);

  const scrollToBottom = () => {
    const el = messagesScrollRef.current;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    } else {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const renderActionWidget = (action: { name: string; parameters?: any }) => {
    switch (action.name) {
      case "play_music":
        return <MusicWidget mood={action.parameters?.mood} track={action.parameters?.track} />;
      case "navigate":
        return <NavigationWidget destination={action.parameters?.destination} />;
      case "call_guardian":
        return <CallWidget />;
      case "show_medications":
        return <MedicationsWidget />;
      case "trigger_emergency":
        return <EmergencyWidget />;
      default:
        return null;
    }
  };

  const handleSend = useCallback(async () => {
    if (isTyping) {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      setIsTyping(false);
      messageIdRef.current += 1;
      setMessages((prev) => [
        ...prev,
        {
          id: messageIdRef.current,
          type: "nova",
          content: "❌ Request interrupted and canceled.",
          timestamp: new Date(),
        }
      ]);
      return;
    }

    if (!inputText.trim()) return;

    const userMessageContent = inputText.trim();
    messageIdRef.current += 1;
    const userMessage: Message = {
      id: messageIdRef.current,
      type: "user",
      content: userMessageContent,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    setIsTyping(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      // Get response and action metadata from NovaBot LLM API
      const result = await sendToNovaBot(userMessageContent, { signal: controller.signal });
      const botResponse = result.response;
      const actions = result.actions || [];

      // Handle custom action side effects like pausing the music
      actions.forEach((act: any) => {
        if (act.name === "pause_music") {
          pauseGlobalAudio();
        }
      });

      messageIdRef.current += 1;
      const novaMessage: Message = {
        id: messageIdRef.current,
        type: "nova",
        content: botResponse,
        timestamp: new Date(),
        action: actions.length > 0 ? actions[0] : undefined,
      };

      setMessages((prev) => [...prev, novaMessage]);

      // Speak the response if TTS is enabled
      if (isTTSEnabled) {
        // Robot TTS: speak through robot's physical speaker
        if (useRobotAudio && robotAvailable) {
          robotSpeak({ text: response }).catch((err) => {
            console.warn('[TTS] Robot speak failed, falling back to browser:', err);
            ttsRef.current?.speak(response);
          });
        } else if (ttsRef.current) {
          // Browser TTS fallback
          ttsRef.current.speak(botResponse);
        }
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('[Talk] Request successfully aborted.');
        return;
      }
      console.error('[Talk] Error getting response:', error);

      messageIdRef.current += 1;
      const errorMessage: Message = {
        id: messageIdRef.current,
        type: "nova",
        content: "I'm sorry, I'm having trouble connecting right now. Please check if the NovaBot server is running and try again.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
      setIsConnected(false);
    } finally {
      setIsTyping(false);
      abortControllerRef.current = null;
    }
  }, [inputText, isTyping, isTTSEnabled]);

  const handleVoiceToggle = useCallback(async () => {
    if (isListening) {
      sttRef.current?.stop();
      setIsListening(false);
      return;
    }

    // Stop TTS if speaking
    ttsRef.current?.stop();

    // Try robot microphone first
    if (useRobotAudio && robotAvailable) {
      setIsListening(true);
      try {
        const result = await robotListen(10, 5);
        if (result.status === "success" && result.text) {
          setInputText(result.text);
        }
      } catch (err) {
        console.warn('[STT] Robot listen failed, falling back to browser:', err);
        // Fall through to browser STT
        if (sttRef.current) {
          sttRef.current.start();
          return; // isListening already true
        }
      } finally {
        setIsListening(false);
      }
      return;
    }

    // Browser STT fallback
    if (sttRef.current) {
      const started = sttRef.current.start();
      if (started) {
        setIsListening(true);
      }
    }
  }, [isListening, useRobotAudio, robotAvailable]);

  const handleTTSToggle = useCallback(() => {
    if (ttsRef.current) {
      const newState = ttsRef.current.toggle();
      setIsTTSEnabled(newState);
    }
  }, []);

  const handleRetryConnection = useCallback(async () => {
    setIsCheckingConnection(true);
    const healthy = await checkHealth();
    setIsConnected(healthy);
    setIsCheckingConnection(false);
  }, []);

  const handleClearHistory = useCallback(async () => {
    try {
      await clearHistory();
      if (typeof window !== "undefined") {
        localStorage.removeItem("novabot_chat_messages");
      }
      setMessages([
        {
          id: 1,
          type: "nova",
          content: "Hi! I'm Nova, your AI assistant. How can I help you today? You can type, speak, or use sign language.",
          timestamp: new Date(),
        },
      ]);
      messageIdRef.current = 1;
    } catch (error) {
      console.error('[Talk] Error clearing history:', error);
    }
  }, []);

  // Handle ASL text confirmation - set it as input and send
  const handleASLConfirm = useCallback((text: string) => {
    setIsASLModalOpen(false);
    if (text.trim()) {
      setInputText(text);
      // Auto-send the ASL text
      setTimeout(() => {
        const sendButton = document.querySelector('[data-send-button]') as HTMLButtonElement;
        if (sendButton) sendButton.click();
      }, 100);
    }
  }, []);

  return (
    <div className="h-full min-h-0 flex flex-col w-full animate-fade-in">
      {/* Connection Status Banner */}
      {isConnected === false && (
        <div className="shrink-0 mb-3 sm:mb-4 p-3 sm:p-4 bg-orange-50 dark:bg-orange-900/30 border border-orange-200 dark:border-orange-800 rounded-2xl flex items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <WifiOff className="w-5 h-5 text-orange-500" />
            <span className="text-orange-700 dark:text-orange-300">
              NovaBot server is not connected. Make sure it&apos;s running on port 5000.
            </span>
          </div>
          <button
            onClick={handleRetryConnection}
            disabled={isCheckingConnection}
            className="flex items-center gap-2 px-4 py-2 bg-orange-500 text-white rounded-xl hover:bg-orange-600 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={cn("w-4 h-4", isCheckingConnection && "animate-spin")} />
            Retry
          </button>
        </div>
      )}

      {/* Header */}
      <div className="shrink-0 flex items-center gap-3 sm:gap-4 mb-4 sm:mb-5">
        <Link
          href="/rover"
          className="rover-btn w-14 h-14 rounded-2xl bg-gray-100 dark:bg-gray-800 flex items-center justify-center hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          <ArrowLeft className="w-6 h-6 text-text-secondary dark:text-gray-400" />
        </Link>
        <div className="flex-1">
          <h1 className="text-3xl font-display font-bold text-text-primary dark:text-white">Talk to Nova</h1>
          <p className="text-text-muted dark:text-gray-400">Voice, text, or sign language input</p>
        </div>
        {/* Connection Status & Clear Historys */}
        <div className="flex items-center gap-3 sm:gap-4">
          <div className="flex items-center gap-4">
            {/* NovaBot API */}
            <div className="flex items-center gap-1.5">
              <div
                className={cn(
                  "w-2.5 h-2.5 rounded-full",
                  isConnected === null
                    ? "bg-gray-400 animate-pulse"
                    : isConnected
                      ? "bg-green-500"
                      : "bg-red-500"
                )}
              />
              <span className="text-xs text-text-muted dark:text-gray-400 hidden sm:inline">
                {isConnected === null ? "..." : isConnected ? "AI" : "AI ✗"}
              </span>
            </div>
            {/* Robot */}
            <div className="flex items-center gap-1.5">
              <Bot className="w-3.5 h-3.5 text-text-muted dark:text-gray-400" />
              <div
                className={cn(
                  "w-2.5 h-2.5 rounded-full",
                  robotAvailable ? "bg-green-500" : "bg-gray-400"
                )}
              />
              <span className="text-xs text-text-muted dark:text-gray-400">
                {robotAvailable ? "Robot" : "Robot ✗"}
              </span>
            </div>
            <button
              onClick={handleClearHistory}
              className="rover-btn flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white rounded-xl text-sm font-semibold transition-all border border-red-200 dark:border-red-800/30"
              title="Clear Conversation"
            >
              <Trash2 className="w-4 h-4" />
              <span>Clear Chat</span>
            </button>
          </div>
        </div>
      </div>

      {/* Input Mode Selector */}
      <div className="shrink-0 flex gap-2 sm:gap-4 mb-3 sm:mb-4">
        <button
          onClick={handleVoiceToggle}
          className={cn(
            "rover-btn flex-1 py-4 px-6 rounded-2xl flex items-center justify-center gap-3 transition-all",
            isListening
              ? "bg-accent text-white animate-pulse"
              : "bg-white dark:bg-gray-800 border-2 border-primary text-primary"
          )}
        >
          {isListening ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
          <span className="text-lg font-semibold">{isListening ? "Stop" : "Voice"}</span>
        </button>
        <button
          onClick={() => setIsASLModalOpen(true)}
          className="rover-btn flex-1 py-4 px-6 rounded-2xl bg-white dark:bg-gray-800 border-2 border-secondary text-secondary flex items-center justify-center gap-3 hover:bg-secondary hover:text-white transition-all"
        >
          <Hand className="w-6 h-6" />
          <span className="text-lg font-semibold">Sign Language</span>
        </button>
        <button
          onClick={handleTTSToggle}
          className={cn(
            "rover-btn flex-1 py-4 px-6 rounded-2xl flex items-center justify-center gap-3 transition-all",
            isTTSEnabled
              ? "bg-purple-500 text-white"
              : "bg-white dark:bg-gray-800 border-2 border-purple-500 text-purple-600 dark:text-purple-400"
          )}
        >
          {isTTSEnabled ? <Volume2 className="w-6 h-6" /> : <VolumeX className="w-6 h-6" />}
          <span className="text-lg font-semibold">{isTTSEnabled ? "Audio On" : "Audio Off"}</span>
        </button>
      </div>

      {/* Chat messages: only this region scrolls; flex-1 + min-h-0 required for nested flex overflow */}
      <div
        ref={messagesScrollRef}
        className="flex-1 min-h-0 bg-white dark:bg-gray-800 rounded-3xl shadow-soft border border-gray-100 dark:border-gray-700 p-4 sm:p-6 overflow-y-auto overscroll-contain mb-3 sm:mb-4"
      >
        <div className="space-y-4 sm:space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex flex-col gap-1 w-full",
                message.type === "user" ? "items-end" : "items-start"
              )}
            >
              <div
                className={cn(
                  "max-w-[80%] rounded-2xl px-6 py-4",
                  message.type === "user"
                    ? "bg-primary text-white rounded-br-md"
                    : "bg-gray-100 dark:bg-gray-700 text-text-primary dark:text-white rounded-bl-md"
                )}
              >
                <p className="text-lg">{message.content}</p>
                <p
                  className={cn(
                    "text-sm mt-2",
                    message.type === "user" ? "text-white/70" : "text-text-muted dark:text-gray-400"
                  )}
                >
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
              {/* Dynamic Action Widget */}
              {message.action && (
                <div className="w-full max-w-[80%]">
                  {renderActionWidget(message.action)}
                </div>
              )}
            </div>
          ))}
          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-gray-100 dark:bg-gray-700 rounded-2xl rounded-bl-md px-6 py-4 flex items-center gap-2">
                <Loader2 className="w-5 h-5 text-primary animate-spin" />
                <span className="text-text-muted dark:text-gray-400">Nova is typing...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick Suggestions */}
      <div className="shrink-0 flex gap-2 sm:gap-3 overflow-x-auto pb-2 mb-2 sm:mb-3 [scrollbar-width:thin]">
        {suggestedQuestions.map((question, index) => (
          <button
            key={index}
            onClick={() => setInputText(question)}
            className="rover-btn whitespace-nowrap px-6 py-3 bg-primary-50 dark:bg-primary-900/30 text-primary rounded-2xl text-base font-medium hover:bg-primary-100 dark:hover:bg-primary-900/50 transition-colors"
          >
            {question}
          </button>
        ))}
      </div>

      {/* Text Input */}
      <div className="shrink-0 flex gap-2 sm:gap-4 pt-1">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          placeholder={isListening ? "Listening..." : "Type your message here..."}
          disabled={isTyping}
          className={cn(
            "flex-1 px-6 py-4 text-lg rounded-2xl border-2 border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-text-primary dark:text-white placeholder:text-text-muted dark:placeholder:text-gray-400 focus:border-primary focus:ring-4 focus:ring-primary/20 outline-none transition-all",
            isTyping && "opacity-50 cursor-not-allowed"
          )}
        />
        <button
          onClick={handleSend}
          disabled={!inputText.trim() && !isTyping}
          data-send-button
          className={cn(
            "rover-btn px-8 py-4 rounded-2xl flex items-center gap-3 transition-all",
            (inputText.trim() || isTyping)
              ? (isTyping
                ? "bg-red-500 hover:bg-red-600 text-white shadow-[0_0_15px_rgba(239,68,68,0.4)] hover:scale-105 active:scale-95"
                : "bg-primary text-white hover:bg-primary-600 hover:scale-105 active:scale-95")
              : "bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed"
          )}
        >
          {isTyping ? (
            <X className="w-6 h-6 animate-pulse" />
          ) : (
            <Send className="w-6 h-6" />
          )}
          <span className="text-lg font-semibold">{isTyping ? "Cancel" : "Send"}</span>
        </button>
      </div>

      {/* ASL Recognition Modal */}
      <ASLRecognitionModal
        isOpen={isASLModalOpen}
        onClose={() => setIsASLModalOpen(false)}
        onConfirm={handleASLConfirm}
      />
    </div>
  );
}
