import { getDynamicUrl } from "./utils";

/**
 * NovaCare — Speech Services (STT & TTS)
 * 
 * Uses standard Web Speech API (SpeechRecognition and SpeechSynthesis).
 */

// ---------------------------------------------------------------------------
// Speech-to-Text (STT)
// ---------------------------------------------------------------------------

interface STTOptions {
  lang?: string;
  continuous?: boolean;
  interimResults?: boolean;
}

export class STTService {
  private recognition: any = null;
  private transcriptCallback: ((text: string) => void) | null = null;
  private errorCallback: ((error: string, message: string) => void) | null = null;
  private options: STTOptions;

  constructor(options: STTOptions = {}) {
    this.options = options;
    this.initRecognition();
  }

  private initRecognition() {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        this.recognition.lang = this.options.lang || 'en-US';
        this.recognition.continuous = this.options.continuous || false;
        this.recognition.interimResults = this.options.interimResults || false;

        this.recognition.onresult = (event: any) => {
          const transcript = Array.from(event.results)
            .map((result: any) => result[0])
            .map((result: any) => result.transcript)
            .join('');
          
          if (this.transcriptCallback) {
            this.transcriptCallback(transcript);
          }
        };

        this.recognition.onerror = (event: any) => {
          if (this.errorCallback) {
            this.errorCallback(event.error, event.message);
          }
        };

        this.recognition.onend = () => {
          if (this.endCallback) {
            this.endCallback();
          }
        };
      }
    }
  }

  onTranscript(cb: (text: string) => void) { this.transcriptCallback = cb; }
  onError(cb: (err: string, msg: string) => void) { this.errorCallback = cb; }
  onEnd(cb: () => void) { this.endCallback = cb; }

  start(): boolean {
    if (!this.recognition) return false;
    try {
      this.recognition.start();
      return true;
    } catch (e) {
      console.error("[STT] Start error:", e);
      return false;
    }
  }

  stop() {
    if (this.recognition) {
      this.recognition.stop();
    }
  }

  abort() {
    if (this.recognition) {
      this.recognition.abort();
    }
  }

  destroy() {
    if (this.recognition) {
      this.recognition.abort();
      this.recognition.onend = null;
      this.recognition.onerror = null;
      this.recognition.onresult = null;
      this.recognition = null;
    }
  }

  recreate() {
    if (!this.recognition) {
      this.initRecognition();
    }
  }
}

// ---------------------------------------------------------------------------
// Text-to-Speech (TTS)
// ---------------------------------------------------------------------------

interface TTSOptions {
  rate?: number;
  pitch?: number;
  volume?: number;
}

function cleanForTts(text: string): string {
  return text
    .replace(/\*\*/g, "")
    .replace(/\*/g, "")
    .replace(/#{1,6}\s/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .trim();
}

export class TTSService {
  private synth: SpeechSynthesis | null = null;
  private enabled: boolean = true;
  private options: TTSOptions;
  private activeAudio: HTMLAudioElement | null = null;
  private objectUrl: string | null = null;

  constructor(options: TTSOptions = {}) {
    if (typeof window !== 'undefined') {
      this.synth = window.speechSynthesis;
    }
    this.options = {
      rate: options.rate || 1.0,
      pitch: options.pitch || 1.0,
      volume: options.volume || 1.0,
    };
  }

  async speak(text: string) {
    this.stop();

    if (!this.enabled || !text.trim()) return;

    const cleanText = cleanForTts(text);
    const pocketUrl = process.env.NEXT_PUBLIC_POCKET_TTS_URL || "http://localhost:8002";

    if (pocketUrl && typeof window !== "undefined") {
      try {
        const body = new URLSearchParams();
        body.set("text", cleanText);
        const voiceUrl = process.env.NEXT_PUBLIC_POCKET_TTS_VOICE_URL;
        if (voiceUrl) {
          body.set("voice_url", voiceUrl);
        }

        const res = await fetch(`${getDynamicUrl(pocketUrl)}/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
          body: body.toString(),
          signal: AbortSignal.timeout(60000), // 60s timeout
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        
        this.objectUrl = URL.createObjectURL(blob);
        const audio = new Audio(this.objectUrl);
        this.activeAudio = audio;
        audio.volume = this.options.volume || 1.0;
        
        await audio.play();
        return;
      } catch (err) {
        console.warn("[TTS] Pocket TTS failed, falling back to browser SpeechSynthesis:", err);
      }
    }

    // Fallback: standard browser speechSynthesis
    if (this.synth) {
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.rate = this.options.rate!;
      utterance.pitch = this.options.pitch!;
      utterance.volume = this.options.volume!;
      
      this.synth.speak(utterance);
    }
  }

  stop() {
    if (this.activeAudio) {
      this.activeAudio.pause();
      this.activeAudio.src = "";
      this.activeAudio = null;
    }
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
    if (this.synth) {
      this.synth.cancel();
    }
  }

  toggle(): boolean {
    this.enabled = !this.enabled;
    if (!this.enabled) this.stop();
    return this.enabled;
  }
}
