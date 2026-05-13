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
  private endCallback: (() => void) | null = null;

  constructor(options: STTOptions = {}) {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        this.recognition.lang = options.lang || 'en-US';
        this.recognition.continuous = options.continuous || false;
        this.recognition.interimResults = options.interimResults || false;

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
}

// ---------------------------------------------------------------------------
// Text-to-Speech (TTS)
// ---------------------------------------------------------------------------

interface TTSOptions {
  rate?: number;
  pitch?: number;
  volume?: number;
}

export class TTSService {
  private synth: SpeechSynthesis | null = null;
  private enabled: boolean = true;
  private options: TTSOptions;

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

  speak(text: string) {
    if (!this.synth || !this.enabled) return;
    
    // Stop current speaking
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = this.options.rate!;
    utterance.pitch = this.options.pitch!;
    utterance.volume = this.options.volume!;
    
    this.synth.speak(utterance);
  }

  stop() {
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
