export class SpeechService {
  private recognition: any = null;
  private synthesis = typeof window !== 'undefined' ? window.speechSynthesis : null;
  private isListening = false;
  private onResultCallback: (text: string) => void = () => {};
  private onInterimCallback: (text: string) => void = () => {};
  private onEndCallback: () => void = () => {};
  private onStartCallback: () => void = () => {};
  private onSpeakEndCallback: () => void = () => {};
  // Chrome speechSynthesis bug workaround — poll to detect when speaking stops
  private speakPollInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
          this.isListening = true;
          this.onStartCallback();
        };

        this.recognition.onresult = (event: any) => {
          let interimTranscript = '';
          let finalTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; ++i) {
            const t = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += t;
            } else {
              interimTranscript += t;
            }
          }

          // Stream interim results immediately for live display
          if (interimTranscript) {
            this.onInterimCallback(interimTranscript);
          }

          // Only trigger command processing on final result
          if (finalTranscript) {
            this.onInterimCallback(''); // clear interim
            this.onResultCallback(finalTranscript.trim());
          }
        };

        this.recognition.onend = () => {
          this.isListening = false;
          this.onEndCallback();
        };

        this.recognition.onerror = (event: any) => {
          // 'no-speech' is harmless — browser auto-ended, onEnd handles restart
          if (event.error !== 'no-speech') {
            console.error('Speech recognition error', event.error);
          }
        };
      } else {
        console.warn('SpeechRecognition not supported in this browser.');
      }
    }
  }

  startListening() {
    if (this.recognition && !this.isListening) {
      try {
        this.recognition.start();
      } catch (e) {
        console.error(e);
      }
    }
  }

  stopListening() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
    }
  }

  speak(text: string, onEnd?: () => void) {
    if (!this.synthesis) return;

    // Cancel any current speech
    if (this.synthesis.speaking) {
      this.synthesis.cancel();
    }
    this._clearSpeakPoll();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';

    const voices = this.synthesis.getVoices();
    const novaVoice = voices.find(
      (v) => v.name.includes('Female') || v.name.includes('Samantha') || v.name.includes('Google US English'),
    );
    if (novaVoice) utterance.voice = novaVoice;

    utterance.pitch = 1.1;
    utterance.rate = 1.0;

    const handleEnd = () => {
      this._clearSpeakPoll();
      this.onSpeakEndCallback();
      onEnd?.();
    };

    utterance.onend = handleEnd;

    // Chrome bug workaround: speechSynthesis.speaking can get stuck.
    // Poll every 200ms; if speaking has stopped and utterance duration has
    // elapsed, fire the end handler manually.
    const startedAt = Date.now();
    const estimatedDurationMs = Math.max(2000, (text.split(' ').length / 3) * 1000);

    this.speakPollInterval = setInterval(() => {
      if (!this.synthesis) return;

      // Workaround: Chrome pauses synthesis on background tabs — resume it
      if (this.synthesis.paused) {
        this.synthesis.resume();
      }

      const elapsed = Date.now() - startedAt;
      if (!this.synthesis.speaking && elapsed > 500) {
        // Speech ended (either naturally or via cancel)
        handleEnd();
      } else if (elapsed > estimatedDurationMs + 3000) {
        // Hard safety timeout: force-end after estimated duration + 3s buffer
        this.synthesis.cancel();
        handleEnd();
      }
    }, 200);

    this.synthesis.speak(utterance);
  }

  stopSpeaking() {
    this._clearSpeakPoll();
    if (this.synthesis && this.synthesis.speaking) {
      this.synthesis.cancel();
    }
    this.onSpeakEndCallback();
  }

  private _clearSpeakPoll() {
    if (this.speakPollInterval !== null) {
      clearInterval(this.speakPollInterval);
      this.speakPollInterval = null;
    }
  }

  onResult(callback: (text: string) => void) {
    this.onResultCallback = callback;
  }

  /** Called repeatedly with partial transcripts as the user speaks */
  onInterim(callback: (text: string) => void) {
    this.onInterimCallback = callback;
  }

  /** Called when TTS finishes speaking */
  onSpeakEnd(callback: () => void) {
    this.onSpeakEndCallback = callback;
  }

  onEnd(callback: () => void) {
    this.onEndCallback = callback;
  }

  onStart(callback: () => void) {
    this.onStartCallback = callback;
  }
}

export const speechService = typeof window !== 'undefined' ? new SpeechService() : null;
