export class SpeechService {
  private recognition: any = null;
  private synthesis = typeof window !== 'undefined' ? window.speechSynthesis : null;
  private isListening = false;
  private onResultCallback: (text: string) => void = () => {};
  private onEndCallback: () => void = () => {};
  private onStartCallback: () => void = () => {};

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
          let finalTranscript = '';
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
              finalTranscript += event.results[i][0].transcript;
            }
          }
          if (finalTranscript) {
            this.onResultCallback(finalTranscript.trim());
          }
        };

        this.recognition.onend = () => {
          this.isListening = false;
          this.onEndCallback();
        };
        
        this.recognition.onerror = (event: any) => {
          console.error("Speech recognition error", event.error);
        };
      } else {
        console.warn("SpeechRecognition not supported in this browser.");
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
    if (this.synthesis.speaking) {
      this.synthesis.cancel();
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    
    const voices = this.synthesis.getVoices();
    const novaVoice = voices.find(v => v.name.includes('Female') || v.name.includes('Samantha') || v.name.includes('Google US English'));
    if (novaVoice) {
      utterance.voice = novaVoice;
    }
    
    utterance.pitch = 1.1;
    utterance.rate = 1.0;
    
    if (onEnd) {
      utterance.onend = onEnd;
    }
    this.synthesis.speak(utterance);
  }
  
  stopSpeaking() {
    if (this.synthesis && this.synthesis.speaking) {
      this.synthesis.cancel();
    }
  }

  onResult(callback: (text: string) => void) {
    this.onResultCallback = callback;
  }

  onEnd(callback: () => void) {
    this.onEndCallback = callback;
  }
  
  onStart(callback: () => void) {
    this.onStartCallback = callback;
  }
}

export const speechService = typeof window !== 'undefined' ? new SpeechService() : null;
