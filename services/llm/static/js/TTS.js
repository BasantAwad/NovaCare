class TTS {
    constructor(options = {}) {
        this.config = {
            enabled: options.enabled !== false,
            rate: options.rate ?? 0.9,
            pitch: options.pitch ?? 0.9,
            volume: options.volume ?? 1.0,
            lang: options.lang ?? 'en-US',
            voiceName: options.voice ?? null
        };

        /** @type {HTMLAudioElement|null} */
        this._edgeAudio = null;
        /** @type {string|null} */
        this._edgeObjectUrl = null;
        this._edgeTimeoutMs = (typeof window !== 'undefined' && window.NOVACARE_EDGE_TTS_TIMEOUT_MS)
            ? parseInt(String(window.NOVACARE_EDGE_TTS_TIMEOUT_MS), 10) || 90000
            : 90000;

        this.state = {
            utterance: null,
            speaking: false
        };

        this.callbacks = {
            start: null,
            end: null,
            error: null,
            status: null
        };

        this.voices = [];
        this._initVoices();
    }

    /* ------------------------------------------------------------------ */
    /* Initialization                                                     */
    /* ------------------------------------------------------------------ */

    _initVoices() {
        if (!this.isSupported()) return;

        const loadVoices = () => {
            this.voices = window.speechSynthesis.getVoices();
        };

        loadVoices();

        if (!this.voices.length) {
            window.speechSynthesis.onvoiceschanged = loadVoices;
        }
    }

    /* ------------------------------------------------------------------ */
    /* Core API                                                           */
    /* ------------------------------------------------------------------ */

    _pocketBaseUrl() {
        if (typeof window === 'undefined' || !window.NOVACARE_POCKET_TTS_URL) return '';
        return String(window.NOVACARE_POCKET_TTS_URL).replace(/\/$/, '');
    }

    _pocketVoiceUrl() {
        if (typeof window === 'undefined' || !window.NOVACARE_POCKET_TTS_VOICE_URL) return '';
        return String(window.NOVACARE_POCKET_TTS_VOICE_URL).trim();
    }

    _edgeBaseUrl() {
        if (typeof window === 'undefined' || !window.NOVACARE_EDGE_TTS_URL) return '';
        return String(window.NOVACARE_EDGE_TTS_URL).replace(/\/$/, '');
    }

    _cleanForTts(text) {
        return text
            .replace(/\*\*/g, '')
            .replace(/\*/g, '')
            .replace(/#{1,6}\s/g, '')
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            .trim();
    }

    _clearEdgeAudio() {
        if (this._edgeAudio) {
            const a = this._edgeAudio;
            a.onplay = null;
            a.onended = null;
            a.onerror = null;
            a.pause();
            a.src = '';
            this._edgeAudio = null;
        }
        if (this._edgeObjectUrl) {
            URL.revokeObjectURL(this._edgeObjectUrl);
            this._edgeObjectUrl = null;
        }
    }

    _speakPocketDirect(baseUrl, cleanText, options, originalText) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this._edgeTimeoutMs);

        const body = new URLSearchParams();
        body.set('text', cleanText);
        const vu = this._pocketVoiceUrl();
        if (vu) body.set('voice_url', vu);

        fetch(`${baseUrl}/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
            body: body.toString(),
            signal: controller.signal
        })
            .then((r) => {
                clearTimeout(timer);
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.blob();
            })
            .then((blob) => this._playWavBlob(blob, cleanText, options, originalText))
            .catch(() => {
                clearTimeout(timer);
                this._speakWebOnly(cleanText, options, originalText);
            });
    }

    _speakEdge(baseUrl, cleanText, options, originalText) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this._edgeTimeoutMs);

        fetch(`${baseUrl}/api/speak`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: cleanText }),
            signal: controller.signal
        })
            .then((r) => {
                clearTimeout(timer);
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.blob();
            })
            .then((blob) => this._playWavBlob(blob, cleanText, options, originalText))
            .catch(() => {
                clearTimeout(timer);
                this._speakWebOnly(cleanText, options, originalText);
            });
    }

    _playWavBlob(blob, cleanText, options, originalText) {
        const url = URL.createObjectURL(blob);
        this._edgeObjectUrl = url;
        const audio = new Audio(url);
        this._edgeAudio = audio;
        audio.volume = options.volume ?? this.config.volume ?? 1.0;
        audio.onplay = () => {
            this.state.speaking = true;
            this._updateStatus('Speaking...');
            this.callbacks.start?.(originalText);
        };
        audio.onended = () => {
            this._clearEdgeAudio();
            this._resetState();
            this._updateStatus('Finished speaking');
            this.callbacks.end?.(originalText);
        };
        audio.onerror = () => {
            this._clearEdgeAudio();
            this._resetState();
            this._speakWebOnly(cleanText, options, originalText);
        };
        return audio.play().catch(() => {
            this._clearEdgeAudio();
            this._speakWebOnly(cleanText, options, originalText);
        });
    }

    _speakWebOnly(cleanText, options, originalText) {
        if (!this.isSupported()) {
            this._emitError('not-supported', 'TTS not supported');
            return;
        }
        const utterance = new SpeechSynthesisUtterance(cleanText);
        this._applyOptions(utterance, options);
        this._attachEvents(utterance, originalText);
        window.speechSynthesis.speak(utterance);
    }

    speak(text, options = {}) {
        if (!this._canSpeak(text)) return false;

        this.stop();

        const cleanText = this._cleanForTts(text);
        const pocket = this._pocketBaseUrl();
        if (pocket && typeof fetch !== 'undefined') {
            this._speakPocketDirect(pocket, cleanText, options, text);
            return true;
        }
        const base = this._edgeBaseUrl();
        if (base && typeof fetch !== 'undefined') {
            this._speakEdge(base, cleanText, options, text);
            return true;
        }

        const utterance = new SpeechSynthesisUtterance(cleanText);
        this._applyOptions(utterance, options);
        this._attachEvents(utterance, text);

        window.speechSynthesis.speak(utterance);
        return true;
    }

    stop() {
        this._clearEdgeAudio();
        if (this.isSupported()) {
            window.speechSynthesis.cancel();
        }
        this._resetState();
        this._updateStatus('Speech stopped');
    }

    pause() {
        if (this.isSupported() && this.isCurrentlySpeaking()) {
            window.speechSynthesis.pause();
            this._updateStatus('Paused');
        }
    }

    resume() {
        if (this.isSupported()) {
            window.speechSynthesis.resume();
            this._updateStatus('Speaking...');
        }
    }

    toggle() {
        this.config.enabled = !this.config.enabled;
        this._updateStatus(
            `Voice output ${this.config.enabled ? 'enabled' : 'disabled'}`
        );

        if (!this.config.enabled) this.stop();
        return this.config.enabled;
    }

    enable() {
        this.config.enabled = true;
    }

    disable() {
        this.config.enabled = false;
        this.stop();
    }

    /* ------------------------------------------------------------------ */
    /* Status Checks                                                      */
    /* ------------------------------------------------------------------ */

    isSupported() {
        return typeof window !== 'undefined' && 'speechSynthesis' in window;
    }

    isCurrentlySpeaking() {
        const edgePlaying =
            this._edgeAudio && !this._edgeAudio.paused && !this._edgeAudio.ended;
        return (
            edgePlaying ||
            this.state.speaking ||
            (this.isSupported() && window.speechSynthesis.speaking)
        );
    }

    isPaused() {
        return this.isSupported() && window.speechSynthesis.paused;
    }

    getVoices() {
        return this.isSupported() ? window.speechSynthesis.getVoices() : [];
    }

    /* ------------------------------------------------------------------ */
    /* Voice Selection                                                    */
    /* ------------------------------------------------------------------ */

    _selectVoice(name) {
        if (!this.voices.length) return null;

        if (name) {
            return this.voices.find(v =>
                v.name.toLowerCase().includes(name.toLowerCase())
            );
        }

        return this._preferredMaleVoice()
            || this.voices.find(v => v.lang.startsWith('en'))
            || this.voices[0];
    }

    _preferredMaleVoice() {
        const priority = ['daniel', 'alex', 'david', 'mark', 'male'];

        return this.voices.find(v =>
            priority.some(p => v.name.toLowerCase().includes(p))
        );
    }

    setVoice(voiceName) {
        this.config.voiceName = voiceName;
    }

    /* ------------------------------------------------------------------ */
    /* Configuration                                                      */
    /* ------------------------------------------------------------------ */

    setRate(rate) {
        this.config.rate = this._clamp(rate, 0.1, 10);
    }

    setPitch(pitch) {
        this.config.pitch = this._clamp(pitch, 0, 2);
    }

    setVolume(volume) {
        this.config.volume = this._clamp(volume, 0, 1);
    }

    setLanguage(lang) {
        this.config.lang = lang;
    }

    /* ------------------------------------------------------------------ */
    /* Callbacks                                                          */
    /* ------------------------------------------------------------------ */

    onStart(cb)  { this.callbacks.start = cb; }
    onEnd(cb)    { this.callbacks.end = cb; }
    onError(cb)  { this.callbacks.error = cb; }
    onStatus(cb) { this.callbacks.status = cb; }

    /* ------------------------------------------------------------------ */
    /* Internal Helpers                                                   */
    /* ------------------------------------------------------------------ */

    _canSpeak(text) {
        if (!this.config.enabled) return false;
        if (!text || !text.trim()) return false;
        if (this._pocketBaseUrl() && typeof fetch !== 'undefined') return true;
        const base = this._edgeBaseUrl();
        if (base && typeof fetch !== 'undefined') return true;
        if (!this.isSupported()) {
            this._emitError('not-supported', 'TTS not supported');
            return false;
        }
        return true;
    }

    _applyOptions(utterance, options) {
        utterance.rate   = options.rate   ?? this.config.rate;
        utterance.pitch  = options.pitch  ?? this.config.pitch;
        utterance.volume = options.volume ?? this.config.volume;
        utterance.lang   = options.lang   ?? this.config.lang;

        utterance.voice = this._selectVoice(
            options.voice ?? this.config.voiceName
        );
    }

    _attachEvents(utterance, text) {
        utterance.onstart = () => {
            this.state.speaking = true;
            this.state.utterance = utterance;
            this._updateStatus('Speaking...');
            this.callbacks.start?.(text);
        };

        utterance.onend = () => {
            this._resetState();
            this._updateStatus('Finished speaking');
            this.callbacks.end?.(text);
        };

        utterance.onerror = (e) => {
            this._resetState();
            this._emitError(e.error, 'TTS error');
        };
    }

    _emitError(code, message) {
        this._updateStatus(message);
        this.callbacks.error?.(code, message);
    }

    _resetState() {
        this.state.speaking = false;
        this.state.utterance = null;
    }

    _updateStatus(message) {
        this.callbacks.status?.(message);
    }

    _clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }
}

/* ------------------------------------------------------------------ */
/* Exports                                                            */
/* ------------------------------------------------------------------ */

if (typeof module !== 'undefined' && module.exports) {
    module.exports = TTS;
}

// window.TTS = TTS;
