'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, X, Sparkles, Shield, Lightbulb } from 'lucide-react';
import { useVoice } from '../VoiceContext';

const ROLE_COLORS: Record<string, string> = {
  medical:  'bg-emerald-500/20 text-emerald-300',
  guardian: 'bg-purple-500/20  text-purple-300',
  admin:    'bg-red-500/20     text-red-300',
  rover:    'bg-blue-500/20    text-blue-300',
  unknown:  'bg-slate-500/20   text-slate-400',
};

// Role-aware example cues shown while listening
const ROLE_CUES: Record<string, string[]> = {
  medical:  ['Show medications', 'Open vitals', 'Go to appointments', 'Show health report'],
  guardian: ['Go to activity', 'Show medications', 'Open communication', 'Go back'],
  admin:    ['Open settings', 'Go to services', 'Show users'],
  rover:    ['Take me to the kitchen', 'Call my guardian', "What's my medication?", 'Play music'],
  unknown:  ['Go to settings', 'Go back', 'Scroll down'],
};

export function VoiceAssistantUI() {
  const {
    isActive, isListening, isSpeaking,
    transcript, interimTranscript, response,
    currentRole, toggleAssistant, processTextCommand,
  } = useVoice();

  const [inputValue, setInputValue] = React.useState('');
  const [cueIndex, setCueIndex] = React.useState(0);

  // Rotate cues every 3s while listening
  const cues = ROLE_CUES[currentRole] ?? ROLE_CUES.unknown;
  React.useEffect(() => {
    if (!isListening) return;
    const id = setInterval(() => setCueIndex((i) => (i + 1) % cues.length), 3000);
    return () => clearInterval(id);
  }, [isListening, cues.length]);

  return (
    <>
      <motion.button
        className="fixed bottom-6 right-6 z-50 p-4 rounded-full bg-blue-600 text-white shadow-lg hover:bg-blue-700 transition-colors focus:outline-none focus:ring-4 focus:ring-blue-300"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={toggleAssistant}
        aria-label="Toggle Nova Voice Assistant"
      >
        {isActive ? <MicOff size={24} /> : <Mic size={24} />}
      </motion.button>

      <AnimatePresence>
        {isActive && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            className="fixed bottom-24 right-6 z-50 w-80 sm:w-96 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border border-white/20 dark:border-slate-700/30 rounded-3xl shadow-2xl overflow-hidden"
          >
            <div className="p-5 flex flex-col h-full relative">
              <button
                onClick={toggleAssistant}
                className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
              >
                <X size={20} />
              </button>

              {/* Header */}
              <div className="flex items-center space-x-2 mb-4">
                <div className="bg-blue-100 dark:bg-blue-900/30 p-2 rounded-full text-blue-600 dark:text-blue-400">
                  <Sparkles size={20} />
                </div>
                <h3 className="text-lg font-semibold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400">
                  Nova AI
                </h3>
                {currentRole !== 'unknown' && (
                  <span className={`ml-auto flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full ${ROLE_COLORS[currentRole] ?? ROLE_COLORS.unknown}`}>
                    <Shield size={9} />
                    {currentRole}
                  </span>
                )}
              </div>

              {/* Chat bubbles */}
              <div className="flex-1 py-2 space-y-3 min-h-[80px]">
                {/* Confirmed transcript */}
                {transcript && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex justify-end"
                  >
                    <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl rounded-tr-none px-4 py-2 text-sm text-slate-700 dark:text-slate-300 max-w-[85%]">
                      {transcript}
                    </div>
                  </motion.div>
                )}

                {/* Live interim transcript (streaming) */}
                <AnimatePresence>
                  {interimTranscript && (
                    <motion.div
                      key="interim"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0 }}
                      className="flex justify-end"
                    >
                      <div className="bg-slate-200/60 dark:bg-slate-700/60 border border-dashed border-slate-300 dark:border-slate-600 rounded-2xl rounded-tr-none px-4 py-2 text-sm text-slate-500 dark:text-slate-400 max-w-[85%] italic">
                        {interimTranscript}
                        <span className="inline-block w-1 h-3 bg-blue-400 ml-1 animate-pulse rounded-sm align-middle" />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Nova response */}
                {response && (
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex justify-start"
                  >
                    <div className="bg-blue-600 text-white rounded-2xl rounded-tl-none px-4 py-2 text-sm max-w-[85%] shadow-md">
                      {response}
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Cue suggestion while listening */}
              <AnimatePresence mode="wait">
                {isListening && !interimTranscript && (
                  <motion.div
                    key={cueIndex}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.3 }}
                    className="flex items-center gap-2 mb-3 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-2xl border border-blue-100 dark:border-blue-800"
                  >
                    <Lightbulb size={13} className="text-blue-400 shrink-0" />
                    <span className="text-xs text-blue-500 dark:text-blue-300 italic truncate">
                      Try: &ldquo;{cues[cueIndex]}&rdquo;
                    </span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Text input */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (inputValue.trim()) {
                    processTextCommand(inputValue.trim());
                    setInputValue('');
                  }
                }}
                className="mb-3 flex items-center space-x-2 bg-slate-100/80 dark:bg-slate-800/80 p-2 rounded-2xl border border-slate-200/50 dark:border-slate-700/50 backdrop-blur-sm"
              >
                <input
                  type="text"
                  placeholder="Type a command…"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  className="flex-1 bg-transparent text-sm outline-none px-2 text-slate-800 dark:text-slate-200 placeholder-slate-400"
                />
                <button
                  type="submit"
                  disabled={!inputValue.trim()}
                  className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white p-1.5 rounded-xl transition-all shadow-md focus:outline-none flex items-center justify-center"
                >
                  <Sparkles size={16} />
                </button>
              </form>

              {/* Status bar */}
              <div className="pt-3 flex flex-col items-center justify-center border-t border-slate-200 dark:border-slate-800">
                <div className="h-8 flex items-center justify-center space-x-1">
                  {isSpeaking || isListening ? (
                    <Waveform isSpeaking={isSpeaking} />
                  ) : (
                    <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Ready
                    </span>
                  )}
                </div>
                <span className="text-xs text-slate-400 mt-1">
                  {isSpeaking
                    ? 'Nova is speaking…'
                    : isListening
                    ? interimTranscript
                      ? 'Hearing you…'
                      : 'Listening…'
                    : 'Say "Hey Nova" or click mic'}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function Waveform({ isSpeaking }: { isSpeaking: boolean }) {
  const bars = 5;
  return (
    <div className="flex items-center space-x-1 h-6">
      {Array.from({ length: bars }).map((_, i) => (
        <motion.div
          key={i}
          className={`w-1.5 rounded-full ${isSpeaking ? 'bg-blue-500' : 'bg-emerald-500'}`}
          animate={{
            height: isSpeaking
              ? ['20%', '100%', '40%', '80%', '30%']
              : ['20%', '80%', '40%', '100%', '20%'],
          }}
          transition={{
            repeat: Infinity,
            duration: isSpeaking ? 0.6 : 1.2,
            ease: 'easeInOut',
            delay: i * 0.1,
          }}
          style={{ height: '20%' }}
        />
      ))}
    </div>
  );
}
