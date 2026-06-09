'use client';

/**
 * VoiceDebugPanel
 *
 * Shows real-time voice navigation debug information:
 * - Detected intent
 * - Resolved route
 * - Route validity
 * - Navigation result
 *
 * Rendered only in development or when explicitly enabled via
 * the `NEXT_PUBLIC_VOICE_DEBUG=true` env variable.
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bug, ChevronDown, ChevronUp, Trash2, CheckCircle2, XCircle, ArrowLeftRight, AlertCircle } from 'lucide-react';
import { useVoice, NavDebugEvent } from '../VoiceContext';

function StatusIcon({ result }: { result: NavDebugEvent['navigationResult'] }) {
  switch (result) {
    case 'success':       return <CheckCircle2  size={14} className="text-emerald-400 flex-shrink-0" />;
    case 'blocked':       return <XCircle       size={14} className="text-red-400     flex-shrink-0" />;
    case 'browser-history': return <ArrowLeftRight size={14} className="text-blue-400  flex-shrink-0" />;
    case 'skipped':       return <AlertCircle   size={14} className="text-amber-400   flex-shrink-0" />;
  }
}

function DebugRow({ event }: { event: NavDebugEvent }) {
  const [open, setOpen] = useState(false);
  const time = new Date(event.timestamp).toLocaleTimeString();

  return (
    <div className="border border-slate-700/60 rounded-lg overflow-hidden text-xs">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 bg-slate-800/60 hover:bg-slate-800 transition-colors text-left"
        onClick={() => setOpen((p) => !p)}
      >
        <StatusIcon result={event.navigationResult} />
        <span className="text-slate-300 font-mono flex-1 truncate">
          &ldquo;{event.transcript}&rdquo;
        </span>
        <span className="text-slate-500 ml-auto mr-1 flex-shrink-0">{time}</span>
        {open ? <ChevronUp size={12} className="text-slate-500 flex-shrink-0" /> : <ChevronDown size={12} className="text-slate-500 flex-shrink-0" />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="px-3 py-2 bg-slate-900/80 grid grid-cols-2 gap-x-4 gap-y-1 font-mono">
              <span className="text-slate-500">Intent</span>
              <span className="text-amber-300">{event.detectedIntent}</span>

              <span className="text-slate-500">Route</span>
              <span className={event.resolvedRoute ? 'text-emerald-300' : 'text-red-400'}>
                {event.resolvedRoute ?? 'none'}
              </span>

              <span className="text-slate-500">Valid</span>
              <span className={event.routeValid ? 'text-emerald-300' : 'text-red-400'}>
                {event.routeValid ? 'yes' : 'no'}
              </span>

              <span className="text-slate-500">Result</span>
              <span className={
                event.navigationResult === 'success' ? 'text-emerald-300' :
                event.navigationResult === 'blocked' ? 'text-red-400' :
                'text-blue-300'
              }>
                {event.navigationResult}
              </span>

              <span className="text-slate-500 col-span-2 mt-1">Reason</span>
              <span className="text-slate-300 col-span-2 break-words">{event.reason}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function VoiceDebugPanel() {
  const { debugLog, clearDebugLog, currentRole } = useVoice();
  const [isOpen, setIsOpen] = useState(false);

  // Only show in development or when env flag is set
  const isDebugEnabled =
    process.env.NODE_ENV === 'development' ||
    process.env.NEXT_PUBLIC_VOICE_DEBUG === 'true';

  if (!isDebugEnabled) return null;

  return (
    <div className="fixed bottom-24 left-4 z-50">
      <motion.button
        onClick={() => setIsOpen((p) => !p)}
        className="flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-900/90 border border-slate-700/60 text-slate-300 text-xs font-mono shadow-lg hover:bg-slate-800 transition-colors backdrop-blur-sm"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        title="Toggle Nova Voice Debug Panel"
      >
        <Bug size={14} className="text-amber-400" />
        <span>Nova Debug</span>
        {debugLog.length > 0 && (
          <span className="ml-1 bg-amber-500/20 text-amber-300 text-[10px] px-1.5 py-0.5 rounded-full">
            {debugLog.length}
          </span>
        )}
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.96 }}
            transition={{ type: 'spring', damping: 22, stiffness: 350 }}
            className="absolute bottom-11 left-0 w-80 bg-slate-950/95 border border-slate-700/60 rounded-2xl shadow-2xl overflow-hidden backdrop-blur-xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/80">
              <div className="flex items-center gap-2">
                <Bug size={14} className="text-amber-400" />
                <span className="text-xs font-semibold text-slate-200 font-mono">Nova Route Debugger</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 font-mono">
                  role: {currentRole}
                </span>
                {debugLog.length > 0 && (
                  <button
                    onClick={clearDebugLog}
                    className="text-slate-500 hover:text-red-400 transition-colors"
                    title="Clear log"
                  >
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
            </div>

            {/* Legend */}
            <div className="flex items-center gap-3 px-4 py-2 bg-slate-900/60 border-b border-slate-800/40 text-[10px] text-slate-500 font-mono">
              <span className="flex items-center gap-1"><CheckCircle2 size={10} className="text-emerald-400" /> success</span>
              <span className="flex items-center gap-1"><XCircle size={10} className="text-red-400" /> blocked</span>
              <span className="flex items-center gap-1"><ArrowLeftRight size={10} className="text-blue-400" /> history</span>
            </div>

            {/* Log entries */}
            <div className="max-h-72 overflow-y-auto p-3 space-y-2">
              {debugLog.length === 0 ? (
                <p className="text-center text-slate-600 text-xs py-6 font-mono">
                  No navigation events yet.<br />Try a voice command.
                </p>
              ) : (
                debugLog.map((event, i) => (
                  <DebugRow key={`${event.timestamp}-${i}`} event={event} />
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
