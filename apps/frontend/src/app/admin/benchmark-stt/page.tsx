"use client";

import { useState, useEffect, useRef } from "react";
import { Mic, MicOff, RefreshCw, CheckCircle, Activity, BarChart2 } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

// Helper to calculate Levenshtein distance for Word Error Rate (WER)
function calculateWER(reference: string, hypothesis: string) {
  const refWords = reference.toLowerCase().replace(/[^\w\s]|_/g, "").split(/\s+/).filter(w => w);
  const hypWords = hypothesis.toLowerCase().replace(/[^\w\s]|_/g, "").split(/\s+/).filter(w => w);

  const matrix = Array(refWords.length + 1).fill(null).map(() => Array(hypWords.length + 1).fill(null));

  for (let i = 0; i <= refWords.length; i++) matrix[i][0] = i;
  for (let j = 0; j <= hypWords.length; j++) matrix[0][j] = j;

  for (let i = 1; i <= refWords.length; i++) {
    for (let j = 1; j <= hypWords.length; j++) {
      if (refWords[i - 1] === hypWords[j - 1]) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1,     // insertion
          matrix[i - 1][j] + 1      // deletion
        );
      }
    }
  }

  const distance = matrix[refWords.length][hypWords.length];
  const wer = distance / Math.max(refWords.length, 1);
  return { distance, wer, refWordCount: refWords.length };
}

const TEST_PHRASES = [
  "Nova take me to the kitchen.",
  "I am feeling dizzy and I need you to call my guardian immediately.",
  "What is my current heart rate and oxygen level?",
  "Please play some relaxing music and tell me my medication schedule.",
  "The quick brown fox jumps over the lazy dog."
];

export default function STTBenchmarkPage() {
  const [currentPhraseIdx, setCurrentPhraseIdx] = useState(0);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event: any) => {
        let currentTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            currentTranscript += event.results[i][0].transcript;
            
            // Calculate WER when final result comes in
            const finalTranscript = currentTranscript.trim();
            setTranscript(finalTranscript);
            
            const expected = TEST_PHRASES[currentPhraseIdx];
            const { distance, wer, refWordCount } = calculateWER(expected, finalTranscript);
            
            setResults(prev => [...prev, {
              phrase: expected,
              transcript: finalTranscript,
              wer: wer,
              errors: distance,
              words: refWordCount
            }]);
            
            setIsListening(false);
            
            if (currentPhraseIdx < TEST_PHRASES.length - 1) {
                setTimeout(() => setCurrentPhraseIdx(p => p + 1), 1500);
            }
          } else {
            setTranscript(currentTranscript + event.results[i][0].transcript);
          }
        }
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error("Speech recognition error", event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, [currentPhraseIdx]);

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      setTranscript("");
      recognitionRef.current?.start();
      setIsListening(true);
    }
  };

  const avgWer = results.length > 0 ? results.reduce((acc, r) => acc + r.wer, 0) / results.length : 0;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8 font-sans">
      <div className="max-w-4xl mx-auto space-y-8">
        
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">STT Benchmarking Suite</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-2">Measure Word Error Rate (WER) using the Web Speech API</p>
          </div>
          <Link href="/admin" className="text-purple-600 hover:text-purple-700 font-medium">
            &larr; Back to Admin
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="col-span-2 bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Live Test</h2>
            
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 border border-purple-100 dark:border-purple-800/30 text-center mb-6">
              <p className="text-sm font-semibold text-purple-600 dark:text-purple-400 uppercase tracking-wider mb-2">Please Read Aloud:</p>
              <p className="text-2xl font-medium text-gray-900 dark:text-white">
                "{TEST_PHRASES[currentPhraseIdx]}"
              </p>
            </div>

            <div className="flex flex-col items-center gap-4">
              <button
                onClick={toggleListening}
                className={cn(
                  "w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 shadow-lg",
                  isListening ? "bg-red-500 hover:bg-red-600 animate-pulse" : "bg-purple-600 hover:bg-purple-700"
                )}
              >
                {isListening ? <MicOff className="w-8 h-8 text-white" /> : <Mic className="w-8 h-8 text-white" />}
              </button>
              
              <div className="w-full mt-4 min-h-[100px] p-4 bg-gray-50 dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700">
                <span className="text-sm text-gray-500 dark:text-gray-400 uppercase font-semibold">Live Transcript:</span>
                <p className="mt-2 text-lg text-gray-800 dark:text-gray-200">{transcript || "Waiting for speech..."}</p>
              </div>
            </div>
          </div>

          <div className="col-span-1 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <div className="flex items-center gap-3 mb-2">
                <BarChart2 className="w-6 h-6 text-purple-600" />
                <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Average WER</h3>
              </div>
              <p className="text-4xl font-bold text-gray-900 dark:text-white">{(avgWer * 100).toFixed(1)}%</p>
              <p className="text-sm text-gray-500 mt-1">Lower is better. &lt;10% is excellent.</p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
              <div className="flex items-center gap-3 mb-2">
                <Activity className="w-6 h-6 text-green-500" />
                <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Progress</h3>
              </div>
              <p className="text-4xl font-bold text-gray-900 dark:text-white">{results.length} / {TEST_PHRASES.length}</p>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 mt-4">
                <div className="bg-green-500 h-2.5 rounded-full" style={{ width: `${(results.length / TEST_PHRASES.length) * 100}%` }}></div>
              </div>
            </div>
          </div>
        </div>

        {results.length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 overflow-x-auto">
            <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Detailed Results</h2>
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700 text-sm text-gray-500 dark:text-gray-400">
                  <th className="pb-3">Reference Phrase</th>
                  <th className="pb-3">Transcript</th>
                  <th className="pb-3">Errors</th>
                  <th className="pb-3">WER</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {results.map((r, i) => (
                  <tr key={i} className="text-sm text-gray-800 dark:text-gray-200">
                    <td className="py-3 pr-4 max-w-[200px] truncate">{r.phrase}</td>
                    <td className="py-3 pr-4 text-gray-500">{r.transcript}</td>
                    <td className="py-3 pr-4 font-mono">{r.errors} / {r.words}</td>
                    <td className="py-3 font-semibold text-purple-600">{(r.wer * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </div>
  );
}
