import { useState, useEffect, useRef } from 'react';
import { getDynamicUrl } from './utils';

const NOVABOT_API = process.env.NEXT_PUBLIC_NOVABOT_API_URL || "http://localhost:5000";

interface EmotionState {
  emotion: string;
  confidence: number;
}

export function useBackgroundEmotion(intervalMs: number = 5000) {
  const [emotionState, setEmotionState] = useState<EmotionState>({ emotion: "unknown", confidence: 0.0 });
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isPolling = useRef<boolean>(false);

  useEffect(() => {
    // Initialize hidden video and canvas elements for silent frame extraction
    videoRef.current = document.createElement("video");
    videoRef.current.autoplay = true;
    videoRef.current.playsInline = true;
    
    canvasRef.current = document.createElement("canvas");
    canvasRef.current.width = 224;
    canvasRef.current.height = 224;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        isPolling.current = true;
      } catch (err) {
        console.warn("Background emotion poller failed to access camera:", err);
      }
    };

    startCamera();

    const pollEmotion = async () => {
      if (!isPolling.current || !videoRef.current || !canvasRef.current) return;
      
      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;

      // Ensure video is playing and has frames
      if (videoRef.current.readyState < 2) return;

      ctx.drawImage(videoRef.current, 0, 0, 224, 224);
      const base64Image = canvasRef.current.toDataURL("image/jpeg", 0.8).split(",")[1];

      try {
        const res = await fetch(`${getDynamicUrl(NOVABOT_API)}/api/emotion/detect`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: base64Image }),
          signal: AbortSignal.timeout(3000)
        });

        if (res.ok) {
          const data = await res.json();
          if (data.status === "success") {
            setEmotionState({ emotion: data.emotion, confidence: data.confidence });
          }
        }
      } catch (err) {
        // Silently ignore network errors to prevent console spam
      }
    };

    const intervalId = setInterval(pollEmotion, intervalMs);

    return () => {
      clearInterval(intervalId);
      isPolling.current = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [intervalMs]);

  return emotionState;
}
