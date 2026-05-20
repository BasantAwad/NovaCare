// Robot UI - Main React Component
// Animated eyes and emotion display for SERBot

import React, { useState, useEffect, useRef } from 'react';
import './RobotUI.css';

interface RobotState {
  emotion: string;
  speaking: boolean;
  listening: boolean;
  audio_level: number;
  battery_level: number;
}

const RobotUI: React.FC = () => {
  const [robotState, setRobotState] = useState<RobotState>({
    emotion: 'idle',
    speaking: false,
    listening: false,
    audio_level: 0,
    battery_level: 100,
  });

  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket connection
  useEffect(() => {
    const connect = () => {
      try {
        wsRef.current = new WebSocket('ws://localhost:9999');

        wsRef.current.onopen = () => {
          console.log('Connected to robot runtime');
          setIsConnected(true);
        };

        wsRef.current.onmessage = (event) => {
          const message = JSON.parse(event.data);

          if (message.type === 'state_update') {
            const { emotion, audio, hardware } = message.data;
            setRobotState({
              emotion: emotion || 'idle',
              speaking: audio?.speaking || false,
              listening: audio?.listening || false,
              audio_level: audio?.amplitude || 0,
              battery_level: hardware?.battery_level || 100,
            });
          }
        };

        wsRef.current.onclose = () => {
          setIsConnected(false);
          setTimeout(connect, 1000);
        };
      } catch (error) {
        console.error('WebSocket error:', error);
      }
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const getEmotionClass = (emotion: string): string => {
    const emotionMap: { [key: string]: string } = {
      happy: 'emotion-happy',
      sad: 'emotion-sad',
      confused: 'emotion-confused',
      thinking: 'emotion-thinking',
      listening: 'emotion-listening',
      speaking: 'emotion-speaking',
      idle: 'emotion-idle',
    };
    return emotionMap[emotion] || 'emotion-idle';
  };

  return (
    <div className={`robot-ui ${getEmotionClass(robotState.emotion)}`}>
      {/* Thinking Indicator */}
      {robotState.emotion === 'thinking' && (
        <div className="thinking-container">
          <div className="thought-cloud">?</div>
          <div className="thought-dot dot-1"></div>
          <div className="thought-dot dot-2"></div>
        </div>
      )}

      {/* Decorative Bow */}
      <div className="bow-container">
        <div className="bow-tie">
          <div className="bow-left"></div>
          <div className="bow-center"></div>
          <div className="bow-right"></div>
        </div>
      </div>

      {/* Eyes Container */}
      <div className="eyes-container">
        <div className="eye eye-left">
          <div className="eyebrow"></div>
          <div className={`pupil ${robotState.listening ? 'listening' : ''}`}></div>
        </div>
        <div className="eye eye-right">
          <div className="eyebrow"></div>
          <div className={`pupil ${robotState.listening ? 'listening' : ''}`}></div>
        </div>
      </div>

      {/* Mouth Element */}
      <div className={`mouth ${robotState.speaking ? 'speaking' : ''}`}>
        <div className="mouth-teeth"></div>
        <div className="mouth-tongue"></div>
      </div>



      {/* Status Indicators */}
      <div className="status-bar">
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? '●' : '○'} Connected
        </div>
        <div className="battery-indicator">
          🔋 {robotState.battery_level.toFixed(0)}%
        </div>
      </div>

      {/* Emotion Text */}
      <div className="emotion-display">{robotState.emotion.toUpperCase()}</div>
    </div>
  );
};

export default RobotUI;
