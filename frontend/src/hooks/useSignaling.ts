"use client";

import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

const SIGNALING_SERVER = process.env.NEXT_PUBLIC_SIGNALING_SERVER || 'http://localhost:8080';

export interface SignalingEvent {
    type: string;
    payload: any;
}

export const useSignaling = (userId: string, role: 'guardian' | 'rover') => {
    const socketRef = useRef<Socket | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [lastEvent, setLastEvent] = useState<SignalingEvent | null>(null);

    useEffect(() => {
        if (!userId) return;

        const socket = io(SIGNALING_SERVER, {
            transports: ['websocket', 'polling'],
            reconnectionAttempts: 5,
        });

        socketRef.current = socket;

        socket.on('connect', () => {
            console.log('Connected to signaling server');
            setIsConnected(true);
            socket.emit('register', { id: userId, type: role });
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from signaling server');
            setIsConnected(false);
        });

        socket.on('EMERGENCY_TRIGGERED', (data) => {
            setLastEvent({ type: 'EMERGENCY_TRIGGERED', payload: data });
        });

        socket.on('NAVIGATE_TO_PATIENT', (data) => {
            setLastEvent({ type: 'NAVIGATE_TO_PATIENT', payload: data });
        });

        return () => {
            socket.disconnect();
        };
    }, [userId, role]);

    const emitEvent = useCallback((eventName: string, data: any) => {
        if (socketRef.current && isConnected) {
            socketRef.current.emit(eventName, data);
        } else {
            console.warn('Socket not connected. Cannot emit event:', eventName);
        }
    }, [isConnected]);

    const navigateToPatient = useCallback((coords?: { x: number, y: number }) => {
        emitEvent('NAVIGATE_TO_PATIENT', coords || { x: -4.5, y: 1.2 });
    }, [emitEvent]);

    return {
        isConnected,
        lastEvent,
        emitEvent,
        navigateToPatient,
    };
};
