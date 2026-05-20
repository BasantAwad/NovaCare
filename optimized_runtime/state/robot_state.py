"""
Robot State Manager — Unified state for all robot operations

Provides a single source of truth for:
- Robot emotions and animations
- Hardware states (battery, sensors)
- Communication states (listening, speaking, thinking)
- Active services and modes
- User context
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod


class EmotionType(Enum):
    """Robot emotion states"""
    IDLE = "idle"
    HAPPY = "happy"
    SAD = "sad"
    CONFUSED = "confused"
    THINKING = "thinking"
    LISTENING = "listening"
    SPEAKING = "speaking"
    SINGING = "singing"
    SLEEPING = "sleeping"
    ALERT = "alert"
    CONCERNED = "concerned"


class RobotMode(Enum):
    """Robot operation modes"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    AUTONOMOUS = "autonomous"
    SLEEP = "sleep"
    ERROR = "error"


@dataclass
class AudioState:
    """Audio pipeline state"""
    is_listening: bool = False
    is_speaking: bool = False
    speaking_amplitude: float = 0.0  # 0.0-1.0
    microphone_active: bool = False
    speaker_volume: float = 0.7  # 0.0-1.0


@dataclass
class HardwareState:
    """Hardware sensor states"""
    battery_level: float = 100.0  # 0-100%
    battery_charging: bool = False
    cpu_temperature: float = 0.0
    cpu_load: float = 0.0  # 0-100%
    memory_available: float = 0.0  # MB
    lidar_connected: bool = False
    camera_connected: bool = False
    microphone_connected: bool = False
    speaker_connected: bool = False
    wifi_strength: int = 0  # -100 to 0 dBm


@dataclass
class ServiceState:
    """State of integrated services"""
    llm_service_available: bool = False
    asl_service_available: bool = False
    tts_service_available: bool = False
    emotion_model_loaded: bool = False
    wake_word_detector_loaded: bool = False
    last_llm_response_time: float = 0.0
    last_asl_detection_time: float = 0.0


@dataclass
class UserContext:
    """Information about current user"""
    current_user_id: Optional[str] = None
    current_user_name: Optional[str] = None
    user_emotion_detected: Optional[str] = None
    last_interaction_time: Optional[datetime] = None
    conversation_history: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnimationState:
    """Animation-related state"""
    eye_open: bool = True
    is_blinking: bool = False
    blink_rate: float = 5.0  # blinks per minute
    eye_position: str = "center"  # center, left, right, up, down
    mouth_position: str = "closed"  # closed, smile, frown, open
    animation_queue: list = field(default_factory=list)


class StateObserver(ABC):
    """Observer pattern for state changes"""
    
    @abstractmethod
    async def on_state_change(self, state_type: str, old_value: Any, new_value: Any):
        """Called when state changes"""
        pass


class RobotState:
    """
    Centralized Robot State Manager
    
    Single source of truth for all robot state information.
    Thread-safe and async-friendly.
    Notifies all observers of state changes.
    """
    
    def __init__(self):
        # Core state containers
        self._emotion: EmotionType = EmotionType.IDLE
        self._mode: RobotMode = RobotMode.IDLE
        self._audio = AudioState()
        self._hardware = HardwareState()
        self._services = ServiceState()
        self._user = UserContext()
        self._animation = AnimationState()
        
        # Metadata
        self._last_update: datetime = datetime.now()
        self._update_count: int = 0
        
        # Observer pattern
        self._observers: list[StateObserver] = []
        self._lock = asyncio.Lock()
        
        # Event emitter
        self._state_change_callbacks: Dict[str, list] = {}
    
    # ========== EMOTION STATE ==========
    
    @property
    def emotion(self) -> EmotionType:
        """Get current emotion"""
        return self._emotion
    
    async def set_emotion(self, emotion: EmotionType):
        """Set emotion with notification"""
        async with self._lock:
            if emotion != self._emotion:
                old = self._emotion
                self._emotion = emotion
                await self._notify_change("emotion", old.value, emotion.value)
    
    # ========== MODE STATE ==========
    
    @property
    def mode(self) -> RobotMode:
        """Get current operating mode"""
        return self._mode
    
    async def set_mode(self, mode: RobotMode):
        """Set operating mode with notification"""
        async with self._lock:
            if mode != self._mode:
                old = self._mode
                self._mode = mode
                await self._notify_change("mode", old.value, mode.value)
    
    # ========== AUDIO STATE ==========
    
    @property
    def audio(self) -> AudioState:
        """Get audio state"""
        return self._audio
    
    async def set_listening(self, is_listening: bool):
        """Set listening state"""
        async with self._lock:
            if self._audio.is_listening != is_listening:
                old = self._audio.is_listening
                self._audio.is_listening = is_listening
                await self._notify_change("audio.listening", old, is_listening)
    
    async def set_speaking(self, is_speaking: bool, amplitude: float = 0.0):
        """Set speaking state with optional amplitude"""
        async with self._lock:
            if self._audio.is_speaking != is_speaking:
                old = self._audio.is_speaking
                self._audio.is_speaking = is_speaking
                self._audio.speaking_amplitude = amplitude
                await self._notify_change("audio.speaking", old, is_speaking)
    
    async def update_speaker_amplitude(self, amplitude: float):
        """Update speaker amplitude during speech"""
        async with self._lock:
            self._audio.speaking_amplitude = max(0.0, min(1.0, amplitude))
            await self._notify_change("audio.amplitude", None, amplitude)
    
    # ========== HARDWARE STATE ==========
    
    @property
    def hardware(self) -> HardwareState:
        """Get hardware state"""
        return self._hardware
    
    async def update_battery(self, level: float, charging: bool = False):
        """Update battery state"""
        async with self._lock:
            self._hardware.battery_level = max(0.0, min(100.0, level))
            self._hardware.battery_charging = charging
            await self._notify_change("hardware.battery", None, level)
    
    async def update_cpu(self, temperature: float, load: float, memory: float):
        """Update CPU metrics"""
        async with self._lock:
            self._hardware.cpu_temperature = temperature
            self._hardware.cpu_load = load
            self._hardware.memory_available = memory
            await self._notify_change("hardware.cpu", None, {"temp": temperature, "load": load})
    
    async def set_hardware_connected(self, device_name: str, connected: bool):
        """Set hardware device connection status"""
        async with self._lock:
            if hasattr(self._hardware, f"{device_name}_connected"):
                old = getattr(self._hardware, f"{device_name}_connected")
                setattr(self._hardware, f"{device_name}_connected", connected)
                await self._notify_change(f"hardware.{device_name}", old, connected)
    
    # ========== SERVICE STATE ==========
    
    @property
    def services(self) -> ServiceState:
        """Get service availability state"""
        return self._services
    
    async def set_service_available(self, service_name: str, available: bool):
        """Set service availability"""
        async with self._lock:
            if hasattr(self._services, f"{service_name}_available"):
                old = getattr(self._services, f"{service_name}_available")
                setattr(self._services, f"{service_name}_available", available)
                await self._notify_change(f"service.{service_name}", old, available)
    
    # ========== USER CONTEXT ==========
    
    @property
    def user(self) -> UserContext:
        """Get user context"""
        return self._user
    
    async def set_current_user(self, user_id: str, user_name: str):
        """Set current user"""
        async with self._lock:
            self._user.current_user_id = user_id
            self._user.current_user_name = user_name
            self._user.last_interaction_time = datetime.now()
            await self._notify_change("user.current", None, user_name)
    
    async def set_user_emotion(self, emotion: str):
        """Set detected user emotion"""
        async with self._lock:
            old = self._user.user_emotion_detected
            self._user.user_emotion_detected = emotion
            await self._notify_change("user.emotion", old, emotion)
    
    # ========== ANIMATION STATE ==========
    
    @property
    def animation(self) -> AnimationState:
        """Get animation state"""
        return self._animation
    
    async def set_eye_state(self, open: bool, position: str = "center"):
        """Set eye animation state"""
        async with self._lock:
            self._animation.eye_open = open
            self._animation.eye_position = position
            await self._notify_change("animation.eye", None, {"open": open, "pos": position})
    
    async def queue_animation(self, animation_name: str, duration: float):
        """Queue an animation to play"""
        async with self._lock:
            self._animation.animation_queue.append({
                "name": animation_name,
                "duration": duration,
                "timestamp": datetime.now()
            })
            await self._notify_change("animation.queued", None, animation_name)
    
    # ========== STATE METADATA ==========
    
    @property
    def last_update(self) -> datetime:
        """Get last state update time"""
        return self._last_update
    
    @property
    def update_count(self) -> int:
        """Get total update count"""
        return self._update_count
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete robot state as dictionary"""
        return {
            "emotion": self._emotion.value,
            "mode": self._mode.value,
            "audio": {
                "listening": self._audio.is_listening,
                "speaking": self._audio.is_speaking,
                "amplitude": self._audio.speaking_amplitude,
                "volume": self._audio.speaker_volume,
            },
            "hardware": {
                "battery_level": self._hardware.battery_level,
                "battery_charging": self._hardware.battery_charging,
                "cpu_temperature": self._hardware.cpu_temperature,
                "cpu_load": self._hardware.cpu_load,
                "memory_available": self._hardware.memory_available,
                "devices": {
                    "lidar": self._hardware.lidar_connected,
                    "camera": self._hardware.camera_connected,
                    "microphone": self._hardware.microphone_connected,
                    "speaker": self._hardware.speaker_connected,
                }
            },
            "services": {
                "llm": self._services.llm_service_available,
                "asl": self._services.asl_service_available,
                "tts": self._services.tts_service_available,
                "emotion_model": self._services.emotion_model_loaded,
            },
            "user": {
                "id": self._user.current_user_id,
                "name": self._user.current_user_name,
                "emotion": self._user.user_emotion_detected,
            },
            "animation": {
                "eyes_open": self._animation.eye_open,
                "eye_position": self._animation.eye_position,
                "animation_queue_length": len(self._animation.animation_queue),
            },
            "metadata": {
                "last_update": self._last_update.isoformat(),
                "update_count": self._update_count,
            }
        }
    
    # ========== OBSERVER PATTERN ==========
    
    def add_observer(self, observer: StateObserver):
        """Add state change observer"""
        self._observers.append(observer)
    
    def remove_observer(self, observer: StateObserver):
        """Remove state change observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def on_state_change(self, state_type: str, callback):
        """Register callback for specific state change"""
        if state_type not in self._state_change_callbacks:
            self._state_change_callbacks[state_type] = []
        self._state_change_callbacks[state_type].append(callback)
    
    async def _notify_change(self, state_type: str, old_value: Any, new_value: Any):
        """Notify all observers of state change"""
        self._last_update = datetime.now()
        self._update_count += 1
        
        # Notify observers
        for observer in self._observers:
            try:
                await observer.on_state_change(state_type, old_value, new_value)
            except Exception as e:
                print(f"Error notifying observer: {e}")
        
        # Call specific callbacks
        if state_type in self._state_change_callbacks:
            for callback in self._state_change_callbacks[state_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(old_value, new_value)
                    else:
                        callback(old_value, new_value)
                except Exception as e:
                    print(f"Error calling state change callback: {e}")


# Global singleton instance
_robot_state: Optional[RobotState] = None


def get_robot_state() -> RobotState:
    """Get or create global robot state instance"""
    global _robot_state
    if _robot_state is None:
        _robot_state = RobotState()
    return _robot_state
