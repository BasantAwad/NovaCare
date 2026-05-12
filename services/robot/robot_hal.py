"""
NovaCare Robot — Hardware Abstraction Layer (HAL)
==================================================
Clean abstraction over the SERBot Prime X ``pop`` library.

All robot-specific hardware calls are encapsulated here so that:
  - Backend services import only this module, never ``pop`` directly.
  - On dev laptops (where ``pop`` is unavailable) every method degrades
    gracefully with a mock/fallback implementation.
  - Thread-safety is managed centrally.

Subsystems
----------
- CameraHAL:   CSI/USB camera via GStreamer (pop.Util)
- MotionHAL:   Omni-wheel control via pop.Pilot.SerBot
- AudioHAL:    Speaker + Microphone via pop.AudioPlay, gTTS, SpeechRecognition
- LidarHAL:    RPLiDAR A1 via pop.LiDAR.Rplidar
"""

import os
import sys
import time
import threading
import tempfile
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_INDEX,
    DEFAULT_SPEED, MAX_SPEED, OBSTACLE_STOP_DISTANCE_MM,
    TTS_LANG, TTS_TEMP_DIR, STT_LANG, STT_TIMEOUT, STT_PHRASE_TIMEOUT,
    LIDAR_ENABLED,
)

# ---------------------------------------------------------------------------
# Try importing the pop library (only available on the SERBot SBC)
# ---------------------------------------------------------------------------
_POP_AVAILABLE = False
_SerBot = None
_Rplidar = None
_AudioPlay = None
_pop_Util = None

try:
    from pop.Pilot import SerBot as _SerBotClass
    from pop.LiDAR import Rplidar as _RplidarClass
    from pop import Util as _pop_Util_module
    _POP_AVAILABLE = True
    _SerBot = _SerBotClass
    _Rplidar = _RplidarClass
    _pop_Util = _pop_Util_module
    print("✓ pop library loaded — running on SERBot hardware")
except ImportError:
    print("⚠ pop library not available — running in MOCK/DEV mode")

try:
    from pop import AudioPlay as _AudioPlayClass
    _AudioPlay = _AudioPlayClass
except ImportError:
    pass

# Optional imports for TTS/STT
try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False
    print("⚠ gTTS not installed (pip install gTTS)")

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False
    print("⚠ SpeechRecognition not installed (pip install SpeechRecognition)")


# ============================================================================
# Camera HAL
# ============================================================================
class CameraHAL:
    """
    Manages the robot camera.

    On SERBot: uses ``pop.Util.gstrmer()`` for a GStreamer pipeline optimised
    for the on-board CSI/USB camera.

    On dev machines: falls back to ``cv2.VideoCapture(CAMERA_INDEX)``.
    """

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._open()

    def _open(self):
        if _pop_Util is not None:
            try:
                _pop_Util.enable_imshow()
            except Exception:
                pass
            try:
                pipeline = _pop_Util.gstrmer(CAMERA_WIDTH, CAMERA_HEIGHT)
                self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self._cap.isOpened():
                    print(f"✓ Camera opened via GStreamer pipeline ({CAMERA_WIDTH}×{CAMERA_HEIGHT})")
                    return
                else:
                    print("⚠ GStreamer pipeline failed — falling back to default camera")
            except Exception as e:
                print(f"⚠ GStreamer error: {e} — falling back to default camera")

        # Fallback: standard OpenCV VideoCapture
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            print(f"✓ Camera opened via VideoCapture({CAMERA_INDEX})")
        else:
            print("✗ No camera available")
            self._cap = None

    @property
    def is_available(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single BGR frame. Thread-safe."""
        with self._lock:
            if not self.is_available:
                return False, None
            ret, frame = self._cap.read()
            return ret, frame

    def read_frame_base64(self, quality: int = 80) -> Optional[str]:
        """Read a frame and return as base64-encoded JPEG string."""
        import base64
        ret, frame = self.read_frame()
        if not ret or frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    def read_frame_jpeg_bytes(self, quality: int = 80) -> Optional[bytes]:
        """Read a frame and return raw JPEG bytes (for MJPEG streaming)."""
        ret, frame = self.read_frame()
        if not ret or frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def release(self):
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None


# ============================================================================
# Motion HAL
# ============================================================================
class MotionHAL:
    """
    Controls the SERBot's 3-axis omni-wheel drive.

    On SERBot: uses ``pop.Pilot.SerBot()``.
    On dev machines: prints movement commands to console (mock mode).
    """

    def __init__(self):
        self._bot = None
        self._lock = threading.Lock()
        self._speed = DEFAULT_SPEED
        self._moving = False

        if _SerBot is not None:
            try:
                self._bot = _SerBot()
                print("✓ SerBot motor controller initialized")
            except Exception as e:
                print(f"✗ SerBot motor init failed: {e}")
        else:
            print("⚠ MotionHAL in MOCK mode (no pop.Pilot)")

    @property
    def is_available(self) -> bool:
        return self._bot is not None

    @property
    def is_moving(self) -> bool:
        return self._moving

    def set_speed(self, speed: int):
        self._speed = max(0, min(speed, MAX_SPEED))

    def move(self, angle: int, speed: Optional[int] = None):
        """Move at *angle* (0=forward, 90=right, …) at given *speed*."""
        s = speed if speed is not None else self._speed
        s = max(0, min(s, MAX_SPEED))
        with self._lock:
            self._moving = True
            if self._bot:
                self._bot.move(angle, s)
            else:
                print(f"[MOCK] move(angle={angle}, speed={s})")

    def forward(self, speed: Optional[int] = None):
        self.move(0, speed)

    def backward(self, speed: Optional[int] = None):
        self.move(180, speed)

    def left(self, speed: Optional[int] = None):
        self.move(270, speed)

    def right(self, speed: Optional[int] = None):
        self.move(90, speed)

    def turn_left(self, speed: Optional[int] = None):
        s = speed if speed is not None else self._speed
        with self._lock:
            self._moving = True
            if self._bot:
                self._bot.turnLeft()
            else:
                print(f"[MOCK] turnLeft(speed={s})")

    def turn_right(self, speed: Optional[int] = None):
        s = speed if speed is not None else self._speed
        with self._lock:
            self._moving = True
            if self._bot:
                self._bot.turnRight()
            else:
                print(f"[MOCK] turnRight(speed={s})")

    def stop(self):
        with self._lock:
            self._moving = False
            if self._bot:
                self._bot.stop()
            else:
                print("[MOCK] stop()")

    def move_for(self, angle: int, duration_s: float, speed: Optional[int] = None):
        """Move in a direction for a fixed duration, then stop."""
        self.move(angle, speed)
        time.sleep(duration_s)
        self.stop()

    def start_tracking(self, target: str = "face"):
        """Start built-in pop.Pilot SerBot tracking (e.g. face/color)."""
        with self._lock:
            self._moving = True
            if self._bot and hasattr(self._bot, "tracking"):
                self._bot.tracking(target)
                print(f"✓ SerBot built-in tracking started for: {target}")
            else:
                print(f"[MOCK] start_tracking(target={target})")

    def stop_tracking(self):
        """Stop built-in pop.Pilot SerBot tracking."""
        with self._lock:
            if self._bot and hasattr(self._bot, "stopTracking"):
                self._bot.stopTracking()
                print("✓ SerBot built-in tracking stopped")
            elif self._bot and hasattr(self._bot, "tracking"):
                # some pop versions stop tracking by passing None or empty
                try:
                    self._bot.tracking("")
                except:
                    pass
            self.stop()

    def navigate_to(self, location_name: str):
        """Start built-in SerBot room navigation / SLAM mapping."""
        with self._lock:
            self._moving = True
            if self._bot and hasattr(self._bot, "navigation"):
                self._bot.navigation(location_name)
                print(f"✓ SerBot navigating to: {location_name}")
            else:
                print(f"[MOCK] navigate_to(location={location_name})")


# ============================================================================
# Audio HAL
# ============================================================================
class AudioHAL:
    """
    Manages the robot's speaker and microphone.

    TTS: gTTS → save to temp WAV/MP3 → play via pop.AudioPlay (or fallback).
    STT: SpeechRecognition library with Google backend.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._audio_player = None
        os.makedirs(TTS_TEMP_DIR, exist_ok=True)

        if _AudioPlay is not None:
            try:
                self._audio_player = _AudioPlay()
                print("✓ AudioPlay initialized (robot speaker)")
            except Exception as e:
                print(f"⚠ AudioPlay init failed: {e}")

    @property
    def tts_available(self) -> bool:
        return _GTTS_AVAILABLE

    @property
    def stt_available(self) -> bool:
        return _SR_AVAILABLE

    def speak(self, text: str, lang: str = None, block: bool = True) -> bool:
        """
        Convert text to speech and play on the robot speaker.

        Falls back to os-level playback commands if pop.AudioPlay is unavailable.
        Returns True if audio was played (or queued).
        """
        if not text.strip():
            return False
        if not _GTTS_AVAILABLE:
            print(f"[TTS-MOCK] {text}")
            return False

        lang = lang or TTS_LANG
        try:
            tts = gTTS(text=text, lang=lang)
            filepath = os.path.join(TTS_TEMP_DIR, f"tts_{int(time.time()*1000)}.mp3")
            tts.save(filepath)

            if block:
                self._play_file(filepath)
            else:
                t = threading.Thread(target=self._play_file, args=(filepath,), daemon=True)
                t.start()
            return True
        except Exception as e:
            print(f"[TTS Error] {e}")
            return False

    def _play_file(self, filepath: str):
        """Play an audio file through the best available backend."""
        with self._lock:
            try:
                if self._audio_player:
                    self._audio_player.play(filepath)
                elif sys.platform == "linux":
                    os.system(f"mpg123 -q '{filepath}' 2>/dev/null || aplay '{filepath}' 2>/dev/null")
                elif sys.platform == "darwin":
                    os.system(f"afplay '{filepath}'")
                else:
                    # Windows or other — use pygame or system player
                    try:
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(filepath)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    except ImportError:
                        os.system(f'start /min "" "{filepath}"')
            finally:
                # Clean up temp file
                try:
                    os.remove(filepath)
                except OSError:
                    pass

    def listen(self, timeout: int = None, phrase_timeout: int = None) -> Optional[str]:
        """
        Listen for speech and return recognised text.
        Returns None if nothing was recognised or STT is unavailable.
        """
        if not _SR_AVAILABLE:
            print("[STT-MOCK] listen() called but SpeechRecognition not available")
            return None

        timeout = timeout or STT_TIMEOUT
        phrase_timeout = phrase_timeout or STT_PHRASE_TIMEOUT

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("[STT] Listening...")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)
                print("[STT] Processing...")
                text = recognizer.recognize_google(audio, language=STT_LANG)
                print(f"[STT] Recognised: {text}")
                return text
        except sr.WaitTimeoutError:
            print("[STT] Timeout — no speech detected")
            return None
        except sr.UnknownValueError:
            print("[STT] Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"[STT] Google API error: {e}")
            return None
        except Exception as e:
            print(f"[STT] Error: {e}")
            return None


# ============================================================================
# LiDAR HAL
# ============================================================================
class LidarHAL:
    """
    Interfaces with the RPLiDAR A1 M8 via ``pop.LiDAR.Rplidar``.

    Provides raw scan vectors and a convenience ``is_obstacle_ahead()`` check.
    """

    def __init__(self):
        self._lidar = None
        self._lock = threading.Lock()

        if LIDAR_ENABLED and _Rplidar is not None:
            try:
                self._lidar = _Rplidar()
                self._lidar.connect()
                self._lidar.startMotor()
                print("✓ RPLiDAR connected and motor started")
            except Exception as e:
                print(f"⚠ LiDAR init failed: {e}")
                self._lidar = None
        else:
            print("⚠ LidarHAL disabled or pop.LiDAR not available")

    @property
    def is_available(self) -> bool:
        return self._lidar is not None

    def get_scan(self) -> List[Dict]:
        """
        Return a list of scan points: [{"angle": float, "distance_mm": float}, …].
        Returns an empty list if LiDAR is unavailable.
        """
        if self._lidar is None:
            return []
        with self._lock:
            try:
                vectors = self._lidar.getVectors()
                return [
                    {"angle": v[0], "distance_mm": v[1]}
                    for v in vectors
                ]
            except Exception as e:
                print(f"[LiDAR] scan error: {e}")
                return []

    def is_obstacle_ahead(self, cone_degrees: int = 30,
                          distance_mm: int = OBSTACLE_STOP_DISTANCE_MM) -> bool:
        """Check if there's an obstacle within *cone_degrees* of forward direction."""
        scan = self.get_scan()
        if not scan:
            return False
        half_cone = cone_degrees / 2
        for point in scan:
            angle = point["angle"]
            # Forward is angle ≈ 0 (or ≈ 360)
            if (angle <= half_cone or angle >= 360 - half_cone):
                if 0 < point["distance_mm"] < distance_mm:
                    return True
        return False

    def shutdown(self):
        if self._lidar:
            try:
                self._lidar.stopMotor()
            except Exception:
                pass


# ============================================================================
# Unified Robot HAL (singleton facade)
# ============================================================================
class RobotHAL:
    """
    Unified facade for all SERBot hardware subsystems.

    Usage::

        from robot_hal import get_robot
        robot = get_robot()
        robot.motion.forward()
        frame = robot.camera.read_frame()
        robot.audio.speak("Hello!")
        robot.motion.stop()
    """

    def __init__(self):
        print("=" * 50)
        print("  NovaCare — SERBot Hardware Abstraction Layer")
        print("=" * 50)
        self.camera = CameraHAL()
        self.motion = MotionHAL()
        self.audio = AudioHAL()
        self.lidar = LidarHAL()
        self._print_status()

    def _print_status(self):
        print("\n  Hardware Status:")
        print(f"    Camera:   {'✓ READY' if self.camera.is_available else '✗ N/A'}")
        print(f"    Motion:   {'✓ READY' if self.motion.is_available else '⚠ MOCK'}")
        print(f"    TTS:      {'✓ READY' if self.audio.tts_available else '✗ N/A'}")
        print(f"    STT:      {'✓ READY' if self.audio.stt_available else '✗ N/A'}")
        print(f"    LiDAR:    {'✓ READY' if self.lidar.is_available else '✗ N/A'}")
        print(f"    pop lib:  {'✓ LOADED' if _POP_AVAILABLE else '⚠ NOT AVAILABLE'}")
        print("=" * 50)

    def shutdown(self):
        """Gracefully release all hardware resources."""
        print("[HAL] Shutting down all hardware...")
        self.motion.stop()
        self.camera.release()
        self.lidar.shutdown()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_robot_instance: Optional[RobotHAL] = None


def get_robot() -> RobotHAL:
    """Get or create the singleton RobotHAL instance."""
    global _robot_instance
    if _robot_instance is None:
        _robot_instance = RobotHAL()
    return _robot_instance
