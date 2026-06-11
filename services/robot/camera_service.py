"""
NovaCare Robot — Lightweight Camera Service
=============================================
Direct Linux V4L2 camera access WITHOUT OpenCV.

Uses ``v4l2-ctl`` (from v4l-utils) for frame capture and PIL/Pillow
for JPEG re-encoding / resizing.  Falls back to raw device reads when
``v4l2-ctl`` is not installed.

Requirements:
    - Linux with V4L2 camera driver (``/dev/video*``)
    - ``v4l-utils`` system package  (``sudo apt install v4l-utils``)
    - ``Pillow`` Python package      (``pip install Pillow``)

No OpenCV, no NumPy, no TensorFlow, no heavy deps.
"""

import io
import os
import struct
import subprocess
import threading
import time
import base64
from typing import Optional, Tuple

# Pillow — lightweight JPEG encode/decode & resize
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    print("[WARN] Pillow not installed (pip install Pillow) — JPEG resize disabled")


# ============================================================================
# V4L2 constants (from <linux/videodev2.h>) — only what we need
# ============================================================================
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_S_FMT = 0xC0CC5605
VIDIOC_G_FMT = 0xC0CC5604
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0445609
VIDIOC_QBUF = 0xC044560F
VIDIOC_DQBUF = 0xC0445611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_PIX_FMT_MJPEG = 0x47504A4D   # 'MJPG' in little-endian
V4L2_PIX_FMT_YUYV = 0x56595559    # 'YUYV' fallback

# Probe order
CAMERA_DEVICES = ["/dev/video0", "/dev/video1", "/dev/video2"]


class LightweightCamera:
    """
    Zero-dependency (beyond Pillow) camera capture for Linux V4L2 devices.

    Preferred capture pipeline:
        1. ``v4l2-ctl --stream-mmap`` → raw JPEG frame   (fast, reliable)
        2. ``ffmpeg -f v4l2 …``       → JPEG frame        (fallback)
        3. Raw ``/dev/videoN`` read    → MJPEG extraction  (last resort)
    """

    def __init__(self, width: int = 320, height: int = 240,
                 fps: int = 10, jpeg_quality: int = 60):
        self._device: Optional[str] = None
        self._width = width
        self._height = height
        self._fps = fps
        self._jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._active_viewers = 0
        self._streaming = False

        # Detect which capture backend is available
        self._use_v4l2ctl = self._check_command("v4l2-ctl")
        self._use_ffmpeg = self._check_command("ffmpeg")

        # Auto-detect camera on init
        self._device = self.detect_camera()
        if self._device:
            print(f"[OK] LightweightCamera: detected camera at {self._device}")
        else:
            print("[WARN] LightweightCamera: no camera detected at /dev/video0-2")

    # ------------------------------------------------------------------
    # Camera detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_camera() -> Optional[str]:
        """Probe /dev/video0 → /dev/video2, return first available device."""
        for dev in CAMERA_DEVICES:
            if os.path.exists(dev):
                # Quick check: can we open it?
                try:
                    fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
                    os.close(fd)
                    print(f"[CAMERA] Device {dev} is accessible")
                    return dev
                except OSError as e:
                    print(f"[CAMERA] Device {dev} exists but cannot open: {e}")
                    continue
        print("[CAMERA] ERROR: No camera found at /dev/video0, /dev/video1, /dev/video2")
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if a camera device was detected."""
        return self._device is not None

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def viewer_count(self) -> int:
        return self._active_viewers

    # ------------------------------------------------------------------
    # Session management (start/stop on demand)
    # ------------------------------------------------------------------

    def start_session(self) -> bool:
        """Register a viewer and mark streaming as active."""
        with self._lock:
            if not self.is_available:
                return False
            self._active_viewers += 1
            self._streaming = True
            print(f"[CAMERA] Session started (viewers: {self._active_viewers})")
            return True

    def stop_session(self) -> None:
        """Unregister a viewer. When zero viewers remain, stop streaming."""
        with self._lock:
            self._active_viewers = max(0, self._active_viewers - 1)
            if self._active_viewers == 0:
                self._streaming = False
            print(f"[CAMERA] Session stopped (viewers: {self._active_viewers})")

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def read_frame_jpeg(self) -> Optional[bytes]:
        """
        Capture a single JPEG frame from the camera.

        Returns raw JPEG bytes or None if capture failed.
        """
        if not self.is_available:
            return None

        with self._lock:
            # Strategy 1: v4l2-ctl (fastest, most reliable)
            if self._use_v4l2ctl:
                frame = self._capture_v4l2ctl()
                if frame:
                    return self._resize_jpeg(frame)

            # Strategy 2: ffmpeg
            if self._use_ffmpeg:
                frame = self._capture_ffmpeg()
                if frame:
                    return self._resize_jpeg(frame)

            # Strategy 3: raw device read (last resort)
            frame = self._capture_raw_device()
            if frame:
                return self._resize_jpeg(frame)

        return None

    def read_frame_base64(self, quality: int = 80) -> Optional[str]:
        """Read a frame and return as base64-encoded JPEG string."""
        jpg = self.read_frame_jpeg()
        if jpg is None:
            return None
        # Re-encode at requested quality if different from stream quality
        if _PIL_AVAILABLE and quality != self._jpeg_quality:
            try:
                img = PILImage.open(io.BytesIO(jpg))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                jpg = buf.getvalue()
            except Exception:
                pass  # use original quality
        return base64.b64encode(jpg).decode("utf-8")

    # ------------------------------------------------------------------
    # Capture backends
    # ------------------------------------------------------------------

    def _capture_v4l2ctl(self) -> Optional[bytes]:
        """Capture one JPEG frame via v4l2-ctl --stream-mmap."""
        try:
            # Set MJPEG pixel format first
            subprocess.run(
                [
                    "v4l2-ctl", "-d", self._device,
                    "--set-fmt-video",
                    f"width={self._width},height={self._height},"
                    f"pixelformat=MJPG",
                ],
                capture_output=True, timeout=5,
            )

            # Capture a single frame to stdout
            result = subprocess.run(
                [
                    "v4l2-ctl", "-d", self._device,
                    "--stream-mmap", "--stream-count=1",
                    "--stream-to=-",
                ],
                capture_output=True, timeout=5,
            )

            if result.returncode == 0 and len(result.stdout) > 100:
                # Verify it looks like JPEG (starts with FF D8)
                data = result.stdout
                jpeg_start = data.find(b'\xff\xd8')
                if jpeg_start >= 0:
                    return data[jpeg_start:]
                return data

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[CAMERA] v4l2-ctl capture error: {e}")

        return None

    def _capture_ffmpeg(self) -> Optional[bytes]:
        """Capture one JPEG frame via ffmpeg."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "v4l2",
                    "-input_format", "mjpeg",
                    "-video_size", f"{self._width}x{self._height}",
                    "-framerate", str(self._fps),
                    "-i", self._device,
                    "-frames:v", "1",
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "-q:v", "5",
                    "pipe:1",
                ],
                capture_output=True, timeout=5,
            )

            if result.returncode == 0 and len(result.stdout) > 100:
                return result.stdout

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[CAMERA] ffmpeg capture error: {e}")

        return None

    def _capture_raw_device(self) -> Optional[bytes]:
        """
        Last-resort: open /dev/videoN directly and read raw bytes.

        This works when the camera driver outputs MJPEG natively.
        We look for JPEG SOI (FF D8) and EOI (FF D9) markers.
        """
        try:
            fd = os.open(self._device, os.O_RDONLY | os.O_NONBLOCK)
            try:
                # Try to read up to 1MB of data
                data = b""
                for _ in range(10):
                    try:
                        chunk = os.read(fd, 1024 * 100)
                        data += chunk
                        if len(data) > 1024 * 1024:
                            break
                    except BlockingIOError:
                        time.sleep(0.05)
                        continue

                if data:
                    # Extract first JPEG frame
                    start = data.find(b'\xff\xd8')
                    end = data.find(b'\xff\xd9', start + 2) if start >= 0 else -1
                    if start >= 0 and end >= 0:
                        return data[start:end + 2]
            finally:
                os.close(fd)

        except OSError as e:
            print(f"[CAMERA] Raw device read error: {e}")

        return None

    # ------------------------------------------------------------------
    # JPEG resize helper
    # ------------------------------------------------------------------

    def _resize_jpeg(self, jpeg_bytes: bytes) -> bytes:
        """Resize and re-compress JPEG if Pillow is available."""
        if not _PIL_AVAILABLE:
            return jpeg_bytes

        try:
            img = PILImage.open(io.BytesIO(jpeg_bytes))

            # Only resize if image is larger than target
            if img.width > self._width or img.height > self._height:
                img = img.resize(
                    (self._width, self._height),
                    PILImage.Resampling.LANCZOS,
                )

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._jpeg_quality)
            return buf.getvalue()

        except Exception as e:
            print(f"[CAMERA] JPEG resize error (using original): {e}")
            return jpeg_bytes

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _check_command(cmd: str) -> bool:
        """Check if a shell command is available."""
        try:
            result = subprocess.run(
                ["which", cmd],
                capture_output=True, timeout=3,
            )
            available = result.returncode == 0
            if available:
                print(f"[CAMERA] {cmd} found: {result.stdout.decode().strip()}")
            return available
        except Exception:
            return False

    def release(self) -> None:
        """Release camera resources."""
        with self._lock:
            self._streaming = False
            self._active_viewers = 0
            print("[CAMERA] Camera released")

    def get_status(self) -> dict:
        """Return camera status as a dict."""
        return {
            "available": self.is_available,
            "device": self._device,
            "streaming": self._streaming,
            "viewers": self._active_viewers,
            "resolution": f"{self._width}x{self._height}",
            "fps": self._fps,
            "backend": (
                "v4l2-ctl" if self._use_v4l2ctl
                else "ffmpeg" if self._use_ffmpeg
                else "raw"
            ),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_camera_instance: Optional[LightweightCamera] = None


def get_camera(width: int = 320, height: int = 240,
               fps: int = 10, jpeg_quality: int = 60) -> LightweightCamera:
    """Get or create the singleton LightweightCamera instance."""
    global _camera_instance
    if _camera_instance is None:
        _camera_instance = LightweightCamera(
            width=width, height=height,
            fps=fps, jpeg_quality=jpeg_quality,
        )
    return _camera_instance
