"""
NovaCare Robot — Unified Camera Service
========================================
Pilot-free camera access for the SERBot Prime X.

Default capture (no OpenCV install required):
  1. ``gst-launch-1.0`` / ``ffmpeg`` subprocess  → JPEG  (Jetson CSI + USB)
  2. ``v4l2-ctl --stream-mmap`` → JPEG
  3. Raw ``/dev/videoN`` MJPEG read

Optional (set ``NOVACARE_USE_OPENCV=1`` — slow to install on SerBot):
  4. ``pop.Util.gstrmer()`` + OpenCV ``VideoCapture(CAP_GSTREAMER)``
  5. OpenCV ``VideoCapture`` on ``/dev/video*`` index

Manual-derived: ``pop.Util.gstrmer``, ``nvarguscamerasrc`` pipeline (§4.1.3).
Forbidden: ``pop.Pilot``, ``Pilot.Camera``.
"""

import base64
import io
import os
import subprocess
import threading
import time
from typing import Any, Optional, Tuple

from camera_pipelines import (
    build_gstreamer_pipeline_string,
    check_command,
    detect_v4l2_device,
    run_subprocess_worker,
)

try:
    from PIL import Image as PILImage
    from PIL import ImageFilter

    _PIL_AVAILABLE = True
except ImportError:
    PILImage = None  # type: ignore
    ImageFilter = None  # type: ignore
    _PIL_AVAILABLE = False
    print("[WARN] Pillow not installed (pip install Pillow) — JPEG resize disabled")

# OpenCV is OPTIONAL — avoid on SerBot (multi-hour pip install).
# Enable only with NOVACARE_USE_OPENCV=1 when pre-installed system-wide.
_USE_OPENCV = os.getenv("NOVACARE_USE_OPENCV", "0").lower() in ("1", "true", "yes")

try:
    if _USE_OPENCV:
        import cv2
        import numpy as np
        _CV2_AVAILABLE = True
    else:
        raise ImportError("OpenCV disabled (set NOVACARE_USE_OPENCV=1 to enable)")
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _CV2_AVAILABLE = False
    if _USE_OPENCV:
        print("[WARN] NOVACARE_USE_OPENCV=1 but OpenCV not installed")
    else:
        print("[OK] OpenCV skipped — using ffmpeg/gst-launch/v4l2-ctl + Pillow")


class LightweightCamera:
    """
    Unified camera service: persistent GStreamer/V4L2 capture when streaming,
    one-shot V4L2/ffmpeg fallback otherwise.
    """

    def __init__(
        self,
        capture_width: int = 640,
        capture_height: int = 480,
        capture_fps: int = 30,
        stream_width: int = 320,
        stream_height: int = 240,
        stream_fps: int = 10,
        jpeg_quality: int = 60,
        gstreamer_flip: int = 0,
        camera_index: int = 0,
    ):
        self._capture_width = capture_width
        self._capture_height = capture_height
        self._capture_fps = capture_fps
        self._stream_width = stream_width
        self._stream_height = stream_height
        self._stream_fps = stream_fps
        self._jpeg_quality = jpeg_quality
        self._gstreamer_flip = gstreamer_flip
        self._camera_index = camera_index

        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._active_viewers = 0
        self._streaming = False

        self._device = detect_v4l2_device()
        self._use_v4l2ctl = check_command("v4l2-ctl")
        self._use_ffmpeg = check_command("ffmpeg")
        self._use_gst = check_command("gst-launch-1.0")

        self._backend = "none"
        self._cap: Any = None
        self._latest_frame: Any = None
        self._latest_jpeg: Optional[bytes] = None

        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop = threading.Event()
        self._subprocess_mode = False

        if self.is_available:
            print(
                f"[OK] LightweightCamera ready "
                f"(capture {capture_width}x{capture_height}@{capture_fps}, "
                f"stream {stream_width}x{stream_height})"
            )
            if self._device:
                print(f"[CAMERA] V4L2 device: {self._device}")
        else:
            print("[WARN] LightweightCamera: no camera backend detected")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when any lightweight capture backend is present."""
        return bool(
            self._device
            or self._use_v4l2ctl
            or self._use_ffmpeg
            or self._use_gst
        )

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
    # Session management
    # ------------------------------------------------------------------

    def start_session(self) -> bool:
        with self._lock:
            if not self.is_available:
                return False
            self._active_viewers += 1
            self._streaming = True
            started = self._ensure_stream_thread()
            print(f"[CAMERA] Session started (viewers: {self._active_viewers}, backend: {self._backend})")
            return started or self.is_available

    def stop_session(self) -> None:
        with self._lock:
            self._active_viewers = max(0, self._active_viewers - 1)
            if self._active_viewers == 0:
                self._streaming = False
                self._stop_stream_thread()
            print(f"[CAMERA] Session stopped (viewers: {self._active_viewers})")

    # ------------------------------------------------------------------
    # Frame capture — public API
    # ------------------------------------------------------------------

    def read_frame(self) -> Tuple[bool, Any]:
        """
        Return ``(success, frame)`` for vision consumers.

        Without OpenCV this returns a PIL grayscale image of the latest JPEG.
        With OpenCV enabled, returns a BGR numpy array.
        """
        if not self.is_available:
            return False, None

        self._ensure_stream_thread()

        with self._frame_lock:
            if self._latest_frame is not None:
                return True, self._latest_frame

        jpg = self.read_frame_jpeg()
        if jpg is None:
            return False, None
        frame = self._jpeg_to_frame(jpg)
        if frame is not None:
            with self._frame_lock:
                self._latest_frame = frame
            return True, frame
        return False, None

    def detect_obstacle_ahead(self) -> bool:
        """
        Lightweight front-obstacle heuristic using JPEG + Pillow only.
        No OpenCV required. Used by summon navigation.
        """
        jpg = self.read_frame_jpeg()
        if not jpg:
            return False
        return detect_obstacle_in_jpeg(jpg)

    def read_frame_jpeg(self) -> Optional[bytes]:
        if not self.is_available:
            return None

        self._ensure_stream_thread()

        with self._frame_lock:
            if self._latest_jpeg is not None:
                return self._resize_jpeg(self._latest_jpeg)

        with self._lock:
            jpg = self._capture_jpeg_once_unlocked()
            if jpg:
                return self._resize_jpeg(jpg)
        return None

    def read_frame_base64(self, quality: int = 80) -> Optional[str]:
        jpg = self.read_frame_jpeg()
        if jpg is None:
            return None
        if _PIL_AVAILABLE and quality != self._jpeg_quality:
            try:
                img = PILImage.open(io.BytesIO(jpg))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                jpg = buf.getvalue()
            except Exception:
                pass
        return base64.b64encode(jpg).decode("utf-8")

    def release(self) -> None:
        with self._lock:
            self._streaming = False
            self._active_viewers = 0
            self._stop_stream_thread()
            with self._frame_lock:
                self._latest_frame = None
                self._latest_jpeg = None
            print("[CAMERA] Camera released")

    def get_status(self) -> dict:
        return {
            "available": self.is_available,
            "device": self._device,
            "streaming": self._streaming,
            "viewers": self._active_viewers,
            "capture_resolution": f"{self._capture_width}x{self._capture_height}",
            "stream_resolution": f"{self._stream_width}x{self._stream_height}",
            "capture_fps": self._capture_fps,
            "stream_fps": self._stream_fps,
            "backend": self._backend,
            "opencv_enabled": _CV2_AVAILABLE,
            "opencv_requested": _USE_OPENCV,
        }

    # ------------------------------------------------------------------
    # Background streaming
    # ------------------------------------------------------------------

    def _ensure_stream_thread(self) -> bool:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return True
        return self._start_stream_thread()

    def _start_stream_thread(self) -> bool:
        # Prefer subprocess capture (ffmpeg/gst-launch) — no OpenCV needed.
        if self._use_ffmpeg or self._use_gst:
            self._backend = "subprocess"
            self._subprocess_mode = True
            self._stream_stop.clear()
            self._stream_thread = threading.Thread(
                target=self._subprocess_stream_loop,
                daemon=True,
                name="camera-subprocess",
            )
            self._stream_thread.start()
            return True

        if _CV2_AVAILABLE:
            cap, backend = self._open_opencv_capture()
            if cap is not None:
                self._cap = cap
                self._backend = backend
                self._subprocess_mode = False
                self._stream_stop.clear()
                self._stream_thread = threading.Thread(
                    target=self._opencv_stream_loop,
                    daemon=True,
                    name="camera-opencv",
                )
                self._stream_thread.start()
                return True

        return False

    def _stop_stream_thread(self) -> None:
        self._stream_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=3.0)
            self._stream_thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._subprocess_mode = False
        if self._backend not in ("none",):
            self._backend = "none"

    def _opencv_stream_loop(self) -> None:
        interval = 1.0 / max(1, self._stream_fps)
        while not self._stream_stop.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(interval)
                continue
            jpg = self._encode_jpeg(frame)
            with self._frame_lock:
                self._latest_frame = frame
                if jpg:
                    self._latest_jpeg = jpg
            time.sleep(interval)

    def _subprocess_stream_loop(self) -> None:
        def on_frame(jpeg: bytes) -> None:
            with self._frame_lock:
                self._latest_jpeg = jpeg
                frame = self._jpeg_to_frame(jpeg)
                if frame is not None:
                    self._latest_frame = frame

        run_subprocess_worker(
            on_frame=on_frame,
            stop_event=self._stream_stop,
            width=self._capture_width,
            height=self._capture_height,
            fps=self._capture_fps,
            jpeg_quality=self._jpeg_quality,
        )

    # ------------------------------------------------------------------
    # One-shot capture
    # ------------------------------------------------------------------

    def _capture_jpeg_once_unlocked(self) -> Optional[bytes]:
        if self._use_v4l2ctl and self._device:
            frame = self._capture_v4l2ctl()
            if frame:
                return frame
        if self._use_ffmpeg and self._device:
            frame = self._capture_ffmpeg()
            if frame:
                return frame
        if self._device:
            frame = self._capture_raw_device()
            if frame:
                return frame
        if self._use_gst:
            return self._capture_gst_once()
        return None

    def _capture_gst_once(self) -> Optional[bytes]:
        """Single JPEG frame via gst-launch (CSI or V4L2), no OpenCV."""
        cmd = [
            "gst-launch-1.0", "-q",
            "nvarguscamerasrc", "num-buffers=1",
            "!",
            f"video/x-raw(memory:NVMM),width={self._capture_width},"
            f"height={self._capture_height},framerate={self._capture_fps}/1",
            "!", "nvvidconv",
            "!", "video/x-raw,format=BGRx",
            "!", "videoconvert",
            "!", "jpegenc", f"quality={self._jpeg_quality}",
            "!", "fdsink", "fd=1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=8)
            if result.returncode == 0 and len(result.stdout) > 100:
                start = result.stdout.find(b"\xff\xd8")
                return result.stdout[start:] if start >= 0 else result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"[CAMERA] gst one-shot error: {exc}")
        return None

    def _jpeg_to_frame(self, jpeg_bytes: bytes) -> Any:
        """Decode JPEG to BGR ndarray (OpenCV) or PIL grayscale image."""
        if _CV2_AVAILABLE:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        if _PIL_AVAILABLE:
            return PILImage.open(io.BytesIO(jpeg_bytes)).convert("L")
        return None

    def _open_opencv_capture(self) -> Tuple[Any, str]:
        """Manual §4.1.3 — GStreamer first, then V4L2 index."""
        if not _CV2_AVAILABLE:
            return None, "none"

        pipeline = build_gstreamer_pipeline_string(
            self._capture_width,
            self._capture_height,
            self._capture_fps,
            self._gstreamer_flip,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap, "gstreamer"
            cap.release()

        indices = [self._camera_index]
        if self._device:
            try:
                indices.insert(0, int(self._device.rsplit("video", 1)[-1]))
            except ValueError:
                pass
        seen = set()
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, f"v4l2-index-{idx}"
                cap.release()
        return None, "none"

    # ------------------------------------------------------------------
    # V4L2 one-shot backends (unchanged behaviour)
    # ------------------------------------------------------------------

    def _capture_v4l2ctl(self) -> Optional[bytes]:
        try:
            subprocess.run(
                [
                    "v4l2-ctl", "-d", self._device,
                    "--set-fmt-video",
                    f"width={self._capture_width},height={self._capture_height},pixelformat=MJPG",
                ],
                capture_output=True,
                timeout=5,
            )
            result = subprocess.run(
                [
                    "v4l2-ctl", "-d", self._device,
                    "--stream-mmap", "--stream-count=1",
                    "--stream-to=-",
                ],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                data = result.stdout
                start = data.find(b"\xff\xd8")
                return data[start:] if start >= 0 else data
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"[CAMERA] v4l2-ctl capture error: {exc}")
        return None

    def _capture_ffmpeg(self) -> Optional[bytes]:
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "v4l2",
                    "-input_format", "mjpeg",
                    "-video_size", f"{self._capture_width}x{self._capture_height}",
                    "-framerate", str(self._capture_fps),
                    "-i", self._device,
                    "-frames:v", "1",
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "-q:v", "5",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"[CAMERA] ffmpeg capture error: {exc}")
        return None

    def _capture_raw_device(self) -> Optional[bytes]:
        try:
            fd = os.open(self._device, os.O_RDONLY | os.O_NONBLOCK)
            try:
                data = b""
                for _ in range(10):
                    try:
                        data += os.read(fd, 1024 * 100)
                        if len(data) > 1024 * 1024:
                            break
                    except BlockingIOError:
                        time.sleep(0.05)
                if data:
                    start = data.find(b"\xff\xd8")
                    end = data.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start >= 0 and end >= 0:
                        return data[start:end + 2]
            finally:
                os.close(fd)
        except OSError as exc:
            print(f"[CAMERA] Raw device read error: {exc}")
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_jpeg(self, frame: Any) -> Optional[bytes]:
        if not _CV2_AVAILABLE:
            return None
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        return buf.tobytes() if ok else None

    def _resize_jpeg(self, jpeg_bytes: bytes) -> bytes:
        if not _PIL_AVAILABLE:
            return jpeg_bytes
        try:
            img = PILImage.open(io.BytesIO(jpeg_bytes))
            if img.width > self._stream_width or img.height > self._stream_height:
                img = img.resize(
                    (self._stream_width, self._stream_height),
                    PILImage.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._jpeg_quality)
            return buf.getvalue()
        except Exception as exc:
            print(f"[CAMERA] JPEG resize error (using original): {exc}")
            return jpeg_bytes


def detect_obstacle_in_jpeg(jpeg_bytes: bytes) -> bool:
    """
    Edge-density obstacle check using Pillow only (no OpenCV).
    Mirrors the summon Canny heuristic on a downscaled grayscale ROI.
    """
    if not _PIL_AVAILABLE or ImageFilter is None:
        return False
    try:
        img = PILImage.open(io.BytesIO(jpeg_bytes)).convert("L")
        img = img.resize((160, 120), PILImage.Resampling.BILINEAR)
        w, h = img.size
        roi = img.crop((0, int(h * 0.45), w, h))
        edges = roi.filter(ImageFilter.FIND_EDGES)
        ew, eh = edges.size
        px = edges.load()
        threshold = 30

        def density(x0: int, x1: int) -> float:
            count = 0
            total = (x1 - x0) * eh
            for y in range(eh):
                for x in range(x0, x1):
                    if px[x, y] > threshold:
                        count += 1
            return count / max(1, total)

        third = ew // 3
        center = density(third, 2 * third)
        left = density(0, third)
        right = density(2 * third, ew)
        return center > 0.03 or max(left, right) > 0.05
    except Exception:
        return False


_camera_instance: Optional[LightweightCamera] = None


def get_camera(
    capture_width: int = 640,
    capture_height: int = 480,
    capture_fps: int = 30,
    stream_width: int = 320,
    stream_height: int = 240,
    stream_fps: int = 10,
    jpeg_quality: int = 60,
    gstreamer_flip: int = 0,
    camera_index: int = 0,
    # Legacy kwargs kept for callers passing width/height/fps only
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
) -> LightweightCamera:
    global _camera_instance
    if width is not None:
        stream_width = width
    if height is not None:
        stream_height = height
    if fps is not None:
        stream_fps = fps

    if _camera_instance is None:
        _camera_instance = LightweightCamera(
            capture_width=capture_width,
            capture_height=capture_height,
            capture_fps=capture_fps,
            stream_width=stream_width,
            stream_height=stream_height,
            stream_fps=stream_fps,
            jpeg_quality=jpeg_quality,
            gstreamer_flip=gstreamer_flip,
            camera_index=camera_index,
        )
    return _camera_instance
