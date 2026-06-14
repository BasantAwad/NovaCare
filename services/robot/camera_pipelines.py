"""
Shared camera capture pipelines for NovaCare robot services.

Derived from the SerBot Prime X manual (§4.1.3 Camera Utilizing):
  - ``pop.Util.gstrmer(width, height, fps, flip)`` for Jetson CSI capture
  - ``nvarguscamerasrc`` GStreamer pipeline when ``pop.Util`` is unavailable
  - V4L2 ``/dev/video*`` via ffmpeg / gst-launch for USB cameras

Does NOT use ``pop.Pilot`` or ``Pilot.Camera``.
"""

import os
import select
import subprocess
import time
from typing import Callable, Generator, List, Optional, Tuple

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"

CAMERA_DEVICES = [f"/dev/video{i}" for i in range(4)]


def detect_v4l2_device(devices: Optional[List[str]] = None) -> Optional[str]:
    """Return the first accessible ``/dev/videoN`` device."""
    for dev in devices or CAMERA_DEVICES:
        if not os.path.exists(dev):
            continue
        try:
            fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
            os.close(fd)
            return dev
        except OSError:
            continue
    return None


def check_command(cmd: str) -> bool:
    try:
        result = subprocess.run(
            ["which", cmd],
            capture_output=True,
            timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


def build_nvargus_pipeline(
    width: int,
    height: int,
    fps: int,
    flip: int = 0,
) -> str:
    """Manual §4.1.3.1 — Jetson CSI ``nvarguscamerasrc`` pipeline for OpenCV."""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def build_gstreamer_pipeline_string(
    width: int,
    height: int,
    fps: int = 30,
    flip: int = 0,
) -> str:
    """
    Prefer ``pop.Util.gstrmer`` (manual §4.1.3.1), else static nvargus pipeline.
    Never imports ``pop.Pilot``.
    """
    try:
        from pop import Util

        pipeline = Util.gstrmer(width=width, height=height, fps=fps, flip=flip)
        if pipeline:
            return pipeline
    except Exception:
        pass
    return build_nvargus_pipeline(width, height, fps, flip)


def build_subprocess_strategies(
    v4l2_dev: Optional[str],
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    jpeg_quality: int = 70,
) -> List[Tuple[str, List[str], float]]:
    """Ordered ffmpeg / gst-launch strategies used by TCP MJPEG worker."""
    size = f"{width}x{height}"
    strategies: List[Tuple[str, List[str], float]] = []

    if v4l2_dev:
        strategies.extend([
            (
                "ffmpeg-mjpeg",
                [
                    "ffmpeg", "-f", "v4l2", "-input_format", "mjpeg",
                    "-framerate", str(fps), "-video_size", size,
                    "-i", v4l2_dev,
                    "-f", "image2pipe", "-vcodec", "copy", "-",
                ],
                2.0,
            ),
            (
                "ffmpeg-raw",
                [
                    "ffmpeg", "-f", "v4l2",
                    "-framerate", str(max(5, fps - 5)), "-video_size", size,
                    "-i", v4l2_dev,
                    "-f", "image2pipe", "-vcodec", "mjpeg",
                    "-q:v", str(max(2, 10 - jpeg_quality // 10)), "-",
                ],
                3.0,
            ),
            (
                "gst-v4l2",
                [
                    "gst-launch-1.0", "-q",
                    "v4l2src", f"device={v4l2_dev}",
                    "!", "videoconvert",
                    "!", "jpegenc", f"quality={jpeg_quality}",
                    "!", "fdsink", "fd=1",
                ],
                2.5,
            ),
        ])

    strategies.append((
        "gst-csi",
        [
            "gst-launch-1.0", "-q",
            "nvarguscamerasrc",
            "!",
            f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1",
            "!", "nvvidconv",
            "!", "video/x-raw,format=BGRx",
            "!", "videoconvert",
            "!", "jpegenc", f"quality={jpeg_quality}",
            "!", "fdsink", "fd=1",
        ],
        3.0,
    ))
    return strategies


def pump_jpeg_pipe(
    proc: subprocess.Popen,
    stop_event=None,
) -> Generator[bytes, None, None]:
    """Read JPEG frames from a subprocess stdout without blocking indefinitely."""
    buf = b""
    stdout = proc.stdout
    if stdout is None:
        return

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            ready, _, _ = select.select([stdout], [], [], 1.0)
        except (ValueError, OSError):
            break
        if not ready:
            if proc.poll() is not None:
                break
            continue
        try:
            chunk = stdout.read(65536)
        except Exception:
            break
        if not chunk:
            break
        buf += chunk
        while True:
            start = buf.find(JPEG_START)
            if start == -1:
                buf = b""
                break
            end = buf.find(JPEG_END, start + 2)
            if end == -1:
                buf = buf[start:]
                break
            yield buf[start:end + 2]
            buf = buf[end + 2:]


def run_subprocess_strategy(
    label: str,
    cmd: List[str],
    startup_wait: float = 2.5,
) -> Optional[subprocess.Popen]:
    """Launch a capture subprocess; return it if still alive after warmup."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError:
        print(f"[CAMERA] {label}: executable not found – skipping.")
        return None
    except Exception as exc:
        print(f"[CAMERA] {label}: launch error – {exc}")
        return None

    time.sleep(startup_wait)
    if proc.poll() is not None:
        err = proc.stderr.read(400).decode("utf-8", errors="replace").strip()
        print(f"[CAMERA] {label}: exited early. stderr → {err or '(empty)'}")
        return None

    print(f"[CAMERA] {label}: process running – streaming frames.")
    return proc


def run_subprocess_worker(
    on_frame: Callable[[bytes], None],
    stop_event,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    jpeg_quality: int = 70,
) -> None:
    """
    Try subprocess capture strategies until one produces JPEG frames.
    Calls ``on_frame(jpeg_bytes)`` for each frame.
    """
    v4l2_dev = detect_v4l2_device()
    if v4l2_dev:
        print(f"[CAMERA] Found V4L2 device: {v4l2_dev}")
    else:
        print("[CAMERA] No /dev/videoX found – will try GStreamer CSI path only.")

    strategies = build_subprocess_strategies(
        v4l2_dev, width=width, height=height, fps=fps, jpeg_quality=jpeg_quality,
    )

    for label, cmd, wait in strategies:
        if stop_event.is_set():
            return
        print(f"[CAMERA] Trying strategy: {label}")
        proc = run_subprocess_strategy(label, cmd, startup_wait=wait)
        if proc is None:
            continue

        got_frame = False
        for jpeg in pump_jpeg_pipe(proc, stop_event=stop_event):
            if not got_frame:
                print(f"[CAMERA] Strategy '{label}' is producing frames.")
                got_frame = True
            on_frame(jpeg)

        if stop_event.is_set():
            try:
                proc.terminate()
            except Exception:
                pass
            return

        if got_frame:
            print(f"[CAMERA] Strategy '{label}' ended – trying next.")
        else:
            print(f"[CAMERA] Strategy '{label}' produced no frames – trying next.")
        try:
            proc.terminate()
        except Exception:
            pass

    print("[CAMERA] All camera strategies exhausted – no video available.")
    print("[CAMERA] Check: ls /dev/video* && ffmpeg -version && gst-launch-1.0 --version")
