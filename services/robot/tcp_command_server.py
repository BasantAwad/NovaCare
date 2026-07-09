#!/usr/bin/env python3
"""TCP command server for AIoT SerBot Rover - Production version"""

import socket
import threading
import os
import json
import sys
import traceback
import time
import struct
import queue
import subprocess
import http.server
import socketserver
import urllib.request
import urllib.error

from camera_pipelines import build_gstreamer_pipeline_string, run_subprocess_worker

HOST = "0.0.0.0"
PORT = 5555
BUFFER_SIZE = 1024

# How long (seconds) to keep moving after the last MOVE command before auto-stopping.
MOVE_TIMEOUT = 0.5

# Port on which raw JPEG frames are streamed to laptop clients.
STREAM_PORT = 5556    # kept for reference; no longer used
MJPEG_PORT  = 5557    # HTTP MJPEG stream – cv2.VideoCapture reads this directly

# ------------------------------------------------------------
# JetAuto Robot HAL (replaces old pop.Pilot.SerBot)
# ------------------------------------------------------------
_hal_robot = None

def _get_robot_hal():
    """Get the JetAuto robot HAL instance (ROS2-based)."""
    global _hal_robot
    if _hal_robot is not None:
        return _hal_robot
    try:
        from robot_hal import get_robot
        _hal_robot = get_robot()
        print("[JETAUTO] Robot HAL initialized — using ROS2 cmd_vel + Nav2")
        return _hal_robot
    except Exception as e:
        print(f"[ERROR] Robot HAL init failed: {e}")
        traceback.print_exc()
        return None


# ------------------------------------------------------------
# Map & state management (unchanged)
# ------------------------------------------------------------
MAP_FILE = "rover_room_map.json"
_current_position = [0, 0]
_home_position = [0, 0]
_visited_positions = [[0, 0]]
_camera_enabled = False
_obstacle_avoidance_enabled = False
_autonomous_enabled = False
_follow_enabled = False

def _dir_to_degree(direction):
    d = (direction or "").strip().lower()
    try:
        return int(d)
    except ValueError:
        pass
    # FIX: corrected mapping so UI arrow matches actual rover movement.
    # Original UI arrows were confused:
    #   UI down  → rover moves left   (was 270, now 180)
    #   UI up    → rover moves right  (was 90,  now 0  )
    #   UI left  → rover moves down   (was 180, now 270)
    #   UI right → rover moves up     (was 0,   now 90 )
    return {
        "up":    180,   # UI up    → rover forward (was mapped as right)
        "down":  0,     # UI down  → rover backward (was mapped as left)
        "left":  90,    # UI left  → rover left turn (was mapped as down)
        "right": 270,   # UI right → rover right turn (was mapped as up)
    }.get(d, None)


def _current_map_state():
    return {
        "home": {"x": _home_position[0], "y": _home_position[1]},
        "current": {"x": _current_position[0], "y": _current_position[1]},
        "visited": [{"x": x, "y": y} for x, y in _visited_positions],
        "camera_enabled": _camera_enabled,
        "obstacle_avoidance": _obstacle_avoidance_enabled,
        "autonomous": _autonomous_enabled,
        "follow_user": _follow_enabled,
    }

def _save_map():
    try:
        with open(MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(_current_map_state(), f, indent=2)
        print(f"[MAP] Saved to {MAP_FILE}")
    except Exception as e:
        print(f"[ERROR] Save map: {e}")

def _read_obstacle_distance(hal):
    """Read the closest obstacle distance from LiDAR via HAL."""
    if hal is None:
        return None
    try:
        _, dist = hal.lidar.get_closest_obstacle()
        return dist if dist != float('inf') else None
    except Exception as e:
        print(f"[WARN] Obstacle distance read failed: {e}")
    return None


# ------------------------------------------------------------
# OpenCV camera fallback
# ------------------------------------------------------------
_cv_capture = None          # cv2.VideoCapture instance (or None)
_cv_thread = None           # background streaming thread (OpenCV or ffmpeg)
_cv_stop_event = threading.Event()
_cv_lock = threading.Lock()
_ffmpeg_proc = None         # subprocess.Popen for the ffmpeg capture process

# ------------------------------------------------------------
# Streaming subscribers (legacy raw-socket clients)
# ------------------------------------------------------------
_stream_subscribers = set()

# ------------------------------------------------------------
# HTTP MJPEG streaming  (port MJPEG_PORT)
# The latest JPEG frame is stored here; the HTTP handler serves it
# in multipart format.  cv2.VideoCapture on the laptop decodes it
# automatically – no custom frame-parsing protocol needed.
# ------------------------------------------------------------
_mjpeg_latest_frame = None          # most recent JPEG bytes
_mjpeg_frame_lock   = threading.Lock()
_mjpeg_frame_event  = threading.Event()  # signals a new frame is ready

def _push_frame(jpeg_bytes: bytes):
    """Store the latest JPEG frame so the MJPEG HTTP server can serve it."""
    global _mjpeg_latest_frame
    with _mjpeg_frame_lock:
        _mjpeg_latest_frame = jpeg_bytes
    _mjpeg_frame_event.set()   # wake any HTTP handler waiting for a new frame


class _MjpegHandler(http.server.BaseHTTPRequestHandler):
    """Serves a continuous MJPEG stream to any HTTP client."""

    def log_message(self, *args):
        pass   # silence per-frame access logs

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        print(f"[MJPEG] Client connected {self.client_address}")
        last_sent = None
        try:
            while True:
                # Wait up to 2 s for a new frame
                _mjpeg_frame_event.wait(timeout=2.0)
                _mjpeg_frame_event.clear()
                with _mjpeg_frame_lock:
                    frame = _mjpeg_latest_frame
                if frame is None or frame is last_sent:
                    continue
                last_sent = frame
                self.wfile.write(
                    b'--jpgboundary\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'\r\n' + frame + b'\r\n'
                )
                self.wfile.flush()
        except Exception:
            pass
        print(f"[MJPEG] Client disconnected {self.client_address}")


def _mjpeg_server_loop():
    """Blocking MJPEG HTTP server loop (runs in a daemon thread)."""
    class _ReuseServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
    with _ReuseServer(('', MJPEG_PORT), _MjpegHandler) as srv:
        print(f"[MJPEG] HTTP stream server ready on port {MJPEG_PORT}")
        print(f"[MJPEG] Laptop: cv2.VideoCapture('http://<ROVER_IP>:{MJPEG_PORT}')")
        srv.serve_forever()


def _start_stream_server():
    """Launch the MJPEG HTTP stream server as a daemon thread."""
    t = threading.Thread(target=_mjpeg_server_loop, daemon=True, name='mjpeg-srv')
    t.start()

def _cv_stream_loop():
    """Background thread: reads frames from the camera and pushes JPEG-encoded
    bytes to all streaming subscribers (laptop vision controller clients)."""
    global _cv_capture
    print("[CAMERA] OpenCV streaming thread started.")
    try:
        import cv2 as _cv2
    except ImportError:
        print("[CAMERA] cv2 not available – cannot stream frames.")
        return
    while not _cv_stop_event.is_set():
        with _cv_lock:
            cap = _cv_capture
        if cap is None or not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            print("[CAMERA] Frame read failed – camera may have disconnected.")
            break
        # Push to the HTTP MJPEG server
        ok, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            _push_frame(buf.tobytes())
        # ~30 fps
        time.sleep(0.033)
    print("[CAMERA] OpenCV streaming thread stopped.")


def _camera_stream_worker():
    """Master capture worker using shared pipeline strategies (no pop.Pilot)."""
    global _ffmpeg_proc

    def on_frame(jpeg_bytes: bytes):
        _push_frame(jpeg_bytes)

    run_subprocess_worker(
        on_frame=on_frame,
        stop_event=_cv_stop_event,
        width=640,
        height=480,
        fps=15,
        jpeg_quality=70,
    )
    _ffmpeg_proc = None


def _start_ffmpeg_camera():
    """Launch the unified camera worker thread."""
    global _cv_thread
    _cv_stop_event.clear()
    _cv_thread = threading.Thread(
        target=_camera_stream_worker, daemon=True, name="cam-worker"
    )
    _cv_thread.start()
    print("[CAMERA] Camera worker started (ffmpeg / GStreamer – no OpenCV needed).")
    return True


def _start_opencv_camera():
    """Start camera streaming.
    Priority: ffmpeg/GStreamer worker → OpenCV fallback."""
    global _cv_capture, _cv_thread, _cv_stop_event

    # Already streaming?
    with _cv_lock:
        if _cv_capture is not None and hasattr(_cv_capture, "isOpened") and _cv_capture.isOpened():
            print("[CAMERA] Already streaming (OpenCV).")
            return True
    if _cv_thread is not None and _cv_thread.is_alive():
        print("[CAMERA] Already streaming.")
        return True

    def _startup_worker():
        global _cv_capture, _cv_thread
        # ── Prefer ffmpeg/GStreamer worker (no OpenCV install needed) ────────
        ffmpeg_ok = False
        gst_ok    = False
        try:
            r = subprocess.run(["ffmpeg", "-version"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ffmpeg_ok = (r.returncode == 0)
        except FileNotFoundError:
            pass
        try:
            r = subprocess.run(["gst-launch-1.0", "--version"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gst_ok = (r.returncode == 0)
        except FileNotFoundError:
            pass

        if ffmpeg_ok or gst_ok:
            _start_ffmpeg_camera()
            return

        # ── OpenCV last resort (manual §4.1.3 — Util.gstrmer + CAP_GSTREAMER) ─
        try:
            import cv2
        except ImportError:
            print("[CAMERA] No camera method available on this rover.")
            print("         Install ffmpeg:  sudo apt install -y ffmpeg")
            return

        pipeline = build_gstreamer_pipeline_string(640, 480, fps=30, flip=0)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                for idx in range(1, 4):
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        break

        if not cap.isOpened():
            print("[CAMERA] WARNING – no camera device found via OpenCV.")
            cap.release()
            return

        with _cv_lock:
            _cv_capture = cap
        _cv_stop_event.clear()
        _cv_thread = threading.Thread(target=_cv_stream_loop, daemon=True, name="cv-stream")
        _cv_thread.start()
        print("[CAMERA] Camera streaming started (OpenCV fallback).")

    threading.Thread(target=_startup_worker, daemon=True, name="cam-startup").start()
    return True

def _stop_opencv_camera():
    """Stop the camera streaming thread and release all resources."""
    global _cv_capture, _cv_thread, _ffmpeg_proc

    _cv_stop_event.set()

    # Terminate ffmpeg process if it is the active capture method
    if _ffmpeg_proc is not None:
        try:
            _ffmpeg_proc.terminate()
        except Exception:
            pass
        _ffmpeg_proc = None

    if _cv_thread is not None:
        _cv_thread.join(timeout=3.0)
        _cv_thread = None

    with _cv_lock:
        if _cv_capture is not None:
            try:
                _cv_capture.release()
            except Exception:
                pass
            _cv_capture = None

    print("[CAMERA] Camera streaming stopped.")


def _set_camera(enabled, bot):
    global _camera_enabled
    _camera_enabled = enabled
    print(f"[CAMERA] {'ON' if enabled else 'OFF'}")

    # OpenCV / subprocess fallback – never uses pop.Pilot camera APIs
    if enabled:
        _start_opencv_camera()
    else:
        _stop_opencv_camera()

    _save_map()


def _set_obstacle_avoidance(enabled, bot):
    global _obstacle_avoidance_enabled
    _obstacle_avoidance_enabled = enabled
    print(f"[SAFETY] Obstacle avoidance {'ON' if enabled else 'OFF'}")
    _set_camera(enabled, bot)
    if bot and hasattr(bot, "setObstacleAvoidance"):
        try:
            bot.setObstacleAvoidance(enabled)
        except Exception as e:
            print(f"[WARN] setObstacleAvoidance: {e}")
    _save_map()

def _set_autonomy(enabled, hal):
    global _autonomous_enabled
    _autonomous_enabled = enabled
    print(f"[AUTONOMY] {'ON' if enabled else 'OFF'}")
    _set_camera(enabled, hal)
    # On JetAuto, autonomy is handled by Nav2 — no direct HAL call needed
    _save_map()

def _set_follow(enabled, hal):
    global _follow_enabled
    _follow_enabled = enabled
    print(f"[FOLLOW] {'ON' if enabled else 'OFF'}")
    _set_camera(enabled, hal)
    if hal:
        try:
            if enabled:
                hal.motion.start_tracking("person")
            else:
                hal.motion.stop_tracking()
        except Exception as e:
            print(f"[WARN] setFollow: {e}")
    _save_map()

def _return_home(hal):
    print(f"[NAV] Return home via Nav2")
    if hal:
        hal.motion.stop()
        # Use Nav2 autonomous navigation to dock/home
        try:
            hal.motion.navigate_to(destination="dock")
        except Exception as e:
            print(f"[WARN] Navigate home: {e}")
    _current_position[:] = _home_position
    if list(_current_position) not in _visited_positions:
        _visited_positions.append(list(_current_position))
    _save_map()

def _record_move(direction):
    if direction == "right":
        _current_position[0] += 1
    elif direction == "left":
        _current_position[0] -= 1
    elif direction == "up":
        _current_position[1] += 1
    elif direction == "down":
        _current_position[1] -= 1
    if list(_current_position) not in _visited_positions:
        _visited_positions.append(list(_current_position))
    _save_map()

def _should_block_move(hal):
    if not _obstacle_avoidance_enabled:
        return False
    if hal:
        try:
            blocked = hal.lidar.is_obstacle_ahead()
            if blocked:
                print(f"[SAFETY] LiDAR obstacle detected – blocking move")
                hal.motion.stop()
                return True
        except Exception:
            pass
    return False


# ------------------------------------------------------------
# Movement auto-stop (coasting timeout)
# ------------------------------------------------------------
_move_timer = None          # threading.Timer that fires auto-stop
_move_timer_lock = threading.Lock()

def _cancel_move_timer():
    global _move_timer
    with _move_timer_lock:
        if _move_timer is not None:
            _move_timer.cancel()
            _move_timer = None

def _auto_stop(bot):
    """Called by the timer when no new MOVE arrives within MOVE_TIMEOUT seconds."""
    print(f"[MOVE] No new command for {MOVE_TIMEOUT}s – auto-stopping rover.")
    if bot:
        try:
            bot.stop()
        except Exception as e:
            print(f"[WARN] Auto-stop failed: {e}")

def _reset_move_timer(bot):
    """Cancel any pending auto-stop and start a fresh one."""
    _cancel_move_timer()
    global _move_timer
    timer = threading.Timer(MOVE_TIMEOUT, _auto_stop, args=(bot,))
    timer.daemon = True
    with _move_timer_lock:
        _move_timer = timer
    timer.start()


# ------------------------------------------------------------
# Command dispatcher
# ------------------------------------------------------------
def _dispatch_command(hal, cmd):
    """Dispatch one command using the JetAuto HAL.
    Returns a response string, or None for default 'OK: <cmd>' ack."""
    if not cmd:
        return None
    cmd = cmd.strip()

    if cmd.startswith("MOVE:"):
        direction = cmd.split(":", 1)[1].lower()
        degree = _dir_to_degree(direction)
        if degree is None:
            return None
        if _should_block_move(hal):
            _cancel_move_timer()
            return "BLOCKED: obstacle detected"
        if hal:
            hal.motion.move(degree, 35)
        _reset_move_timer(hal)
        _record_move(direction)
        return None

    if cmd in ("STOP", "EMERGENCY"):
        _cancel_move_timer()
        if hal:
            hal.motion.stop()
        if cmd == "EMERGENCY" and hal:
            hal.audio.play_alarm(880, 10)
        return None

    if cmd in ("DOCK", "RETURN_HOME"):
        _cancel_move_timer()
        _return_home(hal)
        return None

    # ── Navigation (Nav2) ─────────────────────────────────────────────
    if cmd.startswith("NAVIGATE:"):
        parts = cmd.split(":", 1)[1].split(",")
        if len(parts) == 3:
            try:
                x, y, theta = float(parts[0]), float(parts[1]), float(parts[2])
                if hal:
                    hal.motion.navigate_to(x=x, y=y, theta=theta)
                return "NAV:STARTED"
            except ValueError:
                return "NAV:INVALID_COORDS"
        elif len(parts) == 1:
            dest = parts[0].strip()
            if hal:
                hal.motion.navigate_to(destination=dest)
            return f"NAV:STARTED:{dest}"
        return "NAV:INVALID"

    if cmd == "NAV_STATUS":
        if hal:
            status = hal.get_navigation_status()
            return "NAV_STATUS:" + json.dumps(status)
        return "NAV_STATUS:unavailable"

    if cmd == "NAV_CANCEL":
        if hal:
            hal.motion.cancel_navigation()
        return "NAV:CANCELLED"

    if cmd == "SOS_TRIGGER":
        if hal:
            hal.audio.play_alarm(880, 30)
            hal.audio.speak("Emergency alert activated!", block=False)
        return "SOS:ACTIVE"

    if cmd == "SOS_CANCEL":
        if hal:
            hal.audio.stop_alarm()
        return "SOS:CANCELLED"

    if cmd.startswith("AUTONOMOUS:"):
        _set_autonomy(cmd.split(":", 1)[1].upper() == "ON", hal)
        return None
    if cmd.startswith("FOLLOW_USER:"):
        _set_follow(cmd.split(":", 1)[1].upper() == "ON", hal)
        return None
    if cmd.startswith("OBSTACLE_AVOIDANCE:"):
        _set_obstacle_avoidance(cmd.split(":", 1)[1].upper() == "ON", hal)
        return None
    if cmd.startswith("CAMERA:"):
        _set_camera(cmd.split(":", 1)[1].upper() == "ON", hal)
        return None

    # ── Query commands ────────────────────────────────────────────────
    if cmd == "GET_LIDAR" or cmd == "GET_ULTRASONIC":
        dist = _read_obstacle_distance(hal)
        return f"LIDAR:{dist if dist is not None else 'NONE'}"

    if cmd == "GET_STATE":
        state = _current_map_state()
        if hal:
            pose = hal.get_current_pose()
            if pose:
                state["pose"] = pose
        return "STATE:" + json.dumps(state)

    if cmd == "GET_POSE":
        if hal:
            pose = hal.get_current_pose()
            return "POSE:" + json.dumps(pose or {})
        return "POSE:{}"

    if cmd.startswith("MAP_SAVE"):
        return "MAP:SAVED"

    if cmd.startswith("MAP_LOAD:"):
        name = cmd.split(":", 1)[1].strip()
        return f"MAP:LOADED:{name}"

    # Ignore these commands
    if cmd.startswith("INPUT_MODE:") or cmd.startswith("SENSOR_SET:"):
        return None

    return None

def handle_client(client_socket, addr, bot):
    print(f"[+] Client connected {addr}")
    buffer = ""
    try:
        while True:
            data = client_socket.recv(BUFFER_SIZE)
            if not data:
                break
            buffer += data.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                print(f"[CMD] {line}")
                response = None
                try:
                    response = _dispatch_command(bot, line)
                except Exception as e:
                    print(f"[ERROR] {e}")
                # Use command-specific response if provided, else generic ACK
                reply = (response if response else f"OK: {line}") + "\n"
                client_socket.sendall(reply.encode())
    except Exception as e:
        print(f"[ERROR] {addr}: {e}")
    finally:
        client_socket.close()

def _telemetry_loop():
    """Background thread to push rover telemetry to the backend database."""
    print("[TELEMETRY] Starting background sync to database.")
    # Use environment variable for laptop/backend IP, fallback to gateway
    laptop_ip = os.environ.get("LAPTOP_IP", "192.168.8.1")
    api_base = f"http://{laptop_ip}:8001/api/telemetry"
    rover_id = os.environ.get("ROVER_ID", "RV001")

    while True:
        try:
            # Send Battery
            bat_req = urllib.request.Request(
                f"{api_base}/battery",
                data=json.dumps({
                    "rover_id": rover_id,
                    "battery_percent": 85,
                    "is_charging": False
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(bat_req, timeout=3)

            # Send Vitals (simulated sensor read)
            vit_req = urllib.request.Request(
                f"{api_base}/vitals",
                data=json.dumps({
                    "rover_id": rover_id,
                    "heart_rate": 72,
                    "spo2": 98.5,
                    "temperature": 36.6
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(vit_req, timeout=3)

        except Exception as e:
            pass # Silently fail if backend is unreachable

        time.sleep(10)

def main():
    print("[*] NovaCare JetAuto Rover Server")
    print(f"[*] Listening on {HOST}:{PORT}")
    bot = _get_robot_hal()
    if bot is None:
        print("[!] JetAuto HAL unavailable – movement disabled (map updates only).")
    else:
        print("[*] JetAuto HAL ready – rover will move via ROS2.")

    # Start video streaming server so laptop vision controller can subscribe
    _start_stream_server()

    # Optional telemetry sync to laptop backend (skip in minimal I/O mode)
    if os.environ.get("NOVACARE_MINIMAL", "0").strip().lower() not in ("1", "true", "yes"):
        threading.Thread(target=_telemetry_loop, daemon=True, name='telemetry-sync').start()
    else:
        print("[OK] Telemetry sync disabled (NOVACARE_MINIMAL=1)")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((HOST, PORT))
    except socket.error as e:
        if e.errno == 98:
            print(f"\n[ERROR] Port {PORT} in use. Kill it: sudo kill -9 $(sudo lsof -t -i:{PORT})\n")
        raise
    server.listen(5)
    print("[*] Server ready. Waiting for commands...")

    try:
        while True:
            client, addr = server.accept()
            threading.Thread(target=handle_client, args=(client, addr, bot), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    finally:
        _cancel_move_timer()
        _stop_opencv_camera()
        server.close()

if __name__ == "__main__":
    main()
