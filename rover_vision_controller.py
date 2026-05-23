#!/usr/bin/env python3
"""
rover_vision_controller.py – Laptop-side vision controller for NovaCare SerBot.

Run this on your LAPTOP (not the rover).  It:
  • Receives the live camera stream from the rover (port 5556).
  • Shows the annotated feed in a cv2 window.
  • Detects people (HOG) and autonomously follows the closest one.
  • Detects obstacles (camera edges + ultrasonic) and navigates around them.
  • Sends all movement commands back to the rover's TCP server (port 5555).

Usage:
    python rover_vision_controller.py --host <ROVER_IP> [--follow] [--obstacle]

Keyboard shortcuts while the window is open:
    Q  – quit
    F  – toggle Follow-Me mode
    O  – toggle Obstacle Avoidance mode
    S  – send STOP immediately

Dependencies (install on the LAPTOP):
    pip install opencv-python numpy
"""

import argparse
from typing import List, Optional, Tuple
import json
import queue
import socket
import struct
import threading
import time

import cv2
import numpy as np

# ── Network ports (must match test_rover_server.py) ───────────────────────────
COMMAND_PORT = 5555
MJPEG_PORT   = 5557   # HTTP MJPEG stream  – cv2.VideoCapture reads this directly
STREAM_PORT  = MJPEG_PORT   # alias kept so nothing else needs changing

# ── Follow-Me tuning ──────────────────────────────────────────────────────────
# Fraction of frame width that the person must drift before we turn.
TURN_DEADBAND    = 0.14
# Bounding-box height as a fraction of frame height: closer → bigger number.
CLOSE_THRESHOLD  = 0.60   # stop / back-up when person fills this much height
FAR_THRESHOLD    = 0.18   # move forward when person is this small
# Seconds without detecting a person before sending STOP.
PERSON_LOSE_SEC  = 3.0

# ── Obstacle-avoidance tuning ─────────────────────────────────────────────────
OBSTACLE_DIST_CM    = 40    # ultrasonic: too-close threshold (cm)
EDGE_DENSITY_THRESH = 0.20  # camera centre-zone edge density → obstacle
TURN_DURATION_S     = 0.9   # seconds to turn when dodging
CREEP_DURATION_S    = 0.45  # seconds to creep forward before re-checking

# ── Command rate-limiting ─────────────────────────────────────────────────────
# Minimum gap between consecutive MOVE commands (the server already does 0.5 s
# auto-stop so we must send commands faster than that).
MIN_CMD_INTERVAL_S  = 0.25


# ══════════════════════════════════════════════════════════════════════════════
# Thread-safe TCP command client
# ══════════════════════════════════════════════════════════════════════════════

class CommandClient:
    """Sends commands to the rover's TCP server and reads responses."""

    def __init__(self, host: str, port: int = COMMAND_PORT):
        self._host          = host
        self._port          = port
        self._sock          = None
        self._lock          = threading.Lock()
        self._last_cmd_time = 0.0

    def connect(self, timeout: float = 6.0) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((self._host, self._port))
            s.settimeout(2.0)
            self._sock = s
            print(f"[CMD] Connected to rover at {self._host}:{self._port}")
            return True
        except Exception as exc:
            print(f"[CMD] Connection failed: {exc}")
            return False

    def send(self, cmd: str) -> Optional[str]:
        with self._lock:
            if self._sock is None:
                return None
            try:
                self._sock.sendall((cmd.strip() + "\n").encode())
                resp = b""
                while not resp.endswith(b"\n"):
                    chunk = self._sock.recv(1024)
                    if not chunk:
                        break
                    resp += chunk
                return resp.decode("utf-8", errors="replace").strip()
            except Exception as exc:
                print(f"[CMD] Send error: {exc}")
                # Try to reconnect in the background
                threading.Thread(target=self._reconnect, daemon=True).start()
                self._sock = None
                return None

    def _reconnect(self):
        """Background reconnection loop."""
        time.sleep(1.0)
        for attempt in range(10):
            print(f"[CMD] Reconnection attempt {attempt + 1}…")
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4.0)
                s.connect((self._host, self._port))
                s.settimeout(2.0)
                with self._lock:
                    self._sock = s
                print("[CMD] Reconnected successfully.")
                return
            except Exception:
                time.sleep(2.0)
        print("[CMD] Could not reconnect to rover.")

    def move(self, direction: str):
        """Send a MOVE command, respecting the minimum interval."""
        now = time.time()
        if now - self._last_cmd_time >= MIN_CMD_INTERVAL_S:
            self._last_cmd_time = now
            self.send(f"MOVE:{direction}")

    def stop(self):
        self._last_cmd_time = 0.0    # reset so the next MOVE fires immediately
        self.send("STOP")

    def get_ultrasonic(self) -> Optional[float]:
        resp = self.send("GET_ULTRASONIC")
        if resp and resp.startswith("ULTRASONIC:"):
            val = resp.split(":", 1)[1].strip()
            try:
                return float(val)
            except ValueError:
                pass
        return None

    def close(self):
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None


# ══════════════════════════════════════════════════════════════════════════════
# Non-blocking JPEG stream receiver
# ══════════════════════════════════════════════════════════════════════════════

class StreamReceiver:
    """Receives the rover's MJPEG stream via HTTP.
    cv2.VideoCapture handles HTTP, multipart boundaries, and JPEG decoding
    automatically – no custom protocol parsing needed.
    """

    def __init__(self, host: str, port: int = MJPEG_PORT):
        self._url      = f"http://{host}:{port}"
        self._frame_q: queue.Queue = queue.Queue(maxsize=3)
        self._running  = False
        self._thread   = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._recv_loop, daemon=True, name="stream-recv"
        )
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            cap = None
            try:
                cap = cv2.VideoCapture(self._url)
                if not cap.isOpened():
                    print(f"[STREAM] Cannot open {self._url} – retrying in 2 s…")
                    time.sleep(2.0)
                    continue
                print(f"[STREAM] Connected to MJPEG stream at {self._url}")
                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        print("[STREAM] Frame read failed – reconnecting…")
                        break
                    # Keep only the most recent frame
                    if self._frame_q.full():
                        try:
                            self._frame_q.get_nowait()
                        except queue.Empty:
                            pass
                    self._frame_q.put_nowait(frame)
            except Exception as exc:
                if self._running:
                    print(f"[STREAM] Error: {exc} – retrying in 2 s…")
                    time.sleep(2.0)
            finally:
                if cap is not None:
                    cap.release()

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._running = False


# ══════════════════════════════════════════════════════════════════════════════
# Vision helpers
# ══════════════════════════════════════════════════════════════════════════════

# ── Person detector (HOG – no extra model files, CPU-only) ────────────────────
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_persons(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns [(x, y, w, h), …] for each detected person.
    Detection is run on a 640×480 downscale then mapped back to original size.
    """
    target_w, target_h = 640, 480
    orig_h, orig_w = frame.shape[:2]
    small = cv2.resize(frame, (target_w, target_h))

    boxes, weights = _hog.detectMultiScale(
        small,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05,
        useMeanshiftGrouping=False,
    )

    scale_x = orig_w / target_w
    scale_y = orig_h / target_h
    results = []
    for i, (x, y, w, h) in enumerate(boxes):
        if len(weights) > i and weights[i] < 0.5:
            continue  # low confidence → skip
        results.append((
            int(x * scale_x), int(y * scale_y),
            int(w * scale_x), int(h * scale_y),
        ))
    return results


def analyse_zones(frame: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute edge density in the left / centre / right thirds of the frame
    using the middle vertical band (ignores sky and immediate floor).
    Returns (left_density, centre_density, right_density).
    """
    h, w = frame.shape[:2]
    roi   = frame[h // 4 : 3 * h // 4, :]          # middle half vertically
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    third = w // 3
    l_d = float(np.mean(edges[:, :third]))        / 255.0
    c_d = float(np.mean(edges[:, third:2*third])) / 255.0
    r_d = float(np.mean(edges[:, 2*third:]))      / 255.0
    return l_d, c_d, r_d


# ══════════════════════════════════════════════════════════════════════════════
# Follow-Me controller
# ══════════════════════════════════════════════════════════════════════════════

class FollowController:
    """
    Keeps the detected person horizontally centred and at a comfortable distance.

    Decision logic (priority order):
      1. If person is more than TURN_DEADBAND off-centre → turn toward them.
      2. Else if person bbox is too small (far away)     → move forward.
      3. Else if person bbox is too large (too close)    → stop / back up.
      4. Else                                            → hold position.
    """

    def __init__(self, cmd: CommandClient):
        self._cmd         = cmd
        self._last_seen   = time.time()
        self._lost_warned = False

    def tick(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        display = frame.copy()

        persons = detect_persons(frame)

        if not persons:
            elapsed = time.time() - self._last_seen
            msg = f"Searching for person… ({elapsed:.1f}s)"
            cv2.putText(display, msg, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
            if elapsed > PERSON_LOSE_SEC and not self._lost_warned:
                print(f"[FOLLOW] Person lost for {PERSON_LOSE_SEC:.0f}s → STOP")
                self._cmd.stop()
                self._lost_warned = True
            return display

        # ── Select the largest bounding box (closest person) ──────────────
        bx, by, bw, bh = max(persons, key=lambda b: b[2] * b[3])
        self._last_seen   = time.time()
        self._lost_warned = False

        cx_person = bx + bw / 2.0
        offset    = (cx_person - w / 2.0) / w   # −0.5 … +0.5
        h_frac    = bh / h                       # 0 … 1

        # ── Movement decision ──────────────────────────────────────────────
        if abs(offset) > TURN_DEADBAND:
            direction = "right" if offset > 0 else "left"
            self._cmd.move(direction)
            action = f"TURN {direction.upper()}"
            box_color = (0, 200, 255)
        elif h_frac < FAR_THRESHOLD:
            self._cmd.move("up")
            action = "FORWARD (far)"
            box_color = (0, 255, 80)
        elif h_frac > CLOSE_THRESHOLD:
            self._cmd.stop()
            action = "STOP (too close)"
            box_color = (0, 60, 255)
        else:
            self._cmd.stop()
            action = "HOLD"
            box_color = (180, 255, 80)

        # ── Annotations ───────────────────────────────────────────────────
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), box_color, 2)
        cv2.putText(display, action, (bx, max(by - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

        # Horizontal offset bar
        mid_x = w // 2
        bar_x = int(mid_x + offset * w * 0.7)
        cv2.line(display, (mid_x, h - 18), (bar_x, h - 18), (255, 120, 30), 4)
        cv2.line(display, (mid_x, h - 28), (mid_x, h - 8), (220, 220, 220), 2)

        # Info panel
        cv2.putText(display,
                    f"offset={offset:+.2f}  h_frac={h_frac:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return display


# ══════════════════════════════════════════════════════════════════════════════
# Obstacle-avoidance controller
# ══════════════════════════════════════════════════════════════════════════════

class ObstacleController:
    """
    Finite state machine that navigates around obstacles.

        SCAN ──(obstacle)──► TURN ──(done)──► CREEP ──(done)──► SCAN
               ◄─────────────────────────────────────────────────┘
               (re-check: if obstacle still present, turn again)

    • SCAN  : rover moves forward while the path is clear.
    • TURN  : rover turns away from the obstacle for TURN_DURATION_S seconds.
    • CREEP : rover creeps forward for CREEP_DURATION_S seconds to check.

    The turn direction is chosen by comparing left vs. right edge densities –
    the rover turns toward the side with fewer edges (more open space).
    Ultrasonic is used as a hard distance gate in addition to camera edges.
    """

    _S_SCAN  = "SCAN"
    _S_TURN  = "TURN"
    _S_CREEP = "CREEP"

    def __init__(self, cmd: CommandClient):
        self._cmd          = cmd
        self._state        = self._S_SCAN
        self._state_until  = 0.0
        self._turn_dir     = "right"

    def tick(self, frame: np.ndarray, ultrasonic: Optional[float]) -> np.ndarray:
        display = frame.copy()
        h, w    = frame.shape[:2]

        l_d, c_d, r_d = analyse_zones(frame)
        now = time.time()

        # ── Obstacle detection ─────────────────────────────────────────────
        cam_obstacle  = c_d > EDGE_DENSITY_THRESH
        sonar_obstacle = (
            ultrasonic is not None
            and isinstance(ultrasonic, (int, float))
            and ultrasonic < OBSTACLE_DIST_CM
        )
        obstacle_ahead = cam_obstacle or sonar_obstacle

        # ── State machine ──────────────────────────────────────────────────
        if self._state == self._S_SCAN:
            if obstacle_ahead:
                self._cmd.stop()
                # Turn toward the more open side
                self._turn_dir    = "right" if l_d > r_d else "left"
                self._state       = self._S_TURN
                self._state_until = now + TURN_DURATION_S
                print(f"[AVOID] Obstacle! cam={c_d:.2f} sonar={ultrasonic} "
                      f"→ turning {self._turn_dir}")
            else:
                self._cmd.move("up")

        elif self._state == self._S_TURN:
            if now < self._state_until:
                self._cmd.move(self._turn_dir)
            else:
                self._state       = self._S_CREEP
                self._state_until = now + CREEP_DURATION_S
                print("[AVOID] Turn done → creeping to re-check")

        elif self._state == self._S_CREEP:
            if now < self._state_until:
                self._cmd.move("up")
            else:
                print("[AVOID] Re-entering SCAN")
                self._state = self._S_SCAN

        # ── HUD ────────────────────────────────────────────────────────────
        hud_color = (0, 40, 220) if obstacle_ahead else (30, 220, 80)
        sonar_str = f"{ultrasonic:.0f}cm" if ultrasonic is not None else "N/A"
        cv2.putText(display,
                    f"State: {self._state}  sonar: {sonar_str}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)

        # Edge density bars (left / centre / right)
        bar_y = 50
        for idx, (density, label) in enumerate([(l_d, "L"), (c_d, "C"), (r_d, "R")]):
            bar_x     = 10 + idx * 130
            bar_w     = int(density * 220)
            bar_color = (0, 40, 230) if (label == "C" and cam_obstacle) else (60, 190, 60)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), bar_color, -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + 220, bar_y + 18), (80, 80, 80), 1)
            cv2.putText(display, f"{label}:{density:.2f}",
                        (bar_x, bar_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

        return display


# ══════════════════════════════════════════════════════════════════════════════
# HUD overlay helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_mode_badge(frame: np.ndarray, follow: bool, obstacle: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    badges = []
    if follow:
        badges.append(("FOLLOW-ME", (0, 200, 130)))
    if obstacle:
        badges.append(("OBSTACLE AVOID", (0, 130, 240)))
    if not badges:
        badges.append(("MANUAL", (140, 140, 140)))

    x = 10
    y = h - 14
    for text, color in badges:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + 4), (30, 30, 30), -1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        x += tw + 20
    return frame


_NO_SIGNAL_FRAME: np.ndarray = None   # lazy-initialised placeholder


def _no_signal_frame(w: int = 640, h: int = 480) -> np.ndarray:
    global _NO_SIGNAL_FRAME
    if _NO_SIGNAL_FRAME is None or _NO_SIGNAL_FRAME.shape[:2] != (h, w):
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (20, 20, 30)
        cv2.putText(img, "Connecting to rover stream…",
                    (w // 2 - 190, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 180, 255), 2)
        cv2.putText(img, "Camera is starting on the rover, please wait",
                    (w // 2 - 220, h // 2 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        _NO_SIGNAL_FRAME = img
    return _NO_SIGNAL_FRAME.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Main control loop
# ══════════════════════════════════════════════════════════════════════════════

def run(host: str, follow: bool, obstacle: bool):
    print(f"[*] NovaCare Rover Vision Controller  →  rover @ {host}")
    print(f"    Follow-Me: {'ON' if follow else 'OFF'}   "
          f"Obstacle Avoidance: {'ON' if obstacle else 'OFF'}")
    print("    Keys:  Q=quit  F=toggle follow  O=toggle obstacle  S=stop")

    cmd    = CommandClient(host)
    stream = StreamReceiver(host)

    if not cmd.connect():
        print("[FATAL] Cannot connect to rover command port.  Exiting.")
        return

    stream.start()
    time.sleep(0.6)   # let the stream thread establish its connection

    # ── Always turn camera on first, then set mode flags ─────────────────
    print("[*] Sending CAMERA:ON to rover…")
    cmd.send("CAMERA:ON")
    time.sleep(0.4)   # give rover time to open the camera before streaming
    if follow:
        print("[*] Sending FOLLOW_USER:ON to rover…")
        cmd.send("FOLLOW_USER:ON")
    if obstacle:
        print("[*] Sending OBSTACLE_AVOIDANCE:ON to rover…")
        cmd.send("OBSTACLE_AVOIDANCE:ON")

    follow_ctrl   = FollowController(cmd)
    obstacle_ctrl = ObstacleController(cmd)

    # ── Background ultrasonic poll (rover is queried every 300 ms) ────────
    _sonar = [None]

    def _sonar_loop():
        while True:
            _sonar[0] = cmd.get_ultrasonic()
            time.sleep(0.3)

    threading.Thread(target=_sonar_loop, daemon=True, name="sonar-poll").start()

    # ── Main display / control loop ────────────────────────────────────────
    window_name = "NovaCare Rover – Vision Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    try:
        while True:
            frame = stream.get_frame(timeout=0.5)

            if frame is None:
                display = _no_signal_frame()
            else:
                display = frame.copy()
                if follow:
                    display = follow_ctrl.tick(display)
                if obstacle:
                    display = obstacle_ctrl.tick(display, _sonar[0])
                if not follow and not obstacle:
                    cv2.putText(display,
                                "Manual mode – press F or O to enable autonomous control",
                                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            _draw_mode_badge(display, follow, obstacle)
            cv2.imshow(window_name, display)

            # ── Keyboard input ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:        # Q or ESC → quit
                break

            elif key == ord("f"):                   # F → toggle Follow-Me
                follow = not follow
                cmd.send("FOLLOW_USER:" + ("ON" if follow else "OFF"))
                print(f"[KEY] Follow-Me → {'ON' if follow else 'OFF'}")
                if not follow:
                    cmd.stop()

            elif key == ord("o"):                   # O → toggle Obstacle Avoidance
                obstacle = not obstacle
                cmd.send("OBSTACLE_AVOIDANCE:" + ("ON" if obstacle else "OFF"))
                print(f"[KEY] Obstacle Avoidance → {'ON' if obstacle else 'OFF'}")
                if not obstacle:
                    cmd.stop()

            elif key == ord("s"):                   # S → immediate stop
                print("[KEY] Manual STOP")
                cmd.stop()

    except KeyboardInterrupt:
        pass

    finally:
        print("\n[*] Shutting down vision controller…")
        cmd.stop()
        if follow:
            cmd.send("FOLLOW_USER:OFF")
        if obstacle:
            cmd.send("OBSTACLE_AVOIDANCE:OFF")
        cmd.send("CAMERA:OFF")
        stream.stop()
        cmd.close()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NovaCare SerBot laptop-side vision controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host", required=True,
        help="IP address of the rover (e.g. 192.168.1.42)"
    )
    parser.add_argument(
        "--follow", action="store_true",
        help="Activate Follow-Me mode on startup"
    )
    parser.add_argument(
        "--obstacle", action="store_true",
        help="Activate Obstacle Avoidance mode on startup"
    )
    args = parser.parse_args()
    run(args.host, args.follow, args.obstacle)


if __name__ == "__main__":
    main()
