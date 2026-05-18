"""
Summon Controller — Central Coordinator for Robot Summon Operations
===================================================================

Integrates WebSocket message handling, state machine, RSSI tracking,
and stuck detection into a cohesive summon pipeline.

This controller registers itself with the existing WebSocketServer
and processes summon-related messages from the mobile app.

Architecture:
  WebSocketServer → SummonController → SummonStateMachine
                                     → RSSITracker
                                     → StuckDetector
                                     → Robot HAL (via adapter)
"""

import asyncio
import json
import logging
import os
import time
from enum import Enum
from typing import Optional, Dict, Any, Callable

from .summon_state import SummonState, SummonStateMachine
from .rssi_tracker import RSSITracker
from .stuck_detector import StuckDetector
from .protocol import (
    SummonProtocol, SummonMessageType,
    SummonRequest, SummonStatus, SummonResult,
)

logger = logging.getLogger(__name__)

# Navigation loop frequency
NAV_LOOP_HZ = 5            # 5 Hz navigation update
STATUS_BROADCAST_HZ = 2    # 2 Hz status to mobile
SUMMON_TIMEOUT_S = 300      # 5 minute max summon duration


class RuntimeMode(Enum):
    FULL_AI_MODE = "full_ai"
    LIGHTWEIGHT_MODE = "lightweight"
    NO_CAMERA_MODE = "no_camera"
    SIMULATION_MODE = "simulation"


def get_system_metrics() -> Dict[str, float]:
    """Get system CPU and RAM usage safely and dynamically with zero overhead."""
    metrics = {"cpu_percent": 0.0, "ram_percent": 0.0}
    try:
        import psutil
        metrics["cpu_percent"] = psutil.cpu_percent()
        metrics["ram_percent"] = psutil.virtual_memory().percent
    except ImportError:
        # Fallback reading /proc for Jetson Nano Linux environment
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r") as f:
                    lines = f.readlines()
                mem_total = 0.0
                mem_available = 0.0
                mem_free = 0.0
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        if parts[0] == "MemTotal:":
                            mem_total = float(parts[1])
                        elif parts[0] == "MemFree:":
                            mem_free = float(parts[1])
                        elif parts[0] == "MemAvailable:":
                            mem_available = float(parts[1])
                if mem_total > 0:
                    used = mem_total - (mem_available if mem_available > 0 else mem_free)
                    metrics["ram_percent"] = (used / mem_total) * 100.0
        except Exception:
            pass
    return metrics


class SummonController:
    """
    Central controller for the Summon Robot feature.

    Lifecycle:
    1. Mobile sends summon_request via WebSocket
    2. Controller transitions to REQUESTED → INITIALIZING → SCANNING_RSSI
    3. Navigation loop runs at 5Hz, broadcasting status at 2Hz
    4. On arrival or failure, terminal message sent to mobile
    5. Controller returns to IDLE

    Usage:
        controller = SummonController()
        await controller.initialize(ws_server)
        # Controller automatically handles WebSocket messages
    """

    def __init__(self):
        self.state_machine = SummonStateMachine()
        self.rssi_tracker = RSSITracker()
        self.stuck_detector = StuckDetector()

        self._ws_server = None
        self._active_request: Optional[SummonRequest] = None
        self._nav_task: Optional[asyncio.Task] = None
        self._camera_task: Optional[asyncio.Task] = None
        self._broadcast_callback: Optional[Callable] = None
        self._running = False
        
        # Navigation helpers / hysteresis counters
        self._arrival_confirm = 0
        self._leave_wall_confirm = 0
        self._last_obstacle_detected = False

        # Stabilization, Telemetry and Mode settings
        self._estimated_heading = 0.0
        self._last_turn_time = 0.0
        self._turn_cooldown = 1.2
        self._last_loop_latency_ms = 0.0
        self._camera_obstacle_detected = False
        self._camera_fps = 0.0
        self._camera_latency_ms = 0.0

        # Dynamic mode configuration (default to lightweight for Jetson Nano constraints)
        mode_val = os.getenv("ROBOT_RUNTIME_MODE", "lightweight").lower()
        if mode_val == "full_ai":
            self.runtime_mode = RuntimeMode.FULL_AI_MODE
        elif mode_val == "no_camera":
            self.runtime_mode = RuntimeMode.NO_CAMERA_MODE
        elif mode_val == "simulation":
            self.runtime_mode = RuntimeMode.SIMULATION_MODE
        else:
            self.runtime_mode = RuntimeMode.LIGHTWEIGHT_MODE
            
        logger.info(f"SummonController loaded in ROBOT_RUNTIME_MODE: {self.runtime_mode.name}")

    async def initialize(self, ws_server):
        """
        Initialize controller and register WebSocket handlers.

        Args:
            ws_server: The existing WebSocketServer instance
        """
        self._ws_server = ws_server

        # Register handlers for summon message types
        ws_server.register_handler(
            SummonMessageType.SUMMON_REQUEST.value,
            self._handle_summon_request,
        )
        ws_server.register_handler(
            SummonMessageType.SUMMON_CANCEL.value,
            self._handle_summon_cancel,
        )
        ws_server.register_handler(
            SummonMessageType.RSSI_UPDATE.value,
            self._handle_rssi_update,
        )

        self._running = True
        logger.info("SummonController initialized and handlers registered")

    async def shutdown(self):
        """Stop controller and cancel any active summon."""
        self._running = False
        if self._nav_task and not self._nav_task.done():
            self._nav_task.cancel()
            try:
                await self._nav_task
            except asyncio.CancelledError:
                pass
        if self._camera_task and not self._camera_task.done():
            self._camera_task.cancel()
            try:
                await self._camera_task
            except asyncio.CancelledError:
                pass
        self._cleanup()
        logger.info("SummonController shutdown")

    # ========== WebSocket Message Handlers ==========

    async def _handle_summon_request(self, payload: Dict[str, Any]):
        """Handle incoming summon_request from mobile."""
        if self.state_machine.is_active:
            await self._send_ack(
                SummonMessageType.SUMMON_REQUEST.value,
                success=False,
                message="Summon already in progress",
            )
            return

        request = SummonRequest.from_dict(payload)
        logger.info(f"Summon request received from user={request.user_id}, ble_mac={request.ble_mac}")

        self._active_request = request
        self.state_machine.transition(SummonState.REQUESTED, "mobile_request")

        await self._send_ack(
            SummonMessageType.SUMMON_REQUEST.value,
            success=True,
            message="Summon accepted",
        )

        # Start navigation and background camera polling
        self._nav_task = asyncio.ensure_future(self._navigation_loop())
        self._camera_task = asyncio.ensure_future(self._camera_polling_loop())

    async def _handle_summon_cancel(self, payload: Dict[str, Any]):
        """Handle incoming summon_cancel from mobile."""
        reason = payload.get("reason", "user_cancelled")
        logger.info(f"Summon cancel received: {reason}")

        if self.state_machine.is_active:
            self.state_machine.transition(SummonState.CANCELLED, reason)
            if self._nav_task and not self._nav_task.done():
                self._nav_task.cancel()
            if self._camera_task and not self._camera_task.done():
                self._camera_task.cancel()
            self._cleanup()

        await self._send_ack(
            SummonMessageType.SUMMON_CANCEL.value,
            success=True,
            message="Summon cancelled",
        )

    async def _handle_rssi_update(self, payload: Dict[str, Any]):
        """Handle incoming rssi_update from mobile."""
        rssi = int(payload.get("rssi", -100))
        heading = float(payload.get("heading", -1.0))
        self.rssi_tracker.update(rssi, heading_deg=heading)

    async def _camera_polling_loop(self):
        """Asynchronous background loop to poll camera frames at a low frequency."""
        from services.robot.robot_hal import get_robot
        robot = get_robot()

        if self.runtime_mode in [RuntimeMode.NO_CAMERA_MODE, RuntimeMode.SIMULATION_MODE]:
            logger.info(f"Camera polling loop bypassed in {self.runtime_mode.name} mode.")
            return

        logger.info(f"Starting async background camera poller in {self.runtime_mode.name} mode...")
        
        # Determine frequency based on mode (1Hz in full AI, 0.5Hz in lightweight)
        poll_interval = 1.0 if self.runtime_mode == RuntimeMode.FULL_AI_MODE else 2.0

        while self._running and self.state_machine.is_active:
            if not robot.camera.is_available:
                await asyncio.sleep(poll_interval)
                continue
                
            start_time = time.monotonic()
            try:
                # Run the check in executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                obstacle = await loop.run_in_executor(None, self._check_front_obstacle, robot)
                
                # Telemetry updates
                latency = (time.monotonic() - start_time) * 1000.0
                self._camera_latency_ms = latency
                self._camera_fps = 1.0 / max(0.001, time.monotonic() - start_time)
                self._camera_obstacle_detected = bool(obstacle)
                
                logger.debug(f"[CameraPoller] Obstacle check: {obstacle} (latency: {latency:.1f}ms)")
            except Exception as e:
                logger.error(f"[CameraPoller] Error: {e}")
                await asyncio.sleep(poll_interval)
                continue
                
            await asyncio.sleep(poll_interval)

    # ========== Navigation Loop ==========

    async def _navigation_loop(self):
        """
        Main navigation loop implementing Bug2-style obstacle avoidance
        and RSSI gradient ascent.
        """
        import sys
        import os
        # Ensure root path is accessible to import services
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from services.robot.robot_hal import get_robot
        robot = get_robot()

        try:
            # Phase 1: Initialize
            self.state_machine.transition(SummonState.INITIALIZING, "preparing_sensors")
            self.rssi_tracker.reset()
            self.stuck_detector.reset()
            self._estimated_heading = 0.0
            self._last_turn_time = 0.0
            await asyncio.sleep(0.5)  # Sensor warmup

            # Phase 2: Initial RSSI scan
            self.state_machine.transition(SummonState.SCANNING_RSSI, "initial_scan")

            last_status_time = 0.0
            start_time = time.monotonic()
            
            # Bug2 state
            hit_rssi = -200
            wall_turn_direction = "left"

            # Phase 3: Navigation loop
            while self._running and self.state_machine.is_active:
                loop_start = time.monotonic()
                now = time.monotonic()
                elapsed = now - start_time

                # Timeout check
                if elapsed > SUMMON_TIMEOUT_S:
                    robot.motion.stop()
                    await self._complete_summon(
                        success=False,
                        reason="Timeout — could not reach user within time limit",
                    )
                    break

                # Transition to navigating once we have RSSI data
                if (
                    self.state_machine.state == SummonState.SCANNING_RSSI
                    and self.rssi_tracker.sample_count >= 3
                ):
                    self.state_machine.transition(SummonState.NAVIGATING, "rssi_acquired")

                # Check arrival with hysteresis: require several consecutive confirmations
                if self.rssi_tracker.is_at_target:
                    self._arrival_confirm += 1
                else:
                    self._arrival_confirm = 0

                if self._arrival_confirm >= 3:
                    robot.motion.stop()
                    self.state_machine.transition(SummonState.ARRIVING, "rssi_near_target")
                    await asyncio.sleep(1.0)  # Confirmation dwell
                    if self.rssi_tracker.is_at_target:
                        await self._complete_summon(success=True, reason="Arrived at user")
                        break
                    else:
                        self.state_machine.transition(SummonState.NAVIGATING, "false_arrival")

                # Determine obstacle presence based on RuntimeMode
                obstacle_ahead = False

                if self.runtime_mode == RuntimeMode.NO_CAMERA_MODE:
                    # Bypasses camera checks entirely, uses LiDAR exclusively if available
                    if robot.lidar.is_available:
                        obstacle_ahead = robot.lidar.is_obstacle_ahead(cone_degrees=40, distance_mm=400)
                elif self.runtime_mode == RuntimeMode.SIMULATION_MODE:
                    # In simulation, obstacles are bypassed or mocked
                    obstacle_ahead = False
                else:
                    # In FULL_AI_MODE or LIGHTWEIGHT_MODE:
                    # Use decoupled background camera thread check
                    obstacle_ahead = self._camera_obstacle_detected
                    if not obstacle_ahead and robot.lidar.is_available:
                        # Fallback check using LiDAR
                        obstacle_ahead = robot.lidar.is_obstacle_ahead(cone_degrees=40, distance_mm=400)

                # Doorway Traversal Biasing: if we are near user and signal is actively improving,
                # bias the robot to go straight ahead through a narrow opening instead of triggering Bug2.
                is_near = self.rssi_tracker.is_near_target
                trend = self.rssi_tracker.get_trend()
                if is_near and trend == "improving" and obstacle_ahead:
                    center_clear = True
                    if robot.lidar.is_available:
                        center_clear = robot.lidar.get_distance_at(target_angle=0, cone_degrees=15) > 500
                    if center_clear:
                        logger.info("Doorway traversal bias triggered: strong signal center clear, bypassing wall following")
                        obstacle_ahead = False

                # Check stuck - call update method first!
                is_moving = robot.motion.is_moving
                self.stuck_detector.update(
                    rssi=self.rssi_tracker.current_rssi,
                    heading=self._estimated_heading,
                    is_moving=is_moving,
                    obstacle_ahead=obstacle_ahead
                )

                if self.stuck_detector.is_stuck and self.state_machine.state != SummonState.WALL_FOLLOWING:
                    robot.motion.stop()
                    if self.stuck_detector.should_abort:
                        await self._complete_summon(
                            success=False,
                            reason=f"Navigation failed: {self.stuck_detector.stuck_type.value}",
                        )
                        break
                    self.state_machine.transition(SummonState.RECOVERING, self.stuck_detector.stuck_type.value)
                    self.stuck_detector.record_recovery_attempt()
                    
                    # Recovery action: improved sweep-and-probe routine
                    try:
                        await self._perform_recovery(robot)
                    except Exception as e:
                        logger.error(f"Recovery routine failed: {e}")

                    self.state_machine.transition(SummonState.SCANNING_RSSI, "post_recovery_scan")

                # --- Bug2 Navigation Logic ---
                if self.state_machine.state in [SummonState.NAVIGATING, SummonState.WALL_FOLLOWING]:
                    # Ultrasonic / LiDAR safety: immediate hard stop if obstacle too close
                    try:
                        from services.robot.config import OBSTACLE_STOP_DISTANCE_MM
                        angle_closest, dist_closest = robot.lidar.get_closest_obstacle() if robot.lidar.is_available else (0.0, float('inf'))
                        if dist_closest < OBSTACLE_STOP_DISTANCE_MM:
                            robot.motion.stop()
                            self.state_machine.transition(SummonState.RECOVERING, "safety_stop")
                            # Use the improved recovery routine for safety
                            try:
                                await self._perform_recovery(robot)
                            except Exception:
                                # Fallback quick recovery
                                robot.motion.backward()
                                await asyncio.sleep(0.6)
                                robot.motion.turn_right()
                                await asyncio.sleep(0.6)
                                robot.motion.stop()
                            # After recovery, rescan RSSI
                            self.state_machine.transition(SummonState.SCANNING_RSSI, "post_safety")
                            await asyncio.sleep(0.5)
                            continue
                    except Exception:
                        # If LiDAR unavailable, skip hard stop check
                        pass

                    if self.state_machine.state == SummonState.NAVIGATING:
                        if obstacle_ahead:
                            # Hit an obstacle: Switch to wall following
                            robot.motion.stop()
                            hit_rssi = self.rssi_tracker.current_rssi
                            self.state_machine.transition(SummonState.WALL_FOLLOWING, "obstacle_detected")
                            # Decide turn direction using LiDAR when available
                            if robot.lidar.is_available:
                                left_dist = robot.lidar.get_distance_at(target_angle=270, cone_degrees=30)
                                right_dist = robot.lidar.get_distance_at(target_angle=90, cone_degrees=30)
                                wall_turn_direction = "left" if left_dist > right_dist else "right"
                            else:
                                wall_turn_direction = "left"
                        else:
                            # Gradient Ascent: Move forward, with hysteresis on trend
                            trend = self.rssi_tracker.get_trend()
                            if trend == "degrading":
                                # small corrective turn with anti-oscillation cooldown
                                if now - self._last_turn_time >= self._turn_cooldown:
                                    robot.motion.turn_left()
                                    self._last_turn_time = now
                                    self._estimated_heading = (self._estimated_heading - 45) % 360
                                    await asyncio.sleep(0.4)
                                    robot.motion.stop()
                            else:
                                # Forward bias: prefer small forward movements to traverse partial openings
                                robot.motion.forward()

                    elif self.state_machine.state == SummonState.WALL_FOLLOWING:
                        # Leave condition with hysteresis: require consecutive confirmations
                        path_clear = False
                        if robot.lidar.is_available:
                            path_clear = not robot.lidar.is_obstacle_ahead(cone_degrees=40, distance_mm=400)
                        else:
                            path_clear = not obstacle_ahead

                        if self.rssi_tracker.current_rssi > hit_rssi + 3 and path_clear:
                            self._leave_wall_confirm += 1
                        else:
                            self._leave_wall_confirm = 0

                        if self._leave_wall_confirm >= 2:
                            robot.motion.stop()
                            self.state_machine.transition(SummonState.NAVIGATING, "left_wall")
                            self._leave_wall_confirm = 0
                        else:
                            # Wall following movement
                            if obstacle_ahead:
                                if now - self._last_turn_time >= self._turn_cooldown:
                                    if wall_turn_direction == "left":
                                        robot.motion.turn_left()
                                        self._estimated_heading = (self._estimated_heading - 45) % 360
                                    else:
                                        robot.motion.turn_right()
                                        self._estimated_heading = (self._estimated_heading + 45) % 360
                                    self._last_turn_time = now
                            else:
                                robot.motion.forward()

                # Broadcast status at 2Hz
                if now - last_status_time >= 1.0 / STATUS_BROADCAST_HZ:
                    await self._broadcast_status()
                    last_status_time = now

                # Measure actual loop latency in milliseconds
                self._last_loop_latency_ms = (time.monotonic() - loop_start) * 1000.0

                # Navigation tick
                await asyncio.sleep(1.0 / NAV_LOOP_HZ)

        except asyncio.CancelledError:
            logger.info("Navigation loop cancelled")
            get_robot().motion.stop()
        except Exception as e:
            logger.error(f"Navigation loop error: {e}")
            get_robot().motion.stop()
            await self._complete_summon(success=False, reason=f"Internal error: {e}")

    async def _complete_summon(self, success: bool, reason: str):
        """Complete the summon operation (success or failure)."""
        result = SummonResult(
            success=success,
            reason=reason,
            total_time_seconds=round(self.state_machine.elapsed_seconds, 1),
            total_distance_estimate_m=0,  # TODO: integrate with odometry
            recovery_attempts=self.stuck_detector.recovery_attempts,
            final_rssi=self.rssi_tracker.current_rssi,
        )

        if success:
            self.state_machine.transition(SummonState.ARRIVED, reason)
            msg = SummonProtocol.build_arrived(result)
        else:
            self.state_machine.transition(SummonState.FAILED, reason)
            msg = SummonProtocol.build_failed(result)

        await self._broadcast_message(msg)
        logger.info(f"Summon completed: success={success}, reason={reason}")
        self._cleanup()

    def _cleanup(self):
        """Reset internal state after summon completes."""
        self._active_request = None
        self.rssi_tracker.reset()
        self.stuck_detector.reset()
        if self._camera_task and not self._camera_task.done():
            self._camera_task.cancel()
            self._camera_task = None
        # Don't reset state machine — keep terminal state for history
        # It will be reset on next summon_request

    # ========== Broadcasting ==========

    async def _broadcast_status(self):
        """Send current status and real-time performance telemetry to all connected mobile clients."""
        if not self._ws_server:
            return

        # Fetch lightweight system metrics safely
        metrics = get_system_metrics()

        status = SummonStatus(
            state=self.state_machine.state.value,
            rssi_current=self.rssi_tracker.current_rssi,
            rssi_trend=self.rssi_tracker.get_trend(),
            distance_estimate_m=self.rssi_tracker.estimate_distance_m(),
            obstacle_detected=self._last_obstacle_detected or (getattr(self.rssi_tracker, 'signal_lost', False)),
            is_wall_following=self.state_machine.state == SummonState.WALL_FOLLOWING,
            elapsed_seconds=round(self.state_machine.elapsed_seconds, 1),
            recovery_attempts=self.stuck_detector.recovery_attempts,
            message=self._get_status_message(),
        )

        msg = SummonProtocol.build_status(status)

        # Inject dynamic performance telemetry
        msg["telemetry"] = {
            "loop_latency_ms": round(self._last_loop_latency_ms, 1),
            "cpu_percent": round(metrics["cpu_percent"], 1),
            "ram_percent": round(metrics["ram_percent"], 1),
            "camera_fps": round(self._camera_fps, 1),
            "camera_latency_ms": round(self._camera_latency_ms, 1),
            "runtime_mode": self.runtime_mode.value,
        }

        await self._broadcast_message(msg)

    async def _send_ack(self, original_type: str, success: bool, message: str):
        """Send acknowledgment message."""
        msg = SummonProtocol.build_ack(original_type, success, message)
        await self._broadcast_message(msg)

    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message via the WebSocket server."""
        if self._ws_server:
            try:
                raw = json.dumps(message)
                await self._ws_server._broadcast(raw)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    def _get_status_message(self) -> str:
        """Generate human-readable status message."""
        state = self.state_machine.state
        messages = {
            SummonState.IDLE: "Ready",
            SummonState.REQUESTED: "Summon request received",
            SummonState.INITIALIZING: "Preparing sensors...",
            SummonState.SCANNING_RSSI: "Scanning for your signal...",
            SummonState.NAVIGATING: "Moving toward you...",
            SummonState.WALL_FOLLOWING: "Navigating around obstacle...",
            SummonState.RECOVERING: "Adjusting path...",
            SummonState.ARRIVING: "Almost there!",
            SummonState.ARRIVED: "I'm here!",
            SummonState.FAILED: "Could not reach you",
            SummonState.CANCELLED: "Summon cancelled",
        }
        return messages.get(state, "")

    def _check_front_obstacle(self, robot) -> bool:
        """
        Capture a single camera frame and run a lightweight, fast heuristic
        to detect an obstacle directly in front. Returns True if obstacle
        likely present. This should be called sparingly (triggered only).
        """
        try:
            ret, frame = robot.camera.read_frame()
            if not ret or frame is None:
                return False
            import cv2
            import numpy as np
            h, w = frame.shape[:2]
            roi = frame[int(h * 0.45):, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            third = edges.shape[1] // 3
            left_density = np.count_nonzero(edges[:, :third]) / (edges.shape[0] * max(1, third))
            center_density = np.count_nonzero(edges[:, third:2*third]) / (edges.shape[0] * max(1, third))
            right_density = np.count_nonzero(edges[:, 2*third:]) / (edges.shape[0] * max(1, third))
            # Tuneable thresholds — conservative defaults
            obstacle = center_density > 0.03 or max(left_density, right_density) > 0.05
            # Store last-obstacle for status reporting
            self._last_obstacle_detected = bool(obstacle)
            return obstacle
        except Exception:
            return False

    async def _perform_recovery(self, robot):
        """
        Improved recovery routine:
        - Back up
        - Perform a sweep-turn (incremental rotations) while sampling RSSI
        - Rotate back to the best-scoring heading
        - Probe forward briefly
        - Allow RSSI to re-stabilize
        """
        try:
            logger.info("Starting recovery routine: backup + RSSI sweep")
            # safety stop and backup
            robot.motion.stop()
            robot.motion.backward()
            await asyncio.sleep(0.6)
            robot.motion.stop()

            # Sweep parameters
            n_steps = 7
            step_duration = 0.45  # seconds per step (approx 30-45deg depending on robot)
            sample_pause = 0.45   # time to wait after stopping to let RSSI updates arrive

            rssi_samples = []

            # Perform a rightward sweep, sampling RSSI at each step including initial
            for i in range(n_steps):
                # Allow a short pause for RSSI to settle
                await asyncio.sleep(sample_pause)
                r = self.rssi_tracker.current_rssi
                rssi_samples.append(r)

                # Turn right for next step (except after last sample)
                if i < n_steps - 1:
                    robot.motion.turn_right()
                    await asyncio.sleep(step_duration)
                    robot.motion.stop()

            # Final sample after sweep
            await asyncio.sleep(sample_pause)
            rssi_samples.append(self.rssi_tracker.current_rssi)

            # Determine best sample index
            best_idx = int(max(range(len(rssi_samples)), key=lambda i: rssi_samples[i]))
            logger.info(f"Recovery RSSI samples: {rssi_samples}, best_idx={best_idx}")

            # We ended at right-most position; compute steps to rotate back to best
            end_idx = len(rssi_samples) - 1
            delta_steps = end_idx - best_idx

            if delta_steps > 0:
                # rotate left by delta_steps
                robot.motion.turn_left()
                await asyncio.sleep(delta_steps * step_duration)
                robot.motion.stop()
            elif delta_steps < 0:
                # rotate right further
                robot.motion.turn_right()
                await asyncio.sleep(abs(delta_steps) * step_duration)
                robot.motion.stop()

            # Probe forward briefly to escape tight spot
            robot.motion.forward()
            await asyncio.sleep(0.6)
            robot.motion.stop()

            # Allow RSSI samples to update and smooth
            await asyncio.sleep(1.0)
            logger.info("Recovery routine completed")
        except asyncio.CancelledError:
            robot.motion.stop()
            raise
        except Exception as e:
            robot.motion.stop()
            logger.error(f"Error during recovery routine: {e}")

    # ========== Public API ==========

    def get_state(self) -> Dict[str, Any]:
        """Get complete summon state for external consumption."""
        return {
            "state_machine": self.state_machine.get_snapshot(),
            "rssi": self.rssi_tracker.get_snapshot(),
            "stuck": self.stuck_detector.get_snapshot(),
            "active_request": self._active_request.to_dict() if self._active_request else None,
        }


# Singleton
_controller: Optional[SummonController] = None


def get_summon_controller() -> SummonController:
    """Get or create global SummonController instance."""
    global _controller
    if _controller is None:
        _controller = SummonController()
    return _controller
