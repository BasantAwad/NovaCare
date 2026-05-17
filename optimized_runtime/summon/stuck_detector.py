"""
Stuck Detector — Rule-Based FSM for Navigation Failures
"""

import time
import logging
from enum import Enum
from typing import Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

STUCK_TIME_THRESHOLD_S = 5.0
OSCILLATION_WINDOW = 10
OSCILLATION_THRESHOLD = 6
SPIN_DETECTION_TURNS = 4
MAX_RECOVERY_ATTEMPTS = 5


class StuckType(Enum):
    NONE = "none"
    NO_PROGRESS = "no_progress"
    PHYSICAL_STUCK = "physical_stuck"
    OSCILLATING = "oscillating"
    SPINNING = "spinning"


class StuckDetector:
    """Detects stuck/oscillation conditions during navigation."""

    def __init__(self):
        self._stuck_type = StuckType.NONE
        self._recovery_attempts = 0
        self._last_progress_time = time.monotonic()
        self._last_rssi = -100
        self._heading_history: deque = deque(maxlen=OSCILLATION_WINDOW * 2)
        self._direction_changes: deque = deque(maxlen=OSCILLATION_WINDOW)
        self._consecutive_turns = 0
        self._last_heading = 0.0

    @property
    def is_stuck(self) -> bool:
        return self._stuck_type != StuckType.NONE

    @property
    def stuck_type(self) -> StuckType:
        return self._stuck_type

    @property
    def recovery_attempts(self) -> int:
        return self._recovery_attempts

    @property
    def should_abort(self) -> bool:
        return self._recovery_attempts >= MAX_RECOVERY_ATTEMPTS

    def update(self, rssi: int, heading: float, is_moving: bool, obstacle_ahead: bool = False):
        now = time.monotonic()
        rssi_improved = rssi > self._last_rssi + 2
        if rssi_improved:
            self._last_progress_time = now
            self._stuck_type = StuckType.NONE
            self._consecutive_turns = 0

        time_no_progress = now - self._last_progress_time
        if time_no_progress > STUCK_TIME_THRESHOLD_S and is_moving:
            if not self.is_stuck:
                self._stuck_type = StuckType.NO_PROGRESS
                logger.warning(f"Stuck: no RSSI progress for {time_no_progress:.1f}s")

        heading_delta = self._heading_delta(heading, self._last_heading)
        if abs(heading_delta) > 30:
            self._direction_changes.append(now)

        if len(self._direction_changes) >= OSCILLATION_THRESHOLD:
            if now - self._direction_changes[0] < 10.0:
                self._stuck_type = StuckType.OSCILLATING
                logger.warning("Stuck: oscillating")

        if abs(heading_delta) > 20 and not rssi_improved:
            self._consecutive_turns += 1
        else:
            self._consecutive_turns = 0

        if self._consecutive_turns >= SPIN_DETECTION_TURNS:
            self._stuck_type = StuckType.SPINNING
            logger.warning("Stuck: spinning")

        self._last_rssi = rssi
        self._last_heading = heading
        self._heading_history.append(heading)

    def record_recovery_attempt(self):
        self._recovery_attempts += 1
        self._stuck_type = StuckType.NONE
        self._last_progress_time = time.monotonic()
        self._consecutive_turns = 0
        self._direction_changes.clear()
        logger.info(f"Recovery attempt #{self._recovery_attempts}")

    def reset(self):
        self._stuck_type = StuckType.NONE
        self._recovery_attempts = 0
        self._last_progress_time = time.monotonic()
        self._last_rssi = -100
        self._heading_history.clear()
        self._direction_changes.clear()
        self._consecutive_turns = 0
        self._last_heading = 0.0

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "is_stuck": self.is_stuck,
            "stuck_type": self._stuck_type.value,
            "recovery_attempts": self._recovery_attempts,
            "should_abort": self.should_abort,
            "time_since_progress_s": round(time.monotonic() - self._last_progress_time, 1),
        }

    @staticmethod
    def _heading_delta(h1: float, h2: float) -> float:
        return (h1 - h2 + 180) % 360 - 180
