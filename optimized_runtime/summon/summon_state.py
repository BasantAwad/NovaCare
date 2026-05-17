"""
Summon State Machine
====================

Manages the lifecycle of a summon operation.

States:
  IDLE → REQUESTED → INITIALIZING → SCANNING_RSSI → NAVIGATING → ARRIVING → ARRIVED
                                        ↕                ↕
                                   WALL_FOLLOWING    RECOVERING
                                                         ↓
                                                      FAILED

Transitions are validated — invalid transitions are rejected and logged.
"""

import logging
from enum import Enum
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SummonState(Enum):
    """States in the summon lifecycle."""
    IDLE = "idle"
    REQUESTED = "requested"         # Mobile sent request, awaiting ack
    INITIALIZING = "initializing"   # Robot preparing sensors
    SCANNING_RSSI = "scanning_rssi" # Sampling BLE signal to find direction
    NAVIGATING = "navigating"       # Moving toward user (RSSI gradient)
    WALL_FOLLOWING = "wall_following"  # Bug2: following obstacle boundary
    RECOVERING = "recovering"       # Stuck detected, attempting recovery
    ARRIVING = "arriving"           # Close to user, confirming arrival
    ARRIVED = "arrived"             # Successfully reached user
    FAILED = "failed"               # Could not reach user
    CANCELLED = "cancelled"         # User cancelled summon


# Valid state transitions map
_VALID_TRANSITIONS: Dict[SummonState, List[SummonState]] = {
    SummonState.IDLE: [SummonState.REQUESTED],
    SummonState.REQUESTED: [SummonState.INITIALIZING, SummonState.CANCELLED, SummonState.FAILED],
    SummonState.INITIALIZING: [SummonState.SCANNING_RSSI, SummonState.FAILED, SummonState.CANCELLED],
    SummonState.SCANNING_RSSI: [
        SummonState.NAVIGATING, SummonState.FAILED, SummonState.CANCELLED,
    ],
    SummonState.NAVIGATING: [
        SummonState.WALL_FOLLOWING, SummonState.ARRIVING,
        SummonState.RECOVERING, SummonState.SCANNING_RSSI,
        SummonState.FAILED, SummonState.CANCELLED,
    ],
    SummonState.WALL_FOLLOWING: [
        SummonState.NAVIGATING, SummonState.RECOVERING,
        SummonState.ARRIVING, SummonState.FAILED, SummonState.CANCELLED,
    ],
    SummonState.RECOVERING: [
        SummonState.SCANNING_RSSI, SummonState.NAVIGATING,
        SummonState.FAILED, SummonState.CANCELLED,
    ],
    SummonState.ARRIVING: [
        SummonState.ARRIVED, SummonState.NAVIGATING,
        SummonState.FAILED, SummonState.CANCELLED,
    ],
    # Terminal states — can only reset to IDLE
    SummonState.ARRIVED: [SummonState.IDLE],
    SummonState.FAILED: [SummonState.IDLE],
    SummonState.CANCELLED: [SummonState.IDLE],
}


class SummonStateMachine:
    """
    Manages state transitions for a summon operation.

    Features:
    - Validates transitions against allowed edges
    - Notifies listeners on state changes
    - Tracks transition history for debugging
    - Records timing for each state
    """

    def __init__(self):
        self._state: SummonState = SummonState.IDLE
        self._listeners: List[Callable] = []
        self._history: List[Dict[str, Any]] = []
        self._state_entered_at: datetime = datetime.now()
        self._summon_started_at: Optional[datetime] = None
        self._max_history: int = 50

    @property
    def state(self) -> SummonState:
        """Current state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether a summon is in progress (not idle/terminal)."""
        return self._state not in (
            SummonState.IDLE,
            SummonState.ARRIVED,
            SummonState.FAILED,
            SummonState.CANCELLED,
        )

    @property
    def is_terminal(self) -> bool:
        """Whether the current state is terminal."""
        return self._state in (
            SummonState.ARRIVED,
            SummonState.FAILED,
            SummonState.CANCELLED,
        )

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since summon started (0 if not active)."""
        if self._summon_started_at is None:
            return 0.0
        return (datetime.now() - self._summon_started_at).total_seconds()

    @property
    def state_duration_seconds(self) -> float:
        """Seconds spent in current state."""
        return (datetime.now() - self._state_entered_at).total_seconds()

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Transition history (most recent last)."""
        return list(self._history)

    def transition(self, new_state: SummonState, reason: str = "") -> bool:
        """
        Attempt to transition to a new state.

        Returns True if transition succeeded, False if invalid.
        """
        if new_state == self._state:
            return True  # No-op, already in this state

        allowed = _VALID_TRANSITIONS.get(self._state, [])
        if new_state not in allowed:
            logger.warning(
                f"Invalid summon state transition: {self._state.value} → {new_state.value} "
                f"(allowed: {[s.value for s in allowed]})"
            )
            return False

        old_state = self._state
        now = datetime.now()
        duration = (now - self._state_entered_at).total_seconds()

        # Record history
        self._history.append({
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
            "duration_s": round(duration, 2),
            "timestamp": now.isoformat(),
        })
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Track summon start
        if old_state == SummonState.IDLE and new_state == SummonState.REQUESTED:
            self._summon_started_at = now

        # Update state
        self._state = new_state
        self._state_entered_at = now

        logger.info(
            f"Summon state: {old_state.value} → {new_state.value}"
            + (f" ({reason})" if reason else "")
        )

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(old_state, new_state, reason)
            except Exception as e:
                logger.error(f"State listener error: {e}")

        return True

    def reset(self):
        """Reset to IDLE state (always allowed)."""
        if self._state != SummonState.IDLE:
            old = self._state
            self._state = SummonState.IDLE
            self._state_entered_at = datetime.now()
            self._summon_started_at = None
            logger.info(f"Summon state reset: {old.value} → idle")

    def add_listener(self, callback: Callable):
        """
        Register a state change listener.

        Callback signature: (old_state: SummonState, new_state: SummonState, reason: str)
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        """Remove a state change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def get_snapshot(self) -> Dict[str, Any]:
        """Get current state snapshot for status reporting."""
        return {
            "state": self._state.value,
            "is_active": self.is_active,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "state_duration_seconds": round(self.state_duration_seconds, 1),
            "history_length": len(self._history),
        }
