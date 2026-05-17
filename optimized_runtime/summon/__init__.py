"""
NovaCare Summon Robot Module
============================

Lightweight autonomous navigation system for summoning the robot
to the user's location using BLE RSSI-guided navigation.

Components:
- SummonStateMachine: Manages summon lifecycle states
- SummonController:   Orchestrates navigation, obstacle avoidance, and arrival
- RSSITracker:        BLE signal processing with Kalman filtering
- Bug2Navigator:      Obstacle-aware path following (classical robotics)
- StuckDetector:      Rule-based FSM for detecting stuck/oscillation states

Protocol:
- Mobile sends `summon_request` via WebSocket
- Robot responds with `summon_status` updates at 2Hz
- Mobile can send `summon_cancel` to abort
"""

__version__ = "0.1.0"

from .summon_state import SummonState, SummonStateMachine
from .summon_controller import SummonController
from .protocol import SummonProtocol, SummonMessageType
