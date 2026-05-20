"""
State Management Module

Unified robot state system providing single source of truth
for all robot operations.
"""

from .robot_state import (
    RobotState,
    get_robot_state,
    EmotionType,
    RobotMode,
    AudioState,
    HardwareState,
    ServiceState,
    UserContext,
    AnimationState,
    StateObserver,
)

__all__ = [
    "RobotState",
    "get_robot_state",
    "EmotionType",
    "RobotMode",
    "AudioState",
    "HardwareState",
    "ServiceState",
    "UserContext",
    "AnimationState",
    "StateObserver",
]
