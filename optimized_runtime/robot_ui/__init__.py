"""
Robot UI Package

Lightweight, fullscreen animated robot interface for SERBot.

Components:
- Animated eyes with blinking and emotion
- Audio level visualization
- Status indicators
- Real-time state synchronization via WebSocket
"""

from .config import UIConfig, CONFIG_TEMPLATE

__all__ = ["UIConfig", "CONFIG_TEMPLATE"]
