"""
Robot UI Configuration and Setup

Lightweight, fullscreen React/Electron UI for SERBot
Displays animated eyes, emotions, status, and interactions
"""

import json
from pathlib import Path
from typing import Dict, Any

CONFIG_TEMPLATE = {
    "robot": {
        "name": "SERBot",
        "version": "2.0",
    },
    "ui": {
        "fullscreen": True,
        "resolution": "1280x720",
        "fps": 60,
        "theme": "dark",
    },
    "animation": {
        "blink_rate": 5.0,  # blinks per minute
        "blink_duration": 150,  # ms
        "animation_speed": 1.0,
    },
    "websocket": {
        "url": "ws://localhost:9999",
        "reconnect_interval": 1000,  # ms
        "max_reconnect_attempts": 10,
    },
    "features": {
        "animated_eyes": True,
        "emotion_display": True,
        "audio_visualization": True,
        "status_panel": True,
        "gesture_animation": True,
    }
}


class UIConfig:
    """Robot UI Configuration Manager"""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else Path("config.json")
        self.config = CONFIG_TEMPLATE.copy()
        
        if self.config_path.exists():
            self.load()
    
    def load(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config.update(json.load(f))
        except Exception as e:
            print(f"Failed to load config: {e}")
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set config value by dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
