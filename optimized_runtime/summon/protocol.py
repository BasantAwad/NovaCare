"""
Summon Protocol — WebSocket Message Definitions
================================================

Defines the message contract between mobile app and robot
for the Summon Robot feature.

All messages are JSON-encoded and flow over the existing
WebSocketServer on port 9999.

Mobile → Robot:
  - summon_request:  Initiate summon (includes phone BLE MAC)
  - summon_cancel:   Cancel active summon
  - rssi_update:     Phone-side RSSI reading (if available)

Robot → Mobile:
  - summon_status:   Periodic navigation status update (2Hz)
  - summon_arrived:  Robot has reached the user
  - summon_failed:   Robot could not reach the user
  - summon_ack:      Acknowledgment of request/cancel
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from datetime import datetime


class SummonMessageType(Enum):
    """WebSocket message types for summon protocol."""

    # Mobile → Robot
    SUMMON_REQUEST = "summon_request"
    SUMMON_CANCEL = "summon_cancel"
    RSSI_UPDATE = "rssi_update"

    # Robot → Mobile
    SUMMON_STATUS = "summon_status"
    SUMMON_ARRIVED = "summon_arrived"
    SUMMON_FAILED = "summon_failed"
    SUMMON_ACK = "summon_ack"


@dataclass
class SummonRequest:
    """Payload for summon_request message."""
    user_id: str
    ble_mac: str                   # Phone's BLE MAC address for RSSI tracking
    ble_service_uuid: str = ""     # Optional: specific BLE service UUID
    priority: int = 5              # 1-10, higher = more urgent
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SummonRequest":
        return SummonRequest(
            user_id=data.get("user_id", ""),
            ble_mac=data.get("ble_mac", ""),
            ble_service_uuid=data.get("ble_service_uuid", ""),
            priority=data.get("priority", 5),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class SummonStatus:
    """Payload for summon_status message (robot → mobile, 2Hz)."""
    state: str                      # Current SummonState value
    rssi_current: int = 0           # Latest RSSI reading (dBm)
    rssi_trend: str = "unknown"     # "improving", "degrading", "stable", "unknown"
    distance_estimate_m: float = -1 # Estimated distance in meters (-1 = unknown)
    obstacle_detected: bool = False
    is_wall_following: bool = False
    heading_deg: float = 0.0        # Current heading in degrees
    elapsed_seconds: float = 0.0
    recovery_attempts: int = 0
    message: str = ""               # Human-readable status
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SummonStatus":
        return SummonStatus(**{
            k: data[k] for k in data
            if k in SummonStatus.__dataclass_fields__
        })


@dataclass
class SummonResult:
    """Payload for summon_arrived or summon_failed messages."""
    success: bool
    reason: str = ""
    total_time_seconds: float = 0.0
    total_distance_estimate_m: float = 0.0
    recovery_attempts: int = 0
    final_rssi: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SummonProtocol:
    """
    Helper for constructing and parsing summon protocol messages.

    Usage:
        # Build a message
        msg = SummonProtocol.build_status(state="navigating", rssi=-65)

        # Parse incoming
        msg_type, payload = SummonProtocol.parse(raw_dict)
    """

    @staticmethod
    def build_request(user_id: str, ble_mac: str, **kwargs) -> Dict[str, Any]:
        """Build a summon_request message."""
        req = SummonRequest(user_id=user_id, ble_mac=ble_mac, **kwargs)
        return {
            "type": SummonMessageType.SUMMON_REQUEST.value,
            "payload": req.to_dict(),
        }

    @staticmethod
    def build_cancel(user_id: str, reason: str = "user_cancelled") -> Dict[str, Any]:
        """Build a summon_cancel message."""
        return {
            "type": SummonMessageType.SUMMON_CANCEL.value,
            "payload": {
                "user_id": user_id,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            },
        }

    @staticmethod
    def build_rssi_update(rssi: int, ble_mac: str = "") -> Dict[str, Any]:
        """Build an rssi_update message from the phone."""
        return {
            "type": SummonMessageType.RSSI_UPDATE.value,
            "payload": {
                "rssi": rssi,
                "ble_mac": ble_mac,
                "timestamp": datetime.now().isoformat(),
            },
        }

    @staticmethod
    def build_status(status: SummonStatus) -> Dict[str, Any]:
        """Build a summon_status message."""
        return {
            "type": SummonMessageType.SUMMON_STATUS.value,
            "payload": status.to_dict(),
        }

    @staticmethod
    def build_arrived(result: SummonResult) -> Dict[str, Any]:
        """Build a summon_arrived message."""
        return {
            "type": SummonMessageType.SUMMON_ARRIVED.value,
            "payload": result.to_dict(),
        }

    @staticmethod
    def build_failed(result: SummonResult) -> Dict[str, Any]:
        """Build a summon_failed message."""
        return {
            "type": SummonMessageType.SUMMON_FAILED.value,
            "payload": result.to_dict(),
        }

    @staticmethod
    def build_ack(
        original_type: str, success: bool = True, message: str = ""
    ) -> Dict[str, Any]:
        """Build a summon_ack message."""
        return {
            "type": SummonMessageType.SUMMON_ACK.value,
            "payload": {
                "ack_for": original_type,
                "success": success,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            },
        }

    @staticmethod
    def parse(message: Dict[str, Any]):
        """
        Parse a raw WebSocket message dict.

        Returns:
            (SummonMessageType, payload_dict) or (None, None) if not a summon message.
        """
        msg_type_str = message.get("type", "")
        try:
            msg_type = SummonMessageType(msg_type_str)
        except ValueError:
            return None, None

        payload = message.get("payload", {})
        return msg_type, payload
