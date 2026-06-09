"""
HRYFINE Watch BLE Protocol Decoder
Reverse-engineered protocol format
"""

import struct
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class WatchData:
    """Parsed watch sensor data"""
    heart_rate: Optional[int] = None
    steps: Optional[int] = None
    battery: Optional[int] = None
    timestamp: datetime = None
    raw_message: bytes = None
    
    def __str__(self):
        result = []
        if self.heart_rate is not None:
            result.append(f"❤️  HR: {self.heart_rate} BPM")
        if self.steps is not None:
            result.append(f"👟 Steps: {self.steps}")
        if self.battery is not None:
            result.append(f"🔋 Battery: {self.battery}%")
        return " | ".join(result) if result else "[No data]"


class HRYFINEProtocol:
    """HRYFINE Watch protocol decoder"""
    
    # Standard GATT UUIDs
    BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
    
    # Nordic UART
    NORDIC_RX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9f"
    
    # Message constants
    TELEMETRY_HEADER = bytes([0xDF, 0x00, 0x11])
    
    @staticmethod
    def parse_message(data: bytes, battery_level: Optional[int] = None) -> WatchData:
        """
        Parse a single BLE notification message
        
        Args:
            data: Raw bytes from characteristic notification
            battery_level: Current battery % (from GATT read)
            
        Returns:
            WatchData with parsed sensor values
        """
        result = WatchData(
            timestamp=datetime.now(),
            raw_message=data
        )
        
        # Set battery from standard GATT if provided
        if battery_level is not None:
            result.battery = battery_level
        
        # Single byte: Heart Rate
        if len(data) == 1:
            bpm = data[0]
            # Sanity check: reasonable heart rate is 30-220 BPM
            if 30 <= bpm <= 220:
                result.heart_rate = bpm
            # Low values might be battery percentage
            elif 5 <= bpm <= 100:
                result.battery = bpm
            return result
        
        # 20 bytes: Telemetry packet
        if len(data) == 20 and data[:3] == HRYFINEProtocol.TELEMETRY_HEADER:
            # Byte 12: Steps counter (little-endian)
            steps = struct.unpack('<H', data[12:14])[0]
            result.steps = steps
            
            # Bytes 16-17: Appears to be another counter (possibly distance or calories)
            # Byte 19: Time indicator (increments slowly)
            
            return result
        
        return result
    
    @staticmethod
    def format_telemetry(data: bytes) -> dict:
        """Detailed breakdown of 20-byte telemetry packet"""
        if len(data) < 20 or data[:3] != HRYFINEProtocol.TELEMETRY_HEADER:
            return {}
        
        return {
            'header': data[:3].hex(),
            'byte_3': f'0x{data[3]:02X}',
            'byte_4_11': data[4:12].hex(),
            'steps': struct.unpack('<H', data[12:14])[0],
            'byte_15': f'0x{data[15]:02X}',
            'counter_16_17': struct.unpack('<H', data[16:18])[0],
            'byte_18': f'0x{data[18]:02X}',
            'time_indicator': data[19],
            'raw_hex': data.hex().upper(),
        }


# Test the decoder
if __name__ == "__main__":
    # Test cases from captured data
    test_messages = [
        # Heart rate examples
        bytes.fromhex('61'),  # 97 BPM
        bytes.fromhex('C0'),  # 192 BPM
        bytes.fromhex('74'),  # 116 BPM
        
        # Telemetry examples
        bytes.fromhex('DF00112705010C000C0000003600000020000003'),  # 54 steps
        bytes.fromhex('DF00112E05010C000C0000004400000028000004'),  # 68 steps
        bytes.fromhex('DF0011EC05010C000C000000820000004D000009'),  # 130 steps
    ]
    
    print("=" * 60)
    print("HRYFINE PROTOCOL DECODER TEST")
    print("=" * 60)
    
    for i, msg in enumerate(test_messages):
        print(f"\nMessage {i+1}: {msg.hex().upper()}")
        parsed = HRYFINEProtocol.parse_message(msg, battery_level=36)
        print(f"  Parsed: {parsed}")
        
        if len(msg) == 20:
            details = HRYFINEProtocol.format_telemetry(msg)
            print(f"  Details: Steps={details.get('steps')}, Counter={details.get('counter_16_17')}, Time={details.get('time_indicator')}")
