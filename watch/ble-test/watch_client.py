"""
HRYFINE Smartwatch BLE Client
Reusable class for connecting and reading data from the watch
"""

import asyncio
from bleak import BleakClient
from typing import Callable, Optional
from watch_protocol import HRYFINEProtocol, WatchData


class HRYFINEWatchClient:
    """
    Client for HRYFINE smartwatch
    
    Usage:
        client = HRYFINEWatchClient("C2:FC:28:B7:1C:1B")
        
        def on_data(data: WatchData):
            print(f"HR: {data.heart_rate}, Steps: {data.steps}")
        
        await client.connect()
        await client.start_monitoring(on_data)
        await asyncio.sleep(60)
        await client.disconnect()
    """
    
    def __init__(self, device_address: str):
        self.device_address = device_address
        self.client = None
        self.is_connected = False
        self.data_callback = None
        self._battery_level = None
    
    async def connect(self) -> bool:
        """Connect to the watch"""
        try:
            self.client = BleakClient(self.device_address)
            await self.client.connect()
            self.is_connected = True
            
            # Read initial battery level
            await self.update_battery()
            
            print(f"✅ Connected to {self.device_address}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the watch"""
        if self.client:
            await self.client.disconnect()
            self.is_connected = False
            print("🛑 Disconnected")
    
    async def update_battery(self):
        """Read battery level from standard GATT characteristic"""
        try:
            battery_data = await self.client.read_gatt_char(
                HRYFINEProtocol.BATTERY_UUID
            )
            self._battery_level = battery_data[0]
        except Exception as e:
            print(f"⚠️  Could not read battery: {e}")
    
    async def start_monitoring(
        self, 
        on_data: Callable[[WatchData], None],
        update_battery_interval: int = 10
    ):
        """
        Start monitoring watch data
        
        Args:
            on_data: Callback function(WatchData) called for each sensor update
            update_battery_interval: Read battery level every N seconds
        """
        if not self.is_connected:
            print("❌ Not connected. Call connect() first.")
            return
        
        self.data_callback = on_data
        
        try:
            # Subscribe to Nordic UART notifications
            await self.client.start_notify(
                HRYFINEProtocol.NORDIC_RX_UUID,
                self._on_notification
            )
            
            print(f"🔔 Listening for data...")
            
            # Periodically update battery level
            battery_counter = 0
            while self.is_connected:
                await asyncio.sleep(1)
                battery_counter += 1
                
                if battery_counter >= update_battery_interval:
                    await self.update_battery()
                    battery_counter = 0
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopped")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            try:
                await self.client.stop_notify(HRYFINEProtocol.NORDIC_RX_UUID)
            except:
                pass
    
    def _on_notification(self, sender, data: bytes):
        """Handle incoming BLE notification"""
        try:
            parsed = HRYFINEProtocol.parse_message(data, self._battery_level)
            
            if self.data_callback:
                self.data_callback(parsed)
        except Exception as e:
            print(f"⚠️  Parse error: {e}")


async def main():
    """Example usage"""
    # CHANGE THIS TO YOUR WATCH ADDRESS
    WATCH_ADDRESS = "C2:FC:28:B7:1C:1B"
    
    client = HRYFINEWatchClient(WATCH_ADDRESS)
    
    # Connect
    if not await client.connect():
        return
    
    # Define callback
    def on_watch_data(data: WatchData):
        print(f"{data.timestamp.strftime('%H:%M:%S')} | {data}")
    
    # Start monitoring (will run until Ctrl+C)
    try:
        await client.start_monitoring(on_watch_data)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    print("=" * 60)
    print("HRYFINE WATCH CLIENT - EXAMPLE")
    print("=" * 60)
    print()
    
    asyncio.run(main())
