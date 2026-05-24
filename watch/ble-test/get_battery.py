"""
Simple battery level reader for HRYFINE watch
"""

import asyncio
from bleak import BleakClient


async def get_battery(device_address: str = "C2:FC:28:B7:1C:1B") -> int:
    """Read battery level from the watch"""
    
    BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
    
    try:
        async with BleakClient(device_address) as client:
            print(f"🔗 Connecting to {device_address}...")
            
            battery_data = await client.read_gatt_char(BATTERY_UUID)
            battery_level = battery_data[0]
            
            print(f"✅ Connected!")
            print(f"🔋 Battery: {battery_level}%")
            
            return battery_level
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(get_battery())
