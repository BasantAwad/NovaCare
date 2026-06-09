"""
Step 2: Listen to characteristic notifications to capture real-time data
Useful for understanding the data format being sent by the watch
"""

import asyncio
from bleak import BleakClient
import struct


async def listen_to_characteristic(device_address, char_uuid):
    """
    Connect and listen to notifications from a specific characteristic
    """
    print(f"📡 Connecting to {device_address}")
    
    async with BleakClient(device_address) as client:
        print(f"✅ Connected! Listening to {char_uuid}")
        print("\nWear/interact with the watch to see data changes...")
        print("(Press Ctrl+C to stop)\n")
        
        def notification_handler(sender, data):
            """Called whenever the characteristic sends data"""
            print(f"📊 Raw data from {sender}:")
            print(f"   Hex: {data.hex()}")
            print(f"   Bytes: {list(data)}")
            
            # Try to parse as common formats
            try:
                # Try as 2-byte big-endian int (common for heart rate)
                value = struct.unpack('>H', data[:2])[0]
                print(f"   As uint16: {value}")
            except:
                pass
            
            print()
        
        try:
            # Enable notifications on this characteristic
            await client.start_notify(char_uuid, notification_handler)
            
            # Keep listening
            while True:
                await asyncio.sleep(1)
        
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            await client.stop_notify(char_uuid)


async def main():
    print("=" * 60)
    print("HRYFINE SMARTWATCH - DATA SNIFFER")
    print("=" * 60)
    print()
    
    device_address = input("Enter watch device address: ").strip()
    char_uuid = input("Enter characteristic UUID to listen to: ").strip()
    
    await listen_to_characteristic(device_address, char_uuid)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped listening.")
