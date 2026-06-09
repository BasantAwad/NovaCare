"""
Real-time sniffer for HRYFINE watch data
Listens to both Nordic UART and Vendor-specific characteristics
"""

import asyncio
from bleak import BleakClient
import struct


# UUIDs from the discovery
NORDIC_UART_RX = "6e400003-b5a3-f393-e0a9-e50e24dcca9f"  # Notify
VENDOR_SPECIFIC_RX = "0000ff01-0000-1000-8000-00805f9b34fb"  # Notify
BATTERY_LEVEL = "00002a19-0000-1000-8000-00805f9b34fb"  # Battery

# Keep track of data for pattern analysis
data_log = []


def format_data(sender, data):
    """Pretty print raw data"""
    hex_str = data.hex().upper()
    bytes_list = ' '.join(f'{b:02X}' for b in data)
    return hex_str, bytes_list


def on_nordic_uart_data(sender, data):
    """Callback for Nordic UART notifications"""
    hex_str, bytes_list = format_data(sender, data)
    print(f"\n📡 NORDIC UART ({len(data)} bytes):")
    print(f"   Hex: {hex_str}")
    print(f"   Bytes: {bytes_list}")
    
    # Try to parse common patterns
    try_parse_patterns(data)
    
    data_log.append(('nordic', data, hex_str))


def on_vendor_specific_data(sender, data):
    """Callback for Vendor-specific notifications"""
    hex_str, bytes_list = format_data(sender, data)
    print(f"\n🔧 VENDOR SPECIFIC ({len(data)} bytes):")
    print(f"   Hex: {hex_str}")
    print(f"   Bytes: {bytes_list}")
    
    try_parse_patterns(data)
    
    data_log.append(('vendor', data, hex_str))


def try_parse_patterns(data):
    """Try to identify common sensor data patterns"""
    if len(data) == 0:
        return
    
    # Try different formats
    print(f"   Possible interpretations:")
    
    # Single byte
    if len(data) >= 1:
        val = data[0]
        print(f"     - As uint8: {val} (range 0-255)")
        if 30 <= val <= 200:
            print(f"       └─ Could be Heart Rate: {val} BPM")
        if 0 <= val <= 100:
            print(f"       └─ Could be Battery: {val}%")
    
    # Two bytes (big-endian)
    if len(data) >= 2:
        val = struct.unpack('>H', data[:2])[0]
        print(f"     - As uint16 (big-endian): {val}")
        if val < 50000:
            print(f"       └─ Could be Steps: {val}")
    
    # Two bytes (little-endian)
    if len(data) >= 2:
        val = struct.unpack('<H', data[:2])[0]
        print(f"     - As uint16 (little-endian): {val}")
        if val < 50000:
            print(f"       └─ Could be Steps: {val}")
    
    # Four bytes (little-endian)
    if len(data) >= 4:
        val = struct.unpack('<I', data[:4])[0]
        print(f"     - As uint32 (little-endian): {val}")


async def main():
    device_address = "C2:FC:28:B7:1C:1B"
    
    print("=" * 70)
    print("HRYFINE WATCH - REAL-TIME DATA SNIFFER")
    print("=" * 70)
    print(f"\nConnecting to {device_address}...")
    print("\n⏱️  LISTENING FOR DATA...")
    print("📝 Instructions:")
    print("   1. Move around to generate step data")
    print("   2. Check your heart rate on the watch")
    print("   3. Watch for data patterns below")
    print("\n(Press Ctrl+C to stop)\n")
    
    async with BleakClient(device_address) as client:
        print("✅ Connected!\n")
        
        try:
            # Enable notifications for both characteristics
            print("🔔 Enabling notifications...")
            
            await client.start_notify(NORDIC_UART_RX, on_nordic_uart_data)
            print(f"   ✓ Nordic UART enabled")
            
            await client.start_notify(VENDOR_SPECIFIC_RX, on_vendor_specific_data)
            print(f"   ✓ Vendor Specific enabled")
            
            # Read battery level for baseline
            battery = await client.read_gatt_char(BATTERY_LEVEL)
            print(f"   ✓ Battery: {battery[0]}%")
            
            print("\n" + "=" * 70)
            print("WAITING FOR DATA...")
            print("=" * 70 + "\n")
            
            # Keep listening
            while True:
                await asyncio.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped listening.")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            try:
                await client.stop_notify(NORDIC_UART_RX)
                await client.stop_notify(VENDOR_SPECIFIC_RX)
            except:
                pass
    
    # Print summary
    if data_log:
        print("\n" + "=" * 70)
        print("DATA SUMMARY")
        print("=" * 70)
        print(f"\nCaptured {len(data_log)} messages:")
        
        nordic_count = sum(1 for t, _, _ in data_log if t == 'nordic')
        vendor_count = sum(1 for t, _, _ in data_log if t == 'vendor')
        
        print(f"  - Nordic UART: {nordic_count} messages")
        print(f"  - Vendor Specific: {vendor_count} messages")
        
        # Save log
        with open('data_capture.txt', 'w') as f:
            for source, data, hex_str in data_log:
                f.write(f"{source.upper()}: {hex_str}\n")
        
        print(f"\n💾 Saved capture to data_capture.txt")


if __name__ == "__main__":
    asyncio.run(main())
