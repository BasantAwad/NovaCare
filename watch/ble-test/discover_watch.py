"""
Step 1: Discover the HRYFINE watch and list all its BLE services and characteristics
"""

import asyncio
from bleak import BleakScanner, BleakClient


async def discover_devices():
    """Scan for nearby BLE devices and show those that look like the watch"""
    print("🔍 Scanning for BLE devices (5 seconds)...\n")
    
    devices = await BleakScanner.discover(timeout=5.0)
    
    for device in devices:
        name = device.name if device.name else "[Unknown Device]"
        print(f"Device: {name}")
        print(f"  Address: {device.address}")
        print()
    
    return devices


async def explore_services(device_address):
    """Connect to the watch and list all services/characteristics"""
    print(f"\n📡 Connecting to device: {device_address}")
    
    async with BleakClient(device_address) as client:
        print("✅ Connected!")
        print("\n" + "="*60)
        print("SERVICES AND CHARACTERISTICS")
        print("="*60 + "\n")
        
        services = client.services
        
        for service in services:
            print(f"🔹 Service: {service.uuid}")
            print(f"   Description: {service.description}")
            
            for characteristic in service.characteristics:
                print(f"   ├─ Characteristic: {characteristic.uuid}")
                print(f"   │  Description: {characteristic.description}")
                print(f"   │  Properties: {characteristic.properties}")
                
                # Try to read if readable
                if "read" in characteristic.properties:
                    try:
                        value = await client.read_gatt_char(characteristic.uuid)
                        print(f"   │  Value (raw): {value.hex()}")
                        print(f"   │  Value (decoded): {value}")
                    except Exception as e:
                        print(f"   │  Value: [Could not read: {e}]")
                
                print()


async def main():
    """Main discovery flow"""
    print("=" * 60)
    print("HRYFINE SMARTWATCH BLE DISCOVERY")
    print("=" * 60)
    print()
    
    # Step 1: Find devices
    devices = await discover_devices()
    
    if not devices:
        print("❌ No devices found. Make sure your watch is powered on and nearby.")
        return
    
    # Step 2: Let user select a device
    print("\n" + "=" * 60)
    watch_address = None
    
    for i, device in enumerate(devices):
        if device.name and ("HRYFINE" in device.name.upper() or 
                           "SMARTWATCH" in device.name.upper() or
                           "WATCH" in device.name.upper()):
            print(f"✨ Found potential watch: {device.name} ({device.address})")
            watch_address = device.address
    
    if not watch_address:
        print("\n❓ Didn't auto-detect the watch. Enter the device address manually:")
        watch_address = input("Device address: ").strip()
    
    # Step 3: Explore its services
    if watch_address:
        await explore_services(watch_address)
        print("\n💾 Save this information - you'll need the UUIDs!")


if __name__ == "__main__":
    asyncio.run(main())
