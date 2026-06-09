# HRYFINE Smartwatch BLE Integration

Reverse-engineering the HRYFINE smartwatch BLE protocol to extract heart rate, steps, and other fitness data.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Discover Your Watch
First, put your watch into pairing/discovery mode (check the manual).

```bash
python discover_watch.py
```

This will:
- 🔍 Scan for nearby BLE devices
- 📡 Connect to your watch
- 📋 List all services and characteristics with their UUIDs

**Save the output!** You'll need the UUIDs.

### 3. Sniff Real-Time Data
Once you have a characteristic UUID (usually something like `00002a37-0000-1000-8000-00805f9b34fb`):

```bash
python sniff_data.py
```

Enter:
- Device address (from step 2)
- Characteristic UUID (from step 2)

Move around, take some steps, check your heart rate. You should see raw data being printed!

### 4. Reverse Engineer the Format
Common formats:
- **Heart Rate**: Usually 1-2 bytes, 0-255 BPM
- **Steps**: Usually 2-4 bytes, big-endian integer
- **Battery**: Often 0-100 percentage

Document the format for each UUID.

## Next Steps

Once you understand the data format:
1. Create `watch_protocol.py` - Document the protocol
2. Create `watch_client.py` - Reusable class to read data
3. Integrate into your real project

## Useful Resources
- [BLE/GATT Overview](https://www.bluetooth.com/specifications/specs/)
- [Bleak Library Docs](https://bleak.readthedocs.io/)
- Common GATT UUIDs cheat sheet (see below)

### Common GATT Service UUIDs
- Heart Rate Service: `180d`
- Device Information: `180a`
- Battery Service: `180f`
- Cycling Speed & Cadence: `1816`

### Common GATT Characteristic UUIDs
- Heart Rate Measurement: `2a37`
- Body Sensor Location: `2a38`
- Battery Level: `2a19`
- Steps: `2a35` (usually)

---

## Example Finding

Once you've used `discover_watch.py` and `sniff_data.py`, you might find something like:

```
🔹 Service: 180d (Heart Rate Service)
   ├─ Characteristic: 2a37 (Heart Rate Measurement)
   │  Properties: ['notify']
   │  Raw data: 0x44  (68 BPM)
```

This means:
- The watch uses standard Bluetooth Heart Rate Service
- To read heart rate, subscribe to notifications on `2a37`
- Data is 1 byte: the BPM value
