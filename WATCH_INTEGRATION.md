# Smart Watch Integration for NovaCare Rover

## Overview

This integration enables the NovaCare rover to read real-time health metrics from an HRYFINE smartwatch via Bluetooth Low Energy (BLE), replacing placeholder data with actual user vital signs.

### What's Integrated

✅ **Real Heart Rate**: Live BPM readings from the smartwatch  
✅ **Steps Counter**: Daily activity tracking  
✅ **Battery Level**: Watch battery percentage  
✅ **Auto-Sync**: Continuous background monitoring  
✅ **Fallback Mode**: Simulation mode for testing without hardware  

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│               Frontend (Next.js)                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ /rover                - Home page with real HR          │ │
│  │ /rover/health         - Health check with live metrics  │ │
│  │ useRobotVitals hook   - React hook for vitals polling   │ │
│  │ robot-vitals-api.ts   - API client for robot service    │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               Robot Service (Python Flask)                  │
│  Port 9000                                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ GET /api/vitals/heart-rate    - Latest heart rate      │ │
│  │ GET /api/vitals/current       - All vitals (HR,steps)  │ │
│  │ GET /health                   - Service health + vitals │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ watch_integration.py  - BLE client manager             │ │
│  │   ↓                                                      │ │
│  │   WatchIntegration class                                │ │
│  │   - Thread-safe vital state                             │ │
│  │   - Async BLE connection                                │ │
│  │   - Data parsing & conversion                           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               Watch Integration Scripts                      │
│  /watch/ble-test/                                            │
│  ├── watch_client.py         - Async BLE connection         │
│  ├── watch_protocol.py       - Protocol decoder             │
│  ├── discover_watch.py       - Device discovery             │
│  └── requirements.txt        - BLE dependencies             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               HRYFINE Smartwatch (BLE)                       │
│  Sends:  Heart rate, steps, battery via BLE notifications   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Initialization**
   - Robot service starts at port 9000
   - `watch_integration.py` initializes WatchIntegration manager
   - Connects to HRYFINE watch via BLE (or simulation mode)
   - Starts background monitoring thread

2. **Real-Time Updates**
   - Watch sends BLE notifications every ~1-2 seconds
   - WatchIntegration parses and stores latest vitals
   - Data available via `/api/vitals/*` endpoints

3. **Frontend Polling**
   - React components use `useRobotVitals` hook
   - Hook fetches from `/api/vitals/current` every 2 seconds
   - Heart rate displays in real-time on rover pages

---

## Configuration

### 1. Find Your Watch's BLE Address

First, discover your smartwatch address:

```bash
cd services/robot
python watch_integration.py
```

Or use the discovery script from the watch folder:

```bash
cd watch/ble-test
python discover_watch.py
```

Look for your HRYFINE device address (format: `C2:FC:28:B7:1C:1B`)

### 2. Configure Robot Service

Set environment variables:

```bash
# Watch device address
export WATCH_ADDRESS="C2:FC:28:B7:1C:1B"

# Start in simulation mode (fake data for testing)
export WATCH_SIMULATION="true"

# Or connect to real watch
export WATCH_SIMULATION="false"
```

### 3. Configure Frontend

Update `.env.local`:

```
NEXT_PUBLIC_ROBOT_API_URL=http://localhost:9000
```

---

## Usage

### Starting the System

1. **Start Robot Service** (with watch integration):
   ```bash
   cd services/robot
   python robot_service.py
   ```

   Output:
   ```
   ==================================================
     NovaCare — Robot REST Service
     Listening on 0.0.0.0:9000
   ==================================================

   📱 Initializing watch integration (simulation=true)...
   ✅ Watch monitoring started
   ```

2. **Start Frontend**:
   ```bash
   cd apps/frontend
   npm run dev
   ```

3. **View Real-Time Metrics**:
   - Navigate to `/rover` - See heart rate in quick stats
   - Navigate to `/rover/health` - See full vital signs
   - Heart rate updates every 2 seconds

### API Endpoints

**Get Latest Heart Rate**
```bash
curl http://localhost:9000/api/vitals/heart-rate \
  -H "X-API-Key: novacare-secure-key-2026"
```

Response:
```json
{
  "status": "success",
  "heart_rate": 78,
  "timestamp": "2026-05-24T10:30:45.123456"
}
```

**Get All Current Vitals**
```bash
curl http://localhost:9000/api/vitals/current \
  -H "X-API-Key: novacare-secure-key-2026"
```

Response:
```json
{
  "status": "success",
  "heart_rate": 78,
  "steps": 4521,
  "battery": 85,
  "timestamp": "2026-05-24T10:30:45.123456"
}
```

**Service Health (includes vitals)**
```bash
curl http://localhost:9000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "NovaCare Robot Service",
  "hardware": {
    "camera": true,
    "motion": true,
    "tts": true,
    ...
  },
  "vitals": {
    "heart_rate": 78,
    "steps": 4521,
    "battery": 85,
    "timestamp": "2026-05-24T10:30:45.123456"
  }
}
```

---

## React Hooks & Components

### useRobotVitals Hook

Custom hook for fetching and polling real-time vitals:

```typescript
import { useRobotVitals } from "@/hooks/useRobotVitals";

export function MyComponent() {
  const { 
    vitals,        // RobotVitals | null
    isLoading,     // boolean
    isError,       // boolean
    error,         // string | null
    source,        // "watch" | "dashboard" | "none"
    refetch        // () => Promise<void>
  } = useRobotVitals({
    pollInterval: 2000,        // Poll every 2 seconds
    retryCount: 3,             // Retry 3 times on error
    fallbackToDashboard: true  // Use dashboard API as fallback
  });

  return (
    <div>
      {isLoading ? (
        <p>Loading...</p>
      ) : vitals?.heart_rate ? (
        <p>❤️ {vitals.heart_rate} BPM (from {source})</p>
      ) : (
        <p>No data available</p>
      )}
    </div>
  );
}
```

### API Client Functions

```typescript
import { 
  getHeartRate,
  getRobotVitals,
  getRobotHealth,
  type RobotVitals 
} from "@/lib/robot-vitals-api";

// Get latest heart rate
const hr = await getHeartRate();
console.log(hr.heart_rate); // number or undefined

// Get all vitals
const vitals = await getRobotVitals();
console.log(vitals.steps); // number or undefined

// Get health status
const health = await getRobotHealth();
console.log(health.hardware.camera); // boolean
```

---

## Modes

### Simulation Mode (Default)

Perfect for development and testing:

```bash
export WATCH_SIMULATION="true"
python robot_service.py
```

- ✅ No hardware required
- ✅ Generates realistic heart rate variations (60-100 BPM)
- ✅ Random steps and battery
- ✅ 100% reliable

### Real Hardware Mode

Connect to actual HRYFINE smartwatch:

```bash
export WATCH_ADDRESS="C2:FC:28:B7:1C:1B"
export WATCH_SIMULATION="false"
python robot_service.py
```

- ✅ Live health metrics from watch
- ⚠️ Requires working BLE on your system
- ⚠️ Watch must be paired and in range
- ⚠️ May require platform-specific setup

---

## Troubleshooting

### "Heart rate data not available" (503)

**Causes:**
- Watch not connected
- Watch monitoring thread crashed
- BLE device not found

**Solutions:**
```bash
# Check simulation mode is enabled
export WATCH_SIMULATION="true"

# Or verify watch address
python discover_watch.py

# Check logs for errors
python robot_service.py | grep -i error
```

### Frontend not showing real data

**Check:**
1. Robot service is running on port 9000
2. Environment variable set: `NEXT_PUBLIC_ROBOT_API_URL=http://localhost:9000`
3. Browser console for CORS or fetch errors
4. Check network tab: `/api/vitals/current` should return `{"status": "success", ...}`

### ImportError: bleak module not found

**Solution:**
```bash
cd watch/ble-test
pip install -r requirements.txt
```

### BLE connection failures (Windows)

Windows requires specific Bluetooth drivers. Try:
```bash
# Use simulation mode instead
export WATCH_SIMULATION="true"

# Or install WinRT dependencies
pip install pywin32
python -m pywin32_postinstall -install
```

---

## Files Changed

### Backend (Robot Service)
- ✅ **NEW**: `services/robot/watch_integration.py` - BLE integration manager
- ✅ **MODIFIED**: `services/robot/robot_service.py` - Added vitals endpoints + watch init
- ✅ **MODIFIED**: `services/robot/robot_service.py` docstring - Added vitals API docs

### Frontend (Next.js)
- ✅ **NEW**: `apps/frontend/src/lib/robot-vitals-api.ts` - API client
- ✅ **NEW**: `apps/frontend/src/hooks/useRobotVitals.ts` - React hook
- ✅ **MODIFIED**: `apps/frontend/src/app/rover/page.tsx` - Uses real heart rate
- ✅ **MODIFIED**: `apps/frontend/src/app/rover/health/page.tsx` - Uses real heart rate + live updates

---

## Future Enhancements

- [ ] Database storage for historical vitals
- [ ] Charts/graphs of vital trends
- [ ] Alerts for abnormal readings
- [ ] Multi-watch support
- [ ] Support for other smartwatch brands
- [ ] Export vital data as CSV/PDF
- [ ] Integration with guardian notifications

---

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review logs: `grep -i "watch\|vital" /path/to/logs`
3. Test with simulation mode first
4. Verify BLE device discovery works

---

**Last Updated:** May 24, 2026  
**Integration Version:** 1.0.0
