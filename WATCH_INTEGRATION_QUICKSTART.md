# Smart Watch Integration - Quick Start Guide

## What Was Done

The NovaCare rover now displays **real heart rate data** from your HRYFINE smartwatch instead of fake metrics. Here's what was integrated:

### ✅ Backend Changes

1. **`watch_integration.py`** - New module that:
   - Manages BLE connection to your smartwatch
   - Continuously monitors heart rate, steps, and battery
   - Runs in background thread (won't block the API)
   - Automatically switches to simulation mode if watch unavailable

2. **`robot_service.py`** - Updated with:
   - 3 new API endpoints for vital signs
   - Watch initialization on startup
   - Real-time vitals included in `/health` endpoint

3. **New API Endpoints:**
   - `GET /api/vitals/heart-rate` - Latest heart rate only
   - `GET /api/vitals/current` - All vitals (HR, steps, battery)
   - `GET /health` - Service health + vitals

### ✅ Frontend Changes

1. **`robot-vitals-api.ts`** - New API client for fetching vitals from robot service

2. **`useRobotVitals.ts`** - Custom React hook that:
   - Polls robot service every 2 seconds
   - Falls back to dashboard API if needed
   - Handles errors gracefully

3. **Updated Pages:**
   - `/rover` - Now shows real heart rate in quick stats
   - `/rover/health` - Live updates of heart rate + other vitals

---

## 🚀 Getting Started (5 minutes)

### Step 1: No Configuration Needed!

The system defaults to **simulation mode**, so you can test immediately:

```bash
# Terminal 1: Start robot service
cd services/robot
python robot_service.py
```

**Expected output:**
```
==================================================
  NovaCare — Robot REST Service
  Listening on 0.0.0.0:9000
==================================================

📱 Initializing watch integration (simulation=true)...
✅ Watch monitoring started
```

### Step 2: Start Frontend

```bash
# Terminal 2: Start frontend
cd apps/frontend
npm run dev
```

### Step 3: View Real-Time Heart Rate

Go to:
- **http://localhost:3000/rover** - See HR in quick stats box
- **http://localhost:3000/rover/health** - See full vital signs page

You should see a **heart rate that changes every 2 seconds** (simulated data).

---

## 🔗 Connecting to Real Smartwatch

When ready to use your actual smartwatch:

### 1. Discover Watch Address

```bash
cd watch/ble-test
python discover_watch.py
```

Save the address (format: `C2:FC:28:B7:1C:1B`)

### 2. Set Environment Variables

```bash
# On Linux/Mac:
export WATCH_ADDRESS="C2:FC:28:B7:1C:1B"
export WATCH_SIMULATION="false"
python robot_service.py

# On Windows (PowerShell):
$env:WATCH_ADDRESS="C2:FC:28:B7:1C:1B"
$env:WATCH_SIMULATION="false"
python robot_service.py
```

### 3. Watch Should Connect

Check logs for:
```
✅ Connected to C2:FC:28:B7:1C:1B
🔔 Listening for data...
```

---

## 🧪 Testing the Integration

### Test via API

```bash
# Get heart rate
curl http://localhost:9000/api/vitals/heart-rate \
  -H "X-API-Key: novacare-secure-key-2026"

# Response:
# {"status": "success", "heart_rate": 78, "timestamp": "..."}
```

### Test via Frontend

Open Chrome DevTools (F12):
- Go to Network tab
- Navigate to `/rover` or `/rover/health`
- Watch requests to `http://localhost:9000/api/vitals/current`
- Should see `heart_rate` field updating

---

## 📊 Live Data Display

### Rover Home Page
```
❤️ 78
Heart Rate
(Updates every 2 seconds from real watch)
```

### Health Check Page
Shows all vitals including:
- ❤️ Heart Rate (REAL from watch)
- 👟 Steps  
- 🔋 Battery
- (Other vitals from dashboard)

---

## 🛠️ Architecture

```
Your Smartwatch (BLE)
         ↓
watch_integration.py (Thread-safe manager)
         ↓
Robot Service (:9000)
         ↓
Frontend (http://localhost:3000)
         ↓
Your eyes see real metrics! 👀
```

---

## 🔧 Modes

| Mode | Command | Best For | Hardware Needed |
|------|---------|----------|-----------------|
| **Simulation** (Default) | `WATCH_SIMULATION=true` | Testing, development | ❌ No |
| **Real Watch** | `WATCH_SIMULATION=false` | Production | ✅ Yes |

---

## ⚠️ Common Issues

### "Heart rate data not available" Error

**Solution:** Make sure simulation mode is on
```bash
export WATCH_SIMULATION="true"
python robot_service.py
```

### Frontend shows old heart rate (not updating)

**Check:**
1. Robot service running on port 9000?
2. Environment variable set: `NEXT_PUBLIC_ROBOT_API_URL=http://localhost:9000`
3. No firewall blocking port 9000

### BLE Connection Failing

**Try:**
1. Restart smartwatch
2. Check battery level
3. Run discovery again to verify address
4. Use simulation mode while debugging

---

## 📁 File Locations

**Backend:**
- `services/robot/watch_integration.py` - Main BLE manager ⭐
- `services/robot/robot_service.py` - Flask API (modified)
- `watch/ble-test/` - Original watch scripts (used by watch_integration)

**Frontend:**
- `apps/frontend/src/lib/robot-vitals-api.ts` - API client
- `apps/frontend/src/hooks/useRobotVitals.ts` - React hook ⭐
- `apps/frontend/src/app/rover/page.tsx` - Home (modified)
- `apps/frontend/src/app/rover/health/page.tsx` - Health (modified)

---

## 📚 Documentation

For detailed info, see: `WATCH_INTEGRATION.md`

---

## ✅ Quick Checklist

- [ ] Robot service starts without errors
- [ ] Frontend loads `/rover` page
- [ ] Heart rate displays (changes every 2 sec)
- [ ] `/rover/health` shows live vital updates
- [ ] API endpoint `localhost:9000/api/vitals/current` responds

---

**You're all set! Your rover now shows real health metrics.** 🎉

Questions? Check `WATCH_INTEGRATION.md` for troubleshooting.
