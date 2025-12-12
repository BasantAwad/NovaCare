# NovaCare Fall Detection System - IoT Dashboard UI

A modern, **dark-themed** professional analytics dashboard for the IoT Fall Detection System. This is a **UI-only** Flask project with no backend logic - designed specifically for the NovaCare assistive rover module using Edge AI and Computer Vision.

## ✨ Features

### 🎯 Landing Page
- **Dark Modern Theme** inspired by professional admin dashboards
- IoT Architecture Overview with 4-layer visualization
- System objectives based on fall detection requirements
- Feature showcase with ≥95% detection accuracy
- Team members display
- Fully responsive design

### 📊 Professional Dashboard
- **Sidebar Navigation** - Fixed left sidebar with menu items
- **Top Navigation Bar** - Search box, notifications, user dropdown
- **KPI Cards** - 4 gradient cards showing key metrics:
  - New Alerts (with percentage change)
  - Device Uptime
  - Camera Health
  - MQTT Status
- **Analytics Charts** (Chart.js):
  - Detection Activity (Line chart with dual datasets)
  - Alert Types (Donut chart with percentages)
- **System Performance** - Progress bars for CPU, Memory, Detection Rate, Network
- **Quick Actions** - Interactive buttons for common tasks
- **Recent Activity** - Timeline of system events

### 📑 Additional Pages
- **Alerts Page** - Table view of all system alerts with severity badges
- **System Health** - Detailed status cards for all components
- **Login Page** - Modern card-based authentication UI

### 🎨 Design Highlights
- **Dark Professional Theme** - Inspired by WowDash and modern SaaS dashboards
- **IoT Architecture Focus** - 4-Layer IoT model visualization
- **Professional Color Palette** - Blue primary with gradient accents
- **Smooth Animations** - Hover effects, transitions, card animations
- **Bootstrap 5** - Responsive grid and components
- **Font Awesome Icons** - Professional icon set
- **Chart.js Integration** - Beautiful interactive charts with dark theme

## 📁 Project Structure

```
IoT-Fall-Detection/
├── run.py                          # Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── start.bat                       # Windows quick start script
└── app/
    ├── templates/
    │   ├── base.html              # Base layout with Bootstrap & Chart.js
    │   ├── index.html             # Landing page with hero section
    │   ├── dashboard.html         # Main analytics dashboard
    │   ├── login.html             # Login page
    │   ├── alerts.html            # Alerts management page
    │   └── health.html            # System health monitoring
    └── static/
        ├── css/
        │   └── styles.css         # Professional dashboard styles
        └── js/
            └── main.js            # Interactive functions & Chart.js setup
```

## 🚀 Quick Start

### Option 1: Quick Start Script (Windows)
```bash
start.bat
```

### Option 2: Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

### Option 3: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python run.py
```

**Then open:** `http://localhost:5000`

## 📱 Available Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Homepage | Landing page with IoT architecture overview and team info |
| `/dashboard` | Dashboard | Main analytics dashboard with charts and KPIs |
| `/alerts` | Alerts | Alert management with severity filtering |
| `/health` | System Health | Detailed component status monitoring |
| `/login` | Login | Authentication page (UI only) |

## 🎨 UI Components

### KPI Cards (Dashboard)
- **New Alerts** - Critical alerts count with trend indicator
- **Device Uptime** - System availability percentage
- **Camera Health** - Camera status indicator
- **MQTT Status** - Broker connection status

### Charts (Chart.js)
1. **Detection Activity Chart** (Line)
   - Dual dataset: Detection Events & Fall Alerts
   - 7-day view with smooth curves
   - Interactive tooltips
   
2. **Alert Types Chart** (Donut)
   - Distribution of alert categories
   - Percentage breakdown
   - Color-coded segments

### Interactive Features
- **Simulate Alert** - Triggers fake fall detection alert
- **Open Stream** - Opens modal for camera feed (placeholder)
- **Refresh Dashboard** - Simulates data refresh with loading state
- **Export Report** - Mock report download functionality

### Sidebar Navigation
- Dashboard (with active state)
- Alerts (with notification badge)
- System Health
- Live Stream
- Settings

## ⚠️ Important: UI-Only Mode

This is a **frontend-only** implementation with:
- ❌ No real backend logic
- ❌ No database connections
- ❌ No MQTT integration
- ❌ No camera streaming
- ❌ No authentication system
- ❌ No API endpoints

All data shown is **dummy/placeholder data** for UI demonstration.

## 🎯 Interactive Features (Frontend Only)

### Working Functions
```javascript
simulateAlert()        // Simulates a fall detection alert
openStreamModal()      // Opens camera stream modal
refreshDashboard()     // Simulates data refresh
refreshHealth()        // Updates health status display
exportReport()         // Mock report download
showToast(msg, type)   // Toast notification system
```

### Login Form
When submitting the login form:
> ⚠️ **UI-only version.** Backend integration not implemented.

## 🔧 Backend Integration Roadmap

When you're ready to add real functionality:

1. **MQTT Service**
   - Connect to MQTT broker
   - Subscribe to alert topics
   - Real-time event streaming

2. **Camera Integration**
   - OpenCV video capture
   - WebRTC streaming
   - Flask-SocketIO for live feed

3. **AI Model Integration**
   - MediaPipe Pose / YOLOv8-Pose
   - Real-time inference
   - Fall detection algorithm

4. **Database Layer**
   - SQLAlchemy ORM
   - Store alerts, users, logs
   - Historical data analytics

5. **Authentication & IAM**
   - Flask-Login for sessions
   - JWT tokens for API
   - Role-based access control

6. **RESTful API**
   - Flask-RESTful
   - JSON endpoints
   - API documentation

## 🎨 Customization

### Colors
Edit `app/static/css/styles.css`:
```css
:root {
    --primary-color: #4e73df;    /* Blue */
    --success-color: #1cc88a;    /* Green */
    --warning-color: #f6c23e;    /* Yellow */
    --danger-color: #e74a3b;     /* Red */
    --dark-bg: #0f1419;          /* Dark Background */
    --dark-card: #1a1d23;        /* Card Background */
}
```

### Charts
Edit `app/static/js/main.js` - Look for `initCharts()` function to customize:
- Dataset values
- Colors
- Labels
- Chart types

## 👥 Development Team

- **Basant Awad** (22101405) - Computer Science
- **Nadira El-Sirafy** (22101377) - Computer Science
- **Noureen Yasser** (22101109) - AI Science
- **Muhammad Mustafa** (22101336) - AI Science
- **Ramez Asaad** (22100506) - AI Science

## 📊 Technologies Used

- **Backend**: Flask 3.0.0
- **Frontend**: HTML5, CSS3, JavaScript (ES6)
- **UI Framework**: Bootstrap 5.3
- **Icons**: Font Awesome 6.4
- **Charts**: Chart.js 4.4
- **Animations**: CSS3 Transitions & Keyframes

## 📄 License

IoT Project - NovaCare Fall Detection System © 2025

---

**Built with ❤️ using Flask, Bootstrap, and Chart.js**

*Ready for backend integration - Connect your MQTT, Camera, and AI services!*
