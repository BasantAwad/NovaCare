# NovaCare — Frontend

Next.js 14 web application providing three role-based dashboards for the NovaCare assistant rover.

![Next.js](https://img.shields.io/badge/next.js-14-black)
![React](https://img.shields.io/badge/react-18-blue)
![TypeScript](https://img.shields.io/badge/typescript-5-blue)
![Tailwind](https://img.shields.io/badge/tailwindcss-3-06B6D4)

## Dashboards

| Dashboard | Route | Users |
|-----------|-------|-------|
| **Rover** | `/rover/*` | Primary user (elderly / disabled) — simplified, accessible |
| **Guardian** | `/guardian/*` | Caregivers — monitoring, communication, medications |
| **Medical** | `/medical/*` | Doctors — vitals, records, care plans, appointments |

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── auth/               # Login, Signup
│   │   ├── rover/              # Primary user dashboard
│   │   │   ├── talk/           # NovaBot chat interface
│   │   │   ├── health/         # Health overview
│   │   │   ├── medications/    # Medication reminders
│   │   │   ├── emergency/      # Emergency contacts & SOS
│   │   │   ├── entertainment/  # Media & activities
│   │   │   ├── navigate/       # Robot navigation controls
│   │   │   ├── help/           # ASL guide & help
│   │   │   └── settings/       # Rover preferences
│   │   ├── guardian/           # Caregiver dashboard
│   │   │   ├── activity/       # Activity logs
│   │   │   ├── communication/  # Messaging
│   │   │   ├── medications/    # Medication management
│   │   │   └── settings/       # Guardian preferences
│   │   └── medical/            # Doctor dashboard
│   │       ├── vitals/         # Vital signs charts
│   │       ├── records/        # Patient records
│   │       ├── care-plan/      # Care plans
│   │       ├── appointments/   # Scheduling
│   │       ├── medications/    # Prescriptions
│   │       └── settings/       # Medical preferences
│   ├── components/             # Shared UI components
│   │   ├── ASLRecognitionModal.tsx
│   │   ├── EmotionDetectionModal.tsx
│   │   ├── ThemeProvider.tsx
│   │   └── ui/                 # Design system (Button, Card, etc.)
│   ├── lib/                    # API clients & utilities
│   │   ├── asl-api.ts          # ASL Model API client
│   │   ├── emotion-api.ts      # Emotion detection client
│   │   ├── novabot-api.ts      # LLM chat client
│   │   ├── speech.ts           # Browser STT/TTS + Pocket TTS
│   │   └── utils.ts
│   └── types/                  # TypeScript definitions
├── .env.example                # Sample environment variables
├── next.config.mjs
├── tailwind.config.ts
├── tsconfig.json
└── package.json
```

## Quick Start

### Prerequisites

- **Node.js** v18+ and **npm**
- Backend services running (see root `README.md`)

### Setup

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local
# Edit .env.local — at minimum set:
#   NEXT_PUBLIC_NOVABOT_API_URL=http://localhost:5000
```

### Run

```bash
npm run dev
```

Open **http://localhost:3000**.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_NOVABOT_API_URL` | **Yes** | LLM Backend URL (e.g. `http://localhost:5000`) |
| `NEXT_PUBLIC_POCKET_TTS_URL` | No | Direct Pocket TTS URL for voice output |
| `NEXT_PUBLIC_POCKET_TTS_VOICE_URL` | No | Pocket voice model URL |
| `NEXT_PUBLIC_EDGE_TTS_URL` | No | NovaCare edge TTS proxy URL |
| `NEXT_PUBLIC_EDGE_TTS_TIMEOUT_MS` | No | HTTP TTS timeout (default: `60000`) |

**TTS Fallback:** Pocket TTS direct → edge proxy (`POST /api/speak`) → Web Speech (browser). See [`services/edge-tts-proxy/README.md`](../services/edge-tts-proxy/README.md) for full details.

## Service Communication

| Target | Method | Config |
|--------|--------|--------|
| **LLM Backend** | REST (`POST /api/chat`) | `NEXT_PUBLIC_NOVABOT_API_URL` env var |
| **ASL Model API** | REST (`POST /predict`) | Hardcoded `http://localhost:8000` in `lib/asl-api.ts` |
| **Edge TTS Proxy** | REST (`POST /api/speak`) | `NEXT_PUBLIC_EDGE_TTS_URL` env var (optional) |

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Framework** | Next.js 14 (App Router) |
| **Language** | TypeScript + React 18 |
| **Styling** | Tailwind CSS + Framer Motion |
| **Charts** | Recharts |
| **Icons** | Lucide React |
| **State** | React hooks (no external state library) |
