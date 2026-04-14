    # NovaCare Authentication & Authorization System

Secure, multi-role authentication backend for the NovaCare platform supporting direct email/password and Google OAuth integrations using Flask and JWT.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Flask](https://img.shields.io/badge/flask-3.0+-red)
![JWT](https://img.shields.io/badge/jwt-auth-green)

## Features

- 🔐 **Secure JWT-based Authentication** (access & refresh token support)
- 👥 **Multi-Role Support** (`rover`, `caregiver`, `doctor`)
- 🔑 **Google Auth Integration** for frictionless sign-in/sign-up flows
- 🛡️ **Role-based Access Control (RBAC)**
- ⚡ **Fast and lightweight** Flask RESTful API
- 🏗️ **Extensible Architecture** currently mocking PostgreSQL with scalable paradigms

## Project Structure

```
auth-backend/
├── app/
│   ├── routes/             # API Router definitions
│   │   ├── __init__.py
│   │   └── auth.py         # Registration, login, profile endpoints
│   ├── utils/              # Helper utilities
│   │   ├── __init__.py
│   │   ├── tokens.py       # JWT creation and validation
│   │   └── password.py     # Werkzeug hash generation & verification
│   ├── __init__.py         # Flask App Factory initialize
│   ├── config.py           # Environment and app configuration
│   └── mock_db.py          # In-memory database representing 3NF structure
├── .env.example            # Environment variables template
├── requirements.txt
├── run.py                  # Server entry point
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Ensure placeholders are replaced. Essential variables:
```
SECRET_KEY=your_super_secret_key_change_in_production
JWT_EXPIRATION_HOURS=24
PORT=5001
CORS_ORIGINS=http://localhost:3000
```

### 3. Start API Server

```bash
python run.py
```

The server will start on port `5001`. 
Health check available at: http://localhost:5001/health

## API Usage

### User Registration (Standard)

```python
import requests

response = requests.post(
    "http://localhost:5001/api/auth/register",
    json={
        "email": "doctor@novacare.com",
        "password": "SecurePassword123!",
        "first_name": "John",
        "last_name": "Doe",
        "role": "doctor"
    }
)

print(response.json())
# {"status": "success", "data": {"token": "eyJhbG...", "user": {...}}}
```

### Google OAuth Login

```python
import requests

response = requests.post(
    "http://localhost:5001/api/auth/login/google",
    json={
        "email": "user@gmail.com",
        "google_id": "11223344556677"
    }
)

print(response.json())
# Resolves active account roles and returns full JWT auth token
```

## API Endpoints

| Endpoint | Method | Description | Roles |
|----------|--------|-------------|-------|
| `/api/auth/register` | POST | Register a new user | Public |
| `/api/auth/login` | POST | Login via Email & Password | Public |
| `/api/auth/login/google`| POST | Sign-in/Sign-up using Google OAuth | Public |
| `/api/auth/me` | GET | Retrieve Current Authenticated User | All Roles |
| `/health` | GET | Check service availability | Public |

## Security Implementation

- **Password Hashing:** `werkzeug.security` (PBKDF2 SHA256)
- **Token Format:** PyJWT (HMAC-SHA256)
- **CORS:** Restricts incoming requests to defined `NEXT_PUBLIC` frontends
- **RBAC:** Roles are mapped deeply to users allowing multi-role capability for single-identities

## Database Architecture (3NF Normalization)

Currently utilizes an in-memory mapped mock structured identically to production SQL standard:
1. `users`: Core identity (Email, Hashes, Google IDs)
2. `user_roles`: Foreign mapped cross-joins linking independent identities to specific roles (`rover`, `doctor`, `caregiver`).

## Troubleshooting

### JWT Auth Fails on Windows/Mac
Ensure `Authorization` headers are passed strictly via `Bearer <TOKEN>` in your frontend clients.

### Connection Refused (Port 5001)
Check if port `5001` is actively bound. On macOS (specifically Monterey and up) port 5000 is used by AirPlay, hence NovaCare uses `5001` by default.

## License

Internal Organization Use Only - NovaCare Project
