# NovaCare Auth Backend - Architecture

## System Overview
The Authentication backend serves as the core identity provider and Role-Based Access Control (RBAC) component of the NovaCare platform. It is a lightweight, high-performance Flask microservice designed around a strictly normalized 3NF database schema.

## Core Architectural Patterns
- **Factory Pattern**: The Flask application is instantiated using an App Factory scheme (`app/__init__.py`), enabling scaleable microservice extensions and clean testing environments.
- **Stateless Authentication**: Purely JSON Web Token (JWT) driven, removing server-side session overhead.
- **Route Encapsulation**: Endpoints (`/api/auth/*`) are encapsulated in Flask Blueprints, cleanly separating authentication logic from health-checks or potential feature updates.
- **Database Controller**: Database interaction (`db_controller.py`) actively abstracts away raw SQL queries into a secure MySQL pool.

## Request Flow
```
Client (Next.js Application)
    │
    ▼ (HTTPS / POST /api/auth/register, /login, /login/google)
    │
Flask App Route Layer (`app.routes.auth`)
    │
    ├── Validates Payload
    ├── Identifies User by Email or Google ID
    │
    ▼
Database Controller (`app.db_controller`)
    │
    ├── Verifies `pbkdf2:sha256` Password Hash (`app.utils.password`)
    ├── Extracts Multi-Roles (`rover`, `caregiver`, `doctor`)
    │
    ▼
Token Generator (`app.utils.tokens`)
    │
    ├── Encodes `user_id`, `email`, and `roles` into JWT Payload
    ├── Signs JWT with `SECRET_KEY`
    │
    ▼ (Return `status`, `data.token`, `data.user`)
Client (Frontend Redux / Context API)
```

## Security Strategy
1. **Password Safety**: Weak MD5/SHA-1 hashing is avoided. Using Werkzeug Base PBKDF2 SHA-256 ensuring high brute-force resistance.
2. **CORS Hardening**: Strict CORS origin limits the endpoints to only be callable by known frontend web clients (`NEXT_PUBLIC_URL`).
3. **No Localized Passwords for OAuth**: When `google_id` is supplied, password generation is safely bypassed and nullified to prevent parallel backdoor breaches on local tokens.

## Deployment Profile
- **Environment**: Containerized Python Environment (typically Docker based in production).
- **WSGI Integration**: Flask's built-in werkzeug server is replaced with Gunicorn/uWSGI for production threading stability.
- **Database Scaling**: Mock structures must migrate into PostgreSQL to allow synchronous multi-replica deployments.
