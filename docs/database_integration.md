# NovaCare Database Integration Guide

This document outlines how to interact with the central NovaCare database securely. It acts as a guide for all system admins and maintainers connecting to the centralized remote schema.

## Database Connection Overview

A secure integration controller has been built in the Auth Backend to centrally manage requests to the main MySQL schema:
**Location:** `services/auth-backend/app/db_controller.py`

This controller is a Singleton wrapper that implements connection pooling. It provides an efficient and secure conduit to communicate with the central database and is entirely resilient against SQL injection.

### The Central Schema Profile
- **Host / IP**: `192.168.1.164`
- **Port**: `3306`
- **Database**: `NovaCare_db`

Since the database is actively hosted by the lead admin, we purposely restrict the exposure of direct user credentials within version control (`.env`).

## CRITICAL: Administrator Setup Instructions

Every admin must create their own local connection file to be authenticated. The controller will intentionally fault and cleanly reject connections if the credentials are not provided via your local environment.

1. Navigate to the Auth Backend service directory:
   ```bash
   cd services/auth-backend/
   ```
2. Make a personalized copy of the environment template:
   ```bash
   cp .env.example .env
   ```
3. Open your newly created `.env` file and replace the placeholder fields at the bottom with your specifically assigned administrative username and password:
   ```env
   # Database Configuration
   DB_HOST=192.168.1.164
   DB_PORT=3306
   DB_NAME=NovaCare_db
   
   # REPLACE THE TWO LINES BELOW
   DB_USER=your_admin_username
   DB_PASSWORD=your_admin_password
   ```

## Secure Usage within the Code

All code paths must interact with the database using parameterized queries to maintain a 100% strict shield against SQL injection vectors. All raw methods in the DB controller forcefully accept parameterized injections natively.

To use the database in your Python files, import the `db` singleton from the controller:

```python
from app.db_controller import db

# ✅ CORRECT: Passing dynamic parameters using a tuple keeps everything sanitized
records = db.fetch_all("SELECT * FROM reports WHERE access_level = %s_identifier", (level,))
affected_count = db.execute_query("UPDATE users SET status = %s WHERE id = %s", ("active", user_id))

# ❌ INCORRECT / DANGEROUS: Do NOT use F-strings or manual interpolators! Data breaches occur here.
# db.fetch_all(f"SELECT * FROM reports WHERE access_level='{level}'")
```

### Supported API Methods:
- `db.execute_query(query, params)`: Triggers Data Manipulation functions (`INSERT`, `UPDATE`, `DELETE`) returning exactly how many rows were mapped.
- `db.fetch_all(query, params)`: Executes a comprehensive `SELECT` query, returning a collection of database dictionaries seamlessly.
- `db.fetch_one(query, params)`: Executes targeted `SELECT` statements pulling the single relevant entity returned.

By abstracting this layer, admins can freely perform necessary operations directly without exposing keys, or disrupting the network overhead. 
