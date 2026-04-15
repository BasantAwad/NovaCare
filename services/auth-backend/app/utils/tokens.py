"""
NovaCare Auth Backend — JWT Token Utilities

Generates and validates JWT access / refresh tokens.
"""
import jwt
import uuid
from typing import Optional, List
from datetime import datetime, timedelta, timezone
from app.config import Config


def generate_access_token(user_id: str, roles: List[str]) -> str:
    """Generate a short-lived JWT access token (default 15 min)."""
    payload = {
        "sub": user_id,
        "roles": roles,
        "type": "access",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(seconds=Config.JWT_ACCESS_TOKEN_EXPIRES),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm="HS256")


def generate_refresh_token(user_id: str) -> str:
    """Generate a long-lived JWT refresh token (default 30 days)."""
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(seconds=Config.JWT_REFRESH_TOKEN_EXPIRES),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token. Returns payload or None on failure."""
    try:
        return jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def generate_verification_token() -> str:
    """Generate a random token for email verification or password reset."""
    return str(uuid.uuid4())
