"""
NovaCare Auth Backend — Authentication Middleware

Decorators for JWT validation and role-based access control.
"""
from functools import wraps
from flask import request, jsonify, g
from app.utils.tokens import decode_token


def require_auth(f):
    """Decorator that requires a valid JWT access token.
    Sets g.current_user_id and g.current_roles on success."""

    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ", 1)[1]
        payload = decode_token(token)

        if payload is None:
            return jsonify({"status": "error", "error": "Invalid or expired token"}), 401

        if payload.get("type") != "access":
            return jsonify({"status": "error", "error": "Invalid token type"}), 401

        g.current_user_id = payload["sub"]
        g.current_roles = payload.get("roles", [])
        return f(*args, **kwargs)

    return decorated


def require_role(*allowed_roles: str):
    """Decorator that checks if the current user has at least one of the allowed roles.
    Must be used after @require_auth."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            current_roles = getattr(g, "current_roles", [])
            if not any(role in current_roles for role in allowed_roles):
                return jsonify({"status": "error", "error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator
