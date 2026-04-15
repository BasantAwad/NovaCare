"""
NovaCare Auth Backend — Flask Application Factory
"""
from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS for the Next.js frontend (must allow all methods + headers for preflight)
    CORS(
        app,
        origins=["http://localhost:3000"],
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "Accept"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    )

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)

    # Health check
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "success",
            "data": {
                "service": "auth-backend",
                "version": "1.0.0",
                "database": "live (192.168.1.164)",
            },
        }), 200

    return app
