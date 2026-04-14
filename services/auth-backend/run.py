"""
NovaCare Auth Backend — Entry Point

Run with: python run.py
"""
from app import create_app
from app.config import Config

app = create_app()

if __name__ == "__main__":
    print(f"🔐 NovaCare Auth Backend starting on port {Config.PORT}")
    print(f"   Health: http://localhost:{Config.PORT}/health")
    print(f"   Auth:   http://localhost:{Config.PORT}/api/auth/")
    print(f"   DB:     In-memory mock (placeholder)")
    print()
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
