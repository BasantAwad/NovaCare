"""
NovaCare Auth Backend — Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[3] / '.env'
load_dotenv(env_path)


class Config:
    """Application configuration loaded from environment variables."""

    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "novacare-dev-secret-key-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 900))  # 15 min
    JWT_REFRESH_TOKEN_EXPIRES = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES", 2592000))  # 30 days
    PORT = int(os.getenv("PORT", 5001))
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
