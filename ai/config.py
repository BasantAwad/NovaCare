"""
NovaCare AI - Configuration
Central configuration for all AI modules.
"""
import os


class Config:
    """
    Central configuration for NovaCare AI modules.
    """
    
    # API Token - Priority: Environment Variable > Hardcoded Fallback
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB7oilCjUqHK2zX8KBkafoS7hF_1_BSHsU")
    
    # Model configuration
    MODEL_NAME = "gemini-1.5-flash"
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if API token is configured."""
        return bool(cls.GEMINI_API_KEY) and cls.GEMINI_API_KEY.startswith("AIza")
