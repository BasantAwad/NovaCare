"""
NovaCare AI - Configuration
Central configuration for all AI modules.
"""
import os
import requests
import json
from typing import Dict, Any, Optional


class Config:
    """
    Central configuration for NovaCare AI modules.
    All API settings and model configurations in one place.
    """
    
    # =================================================================
    # GEMINI API CONFIGURATION
    # =================================================================
    
    # API Token
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    
    # Model configuration
    MODEL_NAME = "gemini-2.0-flash"
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    
    # =================================================================
    # HELPER METHODS
    # =================================================================
    
    @classmethod
    def generate_content(cls, prompt: str) -> Optional[str]:
        """
        Generate content using Gemini Pro via REST API.
        This bypasses SDK version constraints on older Python versions.
        """
        if not cls.is_configured():
            return None
            
        url = f"{cls.API_BASE_URL}/{cls.MODEL_NAME}:generateContent?key={cls.GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                # Parse response structure
                # { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
                if 'candidates' in result and result['candidates']:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        return candidate['content']['parts'][0]['text']
            else:
                print(f"[Gemini API] Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[Gemini API] Request failed: {e}")
            
        return None
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if API token is configured."""
        return bool(cls.GEMINI_API_KEY) and cls.GEMINI_API_KEY.startswith("AIza")
