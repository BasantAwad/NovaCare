"""
NovaCare AI Package
Clean, simple imports for AI modules.
"""

from __future__ import annotations

import os
from typing import Any, Union

from ai.impl import EmotionAnalyzer, MedicalQA
from ai.impl.remote_llm_client import FlaskLlmConversationalClient

# Singleton getters
_emotion_analyzer = None
_conversational_ai: Any = None
_medical_qa = None


def __getattr__(name: str):
    """Lazy import for Gemini ConversationalAI (optional dependency)."""
    if name == "ConversationalAI":
        from ai.impl.conversational_ai import ConversationalAI

        return ConversationalAI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_emotion_analyzer() -> EmotionAnalyzer:
    """Get singleton EmotionAnalyzer."""
    global _emotion_analyzer
    if _emotion_analyzer is None:
        _emotion_analyzer = EmotionAnalyzer()
    return _emotion_analyzer


def get_conversational_ai() -> Union[FlaskLlmConversationalClient, Any]:
    """
    NovaBrain chat backend.

    Default: HTTP client to Flask LLM Backend (``NOVABOT_LLM_API_URL`` or ``NEXT_PUBLIC_NOVABOT_API_URL``).

    Set ``NOVABOT_LLM_USE_GEMINI=true`` to use in-process Gemini (requires ``google-genai``).
    """
    global _conversational_ai
    if _conversational_ai is None:
        use_gemini = os.getenv("NOVABOT_LLM_USE_GEMINI", "").lower() in ("1", "true", "yes")
        if use_gemini:
            from ai.impl.conversational_ai import ConversationalAI

            _conversational_ai = ConversationalAI()
        else:
            _conversational_ai = FlaskLlmConversationalClient()
    return _conversational_ai


def get_medical_qa() -> MedicalQA:
    """Get singleton MedicalQA."""
    global _medical_qa
    if _medical_qa is None:
        _medical_qa = MedicalQA()
    return _medical_qa


__all__ = [
    "EmotionAnalyzer",
    "ConversationalAI",
    "MedicalQA",
    "FlaskLlmConversationalClient",
    "get_emotion_analyzer",
    "get_conversational_ai",
    "get_medical_qa",
]
