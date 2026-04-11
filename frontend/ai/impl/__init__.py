"""
NovaCare AI - Implementations Package
Contains concrete implementations of AI interfaces.
"""
from .emotion_analyzer import EmotionAnalyzer
from .medical_qa import MedicalQA
from .remote_llm_client import FlaskLlmConversationalClient

# ConversationalAI (Gemini) is imported lazily via ai.__getattr__ / get_conversational_ai
# so environments without google-genai can use FlaskLlmConversationalClient only.

__all__ = ['EmotionAnalyzer', 'MedicalQA', 'FlaskLlmConversationalClient']
