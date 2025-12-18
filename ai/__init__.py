"""
NovaCare AI Package
Centralized imports following SOLID principles.

Structure:
- ai/interfaces/  - Abstract interfaces (ISP)
- ai/impl/        - Concrete implementations (SRP)
"""

# ==================== INTERFACES ====================
from ai.interfaces import IEmotionAnalyzer, IConversationalAgent, IMedicalQA

# ==================== IMPLEMENTATIONS ====================
from ai.impl import EmotionAnalyzer, ConversationalAI, MedicalQA


# ==================== SINGLETON GETTERS (DIP) ====================
_emotion_analyzer_instance = None
_conversational_ai_instance = None
_medical_qa_instance = None


def get_emotion_analyzer() -> EmotionAnalyzer:
    """Get singleton EmotionAnalyzer instance."""
    global _emotion_analyzer_instance
    if _emotion_analyzer_instance is None:
        _emotion_analyzer_instance = EmotionAnalyzer()
    return _emotion_analyzer_instance


def get_conversational_ai() -> ConversationalAI:
    """Get singleton ConversationalAI instance."""
    global _conversational_ai_instance
    if _conversational_ai_instance is None:
        _conversational_ai_instance = ConversationalAI()
    return _conversational_ai_instance


def get_medical_qa() -> MedicalQA:
    """Get singleton MedicalQA instance."""
    global _medical_qa_instance
    if _medical_qa_instance is None:
        _medical_qa_instance = MedicalQA()
    return _medical_qa_instance


__all__ = [
    # Interfaces
    'IEmotionAnalyzer', 'IConversationalAgent', 'IMedicalQA',
    # Implementations
    'EmotionAnalyzer', 'ConversationalAI', 'MedicalQA',
    # Getters
    'get_emotion_analyzer', 'get_conversational_ai', 'get_medical_qa'
]
