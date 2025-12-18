"""
NovaCare AI - Interfaces Package
Contains all abstract interfaces/protocols for AI modules.
"""
from .emotion_analyzer import IEmotionAnalyzer
from .conversational_agent import IConversationalAgent
from .medical_qa import IMedicalQA

__all__ = ['IEmotionAnalyzer', 'IConversationalAgent', 'IMedicalQA']
