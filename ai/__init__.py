"""
NovaCare AI Package
Centralized imports for all AI modules.
"""

# Emotion Detection (Face)
try:
    from .emotion_detector import EmotionDetector, get_detector
except ImportError as e:
    print(f"[AI] Emotion detector unavailable: {e}")
    EmotionDetector = None
    get_detector = None

# Text Emotion Analysis
try:
    from .text_emotion import TextEmotionAnalyzer, get_text_analyzer
except ImportError as e:
    print(f"[AI] Text emotion unavailable: {e}")
    TextEmotionAnalyzer = None
    get_text_analyzer = None

# Medical QA
try:
    from .medical_qa import MedicalQA, get_medical_qa
except ImportError as e:
    print(f"[AI] Medical QA unavailable: {e}")
    MedicalQA = None
    get_medical_qa = None

# Conversational AI
try:
    from .conversational_ai import ConversationalAI, get_conversational_ai
except ImportError as e:
    print(f"[AI] Conversational AI unavailable: {e}")
    ConversationalAI = None
    get_conversational_ai = None

__all__ = [
    'EmotionDetector', 'get_detector',
    'TextEmotionAnalyzer', 'get_text_analyzer', 
    'MedicalQA', 'get_medical_qa',
    'ConversationalAI', 'get_conversational_ai'
]
