# Lazy imports to allow individual modules to run standalone

# ConversationalAI (requires HuggingFace API key)
try:
    from .conversational_ai import ConversationalAI
except (ImportError, ValueError):
    ConversationalAI = None

# RecognizeEmotion
try:
    from .recognize_emotion import RecognizeEmotion
except ImportError:
    RecognizeEmotion = None

# Emotion Engine (standalone capable)
try:
    from .emotion_engine import (
        EmotionManager,
        FaceGeometryAnalyzer,
        VoiceProcessor,
        EmotionResult,
        get_emotion_manager
    )
except ImportError:
    EmotionManager = None
    FaceGeometryAnalyzer = None
    VoiceProcessor = None
    EmotionResult = None
    get_emotion_manager = None

__all__ = [
    'ConversationalAI',
    'RecognizeEmotion',
    'EmotionManager',
    'FaceGeometryAnalyzer',
    'VoiceProcessor',
    'EmotionResult',
    'get_emotion_manager'
]