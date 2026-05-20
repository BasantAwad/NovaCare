# Package initialization - Lazy imports to avoid dependency errors
# Import individual modules as needed instead of all at once

__all__ = [
    'TextEmotionAnalyzer', 
    'AudioEmotionAnalyzer', 
    'FaceEmotionAnalyzer',
    'EmotionAwareSession',
    'TTSEngine',
    'EmotionDashboard',
]

def get_text_analyzer():
    from .text_predictor import TextEmotionAnalyzer
    return TextEmotionAnalyzer

def get_audio_analyzer():
    from .audio_predictor import AudioEmotionAnalyzer
    return AudioEmotionAnalyzer

def get_face_analyzer():
    from .face_predictor import FaceEmotionAnalyzer
    return FaceEmotionAnalyzer

def get_emotion_session():
    from .emotion_session import EmotionAwareSession
    return EmotionAwareSession

def get_tts_engine():
    from .tts_engine import TTSEngine
    return TTSEngine

def get_emotion_dashboard():
    from .emotion_dashboard import EmotionDashboard
    return EmotionDashboard
