"""
NovaCare AI - EmotionAnalyzer
Uses Google Gemini Pro API (REST) for emotion classification.
"""
import numpy as np
from datetime import datetime
from typing import Union, Dict, Any

from ai.config import Config

# Emotion label mapping for console output
EMOTION_LABELS = {
    'joy': '[HAPPY]', 'happy': '[HAPPY]', 'happiness': '[HAPPY]',
    'sadness': '[SAD]', 'sad': '[SAD]',
    'anger': '[ANGRY]', 'angry': '[ANGRY]',
    'fear': '[FEAR]', 'scared': '[FEAR]',
    'surprise': '[SURPRISE]', 'surprised': '[SURPRISE]',
    'disgust': '[DISGUST]', 'disgusted': '[DISGUST]',
    'neutral': '[NEUTRAL]', 'love': '[LOVE]'
}


class EmotionAnalyzer:
    """Emotion Analyzer using Google Gemini API."""
    
    def __init__(self):
        # Extended keyword dictionary for fallback
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'love', 'amazing', 
                    'fantastic', 'awesome', 'good', 'excellent', 'pleased', 'delighted',
                    'glad', 'cheerful', 'thrilled', 'blessed', 'grateful'],
            'sadness': ['sad', 'unhappy', 'depressed', 'down', 'crying', 'miserable',
                        'heartbroken', 'devastated', 'grief', 'sorrow', 'lonely', 'alone',
                        'hopeless', 'upset', 'disappointed', 'terrible', 'hurt'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated',
                      'rage', 'hate', 'pissed', 'outraged', 'livid'],
            'fear': ['scared', 'afraid', 'frightened', 'terrified', 'nervous', 'anxious',
                     'worried', 'panic', 'fear', 'dread', 'stressed', 'uneasy'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected',
                         'wow', 'omg', 'unbelievable', 'stunning'],
            'disgust': ['disgusted', 'gross', 'revolting', 'sick', 'nauseated', 'awful',
                        'yuck', 'ew', 'repulsed'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'regular', 'so-so', 'meh']
        }
        
        if Config.is_configured():
            print(f"[EmotionAnalyzer] OK - Gemini API Ready")
        else:
            print("[EmotionAnalyzer] WARNING - No API key - using fallback")

    def analyze(self, input_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze emotion from text or face image."""
        if isinstance(input_data, str):
            return self.analyze_text(input_data)
        elif isinstance(input_data, np.ndarray):
            return self.analyze_face(input_data)
        return {'emotion': 'unknown', 'confidence': 0, 'source': 'unknown'}

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze emotion from text using Gemini API."""
        result = {
            'text': text,
            'emotion': 'neutral',
            'confidence': 0.5,
            'source': 'text',
            'method': 'keyword',
            'timestamp': datetime.now().isoformat()
        }

        # Try API first
        if Config.is_configured():
            try:
                api_result = self._call_api(text)
                if api_result:
                    result.update(api_result)
                    result['method'] = 'api'
                    self._print_emotion(text, result)
                    return result
            except Exception:
                pass # Silent fallback

        # Fallback to keyword analysis
        result = self._keyword_analysis(text, result)
        self._print_emotion(text, result)
        return result

    def _call_api(self, text: str) -> Dict:
        """Call Gemini API for emotion classification."""
        prompt = f"""
        Classify the emotion of the following text into one of these categories: 
        joy, sadness, anger, fear, surprise, disgust, neutral.
        
        Text: "{text}"
        
        Return only the category name and a confidence score (0-1) separated by a comma.
        Example: joy, 0.95
        """
        
        response_text = Config.generate_content(prompt)
        
        if response_text:
            parts = response_text.split(',')
            if len(parts) >= 1:
                emotion = parts[0].strip().lower()
                confidence = 0.8  # Default if parsing fails
                
                if len(parts) >= 2:
                    try:
                        confidence = float(parts[1].strip())
                    except:
                        pass
                        
                return {
                    'emotion': emotion,
                    'confidence': confidence
                }
            
        return None

    def _keyword_analysis(self, text: str, result: Dict) -> Dict:
        """Fallback keyword-based emotion analysis."""
        text_lower = text.lower()
        
        scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score
        
        if scores:
            best_emotion = max(scores, key=scores.get)
            result['emotion'] = best_emotion
            result['confidence'] = min(0.5 + scores[best_emotion] * 0.15, 0.95)
        
        return result

    def _print_emotion(self, text: str, result: Dict):
        """Print emotion detection result to console."""
        emotion = result.get('emotion', 'unknown')
        confidence = result.get('confidence', 0)
        method = result.get('method', 'unknown')
        label = EMOTION_LABELS.get(emotion, '[UNKNOWN]')
        
        display_text = text[:40] + "..." if len(text) > 40 else text
        
        print(f"")
        print(f"{'='*50}")
        print(f"  EMOTION DETECTED: {label} {emotion.upper()}")
        print(f"  Confidence: {confidence:.0%} (via {method})")
        print(f"  Text: \"{display_text}\"")
        print(f"{'='*50}")
        print(f"")

    def analyze_face(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Face emotion analysis (requires local model)."""
        return {
            'emotion': 'unknown',
            'confidence': 0,
            'source': 'face',
            'error': 'Face model not available',
            'timestamp': datetime.now().isoformat()
        }

    def train(self, dataset_path: str = None, mode: str = 'text', **kwargs) -> bool:
        """Training not needed for Gemini."""
        print("[EmotionAnalyzer] Gemini API - no training needed")
        return True
