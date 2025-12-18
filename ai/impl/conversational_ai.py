"""
NovaCare AI - ConversationalAI Implementation
Uses Google Gemini Pro API (REST) for conversation generation.
"""
from typing import Optional, List, Dict
import random

from ai.config import Config


class ConversationalAI:
    """Conversational AI using Google Gemini REST API."""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        
        if Config.is_configured():
            print("[ConversationalAI] OK - Gemini API Ready")
        else:
            print("[ConversationalAI] WARNING - No API key - using fallback")

    def generate_response(self, user_input: str, emotion: Optional[str] = None) -> str:
        """Generate a response using Gemini API or fallback."""
        if Config.is_configured():
            try:
                # Construct prompt with simplified context
                prompt = self._build_prompt(user_input, emotion)
                
                response_text = Config.generate_content(prompt)
                if response_text:
                    self._update_history(user_input, response_text)
                    return response_text.strip()
                
            except Exception as e:
                print(f"[ConversationalAI] API error: {e}")
        
        return self._fallback_response(user_input, emotion)

    def _build_prompt(self, user_input: str, emotion: Optional[str]) -> str:
        """Build conversation prompt with history."""
        # Limit history to last 2 turns to keep context window manageable
        history_text = ""
        for turn in self.conversation_history[-2:]:
            history_text += f"User: {turn['user']}\nAI: {turn['ai']}\n"
            
        context = ""
        if emotion and emotion != 'neutral':
            context = f"[User is feeling: {emotion}] "
            
        return f"""You are NovaBot, a helpful and empathetic AI healthcare companion. 
        Keep responses concise (under 2 sentences) and supportive.
        
        {history_text}
        User: {context}{user_input}
        AI:"""

    def _update_history(self, user_input: str, ai_response: str):
        """Update conversation history."""
        self.conversation_history.append({"user": user_input, "ai": ai_response})
        # Keep only last 10 interactions
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def train(self, dataset_path: Optional[str] = None, **kwargs) -> bool:
        """API model - no local training needed."""
        print("[ConversationalAI] Gemini API - no training needed")
        return True

    def _fallback_response(self, user_input: str, emotion: Optional[str] = None) -> str:
        """Curated fallback responses."""
        user_lower = user_input.lower()
        
        emotion_responses = {
            'sad': ["I can sense you're feeling down. I'm here for you. Would you like to talk?",
                    "It's okay to feel sad. I'm here to listen without judgment."],
            'happy': ["I can tell you're in good spirits! What's making you happy?",
                      "Your positive energy is wonderful!"],
            'angry': ["I understand you might be frustrated. Take a deep breath with me.",
                      "It's okay to feel angry. Would you like to talk about it?"],
            'fear': ["I'm here with you. You're safe.",
                     "It's natural to feel scared sometimes. I'm right here."]
        }
        
        if emotion and emotion in emotion_responses:
            return random.choice(emotion_responses[emotion])
        
        if any(w in user_lower for w in ['hello', 'hi', 'hey']):
            return "Hello! I'm NovaBot. How can I help you today?"
        
        return "I'm experiencing some connection issues, but I'm still here with you. How are you feeling?"
