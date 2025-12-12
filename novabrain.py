"""
NovaCare - NovaBrain AI Core
Central AI module integrating all capabilities:
- Conversational AI (LLM)
- Emotion Detection (Face & Text)
- Medical Question Answering
- Health Monitoring & Support
"""
import random
import json
from datetime import datetime

# Try importing AI modules (graceful degradation if not available)
try:
    from ai.text_emotion import get_text_analyzer
    TEXT_EMOTION_AVAILABLE = True
except ImportError:
    TEXT_EMOTION_AVAILABLE = False
    print("[NovaBrain] Text emotion analyzer not available")

try:
    from ai.medical_qa import get_medical_qa
    MEDICAL_QA_AVAILABLE = True
except ImportError:
    MEDICAL_QA_AVAILABLE = False
    print("[NovaBrain] Medical QA not available")

try:
    from ai.emotion_detector import get_detector
    FACE_EMOTION_AVAILABLE = True
except ImportError:
    FACE_EMOTION_AVAILABLE = False
    print("[NovaBrain] Face emotion detector not available")


class NovaBrain:
    """
    Main AI Brain for NovaBot.
    Handles conversation, emotion, medical queries, and emotional support.
    """
    
    def __init__(self, model_name="distilgpt2", use_local=True):
        self.use_local = use_local
        self.model_name = model_name
        self.generator = None
        self.history = []
        self.user_context = {}  # Store user emotional state, preferences

        # Initialize sub-modules
        self.text_analyzer = get_text_analyzer() if TEXT_EMOTION_AVAILABLE else None
        self.medical_qa = get_medical_qa() if MEDICAL_QA_AVAILABLE else None
        self.face_detector = get_detector() if FACE_EMOTION_AVAILABLE else None

        # Intent patterns
        self.intent_patterns = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
            'emergency': ['help', 'emergency', 'fallen', 'fall', 'cant breathe', 'heart attack'],
            'medical': ['symptom', 'medication', 'medicine', 'pain', 'headache', 'fever', 'doctor'],
            'emotional': ['sad', 'lonely', 'depressed', 'anxious', 'scared', 'worried', 'happy'],
            'time': ['time', 'date', 'day'],
            'reminder': ['remind', 'reminder', 'medication time', 'pills'],
            'navigation': ['follow', 'come here', 'stop', 'go to']
        }

        if self.use_local:
            self._load_llm()

    def _load_llm(self):
        """Load local LLM for conversational AI"""
        try:
            from transformers import pipeline
            print(f"[NovaBrain] Loading LLM: {self.model_name}...")
            self.generator = pipeline('text-generation', model=self.model_name)
            print("[NovaBrain] LLM loaded successfully")
        except Exception as e:
            print(f"[NovaBrain] LLM loading failed: {e}")
            self.use_local = False

    def detect_intent(self, text: str) -> str:
        """Detect user intent from text"""
        text_lower = text.lower()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent
        return 'general'

    def analyze_emotion_from_text(self, text: str) -> dict:
        """Analyze emotion from text input"""
        if self.text_analyzer:
            return self.text_analyzer.analyze(text)
        return {'emotion': 'neutral', 'confidence': 0.5}

    def analyze_emotion_from_face(self, face_image) -> dict:
        """Analyze emotion from face image"""
        if self.face_detector:
            return self.face_detector.detect(face_image)
        return {'emotion': 'unknown', 'confidence': 0}

    def get_medical_answer(self, question: str) -> dict:
        """Get answer to medical question"""
        if self.medical_qa:
            return self.medical_qa.query(question)
        return {'answer': 'Medical QA not available', 'confidence': 0}

    def generate_emotional_response(self, emotion: str, user_input: str) -> str:
        """Generate empathetic response based on detected emotion"""
        responses = {
            'sad': [
                "I can sense you might be feeling down. I'm here for you.",
                "It's okay to feel sad sometimes. Would you like to talk about it?",
                "I'm here to listen if you need someone to talk to.",
                "Remember, you're not alone. I'm right here with you."
            ],
            'angry': [
                "I can tell you're frustrated. Take a deep breath with me.",
                "It's okay to feel angry. Would you like to share what's bothering you?",
                "Let's take a moment together. I'm here to help."
            ],
            'fear': [
                "I understand you might be worried. You're safe here.",
                "I'm here with you. Can you tell me what's concerning you?",
                "It's okay to feel nervous. Let me help you feel more at ease."
            ],
            'happy': [
                "I'm glad you're feeling good! That's wonderful to see.",
                "Your happiness makes me happy too! What's making you smile?",
                "It's great to see you in good spirits!"
            ],
            'neutral': [
                "I'm here and ready to help. What would you like to do?",
                "How can I assist you today?",
                "I'm listening. What's on your mind?"
            ]
        }
        
        emotion_responses = responses.get(emotion, responses['neutral'])
        return random.choice(emotion_responses)

    def process_input(self, user_input: str, user_id: int = None, face_image=None) -> dict:
        """
        Process user input and return comprehensive response
        :param user_input: User's text input
        :param user_id: User ID for context
        :param face_image: Optional face image for emotion detection
        :return: dict with response, emotion, intent, etc.
        """
        result = {
            'response': '',
            'intent': 'general',
            'text_emotion': None,
            'face_emotion': None,
            'is_emergency': False,
            'medical_response': None,
            'timestamp': datetime.now().isoformat()
        }

        # Detect intent
        intent = self.detect_intent(user_input)
        result['intent'] = intent

        # Analyze emotions
        text_emotion = self.analyze_emotion_from_text(user_input)
        result['text_emotion'] = text_emotion

        if face_image is not None:
            face_emotion = self.analyze_emotion_from_face(face_image)
            result['face_emotion'] = face_emotion

        # Store in history
        self.history.append({
            'role': 'user',
            'content': user_input,
            'emotion': text_emotion.get('emotion'),
            'timestamp': result['timestamp']
        })

        # Generate response based on intent
        response = ""

        if intent == 'emergency':
            result['is_emergency'] = True
            response = "🚨 EMERGENCY DETECTED! I am alerting your guardians and emergency services immediately. Stay calm, help is on the way!"

        elif intent == 'medical':
            medical_result = self.get_medical_answer(user_input)
            result['medical_response'] = medical_result
            if medical_result.get('is_emergency'):
                result['is_emergency'] = True
                response = medical_result['answer']
            else:
                response = medical_result['answer']
                if medical_result['confidence'] < 0.5:
                    response += " For accurate medical advice, please consult a healthcare professional."

        elif intent == 'emotional':
            emotion = text_emotion.get('emotion', 'neutral')
            response = self.generate_emotional_response(emotion, user_input)

        elif intent == 'greeting':
            responses = [
                "Hello! I'm NovaBot, your AI companion. How are you feeling today?",
                "Hi there! It's great to hear from you. How can I help?",
                "Hello! I'm here and ready to assist you with anything you need."
            ]
            response = random.choice(responses)

        elif intent == 'time':
            now = datetime.now()
            response = f"It is currently {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."

        elif intent == 'reminder':
            response = "I can help you with medication reminders. Would you like me to check your medication schedule?"

        elif intent == 'navigation':
            response = "Navigation commands received. In robot mode, I would process: " + user_input

        else:
            # General conversation with emotion-aware response
            emotion = text_emotion.get('emotion', 'neutral')
            
            # Try LLM generation
            if self.use_local and self.generator:
                try:
                    prompt = f"User: {user_input}\nNovaBot (friendly, helpful):"
                    outputs = self.generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
                    response = outputs[0]['generated_text'].replace(prompt, '').strip()
                    if not response:
                        response = self.generate_emotional_response(emotion, user_input)
                except:
                    response = self.generate_emotional_response(emotion, user_input)
            else:
                response = self.generate_emotional_response(emotion, user_input)

        result['response'] = response

        # Store in history
        self.history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })

        return result

    def get_history(self):
        """Get conversation history"""
        return self.history

    def clear_history(self):
        """Clear conversation history"""
        self.history = []


# Singleton
_nova_instance = None

def get_nova():
    global _nova_instance
    if _nova_instance is None:
        _nova_instance = NovaBrain()
    return _nova_instance
