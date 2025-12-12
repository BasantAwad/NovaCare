"""
NovaCare AI - Medical Question Answering Module
Uses Retrieval-Augmented Generation (RAG) pattern for verified medical info.
Future: Fine-tune on medical dataset for better accuracy.
"""
import os
import json
from datetime import datetime

# Placeholder path for medical knowledge base
MEDICAL_KB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'medical_kb.json')

class MedicalQA:
    def __init__(self, kb_path=None):
        self.kb_path = kb_path or MEDICAL_KB_PATH
        self.knowledge_base = {}
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load medical knowledge base from JSON file"""
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                print(f"[MedicalQA] Loaded {len(self.knowledge_base)} entries from KB")
            except Exception as e:
                print(f"[MedicalQA] Error loading KB: {e}")
                self._init_default_kb()
        else:
            print("[MedicalQA] No KB found, initializing default")
            self._init_default_kb()

    def _init_default_kb(self):
        """Initialize with basic medical information"""
        self.knowledge_base = {
            "symptoms": {
                "headache": "A headache can have many causes including tension, dehydration, or illness. If persistent, consult a doctor.",
                "chest_pain": "CRITICAL: Chest pain can indicate a heart attack. If severe, call emergency services immediately.",
                "fever": "Fever is the body's response to infection. Rest, hydrate, and take acetaminophen if needed. Seek help if >103°F.",
                "dizziness": "Dizziness may indicate low blood pressure, dehydration, or inner ear issues. Sit down and stay hydrated.",
                "shortness_of_breath": "Difficulty breathing may be serious. If sudden onset, seek immediate medical attention."
            },
            "medications": {
                "acetaminophen": "Pain reliever and fever reducer. Max 4g/day. Avoid with liver conditions.",
                "ibuprofen": "Anti-inflammatory. Take with food. Not for those with stomach ulcers or kidney issues.",
                "aspirin": "Blood thinner and pain reliever. Do not give to children. May interact with other blood thinners."
            },
            "emergency_signs": [
                "Difficulty breathing",
                "Chest pain or pressure",
                "Sudden confusion",
                "Severe bleeding",
                "Signs of stroke (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 911)"
            ],
            "wellness_tips": {
                "hydration": "Drink 8 glasses (2L) of water daily. More if active or in hot weather.",
                "sleep": "Adults need 7-9 hours of quality sleep per night.",
                "exercise": "150 minutes of moderate exercise weekly is recommended.",
                "mental_health": "Regular social interaction and stress management are vital."
            }
        }
        self._save_knowledge_base()

    def _save_knowledge_base(self):
        """Save knowledge base to file"""
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2)

    def query(self, question: str) -> dict:
        """
        Answer a medical question using the knowledge base
        :param question: User's medical question
        :return: dict with answer, confidence, and source
        """
        question_lower = question.lower()
        response = {
            "answer": "",
            "confidence": 0.0,
            "source": "NovaCare Medical KB",
            "is_emergency": False,
            "timestamp": datetime.now().isoformat()
        }

        # Check for emergency keywords first
        emergency_keywords = ['heart attack', 'stroke', 'cant breathe', "can't breathe", 'unconscious', 'severe bleeding', 'choking']
        for keyword in emergency_keywords:
            if keyword in question_lower:
                response["is_emergency"] = True
                response["answer"] = "🚨 EMERGENCY: This sounds like a medical emergency. Please call emergency services (911) immediately or press the Emergency button."
                response["confidence"] = 1.0
                return response

        # Search symptoms
        for symptom, info in self.knowledge_base.get("symptoms", {}).items():
            if symptom.replace('_', ' ') in question_lower or symptom in question_lower:
                response["answer"] = info
                response["confidence"] = 0.85
                return response

        # Search medications
        for med, info in self.knowledge_base.get("medications", {}).items():
            if med in question_lower:
                response["answer"] = f"About {med.capitalize()}: {info}"
                response["confidence"] = 0.9
                return response

        # Wellness tips
        for topic, tip in self.knowledge_base.get("wellness_tips", {}).items():
            if topic in question_lower:
                response["answer"] = tip
                response["confidence"] = 0.8
                return response

        # Default response
        response["answer"] = "I don't have specific information about that in my medical knowledge base. For medical concerns, please consult a healthcare professional."
        response["confidence"] = 0.3
        return response

    def add_to_kb(self, category: str, key: str, value: str):
        """Add new entry to knowledge base"""
        if category not in self.knowledge_base:
            self.knowledge_base[category] = {}
        self.knowledge_base[category][key] = value
        self._save_knowledge_base()


# Singleton
_medical_qa_instance = None

def get_medical_qa():
    global _medical_qa_instance
    if _medical_qa_instance is None:
        _medical_qa_instance = MedicalQA()
    return _medical_qa_instance
