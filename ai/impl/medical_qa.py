"""
NovaCare AI - MedicalQA Implementation
Uses Google Gemini Pro API (REST) for medical question answering.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any

from ai.config import Config

# Knowledge base path
AI_DIR = os.path.dirname(os.path.dirname(__file__))
MEDICAL_KB_PATH = os.path.join(AI_DIR, 'data', 'medical_kb.json')


class MedicalQA:
    """Medical QA using Google Gemini API."""
    
    def __init__(self):
        self.knowledge_base = {}
        self._load_knowledge_base()
        
        if Config.is_configured():
            print("[MedicalQA] OK - Gemini API Ready")
        else:
            print("[MedicalQA] WARNING - No API key - using knowledge base")

    def query(self, question: str) -> Dict[str, Any]:
        """Answer a medical question."""
        question_lower = question.lower()
        result = {
            "answer": "", "confidence": 0.0, "source": "NovaCare",
            "is_emergency": False, "timestamp": datetime.now().isoformat()
        }

        # Check emergencies FIRST
        for keyword, response in self.knowledge_base.get("emergency_keywords", {}).items():
            if keyword in question_lower:
                result.update({"is_emergency": True, "answer": response, "confidence": 1.0, "source": "Emergency"})
                return result

        # Try API
        if Config.is_configured():
            try:
                answer = self._call_api(question)
                if answer:
                    result.update({"answer": answer, "confidence": 0.85, "source": "AI Medical Assistant"})
                    return result
            except Exception as e:
                print(f"[MedicalQA] API error: {e}")

        # Knowledge base fallback
        for symptom, info in self.knowledge_base.get("common_symptoms", {}).items():
            if symptom in question_lower:
                result.update({"answer": info, "confidence": 0.7, "source": "Knowledge Base"})
                return result

        result["answer"] = "I'm having trouble connecting to my medical database. For specific concerns, please consult a healthcare professional."
        result["confidence"] = 0.3
        return result

    def _call_api(self, question: str) -> str:
        """Call Gemini API for medical QA."""
        prompt = f"""
        You are an AI medical assistant. Answer the following health question concisely and professionally.
        Disclaimer: Start by stating you are an AI and not a doctor.
        
        Question: {question}
        """
        
        return Config.generate_content(prompt)

    def train(self, **kwargs) -> bool:
        """Training not needed for Gemini."""
        print("[MedicalQA] Gemini API - no training needed")
        return True

    def _load_knowledge_base(self):
        """Load medical knowledge base."""
        if os.path.exists(MEDICAL_KB_PATH):
            with open(MEDICAL_KB_PATH, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        else:
            self._init_default_kb()

    def _init_default_kb(self):
        """Initialize default knowledge base."""
        self.knowledge_base = {
            "emergency_keywords": {
                "heart attack": "🚨 EMERGENCY: Call 911! Symptoms: chest pain, shortness of breath. Chew aspirin if available.",
                "stroke": "🚨 EMERGENCY: Call 911! FAST: Face drooping, Arm weakness, Speech difficulty, Time to call.",
                "choking": "🚨 EMERGENCY: Perform Heimlich maneuver. Call 911 if unconscious.",
                "can't breathe": "🚨 EMERGENCY: Call 911! Stay upright and calm."
            },
            "common_symptoms": {
                "headache": "Rest, stay hydrated, take pain relievers. See doctor if persistent.",
                "fever": "Rest, fluids, acetaminophen. Seek care if >103°F or lasts 3+ days."
            }
        }
        os.makedirs(os.path.dirname(MEDICAL_KB_PATH), exist_ok=True)
        with open(MEDICAL_KB_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2)
