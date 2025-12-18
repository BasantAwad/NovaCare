"""
NovaCare AI - MedicalQA Implementation
Implements IMedicalQA interface.
Uses HuggingFace Med_Dataset for training a medical domain model.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any

# Import interface
from ai.interfaces import IMedicalQA

# Model paths
IMPL_DIR = os.path.dirname(__file__)
AI_DIR = os.path.dirname(IMPL_DIR)
MEDICAL_MODEL_PATH = os.path.join(AI_DIR, 'trained_models', 'medical_qa_model')
MEDICAL_KB_PATH = os.path.join(AI_DIR, 'data', 'medical_kb.json')


class MedicalQA:
    """
    Medical QA system implementing IMedicalQA interface.
    Uses fine-tuned model when available, falls back to knowledge base.
    
    SOLID Principles:
    - Single Responsibility: Medical Q&A only
    - Interface Segregation: Implements focused IMedicalQA
    - Dependency Inversion: Depends on abstraction
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.knowledge_base = {}
        self._load_model()
        self._load_knowledge_base()

    # ==================== IMedicalQA IMPLEMENTATION ====================
    
    def query(self, question: str) -> Dict[str, Any]:
        """Answer a medical question."""
        question_lower = question.lower()
        result = {
            "answer": "",
            "confidence": 0.0,
            "source": "NovaCare Medical AI",
            "is_emergency": False,
            "timestamp": datetime.now().isoformat()
        }

        # Check emergency keywords first
        for keyword, response in self.knowledge_base.get("emergency_keywords", {}).items():
            if keyword in question_lower:
                result["is_emergency"] = True
                result["answer"] = response
                result["confidence"] = 1.0
                result["source"] = "Emergency Protocol"
                print(f"[MedicalQA] EMERGENCY: {keyword}")
                return result

        # Try trained model
        if self.model is not None and self.tokenizer is not None:
            try:
                input_text = f"Medical Question: {question}"
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
                
                outputs = self.model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
                
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                result["answer"] = answer
                result["confidence"] = 0.85
                result["source"] = "Fine-tuned Medical Model"
                return result
                
            except Exception as e:
                print(f"[MedicalQA] Model inference error: {e}")

        # Fallback to knowledge base
        for symptom, info in self.knowledge_base.get("common_symptoms", {}).items():
            if symptom in question_lower:
                result["answer"] = info
                result["confidence"] = 0.7
                result["source"] = "Medical Knowledge Base"
                return result

        # Generic fallback
        result["answer"] = "I can provide general health information, but for specific medical concerns, please consult a healthcare professional."
        result["confidence"] = 0.3
        return result

    def train(self, dataset_name: str = "Med-dataset/Med_Dataset", epochs: int = 3, batch_size: int = 8) -> bool:
        """Fine-tune a medical QA model."""
        try:
            from datasets import load_dataset
            from transformers import (
                AutoModelForSeq2SeqLM, 
                AutoTokenizer, 
                Seq2SeqTrainingArguments, 
                Seq2SeqTrainer,
                DataCollatorForSeq2Seq
            )
            import torch

            print(f"[MedicalQA] Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
            
            base_model = "google/flan-t5-small"
            print(f"[MedicalQA] Loading base model: {base_model}")
            
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

            def preprocess(examples):
                questions = examples.get('question', examples.get('input', []))
                answers = examples.get('answer', examples.get('output', []))
                
                inputs = [f"Medical Question: {q}" for q in questions]
                model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding=True)
                
                labels = tokenizer(answers, max_length=256, truncation=True, padding=True)
                model_inputs["labels"] = labels["input_ids"]
                
                return model_inputs

            print("[MedicalQA] Preprocessing...")
            tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

            training_args = Seq2SeqTrainingArguments(
                output_dir=MEDICAL_MODEL_PATH,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                logging_steps=50,
                save_strategy="epoch",
                predict_with_generate=True,
                fp16=torch.cuda.is_available(),
            )

            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized["train"],
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            print("[MedicalQA] Training...")
            trainer.train()
            
            trainer.save_model(MEDICAL_MODEL_PATH)
            tokenizer.save_pretrained(MEDICAL_MODEL_PATH)
            
            self.model = model
            self.tokenizer = tokenizer
            print("[MedicalQA] Training complete!")
            return True

        except Exception as e:
            print(f"[MedicalQA] Training error: {e}")
            return False

    # ==================== PRIVATE METHODS ====================

    def _load_model(self):
        """Load fine-tuned medical QA model."""
        try:
            if os.path.exists(MEDICAL_MODEL_PATH):
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                print(f"[MedicalQA] Loading model...")
                self.tokenizer = AutoTokenizer.from_pretrained(MEDICAL_MODEL_PATH)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(MEDICAL_MODEL_PATH)
                print("[MedicalQA] Model loaded!")
            else:
                print(f"[MedicalQA] No model at {MEDICAL_MODEL_PATH}")
        except Exception as e:
            print(f"[MedicalQA] Load error: {e}")

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
                "heart attack": "EMERGENCY: Call 911 immediately. Symptoms include chest pain, shortness of breath.",
                "stroke": "EMERGENCY: Call 911. Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call.",
                "choking": "EMERGENCY: Perform Heimlich maneuver. Call 911 if unconscious.",
                "can't breathe": "EMERGENCY: If severe, call 911. Check for airway obstruction."
            },
            "common_symptoms": {
                "headache": "Headaches can be caused by tension, dehydration, or illness. Rest and hydrate.",
                "fever": "Fever indicates your body is fighting infection. Rest and stay hydrated.",
                "dizziness": "Dizziness can indicate low blood pressure or dehydration. Sit down and drink water.",
                "fatigue": "Persistent fatigue may indicate sleep issues or stress. Consult a doctor if it persists."
            }
        }
        os.makedirs(os.path.dirname(MEDICAL_KB_PATH), exist_ok=True)
        with open(MEDICAL_KB_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2)
