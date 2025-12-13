"""
NovaCare AI - Medical Question Answering with Fine-Tuned Model
Uses HuggingFace Med_Dataset for training a medical domain model.
"""
import os
import json
from datetime import datetime

# Model paths
MEDICAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_models', 'medical_qa_model')
MEDICAL_KB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'medical_kb.json')

class MedicalQA:
    """
    Medical Question Answering System
    - Uses fine-tuned model when available
    - Falls back to knowledge base for emergency/common queries
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.knowledge_base = {}
        self._load_model()
        self._load_knowledge_base()

    def _load_model(self):
        """Load fine-tuned medical QA model"""
        try:
            if os.path.exists(MEDICAL_MODEL_PATH):
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                print(f"[MedicalQA] Loading fine-tuned model from {MEDICAL_MODEL_PATH}...")
                self.tokenizer = AutoTokenizer.from_pretrained(MEDICAL_MODEL_PATH)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(MEDICAL_MODEL_PATH)
                print("[MedicalQA] Model loaded successfully!")
            else:
                print(f"[MedicalQA] No trained model at {MEDICAL_MODEL_PATH}. Run train_medical_qa.py first.")
        except Exception as e:
            print(f"[MedicalQA] Model loading error: {e}")

    def _load_knowledge_base(self):
        """Load medical knowledge base for fallback"""
        if os.path.exists(MEDICAL_KB_PATH):
            with open(MEDICAL_KB_PATH, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        else:
            self._init_default_kb()

    def _init_default_kb(self):
        """Initialize default emergency/critical knowledge base"""
        self.knowledge_base = {
            "emergency_keywords": {
                "heart attack": "EMERGENCY: Call 911 immediately. Symptoms include chest pain, shortness of breath.",
                "stroke": "EMERGENCY: Call 911. Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call.",
                "choking": "EMERGENCY: Perform Heimlich maneuver. Call 911 if person becomes unconscious.",
                "severe bleeding": "EMERGENCY: Apply pressure to wound, elevate if possible. Call 911.",
                "can't breathe": "EMERGENCY: If severe, call 911. Check for airway obstruction."
            },
            "common_symptoms": {
                "headache": "Headaches can be caused by tension, dehydration, or illness. Rest, hydrate, and take pain relievers if needed. Consult doctor if persistent.",
                "fever": "Fever indicates your body is fighting infection. Rest, stay hydrated, and take acetaminophen if needed. Seek medical attention if temperature exceeds 103°F.",
                "dizziness": "Dizziness can indicate low blood pressure, dehydration, or inner ear issues. Sit down, drink water, and avoid sudden movements.",
                "fatigue": "Persistent fatigue may indicate sleep issues, stress, or underlying conditions. Ensure adequate sleep and consult a doctor if it persists.",
                "nausea": "Nausea can be caused by food, motion, or illness. Stay hydrated with small sips. Seek help if accompanied by severe pain."
            }
        }
        os.makedirs(os.path.dirname(MEDICAL_KB_PATH), exist_ok=True)
        with open(MEDICAL_KB_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2)

    def train(self, dataset_name="Med-dataset/Med_Dataset", epochs=3, batch_size=8):
        """
        Fine-tune a medical QA model using HuggingFace datasets
        :param dataset_name: HuggingFace dataset name
        :param epochs: Number of training epochs
        :param batch_size: Training batch size
        """
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
            
            # Use a pretrained medical/biomedical model as base
            base_model = "google/flan-t5-small"  # Small but capable
            print(f"[MedicalQA] Loading base model: {base_model}")
            
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

            # Preprocess function
            def preprocess(examples):
                # Assuming dataset has 'question' and 'answer' columns
                # Adapt based on actual dataset structure
                questions = examples.get('question', examples.get('input', []))
                answers = examples.get('answer', examples.get('output', []))
                
                inputs = [f"Medical Question: {q}" for q in questions]
                model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding=True)
                
                labels = tokenizer(answers, max_length=256, truncation=True, padding=True)
                model_inputs["labels"] = labels["input_ids"]
                
                return model_inputs

            print("[MedicalQA] Preprocessing dataset...")
            tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

            # Training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=MEDICAL_MODEL_PATH,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                logging_steps=50,
                save_strategy="epoch",
                evaluation_strategy="epoch" if "validation" in tokenized else "no",
                predict_with_generate=True,
                fp16=torch.cuda.is_available(),
            )

            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized["train"],
                eval_dataset=tokenized.get("validation", tokenized.get("test")),
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            print("[MedicalQA] Starting training...")
            trainer.train()

            # Save
            print(f"[MedicalQA] Saving model to {MEDICAL_MODEL_PATH}")
            trainer.save_model(MEDICAL_MODEL_PATH)
            tokenizer.save_pretrained(MEDICAL_MODEL_PATH)
            
            self.model = model
            self.tokenizer = tokenizer
            print("[MedicalQA] Training complete!")
            
            return True

        except Exception as e:
            print(f"[MedicalQA] Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def query(self, question: str) -> dict:
        """
        Answer a medical question
        :param question: User's medical question
        :return: dict with answer, confidence, source
        """
        question_lower = question.lower()
        result = {
            "answer": "",
            "confidence": 0.0,
            "source": "NovaCare Medical AI",
            "is_emergency": False,
            "timestamp": datetime.now().isoformat()
        }

        # Check emergency keywords first (always use KB for safety)
        for keyword, response in self.knowledge_base.get("emergency_keywords", {}).items():
            if keyword in question_lower:
                result["is_emergency"] = True
                result["answer"] = response
                result["confidence"] = 1.0
                result["source"] = "Emergency Protocol"
                print(f"[MedicalQA] EMERGENCY query detected: {keyword}")
                return result

        # Try trained model
        if self.model is not None and self.tokenizer is not None:
            try:
                input_text = f"Medical Question: {question}"
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=4,
                    early_stopping=True
                )
                
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                result["answer"] = answer
                result["confidence"] = 0.85
                result["source"] = "Fine-tuned Medical Model"
                print(f"[MedicalQA] Model response: {answer[:50]}...")
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
        result["answer"] = "I can provide general health information, but for specific medical concerns, please consult a healthcare professional. Would you like me to help you find a doctor or medical resource?"
        result["confidence"] = 0.3
        return result


# Singleton
_medical_qa_instance = None

def get_medical_qa():
    global _medical_qa_instance
    if _medical_qa_instance is None:
        _medical_qa_instance = MedicalQA()
    return _medical_qa_instance


if __name__ == "__main__":
    # Test training
    qa = MedicalQA()
    print("\nTo train the model, run: python train_medical_qa.py")
    
    # Test query
    test_questions = [
        "What should I do for a headache?",
        "I think I'm having a heart attack",
        "What is diabetes?"
    ]
    
    for q in test_questions:
        result = qa.query(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer'][:100]}...")
        print(f"Confidence: {result['confidence']}")
