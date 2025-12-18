"""
NovaCare AI - ConversationalAI Implementation
Implements IConversationalAgent interface.
Fine-tunable LLM for emotional support, companionship, and context-aware dialogue.
"""
import os
import json
from typing import Optional

# Import interface
from ai.interfaces import IConversationalAgent

# Model paths
IMPL_DIR = os.path.dirname(__file__)
AI_DIR = os.path.dirname(IMPL_DIR)
CONVERSATION_MODEL_PATH = os.path.join(AI_DIR, 'trained_models', 'conversation_model')


class ConversationalAI:
    """
    Conversational AI implementing IConversationalAgent interface.
    Provides emotional support, companionship, and context-aware dialogue.
    
    SOLID Principles:
    - Single Responsibility: Handles conversation only
    - Interface Segregation: Implements focused IConversationalAgent
    - Dependency Inversion: Depends on abstraction
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model = None
        self.tokenizer = None
        self.base_model_name = model_name
        self.conversation_history = []
        self._load_model()

    # ==================== IConversationalAgent IMPLEMENTATION ====================
    
    def generate_response(self, user_input: str, emotion: Optional[str] = None) -> str:
        """Generate a response to user input."""
        if self.model is None:
            return self._fallback_response(user_input, emotion)

        try:
            # Build conversation context
            context = ""
            
            if emotion and emotion not in ['neutral', 'unknown']:
                context += f"[User seems {emotion}] "
            
            for msg in self.conversation_history[-3:]:
                context += f"{msg['role']}: {msg['content']}\n"
            
            context += f"User: {user_input}\nNovaBot:"

            # Tokenize and generate
            inputs = self.tokenizer.encode(context, return_tensors="pt", max_length=256, truncation=True)
            attention_mask = (inputs != self.tokenizer.pad_token_id).long()
            
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "NovaBot:" in response:
                response = response.split("NovaBot:")[-1].strip()
            response = response.split("\n")[0].strip()
            
            if not response or len(response) < 3:
                response = self._fallback_response(user_input, emotion)

            # Update history
            self.conversation_history.append({"role": "User", "content": user_input})
            self.conversation_history.append({"role": "NovaBot", "content": response})
            
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return response

        except Exception as e:
            print(f"[ConversationalAI] Generation error: {e}")
            return self._fallback_response(user_input, emotion)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def train(self, dataset_path: Optional[str] = None, epochs: int = 3, batch_size: int = 4) -> bool:
        """Fine-tune the conversational model."""
        try:
            from transformers import (
                TrainingArguments, 
                Trainer,
                DataCollatorForLanguageModeling
            )
            from datasets import Dataset
            import torch

            print("[ConversationalAI] Initializing training...")
            
            if dataset_path and os.path.exists(dataset_path):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
            else:
                conversations = self._get_emotional_support_data()

            print(f"[ConversationalAI] Training on {len(conversations)} conversations")

            training_texts = []
            for conv in conversations:
                if isinstance(conv, dict):
                    user_msg = conv.get('user', conv.get('input', ''))
                    bot_msg = conv.get('bot', conv.get('output', conv.get('response', '')))
                    training_texts.append(f"User: {user_msg}\nNovaBot: {bot_msg}")

            dataset = Dataset.from_dict({"text": training_texts})

            def tokenize_function(examples):
                return self.tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

            training_args = TrainingArguments(
                output_dir=CONVERSATION_MODEL_PATH,
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                warmup_steps=50,
                weight_decay=0.01,
                logging_steps=10,
                save_strategy="epoch",
                fp16=torch.cuda.is_available(),
            )

            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            trainer.train()
            trainer.save_model(CONVERSATION_MODEL_PATH)
            self.tokenizer.save_pretrained(CONVERSATION_MODEL_PATH)

            print("[ConversationalAI] Training complete!")
            return True

        except Exception as e:
            print(f"[ConversationalAI] Training error: {e}")
            return False

    # ==================== PRIVATE METHODS ====================

    def _load_model(self):
        """Load conversational model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            if os.path.exists(CONVERSATION_MODEL_PATH):
                print(f"[ConversationalAI] Loading fine-tuned model")
                self.tokenizer = AutoTokenizer.from_pretrained(CONVERSATION_MODEL_PATH)
                self.model = AutoModelForCausalLM.from_pretrained(CONVERSATION_MODEL_PATH)
            else:
                print(f"[ConversationalAI] Loading base model: {self.base_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("[ConversationalAI] Model loaded successfully!")
            
        except Exception as e:
            print(f"[ConversationalAI] Model loading error: {e}")
            self.model = None

    def _fallback_response(self, user_input: str, emotion: str = None) -> str:
        """Fallback responses when model isn't available."""
        user_lower = user_input.lower()
        
        emotion_responses = {
            'sad': ["I sense you might be feeling down. I'm here for you."],
            'happy': ["I can tell you're in good spirits! That's wonderful."],
            'angry': ["I understand you might be frustrated. Take a deep breath."],
            'fear': ["I'm here with you. You're safe."]
        }
        
        if emotion and emotion in emotion_responses:
            import random
            return random.choice(emotion_responses[emotion])
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! It's great to hear from you. How are you feeling today?"
        
        if any(word in user_lower for word in ['thank', 'thanks']):
            return "You're very welcome! I'm always here for you."
        
        if any(word in user_lower for word in ['bye', 'goodbye']):
            return "Take care! I'll be here whenever you need me."
        
        return "I'm listening and I'm here for you. Would you like to tell me more?"

    def _get_emotional_support_data(self):
        """Built-in emotional support conversation dataset."""
        return [
            {"user": "I'm feeling really sad today", "bot": "I'm sorry to hear you're feeling sad. I'm here for you."},
            {"user": "I feel lonely", "bot": "Feeling lonely can be really hard. I'm here to keep you company."},
            {"user": "I'm scared", "bot": "It's okay to feel scared. You're safe here with me."},
            {"user": "I'm happy today", "bot": "That's wonderful! What's making you feel so good today?"},
            {"user": "Hello", "bot": "Hello! It's so nice to hear from you. How are you feeling today?"},
            {"user": "Thank you", "bot": "You're very welcome! Is there anything else I can help with?"},
        ]
