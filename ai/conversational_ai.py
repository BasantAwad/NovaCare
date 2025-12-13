"""
NovaCare AI - Conversational AI Module
Fine-tunable LLM for emotional support, companionship, and context-aware dialogue.
"""
import os
import json
from datetime import datetime

# Model paths
CONVERSATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_models', 'conversation_model')

class ConversationalAI:
    """
    Main Conversational AI for NovaBot
    - Provides emotional support and companionship
    - Context-aware dialogue
    - Fine-tunable on custom datasets
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.model = None
        self.tokenizer = None
        self.base_model_name = model_name
        self.conversation_history = []
        self._load_model()

    def _load_model(self):
        """Load conversational model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Check for fine-tuned model first
            if os.path.exists(CONVERSATION_MODEL_PATH):
                print(f"[ConversationalAI] Loading fine-tuned model from {CONVERSATION_MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(CONVERSATION_MODEL_PATH)
                self.model = AutoModelForCausalLM.from_pretrained(CONVERSATION_MODEL_PATH)
            else:
                print(f"[ConversationalAI] Loading base model: {self.base_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("[ConversationalAI] Model loaded successfully!")
            
        except Exception as e:
            print(f"[ConversationalAI] Model loading error: {e}")
            self.model = None

    def train(self, dataset_path=None, epochs=3, batch_size=4):
        """
        Fine-tune the conversational model
        :param dataset_path: Path to conversational dataset (JSON format: [{"user": "", "bot": ""}])
        :param epochs: Number of training epochs
        :param batch_size: Training batch size
        """
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                TrainingArguments, 
                Trainer,
                DataCollatorForLanguageModeling
            )
            from datasets import Dataset
            import torch

            print("[ConversationalAI] Initializing training...")
            
            # Load or create training data
            if dataset_path and os.path.exists(dataset_path):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
            else:
                # Use built-in emotional support conversations for demo
                conversations = self._get_emotional_support_data()

            print(f"[ConversationalAI] Training on {len(conversations)} conversations")

            # Prepare training texts
            training_texts = []
            for conv in conversations:
                if isinstance(conv, dict):
                    user_msg = conv.get('user', conv.get('input', ''))
                    bot_msg = conv.get('bot', conv.get('output', conv.get('response', '')))
                    training_texts.append(f"User: {user_msg}\nNovaBot: {bot_msg}")

            # Create dataset
            dataset = Dataset.from_dict({"text": training_texts})

            # Tokenize
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=256,
                    padding="max_length"
                )

            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

            # Training arguments
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

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            print("[ConversationalAI] Starting training...")
            trainer.train()

            # Save
            print(f"[ConversationalAI] Saving to {CONVERSATION_MODEL_PATH}")
            trainer.save_model(CONVERSATION_MODEL_PATH)
            self.tokenizer.save_pretrained(CONVERSATION_MODEL_PATH)

            print("[ConversationalAI] Training complete!")
            return True

        except Exception as e:
            print(f"[ConversationalAI] Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_emotional_support_data(self):
        """Built-in emotional support conversation dataset"""
        return [
            {"user": "I'm feeling really sad today", "bot": "I'm sorry to hear you're feeling sad. I'm here for you. Would you like to talk about what's on your mind? Sometimes sharing helps."},
            {"user": "I feel lonely", "bot": "Feeling lonely can be really hard. I'm here to keep you company. We can chat, listen to music, or I can tell you an interesting story. What would help you feel better?"},
            {"user": "I'm scared", "bot": "It's okay to feel scared sometimes. You're safe here with me. Can you tell me what's worrying you? I'm here to listen and help."},
            {"user": "Nobody understands me", "bot": "I understand that feeling of not being understood can be frustrating. I'm here to listen without judgment. What's on your mind?"},
            {"user": "I'm happy today", "bot": "That's wonderful to hear! Your happiness makes me happy too. What's making you feel so good today?"},
            {"user": "I'm anxious about tomorrow", "bot": "It's natural to feel anxious about the future. Let's focus on the present moment. Is there anything specific I can help you prepare for?"},
            {"user": "I miss my family", "bot": "Missing loved ones is a sign of how much they mean to you. Would you like me to help you connect with them? I can help you make a call or send a message."},
            {"user": "I feel tired all the time", "bot": "Persistent tiredness can be draining. Are you getting enough sleep? Have you been eating well? If this continues, it might be worth discussing with a doctor."},
            {"user": "I'm frustrated with everything", "bot": "Frustration happens to everyone. Would you like to take a few deep breaths together? Or tell me what's bothering you - sometimes talking it out helps."},
            {"user": "I don't want to talk", "bot": "That's completely okay. I'm here whenever you're ready. We can just sit in silence together, or I can play some calming music if you'd like."},
            {"user": "Thank you for being here", "bot": "I'm always here for you. Supporting you is what I'm here for. Is there anything else I can help you with?"},
            {"user": "Can you tell me something positive", "bot": "Of course! Did you know that laughing for just 15 minutes a day can burn up to 40 calories? Your smile is already making the world brighter!"},
            {"user": "I need help", "bot": "I'm here to help. What do you need? Whether it's a listening ear, medical information, or connecting with your guardian, I'm ready to assist."},
            {"user": "Hello", "bot": "Hello! It's so nice to hear from you. How are you feeling today? I'm here and ready to chat or help with anything you need."},
            {"user": "Good morning", "bot": "Good morning! I hope you had a restful night. How are you feeling today? Is there anything I can help you with this morning?"},
        ]

    def generate_response(self, user_input: str, emotion: str = None) -> str:
        """
        Generate a response to user input
        :param user_input: User's message
        :param emotion: Detected emotion (optional) for context
        :return: Generated response
        """
        if self.model is None:
            return self._fallback_response(user_input, emotion)

        try:
            # Build conversation context
            context = ""
            
            # Add emotion context if available
            if emotion and emotion not in ['neutral', 'unknown']:
                context += f"[User seems {emotion}] "
            
            # Add recent history for context
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
            
            # Extract just the bot's response
            if "NovaBot:" in response:
                response = response.split("NovaBot:")[-1].strip()
            
            # Clean up
            response = response.split("\n")[0].strip()
            
            if not response or len(response) < 3:
                response = self._fallback_response(user_input, emotion)

            # Update history
            self.conversation_history.append({"role": "User", "content": user_input})
            self.conversation_history.append({"role": "NovaBot", "content": response})
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            print(f"[ConversationalAI] Generated: {response[:50]}...")
            return response

        except Exception as e:
            print(f"[ConversationalAI] Generation error: {e}")
            return self._fallback_response(user_input, emotion)

    def _fallback_response(self, user_input: str, emotion: str = None) -> str:
        """Fallback responses when model isn't available"""
        user_lower = user_input.lower()
        
        # Emotion-based responses
        emotion_responses = {
            'sad': [
                "I sense you might be feeling down. I'm here for you. Would you like to talk about it?",
                "It's okay to feel sad sometimes. I'm here to listen and support you."
            ],
            'happy': [
                "I can tell you're in good spirits! That's wonderful. What's making you smile today?",
                "Your positivity is contagious! What would you like to do today?"
            ],
            'angry': [
                "I understand you might be frustrated. Take a deep breath with me. I'm here to help.",
                "It's okay to feel upset. Would you like to talk about what's bothering you?"
            ],
            'fear': [
                "I'm here with you. You're safe. Would you like me to help with something?",
                "It's okay to feel nervous. I'm right here. What can I do to help?"
            ]
        }
        
        if emotion and emotion in emotion_responses:
            import random
            return random.choice(emotion_responses[emotion])
        
        # Keyword-based responses
        if any(word in user_lower for word in ['help', 'need', 'assist']):
            return "I'm here to help you. What do you need? I can assist with information, reminders, or just keeping you company."
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! It's great to hear from you. How are you feeling today?"
        
        if any(word in user_lower for word in ['thank', 'thanks']):
            return "You're very welcome! I'm always here for you. Is there anything else I can help with?"
        
        if any(word in user_lower for word in ['bye', 'goodbye']):
            return "Take care! I'll be here whenever you need me. Stay safe!"
        
        # Default response
        return "I'm listening and I'm here for you. Would you like to tell me more about what's on your mind?"

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Singleton
_conversational_ai_instance = None

def get_conversational_ai():
    global _conversational_ai_instance
    if _conversational_ai_instance is None:
        _conversational_ai_instance = ConversationalAI()
    return _conversational_ai_instance


if __name__ == "__main__":
    ai = ConversationalAI()
    
    print("\n=== Testing Conversational AI ===")
    test_inputs = [
        ("Hello!", None),
        ("I'm feeling a bit sad today", "sad"),
        ("Can you help me?", None),
        ("Thank you for listening", None)
    ]
    
    for user_input, emotion in test_inputs:
        response = ai.generate_response(user_input, emotion)
        print(f"\nUser: {user_input}")
        print(f"NovaBot: {response}")
