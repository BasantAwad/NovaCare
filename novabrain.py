import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class NovaBrain:
    def __init__(self, model_name="distilgpt2", use_local=True):
        """
        Initialize the NovaBrain LLM.
        :param model_name: Name of the HuggingFace model or path.
        :param use_local: If True, uses local HF model. If False, uses Mock/Rule-based.
        """
        self.use_local = use_local
        self.model_name = model_name
        self.generator = None
        self.history = []

        if self.use_local:
            try:
                print(f"Loading NovaBrain model: {model_name}...")
                # Initialize text-generation pipeline
                # Using distilgpt2 for speed on CPU/Edge
                self.generator = pipeline('text-generation', model=model_name)
                print("NovaBrain Model Loaded Successfully.")
            except Exception as e:
                print(f"Error loading model: {e}. Falling back to Mock mode.")
                self.use_local = False

    def process_input(self, user_input: str) -> str:
        """
        Process user input and return a response.
        """
        self.history.append({"role": "user", "content": user_input})
        
        response = ""
        if self.use_local and self.generator:
            try:
                # Generate response
                # Note: max_length includes input length, so we add a buffer
                prompt = f"User: {user_input}\nNovaBot:"
                outputs = self.generator(prompt, max_length=len(prompt)+50, num_return_sequences=1, truncation=True)
                generated_text = outputs[0]['generated_text']
                
                # Extract only the bot's response
                response = generated_text.replace(prompt, "").strip()
                if not response:
                    response = "I'm listening."
            except Exception as e:
                response = f"I'm having trouble thinking right now. ({str(e)})"
        else:
            # Mock / Rule-based fallback
            response = self._mock_response(user_input)

        self.history.append({"role": "assistant", "content": response})
        return response

    def _mock_response(self, text: str) -> str:
        text = text.lower()
        if "hello" in text or "hi" in text:
            return "Hello! I am NovaBot. How can I assist you today?"
        elif "help" in text:
            return "I can help with health monitoring, navigation, or just keeping you company."
        elif "emergency" in text or "fall" in text:
            return "EMERGENCY DETECTED. Alerting guardians and emergency services immediately."
        elif "time" in text:
            from datetime import datetime
            return f"Current time is {datetime.now().strftime('%H:%M')}."
        else:
            options = [
                "I understand.",
                "Could you tell me more?",
                "I am monitoring your environment.",
                "All systems are nominal."
            ]
            return random.choice(options)

    def get_history(self):
        return self.history
