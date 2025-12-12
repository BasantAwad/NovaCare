"""
NovaCare AI - Text-Based Emotion Analysis
Uses sentiment analysis and keyword matching for emotion detection from text.
Future: Train on Kaggle text emotion dataset for better accuracy.
"""
import os
import re
from datetime import datetime

# Placeholder for trained model
TEXT_EMOTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_models', 'text_emotion_model.pkl')

class TextEmotionAnalyzer:
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'glad', 'pleased', 'delighted', 'cheerful', '😊', '😄', '🎉'],
            'sad': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'heartbroken', 'devastated', 'grief', 'sorrow', 'crying', '😢', '😭'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'rage', 'hate', '😠', '😡', '🤬'],
            'fear': ['scared', 'afraid', 'frightened', 'terrified', 'nervous', 'anxious', 'worried', 'panic', 'fear', '😰', '😨'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow', '😮', '😲'],
            'disgust': ['disgusted', 'gross', 'revolting', 'sick', 'nauseated', 'awful', '🤢', '🤮'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'regular']
        }
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load trained text emotion model if available"""
        if os.path.exists(TEXT_EMOTION_MODEL_PATH):
            try:
                import pickle
                with open(TEXT_EMOTION_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print("[TextEmotionAnalyzer] Trained model loaded")
            except Exception as e:
                print(f"[TextEmotionAnalyzer] Could not load model: {e}")

    def train(self, dataset_path, text_column='text', label_column='emotion'):
        """
        Train text emotion classifier on labeled dataset
        :param dataset_path: Path to CSV with text and emotion columns
        :param text_column: Column name for text
        :param label_column: Column name for emotion label
        """
        try:
            import pandas as pd
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            import pickle

            print(f"[TextEmotionAnalyzer] Loading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            X = df[text_column].astype(str)
            y = df[label_column].astype(str)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Build pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', MultinomialNB())
            ])

            print("[TextEmotionAnalyzer] Training model...")
            pipeline.fit(X_train, y_train)

            accuracy = pipeline.score(X_test, y_test)
            print(f"[TextEmotionAnalyzer] Validation Accuracy: {accuracy:.2%}")

            # Save model
            os.makedirs(os.path.dirname(TEXT_EMOTION_MODEL_PATH), exist_ok=True)
            with open(TEXT_EMOTION_MODEL_PATH, 'wb') as f:
                pickle.dump(pipeline, f)
            print(f"[TextEmotionAnalyzer] Model saved to {TEXT_EMOTION_MODEL_PATH}")

            self.model = pipeline
            return accuracy

        except Exception as e:
            print(f"[TextEmotionAnalyzer] Training error: {e}")
            raise

    def analyze(self, text: str) -> dict:
        """
        Analyze emotion from text
        :param text: Input text
        :return: dict with emotion, confidence, and method
        """
        result = {
            "text": text,
            "emotion": "neutral",
            "confidence": 0.5,
            "method": "keyword",
            "timestamp": datetime.now().isoformat()
        }

        # Try trained model first
        if self.model is not None:
            try:
                prediction = self.model.predict([text])[0]
                probabilities = self.model.predict_proba([text])[0]
                max_prob = max(probabilities)
                result["emotion"] = prediction
                result["confidence"] = float(max_prob)
                result["method"] = "ml_model"
                return result
            except:
                pass

        # Fallback to keyword matching
        text_lower = text.lower()
        emotion_scores = {emotion: 0 for emotion in self.emotion_keywords}

        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1

        max_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[max_emotion]

        if max_score > 0:
            result["emotion"] = max_emotion
            result["confidence"] = min(0.5 + (max_score * 0.1), 0.95)
        
        return result


# Singleton
_text_analyzer_instance = None

def get_text_analyzer():
    global _text_analyzer_instance
    if _text_analyzer_instance is None:
        _text_analyzer_instance = TextEmotionAnalyzer()
    return _text_analyzer_instance
