"""
NovaCare AI - EmotionAnalyzer Implementation
Implements IEmotionAnalyzer interface.
Handles both face (image) and text emotion detection.
"""
import os
import numpy as np
from datetime import datetime
from typing import Union, Any, Dict

# Import interface
from ai.interfaces import IEmotionAnalyzer

# Model paths (relative to impl folder)
IMPL_DIR = os.path.dirname(__file__)
AI_DIR = os.path.dirname(IMPL_DIR)
FACE_MODEL_PATH = os.path.join(AI_DIR, 'trained_models', 'emotion_model.h5')
TEXT_MODEL_PATH = os.path.join(AI_DIR, 'trained_models', 'text_emotion_model.pkl')

# Emotion labels (shared across face and text)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class EmotionAnalyzer:
    """
    Unified Emotion Analyzer implementing IEmotionAnalyzer interface.
    Handles both face (image) and text emotion detection.
    
    SOLID Principles:
    - Single Responsibility: All emotion analysis in one place
    - Open/Closed: Extensible for new input types
    - Interface Segregation: Implements focused IEmotionAnalyzer
    - Dependency Inversion: Depends on abstraction (IEmotionAnalyzer)
    """
    
    def __init__(self, face_model_path: str = None, text_model_path: str = None):
        """
        Initialize the unified emotion analyzer.
        :param face_model_path: Path to face emotion model (.h5)
        :param text_model_path: Path to text emotion model (.pkl)
        """
        self.face_model_path = face_model_path or FACE_MODEL_PATH
        self.text_model_path = text_model_path or TEXT_MODEL_PATH
        
        # Models
        self.face_model = None
        self.text_model = None
        
        # Keyword fallback for text analysis
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'glad', 'pleased', 'delighted', 'cheerful', '😊', '😄', '🎉'],
            'sad': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'heartbroken', 'devastated', 'grief', 'sorrow', 'crying', '😢', '😭'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'rage', 'hate', '😠', '😡', '🤬'],
            'fear': ['scared', 'afraid', 'frightened', 'terrified', 'nervous', 'anxious', 'worried', 'panic', 'fear', '😰', '😨'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow', '😮', '😲'],
            'disgust': ['disgusted', 'gross', 'revolting', 'sick', 'nauseated', 'awful', '🤢', '🤮'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'regular']
        }
        
        # Load models
        self._load_face_model()
        self._load_text_model()
        
        print("[EmotionAnalyzer] Initialized")

    # ==================== IEmotionAnalyzer IMPLEMENTATION ====================
    
    def analyze(self, input_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze emotion from input (text or face image).
        Automatically detects input type and routes to appropriate method.
        """
        if isinstance(input_data, str):
            return self.analyze_text(input_data)
        elif isinstance(input_data, np.ndarray):
            return self.analyze_face(input_data)
        else:
            return {
                'emotion': 'unknown',
                'confidence': 0,
                'source': 'unknown',
                'error': f'Unsupported input type: {type(input_data)}'
            }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze emotion from text input."""
        result = {
            'text': text,
            'emotion': 'neutral',
            'confidence': 0.5,
            'source': 'text',
            'method': 'keyword',
            'timestamp': datetime.now().isoformat()
        }

        # Try ML model first
        if self.text_model is not None:
            try:
                prediction = self.text_model.predict([text])[0]
                probabilities = self.text_model.predict_proba([text])[0]
                max_prob = max(probabilities)
                result['emotion'] = prediction
                result['confidence'] = float(max_prob)
                result['method'] = 'ml_model'
                return result
            except Exception:
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
            result['emotion'] = max_emotion
            result['confidence'] = min(0.5 + (max_score * 0.1), 0.95)

        return result

    def analyze_face(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Analyze emotion from face image."""
        result = {
            'emotion': 'unknown',
            'confidence': 0,
            'source': 'face',
            'all_scores': {},
            'timestamp': datetime.now().isoformat()
        }

        if self.face_model is None:
            result['error'] = 'Face model not loaded'
            return result

        try:
            # Preprocess
            if len(face_image.shape) == 2:
                face_image = np.expand_dims(face_image, axis=-1)
            face_image = face_image.astype('float32') / 255.0
            face_image = np.expand_dims(face_image, axis=0)

            # Predict
            predictions = self.face_model.predict(face_image, verbose=0)[0]
            emotion_idx = np.argmax(predictions)

            result['emotion'] = EMOTION_LABELS[emotion_idx]
            result['confidence'] = float(predictions[emotion_idx])
            result['all_scores'] = {label: float(score) for label, score in zip(EMOTION_LABELS, predictions)}
            
        except Exception as e:
            print(f"[EmotionAnalyzer] Face detection error: {e}")
            result['error'] = str(e)

        return result

    def train(self, dataset_path: str, mode: str = 'text', **kwargs) -> Any:
        """Train emotion model."""
        if mode == 'text':
            return self._train_text_model(dataset_path, **kwargs)
        elif mode == 'face':
            return self._train_face_model(dataset_path, **kwargs)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    # ==================== PRIVATE METHODS ====================
    
    def _train_text_model(self, dataset_path: str, text_column: str = 'text', label_column: str = 'emotion'):
        """Train text emotion classifier."""
        try:
            import pandas as pd
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            import pickle

            print(f"[EmotionAnalyzer] Loading text dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            X = df[text_column].astype(str)
            y = df[label_column].astype(str)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', MultinomialNB())
            ])

            print("[EmotionAnalyzer] Training text model...")
            pipeline.fit(X_train, y_train)

            accuracy = pipeline.score(X_test, y_test)
            print(f"[EmotionAnalyzer] Text model accuracy: {accuracy:.2%}")

            os.makedirs(os.path.dirname(self.text_model_path), exist_ok=True)
            with open(self.text_model_path, 'wb') as f:
                pickle.dump(pipeline, f)

            self.text_model = pipeline
            return accuracy

        except Exception as e:
            print(f"[EmotionAnalyzer] Text training error: {e}")
            raise

    def _train_face_model(self, dataset_path: str, epochs: int = 25, batch_size: int = 64):
        """Train face emotion CNN."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.optimizers import Adam

            print(f"[EmotionAnalyzer] Loading face dataset from {dataset_path}")

            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                validation_split=0.2
            )

            train_gen = train_datagen.flow_from_directory(
                dataset_path, target_size=(48, 48), color_mode='grayscale',
                batch_size=batch_size, class_mode='categorical', subset='training'
            )
            val_gen = train_datagen.flow_from_directory(
                dataset_path, target_size=(48, 48), color_mode='grayscale',
                batch_size=batch_size, class_mode='categorical', subset='validation'
            )

            model = Sequential([
                Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
                BatchNormalization(), MaxPooling2D(2, 2), Dropout(0.25),
                Conv2D(128, (3,3), activation='relu'),
                BatchNormalization(), MaxPooling2D(2, 2), Dropout(0.25),
                Conv2D(256, (3,3), activation='relu'),
                BatchNormalization(), MaxPooling2D(2, 2), Dropout(0.25),
                Flatten(), Dense(512, activation='relu'), Dropout(0.5),
                Dense(len(EMOTION_LABELS), activation='softmax')
            ])

            model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)

            os.makedirs(os.path.dirname(self.face_model_path), exist_ok=True)
            model.save(self.face_model_path)

            self.face_model = model
            return history

        except Exception as e:
            print(f"[EmotionAnalyzer] Face training error: {e}")
            raise

    def _load_face_model(self):
        """Load pretrained face emotion model."""
        try:
            from tensorflow.keras.models import load_model
            if os.path.exists(self.face_model_path):
                self.face_model = load_model(self.face_model_path)
                print(f"[EmotionAnalyzer] Face model loaded")
            else:
                print(f"[EmotionAnalyzer] No face model at {self.face_model_path}")
        except ImportError:
            print("[EmotionAnalyzer] TensorFlow not installed - face detection unavailable")
        except Exception as e:
            print(f"[EmotionAnalyzer] Error loading face model: {e}")

    def _load_text_model(self):
        """Load pretrained text emotion model."""
        if os.path.exists(self.text_model_path):
            try:
                import pickle
                with open(self.text_model_path, 'rb') as f:
                    self.text_model = pickle.load(f)
                print(f"[EmotionAnalyzer] Text model loaded")
            except Exception as e:
                print(f"[EmotionAnalyzer] Error loading text model: {e}")
