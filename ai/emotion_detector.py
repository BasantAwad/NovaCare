"""
NovaCare AI - Emotion Detection Module
Uses TensorFlow/Keras for FER (Facial Emotion Recognition)
Dataset: Kaggle FER dataset (to be downloaded by user)
"""
import os
import numpy as np

# Placeholder paths - USER TO CONFIGURE
EMOTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_models', 'emotion_model.h5')
EMOTION_DATASET_PATH = None  # User will set this after downloading Kaggle dataset

# Emotion labels for FER dataset
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class EmotionDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or EMOTION_MODEL_PATH
        self._load_model()

    def _load_model(self):
        """Load pretrained emotion detection model if available"""
        try:
            from tensorflow.keras.models import load_model
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"[EmotionDetector] Model loaded from {self.model_path}")
            else:
                print(f"[EmotionDetector] No trained model found at {self.model_path}. Use train() first.")
        except ImportError:
            print("[EmotionDetector] TensorFlow not installed. Install with: pip install tensorflow")
        except Exception as e:
            print(f"[EmotionDetector] Error loading model: {e}")

    def train(self, dataset_path, epochs=25, batch_size=64):
        """
        Train emotion detection model on FER dataset
        :param dataset_path: Path to FER dataset (should have train/test subdirs with emotion folders)
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.optimizers import Adam

            print(f"[EmotionDetector] Starting training with dataset: {dataset_path}")

            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                validation_split=0.2
            )

            train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=(48, 48),
                color_mode='grayscale',
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )

            val_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=(48, 48),
                color_mode='grayscale',
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )

            # Build CNN model
            model = Sequential([
                Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),

                Conv2D(128, (3,3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),

                Conv2D(256, (3,3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),

                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(len(EMOTION_LABELS), activation='softmax')
            ])

            model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            model.summary()

            # Train
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator
            )

            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model.save(self.model_path)
            print(f"[EmotionDetector] Model saved to {self.model_path}")

            self.model = model
            return history

        except Exception as e:
            print(f"[EmotionDetector] Training error: {e}")
            raise

    def detect(self, face_image):
        """
        Detect emotion from a face image (48x48 grayscale expected)
        :param face_image: numpy array of shape (48, 48) or (48, 48, 1)
        :return: dict with emotion label and confidence scores
        """
        if self.model is None:
            return {'emotion': 'unknown', 'confidence': 0, 'all_scores': {}}

        try:
            # Preprocess
            if len(face_image.shape) == 2:
                face_image = np.expand_dims(face_image, axis=-1)
            face_image = face_image.astype('float32') / 255.0
            face_image = np.expand_dims(face_image, axis=0)

            # Predict
            predictions = self.model.predict(face_image, verbose=0)[0]
            emotion_idx = np.argmax(predictions)

            return {
                'emotion': EMOTION_LABELS[emotion_idx],
                'confidence': float(predictions[emotion_idx]),
                'all_scores': {label: float(score) for label, score in zip(EMOTION_LABELS, predictions)}
            }
        except Exception as e:
            print(f"[EmotionDetector] Detection error: {e}")
            return {'emotion': 'error', 'confidence': 0, 'all_scores': {}}


# Singleton instance for app usage
_detector_instance = None

def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EmotionDetector()
    return _detector_instance
