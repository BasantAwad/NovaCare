"""
Face Emotion Predictor - Specialist 3
Uses ResNet50 fine-tuned on FER2013 dataset for facial emotion recognition.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Union
from PIL import Image

# Try TensorFlow/Keras first, fall back to PyTorch
try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    USE_TENSORFLOW = True
except ImportError:
    USE_TENSORFLOW = False
    from torchvision import models, transforms


class FaceEmotionAnalyzer:
    """
    Face-based emotion analyzer using ResNet50.
    Analyzes facial expressions to detect emotions.
    """
    
    # FER2013 emotion labels
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Image specifications
    TARGET_SIZE = (224, 224)
    FER_SIZE = (48, 48)
    
    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Face Emotion Analyzer.
        
        Args:
            model_path: Path to locally saved model.
            device: Device to run model on ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.use_tensorflow = USE_TENSORFLOW
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("No pre-trained model found. Building base architecture...")
            self._build_model()
    
    def _build_model(self):
        """Build the ResNet50 model architecture."""
        if self.use_tensorflow:
            self._build_tensorflow_model()
        else:
            self._build_pytorch_model()
    
    def _build_tensorflow_model(self):
        """Build model using TensorFlow/Keras."""
        print("Building TensorFlow model...")
        
        # Load ResNet50 without the top classification layer
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers (optional - unfreeze for fine-tuning)
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(7, activation='softmax')(x)  # 7 emotions
        
        self.model = Model(inputs=base_model.input, outputs=outputs)
        print("TensorFlow model built successfully!")
    
    def _build_pytorch_model(self):
        """Build model using PyTorch."""
        print("Building PyTorch model...")
        
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)  # 7 emotions
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transforms = transforms.Compose([
            transforms.Resize(self.TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        
        print(f"PyTorch model built successfully on {self.device}!")
    
    def _load_model(self, model_path: str):
        """Load a saved model from disk."""
        print(f"Loading model from {model_path}...")
        
        try:
            if model_path.endswith('.h5') or model_path.endswith('.keras'):
                # TensorFlow model
                self.model = load_model(model_path)
                self.use_tensorflow = True
            elif model_path.endswith('.pt') or model_path.endswith('.pth'):
                # PyTorch model
                self._build_pytorch_model()
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.use_tensorflow = False
            else:
                # Try to detect format
                if os.path.isdir(model_path):
                    # Likely a SavedModel directory
                    self.model = tf.keras.models.load_model(model_path)
                    self.use_tensorflow = True
                else:
                    raise ValueError(f"Unknown model format: {model_path}")
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Building new model instead...")
            self._build_model()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        Handles grayscale conversion and resizing.
        
        Args:
            image: Input image (BGR format from OpenCV or RGB).
        
        Returns:
            Preprocessed image array.
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            # Grayscale - stack to create 3 channels
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            # BGRA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[-1] == 3:
            # Assume BGR from OpenCV - convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, self.TARGET_SIZE)
        
        return image
    
    def _detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract face from image.
        
        Args:
            image: Input image.
        
        Returns:
            Cropped face image or None if no face detected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        padding = int(0.15 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]
    
    def predict(
        self,
        image_input: Union[str, np.ndarray],
        detect_face: bool = True,
        return_all_scores: bool = False
    ) -> Dict:
        """
        Predict emotion from face image.
        
        Args:
            image_input: Path to image file or numpy array.
            detect_face: If True, detect and crop face first.
            return_all_scores: If True, return scores for all emotions.
        
        Returns:
            Dictionary with emotion and confidence.
            {
                "emotion": "happy",
                "confidence": 0.95,
                "face_detected": True,
                "all_scores": {...}  # Optional
            }
        """
        if self.model is None:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "face_detected": False,
                "error": "Model not loaded"
            }
        
        try:
            # Load image if path provided
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image from {image_input}")
            else:
                image = image_input.copy()
            
            # Detect face if requested
            face_detected = True
            if detect_face:
                face = self._detect_face(image)
                if face is not None:
                    image = face
                else:
                    face_detected = False
                    # Use full image if no face detected
            
            # Preprocess
            processed = self._preprocess_image(image)
            
            # Run inference
            if self.use_tensorflow:
                # TensorFlow inference
                processed = processed.astype(np.float32) / 255.0
                processed = np.expand_dims(processed, axis=0)
                predictions = self.model.predict(processed, verbose=0)[0]
            else:
                # PyTorch inference
                pil_image = Image.fromarray(processed)
                tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(tensor)
                    predictions = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            
            # Get top prediction
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            emotion = self.EMOTION_LABELS[predicted_idx]
            
            response = {
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "face_detected": face_detected
            }
            
            if return_all_scores:
                response["all_scores"] = {
                    self.EMOTION_LABELS[i]: round(float(p), 4)
                    for i, p in enumerate(predictions)
                }
            
            return response
            
        except Exception as e:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "face_detected": False,
                "error": str(e)
            }
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        detect_face: bool = True
    ) -> List[Dict]:
        """
        Predict emotions for multiple images.
        
        Args:
            images: List of image paths or numpy arrays.
            detect_face: If True, detect and crop face first.
        
        Returns:
            List of prediction dictionaries.
        """
        return [self.predict(img, detect_face=detect_face) for img in images]
    
    def predict_from_video_frames(
        self,
        frames: List[np.ndarray],
        detect_face: bool = True
    ) -> Dict:
        """
        Analyze emotions across video frames and aggregate results.
        
        Args:
            frames: List of video frames.
            detect_face: If True, detect face in each frame.
        
        Returns:
            Aggregated emotion analysis.
        """
        all_predictions = []
        emotion_counts = {e: 0 for e in self.EMOTION_LABELS}
        total_confidence = 0.0
        faces_detected = 0
        
        for frame in frames:
            result = self.predict(frame, detect_face=detect_face)
            all_predictions.append(result)
            
            if result['emotion'] != 'unknown':
                emotion_counts[result['emotion']] += 1
                total_confidence += result['confidence']
                if result.get('face_detected', False):
                    faces_detected += 1
        
        # Get dominant emotion
        valid_predictions = [p for p in all_predictions if p['emotion'] != 'unknown']
        
        if valid_predictions:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            avg_confidence = total_confidence / len(valid_predictions)
        else:
            dominant_emotion = 'unknown'
            avg_confidence = 0.0
        
        return {
            "dominant_emotion": dominant_emotion,
            "average_confidence": round(avg_confidence, 4),
            "frames_analyzed": len(frames),
            "faces_detected": faces_detected,
            "emotion_distribution": {
                k: v / len(frames) if len(frames) > 0 else 0
                for k, v in emotion_counts.items()
            },
            "frame_predictions": all_predictions
        }
    
    def save_model(self, save_path: str):
        """
        Save the model to disk.
        
        Args:
            save_path: Path to save the model.
        """
        if self.model is None:
            print("No model to save!")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if self.use_tensorflow:
            self.model.save(save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        
        print(f"Model saved to {save_path}")


# ============================================================================
# Testing / Demo
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Face Emotion Analyzer - Demo")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = FaceEmotionAnalyzer()
    
    # Test with sample image (if available)
    test_image_path = "data/face_data/sample.jpg"
    
    if os.path.exists(test_image_path):
        print(f"\nAnalyzing: {test_image_path}")
        result = analyzer.predict(test_image_path, return_all_scores=True)
        print(f"  → Emotion: {result['emotion']}")
        print(f"  → Confidence: {result['confidence']:.2%}")
        print(f"  → Face detected: {result['face_detected']}")
        
        if 'all_scores' in result:
            print("  → All scores:")
            for emotion, score in sorted(result['all_scores'].items(), key=lambda x: -x[1]):
                print(f"      {emotion}: {score:.2%}")
    else:
        print(f"\nNo test image found at: {test_image_path}")
        print("To test, place an image file at the above path.")
    
    print("\nModel ready for inference!")
