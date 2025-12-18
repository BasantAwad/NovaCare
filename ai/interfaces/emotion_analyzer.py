"""
NovaCare AI - Emotion Analyzer Interface
Defines the contract for emotion analysis (face and text).
"""
from typing import Protocol, Dict, Any, Union, runtime_checkable
from abc import abstractmethod
import numpy as np


@runtime_checkable
class IEmotionAnalyzer(Protocol):
    """
    Interface for emotion analysis.
    Implementations must provide methods for analyzing emotion from various inputs.
    
    SOLID: Interface Segregation Principle
    - Focused interface for emotion analysis only
    """
    
    @abstractmethod
    def analyze(self, input_data: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze emotion from input (auto-detects text vs image).
        :param input_data: Text string or face image array
        :return: Dict with 'emotion', 'confidence', 'source'
        """
        ...
    
    @abstractmethod
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotion from text input.
        :param text: Input text string
        :return: Dict with 'emotion', 'confidence', 'source': 'text'
        """
        ...
    
    @abstractmethod
    def analyze_face(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze emotion from face image.
        :param face_image: Grayscale face image (48x48)
        :return: Dict with 'emotion', 'confidence', 'source': 'face'
        """
        ...
    
    @abstractmethod
    def train(self, dataset_path: str, mode: str = 'text', **kwargs) -> Any:
        """
        Train the emotion model.
        :param dataset_path: Path to training dataset
        :param mode: 'text' or 'face'
        :return: Training result (accuracy for text, history for face)
        """
        ...
