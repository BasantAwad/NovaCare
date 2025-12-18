"""
NovaCare AI - Medical QA Interface
Defines the contract for medical question answering systems.
"""
from typing import Protocol, Dict, Any, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class IMedicalQA(Protocol):
    """
    Interface for medical question answering systems.
    Implementations must provide methods for answering medical queries.
    
    SOLID: Interface Segregation Principle
    - Focused interface for medical Q&A only
    """
    
    @abstractmethod
    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a medical question.
        :param question: User's medical question
        :return: Dict with 'answer', 'confidence', 'source', 'is_emergency'
        """
        ...
    
    @abstractmethod
    def train(self, dataset_name: str, **kwargs) -> bool:
        """
        Train the medical QA model.
        :param dataset_name: HuggingFace dataset name or path
        :return: Success status
        """
        ...
