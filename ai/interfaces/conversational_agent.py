"""
NovaCare AI - Conversational Agent Interface
Defines the contract for conversational AI systems.
"""
from typing import Protocol, Optional, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class IConversationalAgent(Protocol):
    """
    Interface for conversational AI agents.
    Implementations must provide methods for generating responses and managing history.
    
    SOLID: Interface Segregation Principle
    - Focused interface for conversation only
    """
    
    @abstractmethod
    def generate_response(self, user_input: str, emotion: Optional[str] = None) -> str:
        """
        Generate a response to user input.
        :param user_input: User's message
        :param emotion: Detected emotion context (optional)
        :return: Generated response string
        """
        ...
    
    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history."""
        ...
    
    @abstractmethod
    def train(self, dataset_path: Optional[str] = None, **kwargs) -> bool:
        """
        Train/fine-tune the conversational model.
        :param dataset_path: Path to conversation dataset
        :return: Success status
        """
        ...
