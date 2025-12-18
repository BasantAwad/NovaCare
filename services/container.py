"""
NovaCare - Dependency Injection Container (SOLID: Dependency Inversion Principle)
Simple service container for registering and resolving dependencies.
"""
from typing import Dict, Any, Callable, TypeVar, Optional, Type

T = TypeVar('T')


class Container:
    """
    Simple dependency injection container.
    Supports singleton and transient lifetimes.
    """
    
    def __init__(self):
        self._services: Dict[str, Dict[str, Any]] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register_singleton(self, service_type: Type[T], factory: Callable[[], T], name: str = None) -> 'Container':
        """
        Register a singleton service (same instance returned each time).
        :param service_type: The type/interface of the service
        :param factory: Factory function to create the service
        :param name: Optional name for multiple implementations
        """
        key = self._get_key(service_type, name)
        self._services[key] = {
            'factory': factory,
            'singleton': True
        }
        return self
    
    def register_transient(self, service_type: Type[T], factory: Callable[[], T], name: str = None) -> 'Container':
        """
        Register a transient service (new instance each time).
        :param service_type: The type/interface of the service
        :param factory: Factory function to create the service
        :param name: Optional name for multiple implementations
        """
        key = self._get_key(service_type, name)
        self._services[key] = {
            'factory': factory,
            'singleton': False
        }
        return self
    
    def register_instance(self, service_type: Type[T], instance: T, name: str = None) -> 'Container':
        """
        Register an existing instance as a singleton.
        :param service_type: The type/interface of the service
        :param instance: The instance to register
        :param name: Optional name for multiple implementations
        """
        key = self._get_key(service_type, name)
        self._services[key] = {
            'factory': lambda: instance,
            'singleton': True
        }
        self._singletons[key] = instance
        return self
    
    def resolve(self, service_type: Type[T], name: str = None) -> Optional[T]:
        """
        Resolve a service by type.
        :param service_type: The type/interface to resolve
        :param name: Optional name for specific implementation
        :return: The service instance or None if not registered
        """
        key = self._get_key(service_type, name)
        
        if key not in self._services:
            return None
        
        service_info = self._services[key]
        
        if service_info['singleton']:
            if key not in self._singletons:
                self._singletons[key] = service_info['factory']()
            return self._singletons[key]
        
        return service_info['factory']()
    
    def _get_key(self, service_type: Type, name: str = None) -> str:
        """Generate a unique key for a service type and optional name."""
        type_name = getattr(service_type, '__name__', str(service_type))
        if name:
            return f"{type_name}:{name}"
        return type_name
    
    def clear(self) -> None:
        """Clear all registrations and cached singletons."""
        self._services.clear()
        self._singletons.clear()


# Global container instance with lazy initialization
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def configure_services(container: Container) -> Container:
    """
    Configure default services for NovaCare.
    Call this at application startup.
    """
    from ai.interfaces import IEmotionAnalyzer, IConversationalAgent, IMedicalQA
    
    # Register emotion analyzers
    try:
        from ai.emotion_detector import EmotionDetector
        container.register_singleton(IEmotionAnalyzer, EmotionDetector, name='face')
    except ImportError:
        pass
    
    try:
        from ai.text_emotion import TextEmotionAnalyzer
        container.register_singleton(IEmotionAnalyzer, TextEmotionAnalyzer, name='text')
    except ImportError:
        pass
    
    # Register conversational AI
    try:
        from ai.conversational_ai import ConversationalAI
        container.register_singleton(IConversationalAgent, ConversationalAI)
    except ImportError:
        pass
    
    # Register medical QA
    try:
        from ai.medical_qa import MedicalQA
        container.register_singleton(IMedicalQA, MedicalQA)
    except ImportError:
        pass
    
    return container
