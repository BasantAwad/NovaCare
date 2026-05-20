"""
Adapters Module

Service adapters for integrating existing NovaCare microservices
into the optimized runtime layer.
"""

from .service_adapters import (
    ServiceAdapter,
    LLMServiceAdapter,
    ASLServiceAdapter,
    RobotServiceAdapter,
    TTSServiceAdapter,
    ServiceRegistry,
    get_service_registry,
)

__all__ = [
    "ServiceAdapter",
    "LLMServiceAdapter",
    "ASLServiceAdapter",
    "RobotServiceAdapter",
    "TTSServiceAdapter",
    "ServiceRegistry",
    "get_service_registry",
]
