"""
NovaCare - Services Package
Contains dependency injection container and service registration.
"""
from .container import Container, get_container

__all__ = ['Container', 'get_container']
