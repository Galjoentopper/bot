"""Dependency injection container for the trading system."""
from typing import Any, Dict, Type, TypeVar, Callable, Optional
from abc import ABC, abstractmethod
import inspect
from functools import wraps

T = TypeVar('T')


class DIContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._interfaces: Dict[Type, Type] = {}
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> 'DIContainer':
        """Register a singleton service."""
        self._interfaces[interface] = implementation
        return self
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> 'DIContainer':
        """Register a transient service (new instance each time)."""
        self._interfaces[interface] = implementation
        self._services[interface.__name__] = 'transient'
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'DIContainer':
        """Register a factory function."""
        self._factories[interface.__name__] = factory
        return self
    
    def register_instance(self, interface: Type[T], instance: T) -> 'DIContainer':
        """Register a specific instance."""
        self._singletons[interface.__name__] = instance
        return self
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service by interface."""
        interface_name = interface.__name__
        
        # Check if we have a registered instance
        if interface_name in self._singletons:
            return self._singletons[interface_name]
        
        # Check if we have a factory
        if interface_name in self._factories:
            instance = self._factories[interface_name]()
            # Cache singletons
            if interface_name not in self._services or self._services[interface_name] != 'transient':
                self._singletons[interface_name] = instance
            return instance
        
        # Check if we have an implementation registered
        if interface in self._interfaces:
            implementation = self._interfaces[interface]
            instance = self._create_instance(implementation)
            
            # Cache singletons
            if interface_name not in self._services or self._services[interface_name] != 'transient':
                self._singletons[interface_name] = instance
            
            return instance
        
        raise ValueError(f"No registration found for {interface.__name__}")
    
    def _create_instance(self, cls: Type[T]) -> T:
        """Create an instance with dependency injection."""
        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        
        # Resolve dependencies
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except ValueError:
                    # If we can't resolve, check if it has a default value
                    if param.default == inspect.Parameter.empty:
                        raise ValueError(f"Cannot resolve dependency {param.annotation.__name__} for {cls.__name__}")
        
        return cls(**kwargs)
    
    def clear(self):
        """Clear all registrations."""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()
        self._interfaces.clear()


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global container instance."""
    return _container


def inject(interface: Type[T]) -> T:
    """Inject a dependency."""
    return _container.resolve(interface)


def injectable(cls):
    """Decorator to mark a class as injectable."""
    original_init = cls.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Auto-inject dependencies if not provided
        sig = inspect.signature(original_init)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = _container.resolve(param.annotation)
                except ValueError:
                    # If we can't resolve and no default, let the original constructor handle it
                    if param.default == inspect.Parameter.empty:
                        pass
        
        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls


class ServiceLocator:
    """Service locator pattern for backward compatibility."""
    
    @staticmethod
    def get_service(interface: Type[T]) -> T:
        """Get a service by interface."""
        return _container.resolve(interface)
    
    @staticmethod
    def register_service(interface: Type[T], implementation: T):
        """Register a service instance."""
        _container.register_instance(interface, implementation)