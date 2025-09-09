"""Base service class for the trading system."""

import traceback
from abc import ABC
from datetime import datetime
from typing import Any, Dict, Optional

from .container import inject
from .interfaces import ILogger


class BaseService(ABC):
    """Base class for all services in the trading system."""

    def __init__(self, logger: Optional[ILogger] = None):
        self._logger = logger
        self._service_name = self.__class__.__name__
        self._initialized = False
        self._context: Dict[str, Any] = {}

    def initialize(self) -> bool:
        """Initialize the service. Override in subclasses."""
        try:
            self._log_info(f"Initializing {self._service_name}")
            self._initialized = True
            return True
        except Exception as e:
            self._log_error(f"Failed to initialize {self._service_name}", exception=e)
            return False

    def shutdown(self):
        """Shutdown the service. Override in subclasses."""
        try:
            self._log_info(f"Shutting down {self._service_name}")
            self._initialized = False
        except Exception as e:
            self._log_error(f"Error during {self._service_name} shutdown", exception=e)

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    @property
    def service_name(self) -> str:
        """Get service name."""
        return self._service_name

    def set_context(self, key: str, value: Any):
        """Set context information."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information."""
        return self._context.get(key, default)

    def clear_context(self):
        """Clear all context information."""
        self._context.clear()

    def _log_info(self, message: str, context: Dict[str, Any] = None):
        """Log info message with service context."""
        if self._logger is None:
            try:
                self._logger = inject(ILogger)
            except:
                print(f"[{self._service_name}] {message}")  # Fallback to print
                return
        full_context = self._get_full_context(context)
        self._logger.log_info(f"[{self._service_name}] {message}", full_context)

    def _log_warning(self, message: str, context: Dict[str, Any] = None):
        """Log warning message with service context."""
        if self._logger is None:
            try:
                self._logger = inject(ILogger)
            except:
                print(f"WARNING [{self._service_name}] {message}")  # Fallback to print
                return
        full_context = self._get_full_context(context)
        self._logger.log_warning(f"[{self._service_name}] {message}", full_context)

    def _log_error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log error message with service context."""
        if self._logger is None:
            try:
                self._logger = inject(ILogger)
            except:
                print(f"ERROR [{self._service_name}] {message}")  # Fallback to print
                if exception:
                    print(f"Exception: {exception}")
                return
        full_context = self._get_full_context(context)
        if exception:
            full_context["exception_type"] = type(exception).__name__
            full_context["exception_message"] = str(exception)
            full_context["traceback"] = traceback.format_exc()
        self._logger.log_error(f"[{self._service_name}] {message}", full_context, exception)

    def _get_full_context(self, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get full context including service context."""
        full_context = {
            "service": self._service_name,
            "timestamp": datetime.now().isoformat(),
            "initialized": self._initialized,
        }
        full_context.update(self._context)
        if additional_context:
            full_context.update(additional_context)
        return full_context

    def _ensure_initialized(self):
        """Ensure service is initialized before operation."""
        if not self._initialized:
            raise RuntimeError(f"{self._service_name} is not initialized. Call initialize() first.")

    def _safe_execute(self, operation_name: str, operation_func, *args, **kwargs):
        """Safely execute an operation with error handling."""
        try:
            self._ensure_initialized()
            self._log_info(f"Starting {operation_name}")
            result = operation_func(*args, **kwargs)
            self._log_info(f"Completed {operation_name}")
            return result
        except Exception as e:
            self._log_error(f"Failed {operation_name}", exception=e)
            raise


class ServiceHealth:
    """Health check information for a service."""

    def __init__(
        self,
        service_name: str,
        is_healthy: bool,
        message: str = "",
        details: Dict[str, Any] = None,
    ):
        self.service_name = service_name
        self.is_healthy = is_healthy
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "is_healthy": self.is_healthy,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class HealthCheckMixin:
    """Mixin for services that support health checks."""

    def health_check(self) -> ServiceHealth:
        """Perform health check. Override in subclasses."""
        service_name = getattr(self, "_service_name", self.__class__.__name__)
        is_initialized = getattr(self, "_initialized", False)

        if not is_initialized:
            return ServiceHealth(
                service_name=service_name,
                is_healthy=False,
                message="Service not initialized",
            )

        return ServiceHealth(
            service_name=service_name, is_healthy=True, message="Service is healthy"
        )
