"""Core architecture components for the trading system.

Provides foundational services and infrastructure components:
- Base services and interfaces
- Configuration management
- Logging and error handling
- Health monitoring and circuit breakers
- Dependency injection container
- Retry and shutdown handlers
"""

from .advanced_circuit_breaker import (
    AdvancedCircuitBreaker,
    CircuitBreakerException,
    CircuitBreakerManager,
    circuit_breaker,
    get_circuit_breaker_manager,
)
from .base_service import BaseService, ServiceHealth
from .circuit_breaker import CircuitBreakerConfig, CircuitState
from .config_manager import ConfigurationManager
from .container import DIContainer, ServiceLocator
from .enhanced_logger import EnhancedLogger, LogLevel, PerformanceLogger
from .error_handler import ErrorCategory, ErrorSeverity, TradingBotException
from .feature_manager import FeatureSchema, SchemaVersion
from .health_monitor import AlertSeverity, HealthStatus
from .interfaces import (
    IConfigurationManager,
    IDataProvider,
    IFeatureManager,
    ILogger,
    IModelManager,
    INotificationService,
    ITradingEngine,
    ModelMetadata,
    ValidationResult,
)
from .resilience import (
    BulkheadConfig,
    BulkheadHandler,
    BulkheadRejectedException,
    RetryConfig,
    RetryExhaustedException,
    RetryHandler,
    RetryStrategy,
    TimeoutConfig,
    TimeoutException,
    TimeoutHandler,
    bulkhead,
    retry,
    timeout,
)
from .retry_handler import RetryConfig, RetryStrategy
from .shutdown_handler import ShutdownPhase, ShutdownReason
from .structured_logger import (
    CorrelationContextManager,
    LogContext,
    LoggerFactory,
    LoggerType,
    StructuredLogger,
    correlation_context,
    get_correlation_manager,
    get_logger,
    get_model_logger,
    get_performance_logger,
    get_trading_logger,
    log_performance,
)

__all__ = [
    # Base services
    "BaseService",
    "ServiceHealth",
    # Interfaces
    "IConfigurationManager",
    "IFeatureManager",
    "IModelManager",
    "IDataProvider",
    "ITradingEngine",
    "ILogger",
    "INotificationService",
    "ModelMetadata",
    "ValidationResult",
    # Core services
    "ConfigurationManager",
    "EnhancedLogger",
    "LogLevel",
    "PerformanceLogger",
    # Structured logging
    "StructuredLogger",
    "LogContext",
    "CorrelationContextManager",
    "LoggerType",
    "LoggerFactory",
    "get_logger",
    "get_trading_logger",
    "get_model_logger",
    "get_performance_logger",
    "log_performance",
    "correlation_context",
    "get_correlation_manager",
    # Error handling
    "ErrorSeverity",
    "ErrorCategory",
    "TradingBotException",
    # Health monitoring
    "HealthStatus",
    "AlertSeverity",
    # Circuit breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "AdvancedCircuitBreaker",
    "CircuitBreakerManager",
    "CircuitBreakerException",
    "get_circuit_breaker_manager",
    "circuit_breaker",
    # Resilience patterns
    "RetryHandler",
    "TimeoutHandler",
    "BulkheadHandler",
    "RetryConfig",
    "TimeoutConfig",
    "BulkheadConfig",
    "RetryStrategy",
    "RetryExhaustedException",
    "TimeoutException",
    "BulkheadRejectedException",
    "retry",
    "timeout",
    "bulkhead",
    # Shutdown handling
    "ShutdownReason",
    "ShutdownPhase",
    # Dependency injection
    "DIContainer",
    "ServiceLocator",
    # Feature management
    "FeatureSchema",
    "SchemaVersion",
]
