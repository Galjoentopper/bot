"""Enhanced error handling and recovery mechanisms for the trading bot."""

import asyncio
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .interfaces import ILogger


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    NETWORK = "network"
    DATA = "data"
    MODEL = "model"
    TRADING = "trading"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    VALIDATION = "validation"


@dataclass
class ErrorContext:
    """Context information for errors."""

    timestamp: datetime
    component: str
    operation: str
    severity: ErrorSeverity
    category: ErrorCategory
    details: Dict[str, Any]
    message: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class TradingBotException(Exception):
    """Base exception for trading bot errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        component: str = "unknown",
        operation: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.component = component
        self.operation = operation
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.context = ErrorContext(
            timestamp=self.timestamp,
            component=component,
            operation=operation,
            severity=severity,
            category=category,
            details=self.details,
            message=message,
            stack_trace=traceback.format_exc(),
        )


class NetworkException(TradingBotException):
    """Network-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class DataException(TradingBotException):
    """Data-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA, **kwargs)


class ModelException(TradingBotException):
    """Model-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL, **kwargs)


class TradingException(TradingBotException):
    """Trading-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TRADING,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class ConfigurationException(TradingBotException):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)


class ValidationException(TradingBotException):
    """Validation-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class ErrorHandler:
    """Enhanced error handler with recovery mechanisms."""

    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.NETWORK: [self._retry_network_operation],
            ErrorCategory.DATA: [self._reload_data, self._use_cached_data],
            ErrorCategory.MODEL: [self._reload_model, self._use_fallback_model],
            ErrorCategory.TRADING: [self._pause_trading, self._emergency_stop],
            ErrorCategory.CONFIGURATION: [
                self._reload_config,
                self._use_default_config,
            ],
            ErrorCategory.VALIDATION: [self._revalidate_data, self._skip_validation],
        }
        self.max_history_size = 1000
        self.critical_error_threshold = 5  # Max critical errors in 1 hour

    def handle_error(
        self,
        error: Union[Exception, TradingBotException],
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True,
    ) -> bool:
        """Handle an error with optional recovery attempt.

        Args:
            error: The error to handle
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            True if error was recovered, False otherwise
        """
        if isinstance(error, TradingBotException):
            error_context = error.context
            if context:
                error_context.details.update(context)
        else:
            error_context = ErrorContext(
                timestamp=datetime.now(),
                component="unknown",
                operation="unknown",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM,
                details=context or {},
                message=str(error),
                stack_trace=traceback.format_exc(),
            )

        # Log the error
        self._log_error(error, error_context)

        # Add to history
        self._add_to_history(error_context)

        # Check for critical error patterns
        if self._is_critical_pattern():
            self._handle_critical_situation()
            return False

        # Attempt recovery if requested and error is recoverable
        recovery_successful = False
        if attempt_recovery and getattr(error, "recoverable", True):
            recovery_successful = self._attempt_recovery(error_context)
            error_context.recovery_attempted = True
            error_context.recovery_successful = recovery_successful

        return recovery_successful

    def _log_error(self, error: Exception, context: ErrorContext) -> None:
        """Log error with appropriate level."""
        log_message = f"[{context.component}] {context.operation}: {str(error)}"

        if hasattr(self.logger, "logger"):
            # Enhanced logger
            if context.severity == ErrorSeverity.CRITICAL:
                self.logger.logger.critical(log_message, extra={"context": context.details})
            elif context.severity == ErrorSeverity.HIGH:
                self.logger.logger.error(log_message, extra={"context": context.details})
            elif context.severity == ErrorSeverity.MEDIUM:
                self.logger.logger.warning(log_message, extra={"context": context.details})
            else:
                self.logger.logger.info(log_message, extra={"context": context.details})
        else:
            # Enhanced logger or standard logger
            if hasattr(self.logger, "log_critical"):
                # Enhanced logger methods
                if context.severity == ErrorSeverity.CRITICAL:
                    self.logger.log_critical(log_message, {"context": context.details})
                elif context.severity == ErrorSeverity.HIGH:
                    self.logger.log_error(log_message, {"context": context.details})
                elif context.severity == ErrorSeverity.MEDIUM:
                    self.logger.log_warning(log_message, {"context": context.details})
                else:
                    self.logger.log_info(log_message, {"context": context.details})
            else:
                # Standard logger methods
                if context.severity == ErrorSeverity.CRITICAL:
                    self.logger.critical(log_message)
                elif context.severity == ErrorSeverity.HIGH:
                    self.logger.error(log_message)
                elif context.severity == ErrorSeverity.MEDIUM:
                    self.logger.warning(log_message)
                else:
                    self.logger.info(log_message)

    def _add_to_history(self, context: ErrorContext) -> None:
        """Add error to history with size management."""
        self.error_history.append(context)
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size :]

    def _is_critical_pattern(self) -> bool:
        """Check if error pattern indicates critical situation."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)

        recent_critical_errors = [
            error
            for error in self.error_history
            if error.timestamp >= one_hour_ago and error.severity == ErrorSeverity.CRITICAL
        ]

        return len(recent_critical_errors) >= self.critical_error_threshold

    def _handle_critical_situation(self) -> None:
        """Handle critical error situation."""
        if hasattr(self.logger, "logger"):
            self.logger.logger.critical(
                "CRITICAL: Multiple critical errors detected. Initiating emergency procedures."
            )
        elif hasattr(self.logger, "log_critical"):
            self.logger.log_critical(
                "CRITICAL: Multiple critical errors detected. Initiating emergency procedures."
            )
        else:
            self.logger.critical(
                "CRITICAL: Multiple critical errors detected. Initiating emergency procedures."
            )

        # Implement emergency procedures here
        # This could include:
        # - Stopping all trading operations
        # - Sending alerts
        # - Creating system snapshots
        # - Initiating safe shutdown

    def _attempt_recovery(self, context: ErrorContext) -> bool:
        """Attempt to recover from error using registered strategies."""
        strategies = self.recovery_strategies.get(context.category, [])

        for strategy in strategies:
            try:
                if strategy(context):
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info(f"Recovery successful using {strategy.__name__}")
                    elif hasattr(self.logger, "log_info"):
                        self.logger.log_info(f"Recovery successful using {strategy.__name__}")
                    else:
                        self.logger.info(f"Recovery successful using {strategy.__name__}")
                    return True
            except Exception as e:
                if hasattr(self.logger, "logger"):
                    self.logger.logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
                elif hasattr(self.logger, "log_warning"):
                    self.logger.log_warning(f"Recovery strategy {strategy.__name__} failed: {e}")
                else:
                    self.logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")

        return False

    # Recovery strategy implementations
    def _retry_network_operation(self, context: ErrorContext) -> bool:
        """Retry network operation with backoff."""
        # Implementation would depend on the specific operation
        return False

    def _reload_data(self, context: ErrorContext) -> bool:
        """Reload data from source."""
        # Implementation would reload data
        return False

    def _use_cached_data(self, context: ErrorContext) -> bool:
        """Use cached data as fallback."""
        # Implementation would use cached data
        return False

    def _reload_model(self, context: ErrorContext) -> bool:
        """Reload model from disk."""
        # Implementation would reload model
        return False

    def _use_fallback_model(self, context: ErrorContext) -> bool:
        """Use fallback model."""
        # Implementation would switch to fallback model
        return False

    def _pause_trading(self, context: ErrorContext) -> bool:
        """Pause trading operations."""
        # Implementation would pause trading
        return True

    def _emergency_stop(self, context: ErrorContext) -> bool:
        """Emergency stop all operations."""
        # Implementation would stop all operations
        return True

    def _reload_config(self, context: ErrorContext) -> bool:
        """Reload configuration."""
        # Implementation would reload config
        return False

    def _use_default_config(self, context: ErrorContext) -> bool:
        """Use default configuration."""
        # Implementation would use defaults
        return False

    def _revalidate_data(self, context: ErrorContext) -> bool:
        """Revalidate data."""
        # Implementation would revalidate
        return False

    def _skip_validation(self, context: ErrorContext) -> bool:
        """Skip validation as last resort."""
        # Implementation would skip validation
        return True

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {}

        now = datetime.now()
        last_24h = [e for e in self.error_history if (now - e.timestamp).total_seconds() < 86400]

        stats = {
            "total_errors": len(self.error_history),
            "errors_last_24h": len(last_24h),
            "by_severity": {},
            "by_category": {},
            "recovery_rate": 0,
        }

        for error in last_24h:
            # Count by severity
            severity_key = error.severity.value
            stats["by_severity"][severity_key] = stats["by_severity"].get(severity_key, 0) + 1

            # Count by category
            category_key = error.category.value
            stats["by_category"][category_key] = stats["by_category"].get(category_key, 0) + 1

        # Calculate recovery rate
        recovery_attempts = [e for e in last_24h if e.recovery_attempted]
        if recovery_attempts:
            successful_recoveries = [e for e in recovery_attempts if e.recovery_successful]
            stats["recovery_rate"] = len(successful_recoveries) / len(recovery_attempts)

        return stats

    def get_error_history(self, limit: Optional[int] = None) -> List[ErrorContext]:
        """Get error history with optional limit.

        Args:
            limit: Maximum number of errors to return (most recent first)

        Returns:
            List of ErrorContext objects
        """
        if limit is None:
            return list(reversed(self.error_history))
        else:
            return list(reversed(self.error_history[-limit:]))
