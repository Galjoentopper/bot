"""Retry mechanism with exponential backoff and jitter."""

import asyncio
import random
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass

from .interfaces import ILogger
from .error_handler import ErrorHandler, NetworkException, ErrorSeverity


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"                    # Fixed delay between retries
    LINEAR = "linear"                  # Linear backoff
    EXPONENTIAL = "exponential"        # Exponential backoff
    EXPONENTIAL_JITTER = "exponential_jitter"  # Exponential with jitter


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0            # Base delay in seconds
    max_delay: float = 60.0            # Maximum delay in seconds
    backoff_multiplier: float = 2.0    # Multiplier for exponential backoff
    jitter: bool = True                # Add random jitter
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    retryable_exceptions: List[Type[Exception]] = None
    non_retryable_exceptions: List[Type[Exception]] = None

    def __post_init__(self):
        if self.retryable_exceptions is None:
            self.retryable_exceptions = [
                ConnectionError,
                TimeoutError,
                OSError,
                NetworkException
            ]
        if self.non_retryable_exceptions is None:
            self.non_retryable_exceptions = [
                ValueError,
                TypeError,
                KeyError,
                AttributeError
            ]


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    exception: Exception
    delay: float
    timestamp: datetime


@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_attempts: int = 0
    total_retry_time: float = 0.0
    max_attempts_reached: int = 0


class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, attempts: List[RetryAttempt]):
        super().__init__(message)
        self.attempts = attempts


class RetryHandler:
    """Retry handler with configurable strategies and backoff."""
    
    def __init__(
        self,
        name: str,
        config: Optional[RetryConfig] = None,
        logger: Optional[ILogger] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        self.name = name
        self.config = config or RetryConfig()
        self.logger = logger
        self.error_handler = error_handler
        self.stats = RetryStats()
        self.attempts_history: List[RetryAttempt] = []
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedException: When all retry attempts fail
            Original exception: For non-retryable exceptions
        """
        self.stats.total_operations += 1
        attempts = []
        start_time = time.time()
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.stats.total_attempts += 1
            
            try:
                self._log_debug(f"Attempt {attempt}/{self.config.max_attempts} for {self.name}")
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success!
                self.stats.successful_operations += 1
                total_time = time.time() - start_time
                self.stats.total_retry_time += total_time
                
                if attempt > 1:
                    self._log_info(
                        f"Operation {self.name} succeeded on attempt {attempt} "
                        f"after {total_time:.2f}s"
                    )
                
                return result
                
            except Exception as e:
                # Check if exception is retryable
                if not self._is_retryable(e):
                    self._log_warning(
                        f"Non-retryable exception in {self.name}: {type(e).__name__}: {e}"
                    )
                    self.stats.failed_operations += 1
                    raise
                
                # Record attempt
                delay = self._calculate_delay(attempt)
                attempt_info = RetryAttempt(
                    attempt_number=attempt,
                    exception=e,
                    delay=delay,
                    timestamp=datetime.now()
                )
                attempts.append(attempt_info)
                self.attempts_history.append(attempt_info)
                
                # Check if this was the last attempt
                if attempt >= self.config.max_attempts:
                    self.stats.failed_operations += 1
                    self.stats.max_attempts_reached += 1
                    total_time = time.time() - start_time
                    self.stats.total_retry_time += total_time
                    
                    self._log_error(
                        f"All {self.config.max_attempts} attempts failed for {self.name} "
                        f"after {total_time:.2f}s"
                    )
                    
                    # Report to error handler
                    if self.error_handler:
                        retry_error = NetworkException(
                            f"Retry exhausted for {self.name}: {str(e)}",
                            component="retry_handler",
                            operation="retry_execute",
                            details={
                                'operation_name': self.name,
                                'attempts': len(attempts),
                                'total_time': total_time,
                                'last_exception': str(e)
                            }
                        )
                        self.error_handler.handle_error(retry_error, attempt_recovery=False)
                    
                    raise RetryExhaustedException(
                        f"All {self.config.max_attempts} attempts failed for {self.name}",
                        attempts
                    )
                
                # Log retry attempt
                self._log_warning(
                    f"Attempt {attempt} failed for {self.name}: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if an exception is retryable.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if retryable, False otherwise
        """
        # Check non-retryable exceptions first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable exceptions
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        # Default: don't retry unknown exceptions
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            # Add jitter: ±25% of base delay
            jitter_range = base_delay * 0.25
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = base_delay + jitter
        
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        return min(delay, self.config.max_delay)
    
    def _log_debug(self, message: str) -> None:
        """Log debug message."""
        if self.logger:
            if hasattr(self.logger, 'logger'):
                self.logger.logger.debug(message)
            else:
                self.logger.debug(message)
    
    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            if hasattr(self.logger, 'logger'):
                self.logger.logger.info(message)
            else:
                self.logger.info(message)
    
    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            if hasattr(self.logger, 'logger'):
                self.logger.logger.warning(message)
            else:
                self.logger.warning(message)
    
    def _log_error(self, message: str) -> None:
        """Log error message."""
        if self.logger:
            if hasattr(self.logger, 'logger'):
                self.logger.logger.error(message)
            else:
                self.logger.error(message)
    
    def get_stats(self) -> RetryStats:
        """Get retry statistics."""
        return self.stats
    
    def get_recent_attempts(self, limit: int = 10) -> List[RetryAttempt]:
        """Get recent retry attempts.
        
        Args:
            limit: Maximum number of attempts to return
            
        Returns:
            List of recent retry attempts
        """
        return self.attempts_history[-limit:]
    
    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self.stats = RetryStats()
        self.attempts_history.clear()


class RetryManager:
    """Manager for multiple retry handlers."""
    
    def __init__(self, logger: Optional[ILogger] = None, error_handler: Optional[ErrorHandler] = None):
        self.logger = logger
        self.error_handler = error_handler
        self.retry_handlers: Dict[str, RetryHandler] = {}
    
    def get_retry_handler(
        self,
        name: str,
        config: Optional[RetryConfig] = None
    ) -> RetryHandler:
        """Get or create a retry handler.
        
        Args:
            name: Retry handler name
            config: Optional configuration
            
        Returns:
            RetryHandler instance
        """
        if name not in self.retry_handlers:
            self.retry_handlers[name] = RetryHandler(
                name=name,
                config=config,
                logger=self.logger,
                error_handler=self.error_handler
            )
        
        return self.retry_handlers[name]
    
    def get_all_stats(self) -> Dict[str, RetryStats]:
        """Get statistics for all retry handlers."""
        return {
            name: handler.get_stats()
            for name, handler in self.retry_handlers.items()
        }
    
    def reset_all_stats(self) -> None:
        """Reset statistics for all retry handlers."""
        for handler in self.retry_handlers.values():
            handler.reset_stats()


# Decorator for easy retry functionality
def retry(
    name: str = None,
    config: RetryConfig = None,
    logger: ILogger = None,
    error_handler: ErrorHandler = None
):
    """Decorator to add retry functionality to functions.
    
    Args:
        name: Operation name for logging
        config: Retry configuration
        logger: Logger instance
        error_handler: Error handler instance
        
    Returns:
        Decorated function
    """
    def decorator(func):
        operation_name = name or f"{func.__module__}.{func.__name__}"
        retry_handler = RetryHandler(
            name=operation_name,
            config=config,
            logger=logger,
            error_handler=error_handler
        )
        
        async def async_wrapper(*args, **kwargs):
            return await retry_handler.execute(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(retry_handler.execute(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator