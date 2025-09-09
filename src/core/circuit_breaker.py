"""Circuit breaker pattern implementation for external API calls."""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from .error_handler import ErrorHandler, ErrorSeverity, NetworkException
from .interfaces import ILogger


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds
    expected_exception: type = Exception  # Exception type that triggers circuit


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opened_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Circuit breaker implementation for resilient external calls."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        logger: Optional[ILogger] = None,
        error_handler: Optional[ErrorHandler] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logger
        self.error_handler = error_handler

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.stats = CircuitBreakerStats()

        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenException: When circuit is open
            Original exception: When function fails
        """
        async with self._lock:
            self.stats.total_requests += 1

            # Check circuit state
            await self._update_state()

            if self.state == CircuitState.OPEN:
                self._log_info(f"Circuit breaker {self.name} is OPEN - failing fast")
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is open")

            # Execute the function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=self.config.timeout
                    )
                else:
                    result = func(*args, **kwargs)

                await self._on_success()
                return result

            except asyncio.TimeoutError as e:
                self.stats.timeouts += 1
                await self._on_failure(e)
                raise
            except self.config.expected_exception as e:
                await self._on_failure(e)
                raise
            except Exception as e:
                # Unexpected exception - still count as failure but log it
                self._log_warning(f"Unexpected exception in {self.name}: {e}")
                await self._on_failure(e)
                raise

    async def _update_state(self) -> None:
        """Update circuit breaker state based on current conditions."""
        now = time.time()

        if self.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if (
                self.last_failure_time
                and now - self.last_failure_time >= self.config.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self._log_info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")

        elif self.state == CircuitState.HALF_OPEN:
            # Check if we should close the circuit
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self._log_info(f"Circuit breaker {self.name} transitioning to CLOSED")

    async def _on_success(self) -> None:
        """Handle successful function execution."""
        self.stats.successful_requests += 1
        self.stats.last_success_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed function execution."""
        self.stats.failed_requests += 1
        self.stats.last_failure_time = datetime.now()
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.stats.circuit_opened_count += 1
                self._log_warning(
                    f"Circuit breaker {self.name} OPENED after {self.failure_count} failures"
                )

        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self.state = CircuitState.OPEN
            self.failure_count += 1
            self.stats.circuit_opened_count += 1
            self._log_warning(f"Circuit breaker {self.name} OPENED from HALF_OPEN state")

        # Report error to error handler if available
        if self.error_handler:
            network_error = NetworkException(
                f"Circuit breaker {self.name} failure: {str(exception)}",
                component="circuit_breaker",
                operation="external_call",
                details={
                    "circuit_name": self.name,
                    "state": self.state.value,
                    "failure_count": self.failure_count,
                    "exception_type": type(exception).__name__,
                },
            )
            self.error_handler.handle_error(network_error, attempt_recovery=False)

    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(message)
            else:
                self.logger.info(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.warning(message)
            else:
                self.logger.warning(message)

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self.stats

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._log_info(f"Circuit breaker {self.name} manually reset to CLOSED")

    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        self._log_warning(f"Circuit breaker {self.name} manually forced to OPEN")


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        error_handler: Optional[ErrorHandler] = None,
    ):
        self.logger = logger
        self.error_handler = error_handler
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def get_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Optional configuration

        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                config=config,
                logger=self.logger,
                error_handler=self.error_handler,
            )

        return self.circuit_breakers[name]

    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}

    def get_all_states(self) -> Dict[str, CircuitState]:
        """Get states for all circuit breakers."""
        return {name: cb.get_state() for name, cb in self.circuit_breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()

    def force_open_all(self) -> None:
        """Force all circuit breakers to open state."""
        for cb in self.circuit_breakers.values():
            cb.force_open()
