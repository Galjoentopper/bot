"""Advanced circuit breaker implementation for external services."""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from .interfaces import ILogger


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls are failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    expected_exception: Optional[Exception] = None
    fallback_function: Optional[Callable] = None


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_successes: deque = field(default_factory=lambda: deque(maxlen=100))


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str, circuit_name: str, last_failure: Optional[str] = None):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.last_failure = last_failure


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with metrics and fallback support."""

    def __init__(self, name: str, config: CircuitBreakerConfig, logger: Optional[ILogger] = None):
        self.name = name
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._next_attempt_time = 0
        self._lock = threading.RLock()
        self._metrics = CircuitBreakerMetrics()

        # Correlation tracking
        self._active_requests: Dict[str, float] = {}

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        return self._metrics

    def _record_success(self, correlation_id: Optional[str] = None):
        """Record a successful call."""
        with self._lock:
            current_time = time.time()
            self._success_count += 1
            self._metrics.successful_requests += 1
            self._metrics.last_success_time = current_time
            self._metrics.recent_successes.append(current_time)

            if correlation_id and correlation_id in self._active_requests:
                duration = current_time - self._active_requests[correlation_id]
                del self._active_requests[correlation_id]

                if self.logger:
                    self.logger.log_info(
                        f"Circuit breaker {self.name} - request succeeded",
                        context={
                            "correlation_id": correlation_id,
                            "duration_ms": duration * 1000,
                            "state": self._state.value,
                            "success_count": self._success_count,
                        },
                    )

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    self._change_state(CircuitState.CLOSED)
                    self._failure_count = 0

    def _record_failure(self, exception: Exception, correlation_id: Optional[str] = None):
        """Record a failed call."""
        with self._lock:
            current_time = time.time()
            self._failure_count += 1
            self._last_failure_time = current_time
            self._metrics.failed_requests += 1
            self._metrics.last_failure_time = current_time
            self._metrics.recent_failures.append(current_time)

            if correlation_id and correlation_id in self._active_requests:
                duration = current_time - self._active_requests[correlation_id]
                del self._active_requests[correlation_id]

                if self.logger:
                    self.logger.log_error(
                        f"Circuit breaker {self.name} - request failed",
                        context={
                            "correlation_id": correlation_id,
                            "duration_ms": duration * 1000,
                            "state": self._state.value,
                            "failure_count": self._failure_count,
                            "exception_type": type(exception).__name__,
                        },
                        exception=exception,
                    )

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._change_state(CircuitState.OPEN)
                    self._next_attempt_time = current_time + self.config.timeout

            elif self._state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)
                self._next_attempt_time = current_time + self.config.timeout

    def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state."""
        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1

        if self.logger:
            self.logger.log_info(
                f"Circuit breaker {self.name} state changed",
                context={
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "failure_count": self._failure_count,
                    "success_count": self._success_count,
                },
            )

    def _should_attempt_call(self) -> bool:
        """Check if we should attempt the call."""
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            return time.time() >= self._next_attempt_time
        else:  # HALF_OPEN
            return True

    def _handle_open_circuit(self):
        """Handle calls when circuit is open."""
        if time.time() >= self._next_attempt_time:
            self._change_state(CircuitState.HALF_OPEN)
            self._success_count = 0
            return True

        # Circuit is open, try fallback
        if self.config.fallback_function:
            try:
                return self.config.fallback_function()
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        f"Circuit breaker {self.name} fallback failed", exception=e
                    )

        raise CircuitBreakerException(
            f"Circuit breaker {self.name} is open", self.name, str(self._last_failure_time)
        )

    def call(self, func: Callable, *args, **kwargs):
        """Execute a function with circuit breaker protection."""
        correlation_id = str(uuid.uuid4())

        with self._lock:
            self._metrics.total_requests += 1
            self._active_requests[correlation_id] = time.time()

            if not self._should_attempt_call():
                return self._handle_open_circuit()

        try:
            result = func(*args, **kwargs)
            self._record_success(correlation_id)
            return result

        except Exception as e:
            # Check if this is an expected exception type
            if self.config.expected_exception and isinstance(e, self.config.expected_exception):
                self._record_failure(e, correlation_id)
            else:
                # For unexpected exceptions, record failure but re-raise
                self._record_failure(e, correlation_id)
            raise

    async def call_async(self, coro_func: Callable[..., Awaitable], *args, **kwargs):
        """Execute an async function with circuit breaker protection."""
        correlation_id = str(uuid.uuid4())

        with self._lock:
            self._metrics.total_requests += 1
            self._active_requests[correlation_id] = time.time()

            if not self._should_attempt_call():
                return self._handle_open_circuit()

        try:
            result = await coro_func(*args, **kwargs)
            self._record_success(correlation_id)
            return result

        except Exception as e:
            self._record_failure(e, correlation_id)
            raise

    def __call__(self, func: Callable):
        """Decorator usage."""
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)

            return sync_wrapper

    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._next_attempt_time = 0
            self._metrics = CircuitBreakerMetrics()

            if self.logger:
                self.logger.log_info(f"Circuit breaker {self.name} reset")

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "next_attempt_time": self._next_attempt_time,
                "metrics": {
                    "total_requests": self._metrics.total_requests,
                    "successful_requests": self._metrics.successful_requests,
                    "failed_requests": self._metrics.failed_requests,
                    "success_rate": (
                        self._metrics.successful_requests / self._metrics.total_requests
                        if self._metrics.total_requests > 0
                        else 0
                    ),
                    "state_changes": self._metrics.state_changes,
                    "active_requests": len(self._active_requests),
                },
            }


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""

    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger
        self._breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self._lock = threading.RLock()

    def create_breaker(self, name: str, config: CircuitBreakerConfig) -> AdvancedCircuitBreaker:
        """Create a new circuit breaker."""
        with self._lock:
            if name in self._breakers:
                return self._breakers[name]

            breaker = AdvancedCircuitBreaker(name, config, self.logger)
            self._breakers[name] = breaker
            return breaker

    def get_breaker(self, name: str) -> Optional[AdvancedCircuitBreaker]:
        """Get existing circuit breaker."""
        return self._breakers.get(name)

    def remove_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False

    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        with self._lock:
            return {name: breaker.get_status() for name, breaker in self._breakers.items()}


# Global circuit breaker manager instance
_global_cb_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    return _global_cb_manager


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout: float = 60.0,
    expected_exception: Optional[Exception] = None,
    fallback_function: Optional[Callable] = None,
):
    """Circuit breaker decorator."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        expected_exception=expected_exception,
        fallback_function=fallback_function,
    )

    breaker = _global_cb_manager.create_breaker(name, config)
    return breaker
