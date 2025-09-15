"""Resilience patterns including retry, timeout, and bulkhead."""

import asyncio
import logging
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from .interfaces import ILogger

T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry strategies."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    retryable_exceptions: Optional[Set[Type[Exception]]] = None


@dataclass
class TimeoutConfig:
    """Timeout configuration."""

    timeout_seconds: float = 30.0
    cancel_on_timeout: bool = True


@dataclass
class BulkheadConfig:
    """Bulkhead configuration."""

    max_concurrent: int = 10
    max_queue_size: int = 100
    timeout_seconds: float = 30.0


class ResilienceException(Exception):
    """Base exception for resilience patterns."""

    pass


class RetryExhaustedException(ResilienceException):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Exhausted {attempts} retry attempts. Last error: {last_exception}")


class TimeoutException(ResilienceException):
    """Raised when operation times out."""

    pass


class BulkheadRejectedException(ResilienceException):
    """Raised when bulkhead rejects request."""

    pass


class RetryHandler:
    """Advanced retry handler with different strategies."""

    def __init__(self, config: RetryConfig, logger: Optional[ILogger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.initial_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.initial_delay * attempt
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            base_delay = self.config.initial_delay * (
                self.config.backoff_multiplier ** (attempt - 1)
            )
            if self.config.jitter:
                delay = base_delay * (0.5 + random.random() * 0.5)  # 50-100% of base delay
            else:
                delay = base_delay
        else:
            delay = self.config.initial_delay

        return min(delay, self.config.max_delay)

    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        if self.config.retryable_exceptions is None:
            # Default retryable exceptions
            retryable_types = (
                ConnectionError,
                TimeoutError,
                BulkheadRejectedException,
            )
            return isinstance(exception, retryable_types)

        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        correlation_id = str(uuid.uuid4())
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if self.logger:
                    self.logger.log_info(
                        f"Retry attempt {attempt}/{self.config.max_attempts}",
                        context={
                            "correlation_id": correlation_id,
                            "function": func.__name__,
                            "attempt": attempt,
                        },
                    )

                result = func(*args, **kwargs)

                if self.logger and attempt > 1:
                    self.logger.log_info(
                        f"Retry succeeded on attempt {attempt}",
                        context={
                            "correlation_id": correlation_id,
                            "function": func.__name__,
                            "total_attempts": attempt,
                        },
                    )

                return result

            except Exception as e:
                last_exception = e

                if not self._is_retryable(e):
                    if self.logger:
                        self.logger.log_error(
                            f"Non-retryable exception on attempt {attempt}",
                            context={
                                "correlation_id": correlation_id,
                                "function": func.__name__,
                                "exception_type": type(e).__name__,
                            },
                            exception=e,
                        )
                    raise

                if attempt == self.config.max_attempts:
                    if self.logger:
                        self.logger.log_error(
                            f"All retry attempts exhausted",
                            context={
                                "correlation_id": correlation_id,
                                "function": func.__name__,
                                "total_attempts": attempt,
                            },
                            exception=e,
                        )
                    break

                delay = self._calculate_delay(attempt)

                if self.logger:
                    self.logger.log_warning(
                        f"Retry attempt {attempt} failed, retrying in {delay:.2f}s",
                        context={
                            "correlation_id": correlation_id,
                            "function": func.__name__,
                            "delay_seconds": delay,
                            "exception_type": type(e).__name__,
                        },
                    )

                time.sleep(delay)

        raise RetryExhaustedException(self.config.max_attempts, last_exception)

    async def execute_async(self, coro_func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute async function with retry logic."""
        correlation_id = str(uuid.uuid4())
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if self.logger:
                    self.logger.log_info(
                        f"Async retry attempt {attempt}/{self.config.max_attempts}",
                        context={
                            "correlation_id": correlation_id,
                            "function": coro_func.__name__,
                            "attempt": attempt,
                        },
                    )

                result = await coro_func(*args, **kwargs)

                if self.logger and attempt > 1:
                    self.logger.log_info(
                        f"Async retry succeeded on attempt {attempt}",
                        context={
                            "correlation_id": correlation_id,
                            "function": coro_func.__name__,
                            "total_attempts": attempt,
                        },
                    )

                return result

            except Exception as e:
                last_exception = e

                if not self._is_retryable(e):
                    if self.logger:
                        self.logger.log_error(
                            f"Non-retryable async exception on attempt {attempt}",
                            context={
                                "correlation_id": correlation_id,
                                "function": coro_func.__name__,
                                "exception_type": type(e).__name__,
                            },
                            exception=e,
                        )
                    raise

                if attempt == self.config.max_attempts:
                    if self.logger:
                        self.logger.log_error(
                            f"All async retry attempts exhausted",
                            context={
                                "correlation_id": correlation_id,
                                "function": coro_func.__name__,
                                "total_attempts": attempt,
                            },
                            exception=e,
                        )
                    break

                delay = self._calculate_delay(attempt)

                if self.logger:
                    self.logger.log_warning(
                        f"Async retry attempt {attempt} failed, retrying in {delay:.2f}s",
                        context={
                            "correlation_id": correlation_id,
                            "function": coro_func.__name__,
                            "delay_seconds": delay,
                            "exception_type": type(e).__name__,
                        },
                    )

                await asyncio.sleep(delay)

        raise RetryExhaustedException(self.config.max_attempts, last_exception)


class TimeoutHandler:
    """Timeout handler for operations."""

    def __init__(self, config: TimeoutConfig, logger: Optional[ILogger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with timeout."""
        correlation_id = str(uuid.uuid4())

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)

            try:
                result = future.result(timeout=self.config.timeout_seconds)

                if self.logger:
                    self.logger.log_info(
                        f"Function completed within timeout",
                        context={
                            "correlation_id": correlation_id,
                            "function": func.__name__,
                            "timeout_seconds": self.config.timeout_seconds,
                        },
                    )

                return result

            except FutureTimeoutError:
                if self.config.cancel_on_timeout:
                    future.cancel()

                if self.logger:
                    self.logger.log_error(
                        f"Function timed out after {self.config.timeout_seconds}s",
                        context={
                            "correlation_id": correlation_id,
                            "function": func.__name__,
                            "timeout_seconds": self.config.timeout_seconds,
                        },
                    )

                raise TimeoutException(
                    f"Function {func.__name__} timed out after {self.config.timeout_seconds}s"
                )

    async def execute_async(self, coro_func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute async function with timeout."""
        correlation_id = str(uuid.uuid4())

        try:
            result = await asyncio.wait_for(
                coro_func(*args, **kwargs), timeout=self.config.timeout_seconds
            )

            if self.logger:
                self.logger.log_info(
                    f"Async function completed within timeout",
                    context={
                        "correlation_id": correlation_id,
                        "function": coro_func.__name__,
                        "timeout_seconds": self.config.timeout_seconds,
                    },
                )

            return result

        except asyncio.TimeoutError:
            if self.logger:
                self.logger.log_error(
                    f"Async function timed out after {self.config.timeout_seconds}s",
                    context={
                        "correlation_id": correlation_id,
                        "function": coro_func.__name__,
                        "timeout_seconds": self.config.timeout_seconds,
                    },
                )

            raise TimeoutException(
                f"Async function {coro_func.__name__} timed out after {self.config.timeout_seconds}s"
            )


class BulkheadHandler:
    """Bulkhead handler for resource isolation."""

    def __init__(self, name: str, config: BulkheadConfig, logger: Optional[ILogger] = None):
        self.name = name
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self._semaphore = threading.Semaphore(config.max_concurrent)
        self._queue_size = 0
        self._lock = threading.RLock()

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function within bulkhead."""
        correlation_id = str(uuid.uuid4())

        with self._lock:
            if self._queue_size >= self.config.max_queue_size:
                if self.logger:
                    self.logger.log_warning(
                        f"Bulkhead {self.name} rejected request - queue full",
                        context={
                            "correlation_id": correlation_id,
                            "queue_size": self._queue_size,
                            "max_queue_size": self.config.max_queue_size,
                        },
                    )
                raise BulkheadRejectedException(f"Bulkhead {self.name} queue is full")

            self._queue_size += 1

        try:
            acquired = self._semaphore.acquire(timeout=self.config.timeout_seconds)

            if not acquired:
                if self.logger:
                    self.logger.log_warning(
                        f"Bulkhead {self.name} rejected request - timeout acquiring semaphore",
                        context={
                            "correlation_id": correlation_id,
                            "timeout_seconds": self.config.timeout_seconds,
                        },
                    )
                raise BulkheadRejectedException(f"Bulkhead {self.name} timeout acquiring resource")

            try:
                if self.logger:
                    self.logger.log_info(
                        f"Bulkhead {self.name} executing request",
                        context={
                            "correlation_id": correlation_id,
                            "function": func.__name__,
                        },
                    )

                result = func(*args, **kwargs)

                if self.logger:
                    self.logger.log_info(
                        f"Bulkhead {self.name} completed request",
                        context={
                            "correlation_id": correlation_id,
                            "function": func.__name__,
                        },
                    )

                return result

            finally:
                self._semaphore.release()

        finally:
            with self._lock:
                self._queue_size -= 1

    def get_status(self) -> Dict[str, Any]:
        """Get bulkhead status."""
        with self._lock:
            available = self._semaphore._value if hasattr(self._semaphore, "_value") else 0

            return {
                "name": self.name,
                "max_concurrent": self.config.max_concurrent,
                "available_permits": available,
                "active_requests": self.config.max_concurrent - available,
                "queue_size": self._queue_size,
                "max_queue_size": self.config.max_queue_size,
            }


# Decorator functions
def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
):
    """Retry decorator."""
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        backoff_multiplier=backoff_multiplier,
        strategy=strategy,
        retryable_exceptions=retryable_exceptions,
    )

    handler = RetryHandler(config)

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await handler.execute_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return handler.execute(func, *args, **kwargs)

            return sync_wrapper

    return decorator


def timeout(timeout_seconds: float = 30.0, cancel_on_timeout: bool = True):
    """Timeout decorator."""
    config = TimeoutConfig(timeout_seconds=timeout_seconds, cancel_on_timeout=cancel_on_timeout)

    handler = TimeoutHandler(config)

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await handler.execute_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return handler.execute(func, *args, **kwargs)

            return sync_wrapper

    return decorator


# Global bulkhead instances
_bulkheads: Dict[str, BulkheadHandler] = {}
_bulkhead_lock = threading.RLock()


def get_bulkhead(name: str, config: BulkheadConfig) -> BulkheadHandler:
    """Get or create a bulkhead."""
    with _bulkhead_lock:
        if name not in _bulkheads:
            _bulkheads[name] = BulkheadHandler(name, config)
        return _bulkheads[name]


def bulkhead(
    name: str,
    max_concurrent: int = 10,
    max_queue_size: int = 100,
    timeout_seconds: float = 30.0,
):
    """Bulkhead decorator."""
    config = BulkheadConfig(
        max_concurrent=max_concurrent,
        max_queue_size=max_queue_size,
        timeout_seconds=timeout_seconds,
    )

    handler = get_bulkhead(name, config)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return handler.execute(func, *args, **kwargs)

        return wrapper

    return decorator
