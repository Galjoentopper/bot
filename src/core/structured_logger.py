"""Structured logging with correlation IDs and observability."""

import contextlib
import inspect
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import structlog
from structlog.dev import ConsoleRenderer
from structlog.stdlib import LoggerFactory as StructlogLoggerFactory


class LogLevel(str, Enum):
    """Log levels."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerType(str, Enum):
    """Logger types."""

    APPLICATION = "application"
    TRADING = "trading"
    MODEL = "model"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    AUDIT = "audit"


@dataclass
class LogContext:
    """Log context with correlation tracking."""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Trading specific
    symbol: Optional[str] = None
    model_type: Optional[str] = None
    trade_id: Optional[str] = None

    # Performance
    operation: Optional[str] = None
    duration_ms: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}

    def update(self, **kwargs) -> "LogContext":
        """Create new context with updated values."""
        new_context = LogContext(**self.to_dict())
        for key, value in kwargs.items():
            if hasattr(new_context, key):
                setattr(new_context, key, value)
            else:
                new_context.metadata[key] = value
        return new_context


class CorrelationContextManager:
    """Thread-local correlation context manager."""

    def __init__(self):
        self._local = threading.local()

    def get_context(self) -> LogContext:
        """Get current correlation context."""
        if not hasattr(self._local, "context"):
            self._local.context = LogContext()
        return self._local.context

    def set_context(self, context: LogContext):
        """Set correlation context."""
        self._local.context = context

    def update_context(self, **kwargs) -> LogContext:
        """Update current context."""
        current = self.get_context()
        new_context = current.update(**kwargs)
        self.set_context(new_context)
        return new_context

    @contextlib.contextmanager
    def context(self, **kwargs):
        """Context manager for temporary context updates."""
        old_context = self.get_context()
        new_context = old_context.update(**kwargs)
        self.set_context(new_context)
        try:
            yield new_context
        finally:
            self.set_context(old_context)


# Global correlation context manager
_correlation_manager = CorrelationContextManager()


def get_correlation_manager() -> CorrelationContextManager:
    """Get global correlation manager."""
    return _correlation_manager


class TradingBotProcessor:
    """Custom structlog processor for trading bot."""

    def __init__(self, logger_type: LoggerType = LoggerType.APPLICATION):
        self.logger_type = logger_type

    def __call__(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process log entry."""
        # Add timestamp
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add logger type
        event_dict["logger_type"] = self.logger_type.value

        # Add correlation context
        context = _correlation_manager.get_context()
        event_dict.update(context.to_dict())

        # Add caller information
        frame = inspect.currentframe()
        try:
            # Skip structlog frames to get to actual caller
            while frame and "structlog" in str(frame.f_code.co_filename):
                frame = frame.f_back
            if frame and frame.f_back:
                caller_frame = frame.f_back
                event_dict.update(
                    {
                        "file": Path(caller_frame.f_code.co_filename).name,
                        "function": caller_frame.f_code.co_name,
                        "line": caller_frame.f_lineno,
                    }
                )
        finally:
            del frame

        # Add process/thread info
        event_dict.update(
            {
                "process_id": str(os.getpid()) if "os" in sys.modules else None,
                "thread_id": str(threading.get_ident()),
            }
        )

        return event_dict


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data, default=str, ensure_ascii=False)


class StructuredLogger:
    """Enhanced structured logger for trading bot."""

    def __init__(
        self,
        name: str,
        logger_type: LoggerType = LoggerType.APPLICATION,
        level: LogLevel = LogLevel.INFO,
        output_format: str = "json",  # "json" or "console"
        file_path: Optional[str] = None,
    ):
        self.name = name
        self.logger_type = logger_type
        self.output_format = output_format

        # Configure structlog
        processors = [
            TradingBotProcessor(logger_type),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]

        if output_format == "console":
            processors.append(ConsoleRenderer())
        else:
            processors.append(structlog.processors.JSONRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.value)),
            logger_factory=StructlogLoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

        # Create logger
        self._logger = structlog.get_logger(name)

        # Configure standard library logger if file output is needed
        if file_path:
            stdlib_logger = logging.getLogger(name)
            stdlib_logger.setLevel(getattr(logging, level.value))

            # File handler
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(JSONFormatter())
            stdlib_logger.addHandler(file_handler)

    def _log(self, level: str, message: str, **kwargs):
        """Internal log method."""
        getattr(self._logger, level.lower())(message, **kwargs)

    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self._log("debug", message, log_level="TRACE", **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("warning", message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
            kwargs["exception_traceback"] = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        self._log("error", message, **kwargs)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
            kwargs["exception_traceback"] = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        self._log("critical", message, **kwargs)

    # Trading-specific log methods
    def log_trade(
        self,
        symbol: str,
        action: str,
        amount: float,
        price: Optional[float] = None,
        trade_id: Optional[str] = None,
        **kwargs,
    ):
        """Log trading action."""
        with _correlation_manager.context(
            symbol=symbol, trade_id=trade_id or str(uuid.uuid4()), operation="trade"
        ):
            self.info(
                f"Trade executed: {action} {amount} {symbol}",
                action=action,
                amount=amount,
                price=price,
                **kwargs,
            )

    def log_model_prediction(
        self,
        symbol: str,
        model_type: str,
        prediction: float,
        confidence: Optional[float] = None,
        features_hash: Optional[str] = None,
        **kwargs,
    ):
        """Log model prediction."""
        with _correlation_manager.context(
            symbol=symbol, model_type=model_type, operation="prediction"
        ):
            self.info(
                f"Model prediction: {model_type} for {symbol}",
                prediction=prediction,
                confidence=confidence,
                features_hash=features_hash,
                **kwargs,
            )

    def log_performance(self, operation: str, duration_ms: float, success: bool = True, **kwargs):
        """Log performance metrics."""
        with _correlation_manager.context(operation=operation, duration_ms=duration_ms):
            level = "info" if success else "warning"
            self._log(
                level,
                f"Operation {operation} completed in {duration_ms:.2f}ms",
                success=success,
                **kwargs,
            )

    def log_system_event(self, event_type: str, event_data: Dict[str, Any], **kwargs):
        """Log system event."""
        with _correlation_manager.context(operation=f"system_{event_type}"):
            self.info(
                f"System event: {event_type}",
                event_type=event_type,
                event_data=event_data,
                **kwargs,
            )


class LoggerFactory:
    """Factory for creating structured loggers."""

    _loggers: Dict[str, StructuredLogger] = {}
    _lock = threading.Lock()

    @classmethod
    def get_logger(
        cls,
        name: str,
        logger_type: LoggerType = LoggerType.APPLICATION,
        level: LogLevel = LogLevel.INFO,
        output_format: str = "json",
        file_path: Optional[str] = None,
    ) -> StructuredLogger:
        """Get or create logger."""
        logger_key = f"{name}_{logger_type.value}_{level.value}_{output_format}"

        with cls._lock:
            if logger_key not in cls._loggers:
                cls._loggers[logger_key] = StructuredLogger(
                    name=name,
                    logger_type=logger_type,
                    level=level,
                    output_format=output_format,
                    file_path=file_path,
                )
            return cls._loggers[logger_key]


# Performance logging decorator
def log_performance(
    logger: Optional[StructuredLogger] = None,
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
):
    """Decorator for logging function performance."""

    def decorator(func: Callable) -> Callable:
        nonlocal logger, operation

        if logger is None:
            logger = LoggerFactory.get_logger(
                f"{func.__module__}.{func.__qualname__}", LoggerType.PERFORMANCE
            )

        if operation is None:
            operation = f"{func.__module__}.{func.__name__}"

        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                correlation_id = str(uuid.uuid4())

                log_data = {"correlation_id": correlation_id, "operation": operation}

                if log_args:
                    log_data.update(
                        {
                            "args": args[:5],  # Limit args to prevent huge logs
                            "kwargs": {k: v for k, v in list(kwargs.items())[:5]},
                        }
                    )

                try:
                    with _correlation_manager.context(
                        correlation_id=correlation_id, operation=operation
                    ):
                        logger.debug(f"Starting operation: {operation}", **log_data)
                        result = await func(*args, **kwargs)

                        duration_ms = (time.time() - start_time) * 1000

                        result_log_data = {**log_data, "duration_ms": duration_ms, "success": True}
                        if log_result and result is not None:
                            result_log_data["result"] = str(result)[:200]  # Truncate large results

                        logger.log_performance(operation, duration_ms, True, **result_log_data)
                        return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.log_performance(operation, duration_ms, False, error=str(e), **log_data)
                    raise

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                correlation_id = str(uuid.uuid4())

                log_data = {"correlation_id": correlation_id, "operation": operation}

                if log_args:
                    log_data.update(
                        {"args": args[:5], "kwargs": {k: v for k, v in list(kwargs.items())[:5]}}
                    )

                try:
                    with _correlation_manager.context(
                        correlation_id=correlation_id, operation=operation
                    ):
                        logger.debug(f"Starting operation: {operation}", **log_data)
                        result = func(*args, **kwargs)

                        duration_ms = (time.time() - start_time) * 1000

                        result_log_data = {**log_data, "duration_ms": duration_ms, "success": True}
                        if log_result and result is not None:
                            result_log_data["result"] = str(result)[:200]

                        logger.log_performance(operation, duration_ms, True, **result_log_data)
                        return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.log_performance(operation, duration_ms, False, error=str(e), **log_data)
                    raise

            return sync_wrapper

    return decorator


# Convenience functions
def get_logger(name: str, logger_type: LoggerType = LoggerType.APPLICATION) -> StructuredLogger:
    """Get a structured logger."""
    return LoggerFactory.get_logger(name, logger_type)


def get_trading_logger(name: str) -> StructuredLogger:
    """Get a trading-specific logger."""
    return LoggerFactory.get_logger(name, LoggerType.TRADING)


def get_model_logger(name: str) -> StructuredLogger:
    """Get a model-specific logger."""
    return LoggerFactory.get_logger(name, LoggerType.MODEL)


def get_performance_logger(name: str) -> StructuredLogger:
    """Get a performance logger."""
    return LoggerFactory.get_logger(name, LoggerType.PERFORMANCE)


def correlation_context(**kwargs):
    """Context manager for correlation tracking."""
    return _correlation_manager.context(**kwargs)
