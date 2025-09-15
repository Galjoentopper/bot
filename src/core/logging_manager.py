"""
Centralized Logging Manager for Trading Bot Production System
============================================================

This module provides a unified logging architecture for the production trading bot,
replacing the multiple conflicting logging systems with a single, structured approach.

Key Features:
- Purpose-based log categories (trading, model, system, debug)
- Structured logging with JSON support for analysis
- Proper log rotation and file management
- Environment-aware configuration
- Backward compatibility with existing code

Note: This does NOT affect Paperspace training, which uses its own simple console logging.
"""

import json
import logging
import logging.config
import logging.handlers
import os
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LoggerType(str, Enum):
    """Supported logger types for different system components."""

    TRADING = "trading"  # Trade executions, decisions, portfolio updates
    MODEL = "model"  # Model predictions, ensemble decisions, ML operations
    SYSTEM = "system"  # System operations, startup, configuration, errors
    DEBUG = "debug"  # Debug information, development logs
    PERFORMANCE = "performance"  # Performance metrics, timing, resource usage


class TradingBotLogger:
    """
    Centralized logging manager for the trading bot production system.

    Provides structured, purpose-based logging with proper file management
    and environment-aware configuration.
    """

    _instance = None
    _lock = threading.Lock()
    _loggers = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging_system()
            self._initialized = True

    def _setup_logging_system(self):
        """Initialize the logging system with proper configuration."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Load configuration
        config = self._get_logging_config()

        # Apply configuration
        logging.config.dictConfig(config)

        # Set up root logger to prevent unwanted output
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)

        # Log initialization
        system_logger = logging.getLogger("trading_bot.system.logging_manager")
        system_logger.info("✅ Trading Bot Logging System initialized")

    def _get_logging_config(self) -> Dict[str, Any]:
        """Get the logging configuration dictionary."""

        # Determine log levels based on environment
        env = os.getenv("TRADING_ENV", "development")
        debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

        if env == "production":
            default_level = "INFO"
            debug_level = "ERROR"  # Only errors in production
        elif debug_mode:
            default_level = "DEBUG"
            debug_level = "DEBUG"
        else:
            default_level = "INFO"
            debug_level = "WARNING"

        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "trading": {
                    "format": "%(asctime)s | TRADE | %(levelname)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "model": {
                    "format": "%(asctime)s | MODEL | %(levelname)s | %(funcName)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "system": {
                    "format": "%(asctime)s | SYS | %(name)s | %(levelname)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "performance": {
                    "format": "%(asctime)s | PERF | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "debug": {
                    "format": "%(asctime)s | DEBUG | %(name)s:%(lineno)d | %(levelname)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "console": {
                    "format": "%(asctime)s | %(levelname)s | %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "csv": {"format": "%(message)s"},  # Raw CSV data
            },
            "handlers": {
                # Trading operations - Critical data, never lost
                "trading_file": {
                    "class": "logging.FileHandler",
                    "filename": "logs/trading.log",
                    "formatter": "trading",
                    "level": "INFO",
                    "encoding": "utf-8",
                },
                # Trade executions as CSV - Structured data for analysis
                "trades_csv": {
                    "class": "logging.FileHandler",
                    "filename": "logs/trades_report.csv",
                    "formatter": "csv",
                    "level": "INFO",
                    "encoding": "utf-8",
                },
                # Model predictions and ML operations - Rotated daily
                "model_file": {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "filename": "logs/models.log",
                    "when": "midnight",
                    "interval": 1,
                    "backupCount": 7,
                    "formatter": "model",
                    "level": "INFO",
                    "encoding": "utf-8",
                },
                # System operations - Rotated by size
                "system_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "logs/system.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "formatter": "system",
                    "level": "INFO",
                    "encoding": "utf-8",
                },
                # Performance metrics
                "performance_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "logs/performance.log",
                    "maxBytes": 5242880,  # 5MB
                    "backupCount": 3,
                    "formatter": "performance",
                    "level": "INFO",
                    "encoding": "utf-8",
                },
                # Debug logs - Only when needed, smaller retention
                "debug_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "logs/debug.log",
                    "maxBytes": 2097152,  # 2MB
                    "backupCount": 2,
                    "formatter": "debug",
                    "level": debug_level,
                    "encoding": "utf-8",
                },
                # Console output - Clean, minimal
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "level": default_level,
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                # Trading operations logger
                "trading_bot.trading": {
                    "handlers": ["trading_file", "console"],
                    "level": "INFO",
                    "propagate": False,
                },
                # Model operations logger
                "trading_bot.model": {
                    "handlers": ["model_file", "console"],
                    "level": "INFO",
                    "propagate": False,
                },
                # System operations logger
                "trading_bot.system": {
                    "handlers": ["system_file", "console"],
                    "level": "INFO",
                    "propagate": False,
                },
                # Performance logger
                "trading_bot.performance": {
                    "handlers": ["performance_file"],
                    "level": "INFO",
                    "propagate": False,
                },
                # Debug logger
                "trading_bot.debug": {
                    "handlers": ["debug_file"],
                    "level": debug_level,
                    "propagate": False,
                },
            },
            "root": {"level": "WARNING", "handlers": ["console"]},
        }

    @classmethod
    def get_logger(cls, name: str, logger_type: LoggerType = LoggerType.SYSTEM) -> logging.Logger:
        """
        Get a configured logger for a specific component and type.

        Args:
            name: Component name (e.g., "trader", "risk_manager", "telegram")
            logger_type: Type of logger (trading, model, system, debug, performance)

        Returns:
            Configured logger instance
        """
        # Ensure singleton is initialized
        instance = cls()

        # Create unique key
        key = f"{logger_type.value}:{name}"

        if key not in cls._loggers:
            logger_name = f"trading_bot.{logger_type.value}.{name}"
            cls._loggers[key] = logging.getLogger(logger_name)

        return cls._loggers[key]

    @classmethod
    def get_trade_csv_logger(cls) -> logging.Logger:
        """Get the CSV logger for trade data."""
        instance = cls()
        if "trades_csv" not in cls._loggers:
            cls._loggers["trades_csv"] = logging.getLogger("trading_bot.trades_csv")

            # Add the CSV handler specifically
            csv_handler = logging.FileHandler("logs/trades_report.csv", encoding="utf-8")
            csv_handler.setFormatter(logging.Formatter("%(message)s"))
            cls._loggers["trades_csv"].addHandler(csv_handler)
            cls._loggers["trades_csv"].setLevel(logging.INFO)
            cls._loggers["trades_csv"].propagate = False

        return cls._loggers["trades_csv"]


class StructuredTradeLogger:
    """Enhanced trade logger with structured data support."""

    def __init__(self):
        self.logger = TradingBotLogger.get_logger("trades", LoggerType.TRADING)
        self.csv_logger = TradingBotLogger.get_trade_csv_logger()

    def log_trade_execution(
        self,
        trade_id: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        success: bool,
        reason: str = "",
        confidence: float = 0.0,
        portfolio_value: float = 0.0,
        strategy: str = "ENHANCED",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a trade execution with structured data.

        This creates both human-readable logs and CSV data for analysis.
        """
        timestamp = datetime.now()

        # Human-readable log entry
        status = "SUCCESS" if success else "FAILED"
        human_msg = (
            f"TRADE_EXEC | {trade_id} | {symbol} | {action} | "
            f"{quantity:.6f} @ {price:.2f} | {status}"
        )

        if reason:
            human_msg += f" | Reason: {reason}"
        if confidence > 0:
            human_msg += f" | Confidence: {confidence:.3f}"

        self.logger.info(human_msg)

        # CSV entry for analysis (matching current format)
        csv_entry = (
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')},{trade_id},{symbol},"
            f"{action},{quantity:.6f},{price:.4f},{status},{reason},{strategy},"
            f"{confidence:.2f},{portfolio_value:.2f}"
        )

        self.csv_logger.info(csv_entry)

        # Optional: Log as JSON for advanced analysis
        if metadata:
            json_data = {
                "timestamp": timestamp.isoformat(),
                "trade_id": trade_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "success": success,
                "reason": reason,
                "confidence": confidence,
                "portfolio_value": portfolio_value,
                "strategy": strategy,
                **metadata,
            }
            self.logger.debug(f"TRADE_JSON: {json.dumps(json_data)}")

    def log_model_prediction(
        self,
        model_type: str,
        symbol: str,
        prediction: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log model prediction with context."""
        model_logger = TradingBotLogger.get_logger("predictions", LoggerType.MODEL)

        msg = f"PRED | {model_type} | {symbol} | {prediction:.6f} | conf: {confidence:.3f}"

        if metadata:
            context_items = [f"{k}: {v}" for k, v in metadata.items()]
            msg += f" | {', '.join(context_items)}"

        model_logger.info(msg)


class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self):
        self.logger = TradingBotLogger.get_logger("metrics", LoggerType.PERFORMANCE)

    def log_operation_time(self, operation: str, duration_ms: float, success: bool = True):
        """Log operation performance."""
        status = "OK" if success else "FAIL"
        self.logger.info(f"{operation} | {duration_ms:.2f}ms | {status}")

    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system performance metrics."""
        metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"SYSTEM_METRICS | {metrics_str}")


# Convenience functions for easy migration
def get_trading_logger(name: str) -> logging.Logger:
    """Get a trading logger - for trade decisions and executions."""
    return TradingBotLogger.get_logger(name, LoggerType.TRADING)


def get_model_logger(name: str) -> logging.Logger:
    """Get a model logger - for ML predictions and model operations."""
    return TradingBotLogger.get_logger(name, LoggerType.MODEL)


def get_system_logger(name: str) -> logging.Logger:
    """Get a system logger - for system operations and errors."""
    return TradingBotLogger.get_logger(name, LoggerType.SYSTEM)


def get_debug_logger(name: str) -> logging.Logger:
    """Get a debug logger - for development and troubleshooting."""
    return TradingBotLogger.get_logger(name, LoggerType.DEBUG)


def get_performance_logger(name: str) -> logging.Logger:
    """Get a performance logger - for metrics and timing."""
    return TradingBotLogger.get_logger(name, LoggerType.PERFORMANCE)


# Legacy compatibility - for existing code migration
class Logger:
    """
    Legacy compatibility wrapper.

    Provides the .logger attribute expected by existing code while
    using the new centralized logging system.
    """

    def __init__(
        self,
        name: str = "crypto_trading_bot",
        config: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        # Determine appropriate logger type from name
        if any(keyword in name.lower() for keyword in ["trade", "order", "execution"]):
            logger_type = LoggerType.TRADING
        elif any(keyword in name.lower() for keyword in ["model", "prediction", "ensemble"]):
            logger_type = LoggerType.MODEL
        elif any(keyword in name.lower() for keyword in ["debug", "dev"]):
            logger_type = LoggerType.DEBUG
        else:
            logger_type = LoggerType.SYSTEM

        self.logger = TradingBotLogger.get_logger(name, logger_type)

    def getChild(self, suffix: str) -> logging.Logger:
        """Return a child logger of the wrapped logger."""
        return self.logger.getChild(suffix)


# Initialize the system on import
_system_logger = get_system_logger("logging_manager")
_system_logger.info("Enhanced logging system ready for production trading")
