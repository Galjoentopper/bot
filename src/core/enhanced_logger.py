"""Enhanced logging system with structured logging and context tracking."""
import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import traceback
from enum import Enum

from .interfaces import ILogger
from .base_service import BaseService
from .container import injectable


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context if available
        if hasattr(record, 'context') and record.context:
            log_entry['context'] = record.context
        
        # Add exception info if available
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class EnhancedLogger(BaseService, ILogger):
    """Enhanced logger with structured logging and context tracking."""
    
    def __init__(self):
        super().__init__(logger=None)  # Don't inject logger into itself
        self._loggers: Dict[str, logging.Logger] = {}
        self._log_dir = Path("logs")
        self._context_stack: List[Dict[str, Any]] = []
        self._global_context: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the logging system."""
        try:
            # Create logs directory
            self._log_dir.mkdir(exist_ok=True)
            
            # Setup main logger
            self._setup_main_logger()
            
            # Setup component loggers
            self._setup_component_loggers()
            
            self._initialized = True
            self.log_info("Enhanced logging system initialized")
            return True
            
        except Exception as e:
            print(f"Failed to initialize logging system: {e}")
            traceback.print_exc()
            return False
    
    def log_info(self, message: str, context: Dict[str, Any] = None):
        """Log info message."""
        self._log(LogLevel.INFO, message, context)
    
    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, context)
    
    def log_error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log error message."""
        if exception:
            # Add exception info to context
            exc_context = context.copy() if context else {}
            exc_context.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception)
            })
            context = exc_context
        
        self._log(LogLevel.ERROR, message, context, exception)
    
    def log_trade(self, symbol: str, action: str, amount: float, metadata: Dict[str, Any]):
        """Log trade execution."""
        trade_context = {
            'symbol': symbol,
            'action': action,
            'amount': amount,
            'trade_metadata': metadata,
            'trade_timestamp': datetime.now().isoformat()
        }
        
        self._log(LogLevel.INFO, f"Trade executed: {action} {amount} {symbol}", trade_context)
        
        # Also log to dedicated trade log
        trade_logger = self._get_logger('trades')
        trade_logger.info(f"TRADE: {action} {amount} {symbol}", extra={'context': trade_context})
    
    def log_debug(self, message: str, context: Dict[str, Any] = None):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, context)
    
    def log_critical(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, context, exception)
    
    def push_context(self, context: Dict[str, Any]):
        """Push context onto the context stack."""
        self._context_stack.append(context.copy())
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """Pop context from the context stack."""
        if self._context_stack:
            return self._context_stack.pop()
        return None
    
    def set_global_context(self, key: str, value: Any):
        """Set global context that applies to all log messages."""
        self._global_context[key] = value
    
    def clear_global_context(self):
        """Clear all global context."""
        self._global_context.clear()
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a named logger."""
        return self._get_logger(name)
    
    def _log(self, level: LogLevel, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Internal logging method."""
        if not self._initialized:
            # Fallback to print if not initialized
            print(f"[{level.value}] {message}")
            if exception:
                traceback.print_exc()
            return
        
        # Build full context
        full_context = self._build_full_context(context)
        
        # Get appropriate logger
        logger = self._get_logger('main')
        
        # Log the message
        log_method = getattr(logger, level.value.lower())
        
        if exception:
            log_method(message, exc_info=exception, extra={'context': full_context})
        else:
            log_method(message, extra={'context': full_context})
    
    def _build_full_context(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build full context from global, stack, and local context."""
        full_context = {}
        
        # Add global context
        full_context.update(self._global_context)
        
        # Add stacked contexts
        for stack_context in self._context_stack:
            full_context.update(stack_context)
        
        # Add local context
        if context:
            full_context.update(context)
        
        return full_context
    
    def _get_logger(self, name: str) -> logging.Logger:
        """Get or create a named logger."""
        if name not in self._loggers:
            logger = logging.getLogger(f"trading_system.{name}")
            logger.setLevel(logging.DEBUG)
            
            # Add file handler
            log_file = self._log_dir / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(StructuredFormatter())
            logger.addHandler(file_handler)
            
            # Add console handler for main logger
            if name == 'main':
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                logger.addHandler(console_handler)
            
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def _setup_main_logger(self):
        """Setup the main system logger."""
        main_logger = self._get_logger('main')
        main_logger.info("Main logger initialized")
    
    def _setup_component_loggers(self):
        """Setup component-specific loggers."""
        components = [
            'config_manager',
            'feature_manager',
            'model_manager',
            'trading_engine',
            'data_provider',
            'trades',
            'errors',
            'performance'
        ]
        
        for component in components:
            self._get_logger(component)


class LoggerContext:
    """Context manager for logging context."""
    
    def __init__(self, logger: EnhancedLogger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        self.logger.push_context(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.pop_context()


class PerformanceLogger:
    """Performance logging utility."""
    
    def __init__(self, logger: EnhancedLogger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log_debug(f"Starting {self.operation_name}", {
            'operation': self.operation_name,
            'start_time': self.start_time.isoformat()
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        context = {
            'operation': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'success': exc_type is None
        }
        
        if exc_type is None:
            self.logger.log_info(f"Completed {self.operation_name} in {duration:.3f}s", context)
        else:
            context['error'] = str(exc_val)
            self.logger.log_error(f"Failed {self.operation_name} after {duration:.3f}s", context, exc_val)


# Convenience functions
def with_logging_context(logger: EnhancedLogger, context: Dict[str, Any]):
    """Create a logging context manager."""
    return LoggerContext(logger, context)


def with_performance_logging(logger: EnhancedLogger, operation_name: str):
    """Create a performance logging context manager."""
    return PerformanceLogger(logger, operation_name)