"""Enhanced error handling and circuit breaker pattern for the paper trader."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 300  # seconds
    success_threshold: int = 3  # successful calls needed to close from half-open
    

class TradingCircuitBreaker:
    """Circuit breaker implementation for trading operations."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
        self.logger = logging.getLogger(__name__)
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Recovery timeout not reached."
                )
        
        try:
            result = await self._execute_function(func, *args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._reset()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker opened due to failure in HALF_OPEN state: {exception}")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.error(f"Circuit breaker opened after {self.failure_count} failures: {exception}")
    
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info("Circuit breaker reset to CLOSED state")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'time_to_recovery': max(0, self.config.recovery_timeout - (
                time.time() - (self.last_failure_time or 0)
            )) if self.last_failure_time else 0
        }


class SystemRecoveryManager:
    """Manages automatic recovery from system failures."""
    
    def __init__(self, paper_trader):
        self.trader = paper_trader
        self.logger = logging.getLogger(__name__)
        self.recovery_strategies = {
            'websocket_disconnected': self._recover_websocket,
            'api_rate_limited': self._handle_rate_limit,
            'model_loading_failed': self._reload_models,
            'data_corruption': self._reinitialize_buffers,
            'prediction_failure': self._handle_prediction_failure,
            'telegram_failure': self._recover_telegram
        }
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
    
    async def handle_system_failure(self, failure_type: str, error_details: Dict[str, Any]) -> bool:
        """Handle system failure with appropriate recovery strategy."""
        
        # Track recovery attempts
        if failure_type not in self.recovery_attempts:
            self.recovery_attempts[failure_type] = 0
        
        self.recovery_attempts[failure_type] += 1
        
        if self.recovery_attempts[failure_type] > self.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts reached for {failure_type}")
            return False
        
        recovery_func = self.recovery_strategies.get(failure_type)
        if not recovery_func:
            self.logger.error(f"No recovery strategy for failure type: {failure_type}")
            return False
        
        try:
            self.logger.info(f"Attempting recovery for {failure_type} (attempt {self.recovery_attempts[failure_type]})")
            success = await recovery_func(error_details)
            
            if success:
                self.recovery_attempts[failure_type] = 0  # Reset on success
                await self._send_recovery_notification(failure_type, "System recovered successfully")
                self.logger.info(f"Successfully recovered from {failure_type}")
            else:
                self.logger.warning(f"Recovery attempt failed for {failure_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during recovery for {failure_type}: {e}")
            return False
    
    async def _recover_websocket(self, error_details: Dict[str, Any]) -> bool:
        """Recover from WebSocket disconnection."""
        try:
            # Cancel existing tasks
            if hasattr(self.trader, 'data_feed_task') and self.trader.data_feed_task:
                self.trader.data_feed_task.cancel()
            
            # Wait a bit before reconnecting
            await asyncio.sleep(5)
            
            # Restart WebSocket feed
            self.trader.data_feed_task = asyncio.create_task(
                self.trader.data_collector.start_websocket_feed(self.trader.settings.symbols)
            )
            
            # Wait to verify connection
            await asyncio.sleep(10)
            return not self.trader.data_feed_task.done()
            
        except Exception as e:
            self.logger.error(f"WebSocket recovery failed: {e}")
            return False
    
    async def _handle_rate_limit(self, error_details: Dict[str, Any]) -> bool:
        """Handle API rate limiting."""
        try:
            # Extract wait time from error or use default
            wait_time = error_details.get('retry_after', 60)
            self.logger.info(f"Rate limited, waiting {wait_time} seconds")
            
            await asyncio.sleep(wait_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit handling failed: {e}")
            return False
    
    async def _reload_models(self, error_details: Dict[str, Any]) -> bool:
        """Reload ML models."""
        try:
            symbol = error_details.get('symbol')
            if symbol:
                # Reload specific symbol models
                success = await self.trader.model_loader.load_symbol_models(symbol)
                return success
            else:
                # Reload all models
                await self.trader.load_models()
                return True
                
        except Exception as e:
            self.logger.error(f"Model reloading failed: {e}")
            return False
    
    async def _reinitialize_buffers(self, error_details: Dict[str, Any]) -> bool:
        """Reinitialize data buffers."""
        try:
            symbol = error_details.get('symbol')
            if symbol:
                # Reinitialize specific symbol buffer
                await self.trader.data_collector.initialize_buffer(symbol)
                await self.trader.data_collector.ensure_sufficient_data(symbol, min_length=500)
            else:
                # Reinitialize all buffers
                await self.trader.data_collector.initialize_buffers(self.trader.settings.symbols)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Buffer reinitialization failed: {e}")
            return False
    
    async def _handle_prediction_failure(self, error_details: Dict[str, Any]) -> bool:
        """Handle prediction failures."""
        try:
            symbol = error_details.get('symbol')
            if symbol:
                # Skip this symbol for a few cycles
                if not hasattr(self.trader, 'skip_symbols'):
                    self.trader.skip_symbols = {}
                
                self.trader.skip_symbols[symbol] = datetime.now() + timedelta(minutes=5)
                self.logger.info(f"Skipping predictions for {symbol} for 5 minutes")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Prediction failure handling failed: {e}")
            return False
    
    async def _recover_telegram(self, error_details: Dict[str, Any]) -> bool:
        """Recover Telegram connection."""
        try:
            if hasattr(self.trader, 'telegram_notifier'):
                await self.trader.telegram_notifier.test_connection()
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Telegram recovery failed: {e}")
            return False
    
    async def _send_recovery_notification(self, failure_type: str, message: str):
        """Send recovery notification if possible."""
        try:
            if hasattr(self.trader, 'telegram_notifier'):
                await self.trader.telegram_notifier.send_system_status(
                    "RECOVERED", f"{failure_type}: {message}"
                )
        except Exception as e:
            self.logger.warning(f"Could not send recovery notification: {e}")


class HealthMonitor:
    """Monitors system health and performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'predictions_per_minute': 0,
            'avg_api_response_time': 0,
            'memory_usage_mb': 0,
            'buffer_health': {},
            'last_successful_prediction': None,
            'error_count_last_hour': 0,
            'websocket_status': 'unknown',
            'model_load_status': {},
            'uptime_seconds': 0
        }
        self.start_time = time.time()
        self.prediction_count = 0
        self.api_response_times = []
        self.error_timestamps = []
        self.logger = logging.getLogger(__name__)
    
    async def collect_metrics(self, trader):
        """Collect current system health metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            # Update basic metrics
            self.metrics['uptime_seconds'] = time.time() - self.start_time
            self.metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            
            # Update error count (last hour)
            cutoff_time = time.time() - 3600
            self.error_timestamps = [t for t in self.error_timestamps if t > cutoff_time]
            self.metrics['error_count_last_hour'] = len(self.error_timestamps)
            
            # Update WebSocket status
            if hasattr(trader, 'data_feed_task') and trader.data_feed_task:
                self.metrics['websocket_status'] = 'connected' if not trader.data_feed_task.done() else 'disconnected'
            else:
                self.metrics['websocket_status'] = 'not_started'
            
            # Update buffer health
            if hasattr(trader, 'data_collector'):
                buffer_status = trader.data_collector.get_detailed_buffer_status()
                self.metrics['buffer_health'] = {
                    symbol: status.get('status', 'unknown')
                    for symbol, status in buffer_status.items()
                }
            
            # Update model status
            if hasattr(trader, 'model_loader'):
                self.metrics['model_load_status'] = {
                    'lstm_models': len(getattr(trader.model_loader, 'lstm_models', {})),
                    'xgb_models': len(getattr(trader.model_loader, 'xgb_models', {}))
                }
            
            # Calculate predictions per minute
            if self.metrics['uptime_seconds'] > 0:
                self.metrics['predictions_per_minute'] = (
                    self.prediction_count / (self.metrics['uptime_seconds'] / 60)
                )
            
            # Calculate average API response time
            if self.api_response_times:
                self.metrics['avg_api_response_time'] = sum(self.api_response_times) / len(self.api_response_times)
                # Keep only last 100 measurements
                self.api_response_times = self.api_response_times[-100:]
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def record_prediction(self):
        """Record a prediction event."""
        self.prediction_count += 1
        self.metrics['last_successful_prediction'] = datetime.now()
    
    def record_api_response_time(self, response_time: float):
        """Record API response time."""
        self.api_response_times.append(response_time)
    
    def record_error(self):
        """Record an error event."""
        self.error_timestamps.append(time.time())
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status with overall assessment."""
        
        # Determine overall health
        health_score = 100
        issues = []
        
        # Check critical metrics
        if self.metrics['error_count_last_hour'] > 10:
            health_score -= 30
            issues.append(f"High error rate: {self.metrics['error_count_last_hour']} errors/hour")
        
        if self.metrics['websocket_status'] != 'connected':
            health_score -= 25
            issues.append(f"WebSocket not connected: {self.metrics['websocket_status']}")
        
        if self.metrics['memory_usage_mb'] > 1000:  # > 1GB
            health_score -= 15
            issues.append(f"High memory usage: {self.metrics['memory_usage_mb']:.1f}MB")
        
        if self.metrics['last_successful_prediction']:
            time_since_prediction = datetime.now() - self.metrics['last_successful_prediction']
            if time_since_prediction.total_seconds() > 300:  # 5 minutes
                health_score -= 20
                issues.append("No recent predictions")
        
        # Assess buffer health
        unhealthy_buffers = [
            symbol for symbol, status in self.metrics['buffer_health'].items()
            if status != 'healthy'
        ]
        if unhealthy_buffers:
            health_score -= len(unhealthy_buffers) * 5
            issues.append(f"Unhealthy buffers: {unhealthy_buffers}")
        
        # Overall status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        elif health_score >= 40:
            status = "poor"
        else:
            status = "critical"
        
        return {
            'overall_status': status,
            'health_score': max(0, health_score),
            'issues': issues,
            'metrics': self.metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics_summary(self) -> str:
        """Get a formatted summary of key metrics."""
        status = self.get_health_status()
        
        return f"""
🏥 System Health: {status['overall_status'].upper()} ({status['health_score']}/100)
⏱️  Uptime: {self.metrics['uptime_seconds']/3600:.1f} hours
🔮 Predictions/min: {self.metrics['predictions_per_minute']:.1f}
📊 Memory: {self.metrics['memory_usage_mb']:.1f}MB
🌐 WebSocket: {self.metrics['websocket_status']}
❌ Errors (1h): {self.metrics['error_count_last_hour']}
📡 API Response: {self.metrics['avg_api_response_time']:.3f}s
"""