"""Comprehensive test suite for enhanced paper trader."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add paper_trader to path
sys.path.append(str(Path(__file__).parent.parent))

from paper_trader.utils import (
    TradingCircuitBreaker,
    CircuitBreakerConfig,
    DataQualityMonitor,
    PerformanceMonitor,
    SmartNotificationManager,
    NotificationPriority,
    NotificationType
)


class TestTradingCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
        return TradingCircuitBreaker(config)
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, circuit_breaker):
        """Test successful function execution."""
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_failure_handling(self, circuit_breaker):
        """Test failure handling and circuit opening."""
        async def failing_func():
            raise Exception("Test error")
        
        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state.value == "open"
        
        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(Exception):  # CircuitBreakerOpenError
            await circuit_breaker.call(failing_func)


class TestDataQualityMonitor:
    """Test data quality monitoring functionality."""
    
    @pytest.fixture
    def quality_monitor(self):
        return DataQualityMonitor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        np.random.seed(42)  # For reproducible results
        
        base_price = 100.0
        data = {
            'open': base_price + np.random.randn(100) * 0.5,
            'high': base_price + np.random.randn(100) * 0.5 + 0.5,
            'low': base_price + np.random.randn(100) * 0.5 - 0.5,
            'close': base_price + np.random.randn(100) * 0.5,
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        # Ensure OHLC relationships are valid
        for i in range(100):
            data['high'][i] = max(data['open'][i], data['close'][i], data['high'][i])
            data['low'][i] = min(data['open'][i], data['close'][i], data['low'][i])
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_valid_data_quality(self, quality_monitor, sample_data):
        """Test quality validation with good data."""
        report = quality_monitor.validate_data_quality("BTCEUR", sample_data)
        
        assert report.symbol == "BTCEUR"
        assert report.quality_score > 80  # Should be high quality
        assert report.is_tradeable
        assert report.overall_quality.value in ['excellent', 'good']
    
    def test_invalid_data_quality(self, quality_monitor):
        """Test quality validation with bad data."""
        # Create data with issues
        bad_data = pd.DataFrame({
            'open': [100, 101, None, 103],  # Has null values
            'high': [99, 102, 104, 105],    # Invalid: high < open for first row
            'low': [98, 100, 102, 102],
            'close': [101, 101, 103, 104],
            'volume': [1000, 2000, 3000, -100]  # Negative volume
        })
        
        report = quality_monitor.validate_data_quality("TESTEUR", bad_data)
        
        assert report.quality_score < 60  # Should be low quality
        assert not report.is_tradeable
        assert len(report.issues) > 0
    
    def test_quality_metrics_calculation(self, quality_monitor, sample_data):
        """Test quality metrics calculation."""
        report = quality_monitor.validate_data_quality("BTCEUR", sample_data)
        
        assert 'total_rows' in report.metrics
        assert 'completeness_pct' in report.metrics
        assert 'price_volatility' in report.metrics
        assert 'avg_volume' in report.metrics


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def performance_monitor(self):
        return PerformanceMonitor()
    
    @pytest.mark.asyncio
    async def test_operation_monitoring(self, performance_monitor):
        """Test operation performance monitoring."""
        async def test_operation():
            await asyncio.sleep(0.01)  # Simulate work
            return "result"
        
        result = await performance_monitor.monitor_operation(
            "test_op", test_operation, use_cache=False
        )
        
        assert result == "result"
        
        # Check that metrics were recorded
        summary = performance_monitor.get_performance_summary("test_op")
        assert summary['total_calls'] == 1
        assert summary['avg_execution_time'] > 0
    
    def test_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        # Manually add some metrics
        performance_monitor.operation_stats['test_op'] = {
            'count': 10,
            'total_time': 1.0,
            'avg_time': 0.1,
            'min_time': 0.05,
            'max_time': 0.2,
            'cache_hits': 5,
            'errors': 1
        }
        
        summary = performance_monitor.get_performance_summary("test_op")
        
        assert summary['total_calls'] == 10
        assert summary['avg_execution_time'] == 0.1
        assert summary['cache_hit_rate'] == 0.5
        assert summary['error_rate'] == 0.1
    
    def test_slowest_operations(self, performance_monitor):
        """Test slowest operations tracking."""
        # Add multiple operations with different speeds
        performance_monitor.operation_stats['fast_op'] = {
            'count': 10, 'total_time': 0.1, 'avg_time': 0.01,
            'min_time': 0.01, 'max_time': 0.01, 'cache_hits': 0, 'errors': 0
        }
        performance_monitor.operation_stats['slow_op'] = {
            'count': 5, 'total_time': 5.0, 'avg_time': 1.0,
            'min_time': 0.8, 'max_time': 1.2, 'cache_hits': 0, 'errors': 0
        }
        
        slowest = performance_monitor.get_slowest_operations(limit=2)
        
        assert len(slowest) == 2
        assert slowest[0]['operation'] == 'slow_op'
        assert slowest[1]['operation'] == 'fast_op'


class TestSmartNotificationManager:
    """Test smart notification system."""
    
    @pytest.fixture
    def notification_manager(self):
        mock_telegram = Mock()
        mock_telegram.is_enabled.return_value = True
        mock_telegram.send_message = AsyncMock()
        return SmartNotificationManager(mock_telegram)
    
    @pytest.mark.asyncio
    async def test_notification_queuing(self, notification_manager):
        """Test notification queuing functionality."""
        success = await notification_manager.send_notification(
            NotificationType.TRADE_OPENED,
            "Test Trade",
            "Test message",
            NotificationPriority.HIGH
        )
        
        assert success
        
        # Check that notification was queued
        stats = notification_manager.get_notification_stats()
        assert stats['total_queued'] > 0
        assert stats['queue_sizes']['high'] > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, notification_manager):
        """Test rate limiting functionality."""
        # Send multiple notifications rapidly
        for i in range(25):  # Exceed the default limit
            await notification_manager.send_notification(
                NotificationType.ERROR,
                f"Error {i}",
                f"Error message {i}"
            )
        
        stats = notification_manager.get_notification_stats()
        
        # Some notifications should have been dropped due to rate limiting
        assert stats['stats']['dropped'][NotificationType.ERROR] > 0
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, notification_manager):
        """Test that high priority notifications are processed first."""
        # Add notifications in reverse priority order
        await notification_manager.send_notification(
            NotificationType.PORTFOLIO_UPDATE, "Low", "Low priority",
            NotificationPriority.LOW
        )
        await notification_manager.send_notification(
            NotificationType.ERROR, "Critical", "Critical message",
            NotificationPriority.CRITICAL
        )
        
        # Critical should be in critical queue, low in low queue
        assert len(notification_manager.queues[NotificationPriority.CRITICAL]) > 0
        assert len(notification_manager.queues[NotificationPriority.LOW]) > 0
    
    @pytest.mark.asyncio
    async def test_trade_notification(self, notification_manager):
        """Test trade-specific notification formatting."""
        success = await notification_manager.send_trade_notification(
            action="BUY",
            symbol="BTCEUR",
            price=50000.0,
            quantity=0.001,
            pnl=None
        )
        
        assert success
        
        # Test sell notification with P&L
        success = await notification_manager.send_trade_notification(
            action="SELL",
            symbol="BTCEUR",
            price=51000.0,
            quantity=0.001,
            pnl=1.0,
            reason="take_profit"
        )
        
        assert success


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_data_pipeline_integration(self):
        """Test complete data processing pipeline."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=500, freq='1min')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(500) * 0.5,
            'high': 100 + np.random.randn(500) * 0.5 + 0.5,
            'low': 100 + np.random.randn(500) * 0.5 - 0.5,
            'close': 100 + np.random.randn(500) * 0.5,
            'volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        
        # Fix OHLC relationships
        for i in range(500):
            data.loc[data.index[i], 'high'] = max(
                data.iloc[i]['open'], data.iloc[i]['close'], data.iloc[i]['high']
            )
            data.loc[data.index[i], 'low'] = min(
                data.iloc[i]['open'], data.iloc[i]['close'], data.iloc[i]['low']
            )
        
        # Test data quality
        quality_monitor = DataQualityMonitor()
        quality_report = quality_monitor.validate_data_quality("BTCEUR", data)
        
        assert quality_report.is_tradeable
        assert quality_report.quality_score > 70
        
        # Test performance monitoring with data processing
        performance_monitor = PerformanceMonitor()
        
        async def process_data():
            # Simulate feature engineering
            await asyncio.sleep(0.01)
            return data.copy()
        
        result = await performance_monitor.monitor_operation(
            "data_processing",
            process_data
        )
        
        assert result is not None
        assert len(result) == 500
        
        # Check performance metrics
        summary = performance_monitor.get_performance_summary("data_processing")
        assert summary['total_calls'] == 1
        assert summary['avg_execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self):
        """Test error handling and recovery mechanisms."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        circuit_breaker = TradingCircuitBreaker(config)
        
        call_count = 0
        
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Simulated failure")
            return "success"
        
        # First two calls should fail
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(flaky_function)
        
        # Circuit should be open now
        assert circuit_breaker.state.value == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should succeed and reset circuit
        result = await circuit_breaker.call(flaky_function)
        assert result == "success"
        assert circuit_breaker.state.value == "closed"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])