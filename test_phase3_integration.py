"""Simplified Phase 3 integration tests focusing on core functionality."""

import asyncio
import time
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import Logger
from src.core.error_handler import (
    ErrorHandler, TradingBotException, NetworkException, 
    ErrorSeverity, ErrorCategory
)
from src.core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerManager,
    CircuitState, CircuitBreakerOpenException
)
from src.core.retry_handler import (
    RetryHandler, RetryConfig, RetryStrategy, RetryExhaustedException
)
from src.core.health_monitor import (
    HealthMonitor, HealthStatus, AlertSeverity, HealthCheck
)
from src.core.shutdown_handler import (
    ShutdownHandler, ShutdownTask, ShutdownReason, ShutdownPhase,
    initialize_shutdown_handler
)


class TestPhase3Integration:
    """Simplified integration tests for Phase 3 components."""
    
    def __init__(self):
        self.logger = Logger("test_phase3")
        self.error_handler = ErrorHandler(self.logger)
        self.circuit_manager = CircuitBreakerManager(self.logger, self.error_handler)
        self.health_monitor = HealthMonitor(self.logger, self.error_handler)
        self.shutdown_handler = initialize_shutdown_handler(self.logger, self.error_handler)
        
        # Test counters
        self.tests_passed = 0
        self.tests_failed = 0
    
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        status = "PASS" if success else "FAIL"
        full_message = f"[{status}] {test_name}"
        if message:
            full_message += f": {message}"
        
        if success:
            self.tests_passed += 1
            self.logger.logger.info(full_message)
        else:
            self.tests_failed += 1
            self.logger.logger.error(full_message)
    
    async def test_error_handler_basic(self) -> bool:
        """Test basic error handler functionality."""
        try:
            # Test basic error handling
            error = TradingBotException(
                "Test error",
                component="test",
                operation="integration_test"
            )
            
            result = self.error_handler.handle_error(error)
            
            # Check error was recorded
            history = self.error_handler.get_error_history(limit=1)
            if not history or "Test error" not in str(history[0]):
                self.log_test_result("Error Handler - Basic", False, "Error not recorded properly")
                return False
            
            self.log_test_result("Error Handler - Basic", True)
            return True
            
        except Exception as e:
            self.log_test_result("Error Handler - Basic", False, str(e))
            return False
    
    async def test_circuit_breaker_basic(self) -> bool:
        """Test basic circuit breaker functionality."""
        try:
            # Create circuit breaker with low thresholds for testing
            config = CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=0.5,
                success_threshold=1
            )
            
            circuit = self.circuit_manager.get_circuit_breaker("test_api", config)
            
            # Test normal operation
            async def successful_call():
                return "success"
            
            result = await circuit.call(successful_call)
            if result != "success":
                self.log_test_result("Circuit Breaker - Normal Operation", False, "Unexpected result")
                return False
            
            # Test failure and circuit opening
            async def failing_call():
                raise ConnectionError("API unavailable")
            
            # Trigger failures to open circuit
            for i in range(2):
                try:
                    await circuit.call(failing_call)
                except ConnectionError:
                    pass
            
            # Circuit should be open now
            if circuit.get_state() != CircuitState.OPEN:
                self.log_test_result("Circuit Breaker - Opening", False, f"Expected OPEN, got {circuit.get_state()}")
                return False
            
            self.log_test_result("Circuit Breaker - Basic", True)
            return True
            
        except Exception as e:
            self.log_test_result("Circuit Breaker - Basic", False, str(e))
            return False
    
    async def test_retry_handler_basic(self) -> bool:
        """Test basic retry handler functionality."""
        try:
            # Create retry handler with quick retries for testing
            config = RetryConfig(
                max_attempts=3,
                base_delay=0.05,
                strategy=RetryStrategy.EXPONENTIAL_JITTER
            )
            
            retry_handler = RetryHandler("test_operation", config, self.logger, self.error_handler)
            
            # Test successful operation (no retries needed)
            async def successful_operation():
                return "success"
            
            result = await retry_handler.execute(successful_operation)
            if result != "success":
                self.log_test_result("Retry Handler - Success", False, "Unexpected result")
                return False
            
            # Test operation that succeeds on second attempt
            attempt_count = 0
            
            async def eventually_successful():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 2:
                    raise ConnectionError("Temporary failure")
                return "success after retry"
            
            result = await retry_handler.execute(eventually_successful)
            if result != "success after retry":
                self.log_test_result("Retry Handler - Retry Success", False, "Retry did not work")
                return False
            
            self.log_test_result("Retry Handler - Basic", True)
            return True
            
        except Exception as e:
            self.log_test_result("Retry Handler - Basic", False, str(e))
            return False
    
    async def test_health_monitor_basic(self) -> bool:
        """Test basic health monitoring functionality."""
        try:
            # Add simple health check
            async def simple_health_check():
                from src.core.health_monitor import HealthMetric, HealthStatus
                
                return HealthMetric(
                    name="test_metric",
                    value=50.0,
                    status=HealthStatus.HEALTHY,
                    threshold_warning=80.0,
                    unit="%"
                )
            
            health_check = HealthCheck(
                name="simple_check",
                check_function=simple_health_check,
                interval=1.0,  # Use a more reasonable interval
                timeout=10.0,
                enabled=True
            )
            
            self.health_monitor.register_health_check(health_check)
            print(f"Registered health checks: {list(self.health_monitor.health_checks.keys())}")
            print(f"Health check enabled: {health_check.enabled if hasattr(health_check, 'enabled') else 'No enabled attribute'}")
            
            # Start monitoring briefly
            await self.health_monitor.start()
            
            # Wait longer for custom health check to run (interval is 1.0s)
            await asyncio.sleep(3.0)  # Wait for at least 3 intervals
            
            # Check system health
            system_health = self.health_monitor.get_system_health()
            print(f"System health metrics: {list(system_health.metrics.keys())}")
            print(f"Looking for test_metric in: {system_health.metrics}")
            
            await self.health_monitor.stop()
            
            # Check if our custom metric exists
            if "test_metric" not in system_health.metrics:
                # Log available metrics for debugging
                available = list(system_health.metrics.keys())
                self.log_test_result("Health Monitor - Basic", False, f"Custom metric not found. Available: {available}")
                return False
            
            self.log_test_result("Health Monitor - Basic", True)
            return True
            
        except Exception as e:
            await self.health_monitor.stop()
            self.log_test_result("Health Monitor - Basic", False, str(e))
            return False
    
    async def test_shutdown_handler_basic(self) -> bool:
        """Test basic shutdown handler functionality."""
        try:
            # Register simple shutdown task
            task_executed = False
            
            def simple_task():
                nonlocal task_executed
                task_executed = True
            
            self.shutdown_handler.register_shutdown_task(
                ShutdownTask(
                    name="simple_task",
                    function=simple_task,
                    phase=ShutdownPhase.STOPPING_SERVICES
                )
            )
            
            # Execute shutdown
            success = await self.shutdown_handler.shutdown(ShutdownReason.USER_REQUEST, timeout=2.0)
            
            if not success:
                self.log_test_result("Shutdown Handler - Basic", False, "Shutdown failed")
                return False
            
            if not task_executed:
                self.log_test_result("Shutdown Handler - Basic", False, "Task not executed")
                return False
            
            self.log_test_result("Shutdown Handler - Basic", True)
            return True
            
        except Exception as e:
            self.log_test_result("Shutdown Handler - Basic", False, str(e))
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all Phase 3 integration tests."""
        self.logger.logger.info("Starting Phase 3 Integration Tests")
        self.logger.logger.info("=" * 50)
        
        tests = [
            ("Error Handler Basic", self.test_error_handler_basic),
            ("Circuit Breaker Basic", self.test_circuit_breaker_basic),
            ("Retry Handler Basic", self.test_retry_handler_basic),
            ("Health Monitor Basic", self.test_health_monitor_basic),
            ("Shutdown Handler Basic", self.test_shutdown_handler_basic)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            self.logger.logger.info(f"\nRunning {test_name}...")
            try:
                success = await test_func()
                if not success:
                    all_passed = False
            except Exception as e:
                self.log_test_result(test_name, False, f"Unexpected error: {e}")
                all_passed = False
        
        # Print summary
        self.logger.logger.info("\n" + "=" * 50)
        self.logger.logger.info("Phase 3 Integration Test Summary")
        self.logger.logger.info(f"Tests Passed: {self.tests_passed}")
        self.logger.logger.info(f"Tests Failed: {self.tests_failed}")
        
        if self.tests_passed + self.tests_failed > 0:
            success_rate = (self.tests_passed / (self.tests_passed + self.tests_failed)) * 100
            self.logger.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if all_passed:
            self.logger.logger.info("\n🎉 All Phase 3 integration tests PASSED!")
            self.logger.logger.info("Enhanced error handling and recovery system is working correctly.")
            self.logger.logger.info("\nPhase 3 Components Verified:")
            self.logger.logger.info("• ✓ Error Handler - Exception handling and logging")
            self.logger.logger.info("• ✓ Circuit Breaker - Fault tolerance for external calls")
            self.logger.logger.info("• ✓ Retry Handler - Automatic retry with backoff")
            self.logger.logger.info("• ✓ Health Monitor - System health tracking")
            self.logger.logger.info("• ✓ Shutdown Handler - Graceful shutdown procedures")
        else:
            self.logger.logger.error("\n❌ Some Phase 3 integration tests FAILED!")
            self.logger.logger.error("Please review the failed tests and fix the issues.")
        
        return all_passed


async def main():
    """Main test execution function."""
    test_suite = TestPhase3Integration()
    
    try:
        success = await test_suite.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"Test execution failed: {e}")
        return 1
    finally:
        # Cleanup
        try:
            await test_suite.health_monitor.stop()
        except:
            pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)