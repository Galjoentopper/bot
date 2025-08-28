"""Minimal Phase 3 tests to identify specific issues."""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import Logger
from src.core.error_handler import ErrorHandler, TradingBotException
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from src.core.health_monitor import HealthMonitor, HealthCheck, HealthMetric, HealthStatus


async def test_circuit_breaker():
    """Test circuit breaker in isolation."""
    print("Testing Circuit Breaker...")
    
    try:
        logger = Logger("test_cb")
        error_handler = ErrorHandler(logger)
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            success_threshold=1
        )
        
        circuit = CircuitBreaker("test_api", config, logger, error_handler)
        
        # Test successful call
        async def success_call():
            return "success"
        
        result = await circuit.call(success_call)
        print(f"Success call result: {result}")
        
        # Test failing calls
        async def fail_call():
            raise ConnectionError("Test failure")
        
        # Trigger failures
        for i in range(2):
            try:
                await circuit.call(fail_call)
            except ConnectionError as e:
                print(f"Expected failure {i+1}: {e}")
        
        print(f"Circuit state after failures: {circuit.get_state()}")
        
        if circuit.get_state() == CircuitState.OPEN:
            print("✓ Circuit Breaker test PASSED")
            return True
        else:
            print("✗ Circuit Breaker test FAILED - Circuit not opened")
            return False
            
    except Exception as e:
        print(f"✗ Circuit Breaker test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_health_monitor():
    """Test health monitor in isolation."""
    print("\nTesting Health Monitor...")
    
    try:
        logger = Logger("test_hm")
        error_handler = ErrorHandler(logger)
        health_monitor = HealthMonitor(logger, error_handler)
        
        # Create simple health check
        async def simple_check():
            return HealthMetric(
                name="test_metric",
                value=50.0,
                status=HealthStatus.HEALTHY,
                threshold_warning=80.0,
                unit="%"
            )
        
        health_check = HealthCheck(
            name="simple_check",
            check_function=simple_check,
            interval=0.2
        )
        
        health_monitor.register_health_check(health_check)
        print("Health check registered")
        
        # Start monitoring
        await health_monitor.start()
        print("Health monitor started")
        
        # Wait for checks to run
        await asyncio.sleep(0.5)
        
        # Get system health
        system_health = health_monitor.get_system_health()
        print(f"System health metrics: {list(system_health.metrics.keys())}")
        
        await health_monitor.stop()
        print("Health monitor stopped")
        
        if "test_metric" in system_health.metrics:
            print("✓ Health Monitor test PASSED")
            return True
        else:
            print("✗ Health Monitor test FAILED - test_metric not found")
            print(f"Available metrics: {list(system_health.metrics.keys())}")
            return False
            
    except Exception as e:
        print(f"✗ Health Monitor test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        try:
            await health_monitor.stop()
        except:
            pass
        return False


async def main():
    """Run minimal tests."""
    print("Running minimal Phase 3 tests...")
    print("=" * 40)
    
    results = []
    
    # Test Circuit Breaker
    cb_result = await test_circuit_breaker()
    results.append(cb_result)
    
    # Test Health Monitor
    hm_result = await test_health_monitor()
    results.append(hm_result)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED!")
        return 0
    else:
        print("❌ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)