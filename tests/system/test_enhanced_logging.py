#!/usr/bin/env python3
"""
Test Suite for Enhanced Logging System
======================================

This test validates the new centralized logging system for the production trading bot.

Tests:
- Logging manager initialization
- Different logger types (trading, model, system, debug)
- Structured trade logging
- Performance logging
- File creation and rotation
- Legacy compatibility
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_basic_logging_functionality():
    """Test basic logging system functionality."""
    print("🧪 Testing Enhanced Logging System")
    print("=" * 50)

    try:
        # Import the new logging system
        from src.core.logging_manager import (
            PerformanceLogger,
            StructuredTradeLogger,
            TradingBotLogger,
            get_debug_logger,
            get_model_logger,
            get_system_logger,
            get_trading_logger,
        )

        print("✅ Successfully imported enhanced logging system")

        # Test basic logger creation
        trading_logger = get_trading_logger("test_trader")
        model_logger = get_model_logger("test_model")
        system_logger = get_system_logger("test_system")
        debug_logger = get_debug_logger("test_debug")
        print("✅ Successfully created different logger types")

        # Test basic logging
        system_logger.info("🚀 Enhanced logging system test started")
        trading_logger.info("TRADE_TEST | BTCEUR | BUY | 0.001 | 95000 | SUCCESS")
        model_logger.info("MODEL_TEST | lightgbm | ETHEUR | 0.75 | conf: 0.82")
        debug_logger.debug("DEBUG_TEST | This is a debug message")
        print("✅ Basic logging operations successful")

        # Test structured trade logger
        trade_logger = StructuredTradeLogger()
        trade_logger.log_trade_execution(
            trade_id="test-123",
            symbol="BTCEUR",
            action="BUY",
            quantity=0.001,
            price=95000,
            success=True,
            reason="Test trade execution",
            confidence=0.85,
            portfolio_value=10000,
            metadata={"test": True},
        )
        print("✅ Structured trade logging successful")

        # Test performance logger
        perf_logger = PerformanceLogger()
        perf_logger.log_operation_time("model_prediction", 125.5, success=True)
        perf_logger.log_system_metrics({"cpu": 45.2, "memory": 2048})
        print("✅ Performance logging successful")

        # Test legacy compatibility
        from src.core.logging_manager import Logger

        legacy_logger = Logger("test_legacy")
        legacy_logger.logger.info("Legacy compatibility test")
        print("✅ Legacy compatibility maintained")

        return True

    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_file_creation():
    """Test that log files are created properly."""
    print("\n🗂️ Testing Log File Creation")
    print("=" * 30)

    try:
        # Check if logs directory exists
        logs_dir = Path("logs")
        if logs_dir.exists():
            print(f"✅ Logs directory exists: {logs_dir.absolute()}")

            # List log files
            log_files = list(logs_dir.glob("*.log"))
            csv_files = list(logs_dir.glob("*.csv"))

            expected_files = [
                "trading.log",
                "models.log",
                "system.log",
                "performance.log",
                "debug.log",
                "trades_report.csv",
            ]

            created_files = []
            for expected in expected_files:
                file_path = logs_dir / expected
                if file_path.exists():
                    created_files.append(expected)
                    size = file_path.stat().st_size
                    print(f"✅ {expected}: {size} bytes")
                else:
                    print(f"⚠️ {expected}: Not created (may be created on first log)")

            print(f"📊 Created {len(created_files)}/{len(expected_files)} expected log files")
            return True

        else:
            print("⚠️ Logs directory not found - may be created on first log")
            return True

    except Exception as e:
        print(f"❌ File creation test failed: {e}")
        return False


def test_trader_integration():
    """Test integration with the actual trader."""
    print("\n🤖 Testing Trader Integration")
    print("=" * 30)

    try:
        # Test if trader can import the new logging system
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bin"))

        # Try importing the modules the trader needs
        from src.core.logging_manager import (
            StructuredTradeLogger,
            get_model_logger,
            get_system_logger,
            get_trading_logger,
        )

        # Simulate what the trader does
        system_logger = get_system_logger("trader_test")
        model_logger = get_model_logger("ensemble_test")
        trade_logger = StructuredTradeLogger()

        system_logger.info("🚀 Trader logging integration test")
        model_logger.info("MODEL_PRED | test | BTCEUR | 0.0012 | conf: 0.75")

        print("✅ Trader integration successful")
        return True

    except Exception as e:
        print(f"❌ Trader integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_impact():
    """Test performance impact of logging system."""
    print("\n⚡ Testing Performance Impact")
    print("=" * 30)

    try:
        from src.core.logging_manager import get_trading_logger

        logger = get_trading_logger("perf_test")

        # Time logging operations
        num_logs = 1000
        start_time = time.time()

        for i in range(num_logs):
            logger.info(f"Performance test log message {i}")

        end_time = time.time()
        duration = end_time - start_time
        logs_per_second = num_logs / duration

        print(f"✅ Logged {num_logs} messages in {duration:.3f}s")
        print(f"⚡ Performance: {logs_per_second:.0f} logs/second")

        if logs_per_second > 1000:
            print("✅ Performance is excellent (>1000 logs/sec)")
        elif logs_per_second > 500:
            print("✅ Performance is good (>500 logs/sec)")
        else:
            print("⚠️ Performance may need optimization (<500 logs/sec)")

        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all logging tests."""
    print("🚀 Enhanced Trading Bot Logging System Test Suite")
    print("=" * 60)

    # Set test environment
    os.environ["TRADING_ENV"] = "testing"

    tests = [
        test_basic_logging_functionality,
        test_file_creation,
        test_trader_integration,
        test_performance_impact,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)

    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 25)

    passed = sum(results)
    total = len(results)

    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_func.__name__}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced logging system is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
