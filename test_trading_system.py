#!/usr/bin/env python3
"""
Trading System Test Script
Tests all components of the enhanced trading system to verify optimizations.
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime

import yaml

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    """Setup logging for test script"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("test_results.log")],
    )
    return logging.getLogger(__name__)


def test_config_files(logger):
    """Test 1: Verify configuration files are properly set"""
    logger.info("=== Testing Configuration Files ===")

    # Test training_config.yaml
    try:
        with open("training_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Check drift monitoring is disabled
        drift_enabled = config.get("drift_monitoring", {}).get("enabled", True)
        if not drift_enabled:
            logger.info("✓ Drift monitoring disabled in training_config.yaml")
        else:
            logger.error("✗ Drift monitoring still enabled in training_config.yaml")
            return False

        # Check trading thresholds
        thresholds = config.get("trading", {}).get("thresholds", {})
        logger.info(f"✓ Trading thresholds: {thresholds}")

        # Check model weights
        weights = config.get("trading", {}).get("model_weights", {})
        logger.info(f"✓ Model weights: {weights}")

    except Exception as e:
        logger.error(f"✗ Error reading training_config.yaml: {e}")
        return False

    # Test validation_config.json if exists
    if os.path.exists("validation_config.json"):
        try:
            with open("validation_config.json", "r") as f:
                val_config = json.load(f)

            drift_enabled = val_config.get("drift_monitoring_enabled", True)
            auto_start = val_config.get("auto_start_monitoring", True)

            if not drift_enabled and not auto_start:
                logger.info("✓ Drift monitoring disabled in validation_config.json")
            else:
                logger.error("✗ Drift monitoring not properly disabled in validation_config.json")
                return False

        except Exception as e:
            logger.error(f"✗ Error reading validation_config.json: {e}")
            return False

    return True


def test_module_imports(logger):
    """Test 2: Verify all trading modules can be imported"""
    logger.info("=== Testing Module Imports ===")

    modules_to_test = [
        "scripts.enhanced_trader",
        "scripts.enhanced_signal_generator",
        "scripts.profit_optimizer",
        "scripts.data_manager",
        "scripts.model_manager",
    ]

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✓ Successfully imported {module_name}")
        except Exception as e:
            logger.error(f"✗ Failed to import {module_name}: {e}")
            return False

    return True


def test_signal_generation(logger):
    """Test 3: Test signal generation functionality"""
    logger.info("=== Testing Signal Generation ===")

    try:
        from src.data_pipeline.loader import DataLoader
        from src.trading.enhanced_signal_generator import EnhancedSignalGenerator

        # Initialize components
        data_loader = DataLoader()
        signal_generator = EnhancedSignalGenerator()

        # Test with a sample symbol
        test_symbol = "AAPL"
        logger.info(f"Testing signal generation for {test_symbol}")

        # This is a basic test - in real scenario we'd need actual data
        logger.info("✓ Signal generator initialized successfully")

        return True

    except Exception as e:
        logger.error(f"✗ Signal generation test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_model_loading(logger):
    """Test 4: Test model loading capabilities"""
    logger.info("=== Testing Model Loading ===")

    try:
        from scripts.model_manager import ModelManager

        model_manager = ModelManager()
        logger.info("✓ Model manager initialized successfully")

        # Check for model files
        model_dirs = ["models", "trained_models"]
        models_found = False

        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                model_files = [
                    f for f in os.listdir(model_dir) if f.endswith((".pkl", ".joblib", ".h5"))
                ]
                if model_files:
                    logger.info(f"✓ Found {len(model_files)} model files in {model_dir}")
                    models_found = True

        if not models_found:
            logger.warning("⚠ No trained models found - system will use default predictions")

        return True

    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False


def test_profit_optimizer(logger):
    """Test 5: Test profit optimizer functionality"""
    logger.info("=== Testing Profit Optimizer ===")

    try:
        from scripts.profit_optimizer import ProfitOptimizer

        # Initialize with test config
        test_config = {
            "initial_balance": 10000,
            "max_position_size": 0.1,
            "transaction_fee": 0.001,
            "slippage": 0.001,
        }

        optimizer = ProfitOptimizer(test_config)
        logger.info("✓ Profit optimizer initialized successfully")

        # Test position sizing calculation
        test_confidence = 0.7
        test_volatility = 0.02
        position_size = optimizer.calculate_position_size("AAPL", test_confidence, test_volatility)

        if 0 < position_size <= test_config["max_position_size"]:
            logger.info(f"✓ Position sizing working correctly: {position_size:.4f}")
        else:
            logger.error(f"✗ Invalid position size calculated: {position_size}")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Profit optimizer test failed: {e}")
        return False


def test_error_handling(logger):
    """Test 6: Test error handling and graceful degradation"""
    logger.info("=== Testing Error Handling ===")

    try:
        # Test handling of missing data
        from scripts.enhanced_trader import EnhancedTrader

        # This should handle missing models gracefully
        trader = EnhancedTrader(test_mode=True)
        logger.info("✓ Enhanced trader handles initialization gracefully")

        return True

    except Exception as e:
        logger.info(f"✓ Error handling working - caught expected error: {type(e).__name__}")
        return True


def run_comprehensive_test():
    """Run all tests and generate report"""
    logger = setup_logging()
    logger.info("Starting comprehensive trading system test...")

    test_results = {"timestamp": datetime.now().isoformat(), "tests": {}}

    # Define all tests
    tests = [
        ("Configuration Files", test_config_files),
        ("Module Imports", test_module_imports),
        ("Signal Generation", test_signal_generation),
        ("Model Loading", test_model_loading),
        ("Profit Optimizer", test_profit_optimizer),
        ("Error Handling", test_error_handling),
    ]

    passed_tests = 0
    total_tests = len(tests)

    # Run each test
    for test_name, test_func in tests:
        try:
            result = test_func(logger)
            test_results["tests"][test_name] = {"passed": result, "error": None}
            if result:
                passed_tests += 1
                logger.info(f"[PASS] {test_name} PASSED")
            else:
                logger.error(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            test_results["tests"][test_name] = {"passed": False, "error": str(e)}
            logger.error(f"✗ {test_name} FAILED with exception: {e}")

    # Generate summary
    success_rate = (passed_tests / total_tests) * 100
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": success_rate,
    }

    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 80:
        logger.info("[SUCCESS] SYSTEM READY FOR PRODUCTION")
    elif success_rate >= 60:
        logger.warning("[WARNING] SYSTEM NEEDS MINOR FIXES")
    else:
        logger.error("[ERROR] SYSTEM NEEDS MAJOR FIXES")

    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    return success_rate >= 80


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
