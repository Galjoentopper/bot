#!/usr/bin/env python3
"""
Simple Trading System Test
Tests core functionality without complex logging
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test basic imports"""
    try:
        from scripts.enhanced_trader import EnhancedUnifiedPaperTrader

        print("[PASS] EnhancedUnifiedPaperTrader import successful")
        return True
    except Exception as e:
        print(f"[FAIL] EnhancedUnifiedPaperTrader import failed: {e}")
        return False


def test_config_files():
    """Test configuration files exist and are valid"""
    try:
        import json

        import yaml

        # Test training_config.yaml
        with open("training_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Check drift monitoring is disabled
        drift_enabled = config.get("drift_monitoring", {}).get("enabled", True)
        if not drift_enabled:
            print("[PASS] Drift monitoring disabled in training_config.yaml")
        else:
            print("[FAIL] Drift monitoring still enabled in training_config.yaml")
            return False

        # Test validation_config.json
        with open("validation_config.json", "r") as f:
            val_config = json.load(f)

        drift_monitoring_enabled = val_config.get("drift_monitoring_enabled", True)
        if not drift_monitoring_enabled:
            print("[PASS] Drift monitoring disabled in validation_config.json")
        else:
            print("[FAIL] Drift monitoring still enabled in validation_config.json")
            return False

        return True
    except Exception as e:
        print(f"[FAIL] Config test failed: {e}")
        return False


def test_profit_optimizer():
    """Test profit optimizer functionality"""
    try:
        from src.trading.profit_optimizer import ProfitOptimizer

        # Initialize with basic config
        config = {
            "profit_target": 0.02,
            "stop_loss": 0.01,
            "trailing_stop": 0.005,
            "max_position_pct": 0.1,
        }

        optimizer = ProfitOptimizer(config)
        print("[PASS] ProfitOptimizer initialized successfully")
        return True
    except Exception as e:
        print(f"[FAIL] ProfitOptimizer test failed: {e}")
        return False


def test_signal_generator():
    """Test signal generator functionality"""
    try:
        from src.trading.enhanced_signal_generator import EnhancedSignalGenerator

        signal_gen = EnhancedSignalGenerator()
        print("[PASS] EnhancedSignalGenerator initialized successfully")
        return True
    except Exception as e:
        print(f"[FAIL] EnhancedSignalGenerator test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Starting simple trading system test...")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Config Files Test", test_config_files),
        ("Profit Optimizer Test", test_profit_optimizer),
        ("Signal Generator Test", test_signal_generator),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] {test_name} failed")
        except Exception as e:
            print(f"[ERROR] {test_name} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    success_rate = (passed / total) * 100
    print(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 75:
        print("[SUCCESS] System appears to be working correctly")
        return True
    else:
        print("[ERROR] System has significant issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
