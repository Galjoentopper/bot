#!/usr/bin/env python3
"""
Comprehensive Trading System Test Script
Tests all components and writes detailed results to a log file
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def write_log(message):
    """Write message to both console and log file"""
    print(message)
    with open('test_results_detailed.log', 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def test_enhanced_trader_import():
    """Test EnhancedUnifiedPaperTrader import"""
    try:
        from scripts.enhanced_trader import EnhancedUnifiedPaperTrader
        write_log("[PASS] EnhancedUnifiedPaperTrader import successful")
        return True
    except Exception as e:
        write_log(f"[FAIL] EnhancedUnifiedPaperTrader import failed: {e}")
        write_log(f"Traceback: {traceback.format_exc()}")
        return False

def test_profit_optimizer():
    """Test ProfitOptimizer initialization"""
    try:
        from src.trading.profit_optimizer import ProfitOptimizer
        optimizer = ProfitOptimizer()
        write_log("[PASS] ProfitOptimizer initialization successful")
        return True
    except Exception as e:
        write_log(f"[FAIL] ProfitOptimizer initialization failed: {e}")
        write_log(f"Traceback: {traceback.format_exc()}")
        return False

def test_signal_generator():
    """Test EnhancedSignalGenerator initialization"""
    try:
        from src.trading.enhanced_signal_generator import EnhancedSignalGenerator
        from src.trading.profit_optimizer import ProfitOptimizer
        from src.config.config_loader import ConfigLoader
        
        config = ConfigLoader().config
        profit_optimizer = ProfitOptimizer()
        signal_gen = EnhancedSignalGenerator(config, profit_optimizer)
        write_log("[PASS] EnhancedSignalGenerator initialization successful")
        return True
    except Exception as e:
        write_log(f"[FAIL] EnhancedSignalGenerator initialization failed: {e}")
        write_log(f"Traceback: {traceback.format_exc()}")
        return False

def test_config_files():
    """Test configuration file loading"""
    try:
        from src.config.config_loader import ConfigLoader
        config = ConfigLoader().config
        write_log("[PASS] Configuration files loaded successfully")
        return True
    except Exception as e:
        write_log(f"[FAIL] Configuration file loading failed: {e}")
        write_log(f"Traceback: {traceback.format_exc()}")
        return False

def test_trader_initialization():
    """Test EnhancedUnifiedPaperTrader initialization"""
    try:
        from scripts.enhanced_trader import EnhancedUnifiedPaperTrader
        from src.config.config_loader import ConfigLoader
        
        config = ConfigLoader().config
        trader = EnhancedUnifiedPaperTrader(config=config, show_available_mode=True)
        write_log("[PASS] EnhancedUnifiedPaperTrader initialization successful")
        return True
    except Exception as e:
        write_log(f"[FAIL] EnhancedUnifiedPaperTrader initialization failed: {e}")
        write_log(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests"""
    # Clear previous log
    with open('test_results_detailed.log', 'w', encoding='utf-8') as f:
        f.write("=== Comprehensive Trading System Test Results ===\n\n")
    
    write_log("=== Comprehensive Trading System Test Results ===")
    write_log("")
    
    tests = [
        ("Enhanced Trader Import", test_enhanced_trader_import),
        ("ProfitOptimizer", test_profit_optimizer),
        ("EnhancedSignalGenerator", test_signal_generator),
        ("Configuration Files", test_config_files),
        ("Trader Initialization", test_trader_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        write_log(f"\nTesting {test_name}...")
        if test_func():
            passed += 1
    
    write_log(f"\n=== SUMMARY ===")
    write_log(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        write_log("[SUCCESS] All tests passed! System is ready.")
        return 0
    else:
        write_log(f"[WARNING] {total-passed} tests failed. Check details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())