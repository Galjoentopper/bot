#!/usr/bin/env python3
"""
Quick Trading System Test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def main():
    print("=== Quick Trading System Test ===")

    # Test 1: Import EnhancedUnifiedPaperTrader
    try:
        from scripts.enhanced_trader import EnhancedUnifiedPaperTrader

        print("[PASS] EnhancedUnifiedPaperTrader import")
    except Exception as e:
        print(f"[FAIL] EnhancedUnifiedPaperTrader import: {e}")
        return 1

    # Test 2: Load config
    try:
        from src.config.config_loader import ConfigLoader

        config = ConfigLoader().config
        print("[PASS] Configuration loading")
    except Exception as e:
        print(f"[FAIL] Configuration loading: {e}")
        return 1

    # Test 3: Initialize ProfitOptimizer with config
    try:
        from src.trading.profit_optimizer import ProfitOptimizer

        optimizer = ProfitOptimizer(config)
        print("[PASS] ProfitOptimizer initialization")
    except Exception as e:
        print(f"[FAIL] ProfitOptimizer initialization: {e}")
        return 1

    # Test 4: Initialize EnhancedSignalGenerator
    try:
        from src.trading.enhanced_signal_generator import EnhancedSignalGenerator

        signal_gen = EnhancedSignalGenerator(config, optimizer)
        print("[PASS] EnhancedSignalGenerator initialization")
    except Exception as e:
        print(f"[FAIL] EnhancedSignalGenerator initialization: {e}")
        return 1

    # Test 5: Try to initialize trader (with show_available_mode to avoid errors)
    try:
        trader = EnhancedUnifiedPaperTrader(config=config, show_available_mode=True)
        print("[PASS] EnhancedUnifiedPaperTrader initialization")
    except Exception as e:
        print(f"[FAIL] EnhancedUnifiedPaperTrader initialization: {e}")
        return 1

    print("\n[SUCCESS] All core components initialized successfully!")
    print("Trading system is ready for testing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
