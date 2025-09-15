#!/usr/bin/env python3
"""
Quick Trading System Test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def main():
    print("=== Quick Trading System Test ===")

    # Test 1: Check restructured directories
    try:
        root = project_root

        # Check new directory structure
        assert (root / "bin").exists(), "bin directory missing"
        assert (root / "config").exists(), "config directory missing"
        assert (root / "tests").exists(), "tests directory missing"
        assert (root / "deployment").exists(), "deployment directory missing"

        print("[PASS] Directory structure")
    except Exception as e:
        print(f"[FAIL] Directory structure: {e}")
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

    # Test 5: Check if new bin utilities work
    try:
        import subprocess

        result = subprocess.run(
            [str(project_root / "bin" / "metadata_manager"), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("[PASS] Metadata manager utility")
        else:
            print(f"[FAIL] Metadata manager utility: {result.stderr}")
            return 1
    except Exception as e:
        print(f"[FAIL] Metadata manager utility: {e}")
        return 1

    print("\n[SUCCESS] Core restructured components working!")
    print("Directory structure and utilities are functional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
