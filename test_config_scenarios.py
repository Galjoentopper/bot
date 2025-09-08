#!/usr/bin/env python3
"""
Test different configuration scenarios for the deploy_trader functionality
"""

import yaml
import tempfile
import os
import subprocess


def create_test_config(content):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(content, f)
        return f.name


def run_config_scenario(
    scenario_name, config_content, expected_symbols, expected_models
):
    """Test a specific configuration scenario."""
    print(f"\n=== Testing: {scenario_name} ===")

    config_file = create_test_config(config_content)

    try:
        # Test symbol extraction
        result = subprocess.run(
            ["python", "test_symbol_extraction.py", config_file],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and "SYMBOLS_FOUND:" in result.stdout:
            found_symbols = result.stdout.strip().split(":")[1].split(",")
            found_symbols = [s.strip() for s in found_symbols if s.strip()]

            if set(found_symbols) == set(expected_symbols):
                print(f"✓ Symbols extracted correctly: {found_symbols}")
            else:
                print(
                    f"✗ Symbol mismatch - Expected: {expected_symbols}, Found: {found_symbols}"
                )
                return False
        else:
            print(f"✗ Symbol extraction failed: {result.stdout}")
            return False

        # Test enhanced trader with these symbols
        cmd = [
            "python",
            "scripts/enhanced_trader.py",
            "--config",
            config_file,
            "--test-mode",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode == 0:
            print("✓ Enhanced trader accepts configuration")
            return True
        else:
            print(f"✗ Enhanced trader failed: {result.stderr}")
            return False

    finally:
        os.unlink(config_file)


def main():
    """Test various configuration scenarios."""
    print("Testing Deploy Trader Configuration Scenarios")
    print("=" * 50)

    scenarios = [
        {
            "name": "Standard data_acquisition format",
            "config": {
                "data_acquisition": {
                    "symbols": ["BTCEUR", "ETHEUR"],
                    "interval": "30m",
                },
                "training": {"models": ["lightgbm", "gru"]},
            },
            "expected_symbols": ["BTCEUR", "ETHEUR"],
            "expected_models": ["lightgbm", "gru"],
        },
        {
            "name": "Alternative data section format",
            "config": {
                "data": {"symbols": ["ADAEUR", "DOTEUR"], "interval": "1h"},
                "training": {"models": ["ppo"]},
            },
            "expected_symbols": ["ADAEUR", "DOTEUR"],
            "expected_models": ["ppo"],
        },
        {
            "name": "Root level symbols",
            "config": {
                "symbols": ["LINKEUR"],
                "interval": "15m",
                "training": {"models": ["gru", "lightgbm", "ppo"]},
            },
            "expected_symbols": ["LINKEUR"],
            "expected_models": ["gru", "lightgbm", "ppo"],
        },
        {
            "name": "Single symbol configuration",
            "config": {"data_acquisition": {"symbols": ["BTCEUR"], "interval": "30m"}},
            "expected_symbols": ["BTCEUR"],
            "expected_models": ["gru", "lightgbm", "ppo"],  # defaults
        },
        {
            "name": "Multiple symbols with specific models",
            "config": {
                "data_acquisition": {
                    "symbols": ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"],
                    "interval": "30m",
                },
                "training": {"models": ["lightgbm"]},
            },
            "expected_symbols": ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"],
            "expected_models": ["lightgbm"],
        },
    ]

    passed = 0
    total = len(scenarios)

    for scenario in scenarios:
        if run_config_scenario(
            scenario["name"],
            scenario["config"],
            scenario["expected_symbols"],
            scenario["expected_models"],
        ):
            passed += 1

    print(f"\n{'='*50}")
    print(f"Configuration Scenario Tests: {passed}/{total} passed")

    if passed == total:
        print("✅ All configuration scenarios work correctly!")
        return 0
    else:
        print("❌ Some configuration scenarios failed")
        return 1


if __name__ == "__main__":
    exit(main())


# Pytest-friendly wrapper: run the script-like test suite
def test_config_scenarios_script():
    assert main() == 0
