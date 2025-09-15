#!/usr/bin/env python3
"""
Integration Test for All Models
================================

Verify that all model types (PPO, GRU, LightGBM) can receive the correct
number of features and work properly with the new feature routing system.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, "/opt/trading_bot/bot/src")

# Import required modules
from data_pipeline.model_feature_router import ModelFeatureRouter

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_market_data(symbol: str, n_rows: int = 1000) -> pd.DataFrame:
    """Generate realistic market data for testing."""
    np.random.seed(hash(symbol) % 1000)  # Different seed per symbol

    # Base price based on symbol
    base_prices = {
        "BTCEUR": 50000,
        "ETHEUR": 3000,
        "ADAEUR": 0.5,
        "DOTEUR": 10,
        "LINKEUR": 15,
    }

    base_price = base_prices.get(symbol, 100)

    # Generate realistic price movement
    returns = np.random.randn(n_rows) * 0.02  # 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_rows) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_rows) * 0.002)),
            "low": prices * (1 - np.abs(np.random.randn(n_rows) * 0.002)),
            "close": prices,
            "volume": np.abs(np.random.randn(n_rows) * 1000000) + 100000,
        }
    )

    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def test_model_features(router: ModelFeatureRouter, model_type: str, symbol: str) -> dict:
    """Test feature generation for a specific model and symbol."""
    result = {
        "model_type": model_type,
        "symbol": symbol,
        "success": False,
        "feature_count": 0,
        "expected_count": 0,
        "method_used": "",
        "warnings": [],
        "error": None,
    }

    try:
        # Generate sample data
        df = generate_market_data(symbol)
        logger.info(f"Testing {model_type.upper()}_{symbol} with {len(df)} rows")

        # Route features
        routed_df, routing_info = router.route_features_for_model(df, model_type, symbol)

        # Get expected feature count
        expected_count = router._get_expected_feature_count(model_type)

        # Extract results
        result["success"] = routing_info.get("success", False)
        result["feature_count"] = routing_info.get("feature_count", 0)
        result["expected_count"] = expected_count
        result["method_used"] = routing_info.get("method_used", "unknown")
        result["warnings"] = routing_info.get("warnings", [])

        # Validate feature count
        if result["feature_count"] == expected_count:
            result["validation"] = "PASSED"
            logger.info(
                f"✅ {model_type.upper()}_{symbol}: {result['feature_count']} features (method: {result['method_used']})"
            )
        else:
            result["validation"] = "FAILED"
            logger.error(
                f"❌ {model_type.upper()}_{symbol}: Feature count mismatch - expected {expected_count}, got {result['feature_count']}"
            )

        # Check for data quality
        if routed_df is not None:
            nan_count = routed_df.isnull().sum().sum()
            inf_count = np.isinf(routed_df.select_dtypes(include=[np.number])).sum().sum()

            if nan_count > 0:
                result["warnings"].append(f"{nan_count} NaN values found")
            if inf_count > 0:
                result["warnings"].append(f"{inf_count} infinite values found")

    except Exception as e:
        result["error"] = str(e)
        result["validation"] = "ERROR"
        logger.error(f"❌ {model_type.upper()}_{symbol}: Error - {e}")

    return result


def main():
    """Main test runner."""
    print("\n" + "=" * 80)
    print("INTEGRATION TEST: ALL MODELS FEATURE ROUTING")
    print("=" * 80 + "\n")

    # Initialize router
    router = ModelFeatureRouter()

    # Test configurations
    models = ["ppo", "gru", "lightgbm"]
    symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]

    # Results storage
    all_results = []

    # Test each combination
    print("Testing Model-Symbol Combinations:")
    print("-" * 50)

    for model_type in models:
        for symbol in symbols:
            result = test_model_features(router, model_type, symbol)
            all_results.append(result)

    # Summary statistics
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80 + "\n")

    # Model-level summary
    print("By Model Type:")
    print("-" * 50)
    for model_type in models:
        model_results = [r for r in all_results if r["model_type"] == model_type]
        passed = sum(1 for r in model_results if r["validation"] == "PASSED")
        failed = sum(1 for r in model_results if r["validation"] == "FAILED")
        errors = sum(1 for r in model_results if r["validation"] == "ERROR")

        status = "✅" if passed == len(model_results) else "❌"
        print(
            f"{model_type.upper():10} - Passed: {passed}/{len(model_results)}, Failed: {failed}, Errors: {errors} {status}"
        )

        # Show expected feature count
        if model_results:
            expected = model_results[0]["expected_count"]
            print(f"            Expected features: {expected}")

    # Symbol-level summary
    print("\nBy Symbol:")
    print("-" * 50)
    for symbol in symbols:
        symbol_results = [r for r in all_results if r["symbol"] == symbol]
        passed = sum(1 for r in symbol_results if r["validation"] == "PASSED")
        failed = sum(1 for r in symbol_results if r["validation"] == "FAILED")
        errors = sum(1 for r in symbol_results if r["validation"] == "ERROR")

        status = "✅" if passed == len(symbol_results) else "❌"
        print(
            f"{symbol:10} - Passed: {passed}/{len(symbol_results)}, Failed: {failed}, Errors: {errors} {status}"
        )

    # Method usage summary
    print("\nRouting Methods Used:")
    print("-" * 50)
    method_counts = {}
    for result in all_results:
        method = result["method_used"]
        if method:
            method_counts[method] = method_counts.get(method, 0) + 1

    for method, count in sorted(method_counts.items()):
        print(f"{method:25} - {count} times")

    # Overall summary
    print("\n" + "=" * 80)
    total_passed = sum(1 for r in all_results if r["validation"] == "PASSED")
    total_failed = sum(1 for r in all_results if r["validation"] == "FAILED")
    total_errors = sum(1 for r in all_results if r["validation"] == "ERROR")
    total_tests = len(all_results)

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"OVERALL RESULTS: {total_passed}/{total_tests} tests passed ({success_rate:.1f}%)")
    print(f"Failed: {total_failed}, Errors: {total_errors}")

    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED! Feature routing is working correctly for all models.")
    else:
        print("\n❌ SOME TESTS FAILED. Please review the errors above.")

    # Show any warnings
    warnings_found = [r for r in all_results if r["warnings"]]
    if warnings_found:
        print("\n⚠️ Warnings found in {len(warnings_found)} tests:")
        for result in warnings_found[:5]:  # Show first 5
            print(
                f"  - {result['model_type'].upper()}_{result['symbol']}: {', '.join(result['warnings'])}"
            )

    print("=" * 80)

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
