#!/usr/bin/env python3
"""
Quick Test: Verify Prediction Fixes
===================================

Tests if the prediction pipeline now works with the new metadata files.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.config.config_loader import ConfigLoader
from src.data_pipeline.feature_engine import FeatureEngine


def create_test_data():
    """Create test market data."""
    np.random.seed(42)

    dates = pd.date_range(start="2025-01-01", periods=200, freq="30T")
    base_price = 50000

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": base_price + np.random.randn(200) * 100,
            "high": base_price + np.random.randn(200) * 120 + 50,
            "low": base_price + np.random.randn(200) * 80 - 50,
            "close": base_price + np.random.randn(200) * 100,
            "volume": np.random.exponential(1000000, 200),
            "quote_volume": np.random.exponential(50000000, 200),
            "trades": np.random.randint(1000, 5000, 200),
        }
    )

    # Fix OHLC relationships
    df["high"] = np.maximum(df[["open", "close"]].max(axis=1), df["high"])
    df["low"] = np.minimum(df[["open", "close"]].min(axis=1), df["low"])

    return df


def test_prediction_pipeline():
    """Test the complete prediction pipeline."""
    print("Testing Prediction Pipeline with Metadata Fixes...")
    print("=" * 50)

    # Initialize components
    config_loader = ConfigLoader("training_config.yaml")
    config = config_loader.config
    feature_engine = FeatureEngine(config)

    # Test data
    test_data = create_test_data()
    print(f"✅ Created test data: {len(test_data)} rows")

    # Generate features
    try:
        features_df = feature_engine.generate_all_features(test_data)
        print(f"✅ Generated features: {len(features_df.columns)} columns")

        # Test feature alignment for each model type
        symbols = ["BTCEUR", "ETHEUR"]
        model_types = ["gru", "lightgbm", "ppo"]

        results = {}

        for symbol in symbols:
            results[symbol] = {}
            print(f"\n--- Testing {symbol} ---")

            for model_type in model_types:
                try:
                    aligned_features = feature_engine.pad_features_for_model(
                        features_df, model_type, symbol
                    )

                    if aligned_features is not None and not aligned_features.empty:
                        print(f"✅ {model_type}: {len(aligned_features.columns)} features aligned")
                        results[symbol][model_type] = "SUCCESS"
                    else:
                        print(f"❌ {model_type}: Failed - returned None/empty")
                        results[symbol][model_type] = "FAILED"

                except Exception as e:
                    print(f"❌ {model_type}: Error - {e}")
                    results[symbol][model_type] = f"ERROR: {e}"

        # Summary
        print(f"\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)

        total_tests = len(symbols) * len(model_types)
        successful_tests = sum(1 for s in results.values() for r in s.values() if r == "SUCCESS")

        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")

        if successful_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Prediction pipeline should work now.")
            return True
        else:
            print("⚠️  Some tests failed. Check the errors above.")
            return False

    except Exception as e:
        print(f"❌ Feature generation failed: {e}")
        return False


if __name__ == "__main__":
    success = test_prediction_pipeline()
    sys.exit(0 if success else 1)
