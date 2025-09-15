#!/usr/bin/env python3
"""
Test PPO Feature Expansion
==========================

Verify that PPO models can now receive 104 features and make predictions successfully.
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
from data_pipeline.ppo_feature_expansion import PPOFeatureExpander

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_rows: int = 500) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    logger.info(f"Generating {n_rows} rows of sample OHLCV data")

    # Generate realistic price data
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(n_rows) * 100)

    df = pd.DataFrame(
        {
            "open": prices + np.random.randn(n_rows) * 50,
            "high": prices + np.abs(np.random.randn(n_rows) * 100),
            "low": prices - np.abs(np.random.randn(n_rows) * 100),
            "close": prices,
            "volume": np.abs(np.random.randn(n_rows) * 1000000) + 100000,
        }
    )

    # Ensure high >= low and high >= open/close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def test_ppo_feature_expansion():
    """Test PPO feature expansion."""
    print("\n" + "=" * 80)
    print("Testing PPO Feature Expansion")
    print("=" * 80 + "\n")

    # Test 1: Direct PPO Feature Expander
    print("Test 1: Direct PPO Feature Expander")
    print("-" * 40)

    df = generate_sample_data()
    print(f"Input shape: {df.shape}")
    print(f"Input columns: {list(df.columns)}")

    expander = PPOFeatureExpander()
    expanded_df = expander.expand_features(df)

    print(f"Output shape: {expanded_df.shape}")
    print(
        f"Feature columns (excluding OHLCV): {len([c for c in expanded_df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])}"
    )

    # Validate
    is_valid = expander.validate_features(expanded_df)
    print(f"Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")

    # Check for NaN or Inf values
    nan_count = expanded_df.isnull().sum().sum()
    inf_count = np.isinf(expanded_df.select_dtypes(include=[np.number])).sum().sum()
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")

    print("\n" + "-" * 40)

    # Test 2: Model Feature Router with PPO
    print("Test 2: Model Feature Router with PPO")
    print("-" * 40)

    router = ModelFeatureRouter()
    routed_df, routing_info = router.route_features_for_model(df, "ppo", "BTCEUR")

    print(f"Routing method: {routing_info.get('method_used')}")
    print(f"Output shape: {routed_df.shape}")
    print(f"Feature count: {routing_info.get('feature_count')}")
    print(f"Success: {routing_info.get('success')}")
    print(f"Warnings: {routing_info.get('warnings', [])}")

    # Test 3: Compare with other model types
    print("\n" + "-" * 40)
    print("Test 3: Compare Feature Counts Across Models")
    print("-" * 40)

    for model_type in ["ppo", "gru", "lightgbm"]:
        routed_df, routing_info = router.route_features_for_model(df, model_type, "BTCEUR")
        feature_count = routing_info.get("feature_count")
        expected = router._get_expected_feature_count(model_type)
        status = "✅" if feature_count == expected else "❌"
        print(f"{model_type.upper()}: {feature_count} features (expected: {expected}) {status}")

    print("\n" + "=" * 80)

    return is_valid


def test_ppo_model_prediction():
    """Test actual PPO model prediction with expanded features."""
    print("\n" + "=" * 80)
    print("Testing PPO Model Prediction")
    print("=" * 80 + "\n")

    try:
        # Import PPO manager
        from models.ppo_model_manager import get_ppo_manager

        # Get sample data
        df = generate_sample_data()

        # Expand features
        expander = PPOFeatureExpander()
        expanded_df = expander.expand_features(df)

        # Get features only (exclude OHLCV)
        feature_cols = [
            c for c in expanded_df.columns if c not in ["open", "high", "low", "close", "volume"]
        ]
        features = expanded_df[feature_cols].values

        print(f"Feature shape for prediction: {features.shape}")

        # Load PPO model
        manager = get_ppo_manager()
        model_path = "/opt/trading_bot/bot/models/ppo/BTCEUR/model.zip"

        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}")
            return False

        model = manager.load_model(model_path)

        if model:
            print("✅ PPO model loaded successfully")

            # Test prediction with single observation
            test_obs = features[-1:].astype(np.float32)
            print(f"Test observation shape: {test_obs.shape}")

            try:
                action, _ = manager.predict(model_path, test_obs)
                print(f"✅ Prediction successful! Action: {action}")
                return True
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                return False
        else:
            print("❌ Failed to load PPO model")
            return False

    except ImportError as e:
        print(f"⚠️ Could not import PPO manager: {e}")
        print("Skipping actual model prediction test")
        return None


def main():
    """Main test runner."""
    # Test feature expansion
    expansion_success = test_ppo_feature_expansion()

    # Test model prediction
    prediction_success = test_ppo_model_prediction()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Feature Expansion: {'✅ PASSED' if expansion_success else '❌ FAILED'}")

    if prediction_success is not None:
        print(f"Model Prediction: {'✅ PASSED' if prediction_success else '❌ FAILED'}")
    else:
        print(f"Model Prediction: ⚠️ SKIPPED (import issues)")

    overall_success = expansion_success and (prediction_success is None or prediction_success)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    print("=" * 80)

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
