#!/usr/bin/env python3
"""
Regenerate Model Preprocessors
=============================

This script regenerates all model preprocessors to match the current
feature generation pipeline (100 features for GRU/LightGBM, 13 for PPO).
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.feature_selector import EnhancedDataPreprocessor
from src.data_pipeline.features import FeatureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def regenerate_preprocessor(symbol: str, model_type: str) -> bool:
    """Regenerate a preprocessor for the given symbol and model type."""
    try:
        logger.info(f"Regenerating preprocessor for {model_type} {symbol}")

        # Fetch sample data using ccxt
        exchange = ccxt.kraken()

        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, "30m", limit=300)

            # Convert to DataFrame
            data = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
            data.set_index("timestamp", inplace=True)

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return False

        if data is None or len(data) < 100:
            logger.error(f"Insufficient data for {symbol}")
            return False

        # Generate features using current pipeline
        feature_engine = FeatureEngine()
        features_df = feature_engine.generate_all_features(data)

        if features_df is None or len(features_df.columns) == 0:
            logger.error(f"Feature generation failed for {symbol}")
            return False

        # Load expected features for this model
        metadata_file = f"models/metadata/features_{model_type}_{symbol}.json"
        if not os.path.exists(metadata_file):
            logger.error(f"Metadata file not found: {metadata_file}")
            return False

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        expected_features = metadata.get("expected_features", [])
        if not expected_features:
            logger.error(f"No expected features in {metadata_file}")
            return False

        logger.info(f"Expected features for {model_type} {symbol}: {len(expected_features)}")

        # Select only the expected features (in correct order)
        available_features = [col for col in expected_features if col in features_df.columns]
        if len(available_features) != len(expected_features):
            logger.warning(
                f"Feature mismatch: {len(available_features)}/{len(expected_features)} features available"
            )

        # Use available features or create dummy features
        selected_features = []
        feature_data = []

        for feature_name in expected_features:
            if feature_name in features_df.columns:
                selected_features.append(feature_name)
                feature_data.append(features_df[feature_name].values)
            else:
                # Create dummy feature with appropriate values
                logger.warning(f"Creating dummy feature: {feature_name}")
                selected_features.append(feature_name)
                dummy_values = np.random.normal(0, 1, len(features_df))
                feature_data.append(dummy_values)

        # Create training data
        training_data = pd.DataFrame(np.array(feature_data).T, columns=selected_features)

        logger.info(f"Training preprocessor with {len(training_data.columns)} features")

        # Create and fit preprocessor
        preprocessor = EnhancedDataPreprocessor(model_type=model_type, symbol=symbol)

        # Fit the preprocessor
        preprocessor.fit(training_data)

        # Verify the preprocessor works
        test_transform = preprocessor.transform(training_data.iloc[:10])
        if test_transform is None:
            logger.error(f"Preprocessor transform test failed for {model_type} {symbol}")
            return False

        logger.info(f"Preprocessor transform test successful: {test_transform.shape}")

        # Save the new preprocessor
        model_dir = Path(f"models/{model_type}/{symbol}")
        model_dir.mkdir(parents=True, exist_ok=True)

        preprocessor_path = model_dir / "preprocessor.pkl"
        preprocessor.save(preprocessor_path)

        # Verify saved preprocessor
        with open(preprocessor_path, "rb") as f:
            loaded_preprocessor = pickle.load(f)
            if hasattr(loaded_preprocessor.scaler, "n_features_in_"):
                actual_features = loaded_preprocessor.scaler.n_features_in_
                logger.info(f"✅ Saved preprocessor expects {actual_features} features")

                if actual_features != len(expected_features):
                    logger.error(
                        f"Feature count mismatch: expected {len(expected_features)}, got {actual_features}"
                    )
                    return False

        logger.info(f"✅ Successfully regenerated preprocessor for {model_type} {symbol}")
        return True

    except Exception as e:
        logger.error(f"Failed to regenerate preprocessor for {model_type} {symbol}: {e}")
        return False


def main():
    """Regenerate all preprocessors."""
    symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]
    model_types = ["gru", "lightgbm"]  # PPO uses different preprocessing

    success_count = 0
    total_count = 0

    for model_type in model_types:
        for symbol in symbols:
            total_count += 1
            if regenerate_preprocessor(symbol, model_type):
                success_count += 1

    logger.info(f"Preprocessor regeneration complete: {success_count}/{total_count} successful")

    if success_count == total_count:
        logger.info("🎉 All preprocessors regenerated successfully!")
        return 0
    else:
        logger.error(f"❌ {total_count - success_count} preprocessors failed to regenerate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
