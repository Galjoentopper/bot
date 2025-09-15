"""
Feature Selection and Alignment Module
=====================================

Handles feature selection and alignment between training and deployment.
Ensures models receive exactly the features they were trained on.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selector that aligns runtime features with training-time selections.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature selector."""
        self.config = config or {}
        self.feature_mappings = self._load_feature_mappings()
        self.model_feature_cache = {}

    def align_features_for_model(
        self, features_df: pd.DataFrame, model_type: str, symbol: str
    ) -> pd.DataFrame:
        """
        Align features for specific model type and symbol.

        Args:
            features_df: DataFrame with generated features
            model_type: Type of model ('gru', 'lightgbm', 'ppo')
            symbol: Trading symbol (e.g., 'BTCEUR')

        Returns:
            DataFrame with model-aligned features
        """
        logger.info(f"Aligning features for {model_type}_{symbol}")

        # Try to load model-specific feature selection
        aligned_df = self._apply_model_specific_selection(features_df, model_type, symbol)
        if aligned_df is not None:
            return aligned_df

        # Fall back to count-based alignment with intelligent selection
        return self._apply_intelligent_feature_selection(features_df, model_type, symbol)

    def _apply_model_specific_selection(
        self, features_df: pd.DataFrame, model_type: str, symbol: str
    ) -> Optional[pd.DataFrame]:
        """Apply model-specific feature selection if metadata available."""
        # CRITICAL FIX: Validate input DataFrame
        if features_df is None or features_df.empty:
            logger.warning(f"Invalid input DataFrame for {model_type}_{symbol}")
            return features_df if features_df is not None else pd.DataFrame()

        # Try to load saved feature names from model metadata
        model_key = f"{model_type}_{symbol}"

        if model_key in self.model_feature_cache:
            expected_features = self.model_feature_cache[model_key]
        else:
            expected_features = self._load_model_feature_names(model_type, symbol)
            if expected_features:
                self.model_feature_cache[model_key] = expected_features

        if not expected_features:
            logger.debug(f"No metadata found for {model_key}, using fallback")
            return None

        logger.info(f"Found {len(expected_features)} expected features for {model_key}")

        # Align features with expected names
        aligned_df = features_df.copy()

        # Add missing features as zeros
        for feature in expected_features:
            if feature not in aligned_df.columns:
                aligned_df[feature] = 0.0

        # Keep only expected features in correct order
        try:
            aligned_df = aligned_df[expected_features]
            logger.info(f"Successfully aligned {len(expected_features)} features for {model_key}")
            return aligned_df
        except KeyError as e:
            logger.warning(f"Could not align all expected features for {model_key}: {e}")
            return None

    def _apply_intelligent_feature_selection(
        self, features_df: pd.DataFrame, model_type: str, symbol: str
    ) -> pd.DataFrame:
        """Apply intelligent feature selection based on model requirements."""
        # Get expected feature count from mapping
        expected_count = self._get_expected_feature_count(model_type, symbol)
        current_features = self._get_numeric_features(features_df)
        current_count = len(current_features)

        logger.info(
            f"Intelligent selection: {current_count} -> {expected_count} features for {model_type}_{symbol}"
        )

        if current_count == expected_count:
            return features_df[current_features]
        elif current_count < expected_count:
            # Pad with zero features
            result_df = features_df[current_features].copy()
            for i in range(current_count, expected_count):
                result_df[f"pad_feature_{i}"] = 0.0
            return result_df
        else:
            # Select best features
            return self._select_best_features(
                features_df, current_features, expected_count, model_type
            )

    def _select_best_features(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        target_count: int,
        model_type: str,
    ) -> pd.DataFrame:
        """Select best features using importance-based selection."""
        try:
            # Use price-based importance for feature selection
            if "close" not in features_df.columns:
                logger.warning("No target variable available, using random selection")
                selected_features = feature_cols[:target_count]
                return features_df[selected_features]

            # Create target based on future returns
            target = features_df["close"].pct_change().shift(-1).fillna(0)
            feature_data = features_df[feature_cols].fillna(0)

            # Remove features with zero variance
            feature_data = feature_data.loc[:, feature_data.var() > 1e-10]
            remaining_features = list(feature_data.columns)

            if len(remaining_features) <= target_count:
                logger.info(f"After variance filtering: {len(remaining_features)} features")
                result_df = feature_data.copy()
                # Pad if needed
                for i in range(len(remaining_features), target_count):
                    result_df[f"pad_feature_{i}"] = 0.0
                return result_df

            # Use mutual information for feature selection
            logger.info(f"Selecting {target_count} best features from {len(remaining_features)}")
            selector = SelectKBest(score_func=mutual_info_regression, k=target_count)

            selected_data = selector.fit_transform(feature_data, target)
            selected_feature_names = [
                remaining_features[i] for i in selector.get_support(indices=True)
            ]

            result_df = pd.DataFrame(
                selected_data, columns=selected_feature_names, index=feature_data.index
            )
            logger.info(
                f"Selected features for {model_type}: {selected_feature_names[:10]}..."
                if len(selected_feature_names) > 10
                else f"Selected features: {selected_feature_names}"
            )

            return result_df

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using first {target_count} features")
            selected_features = feature_cols[:target_count]
            return features_df[selected_features]

    def _get_expected_feature_count(self, model_type: str, symbol: str) -> int:
        """Get expected feature count from mappings."""
        if self.feature_mappings:
            # Try model-specific mapping first
            model_key = f"{model_type}_{symbol}"
            if "models" in self.feature_mappings and model_key in self.feature_mappings["models"]:
                return self.feature_mappings["models"][model_key].get("expected_feature_count", 100)

            # Try feature_counts mapping
            if (
                "feature_counts" in self.feature_mappings
                and model_key in self.feature_mappings["feature_counts"]
            ):
                return self.feature_mappings["feature_counts"][model_key]

        # Use correct feature counts matching the trained models
        defaults = {
            "lightgbm": 113,  # LightGBM models trained on 113 features
            "gru": 113,  # GRU models trained on 113 features
            "ppo": 104,  # PPO models expect 104 features as defined in PPOFeatureExpander
        }
        default_count = defaults.get(model_type, 100)
        logger.info(f"Using dynamic default feature count {default_count} for {model_type}")
        return default_count

    def _get_numeric_features(self, df: pd.DataFrame) -> List[str]:
        """Get numeric feature columns excluding OHLCV."""
        excluded_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "timestamp",
            "target",
        ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in excluded_cols]
        return feature_cols

    def _load_model_feature_names(self, model_type: str, symbol: str) -> Optional[List[str]]:
        """Load model-specific feature names from metadata."""
        search_paths = [
            # PRIORITY: Load from our feature metadata files first
            Path(f"models/metadata/features_{model_type}_{symbol}.json"),
            # Fallback paths for compatibility
            Path(f"models/{model_type}/{symbol}/features.json"),
            Path(f"models/{model_type}/{symbol}/metadata.json"),
            Path(f"models/{model_type}/{symbol}/feature_names.pkl"),
            Path(f"models/imported/{model_type}_{symbol}_features.json"),
            Path(f"models/imported/{model_type}_{symbol}_feature_names.pkl"),
        ]

        for path in search_paths:
            if path.exists():
                try:
                    if path.suffix == ".json":
                        with open(path, "r") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                return data
                            elif isinstance(data, dict):
                                # Support multiple metadata formats
                                features = (
                                    data.get("expected_features")
                                    or data.get("feature_names")
                                    or data.get("features")
                                )
                                if features and isinstance(features, list):
                                    logger.info(f"Loaded {len(features)} features from {path}")
                                    return features
                    elif path.suffix == ".pkl":
                        with open(path, "rb") as f:
                            feature_names = pickle.load(f)
                            if isinstance(feature_names, list):
                                return feature_names
                except Exception as e:
                    logger.debug(f"Failed to load feature names from {path}: {e}")
                    continue

        logger.debug(f"No feature names found for {model_type}_{symbol}")
        return None

    def _load_feature_mappings(self) -> Optional[Dict]:
        """Load feature mappings from metadata files."""
        try:
            mapping_paths = [
                Path("feature_mapping.json"),
                Path("config/feature_mapping.json"),
                Path("models/feature_mapping.json"),
            ]

            for path in mapping_paths:
                if path.exists():
                    with open(path, "r") as f:
                        mappings = json.load(f)
                        logger.info(f"Loaded feature mappings from {path}")
                        return mappings

        except Exception as e:
            logger.warning(f"Failed to load feature mappings: {e}")

        return None

    def save_feature_selection(
        self,
        model_type: str,
        symbol: str,
        selected_features: List[str],
        method: str = "deployment",
    ) -> None:
        """Save feature selection for future use."""
        try:
            output_dir = Path(f"models/{model_type}/{symbol}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save as JSON
            feature_data = {
                "feature_names": selected_features,
                "count": len(selected_features),
                "method": method,
                "created_at": pd.Timestamp.now().isoformat(),
            }

            json_path = output_dir / "features.json"
            with open(json_path, "w") as f:
                json.dump(feature_data, f, indent=2)

            # Save as pickle for faster loading
            pkl_path = output_dir / "feature_names.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(selected_features, f)

            logger.info(
                f"Saved feature selection for {model_type}_{symbol}: {len(selected_features)} features"
            )

        except Exception as e:
            logger.error(f"Failed to save feature selection: {e}")


class EnhancedDataPreprocessor:
    """Enhanced preprocessor with persistence and validation."""

    def __init__(self, model_type: str = None, symbol: str = None):
        """Initialize enhanced preprocessor."""
        self.model_type = model_type
        self.symbol = symbol
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        self.metadata = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "EnhancedDataPreprocessor":
        """Fit preprocessor to training data."""
        from sklearn.preprocessing import RobustScaler

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.scaler = RobustScaler()
        self.scaler.fit(X)
        self.is_fitted = True

        # Store metadata
        self.metadata = {
            "feature_count": X.shape[1],
            "fitted_at": pd.Timestamp.now().isoformat(),
            "model_type": self.model_type,
            "symbol": self.symbol,
        }

        logger.info(
            f"Fitted preprocessor for {self.model_type}_{self.symbol}: {X.shape[1]} features"
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform features using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.scaler.transform(X)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Fit preprocessor and transform data."""
        return self.fit(X, y).transform(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save preprocessor to file."""
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Saved preprocessor to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {e}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EnhancedDataPreprocessor":
        """Load preprocessor from file."""
        try:
            with open(filepath, "rb") as f:
                preprocessor = pickle.load(f)
            logger.info(f"Loaded preprocessor from {filepath}")
            return preprocessor
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            return None
