"""
Model Feature Router
===================

Professional routing system that ensures each model type receives the exact
feature set it was trained on, maintaining consistency between training and inference.

Features:
- Model-specific feature routing
- Backward compatibility with existing models
- Intelligent feature mapping and alignment
- Performance monitoring and validation
- Error handling and fallback mechanisms
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .enhanced_feature_engine import EnhancedFeatureEngine
from .feature_selector import FeatureSelector
from .ppo_feature_expansion import PPOFeatureExpander

logger = logging.getLogger(__name__)


class ModelFeatureRouter:
    """
    Professional feature routing system for different model types.

    This router ensures that:
    1. PPO models receive exactly 104 features
    2. GRU models receive exactly 100 features
    3. LightGBM models receive exactly 100 features
    4. Feature names and order are consistent with trained models
    5. Graceful handling of missing features or model metadata
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model feature router.

        Args:
            config_path: Path to feature configuration file
        """
        self.config_path = config_path or "config/feature_config.yaml"
        self.config = self._load_config()

        # Initialize core components
        self.enhanced_engine = EnhancedFeatureEngine(config_path)
        self.feature_selector = FeatureSelector(self.config)
        self.ppo_expander = PPOFeatureExpander()  # PPO-specific feature expander

        # Model metadata cache
        self.model_metadata_cache = {}

        # Performance tracking
        self.routing_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "fallback_routes": 0,
            "errors": 0,
        }

        logger.info("🔀 Model Feature Router initialized")
        logger.info(f"Configuration: {self.config_path}")

    def route_features_for_model(
        self,
        df: pd.DataFrame,
        model_type: str,
        symbol: str,
        use_enhanced_engine: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Route features to the appropriate model with comprehensive validation.

        Args:
            df: Input DataFrame with OHLCV data
            model_type: Model type ('ppo', 'gru', 'lightgbm')
            symbol: Trading symbol (e.g., 'BTCEUR')
            use_enhanced_engine: Whether to use enhanced feature generation

        Returns:
            Tuple of (processed_dataframe, routing_info)
        """
        self.routing_stats["total_requests"] += 1

        routing_info = {
            "model_type": model_type,
            "symbol": symbol,
            "input_shape": df.shape,
            "method_used": "unknown",
            "feature_count": 0,
            "success": False,
            "warnings": [],
            "processing_time": 0,
        }

        try:
            import time

            start_time = time.time()

            logger.info(f"🔀 Routing features for {model_type.upper()}_{symbol}")

            # Validate input
            self._validate_input(df, model_type, symbol)

            # Try different routing strategies in order of preference
            result_df = None

            # PPO-specific routing: Use PPO Feature Expander for 104 features
            if model_type == "ppo":
                result_df, method_info = self._route_ppo_features(df, symbol)
                if result_df is not None and not result_df.empty:
                    routing_info["method_used"] = "ppo_feature_expander"
                    routing_info.update(method_info)

            # Strategy 1: Use model-specific metadata if available
            if result_df is None:
                result_df, method_info = self._route_with_model_metadata(df, model_type, symbol)
                if result_df is not None and not result_df.empty:
                    routing_info["method_used"] = "model_metadata"
                    routing_info.update(method_info)

            # Strategy 2: Use enhanced feature engine
            if result_df is None and use_enhanced_engine:
                result_df, method_info = self._route_with_enhanced_engine(df, model_type, symbol)
                if result_df is not None and not result_df.empty:
                    routing_info["method_used"] = "enhanced_engine"
                    routing_info.update(method_info)

            # Strategy 3: Fallback to feature selector alignment
            if result_df is None:
                result_df, method_info = self._route_with_feature_selector(df, model_type, symbol)
                if result_df is not None and not result_df.empty:
                    routing_info["method_used"] = "feature_selector"
                    routing_info.update(method_info)
                    self.routing_stats["fallback_routes"] += 1

            # Strategy 4: Emergency fallback
            if result_df is None or result_df.empty:
                result_df, method_info = self._emergency_fallback_routing(df, model_type, symbol)
                routing_info["method_used"] = "emergency_fallback"
                routing_info.update(method_info)
                routing_info["warnings"].append("Used emergency fallback routing")

            # Final validation and statistics
            routing_info["output_shape"] = result_df.shape
            routing_info["feature_count"] = len(self._get_feature_columns(result_df))
            routing_info["processing_time"] = time.time() - start_time
            routing_info["success"] = True

            # Validate output meets model requirements
            self._validate_output(result_df, model_type, symbol, routing_info)

            self.routing_stats["successful_routes"] += 1

            logger.info(
                f"✅ Successfully routed {routing_info['feature_count']} features for {model_type.upper()}_{symbol}"
            )
            logger.debug(
                f"Routing method: {routing_info['method_used']}, Time: {routing_info['processing_time']:.3f}s"
            )

            return result_df, routing_info

        except Exception as e:
            self.routing_stats["errors"] += 1
            routing_info["success"] = False
            routing_info["error"] = str(e)

            logger.error(f"❌ Feature routing failed for {model_type}_{symbol}: {e}")

            # If strict pinning error, do not silently recover
            if "PPO_FEATURE_INDEX_MISSING_STRICT" in str(e):
                logger.error("Strict pinning is enabled; aborting feature routing to surface error")
                raise e

            # Return emergency fallback for non-strict errors
            try:
                emergency_df, _ = self._emergency_fallback_routing(df, model_type, symbol)
                routing_info["method_used"] = "error_recovery"
                routing_info["warnings"].append(f"Error recovery used due to: {e}")
                return emergency_df, routing_info
            except:
                logger.error(f"❌ Emergency fallback also failed for {model_type}_{symbol}")
                raise e

    def _route_ppo_features(
        self, df: pd.DataFrame, symbol: str
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Route PPO features using dedicated PPO Feature Expander."""
        try:
            logger.info(f"🚀 Using PPO Feature Expander for {symbol}")

            # Use PPO-specific feature expansion, with optional per-symbol pinning
            result_df = self.ppo_expander.expand_features(df, symbol=symbol)

            # Validate the expansion
            if self.ppo_expander.validate_features(result_df):
                feature_names = self.ppo_expander.get_feature_names()
                logger.info(f"✅ PPO features expanded successfully: {len(feature_names)} features")

                return result_df, {
                    "expanded_features": len(feature_names),
                    "method": "ppo_feature_expander",
                    "feature_names": (
                        feature_names[:10] + ["..."] if len(feature_names) > 10 else feature_names
                    ),
                }
            else:
                logger.error("PPO feature expansion validation failed")
                return None, {"reason": "validation_failed"}

        except Exception as e:
            msg = str(e)
            if "PPO_FEATURE_INDEX_MISSING_STRICT" in msg:
                logger.error(f"PPO strict pinning error: {e}")
                # Escalate to caller to prevent silent fallbacks
                raise
            logger.error(f"PPO feature routing failed: {e}")
            return None, {"reason": "ppo_expansion_error", "error": msg}

    def _route_with_model_metadata(
        self, df: pd.DataFrame, model_type: str, symbol: str
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Route using saved model metadata."""
        try:
            metadata = self._load_model_metadata(model_type, symbol)
            if not metadata:
                return None, {"reason": "no_metadata"}

            expected_features = metadata.get("expected_features", [])
            if not expected_features:
                return None, {"reason": "no_feature_list"}

            # Generate comprehensive features first
            comprehensive_df = self.enhanced_engine.generate_features_for_model(
                df, model_type, symbol
            )

            # Align with expected features
            result_df = comprehensive_df.copy()

            # Add missing features as zeros
            for feature in expected_features:
                if feature not in result_df.columns:
                    result_df[feature] = 0.0

            # Keep only expected features (preserve OHLCV)
            ohlcv_cols = [
                col
                for col in result_df.columns
                if col in ["open", "high", "low", "close", "volume"]
            ]
            final_cols = ohlcv_cols + expected_features
            result_df = result_df[final_cols]

            logger.debug(f"Routed with model metadata: {len(expected_features)} features")

            return result_df, {
                "expected_features_count": len(expected_features),
                "metadata_source": metadata.get("source", "unknown"),
                "alignment_method": "metadata_driven",
            }

        except Exception as e:
            logger.debug(f"Model metadata routing failed: {e}")
            return None, {"reason": "metadata_error", "error": str(e)}

    def _route_with_enhanced_engine(
        self, df: pd.DataFrame, model_type: str, symbol: str
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Route using enhanced feature engine."""
        try:
            result_df = self.enhanced_engine.generate_features_for_model(df, model_type, symbol)
            feature_count = len(self._get_feature_columns(result_df))

            logger.debug(f"Enhanced engine generated {feature_count} features")

            return result_df, {
                "generated_features": feature_count,
                "engine_version": "enhanced",
                "generation_method": "comprehensive",
            }

        except Exception as e:
            logger.debug(f"Enhanced engine routing failed: {e}")
            return None, {"reason": "enhanced_engine_error", "error": str(e)}

    def _route_with_feature_selector(
        self, df: pd.DataFrame, model_type: str, symbol: str
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Route using feature selector (backward compatibility)."""
        try:
            # Generate basic features first (using existing pipeline)
            from .features import FeatureEngine

            basic_engine = FeatureEngine()
            features_df = basic_engine.generate_all_features(df)

            # Use feature selector for alignment
            result_df = self.feature_selector.align_features_for_model(
                features_df, model_type, symbol
            )

            feature_count = len(self._get_feature_columns(result_df))

            logger.debug(f"Feature selector aligned to {feature_count} features")

            return result_df, {
                "aligned_features": feature_count,
                "selector_method": "intelligent_selection",
                "fallback_level": "feature_selector",
            }

        except Exception as e:
            logger.debug(f"Feature selector routing failed: {e}")
            return None, {"reason": "selector_error", "error": str(e)}

    def _emergency_fallback_routing(
        self, df: pd.DataFrame, model_type: str, symbol: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Emergency fallback routing with minimal features."""
        logger.warning(f"Using emergency fallback routing for {model_type}_{symbol}")

        # Generate minimal feature set
        result_df = df.copy()

        # Add basic technical indicators
        result_df["sma_20"] = df["close"].rolling(20).mean().fillna(df["close"])
        result_df["rsi_14"] = self._simple_rsi(df["close"], 14)
        result_df["price_change"] = df["close"].pct_change().fillna(0)
        result_df["volume_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()).fillna(1)

        # Get expected count for model
        expected_count = self._get_expected_feature_count(model_type)
        current_features = self._get_feature_columns(result_df)

        # Pad with zeros if needed
        while len(current_features) < expected_count:
            pad_feature_name = f"emergency_pad_{len(current_features)}"
            result_df[pad_feature_name] = 0.0
            current_features.append(pad_feature_name)

        # Truncate if too many features
        if len(current_features) > expected_count:
            feature_cols_to_keep = current_features[:expected_count]
            ohlcv_cols = [
                col
                for col in result_df.columns
                if col in ["open", "high", "low", "close", "volume"]
            ]
            result_df = result_df[ohlcv_cols + feature_cols_to_keep]
            current_features = feature_cols_to_keep

        return result_df, {
            "emergency_features": len(current_features),
            "expected_count": expected_count,
            "method": "minimal_technical_indicators",
        }

    def _simple_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Simple RSI calculation for emergency fallback."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _load_model_metadata(self, model_type: str, symbol: str) -> Optional[Dict]:
        """Load model-specific metadata."""
        cache_key = f"{model_type}_{symbol}"

        if cache_key in self.model_metadata_cache:
            return self.model_metadata_cache[cache_key]

        metadata_paths = [
            Path(f"models/metadata/features_{model_type}_{symbol}.json"),
            Path(f"models/{model_type}/{symbol}/metadata.json"),
            Path(f"models/{model_type}/{symbol}/features.json"),
        ]

        for path in metadata_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        metadata = json.load(f)

                    metadata["source"] = str(path)
                    self.model_metadata_cache[cache_key] = metadata

                    logger.debug(f"Loaded metadata for {cache_key} from {path}")
                    return metadata

                except Exception as e:
                    logger.debug(f"Failed to load metadata from {path}: {e}")
                    continue

        logger.debug(f"No metadata found for {cache_key}")
        return None

    def _get_expected_feature_count(self, model_type: str) -> int:
        """Get expected feature count for model type."""
        defaults = {
            "ppo": 104,  # PPO models trained with 104 features
            "gru": 100,
            "lightgbm": 100,
        }

        config_count = (
            self.config.get("models", {}).get(model_type, {}).get("expected_feature_count")
        )
        if config_count:
            return config_count

        return defaults.get(model_type, 100)

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names (excluding OHLCV)."""
        excluded_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "target",
        ]
        return [col for col in df.columns if col not in excluded_cols]

    def _validate_input(self, df: pd.DataFrame, model_type: str, symbol: str):
        """Validate input parameters."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if model_type not in ["ppo", "gru", "lightgbm"]:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _validate_output(
        self,
        df: pd.DataFrame,
        model_type: str,
        symbol: str,
        routing_info: Dict[str, Any],
    ):
        """Validate output meets model requirements."""
        expected_count = self._get_expected_feature_count(model_type)
        actual_count = routing_info["feature_count"]

        if actual_count != expected_count:
            warning_msg = f"Feature count mismatch: expected {expected_count}, got {actual_count}"
            routing_info["warnings"].append(warning_msg)
            logger.warning(f"⚠️ {warning_msg} for {model_type}_{symbol}")

        # Check for NaN or infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_count = df[numeric_cols].isnull().sum().sum()
        inf_count = np.isinf(df[numeric_cols]).sum().sum()

        if nan_count > 0:
            routing_info["warnings"].append(f"Found {nan_count} NaN values")

        if inf_count > 0:
            routing_info["warnings"].append(f"Found {inf_count} infinite values")

    def _load_config(self) -> Dict:
        """Load routing configuration."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                return {}

            with open(config_file, "r") as f:
                return yaml.safe_load(f)

        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        total = self.routing_stats["total_requests"]
        if total == 0:
            success_rate = 0.0
        else:
            success_rate = self.routing_stats["successful_routes"] / total * 100

        return {
            "total_requests": total,
            "successful_routes": self.routing_stats["successful_routes"],
            "fallback_routes": self.routing_stats["fallback_routes"],
            "errors": self.routing_stats["errors"],
            "success_rate": success_rate,
            "cache_size": len(self.model_metadata_cache),
        }

    def clear_cache(self):
        """Clear metadata cache."""
        self.model_metadata_cache.clear()
        logger.info("Metadata cache cleared")

    def preload_model_metadata(self, model_types: List[str], symbols: List[str]):
        """Preload metadata for specified models and symbols."""
        logger.info(
            f"Preloading metadata for {len(model_types)} model types and {len(symbols)} symbols"
        )

        loaded_count = 0
        for model_type in model_types:
            for symbol in symbols:
                metadata = self._load_model_metadata(model_type, symbol)
                if metadata:
                    loaded_count += 1

        logger.info(f"Preloaded metadata for {loaded_count} model configurations")


# Convenience function for quick feature routing
def route_features(
    df: pd.DataFrame,
    model_type: str,
    symbol: str = "GENERIC",
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to route features for a specific model.

    Args:
        df: Input DataFrame with OHLCV data
        model_type: Model type ('ppo', 'gru', 'lightgbm')
        symbol: Trading symbol
        config_path: Path to configuration file

    Returns:
        DataFrame with routed features
    """
    router = ModelFeatureRouter(config_path)
    result_df, routing_info = router.route_features_for_model(df, model_type, symbol)

    if not routing_info["success"]:
        logger.warning(f"Feature routing had issues: {routing_info.get('warnings', [])}")

    return result_df
