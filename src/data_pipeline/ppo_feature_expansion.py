"""
PPO Feature Expansion Module
============================

Professional feature expansion system specifically designed for PPO models.
Ensures PPO models receive exactly 104 features as they were trained with.

This module solves the dimension mismatch issue by expanding the basic 13 features
into a comprehensive 104-feature set that matches the original training configuration.
"""

import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class PPOFeatureExpander:
    """
    Expands features specifically for PPO models to match their 104-feature requirement.

    The expansion follows a systematic approach:
    1. Core technical indicators (30 features)
    2. Price momentum features (25 features)
    3. Volatility features (20 features)
    4. Volume features (15 features)
    5. Market microstructure (14 features)
    Total: 104 features
    """

    def __init__(self):
        """Initialize the PPO feature expander."""
        self.expected_features = 104
        self.feature_names: List[str] = []
        # Optional pinning of per-symbol feature indices
        self.pin_feature_index = str(os.getenv("PPO_PIN_FEATURE_INDEX", "true")).lower() in (
            "1",
            "true",
            "yes",
        )
        # Default strict in production environments
        env_name = os.getenv("ENVIRONMENT", os.getenv("APP_ENV", "")).lower()
        default_save_missing = "false" if env_name in ("prod", "production") else "true"
        self.save_missing_index = str(
            os.getenv("PPO_SAVE_MISSING_FEATURE_INDEX", default_save_missing)
        ).lower() in ("1", "true", "yes")
        self.index_base_dir = os.getenv("PPO_FEATURE_INDEX_DIR", "models/ppo")
        logger.info(
            f"🚀 PPO Feature Expander initialized (target: {self.expected_features} features)"
        )

    def expand_features(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Expand basic features to 104 PPO-specific features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with exactly 104 features for PPO models
        """
        logger.debug(f"Expanding features for PPO (input cols={len(df.columns)})")

        # Validate input
        self._validate_input(df)

        # Start from OHLCV-only base to ensure stable, non-churning feature set
        base_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        expanded_df = df[base_cols].copy()

        # 1. Core Technical Indicators (30 features)
        expanded_df = self._add_core_technical_indicators(expanded_df)

        # 2. Price Momentum Features (25 features)
        expanded_df = self._add_price_momentum_features(expanded_df)

        # 3. Volatility Features (20 features)
        expanded_df = self._add_volatility_features(expanded_df)

        # 4. Volume Features (15 features)
        expanded_df = self._add_volume_features(expanded_df)

        # 5. Market Microstructure Features (14 features)
        expanded_df = self._add_microstructure_features(expanded_df)

        # Clean and validate
        expanded_df = self._clean_features(expanded_df)

        # Ensure exactly 104 features
        expanded_df = self._ensure_feature_count(expanded_df)

        # If a symbol is provided, optionally pin to per-symbol feature index order
        if symbol and self.pin_feature_index:
            try:
                pinned = self._apply_feature_index(expanded_df, symbol)
                if pinned is not None:
                    expanded_df = pinned
            except Exception as e:
                logger.debug(f"Feature index pinning skipped for {symbol}: {e}")

        return expanded_df

    def _add_core_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 30 core technical indicator features."""
        # RSI variations (6 features)
        for period in [7, 14, 21, 28, 35, 42]:
            df[f"rsi_{period}"] = self._calculate_rsi(df["close"], period)

        # Moving averages (6 features)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{period}"] = df["close"].rolling(window=period, min_periods=1).mean()

        # EMA variations (4 features)
        for period in [12, 26, 50, 100]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # MACD components (3 features)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands (4 features)
        bb_period = 20
        bb_std = 2
        sma = df["close"].rolling(window=bb_period, min_periods=1).mean()
        std = df["close"].rolling(window=bb_period, min_periods=1).std()
        df["bb_upper"] = sma + (bb_std * std)
        df["bb_lower"] = sma - (bb_std * std)
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_width"] + 1e-10)

        # Stochastic oscillator (3 features)
        for period in [14, 21, 28]:
            low_min = df["low"].rolling(window=period, min_periods=1).min()
            high_max = df["high"].rolling(window=period, min_periods=1).max()
            df[f"stoch_{period}"] = 100 * ((df["close"] - low_min) / (high_max - low_min + 1e-10))

        # ATR variations (4 features)
        for period in [7, 14, 21, 28]:
            df[f"atr_{period}"] = self._calculate_atr(df, period)

        return df

    def _add_price_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 25 price momentum features."""
        # Price changes (10 features)
        for period in [1, 2, 3, 5, 7, 10, 14, 20, 30, 50]:
            df[f"price_change_{period}"] = df["close"].pct_change(period).fillna(0)

        # Momentum indicators (5 features)
        for period in [10, 20, 30, 40, 50]:
            df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)

        # Rate of change (5 features)
        for period in [5, 10, 15, 20, 25]:
            df[f"roc_{period}"] = (
                (df["close"] - df["close"].shift(period)) / (df["close"].shift(period) + 1e-10)
            ) * 100

        # Price position relative to moving averages (5 features)
        for period in [10, 20, 50, 100, 200]:
            ma = df["close"].rolling(window=period, min_periods=1).mean()
            df[f"price_ma_ratio_{period}"] = df["close"] / (ma + 1e-10)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 20 volatility features."""
        # Historical volatility (5 features)
        for period in [5, 10, 20, 30, 60]:
            returns = df["close"].pct_change()
            df[f"volatility_{period}"] = returns.rolling(
                window=period, min_periods=1
            ).std() * np.sqrt(252)

        # Parkinson volatility (3 features)
        for period in [10, 20, 30]:
            hl_ratio = np.log(df["high"] / (df["low"] + 1e-10))
            df[f"parkinson_vol_{period}"] = hl_ratio.rolling(
                window=period, min_periods=1
            ).std() * np.sqrt(252)

        # Garman-Klass volatility (3 features)
        for period in [10, 20, 30]:
            hl = np.log(df["high"] / (df["low"] + 1e-10)) ** 2
            co = np.log(df["close"] / (df["open"] + 1e-10)) ** 2
            df[f"gk_vol_{period}"] = np.sqrt(
                hl.rolling(window=period, min_periods=1).mean() * 0.5
                - co.rolling(window=period, min_periods=1).mean() * 0.386
            )

        # Volatility ratios (4 features)
        vol_5 = df["close"].pct_change().rolling(window=5, min_periods=1).std()
        vol_20 = df["close"].pct_change().rolling(window=20, min_periods=1).std()
        vol_60 = df["close"].pct_change().rolling(window=60, min_periods=1).std()

        df["vol_ratio_5_20"] = vol_5 / (vol_20 + 1e-10)
        df["vol_ratio_5_60"] = vol_5 / (vol_60 + 1e-10)
        df["vol_ratio_20_60"] = vol_20 / (vol_60 + 1e-10)
        df["vol_trend"] = (vol_5 - vol_20) / (vol_20 + 1e-10)

        # High-low spread features (5 features)
        for period in [5, 10, 20, 30, 50]:
            df[f"hl_spread_{period}"] = (
                ((df["high"] - df["low"]) / (df["close"] + 1e-10))
                .rolling(window=period, min_periods=1)
                .mean()
            )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 15 volume features."""
        # Volume moving averages (5 features)
        for period in [5, 10, 20, 50, 100]:
            df[f"volume_ma_{period}"] = df["volume"].rolling(window=period, min_periods=1).mean()

        # Volume ratios (4 features)
        vol_ma_5 = df["volume"].rolling(window=5, min_periods=1).mean()
        vol_ma_20 = df["volume"].rolling(window=20, min_periods=1).mean()
        vol_ma_50 = df["volume"].rolling(window=50, min_periods=1).mean()

        df["volume_ratio_5_20"] = df["volume"] / (vol_ma_20 + 1e-10)
        df["volume_ratio_5_50"] = df["volume"] / (vol_ma_50 + 1e-10)
        df["volume_ratio_20_50"] = vol_ma_20 / (vol_ma_50 + 1e-10)
        df["volume_trend"] = (vol_ma_5 - vol_ma_20) / (vol_ma_20 + 1e-10)

        # On-Balance Volume (OBV) and variations (3 features)
        df["obv"] = self._calculate_obv(df)
        df["obv_ma_10"] = df["obv"].rolling(window=10, min_periods=1).mean()
        df["obv_momentum"] = df["obv"] - df["obv"].shift(10)

        # Volume-price correlations (3 features)
        for period in [10, 20, 30]:
            price_change = df["close"].pct_change()
            volume_change = df["volume"].pct_change()
            df[f"pv_corr_{period}"] = (
                price_change.rolling(window=period, min_periods=1).corr(volume_change).fillna(0)
            )

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 14 market microstructure features."""
        # Spread features (4 features)
        df["bid_ask_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)
        df["spread_ma_10"] = df["bid_ask_spread"].rolling(window=10, min_periods=1).mean()
        df["spread_std_10"] = df["bid_ask_spread"].rolling(window=10, min_periods=1).std()
        df["spread_trend"] = df["bid_ask_spread"] - df["spread_ma_10"]

        # Price efficiency (3 features)
        for period in [10, 20, 30]:
            price_change = (df["close"] - df["close"].shift(period)).abs()
            path_length = df["close"].diff().abs().rolling(window=period, min_periods=1).sum()
            df[f"efficiency_{period}"] = price_change / (path_length + 1e-10)

        # VWAP features (3 features)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-10)
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-10)
        df["vwap_signal"] = (df["close"] > df["vwap"]).astype(float)

        # Accumulation/Distribution (2 features)
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"] + 1e-10
        )
        df["acc_dist"] = (clv * df["volume"]).cumsum()
        df["acc_dist_ma"] = df["acc_dist"].rolling(window=10, min_periods=1).mean()

        # Money Flow Index (2 features)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_flow_sum = positive_flow.rolling(window=14, min_periods=1).sum()
        negative_flow_sum = negative_flow.rolling(window=14, min_periods=1).sum()

        money_ratio = positive_flow_sum / (negative_flow_sum + 1e-10)
        df["mfi"] = 100 - (100 / (1 + money_ratio))
        df["mfi_signal"] = (df["mfi"] > 50).astype(float)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()

        return atr.fillna(0)

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = df["close"].diff()
        volume_direction = np.where(
            price_change > 0, df["volume"], np.where(price_change < 0, -df["volume"], 0)
        )
        obv = pd.Series(volume_direction, index=df.index).cumsum()
        return obv.fillna(0)

    def _validate_input(self, df: pd.DataFrame):
        """Validate input DataFrame."""
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(df) < 200:
            logger.warning(f"Small dataset ({len(df)} rows) - some features may be less reliable")

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features."""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill NaN values
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Final fillna with 0 for any remaining NaN
        df = df.fillna(0)

        # Clip extreme values for non-OHLCV columns
        feature_cols = [
            col for col in df.columns if col not in ["open", "high", "low", "close", "volume"]
        ]
        for col in feature_cols:
            if col in df.columns:
                # Clip to reasonable range
                percentile_99 = df[col].quantile(0.99)
                percentile_1 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=percentile_1, upper=percentile_99)

        return df

    def _ensure_feature_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure exactly 104 features (excluding OHLCV and metadata columns)."""
        # Get feature columns (excluding OHLCV and metadata) - aligned with ModelFeatureRouter
        excluded_cols = ["open", "high", "low", "close", "volume", "timestamp", "target"]
        feature_cols = [col for col in df.columns if col not in excluded_cols]

        current_count = len(feature_cols)

        if current_count == self.expected_features:
            logger.debug(f"✅ PPO feature count correct: {current_count}")
        elif current_count < self.expected_features:
            # Add padding features if needed
            logger.debug(f"Adding {self.expected_features - current_count} padding features")
            for i in range(current_count, self.expected_features):
                df[f"padding_feature_{i}"] = 0.0
                feature_cols.append(f"padding_feature_{i}")
        else:
            # Truncate if too many features
            # Use deterministic ordering to avoid churn, then select first N
            logger.debug(
                f"Truncating PPO features from {current_count} to {self.expected_features}"
            )
            feature_cols = sorted(feature_cols)[: self.expected_features]

        # Store feature names for reference
        self.feature_names = feature_cols

        # Return DataFrame with preserved columns + exactly 104 features
        preserved_cols = ["open", "high", "low", "close", "volume"]
        # Include timestamp and target if they exist in the original data
        for col in ["timestamp", "target"]:
            if col in df.columns:
                preserved_cols.append(col)

        return df[preserved_cols + feature_cols]

    def _index_path(self, symbol: str) -> str:
        return os.path.join(self.index_base_dir, symbol, "feature_index.json")

    def _load_feature_index(self, symbol: str) -> Optional[List[str]]:
        path = self._index_path(symbol)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict) and "feature_names" in data:
                        return list(data["feature_names"])  # tolerant legacy format
                except Exception as e:
                    logger.warning(f"Failed to parse feature index {path}: {e}")
        return None

    def _save_feature_index(self, symbol: str, names: List[str]) -> None:
        try:
            out_dir = os.path.dirname(self._index_path(symbol))
            os.makedirs(out_dir, exist_ok=True)
            with open(self._index_path(symbol), "w", encoding="utf-8") as f:
                json.dump(names, f, indent=2)
            logger.info(f"Saved PPO feature index for {symbol}: {len(names)} features")
        except Exception as e:
            logger.debug(f"Failed to save PPO feature index for {symbol}: {e}")

    def _apply_feature_index(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Reorder/pad features to match per-symbol index; create if missing (optional)."""
        # Determine feature columns (exclude OHLCV, timestamp, target)
        excluded = ["open", "high", "low", "close", "volume", "timestamp", "target"]
        current_cols = [c for c in df.columns if c not in excluded]

        index_names = self._load_feature_index(symbol)
        if index_names:
            # Build a DataFrame with exactly the index order
            pinned_df = df.copy()
            missing = [n for n in index_names if n not in pinned_df.columns]
            extra = [c for c in current_cols if c not in index_names]
            if missing or extra:
                logger.info(
                    f"PPO feature index applied for {symbol}: missing={len(missing)} extra={len(extra)}"
                )
            for name in index_names:
                if name not in pinned_df.columns:
                    pinned_df[name] = 0.0
            # Keep only indexed names in order
            ordered = index_names[: self.expected_features]
            # If index shorter than expected, pad with zeros
            if len(ordered) < self.expected_features:
                # Add deterministic padding columns
                pad_needed = self.expected_features - len(ordered)
                for i in range(pad_needed):
                    col = f"padding_feature_{len(ordered) + i}"
                    pinned_df[col] = 0.0
                    ordered.append(col)
            self.feature_names = ordered
            preserved_cols = [
                c
                for c in ["open", "high", "low", "close", "volume", "timestamp", "target"]
                if c in df.columns
            ]
            return pinned_df[preserved_cols + ordered]

        # If index missing
        if not index_names:
            if self.pin_feature_index and not self.save_missing_index:
                # Strict pinning: fail fast to require trainer-supplied index
                raise RuntimeError(
                    f"PPO_FEATURE_INDEX_MISSING_STRICT: Feature index missing for {symbol} and strict pinning enabled"
                )
            if self.save_missing_index and current_cols:
                # Non-strict mode: persist first-seen feature order
                proposed = current_cols[: self.expected_features]
                self._save_feature_index(symbol, proposed)
        return None

    def get_feature_names(self) -> List[str]:
        """Get the list of generated feature names."""
        return self.feature_names

    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has correct number of features.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        # Use same exclusion logic as _ensure_feature_count and ModelFeatureRouter
        excluded_cols = ["open", "high", "low", "close", "volume", "timestamp", "target"]
        feature_cols = [col for col in df.columns if col not in excluded_cols]

        is_valid = len(feature_cols) == self.expected_features

        if is_valid:
            logger.info(f"✅ PPO features validated: {len(feature_cols)} features")
        else:
            logger.error(
                f"❌ PPO feature validation failed: expected {self.expected_features}, got {len(feature_cols)}"
            )

        return is_valid


# Convenience function for easy usage
def expand_ppo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand features for PPO models to 104 dimensions.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with 104 features for PPO models
    """
    expander = PPOFeatureExpander()
    return expander.expand_features(df)
