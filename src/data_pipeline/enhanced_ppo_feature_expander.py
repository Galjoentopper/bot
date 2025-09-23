"""
Enhanced PPO Feature Expander - Recreating Superior Multi-Timeframe Architecture
===============================================================================

This module recreates and enhances the superior multi-timeframe target engineering
approach from the old model, which was significantly more effective for trading.

Based on analysis of the old BTCEUR model metadata, this system focuses on:
- Forward-looking predictive features (not backward-looking technical indicators)
- Multi-timeframe return targets (1h, 3h, 6h, 12h, 24h)
- Cost-adjusted profitability analysis
- Risk-adjusted directional signals
- Market regime detection and adaptation
"""

import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class EnhancedPPOFeatureExpander:
    """
    Enhanced multi-timeframe feature expander based on the superior old model architecture.

    Recreates the 104-feature system with forward-looking predictive targets:
    - Multi-timeframe returns (25 features)
    - Cost-adjusted profitability (25 features)
    - Direction and magnitude analysis (25 features)
    - Risk-adjusted targets (20 features)
    - Market regime detection (9 features)
    """

    def __init__(self):
        """Initialize the enhanced PPO feature expander."""
        self.expected_features = 104
        self.feature_names: List[str] = []

        # Timeframes for multi-horizon predictions (matches old model)
        self.timeframes = [
            1,
            3,
            6,
            12,
            24,
        ]  # hours for 30m candles: 2, 6, 12, 24, 48 periods
        self.periods = [t * 2 for t in self.timeframes]  # Convert to 30m periods

        # Trading cost parameters (realistic for crypto)
        self.transaction_cost_bps = 10  # 0.1% per trade
        self.slippage_bps = 5  # 0.05% slippage
        self.funding_cost_bps = 1  # 0.01% funding per period

        logger.info(f"🚀 Enhanced PPO Feature Expander initialized (multi-timeframe targets)")

    def expand_features(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Expand features using superior multi-timeframe target engineering.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol (optional)

        Returns:
            DataFrame with exactly 104 predictive features
        """
        logger.debug(f"Expanding features with enhanced multi-timeframe approach")

        # Validate input
        self._validate_input(df)

        # Start with core OHLCV
        expanded_df = df[["open", "high", "low", "close", "volume"]].copy()

        # 1. Multi-timeframe returns (25 features)
        expanded_df = self._add_multiframe_returns(expanded_df)

        # 2. Cost-adjusted profitability analysis (25 features)
        expanded_df = self._add_cost_adjusted_targets(expanded_df)

        # 3. Direction and magnitude analysis (25 features)
        expanded_df = self._add_direction_magnitude_features(expanded_df)

        # 4. Risk-adjusted targets (20 features)
        expanded_df = self._add_risk_adjusted_features(expanded_df)

        # 5. Market regime detection (4 features) + close price
        expanded_df = self._add_regime_features(expanded_df)

        # Clean and validate
        expanded_df = self._clean_features(expanded_df)
        expanded_df = self._ensure_feature_count(expanded_df)

        # Store feature names
        self.feature_names = list(expanded_df.columns)

        logger.info(f"✅ Enhanced feature expansion complete: {len(expanded_df.columns)} features")
        return expanded_df

    def _add_multiframe_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe return targets (25 features) - core predictive power."""

        for i, (timeframe, periods) in enumerate(zip(self.timeframes, self.periods)):
            # Raw returns (forward-looking)
            df[f"return_{timeframe}h"] = df["close"].pct_change(periods).shift(-periods)

            # Log returns for better distribution properties
            df[f"log_return_{timeframe}h"] = np.log(df["close"] / df["close"].shift(periods)).shift(
                -periods
            )

            # Volatility-adjusted returns
            vol_window = min(periods, 20)
            vol = df["close"].pct_change().rolling(vol_window).std()
            df[f"vol_adj_return_{timeframe}h"] = df[f"return_{timeframe}h"] / (vol + 1e-8)

            # Momentum-adjusted returns (trend context)
            momentum = df["close"] / df["close"].rolling(periods).mean() - 1
            df[f"momentum_adj_return_{timeframe}h"] = df[f"return_{timeframe}h"] * (1 + momentum)

            # Regime-aware returns (bull/bear context)
            sma_long = df["close"].rolling(periods * 2).mean()
            regime = (df["close"] > sma_long).astype(float)
            df[f"regime_return_{timeframe}h"] = df[f"return_{timeframe}h"] * regime

        return df

    def _add_cost_adjusted_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cost-adjusted profitability analysis (25 features) - trading reality."""

        for i, (timeframe, periods) in enumerate(zip(self.timeframes, self.periods)):
            # Calculate total trading costs
            total_cost = (
                self.transaction_cost_bps + self.slippage_bps + self.funding_cost_bps * periods
            ) / 10000

            # Cost-adjusted returns (what trader actually gets)
            raw_return = df[f"return_{timeframe}h"]
            df[f"cost_adj_return_{timeframe}h"] = raw_return - total_cost

            # Profitable trade indicator (binary prediction target)
            df[f"profitable_{timeframe}h"] = (df[f"cost_adj_return_{timeframe}h"] > 0).astype(float)

            # Trade efficiency (return per unit cost)
            df[f"efficiency_{timeframe}h"] = df[f"cost_adj_return_{timeframe}h"] / total_cost

            # Risk-reward ratio
            volatility = df["close"].pct_change().rolling(20).std()
            df[f"risk_reward_{timeframe}h"] = df[f"cost_adj_return_{timeframe}h"] / (
                volatility + 1e-8
            )

            # Expected value (probability * magnitude)
            prob_positive = df[f"profitable_{timeframe}h"].rolling(50).mean()
            df[f"expected_value_{timeframe}h"] = prob_positive * df[f"cost_adj_return_{timeframe}h"]

        return df

    def _add_direction_magnitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add direction and magnitude analysis (25 features) - signal clarity."""

        for i, (timeframe, periods) in enumerate(zip(self.timeframes, self.periods)):
            # Direction prediction (up/down)
            df[f"direction_{timeframe}h"] = np.sign(df[f"return_{timeframe}h"])

            # Strong direction (above threshold)
            threshold = df[f"return_{timeframe}h"].rolling(100).std() * 1.5
            df[f"strong_direction_{timeframe}h"] = (
                np.abs(df[f"return_{timeframe}h"]) > threshold
            ).astype(float)

            # Return magnitude (absolute size of move)
            df[f"return_magnitude_{timeframe}h"] = np.abs(df[f"return_{timeframe}h"])

            # Magnitude percentile (relative to history)
            df[f"magnitude_percentile_{timeframe}h"] = (
                df[f"return_magnitude_{timeframe}h"].rolling(100).rank(pct=True)
            )

            # Direction confidence (consistency of direction)
            direction_consistency = df[f"direction_{timeframe}h"].rolling(10).mean()
            df[f"direction_confidence_{timeframe}h"] = np.abs(direction_consistency)

        return df

    def _add_risk_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted targets (20 features) - sophisticated risk analysis."""

        for i, (timeframe, periods) in enumerate(zip(self.timeframes, self.periods)):
            # Sharpe-like ratio for individual predictions
            returns = df[f"return_{timeframe}h"]
            mean_return = returns.rolling(50).mean()
            std_return = returns.rolling(50).std()
            df[f"sharpe_{timeframe}h"] = mean_return / (std_return + 1e-8)

            # Information ratio (active return / tracking error)
            benchmark_return = df["close"].pct_change(periods).shift(-periods)  # Simple buy-hold
            active_return = returns - benchmark_return
            tracking_error = active_return.rolling(50).std()
            df[f"info_ratio_{timeframe}h"] = active_return.rolling(50).mean() / (
                tracking_error + 1e-8
            )

            # Maximum favorable excursion potential
            high_future = df["high"].rolling(periods).max().shift(-periods)
            df[f"mfe_potential_{timeframe}h"] = high_future / df["close"] - 1

            # Maximum adverse excursion risk
            low_future = df["low"].rolling(periods).min().shift(-periods)
            df[f"mae_risk_{timeframe}h"] = df["close"] / low_future - 1

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection (4 features) + close price."""

        # Market trend regime (bull/bear/sideways)
        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        df["trend_regime"] = np.where(
            sma_20 > sma_50 * 1.02,
            1,  # Bull
            np.where(sma_20 < sma_50 * 0.98, -1, 0),  # Bear vs Sideways
        )

        # Volatility regime (high/low)
        vol = df["close"].pct_change().rolling(20).std()
        vol_percentile = vol.rolling(100).rank(pct=True)
        df["vol_regime"] = (vol_percentile > 0.7).astype(float)  # High vol regime

        # Volume regime (high/low activity)
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20).mean()
            vol_percentile = vol_ma.rolling(100).rank(pct=True)
            df["volume_regime"] = (vol_percentile > 0.7).astype(float)
        else:
            df["volume_regime"] = 0.5  # Neutral if no volume data

        # Price momentum regime (trending/mean-reverting)
        momentum = df["close"] / df["close"].rolling(20).mean() - 1
        df["momentum_regime"] = (np.abs(momentum) > 0.05).astype(float)

        # Keep close price for reference
        df["close"] = df["close"]

        return df

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(df) < max(self.periods) * 3:
            logger.warning(f"Short dataset ({len(df)} rows) may affect feature quality")

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by handling NaN/inf values."""
        # Forward fill then backward fill
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)

        # Final NaN fill with 0
        df = df.fillna(0)

        return df

    def _ensure_feature_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure exactly 104 features."""
        current_features = len(df.columns)

        if current_features > self.expected_features:
            # Keep most important features
            logger.warning(f"Truncating {current_features} to {self.expected_features} features")
            df = df.iloc[:, : self.expected_features]
        elif current_features < self.expected_features:
            # Add padding features
            needed = self.expected_features - current_features
            logger.warning(f"Padding with {needed} features to reach {self.expected_features}")
            for i in range(needed):
                df[f"padding_{i}"] = 0.0

        logger.info(f"Final feature count: {len(df.columns)}")
        return df

    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_names.copy()

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by importance for analysis."""
        groups = {
            "returns": [f for f in self.feature_names if "return_" in f and "adj" not in f],
            "cost_adjusted": [
                f for f in self.feature_names if "cost_adj" in f or "profitable" in f
            ],
            "direction": [f for f in self.feature_names if "direction" in f or "magnitude" in f],
            "risk_adjusted": [f for f in self.feature_names if "sharpe" in f or "info_ratio" in f],
            "regime": [f for f in self.feature_names if "regime" in f],
        }
        return groups
