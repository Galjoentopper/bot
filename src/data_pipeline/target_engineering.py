"""
Trading Target Engineering Module
=================================

Creates trading-optimized targets that account for transaction costs,
market regimes, and multi-objective optimization for profitability.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class TradingTargetEngine:
    """
    Creates trading-optimized targets for model training.

    Features:
    - Transaction cost adjustment
    - Multi-horizon targets
    - Risk-adjusted targets
    - Market regime-specific targets
    - Classification and regression targets
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TradingTargetEngine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        logger.info("TradingTargetEngine initialized")

    def _get_default_config(self) -> Dict:
        """Get default target configuration."""
        return {
            # Transaction costs (in basis points)
            "transaction_cost_bps": 10,  # 0.1% per trade (typical for crypto)
            "slippage_bps": 5,  # 0.05% slippage
            "funding_cost_bps": 1,  # 0.01% funding cost per period
            # Target horizons (in periods)
            "horizons": [1, 3, 6, 12, 24],  # 30min to 12h for 30min data
            # Risk adjustment
            "risk_free_rate": 0.0,  # Assume 0% risk-free rate for crypto
            "vol_adjustment": True,  # Risk-adjust returns by volatility
            "vol_window": 20,  # Window for volatility calculation
            # Classification thresholds (returns)
            "buy_threshold": 0.005,  # 0.5% minimum for buy signal
            "sell_threshold": -0.005,  # -0.5% minimum for sell signal
            "strong_signal_multiplier": 2.0,  # 2x threshold for strong signals
            # Market regime adjustment
            "regime_adjustment": True,
            "bull_multiplier": 1.2,  # Higher threshold in bull markets
            "bear_multiplier": 0.8,  # Lower threshold in bear markets
            # Multi-target configuration
            "include_magnitude": True,  # Include return magnitude targets
            "include_direction": True,  # Include direction classification
            "include_confidence": True,  # Include confidence targets
            "include_risk_adjusted": True,  # Include risk-adjusted targets
        }

    def create_trading_targets(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """
        Create comprehensive trading targets.

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price data

        Returns:
            DataFrame with trading targets
        """
        logger.info("🎯 Creating trading-optimized targets")

        # Validate input
        if price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame")

        targets_df = pd.DataFrame(index=df.index)

        # Basic returns for all horizons
        targets_df = self._add_basic_returns(df, targets_df, price_col)

        # Transaction cost adjusted returns
        targets_df = self._add_cost_adjusted_returns(df, targets_df, price_col)

        # Classification targets
        if self.config["include_direction"]:
            targets_df = self._add_direction_targets(df, targets_df, price_col)

        # Magnitude targets
        if self.config["include_magnitude"]:
            targets_df = self._add_magnitude_targets(df, targets_df, price_col)

        # Risk-adjusted targets
        if self.config["include_risk_adjusted"]:
            targets_df = self._add_risk_adjusted_targets(df, targets_df, price_col)

        # Confidence targets
        if self.config["include_confidence"]:
            targets_df = self._add_confidence_targets(df, targets_df, price_col)

        # Market regime specific targets
        if self.config["regime_adjustment"]:
            targets_df = self._add_regime_targets(df, targets_df, price_col)

        # Clean targets
        targets_df = self._clean_targets(targets_df)

        logger.info(f"✅ Created {len(targets_df.columns)} trading targets")
        return targets_df

    def _add_basic_returns(
        self, df: pd.DataFrame, targets_df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add basic forward-looking returns for all horizons."""
        logger.debug("Adding basic returns")

        for horizon in self.config["horizons"]:
            # Forward returns
            future_price = df[price_col].shift(-horizon)
            returns = (future_price / df[price_col]) - 1
            targets_df[f"return_{horizon}h"] = returns

            # Log returns (better for multiplicative effects)
            log_returns = np.log(future_price / df[price_col])
            targets_df[f"log_return_{horizon}h"] = log_returns

        return targets_df

    def _add_cost_adjusted_returns(
        self, df: pd.DataFrame, targets_df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add transaction cost adjusted returns."""
        logger.debug("Adding cost-adjusted returns")

        # Total transaction cost per trade
        total_cost_bps = self.config["transaction_cost_bps"] + self.config["slippage_bps"]
        total_cost_ratio = total_cost_bps / 10000

        # Funding cost per period
        funding_cost_ratio = self.config["funding_cost_bps"] / 10000

        for horizon in self.config["horizons"]:
            # Basic return
            raw_return = targets_df[f"return_{horizon}h"]

            # Subtract transaction costs (entry + exit)
            cost_adjusted = raw_return - (2 * total_cost_ratio)

            # Subtract funding costs for the holding period
            cost_adjusted = cost_adjusted - (horizon * funding_cost_ratio)

            targets_df[f"cost_adj_return_{horizon}h"] = cost_adjusted

            # Only profitable after costs
            targets_df[f"profitable_{horizon}h"] = (cost_adjusted > 0).astype(int)

        return targets_df

    def _add_direction_targets(
        self, df: pd.DataFrame, targets_df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add directional classification targets."""
        logger.debug("Adding direction targets")

        for horizon in self.config["horizons"]:
            cost_adj_return = targets_df[f"cost_adj_return_{horizon}h"]

            # Basic direction (accounting for costs)
            buy_threshold = self.config["buy_threshold"]
            sell_threshold = self.config["sell_threshold"]

            direction = np.where(
                cost_adj_return > buy_threshold,
                1,  # Buy
                np.where(cost_adj_return < sell_threshold, -1, 0),  # Sell or Hold
            )
            targets_df[f"direction_{horizon}h"] = direction

            # Strong signals (2x threshold)
            strong_buy_threshold = buy_threshold * self.config["strong_signal_multiplier"]
            strong_sell_threshold = sell_threshold * self.config["strong_signal_multiplier"]

            strong_direction = np.where(
                cost_adj_return > strong_buy_threshold,
                2,  # Strong Buy
                np.where(
                    cost_adj_return < strong_sell_threshold, -2, direction  # Strong Sell
                ),  # Use regular direction
            )
            targets_df[f"strong_direction_{horizon}h"] = strong_direction

            # Binary profitable classification
            targets_df[f"is_profitable_{horizon}h"] = (cost_adj_return > 0).astype(int)

        return targets_df

    def _add_magnitude_targets(
        self, df: pd.DataFrame, targets_df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add return magnitude targets."""
        logger.debug("Adding magnitude targets")

        for horizon in self.config["horizons"]:
            cost_adj_return = targets_df[f"cost_adj_return_{horizon}h"]

            # Absolute magnitude
            targets_df[f"return_magnitude_{horizon}h"] = np.abs(cost_adj_return)

            # Positive magnitude only (for profitable trades)
            positive_magnitude = np.where(cost_adj_return > 0, cost_adj_return, 0)
            targets_df[f"positive_magnitude_{horizon}h"] = positive_magnitude

            # Magnitude categories
            magnitude = np.abs(cost_adj_return)
            mag_categories = pd.cut(
                magnitude,
                bins=[0, 0.01, 0.02, 0.05, np.inf],
                labels=[0, 1, 2, 3],
                include_lowest=True,
            ).astype(float)
            targets_df[f"magnitude_category_{horizon}h"] = mag_categories

        return targets_df

    def _add_risk_adjusted_targets(
        self, df: pd.DataFrame, targets_df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add risk-adjusted targets."""
        logger.debug("Adding risk-adjusted targets")

        # Calculate rolling volatility
        returns = df[price_col].pct_change()
        volatility = returns.rolling(self.config["vol_window"]).std()

        for horizon in self.config["horizons"]:
            cost_adj_return = targets_df[f"cost_adj_return_{horizon}h"]

            # Risk-adjusted return (Sharpe-like)
            risk_adj_return = cost_adj_return / (volatility + 1e-8)
            targets_df[f"risk_adj_return_{horizon}h"] = risk_adj_return

            # Information ratio (vs market benchmark)
            market_return = targets_df[f"cost_adj_return_{horizon}h"].mean()
            excess_return = cost_adj_return - market_return
            targets_df[f"info_ratio_{horizon}h"] = excess_return / (volatility + 1e-8)

            # Risk-adjusted direction
            risk_adj_direction = np.where(
                risk_adj_return > 0.5,
                1,  # Buy if risk-adj return > 0.5
                np.where(risk_adj_return < -0.5, -1, 0),  # Sell if < -0.5
            )
            targets_df[f"risk_adj_direction_{horizon}h"] = risk_adj_direction

        return targets_df

    def _add_confidence_targets(
        self, df: pd.DataFrame, targets_df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add prediction confidence targets."""
        logger.debug("Adding confidence targets")

        for horizon in self.config["horizons"]:
            cost_adj_return = targets_df[f"cost_adj_return_{horizon}h"]

            # Confidence based on return magnitude and consistency
            magnitude = np.abs(cost_adj_return)

            # Historical consistency (how often similar returns occurred)
            rolling_window = min(50, len(df) // 4)
            consistency = cost_adj_return.rolling(rolling_window).apply(
                lambda x: (np.sign(x) == np.sign(x.iloc[-1])).mean(), raw=False
            )

            # Combined confidence score
            confidence = (magnitude * consistency).fillna(0)

            # Normalize confidence to [0, 1]
            if confidence.max() > 0:
                confidence = confidence / confidence.max()

            targets_df[f"confidence_{horizon}h"] = confidence

            # High confidence threshold
            high_conf_threshold = confidence.quantile(0.8)
            targets_df[f"high_confidence_{horizon}h"] = (confidence > high_conf_threshold).astype(
                int
            )

        return targets_df

    def _add_regime_targets(
        self, df: pd.DataFrame, targets_df: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """Add market regime-specific targets."""
        logger.debug("Adding regime-specific targets")

        # Detect market regime
        sma_short = df[price_col].rolling(20).mean()
        sma_long = df[price_col].rolling(50).mean()
        is_bull_market = sma_short > sma_long

        # Volatility regime
        returns = df[price_col].pct_change()
        vol_short = returns.rolling(10).std()
        vol_long = returns.rolling(30).std()
        is_high_vol = vol_short > vol_long

        for horizon in self.config["horizons"]:
            cost_adj_return = targets_df[f"cost_adj_return_{horizon}h"]

            # Regime-adjusted thresholds
            bull_threshold = self.config["buy_threshold"] * self.config["bull_multiplier"]
            bear_threshold = self.config["buy_threshold"] * self.config["bear_multiplier"]

            regime_buy_threshold = np.where(is_bull_market, bull_threshold, bear_threshold)
            regime_sell_threshold = -regime_buy_threshold

            # Regime-adjusted direction
            regime_direction = np.where(
                cost_adj_return > regime_buy_threshold,
                1,
                np.where(cost_adj_return < regime_sell_threshold, -1, 0),
            )
            targets_df[f"regime_direction_{horizon}h"] = regime_direction

            # Bull market specific targets
            bull_return = np.where(is_bull_market, cost_adj_return, np.nan)
            targets_df[f"bull_return_{horizon}h"] = bull_return

            # Bear market specific targets
            bear_return = np.where(~is_bull_market, cost_adj_return, np.nan)
            targets_df[f"bear_return_{horizon}h"] = bear_return

            # High volatility targets
            high_vol_return = np.where(is_high_vol, cost_adj_return, np.nan)
            targets_df[f"high_vol_return_{horizon}h"] = high_vol_return

            # Low volatility targets
            low_vol_return = np.where(~is_high_vol, cost_adj_return, np.nan)
            targets_df[f"low_vol_return_{horizon}h"] = low_vol_return

        return targets_df

    def _clean_targets(self, targets_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize targets."""
        logger.debug("Cleaning targets")

        # Remove rows where all targets are NaN (end of series)
        targets_df = targets_df.dropna(how="all")

        # Handle infinite values
        targets_df = targets_df.replace([np.inf, -np.inf], np.nan)

        # Forward fill NaN values for regime-specific targets
        regime_cols = [
            col
            for col in targets_df.columns
            if "regime" in col or "bull" in col or "bear" in col or "vol" in col
        ]
        targets_df[regime_cols] = targets_df[regime_cols].fillna(method="ffill")

        # Fill remaining NaN with 0 for classification targets
        classification_cols = [
            col
            for col in targets_df.columns
            if "direction" in col or "profitable" in col or "confidence" in col
        ]
        targets_df[classification_cols] = targets_df[classification_cols].fillna(0)

        return targets_df

    def get_target_for_model(
        self, targets_df: pd.DataFrame, model_type: str, horizon: int = 1
    ) -> Tuple[pd.Series, str]:
        """
        Get the most appropriate target for a specific model type.

        Args:
            targets_df: DataFrame with all targets
            model_type: Type of model ('gru', 'lgbm', 'ppo')
            horizon: Target horizon

        Returns:
            Tuple of (target_series, target_name)
        """
        target_mapping = {
            "gru": f"cost_adj_return_{horizon}h",
            "lgbm": f"direction_{horizon}h",
            "ppo": f"cost_adj_return_{horizon}h",
        }

        target_name = target_mapping.get(model_type.lower(), f"cost_adj_return_{horizon}h")

        if target_name not in targets_df.columns:
            # Fallback to basic return
            target_name = f"return_{horizon}h"

        if target_name not in targets_df.columns:
            raise ValueError(f"Target {target_name} not found for model {model_type}")

        return targets_df[target_name], target_name

    def get_evaluation_targets(self, targets_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get targets specifically for model evaluation.

        Args:
            targets_df: DataFrame with all targets

        Returns:
            Dictionary of evaluation targets
        """
        eval_targets = {}

        # Primary evaluation targets
        primary_horizon = self.config["horizons"][0]  # Shortest horizon

        eval_targets["returns"] = targets_df[f"cost_adj_return_{primary_horizon}h"]
        eval_targets["direction"] = targets_df[f"direction_{primary_horizon}h"]
        eval_targets["profitable"] = targets_df[f"profitable_{primary_horizon}h"]

        if f"confidence_{primary_horizon}h" in targets_df.columns:
            eval_targets["confidence"] = targets_df[f"confidence_{primary_horizon}h"]

        if f"risk_adj_return_{primary_horizon}h" in targets_df.columns:
            eval_targets["risk_adjusted"] = targets_df[f"risk_adj_return_{primary_horizon}h"]

        return eval_targets


def create_trading_targets(
    df: pd.DataFrame, price_col: str = "close", config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Convenience function to create trading targets.

    Args:
        df: OHLCV DataFrame
        price_col: Price column name
        config: Target configuration

    Returns:
        DataFrame with trading targets
    """
    engine = TradingTargetEngine(config)
    return engine.create_trading_targets(df, price_col)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=1000, freq="30T")
    sample_data = pd.DataFrame(
        {
            "open": np.random.uniform(100, 110, 1000),
            "high": np.random.uniform(105, 115, 1000),
            "low": np.random.uniform(95, 105, 1000),
            "close": np.random.uniform(100, 110, 1000),
            "volume": np.random.uniform(1000, 10000, 1000),
        },
        index=dates,
    )

    # Ensure OHLC consistency
    sample_data["high"] = np.maximum.reduce(
        [sample_data["open"], sample_data["high"], sample_data["close"]]
    )
    sample_data["low"] = np.minimum.reduce(
        [sample_data["open"], sample_data["low"], sample_data["close"]]
    )

    # Create targets
    engine = TradingTargetEngine()
    targets = engine.create_trading_targets(sample_data)

    print(f"Created {len(targets.columns)} targets")
    print(f"Target types: {list(targets.columns)[:10]}")

    # Get model-specific targets
    gru_target, gru_name = engine.get_target_for_model(targets, "gru", 1)
    print(f"GRU target: {gru_name}")

    eval_targets = engine.get_evaluation_targets(targets)
    print(f"Evaluation targets: {list(eval_targets.keys())}")
