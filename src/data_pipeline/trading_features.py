"""
Trading-Specific Feature Engineering Module
==========================================

Advanced feature engineering specifically designed for cryptocurrency trading.
Focuses on multi-timeframe analysis, market microstructure, and trading profitability.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ta  # Technical Analysis library
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class TradingFeatureEngine:
    """
    Advanced feature engineering class optimized for cryptocurrency trading.

    Focuses on:
    - Multi-timeframe momentum and volatility
    - Market microstructure indicators
    - Risk-adjusted returns
    - Regime detection
    - Support/resistance levels
    - Volume-price relationships
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TradingFeatureEngine.

        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config or self._get_default_config()
        logger.info("TradingFeatureEngine initialized for cryptocurrency trading")

    def _get_default_config(self) -> Dict:
        """Get trading-optimized feature configuration."""
        return {
            # Multi-timeframe periods (in number of candles)
            "momentum_periods": [5, 10, 20, 50, 100, 200],
            "volatility_periods": [5, 10, 20, 50, 100],
            "volume_periods": [5, 10, 20, 50],
            "price_periods": [5, 10, 20, 50, 100],
            # Technical indicators
            "rsi_periods": [7, 14, 21],
            "macd_configs": [
                {"fast": 12, "slow": 26, "signal": 9},
                {"fast": 5, "slow": 15, "signal": 7},  # Faster for crypto
            ],
            "bollinger_periods": [20, 50],
            "bollinger_std": 2.0,
            "atr_periods": [14, 21],
            # Market microstructure
            "support_resistance_window": 50,
            "volume_profile_bins": 20,
            "order_flow_window": 10,
            # Risk metrics
            "sharpe_windows": [20, 50, 100],
            "max_drawdown_windows": [20, 50, 100],
            "var_confidence": 0.05,
            "var_windows": [20, 50],
            # Market regime
            "regime_detection_window": 100,
            "trend_strength_window": 50,
            "volatility_regime_window": 50,
            # Feature selection
            "min_periods_ratio": 0.7,  # Minimum data availability
            "remove_outliers": True,
            "outlier_std_threshold": 5.0,
        }

    def generate_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive trading features.

        Args:
            df: OHLCV DataFrame with DatetimeIndex

        Returns:
            DataFrame with trading features
        """
        logger.info("🔧 Generating trading-specific features")

        # Validate input data
        self._validate_input_data(df)

        # Start with original data
        features_df = df.copy()

        # Core price/volume features
        features_df = self._add_price_features(features_df)
        features_df = self._add_volume_features(features_df)

        # Multi-timeframe momentum
        features_df = self._add_momentum_features(features_df)

        # Volatility and risk features
        features_df = self._add_volatility_features(features_df)
        features_df = self._add_risk_features(features_df)

        # Technical indicators
        features_df = self._add_technical_indicators(features_df)

        # Market microstructure
        features_df = self._add_microstructure_features(features_df)

        # Market regime detection
        features_df = self._add_regime_features(features_df)

        # Advanced features
        features_df = self._add_advanced_features(features_df)

        # Clean and finalize
        features_df = self._clean_features(features_df)

        logger.info(f"✅ Generated {len(features_df.columns)} trading features")
        return features_df

    def _validate_input_data(self, df: pd.DataFrame):
        """Validate input DataFrame."""
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex - some time-based features may not work")

        if len(df) < 200:
            logger.warning(f"Small dataset ({len(df)} rows) - some features may be unreliable")

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        logger.debug("Adding price features")

        # Basic price relationships
        df["hl_ratio"] = df["high"] / df["low"]
        df["oc_ratio"] = df["open"] / df["close"]
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

        # Price ranges
        df["true_range"] = np.maximum.reduce(
            [
                df["high"] - df["low"],
                np.abs(df["high"] - df["close"].shift(1)),
                np.abs(df["low"] - df["close"].shift(1)),
            ]
        )

        # Typical price and weighted close
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["weighted_close"] = (df["high"] + df["low"] + 2 * df["close"]) / 4

        # Gaps
        df["gap_up"] = (df["open"] > df["close"].shift(1)).astype(int)
        df["gap_down"] = (df["open"] < df["close"].shift(1)).astype(int)
        df["gap_size"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Multi-period price features
        try:
            periods = self.config.get("price_periods", [5, 10, 20, 50, 100])
            logger.debug(f"Using price_periods: {periods}")
        except Exception as e:
            logger.warning(f"Error accessing price_periods config: {e}")
            periods = [5, 10, 20, 50, 100]  # fallback
            
        for period in periods:
            # Price momentum
            df[f"price_momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

            # High/low momentum
            df[f"high_momentum_{period}"] = df["high"] / df["high"].shift(period) - 1
            df[f"low_momentum_{period}"] = df["low"] / df["low"].shift(period) - 1

            # Price acceleration
            if period >= 10:
                momentum = df["close"].pct_change(period)
                df[f"price_acceleration_{period}"] = momentum - momentum.shift(period)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        logger.debug("Adding volume features")

        # Basic volume features
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # Volume-price relationships
        df["volume_price_trend"] = df["volume"] * df["close"].pct_change()
        df["price_volume_ratio"] = df["close"].pct_change() / (df["volume"].pct_change() + 1e-8)

        # On-Balance Volume
        df["obv_direction"] = np.where(df["close"] > df["close"].shift(1), 1, -1)
        df["obv"] = (df["volume"] * df["obv_direction"]).cumsum()

        # Volume Rate of Change
        volume_periods = self.config.get("volume_periods", [5, 10, 20, 50])
        for period in volume_periods:
            df[f"volume_roc_{period}"] = df["volume"].pct_change(period)
            df[f"volume_sma_{period}"] = df["volume"].rolling(period).mean()
            df[f"volume_std_{period}"] = df["volume"].rolling(period).std()
            df[f"volume_zscore_{period}"] = (df["volume"] - df[f"volume_sma_{period}"]) / df[
                f"volume_std_{period}"
            ]

        # Volume Weight Average Price (VWAP)
        df["vwap_20"] = (df["typical_price"] * df["volume"]).rolling(20).sum() / df[
            "volume"
        ].rolling(20).sum()
        df["vwap_distance"] = (df["close"] - df["vwap_20"]) / df["vwap_20"]

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe momentum features."""
        logger.debug("Adding momentum features")

        momentum_periods = self.config.get("momentum_periods", [5, 10, 20, 50, 100, 200])
        for period in momentum_periods:
            # Standard momentum
            df[f"momentum_{period}"] = df["close"].pct_change(period)

            # Smoothed momentum (reduce noise)
            df[f"momentum_smooth_{period}"] = df["close"].pct_change(period).rolling(5).mean()

            # Momentum strength
            df[f"momentum_strength_{period}"] = np.abs(df["close"].pct_change(period))

            # Momentum persistence
            momentum = df["close"].pct_change()
            df[f"momentum_persistence_{period}"] = momentum.rolling(period).apply(
                lambda x: (x > 0).sum() / len(x), raw=True
            )

            # Relative strength vs different timeframes
            if period >= 20:
                short_momentum = df["close"].pct_change(period // 4)
                long_momentum = df["close"].pct_change(period)
                df[f"momentum_ratio_{period}"] = short_momentum / (long_momentum + 1e-8)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility and risk features."""
        logger.debug("Adding volatility features")

        # Returns for volatility calculation
        returns = df["close"].pct_change()

        for period in self.config["volatility_periods"]:
            # Standard volatility metrics
            df[f"volatility_{period}"] = returns.rolling(period).std()
            df[f"volatility_ann_{period}"] = df[f"volatility_{period}"] * np.sqrt(
                24 * 365
            )  # Annualized for crypto

            # Parkinson volatility (uses high-low)
            parkinson = np.log(df["high"] / df["low"]) ** 2
            df[f"parkinson_vol_{period}"] = parkinson.rolling(period).mean() * np.sqrt(24 * 365)

            # Garman-Klass volatility
            gk = (
                0.5 * np.log(df["high"] / df["low"]) ** 2
                - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
            )
            df[f"gk_volatility_{period}"] = gk.rolling(period).mean() * np.sqrt(24 * 365)

            # Volatility of volatility
            if period >= 10:
                vol = returns.rolling(period // 2).std()
                df[f"vol_of_vol_{period}"] = vol.rolling(period // 2).std()

            # Volatility regime (high/medium/low)
            volatility_series = df[f"volatility_{period}"]
            vol_33 = volatility_series.rolling(period * 2).quantile(0.33)
            vol_67 = volatility_series.rolling(period * 2).quantile(0.67)
            vol_33_median = vol_33.median()
            vol_67_median = vol_67.median()
            
            # Handle case where quantiles are identical (constant volatility)
            if pd.isna(vol_33_median) or pd.isna(vol_67_median) or vol_33_median == vol_67_median:
                # Use simple binary regime based on median
                vol_median = volatility_series.median()
                df[f"vol_regime_{period}"] = (volatility_series > vol_median).astype(float)
            else:
                df[f"vol_regime_{period}"] = pd.cut(
                    volatility_series,
                    bins=[
                        -np.inf,
                        vol_33_median,
                        vol_67_median,
                        np.inf,
                    ],
                    labels=[0, 1, 2],  # Low, Medium, High
                    duplicates='drop'
                ).astype(float)

        return df

    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted performance features."""
        logger.debug("Adding risk features")

        returns = df["close"].pct_change()

        for window in self.config["sharpe_windows"]:
            # Sharpe ratio (assuming risk-free rate = 0 for crypto)
            mean_return = returns.rolling(window).mean()
            vol = returns.rolling(window).std()
            df[f"sharpe_{window}"] = mean_return / (vol + 1e-8) * np.sqrt(24 * 365)

            # Sortino ratio (downside deviation)
            downside_returns = returns.where(returns < 0, 0)
            downside_vol = downside_returns.rolling(window).std()
            df[f"sortino_{window}"] = mean_return / (downside_vol + 1e-8) * np.sqrt(24 * 365)

            # Calmar ratio (return/max drawdown)
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.rolling(window).max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.rolling(window).min()
            df[f"calmar_{window}"] = (mean_return * 24 * 365) / (abs(max_drawdown) + 1e-8)

        # Value at Risk (VaR)
        for window in self.config["var_windows"]:
            df[f"var_{window}"] = returns.rolling(window).quantile(self.config["var_confidence"])
            df[f"cvar_{window}"] = returns[returns <= df[f"var_{window}"]].rolling(window).mean()

        # Maximum Drawdown
        for window in self.config["max_drawdown_windows"]:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.rolling(window).max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            df[f"max_drawdown_{window}"] = drawdown.rolling(window).min()

            # Drawdown duration
            is_drawdown = drawdown < 0
            df[f"drawdown_duration_{window}"] = is_drawdown.rolling(window).sum()

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        logger.debug("Adding technical indicators")

        # RSI for multiple periods
        for period in self.config["rsi_periods"]:
            df[f"rsi_{period}"] = ta.momentum.RSIIndicator(df["close"], window=period).rsi()

            # RSI divergence
            rsi = df[f"rsi_{period}"]
            price_peaks, _ = find_peaks(df["close"], distance=10)
            rsi_peaks, _ = find_peaks(rsi, distance=10)
            # Simplified divergence - could be enhanced
            df[f"rsi_divergence_{period}"] = 0  # Placeholder for more complex divergence logic

        # MACD for multiple configurations
        for i, macd_config in enumerate(self.config["macd_configs"]):
            macd = ta.trend.MACD(
                df["close"],
                window_fast=macd_config["fast"],
                window_slow=macd_config["slow"],
                window_sign=macd_config["signal"],
            )
            df[f"macd_{i}"] = macd.macd()
            df[f"macd_signal_{i}"] = macd.macd_signal()
            df[f"macd_histogram_{i}"] = macd.macd_diff()

        # Bollinger Bands
        for period in self.config["bollinger_periods"]:
            bb = ta.volatility.BollingerBands(
                df["close"], window=period, window_dev=self.config["bollinger_std"]
            )
            df[f"bb_upper_{period}"] = bb.bollinger_hband()
            df[f"bb_lower_{period}"] = bb.bollinger_lband()
            df[f"bb_width_{period}"] = (df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]) / df[
                "close"
            ]
            df[f"bb_position_{period}"] = (df["close"] - df[f"bb_lower_{period}"]) / (
                df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]
            )

        # ATR (Average True Range)
        for period in self.config["atr_periods"]:
            df[f"atr_{period}"] = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=period
            ).average_true_range()
            df[f"atr_ratio_{period}"] = df[f"atr_{period}"] / df["close"]

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        logger.debug("Adding microstructure features")

        # Bid-ask spread proxy (high-low as approximation)
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
        df["spread_ma"] = df["spread_proxy"].rolling(20).mean()
        df["spread_relative"] = df["spread_proxy"] / df["spread_ma"]

        # Price impact (relationship between volume and price movement)
        price_change = df["close"].pct_change()
        volume_change = df["volume"].pct_change()
        df["price_impact"] = price_change / (volume_change + 1e-8)

        # Kyle's Lambda (price impact measure)
        window = self.config["order_flow_window"]
        df["kyle_lambda"] = (
            price_change.abs().rolling(window).mean() / df["volume"].rolling(window).mean()
        )

        # Support and Resistance levels
        window = self.config["support_resistance_window"]
        df["resistance_level"] = df["high"].rolling(window).max()
        df["support_level"] = df["low"].rolling(window).min()
        df["support_distance"] = (df["close"] - df["support_level"]) / df["close"]
        df["resistance_distance"] = (df["resistance_level"] - df["close"]) / df["close"]

        # Volume profile approximation
        bins = self.config["volume_profile_bins"]
        price_range = df["high"].max() - df["low"].min()
        bin_size = price_range / bins

        # Simplified volume profile
        df["volume_profile_level"] = ((df["close"] - df["low"].min()) // bin_size).astype(int)

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        logger.debug("Adding regime features")

        window = self.config["regime_detection_window"]
        returns = df["close"].pct_change()

        # Trend strength using ADX logic
        trend_window = self.config["trend_strength_window"]
        plus_dm = np.where(
            (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
            np.maximum(df["high"] - df["high"].shift(1), 0),
            0,
        )
        minus_dm = np.where(
            (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
            np.maximum(df["low"].shift(1) - df["low"], 0),
            0,
        )

        tr = df["true_range"]
        plus_di = 100 * (plus_dm / tr).rolling(trend_window).mean()
        minus_di = 100 * (minus_dm / tr).rolling(trend_window).mean()
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df["trend_strength"] = dx.rolling(trend_window).mean()

        # Market regime classification
        # Bull/Bear based on price trend
        sma_short = df["close"].rolling(20).mean()
        sma_long = df["close"].rolling(50).mean()
        df["bull_bear_regime"] = np.where(sma_short > sma_long, 1, 0)  # 1=Bull, 0=Bear

        # Volatility regime
        vol_window = self.config["volatility_regime_window"]
        volatility = returns.rolling(vol_window).std()
        vol_median = volatility.rolling(window).median()
        df["volatility_regime"] = np.where(volatility > vol_median, 1, 0)  # 1=High Vol, 0=Low Vol

        # Mean reversion vs momentum regime
        # Based on Hurst exponent approximation
        def hurst_estimate(ts, max_lag=20):
            """Simplified Hurst exponent estimate."""
            if len(ts) < max_lag * 2:
                return 0.5

            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        df["hurst_exponent"] = returns.rolling(window).apply(hurst_estimate, raw=True)
        df["mean_reversion_regime"] = np.where(df["hurst_exponent"] < 0.5, 1, 0)  # 1=Mean reverting

        return df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced trading features."""
        logger.debug("Adding advanced features")

        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            df["month"] = df.index.month
            df["quarter"] = df.index.quarter

            # Cyclical encoding
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Fractal features
        returns = df["close"].pct_change()

        # Skewness and Kurtosis
        for window in [20, 50, 100]:
            df[f"returns_skew_{window}"] = returns.rolling(window).skew()
            df[f"returns_kurtosis_{window}"] = returns.rolling(window).kurt()

        # Auto-correlation features
        for lag in [1, 5, 10]:
            df[f"returns_autocorr_lag{lag}"] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else 0, raw=False
            )

        # Efficiency ratio (Kaufman)
        for period in [10, 20, 50]:
            direction = abs(df["close"] - df["close"].shift(period))
            volatility = np.abs(df["close"].diff()).rolling(period).sum()
            df[f"efficiency_ratio_{period}"] = direction / (volatility + 1e-8)

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features."""
        logger.debug("Cleaning features")

        # Remove outliers if configured
        if self.config["remove_outliers"]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ["open", "high", "low", "close", "volume"]:
                    continue  # Don't modify original OHLCV data

                mean = df[col].mean()
                std = df[col].std()
                threshold = self.config["outlier_std_threshold"]

                outlier_mask = np.abs(df[col] - mean) > threshold * std
                df.loc[outlier_mask, col] = np.nan

        # Forward fill NaN values (conservative approach)
        df = df.fillna(method="ffill")

        # Remove columns with too many NaN values
        min_periods = int(len(df) * self.config["min_periods_ratio"])
        df = df.dropna(thresh=min_periods, axis=1)

        # Final NaN cleanup
        df = df.fillna(0)

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        return df


def generate_trading_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to generate trading features.

    Args:
        df: OHLCV DataFrame
        config: Feature configuration

    Returns:
        DataFrame with trading features
    """
    engine = TradingFeatureEngine(config)
    return engine.generate_trading_features(df)


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

    # Generate features
    engine = TradingFeatureEngine()
    features = engine.generate_trading_features(sample_data)

    print(f"Generated {len(features.columns)} features")
    print(
        f"Feature categories: {[col for col in features.columns if not col in ['open', 'high', 'low', 'close', 'volume']][:10]}"
    )
