"""
Enhanced Feature Engineering Engine
===================================

Professional-grade feature engineering system designed for cryptocurrency trading models.
Generates comprehensive feature sets tailored for different model types (PPO, GRU, LightGBM).

Features:
- Model-specific feature generation (PPO: 104, GRU/LightGBM: 100)
- Advanced technical indicators and market microstructure features
- Intelligent feature selection and routing
- Robust error handling and validation
- Configuration-driven architecture
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class EnhancedFeatureEngine:
    """
    Enhanced feature engineering engine for cryptocurrency trading models.

    This engine generates model-specific feature sets:
    - PPO: 104 features for reinforcement learning
    - GRU: 100 features for sequential modeling
    - LightGBM: 100 features for gradient boosting
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced feature engine.

        Args:
            config_path: Path to feature configuration YAML file
        """
        self.config_path = config_path or "config/feature_config.yaml"
        self.config = self._load_config()

        # Initialize feature groups and mappings
        self.feature_groups = self.config.get("feature_groups", {})
        self.model_configs = self.config.get("models", {})

        # Cache for generated features
        self._feature_cache = {}

        logger.info("🚀 Enhanced Feature Engine initialized")
        logger.info(f"Configuration loaded from: {self.config_path}")
        logger.info(f"Supported models: {list(self.model_configs.keys())}")

    def _load_config(self) -> Dict:
        """Load feature configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._get_default_config()

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            logger.info(f"✅ Feature configuration loaded successfully")
            return config

        except Exception as e:
            logger.error(f"❌ Error loading config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration if file loading fails."""
        return {
            "models": {
                "ppo": {"expected_feature_count": 104},
                "gru": {"expected_feature_count": 100},
                "lightgbm": {"expected_feature_count": 100},
            },
            "feature_engineering": {
                "enable_advanced_features": True,
                "remove_outliers": True,
                "outlier_std_threshold": 5.0,
            },
        }

    def generate_features_for_model(
        self, df: pd.DataFrame, model_type: str, symbol: str = "GENERIC"
    ) -> pd.DataFrame:
        """
        Generate features tailored for specific model type.

        Args:
            df: Input DataFrame with OHLCV data
            model_type: Model type ('ppo', 'gru', 'lightgbm')
            symbol: Trading symbol for model-specific optimizations

        Returns:
            DataFrame with model-specific features
        """
        logger.info(f"🔧 Generating features for {model_type.upper()} model ({symbol})")

        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df

        # Validate input data
        self._validate_input_data(df)

        # Generate comprehensive feature set
        features_df = self._generate_comprehensive_features(df)

        # Apply model-specific feature selection
        model_features_df = self._select_features_for_model(features_df, model_type, symbol)

        # Apply final validation and cleaning
        final_df = self._apply_final_validation(model_features_df, model_type)

        feature_count = len(
            [
                col
                for col in final_df.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
        )
        expected_count = self.model_configs.get(model_type, {}).get("expected_feature_count", 100)

        logger.info(
            f"✅ Generated {feature_count} features for {model_type.upper()} (expected: {expected_count})"
        )

        return final_df

    def _generate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set covering all categories."""
        logger.debug("Generating comprehensive feature set")

        # Start with validated source data
        features_df = self._validate_and_clean_source_data(df.copy())

        # Core technical indicators (15 features)
        features_df = self._add_core_technical_features(features_df)

        # Price momentum features (20 features)
        features_df = self._add_price_momentum_features(features_df)

        # Volatility and risk features (15 features)
        features_df = self._add_volatility_risk_features(features_df)

        # Volume analysis features (12 features)
        features_df = self._add_volume_analysis_features(features_df)

        # Market microstructure features (10 features)
        features_df = self._add_microstructure_features(features_df)

        # Regime detection features (8 features)
        features_df = self._add_regime_detection_features(features_df)

        # Time-based features (6 features)
        features_df = self._add_time_features(features_df)

        # Advanced technical features (12 features)
        features_df = self._add_advanced_technical_features(features_df)

        # Statistical features (6 features)
        features_df = self._add_statistical_features(features_df)

        # Apply intermediate cleaning
        features_df = self._clean_intermediate_features(features_df)

        logger.debug(f"Generated {len(features_df.columns)} total features")
        return features_df

    def _add_core_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add core technical indicators (15 features)."""
        # RSI for multiple periods
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = self._calculate_rsi(df["close"], period)

        # MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_histogram"] = macd_histogram

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df["close"])
        df["bb_upper_20"] = bb_upper
        df["bb_lower_20"] = bb_lower
        df["bb_width_20"] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
        df["bb_position_20"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # ATR
        df["atr_14"] = self._calculate_atr(df, 14)
        df["atr_ratio_14"] = df["atr_14"] / (df["close"] + 1e-10)

        # CCI and ADX
        df["cci_20"] = self._calculate_cci(df, 20)
        df["adx_14"] = self._calculate_adx(df, 14)

        # Stochastic
        stoch_k, _ = self._calculate_stochastic(df, 14, 3)
        df["stoch_k_14"] = stoch_k

        return df

    def _add_price_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price momentum features (20 features)."""
        # Basic momentum for multiple periods
        for period in [5, 10, 20, 50]:
            df[f"momentum_{period}"] = df["close"].pct_change(period)
            df[f"price_momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

        # Price acceleration
        for period in [10, 20]:
            momentum = df["close"].pct_change(period)
            df[f"price_acceleration_{period}"] = momentum - momentum.shift(period)

        # Rate of Change
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (
                (df["close"] - df["close"].shift(period)) / (df["close"].shift(period) + 1e-10)
            ) * 100

        # Williams %R
        for period in [14, 21]:
            df[f"williams_r_{period}"] = self._calculate_williams_r(df, period)

        # Efficiency Ratio (Kaufman)
        for period in [10, 20, 50]:
            direction = abs(df["close"] - df["close"].shift(period))
            volatility = np.abs(df["close"].diff()).rolling(period).sum()
            df[f"efficiency_ratio_{period}"] = direction / (volatility + 1e-8)

        # Moving average ratios
        sma_5 = df["close"].rolling(5).mean()
        ema_5 = df["close"].ewm(span=5).mean()
        df["sma_5_ratio"] = df["close"] / (sma_5 + 1e-10)
        df["ema_5_ratio"] = df["close"] / (ema_5 + 1e-10)

        return df

    def _add_volatility_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility and risk features (15 features)."""
        returns = df["close"].pct_change()

        # Standard volatility
        for period in [10, 20, 50]:
            df[f"volatility_{period}"] = returns.rolling(period).std()

        # Parkinson volatility
        parkinson = np.log(df["high"] / (df["low"] + 1e-10)) ** 2
        df["parkinson_vol_20"] = parkinson.rolling(20).mean() * np.sqrt(365 * 24)

        # Garman-Klass volatility
        gk = (
            0.5 * np.log(df["high"] / (df["low"] + 1e-10)) ** 2
            - (2 * np.log(2) - 1) * np.log(df["close"] / (df["open"] + 1e-10)) ** 2
        )
        df["gk_volatility_20"] = gk.rolling(20).mean() * np.sqrt(365 * 24)

        # Volatility of volatility
        vol = returns.rolling(10).std()
        df["vol_of_vol_20"] = vol.rolling(10).std()

        # Volatility regime
        volatility_series = df["volatility_20"]
        vol_median = volatility_series.rolling(40).median()
        df["vol_regime_20"] = (volatility_series > vol_median).astype(float)

        # Risk metrics
        for window in [20, 50]:
            mean_return = returns.rolling(window).mean()
            vol = returns.rolling(window).std()
            df[f"sharpe_{window}"] = mean_return / (vol + 1e-8) * np.sqrt(365 * 24)

            # Sortino ratio
            downside_returns = returns.where(returns < 0, 0)
            downside_vol = downside_returns.rolling(window).std()
            df[f"sortino_{window}"] = mean_return / (downside_vol + 1e-8) * np.sqrt(365 * 24)

        # Calmar ratio
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(20).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(20).min()
        df["calmar_20"] = (returns.rolling(20).mean() * 365 * 24) / (abs(max_drawdown) + 1e-8)

        # Value at Risk
        df["var_20"] = returns.rolling(20).quantile(0.05)

        # Maximum Drawdown
        for window in [20, 50]:
            df[f"max_drawdown_{window}"] = drawdown.rolling(window).min()

        # Volatility ratios
        df["vol_ratio_10_20"] = df["volatility_10"] / (df["volatility_20"] + 1e-8)

        return df

    def _add_volume_analysis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis features (12 features)."""
        # Basic volume features
        volume_sma_20 = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (volume_sma_20 + 1e-10)

        # Volume Rate of Change
        for period in [5, 10]:
            df[f"volume_roc_{period}"] = df["volume"].pct_change(period)

        # Volume Z-score
        volume_std_20 = df["volume"].rolling(20).std()
        df["volume_zscore_20"] = (df["volume"] - volume_sma_20) / (volume_std_20 + 1e-10)

        # On-Balance Volume
        df["obv"] = self._calculate_obv(df)

        # VWAP features
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap_20"] = (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(
            20
        ).sum()
        df["vwap_distance"] = (df["close"] - df["vwap_20"]) / (df["vwap_20"] + 1e-10)

        # Advanced volume features
        df["vwap_deviation"] = self._calculate_vwap_deviation(df)
        df["accumulation_distribution"] = self._calculate_accumulation_distribution(df)

        # Volume-price relationships
        df["volume_price_trend"] = df["volume"] * df["close"].pct_change()
        df["price_volume_ratio"] = df["close"].pct_change() / (df["volume"].pct_change() + 1e-8)

        # High volume indicator
        df["high_volume"] = (df["volume_ratio"] > 1.5).astype(int)

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features (10 features)."""
        # Spread proxy
        df["spread_proxy"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)
        spread_ma = df["spread_proxy"].rolling(20).mean()
        df["spread_relative"] = df["spread_proxy"] / (spread_ma + 1e-10)

        # Price impact
        price_change = df["close"].pct_change()
        volume_change = df["volume"].pct_change()
        df["price_impact"] = price_change / (volume_change + 1e-8)

        # Kyle's Lambda
        df["kyle_lambda"] = price_change.abs().rolling(10).mean() / df["volume"].rolling(10).mean()

        # Support and resistance
        window = 50
        support_level = df["low"].rolling(window).min()
        resistance_level = df["high"].rolling(window).max()
        df["support_distance"] = (df["close"] - support_level) / (df["close"] + 1e-10)
        df["resistance_distance"] = (resistance_level - df["close"]) / (df["close"] + 1e-10)

        # Price position
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        # Gap analysis
        df["gap_size"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-10)

        # True range
        df["true_range"] = np.maximum.reduce(
            [
                df["high"] - df["low"],
                np.abs(df["high"] - df["close"].shift(1)),
                np.abs(df["low"] - df["close"].shift(1)),
            ]
        )

        # High-low ratio
        df["hl_ratio"] = df["high"] / (df["low"] + 1e-10)

        return df

    def _add_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features (8 features)."""
        returns = df["close"].pct_change()

        # Trend strength (simplified ADX)
        window = 50
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
        plus_di = 100 * (plus_dm / (tr + 1e-10)).rolling(window).mean()
        minus_di = 100 * (minus_dm / (tr + 1e-10)).rolling(window).mean()
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df["trend_strength"] = dx.rolling(window).mean()

        # Bull/Bear regime
        sma_short = df["close"].rolling(20).mean()
        sma_long = df["close"].rolling(50).mean()
        df["bull_bear_regime"] = (sma_short > sma_long).astype(float)

        # Volatility regime
        volatility = returns.rolling(50).std()
        vol_median = volatility.rolling(100).median()
        df["volatility_regime"] = (volatility > vol_median).astype(float)

        # Mean reversion regime (Hurst exponent approximation)
        def simple_hurst(ts, max_lag=20):
            if len(ts) < max_lag * 2:
                return 0.5
            lags = range(2, min(max_lag, len(ts) // 2))
            if len(lags) < 2:
                return 0.5
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            if len(tau) < 2 or any(t <= 0 for t in tau):
                return 0.5
            try:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5

        df["hurst_exponent"] = returns.rolling(100).apply(simple_hurst, raw=True)
        df["mean_reversion_regime"] = (df["hurst_exponent"] < 0.5).astype(float)

        # Additional regime indicators
        df["trend_strength_index"] = self._calculate_trend_strength_index(df)
        df["market_regime"] = self._calculate_market_regime(df)
        df["ichimoku_signal"] = self._calculate_ichimoku_signal(df)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (6 features)."""
        if isinstance(df.index, pd.DatetimeIndex):
            # Cyclical encoding
            df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
            df["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

            # Binary indicators
            df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
            df["is_night"] = ((df.index.hour >= 22) | (df.index.hour <= 6)).astype(int)
        else:
            # Default values if no datetime index
            for feature in [
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
                "is_weekend",
                "is_night",
            ]:
                df[feature] = 0.0

        return df

    def _add_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical features (12 features)."""
        # Ichimoku components
        df["ichimoku_tenkan"] = self._calculate_ichimoku_tenkan(df)
        df["ichimoku_kijun"] = self._calculate_ichimoku_kijun(df)
        df["ichimoku_senkou_a"] = self._calculate_ichimoku_senkou_a(df)

        # Signal features
        if "bb_width_20" in df.columns:
            bb_width_rolling = df["bb_width_20"].rolling(20).mean()
            df["bb_squeeze"] = (df["bb_width_20"] < bb_width_rolling).astype(int)
        else:
            df["bb_squeeze"] = 0

        if "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)
        else:
            df["macd_bullish"] = 0

        # RSI signals
        if "rsi_14" in df.columns:
            df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
            df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        else:
            df["rsi_oversold"] = 0
            df["rsi_overbought"] = 0

        # Moving average crosses
        sma_5 = df["close"].rolling(5).mean()
        sma_20 = df["close"].rolling(20).mean()
        ema_5 = df["close"].ewm(span=5).mean()
        ema_20 = df["close"].ewm(span=20).mean()

        df["sma_cross"] = (sma_5 > sma_20).astype(int)
        df["ema_cross"] = (ema_5 > ema_20).astype(int)
        df["price_above_sma20"] = (df["close"] > sma_20).astype(int)
        df["price_above_ema20"] = (df["close"] > ema_20).astype(int)

        # Bollinger Band breakouts
        if "bb_upper_20" in df.columns and "bb_lower_20" in df.columns:
            df["bb_breakout_upper"] = (df["close"] > df["bb_upper_20"]).astype(int)
        else:
            df["bb_breakout_upper"] = 0

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features (6 features)."""
        returns = df["close"].pct_change()

        # Skewness and Kurtosis
        for window in [20]:
            df[f"returns_skew_{window}"] = returns.rolling(window).skew()
            df[f"returns_kurtosis_{window}"] = returns.rolling(window).kurt()

        # Auto-correlation
        for lag in [1, 5]:
            df[f"returns_autocorr_lag{lag}"] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else 0, raw=False
            )

        # Momentum persistence and strength
        momentum = df["close"].pct_change()
        df["momentum_persistence_20"] = momentum.rolling(20).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5, raw=True
        )
        df["momentum_strength_20"] = momentum.abs().rolling(20).mean()

        return df

    def _select_features_for_model(
        self, features_df: pd.DataFrame, model_type: str, symbol: str
    ) -> pd.DataFrame:
        """Select appropriate features for specific model type."""
        model_config = self.model_configs.get(model_type, {})
        expected_count = model_config.get("expected_feature_count", 100)

        # Get non-OHLCV features
        feature_cols = self._get_feature_columns(features_df)
        current_count = len(feature_cols)

        logger.debug(
            f"Selecting {expected_count} features from {current_count} available for {model_type}"
        )

        if current_count == expected_count:
            # Perfect match
            result_df = features_df.copy()
        elif current_count < expected_count:
            # Need to pad with additional features
            result_df = features_df.copy()
            for i in range(current_count, expected_count):
                result_df[f"pad_feature_{i}"] = 0.0
        else:
            # Need to select best features
            result_df = self._select_best_features(
                features_df, feature_cols, expected_count, model_type
            )

        return result_df

    def _select_best_features(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        target_count: int,
        model_type: str,
    ) -> pd.DataFrame:
        """Select best features using model-specific criteria."""
        try:
            # Model-specific feature priorities
            priority_patterns = {
                "ppo": [
                    "momentum",
                    "volatility",
                    "regime",
                    "rsi",
                    "macd",
                    "volume",
                    "trend",
                ],
                "gru": ["momentum", "technical", "price", "volatility", "volume"],
                "lightgbm": ["all"],  # LightGBM handles feature selection well
            }

            patterns = priority_patterns.get(model_type, ["all"])

            if "all" in patterns:
                # Use all features, truncated if necessary
                selected_features = feature_cols[:target_count]
            else:
                # Select features by priority patterns
                selected_features = []
                remaining_features = feature_cols.copy()

                # First pass: select features matching priority patterns
                for pattern in patterns:
                    matching = [f for f in remaining_features if pattern.lower() in f.lower()]
                    selected_features.extend(
                        matching[: min(len(matching), target_count - len(selected_features))]
                    )
                    remaining_features = [
                        f for f in remaining_features if f not in selected_features
                    ]

                    if len(selected_features) >= target_count:
                        break

                # Second pass: fill remaining slots with other features
                if len(selected_features) < target_count:
                    needed = target_count - len(selected_features)
                    selected_features.extend(remaining_features[:needed])

            # Ensure we have exactly the target count
            selected_features = selected_features[:target_count]

            # Create result DataFrame with original OHLCV + selected features
            ohlcv_cols = [
                col
                for col in features_df.columns
                if col in ["open", "high", "low", "close", "volume"]
            ]
            result_cols = ohlcv_cols + selected_features
            result_df = features_df[result_cols].copy()

            logger.debug(f"Selected {len(selected_features)} features for {model_type}")
            return result_df

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using first {target_count} features")
            selected_features = feature_cols[:target_count]
            ohlcv_cols = [
                col
                for col in features_df.columns
                if col in ["open", "high", "low", "close", "volume"]
            ]
            result_cols = ohlcv_cols + selected_features
            return features_df[result_cols].copy()

    # Technical indicator calculation methods (optimized versions)
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with improved error handling."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(np.clip(rsi.fillna(50), 0, 100), index=rsi.index)

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return (macd_line.fillna(0), macd_signal.fillna(0), macd_histogram.fillna(0))

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std().fillna(0)
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.fillna(prices), middle.fillna(prices), lower.fillna(prices)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.Series(np.maximum.reduce([high_low, high_close, low_close]), index=df.index)
        return true_range.rolling(window=period).mean().fillna(0)

    def _calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = df["low"].rolling(window=k_period).min()
        highest_high = df["high"].rolling(window=k_period).max()
        denominator = highest_high - lowest_low
        k_percent = 100 * ((df["close"] - lowest_low) / (denominator + 1e-10))
        k_percent = k_percent.fillna(50)
        d_percent = k_percent.rolling(window=d_period).mean().fillna(50)
        return k_percent, d_percent

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean() if len(x) > 0 else 0
        )
        cci = (typical_price - sma) / (0.015 * mean_deviation + 1e-10)
        return pd.Series(np.clip(cci.fillna(0), -500, 500), index=cci.index)

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high_diff = df["high"] - df["high"].shift()
        low_diff = df["low"].shift() - df["low"]

        plus_dm = pd.Series(
            np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0),
            index=df.index,
        )

        tr = self._calculate_atr(df, 1)  # True range for single period
        alpha = 1.0 / period

        atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100 * (plus_dm_smooth / (atr_smooth + 1e-10))
        minus_di = 100 * (minus_dm_smooth / (atr_smooth + 1e-10))
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return pd.Series(np.clip(adx.fillna(25), 0, 100), index=adx.index)

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()
        denominator = highest_high - lowest_low
        williams_r = -100 * ((highest_high - df["close"]) / (denominator + 1e-10))
        return pd.Series(np.clip(williams_r.fillna(-50), -100, 0), index=williams_r.index)

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = df["close"].diff()
        volume_direction = np.where(
            price_change > 0, df["volume"], np.where(price_change < 0, -df["volume"], 0)
        )
        obv = pd.Series(volume_direction, index=df.index).cumsum()
        return obv.fillna(0)

    def _calculate_vwap_deviation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP deviation as percentage."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1e-10)
        vwap_dev = (df["close"] - vwap) / (vwap + 1e-10) * 100
        return vwap_dev.fillna(0)

    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"] + 1e-10
        )
        mfv = clv * df["volume"]
        return mfv.cumsum().fillna(0)

    def _calculate_trend_strength_index(self, df: pd.DataFrame, period: int = 25) -> pd.Series:
        """Calculate Trend Strength Index."""
        momentum = df["close"] - df["close"].shift(period)
        abs_momentum_sum = momentum.abs().rolling(window=period).sum()
        positive_momentum = momentum.where(momentum > 0, 0).rolling(window=period).sum()
        negative_momentum = momentum.where(momentum < 0, 0).abs().rolling(window=period).sum()
        tsi = 100 * (positive_momentum - negative_momentum) / (abs_momentum_sum + 1e-10)
        return tsi.fillna(0)

    def _calculate_market_regime(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate market regime indicator."""
        price_change = abs(df["close"] - df["close"].shift(period))
        volatility_sum = df["close"].diff().abs().rolling(window=period).sum()
        efficiency_ratio = price_change / (volatility_sum + 1e-10)
        regime = efficiency_ratio.ewm(span=10).mean()
        return (regime * 100).fillna(50)

    def _calculate_ichimoku_tenkan(self, df: pd.DataFrame, period: int = 9) -> pd.Series:
        """Calculate Ichimoku Tenkan-sen."""
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        return ((high_max + low_min) / 2).fillna(df["close"])

    def _calculate_ichimoku_kijun(self, df: pd.DataFrame, period: int = 26) -> pd.Series:
        """Calculate Ichimoku Kijun-sen."""
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        return ((high_max + low_min) / 2).fillna(df["close"])

    def _calculate_ichimoku_senkou_a(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ichimoku Senkou Span A."""
        tenkan = self._calculate_ichimoku_tenkan(df)
        kijun = self._calculate_ichimoku_kijun(df)
        return ((tenkan + kijun) / 2).fillna(df["close"])

    def _calculate_ichimoku_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ichimoku signal (simplified)."""
        tenkan = self._calculate_ichimoku_tenkan(df)
        kijun = self._calculate_ichimoku_kijun(df)
        return (tenkan > kijun).astype(float)

    # Utility methods
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
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        return feature_cols

    def _validate_input_data(self, df: pd.DataFrame):
        """Validate input DataFrame."""
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(df) < 100:
            logger.warning(f"Small dataset ({len(df)} rows) - some features may be unreliable")

    def _validate_and_clean_source_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean source OHLCV data."""
        logger.debug("Validating and cleaning source data")

        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill and backward fill NaN values
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Clip extreme values
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = np.clip(df[col], 0.01, 1000000)

        if "volume" in df.columns:
            df["volume"] = np.clip(df["volume"], 0, 1e12)

        return df

    def _clean_intermediate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features during intermediate processing."""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill NaN values
        df = df.fillna(method="ffill").fillna(0)

        # Remove outliers if configured
        if self.config.get("feature_engineering", {}).get("remove_outliers", True):
            threshold = self.config.get("feature_engineering", {}).get("outlier_std_threshold", 5.0)
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col in ["open", "high", "low", "close", "volume"]:
                    continue

                mean_val = df[col].mean()
                std_val = df[col].std()

                if std_val > 0:
                    outlier_mask = np.abs(df[col] - mean_val) > threshold * std_val
                    df.loc[outlier_mask, col] = np.nan

        # Final cleanup
        df = df.fillna(method="ffill").fillna(0)

        return df

    def _apply_final_validation(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """Apply final validation and cleaning."""
        logger.debug(f"Applying final validation for {model_type}")

        # Replace any remaining infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill remaining NaN values
        df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Ensure all values are finite
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.Series(np.where(np.isfinite(df[col]), df[col], 0.0), index=df.index)

        # Log final statistics
        feature_count = len(self._get_feature_columns(df))
        nan_count = df.isnull().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

        if nan_count > 0 or inf_count > 0:
            logger.warning(
                f"Final validation found {nan_count} NaN and {inf_count} infinite values"
            )

        logger.debug(f"Final validation complete: {feature_count} features, {len(df)} rows")

        return df


# Convenience function for backward compatibility
def generate_enhanced_features(
    df: pd.DataFrame,
    model_type: str = "gru",
    symbol: str = "GENERIC",
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate enhanced features for specified model type.

    Args:
        df: Input DataFrame with OHLCV data
        model_type: Model type ('ppo', 'gru', 'lightgbm')
        symbol: Trading symbol
        config_path: Path to feature configuration file

    Returns:
        DataFrame with model-specific features
    """
    engine = EnhancedFeatureEngine(config_path)
    return engine.generate_features_for_model(df, model_type, symbol)
