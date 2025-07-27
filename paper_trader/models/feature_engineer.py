"""Feature engineering for technical analysis and model input preparation."""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List
# import pandas_ta as ta  # Temporarily disabled due to numpy compatibility issues
# import talib  # Commented out - not available on Windows without special installation

# Simple technical indicators implementation as fallback
class TechnicalIndicators:
    """Fallback technical indicators implementation"""
    
    @staticmethod
    def rsi(prices, length=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        # Return as DataFrame to match pandas_ta format
        result = pd.DataFrame({
            'MACD_12_26_9': macd,
            'MACDh_12_26_9': histogram,
            'MACDs_12_26_9': signal_line
        })
        return result
    
    @staticmethod
    def bbands(prices, length=20, std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=length).mean()
        std_dev = prices.rolling(window=length).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        # Return as DataFrame to match pandas_ta format
        result = pd.DataFrame({
            'BBL_20_2.0': lower_band,
            'BBM_20_2.0': sma,
            'BBU_20_2.0': upper_band,
            'BBB_20_2.0': (upper_band - lower_band) / sma,
            'BBP_20_2.0': (prices - lower_band) / (upper_band - lower_band)
        })
        return result
    
    @staticmethod
    def atr(high, low, close, length=14):
        """Calculate ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=length).mean()
    
    @staticmethod
    def roc(prices, length=10):
        """Calculate Rate of Change"""
        return ((prices - prices.shift(length)) / prices.shift(length)) * 100
    
    @staticmethod
    def stoch(high, low, close, k=14, d=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k).min()
        highest_high = high.rolling(window=k).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d).mean()
        # Return as DataFrame to match pandas_ta format
        result = pd.DataFrame({
            'STOCHk_14_3_3': k_percent,
            'STOCHd_14_3_3': d_percent
        })
        return result
    
    @staticmethod
    def willr(high, low, close, length=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=length).max()
        lowest_low = low.rolling(window=length).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))

# Use fallback implementation
ta = TechnicalIndicators()

# Feature order used during model training for LSTM scaling - Updated to match trained models
LSTM_FEATURES: List[str] = [
    'price_vs_ema_30min', 'price_vs_ema_1h', 'price_vs_ema_2h', 'price_vs_ema_4h',
    'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'volume_change_24h',
    'atr_14', 'adx_14', 'momentum_30min', 'volume_ema_ratio'
]

# Legacy LSTM features list (36 features) - kept for compatibility
LSTM_FEATURES_LEGACY: List[str] = [
    'close', 'volume', 'returns', 'log_returns',
    'volatility_20', 'atr_ratio', 'rsi', 'macd', 'bb_position',
    'volume_ratio', 'price_vs_ema9', 'price_vs_ema21',
    'buying_pressure', 'selling_pressure', 'spread_ratio',
    'momentum_10', 'price_zscore_20',
    # Jump-specific features
    'volume_surge_5', 'volume_surge_10', 'volume_acceleration',
    'momentum_acceleration', 'momentum_velocity', 'price_acceleration',
    'volatility_breakout', 'atr_breakout', 'resistance_breakout',
    'support_bounce', 'price_gap_up', 'momentum_convergence',
    'squeeze_setup', 'squeeze_breakout',
    # Market context features
    'bull_market', 'market_momentum_alignment', 'market_stress',
    'strong_trend', 'weak_trend'
]

# Sequence length used during model training
LSTM_SEQUENCE_LENGTH: int = 96

# Full feature list used during model training
TRAINING_FEATURES: List[str] = [
    # Enhanced Price features with multi-timeframe analysis
    'returns', 'log_returns', 'price_change_30min', 'price_change_1h', 
    'price_change_4h', 'price_change_24h', 'price_zscore_20', 'price_zscore_50',
    # Multi-timeframe volatility features
    'volatility_15min', 'volatility_30min', 'volatility_1h', 'volatility_4h',
    'vol_ratio_15min_30min', 'vol_ratio_30min_1h', 'vol_ratio_1h_4h',
    # Lag features (important for time series)
    'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5', 
    'returns_lag_10', 'log_returns_lag_1', 'log_returns_lag_2', 'log_returns_lag_3',
    # Rolling statistics
    'returns_mean_10', 'returns_std_10', 'returns_skew_20', 'returns_kurt_20',
    # Enhanced Volume features
    'volume_ratio', 'volume_change', 'volume_zscore', 'volume_price_trend', 
    'volume_weighted_price',
    # Market microstructure
    'spread', 'spread_ratio', 'buying_pressure', 'selling_pressure', 'net_pressure',
    # Enhanced Volatility
    'volatility_20', 'volatility_50', 'volatility_ratio', 'atr', 'atr_ratio', 
    'realized_vol_5', 'realized_vol_20',
    # Technical indicators
    'rsi', 'rsi_9', 'rsi_21', 'rsi_oversold', 'rsi_overbought', 'rsi_divergence',
    'macd', 'macd_signal', 'macd_hist', 'macd_bullish',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    # VWAP and price vs MA features
    'vwap', 'price_vs_vwap', 'sma_20', 'sma_50', 'sma_200',
    'ema_9', 'ema_21', 'ema_50', 'ema_100',
    'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
    'price_vs_ema9', 'price_vs_ema21', 'price_vs_ema50',
    # MA alignment and momentum
    'ma_alignment', 'ema9_vs_ema21', 'ema21_vs_ema50', 'ema50_vs_ema100',
    'ma_slope_9', 'ma_slope_21', 'momentum_10', 'roc_10',
    # Candle patterns and time features
    'candle_body', 'upper_wick', 'lower_wick', 'hour', 'day_of_week', 'is_weekend',
    # Support/Resistance and Stochastic
    'high_20', 'low_20', 'near_resistance', 'near_support',
    'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought', 'williams_r',
    # Volume regime
    'vol_regime'
]

class FeatureEngineer:
    """Creates technical indicators and features for ML models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_lstm_feature_columns(self) -> List[str]:
        """Return LSTM feature column order used during training."""
        return LSTM_FEATURES

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate that all required features are present."""
        missing_features = set(TRAINING_FEATURES) - set(df.columns)
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            return False
        return True
    def create_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create features - alias for engineer_features to maintain compatibility."""
        return self.engineer_features(data)
        
    def engineer_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create the same technical indicators used during model training."""
        if len(data) < 250:
            self.logger.warning(f"Insufficient data for feature engineering. Need at least 250 rows, got {len(data)}.")
            return None

        try:
            df = data.copy()
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')

            # Calculate all features in one go using a dictionary
            features = {}

            # Basic price features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Multi-timeframe price changes (assuming 15-minute intervals as in training)
            features['price_change_30min'] = df['close'].pct_change(2)  # 2 * 15min = 30min
            features['price_change_1h'] = df['close'].pct_change(4)    # 4 * 15min = 1h
            features['price_change_4h'] = df['close'].pct_change(16)   # 16 * 15min = 4h
            features['price_change_24h'] = df['close'].pct_change(96)  # 96 * 15min = 24h

            # Price statistics
            price_rolling_20 = df['close'].rolling(20)
            price_rolling_50 = df['close'].rolling(50)
            features['price_zscore_20'] = (df['close'] - price_rolling_20.mean()) / price_rolling_20.std()
            features['price_zscore_50'] = (df['close'] - price_rolling_50.mean()) / price_rolling_50.std()

            # Returns lag features
            for lag in [1, 2, 3, 5, 10, 20]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'log_returns_lag_{lag}'] = features['log_returns'].shift(lag)

            # Returns statistics
            returns_rolling_10 = features['returns'].rolling(10)
            returns_rolling_20 = features['returns'].rolling(20)
            features['returns_mean_10'] = returns_rolling_10.mean()
            features['returns_std_10'] = returns_rolling_10.std()
            features['returns_skew_20'] = returns_rolling_20.skew()
            features['returns_kurt_20'] = returns_rolling_20.kurt()

            # Volume features
            volume_rolling_20 = df['volume'].rolling(20)
            volume_sma_20 = volume_rolling_20.mean()
            volume_std_20 = volume_rolling_20.std()
            features['volume_sma_20'] = volume_sma_20
            features['volume_ratio'] = df['volume'] / volume_sma_20
            features['volume_change'] = df['volume'].pct_change()
            features['volume_change_24h'] = df['volume'].pct_change(96)  # Required for LSTM
            features['volume_zscore'] = (df['volume'] - volume_sma_20) / volume_std_20
            features['volume_price_trend'] = df['volume'] * features['returns']
            features['volume_weighted_price'] = (
                (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
            )
            
            # Volume EMA features (required for LSTM)
            features['volume_ema_14'] = df['volume'].ewm(span=14).mean()
            features['volume_ema_ratio'] = df['volume'] / features['volume_ema_14']

            # Market microstructure
            features['spread'] = (df['high'] - df['low']) / df['close']

            # Volatility features
            features['volatility_20'] = features['returns'].rolling(20).std()
            features['volatility_5'] = features['returns'].rolling(5).std()
            features['volatility_50'] = features['returns'].rolling(50).std()
            features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
            features['realized_vol_5'] = features['returns'].rolling(5).std() * np.sqrt(5)
            features['realized_vol_20'] = features['returns'].rolling(20).std() * np.sqrt(20)
            
            # Multi-timeframe volatility features (assuming 15-minute intervals as in training)
            features['volatility_15min'] = features['returns'].rolling(4).std()   # 4 periods = 1 hour of 15min data
            features['volatility_30min'] = features['returns'].rolling(8).std()   # 8 periods = 2 hours
            features['volatility_1h'] = features['returns'].rolling(16).std()     # 16 periods = 4 hours
            features['volatility_4h'] = features['returns'].rolling(64).std()     # 64 periods = 16 hours
            
            # Cross-timeframe volatility ratios for regime detection
            features['vol_ratio_15min_30min'] = features['volatility_15min'] / features['volatility_30min']
            features['vol_ratio_30min_1h'] = features['volatility_30min'] / features['volatility_1h']
            features['vol_ratio_1h_4h'] = features['volatility_1h'] / features['volatility_4h']

            # Technical indicators using pandas_ta
            df_temp = pd.concat([df, pd.DataFrame(features)], axis=1)
            
            # RSI
            rsi_values = ta.rsi(df['close'], length=14)
            features['rsi'] = rsi_values
            features['rsi_14'] = rsi_values  # Alias for LSTM compatibility
            features['rsi_7'] = ta.rsi(df['close'], length=7)  # Required for LSTM
            features['rsi_9'] = ta.rsi(df['close'], length=9)
            features['rsi_21'] = ta.rsi(df['close'], length=21)
            features['rsi_oversold'] = (rsi_values < 30).astype(int)
            features['rsi_overbought'] = (rsi_values > 70).astype(int)
            features['rsi_divergence'] = features['rsi'].diff(5) * df['close'].pct_change(5)

            # MACD
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_result is not None and not macd_result.empty:
                features['macd'] = macd_result.iloc[:, 0]  # MACD line
                features['macd_signal'] = macd_result.iloc[:, 1]  # Signal line
                macd_hist = macd_result.iloc[:, 2]
                features['macd_hist'] = macd_hist
                features['macd_histogram'] = macd_hist
            else:
                features['macd'] = pd.Series(0, index=df.index)
                features['macd_signal'] = pd.Series(0, index=df.index)
                features['macd_hist'] = pd.Series(0, index=df.index)
                features['macd_histogram'] = pd.Series(0, index=df.index)
            features['macd_bullish'] = (features['macd'] > features['macd_signal']).astype(int)

            # Bollinger Bands
            bb_result = ta.bbands(df['close'], length=20, std=2)
            if bb_result is not None and not bb_result.empty:
                bb_upper = bb_result.iloc[:, 0]  # Upper band
                bb_middle = bb_result.iloc[:, 1]  # Middle band (SMA)
                bb_lower = bb_result.iloc[:, 2]  # Lower band
                features['bb_upper'] = bb_upper
                features['bb_middle'] = bb_middle
                features['bb_lower'] = bb_lower
                features['bb_width'] = (bb_upper - bb_lower) / bb_middle
                features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            else:
                features['bb_upper'] = df['close']
                features['bb_middle'] = df['close']
                features['bb_lower'] = df['close']
                features['bb_width'] = pd.Series(0, index=df.index)
                features['bb_position'] = pd.Series(0.5, index=df.index)

            # ATR
            atr_values = ta.atr(df['high'], df['low'], df['close'], length=14)
            features['atr'] = atr_values if atr_values is not None else pd.Series(0, index=df.index)
            features['atr_14'] = features['atr']  # Alias for LSTM compatibility
            features['atr_ratio'] = features['atr'] / df['close']
            
            # ADX (required for LSTM)
            # Simple approximation using directional movement
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
            minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
            
            # Smooth the directional movements
            plus_dm_smooth = plus_dm.rolling(14).mean()
            minus_dm_smooth = minus_dm.rolling(14).mean()
            tr_smooth = features['atr']
            
            # Calculate DI+ and DI-
            plus_di = 100 * plus_dm_smooth / tr_smooth
            minus_di = 100 * minus_dm_smooth / tr_smooth
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            features['adx_14'] = dx.rolling(14).mean().fillna(25)  # Default neutral ADX

            # VWAP
            vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            features['vwap'] = vwap
            features['price_vs_vwap'] = (df['close'] - vwap) / vwap

            # Moving averages
            features['sma_20'] = df['close'].rolling(20).mean()
            features['sma_50'] = df['close'].rolling(50).mean()
            features['sma_200'] = df['close'].rolling(200).mean()
            features['ema_9'] = df['close'].ewm(span=9).mean()
            features['ema_21'] = df['close'].ewm(span=21).mean()
            features['ema_50'] = df['close'].ewm(span=50).mean()
            features['ema_100'] = df['close'].ewm(span=100).mean()

            # Price vs MA features
            features['price_vs_sma20'] = (df['close'] - features['sma_20']) / features['sma_20']
            features['price_vs_sma50'] = (df['close'] - features['sma_50']) / features['sma_50']
            features['price_vs_sma200'] = (df['close'] - features['sma_200']) / features['sma_200']
            features['price_vs_ema9'] = (df['close'] - features['ema_9']) / features['ema_9']
            features['price_vs_ema21'] = (df['close'] - features['ema_21']) / features['ema_21']
            features['price_vs_ema50'] = (df['close'] - features['ema_50']) / features['ema_50']
            
            # LSTM-specific price vs EMA features (using different timeframe approximations)
            features['price_vs_ema_30min'] = features['price_vs_ema9']  # Short-term
            features['price_vs_ema_1h'] = features['price_vs_ema21']    # Medium-term  
            features['price_vs_ema_2h'] = features['price_vs_ema50']    # Longer-term
            features['price_vs_ema_4h'] = (df['close'] - features['ema_100']) / features['ema_100']  # Very long-term

            # MA alignment and crossovers
            ma_9_vs_21 = (features['ema_9'] > features['ema_21']).astype(int)
            ma_21_vs_50 = (features['ema_21'] > features['ema_50']).astype(int)
            features['ma_alignment'] = ma_9_vs_21 + ma_21_vs_50 - 1  # -1, 0, or 1
            features['ema9_vs_ema21'] = (features['ema_9'] - features['ema_21']) / features['ema_21']
            features['ema21_vs_ema50'] = (features['ema_21'] - features['ema_50']) / features['ema_50']
            features['ema50_vs_ema100'] = (features['ema_50'] - features['ema_100']) / features['ema_100']
            features['ma_slope_9'] = features['ema_9'].pct_change(5)
            features['ma_slope_21'] = features['ema_21'].pct_change(5)

            # Order flow features
            features['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'])).fillna(0.5)
            features['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'])).fillna(0.5)
            features['net_pressure'] = features['buying_pressure'] - features['selling_pressure']
            features['spread_ratio'] = (df['high'] - df['low']) / df['close']

            # Momentum
            features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            features['momentum_30min'] = df['close'].pct_change(periods=2)  # Approximation for 30min momentum
            features['roc_10'] = ta.roc(df['close'], length=10)

            # Candle patterns
            features['candle_body'] = abs(df['close'] - df['open']) / df['open']
            features['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
            features['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']

            # Time features (only if index is datetime)
            if hasattr(df.index, 'hour'):
                features['hour'] = df.index.hour
                features['day_of_week'] = df.index.dayofweek
                features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            else:
                # Default values when no datetime index available
                features['hour'] = pd.Series(12, index=df.index)  # Default to noon
                features['day_of_week'] = pd.Series(2, index=df.index)  # Default to Tuesday
                features['is_weekend'] = pd.Series(0, index=df.index)  # Default to weekday

            # Support/Resistance
            features['high_20'] = df['high'].rolling(20).max()
            features['low_20'] = df['low'].rolling(20).min()
            features['near_resistance'] = (df['close'] / features['high_20'] > 0.98).astype(int)
            features['near_support'] = (df['close'] / features['low_20'] < 1.02).astype(int)

            # Stochastic
            stoch_result = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch_result is not None and not stoch_result.empty:
                stoch_k = stoch_result.iloc[:, 0]
                stoch_d = stoch_result.iloc[:, 1]
                features['stoch_k'] = stoch_k
                features['stoch_d'] = stoch_d
                features['stoch_oversold'] = (stoch_k < 20).astype(int)
                features['stoch_overbought'] = (stoch_k > 80).astype(int)
            else:
                features['stoch_k'] = pd.Series(50, index=df.index)
                features['stoch_d'] = pd.Series(50, index=df.index)
                features['stoch_oversold'] = pd.Series(0, index=df.index)
                features['stoch_overbought'] = pd.Series(0, index=df.index)

            # Williams %R
            features['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)

            # Volume regime
            volume_percentile = df['volume'].rolling(100).rank(pct=True)
            features['vol_regime'] = (volume_percentile > 0.8).astype(int)

            # Jump-specific features for enhanced price jump detection
            # Volume surge indicators
            features['volume_surge_5'] = (df['volume'] / df['volume'].rolling(5).mean() > 2.0).astype(int)
            features['volume_surge_10'] = (df['volume'] / df['volume'].rolling(10).mean() > 1.5).astype(int)
            features['volume_acceleration'] = df['volume'].pct_change(1) - df['volume'].pct_change(2)
            
            # Momentum acceleration (detecting momentum buildup)
            features['momentum_acceleration'] = features['momentum_10'].diff()
            features['momentum_velocity'] = features['momentum_10'].pct_change()
            features['price_acceleration'] = df['close'].pct_change().diff()
            
            # Volatility breakout signals
            volatility_zscore = (features['volatility_20'] - features['volatility_20'].rolling(50).mean()) / features['volatility_20'].rolling(50).std()
            features['volatility_breakout'] = (volatility_zscore > 1.5).astype(int)
            features['atr_breakout'] = (features['atr'] > features['atr'].rolling(20).quantile(0.8)).astype(int)
            
            # Support/resistance breakout indicators  
            features['resistance_breakout'] = (df['close'] > features['high_20'].shift(1)).astype(int)
            features['support_bounce'] = (df['close'] > features['low_20'].shift(1) * 1.005).astype(int)
            
            # Price gap detection
            features['price_gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1) > 0.002).astype(int)
            features['price_gap_down'] = ((df['close'].shift(1) - df['open']) / df['close'].shift(1) > 0.002).astype(int)
            
            # Multi-timeframe momentum convergence
            short_momentum = df['close'].pct_change(3)
            medium_momentum = df['close'].pct_change(7)
            long_momentum = df['close'].pct_change(15)
            features['momentum_convergence'] = ((short_momentum > 0) & (medium_momentum > 0) & (long_momentum > 0)).astype(int)
            
            # Squeeze breakout (low volatility followed by expansion)
            bb_squeeze = features['bb_width'] < features['bb_width'].rolling(20).quantile(0.2)
            features['squeeze_setup'] = bb_squeeze.astype(int)
            features['squeeze_breakout'] = (bb_squeeze.shift(1) & (features['bb_width'] > features['bb_width'].shift(1) * 1.2)).astype(int)
            
            # Market context features for broader market sentiment
            # Market regime detection
            sma_200 = df['close'].rolling(200).mean()
            features['bull_market'] = (df['close'] > sma_200).astype(int)
            features['bear_market'] = (df['close'] < sma_200 * 0.95).astype(int)
            
            # Market momentum context
            market_momentum_short = df['close'].pct_change(24)  # 6 hours
            market_momentum_medium = df['close'].pct_change(96)  # 24 hours  
            market_momentum_long = df['close'].pct_change(288)  # 72 hours
            features['market_momentum_alignment'] = (
                (market_momentum_short > 0) & 
                (market_momentum_medium > 0) & 
                (market_momentum_long > 0)
            ).astype(int)
            
            # Market stress indicators
            price_volatility_percentile = features['volatility_20'].rolling(200).rank(pct=True)
            volume_volatility_percentile = df['volume'].rolling(200).rank(pct=True)
            features['market_stress'] = (
                (price_volatility_percentile > 0.8) | 
                (volume_volatility_percentile > 0.8)
            ).astype(int)
            
            # Trend strength context
            trend_consistency = (df['close'] > df['close'].shift(1)).rolling(10).sum() / 10
            features['strong_trend'] = (trend_consistency > 0.7).astype(int)
            features['weak_trend'] = (trend_consistency < 0.3).astype(int)

            # Complex features - calculate after basic features are available
            # Update df_temp with all basic features first
            df_temp = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)
            
            # Now calculate complex features using the updated dataframe
            complex_features = {}
            complex_features['rsi_macd_signal'] = features['rsi'] * features['macd_signal']
            complex_features['rsi_macd_combo'] = features['rsi'] * features['macd_signal']
            complex_features['volatility_ema_ratio'] = features['volatility_20'] / features['ema_21']
            complex_features['volume_price_momentum'] = features['volume_ratio'] * features['momentum_10']
            complex_features['bb_rsi_signal'] = features['bb_position'] * features['rsi']
            complex_features['trend_strength'] = features['price_vs_ema9'] * features['price_vs_ema21']
            complex_features['volatility_breakout'] = features['atr'] * features['bb_width']
            complex_features['momentum_vol_signal'] = features['momentum_10'] * features['volume_ratio'] * features['volatility_ratio']
            complex_features['trend_momentum_align'] = features['ma_alignment'] * features['momentum_10']
            complex_features['pressure_volume_signal'] = features['net_pressure'] * features['volume_zscore']
            complex_features['volatility_regime_signal'] = features['vol_regime'] * features['rsi']
            complex_features['multi_timeframe_signal'] = features['price_change_1h'] * features['price_change_4h'] * features['price_change_24h']
            complex_features['oscillator_consensus'] = (features['rsi_oversold'] + features['stoch_oversold']) - (features['rsi_overbought'] + features['stoch_overbought'])
            
            # Advanced feature interactions and polynomial terms for enhanced prediction
            complex_features['price_volume_interaction'] = df['close'] * features['volume_ratio']
            complex_features['volatility_momentum_cross'] = features['volatility_20'] * features['momentum_10']
            complex_features['rsi_bb_cross'] = features['rsi'] * features['bb_position']
            complex_features['macd_volume_signal'] = features['macd_histogram'] * features['volume_ratio']
            
            # Polynomial features for non-linear relationships
            complex_features['rsi_squared'] = features['rsi'] ** 2
            complex_features['volatility_squared'] = features['volatility_20'] ** 2
            complex_features['momentum_squared'] = features['momentum_10'] ** 2
            
            # Cross-timeframe feature interactions  
            complex_features['short_long_momentum_ratio'] = features['price_change_1h'] / (features['price_change_24h'] + 1e-8)
            complex_features['volatility_momentum_regime'] = features['vol_regime'] * features['momentum_10']
            
            # Calculate trend_regime first before using it
            complex_features['trend_regime'] = ((features['ma_alignment'] == 1) & (features['price_vs_sma200'] > 0)).astype(int)
            complex_features['trend_volatility_signal'] = complex_features['trend_regime'] * features['volatility_breakout']
            
            # Market microstructure advanced features
            complex_features['order_flow_strength'] = features['buying_pressure'] - features['selling_pressure']
            complex_features['volume_weighted_momentum'] = features['momentum_10'] * features['volume_weighted_price']
            complex_features['pressure_asymmetry'] = abs(features['buying_pressure'] - features['selling_pressure'])
            
            # Advanced trend detection
            trend_momentum_5 = df['close'].pct_change(5)
            trend_momentum_15 = df['close'].pct_change(15)
            trend_momentum_30 = df['close'].pct_change(30)
            denominator = np.maximum(trend_momentum_15 - trend_momentum_30, 1e-4)
            complex_features['trend_acceleration'] = (trend_momentum_5 - trend_momentum_15) / denominator
            
            # Market regime strength indicators
            complex_features['bull_strength'] = features['bull_market'] * (df['close'] / features['sma_200'] - 1)
            complex_features['bear_strength'] = features['bear_market'] * (1 - df['close'] / features['sma_200'])
            
            # Volume profile features
            volume_profile_20 = df['volume'].rolling(20)
            complex_features['volume_percentile_20'] = volume_profile_20.rank(pct=True)
            complex_features['volume_zscore_20'] = (df['volume'] - volume_profile_20.mean()) / volume_profile_20.std()
            
            # Price action patterns
            complex_features['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum() / 5
            complex_features['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum() / 5
            complex_features['price_pattern_strength'] = complex_features['higher_highs'] - complex_features['lower_lows']
            
            # Enhanced oscillator combinations
            complex_features['stoch_rsi_divergence'] = (features['stoch_k'].diff() * features['rsi'].diff())
            complex_features['williams_rsi_combo'] = features['williams_r'] * features['rsi'] / 100
            complex_features['oscillator_momentum'] = (features['rsi'] + features['stoch_k'] + abs(features['williams_r'])) / 3
            
            # Market timing features
            if isinstance(df.index, pd.DatetimeIndex):
                # Trading session strength (based on typical crypto trading patterns)
                asian_hours = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
                european_hours = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)  
                american_hours = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)
                
                complex_features['session_volume_strength'] = (
                    asian_hours * 0.7 + european_hours * 1.0 + american_hours * 1.2
                ) * features['volume_ratio']
                
                # Day of week patterns
                weekday_strength = pd.Series(0.0, index=df.index)
                weekday_strength[df.index.dayofweek <= 4] = 1.0  # Weekdays
                weekday_strength[df.index.dayofweek >= 5] = 0.8  # Weekends (lower activity)
                complex_features['weekday_volume_pattern'] = weekday_strength * features['volume_ratio']
            else:
                # Default values when no datetime index available
                complex_features['session_volume_strength'] = features['volume_ratio']
                complex_features['weekday_volume_pattern'] = features['volume_ratio']
            
            # BB width rolling calculation
            bb_width_rolling = features['bb_width'].rolling(50)
            atr_ratio_rolling = features['atr_ratio'].rolling(50)
            complex_features['consolidation_regime'] = (
                (features['bb_width'] < bb_width_rolling.quantile(0.3)) & 
                (features['atr_ratio'] < atr_ratio_rolling.quantile(0.3))
            ).astype(int)

            # Combine all features
            all_features = {**features, **complex_features}

            # Concatenate all features at once to avoid DataFrame fragmentation
            df_new_features = pd.DataFrame(all_features, index=df.index)
            df_final = pd.concat([df, df_new_features], axis=1)

            # Replace inf values with NaN then drop
            numeric_data = df_final.select_dtypes(include=[np.number])
            if np.isinf(numeric_data.to_numpy()).any():
                self.logger.debug("Replacing inf values in engineered features")
                df_final.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop rows with NaN values
            before_rows = len(df_final)
            df_final = df_final.dropna()
            dropped = before_rows - len(df_final)
            if dropped > 0:
                self.logger.debug(f"Dropped {dropped} rows due to NaN/inf values")

            if len(df_final) < 30:
                self.logger.warning(f"Insufficient data after feature engineering and NaN removal. Remaining rows: {len(df_final)}")
                return None

            # Validate that all required training features are present
            if not self.validate_features(df_final):
                self.logger.warning("Not all required training features are present")
                return None
                
            self.logger.info(f"Successfully created {len(df_final.columns)} features from {len(df_final)} samples")
            return df_final
        
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}", exc_info=True)
            return None
        
    def prepare_lstm_sequences(self, data: pd.DataFrame, sequence_length: int = 96,
                              target_col: str = 'close') -> tuple:
        """Prepare sequences for LSTM model input."""
        try:
            if len(data) < sequence_length + 1:
                self.logger.warning(f"Insufficient data for sequence length {sequence_length}")
                return None, None
            
            # Select features for LSTM (exclude target and non-numeric columns)
            feature_cols = [col for col in data.columns 
                          if col != target_col and data[col].dtype in ['float64', 'int64']]
            
            features = data[feature_cols].values
            target = data[target_col].values
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features)):
                X.append(features[i-sequence_length:i])
                y.append(target[i])
            
            X = np.array(X)
            y = np.array(y)
            
            self.logger.debug(f"Created LSTM sequences: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing LSTM sequences: {e}")
            return None, None
    
    def get_feature_names(self, data: pd.DataFrame) -> list:
        """Get list of feature column names."""
        return [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
    
    def normalize_features(self, data: pd.DataFrame, scaler=None) -> tuple:
        """Normalize features using provided scaler or return data as-is."""
        try:
            if scaler is not None:
                feature_cols = self.get_feature_names(data)
                data_scaled = data.copy()
                data_scaled[feature_cols] = scaler.transform(
                    data[feature_cols].to_numpy()
                )
                return data_scaled, scaler
            else:
                return data, None
                
        except Exception as e:
            self.logger.error(f"Error normalizing features: {e}")
            return data, scaler
