"""Feature engineering for technical analysis and model input preparation."""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List
import pandas_ta as ta
# import talib  # Commented out - not available on Windows without special installation

# Feature order used during model training for LSTM scaling
LSTM_FEATURES: List[str] = [
    'close', 'volume', 'returns', 'log_returns',
    'volatility_20', 'atr_ratio', 'rsi', 'macd', 'bb_position',
    'volume_ratio', 'price_vs_ema9', 'price_vs_ema21',
    'buying_pressure', 'selling_pressure', 'spread_ratio',
    'momentum_10', 'price_zscore_20'
]

# Full feature list used during model training
TRAINING_FEATURES: List[str] = [
    'returns', 'log_returns', 'price_change_1h', 'price_change_4h',
    'volatility_20', 'atr_ratio', 'rsi', 'macd', 'bb_position',
    'volume_ratio', 'price_vs_ema9', 'price_vs_ema21', 'buying_pressure',
    'selling_pressure', 'spread_ratio', 'momentum_10', 'price_zscore_20'
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
        
    def engineer_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create the same technical indicators used during model training."""
        try:
            if len(data) < 50:
                self.logger.warning("Insufficient data for feature engineering")
                return None

            df = data.copy()
            
            # Ensure proper column names
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # Price based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_change_1h'] = df['close'].pct_change(4)
            df['price_change_4h'] = df['close'].pct_change(16)
            df['price_change_24h'] = df['close'].pct_change(96)

            df['price_zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            df['price_zscore_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()

            for lag in [1, 2, 3, 5, 10, 20]:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'log_returns_lag_{lag}'] = df['log_returns'].shift(lag)

            df['returns_mean_10'] = df['returns'].rolling(10).mean()
            df['returns_std_10'] = df['returns'].rolling(10).std()
            df['returns_skew_20'] = df['returns'].rolling(20).skew()
            df['returns_kurt_20'] = df['returns'].rolling(20).kurt()

            # Volume features
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['volume_change'] = df['volume'].pct_change()
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

            df['volume_price_trend'] = df['volume'] * df['returns']
            df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()

            # Market microstructure
            df['spread'] = (df['high'] - df['low']) / df['close']
            df['spread_ma'] = df['spread'].rolling(20).mean()
            df['spread_ratio'] = df['spread'] / df['spread_ma']

            df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
            df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']

            # Volatility
            df['volatility_20'] = df['returns'].rolling(20).std()
            df['volatility_50'] = df['returns'].rolling(50).std()
            df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_ratio'] = df['atr'] / df['close']

            df['realized_vol_5'] = df['returns'].rolling(5).std() * np.sqrt(5)
            df['realized_vol_20'] = df['returns'].rolling(20).std() * np.sqrt(20)

            df['vol_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).quantile(0.75)).astype(int)

            # Moving averages
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_100'] = ta.ema(df['close'], length=100)
            df['sma_200'] = ta.sma(df['close'], length=200)

            df['price_vs_ema9'] = (df['close'] - df['ema_9']) / df['ema_9']
            df['price_vs_ema21'] = (df['close'] - df['ema_21']) / df['ema_21']
            df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['ema_50']
            df['price_vs_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']

            df['ema9_vs_ema21'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
            df['ema21_vs_ema50'] = (df['ema_21'] - df['ema_50']) / df['ema_50']
            df['ema50_vs_ema100'] = (df['ema_50'] - df['ema_100']) / df['ema_100']

            df['ma_alignment'] = ((df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50']) & (df['ema_50'] > df['ema_100'])).astype(int)
            df['ma_slope_9'] = df['ema_9'].pct_change(5)
            df['ma_slope_21'] = df['ema_21'].pct_change(5)

            # Oscillators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['rsi_9'] = ta.rsi(df['close'], length=9)
            df['rsi_21'] = ta.rsi(df['close'], length=21)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_divergence'] = df['rsi'].diff(5) * df['close'].pct_change(5)

            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
            df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)

            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)

            macd_data = ta.macd(df['close'])
            df['macd'] = macd_data['MACD_12_26_9']
            df['macd_signal'] = macd_data['MACDs_12_26_9']
            df['macd_histogram'] = macd_data['MACDh_12_26_9']
            df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)

            bb_data = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb_data['BBU_20_2.0']
            df['bb_lower'] = bb_data['BBL_20_2.0']
            df['bb_middle'] = bb_data['BBM_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']

            df['candle_body'] = abs(df['close'] - df['open']) / df['open']
            df['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
            df['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']

            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

            df['momentum_10'] = ta.mom(df['close'], length=10)
            df['roc_10'] = ta.roc(df['close'], length=10)

            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            df['near_resistance'] = (df['close'] / df['high_20'] > 0.98).astype(int)
            df['near_support'] = (df['close'] / df['low_20'] < 1.02).astype(int)

            df['rsi_macd_combo'] = df['rsi'] * df['macd_signal']
            df['volatility_ema_ratio'] = df['volatility_20'] / df['ema_21']
            df['volume_price_momentum'] = df['volume_ratio'] * df['momentum_10']
            df['bb_rsi_signal'] = df['bb_position'] * df['rsi']
            df['trend_strength'] = df['price_vs_ema9'] * df['price_vs_ema21']
            df['volatility_breakout'] = df['atr'] * df['bb_width']

            df['momentum_vol_signal'] = df['momentum_10'] * df['volume_ratio'] * df['volatility_ratio']
            df['trend_momentum_align'] = df['ma_alignment'] * df['momentum_10']
            df['pressure_volume_signal'] = df['net_pressure'] * df['volume_zscore']
            df['volatility_regime_signal'] = df['vol_regime'] * df['rsi']
            df['multi_timeframe_signal'] = df['price_change_1h'] * df['price_change_4h'] * df['price_change_24h']
            df['oscillator_consensus'] = (df['rsi_oversold'] + df['stoch_oversold']) - (df['rsi_overbought'] + df['stoch_overbought'])

            df['trend_regime'] = ((df['ma_alignment'] == 1) & (df['price_vs_sma200'] > 0)).astype(int)
            df['consolidation_regime'] = ((df['bb_width'] < df['bb_width'].rolling(50).quantile(0.3)) &
                                          (df['atr_ratio'] < df['atr_ratio'].rolling(50).quantile(0.3))).astype(int)

            # Drop rows with NaN values
            df = df.dropna()
            
            # Drop rows with NaN values
            df = df.dropna()
            
            if len(df) < 30:
                self.logger.warning("Insufficient data after feature engineering")
                return None
                
            self.logger.debug(f"Created {len(df.columns)} features from {len(df)} samples")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            return None
    
    def prepare_lstm_sequences(self, data: pd.DataFrame, sequence_length: int = 60, 
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
                data_scaled[feature_cols] = scaler.transform(data[feature_cols])
                return data_scaled, scaler
            else:
                return data, None
                
        except Exception as e:
            self.logger.error(f"Error normalizing features: {e}")
            return data, scaler