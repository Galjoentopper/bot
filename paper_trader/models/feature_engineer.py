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
        if len(data) < 250:
            self.logger.warning(f"Insufficient data for feature engineering. Need at least 250 rows, got {len(data)}.")
            return None

        try:
            df = data.copy()
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')

            # Use a dictionary to store new features and assign them all at once
            features = {}

            # Price based features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['price_change_1h'] = df['close'].pct_change(4)
            features['price_change_4h'] = df['close'].pct_change(16)
            features['price_change_24h'] = df['close'].pct_change(96)
            features['price_zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            features['price_zscore_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()

            for lag in [1, 2, 3, 5, 10, 20]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'log_returns_lag_{lag}'] = features['log_returns'].shift(lag)

            features['returns_mean_10'] = features['returns'].rolling(10).mean()
            features['returns_std_10'] = features['returns'].rolling(10).std()
            features['returns_skew_20'] = features['returns'].rolling(20).skew()
            features['returns_kurt_20'] = features['returns'].rolling(20).kurt()

            # Volume features
            volume_sma_20 = df['volume'].rolling(20).mean()
            features['volume_sma_20'] = volume_sma_20
            features['volume_ratio'] = df['volume'] / volume_sma_20
            features['volume_change'] = df['volume'].pct_change()
            features['volume_zscore'] = (df['volume'] - volume_sma_20) / df['volume'].rolling(20).std()
            features['volume_price_trend'] = df['volume'] * features['returns']
            features['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()

            # Market microstructure
            spread = (df['high'] - df['low']) / df['close']
            spread_ma = spread.rolling(20).mean()
            features['spread'] = spread
            features['spread_ma'] = spread_ma
            features['spread_ratio'] = spread / spread_ma
            features['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            features['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
            features['net_pressure'] = features['buying_pressure'] - features['selling_pressure']

            # Volatility
            volatility_20 = features['returns'].rolling(20).std()
            volatility_50 = features['returns'].rolling(50).std()
            features['volatility_20'] = volatility_20
            features['volatility_50'] = volatility_50
            features['volatility_ratio'] = volatility_20 / volatility_50
            features['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            features['atr_ratio'] = features['atr'] / df['close']
            features['realized_vol_5'] = features['returns'].rolling(5).std() * np.sqrt(5)
            features['realized_vol_20'] = features['returns'].rolling(20).std() * np.sqrt(20)
            features['vol_regime'] = (volatility_20 > volatility_20.rolling(100).quantile(0.75)).astype(int)

            # Moving averages
            ema_9 = ta.ema(df['close'], length=9)
            ema_21 = ta.ema(df['close'], length=21)
            ema_50 = ta.ema(df['close'], length=50)
            ema_100 = ta.ema(df['close'], length=100)
            sma_200 = ta.sma(df['close'], length=200)
            features['ema_9'] = ema_9
            features['ema_21'] = ema_21
            features['ema_50'] = ema_50
            features['ema_100'] = ema_100
            features['sma_200'] = sma_200
            features['price_vs_ema9'] = (df['close'] - ema_9) / ema_9
            features['price_vs_ema21'] = (df['close'] - ema_21) / ema_21
            features['price_vs_ema50'] = (df['close'] - ema_50) / ema_50
            features['price_vs_sma200'] = (df['close'] - sma_200) / sma_200
            features['ema9_vs_ema21'] = (ema_9 - ema_21) / ema_21
            features['ema21_vs_ema50'] = (ema_21 - ema_50) / ema_50
            features['ema50_vs_ema100'] = (ema_50 - ema_100) / ema_100
            features['ma_alignment'] = ((ema_9 > ema_21) & (ema_21 > ema_50) & (ema_50 > ema_100)).astype(int)
            features['ma_slope_9'] = ema_9.pct_change(5)
            features['ma_slope_21'] = ema_21.pct_change(5)

            # Oscillators
            rsi = ta.rsi(df['close'], length=14)
            features['rsi'] = rsi
            features['rsi_9'] = ta.rsi(df['close'], length=9)
            features['rsi_21'] = ta.rsi(df['close'], length=21)
            features['rsi_oversold'] = (rsi < 30).astype(int)
            features['rsi_overbought'] = (rsi > 70).astype(int)
            features['rsi_divergence'] = rsi.diff(5) * df['close'].pct_change(5)

            stoch = ta.stoch(df['high'], df['low'], df['close'])
            features['stoch_k'] = stoch['STOCHk_14_3_3']
            features['stoch_d'] = stoch['STOCHd_14_3_3']
            features['stoch_oversold'] = (features['stoch_k'] < 20).astype(int)
            features['stoch_overbought'] = (features['stoch_k'] > 80).astype(int)

            features['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)

            macd_data = ta.macd(df['close'])
            features['macd'] = macd_data['MACD_12_26_9']
            features['macd_signal'] = macd_data['MACDs_12_26_9']
            features['macd_histogram'] = macd_data['MACDh_12_26_9']
            features['macd_bullish'] = (features['macd'] > features['macd_signal']).astype(int)

            bb_data = ta.bbands(df['close'], length=20, std=2)
            features['bb_upper'] = bb_data['BBU_20_2.0']
            features['bb_lower'] = bb_data['BBL_20_2.0']
            features['bb_middle'] = bb_data['BBM_20_2.0']
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            features['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            features['price_vs_vwap'] = (df['close'] - features['vwap']) / features['vwap']

            features['candle_body'] = abs(df['close'] - df['open']) / df['open']
            features['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
            features['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']

            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

            features['momentum_10'] = ta.mom(df['close'], length=10)
            features['roc_10'] = ta.roc(df['close'], length=10)

            high_20 = df['high'].rolling(20).max()
            low_20 = df['low'].rolling(20).min()
            features['high_20'] = high_20
            features['low_20'] = low_20
            features['near_resistance'] = (df['close'] / high_20 > 0.98).astype(int)
            features['near_support'] = (df['close'] / low_20 < 1.02).astype(int)

            features['rsi_macd_combo'] = features['rsi'] * features['macd_signal']
            features['volatility_ema_ratio'] = features['volatility_20'] / features['ema_21']
            features['volume_price_momentum'] = features['volume_ratio'] * features['momentum_10']
            features['bb_rsi_signal'] = features['bb_position'] * features['rsi']
            features['trend_strength'] = features['price_vs_ema9'] * features['price_vs_ema21']
            features['volatility_breakout'] = features['atr'] * features['bb_width']

            features['momentum_vol_signal'] = features['momentum_10'] * features['volume_ratio'] * features['volatility_ratio']
            features['trend_momentum_align'] = features['ma_alignment'] * features['momentum_10']
            features['pressure_volume_signal'] = features['net_pressure'] * features['volume_zscore']
            features['volatility_regime_signal'] = features['vol_regime'] * features['rsi']
            features['multi_timeframe_signal'] = features['price_change_1h'] * features['price_change_4h'] * features['price_change_24h']
            features['oscillator_consensus'] = (features['rsi_oversold'] + features['stoch_oversold']) - (features['rsi_overbought'] + features['stoch_overbought'])
            features['trend_regime'] = ((features['ma_alignment'] == 1) & (features['price_vs_sma200'] > 0)).astype(int)
            features['consolidation_regime'] = ((features['bb_width'] < features['bb_width'].rolling(50).quantile(0.3)) & (features['atr_ratio'] < features['atr_ratio'].rolling(50).quantile(0.3))).astype(int)

            # Assign all new features at once
            df = df.assign(**features)

            # Drop rows with NaN values only once at the end
            df = df.dropna()

            if len(df) < 30:
                self.logger.warning(f"Insufficient data after feature engineering and NaN removal. Remaining rows: {len(df)}")
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