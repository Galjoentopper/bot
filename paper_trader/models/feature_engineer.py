"""Feature engineering for technical analysis and model input preparation."""

import logging
import numpy as np
import pandas as pd
from typing import Optional
import pandas_ta as ta
# import talib  # Commented out - not available on Windows without special installation

class FeatureEngineer:
    """Creates technical indicators and features for ML models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create comprehensive technical features from OHLCV data."""
        try:
            if len(data) < 50:
                self.logger.warning("Insufficient data for feature engineering")
                return None
                
            df = data.copy()
            
            # Ensure proper column names
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_change'] = df['close'] - df['open']
            df['price_range'] = df['high'] - df['low']
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Volume features
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
            df['price_volume'] = df['close'] * df['volume']
            df['vwap'] = (df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum())
            
            # Moving Averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
                df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
            
            # RSI
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            df['rsi_7'] = ta.rsi(df['close'], length=7)
            df['rsi_21'] = ta.rsi(df['close'], length=21)
            
            # MACD
            macd_data = ta.macd(df['close'])
            if macd_data is not None:
                df['macd'] = macd_data['MACD_12_26_9']
                df['macd_signal'] = macd_data['MACDs_12_26_9']
                df['macd_histogram'] = macd_data['MACDh_12_26_9']
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20)
            if bb_data is not None:
                df['bb_upper'] = bb_data['BBU_20_2.0']
                df['bb_middle'] = bb_data['BBM_20_2.0']
                df['bb_lower'] = bb_data['BBL_20_2.0']
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            stoch_data = ta.stoch(df['high'], df['low'], df['close'])
            if stoch_data is not None:
                df['stoch_k'] = stoch_data['STOCHk_14_3_3']
                df['stoch_d'] = stoch_data['STOCHd_14_3_3']
            
            # ATR (Average True Range)
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_7'] = ta.atr(df['high'], df['low'], df['close'], length=7)
            
            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            
            # Commodity Channel Index
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # Money Flow Index
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            
            # ADX
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_data is not None:
                df['adx'] = adx_data['ADX_14']
            
            # On Balance Volume
            df['obv'] = ta.obv(df['close'], df['volume'])
            df['obv_sma'] = df['obv'].rolling(10).mean()
            
            # Volatility features
            df['volatility_10'] = df['returns'].rolling(10).std()
            df['volatility_20'] = df['returns'].rolling(20).std()
            
            # Support and Resistance levels
            df['support'] = df['low'].rolling(20).min()
            df['resistance'] = df['high'].rolling(20).max()
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
            
            # Trend features
            df['trend_5'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
            df['trend_10'] = np.where(df['close'] > df['close'].shift(10), 1, -1)
            df['trend_20'] = np.where(df['close'] > df['close'].shift(20), 1, -1)
            
            # Price momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Rate of Change
            df['roc_5'] = ta.roc(df['close'], length=5)
            df['roc_10'] = ta.roc(df['close'], length=10)
            
            # Time-based features
            if df.index.dtype == 'datetime64[ns]':
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.dayofweek
                df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            
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