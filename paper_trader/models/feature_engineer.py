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

# Sequence length used during model training
LSTM_SEQUENCE_LENGTH: int = 96

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

            # Calculate all features in one go using a dictionary
            features = {}

            # Basic price features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['price_change_1h'] = df['close'].pct_change(4)
            features['price_change_4h'] = df['close'].pct_change(16)
            features['price_change_24h'] = df['close'].pct_change(96)

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
            features['volume_zscore'] = (df['volume'] - volume_sma_20) / volume_std_20
            features['volume_price_trend'] = df['volume'] * features['returns']
            features['volume_weighted_price'] = (
                (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
            )

            # Market microstructure
            features['spread'] = (df['high'] - df['low']) / df['close']

            # Volatility features
            features['volatility_20'] = features['returns'].rolling(20).std()
            features['volatility_5'] = features['returns'].rolling(5).std()
            features['volatility_50'] = features['returns'].rolling(50).std()
            features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
            features['realized_vol_5'] = features['returns'].rolling(5).std() * np.sqrt(5)
            features['realized_vol_20'] = features['returns'].rolling(20).std() * np.sqrt(20)

            # Technical indicators using pandas_ta
            df_temp = pd.concat([df, pd.DataFrame(features)], axis=1)
            
            # RSI
            rsi_values = ta.rsi(df['close'], length=14)
            features['rsi'] = rsi_values
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
            features['atr_ratio'] = features['atr'] / df['close']

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
            features['roc_10'] = ta.roc(df['close'], length=10)

            # Candle patterns
            features['candle_body'] = abs(df['close'] - df['open']) / df['open']
            features['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
            features['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']

            # Time features
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

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
            complex_features['trend_regime'] = ((features['ma_alignment'] == 1) & (features['price_vs_sma200'] > 0)).astype(int)
            
            # BB width rolling calculation
            bb_width_rolling = features['bb_width'].rolling(50)
            atr_ratio_rolling = features['atr_ratio'].rolling(50)
            complex_features['consolidation_regime'] = (
                (features['bb_width'] < bb_width_rolling.quantile(0.3)) & 
                (features['atr_ratio'] < atr_ratio_rolling.quantile(0.3))
            ).astype(int)

            # Combine all features
            all_features = {**features, **complex_features}

            # Assign all features at once to avoid fragmentation
            df_final = df.assign(**all_features)

            # Drop rows with NaN values
            df_final = df_final.dropna()

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
