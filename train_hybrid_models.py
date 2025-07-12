import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_cuda_graphs=false"

#!/usr/bin/env python3
"""
Hybrid LSTM + XGBoost Model Training Script with Walk-Forward Analysis
====================================================================

This script implements a walk-forward training pipeline:
1. Generate time windows (train N months → test 1 month)
2. For each window:
   - Train LSTM on training data
   - Generate lstm_delta predictions
   - Train XGBoost with lstm_delta + technical features
   - Evaluate on test period
3. Aggregate performance metrics across all folds

Supported pairs: BTCEUR, ETHEUR, ADAEUR, SOLEUR, XRPEUR
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta
import pickle
import json
import argparse
from typing import Dict, Tuple, List

# ML Libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score, classification_report
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, GlobalAveragePooling1D, Dot
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import traceback

# Configure GPU for maximum utilization and performance
try:
    # Enable GPU memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Remove memory limit for maximum GPU utilization
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)]  # 15GB limit removed
            # )
        print(f"🚀 GPU Configuration: Found {len(gpus)} GPU(s), memory growth enabled with NO LIMIT for maximum utilization")
        
        # Disable mixed precision to reduce overhead and improve speed
        # from tensorflow.keras import mixed_precision
        # policy = mixed_precision.Policy('mixed_float16')
        # mixed_precision.set_global_policy(policy)
        print("⚡ Mixed precision DISABLED for maximum speed (using float32)")
        
        # Additional GPU optimizations
        tf.config.optimizer.set_jit(True)  # Enable XLA compilation
        
        # Disable CUDA graphs to prevent graph execution failures
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_graph_level=0'
        
        # Enable TF32 for better performance (if available)
        try:
            tf.config.experimental.enable_tensor_float_32()
            print("🔥 Advanced GPU optimizations: XLA compilation, TF32 enabled, CUDA graphs disabled")
        except AttributeError:
            # TF32 not available in this TensorFlow version
            print("🔥 Advanced GPU optimizations: XLA compilation enabled, CUDA graphs disabled (TF32 not available)")
        except Exception as tf32_error:
            print(f"🔥 Advanced GPU optimizations: XLA compilation enabled, CUDA graphs disabled (TF32 error: {tf32_error})")
    else:
        print("💻 No GPU detected, using CPU")
except Exception as e:
    print(f"⚠️ GPU configuration warning: {e}")

# Technical Analysis
import pandas_ta as ta

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class HybridModelTrainer:
    """
    Hybrid LSTM + XGBoost Model Trainer for Cryptocurrency Trading
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models", 
                 train_months: int = 3, test_months: int = 1, step_months: int = 1,
                 symbols: List[str] = None):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.symbols = symbols or ['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR']
        
        # Walk-forward parameters
        self.train_months = train_months    # Training window size in months
        self.test_months = test_months      # Test window size in months
        self.step_months = step_months      # Step size for rolling window
        self.min_training_samples = 5000    # Minimum samples for training
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(f"{self.models_dir}/lstm", exist_ok=True)
        os.makedirs(f"{self.models_dir}/xgboost", exist_ok=True)
        os.makedirs(f"{self.models_dir}/scalers", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/feature_importance", exist_ok=True)
        
        # Enhanced Model parameters
        self.lstm_sequence_length = 96      # 24 hours of 15-min candles (increased from 60)
        self.prediction_horizon = 1         # Next 15-min candle
        self.price_change_threshold = 0.001 # 0.1% for binary classification (more sensitive)
        
        # Advanced LSTM parameters (optimized for better GPU utilization)
        self.lstm_units = [256, 128, 64]    # Increased units for better GPU utilization
        self.dropout_rate = 0.3             # Increased dropout for better regularization
        self.attention_units = 128          # Increased attention units for more computation
        self.use_attention = True           # Enable attention mechanism
        
        # Optimized XGBoost parameters for speed and performance balance
        self.xgb_params = {
            'n_estimators': 300,            # Reduced for faster training
            'max_depth': 6,                 # Reduced depth for faster training
            'learning_rate': 0.1,           # Increased learning rate for faster convergence
            'subsample': 0.8,               # Standard subsampling
            'colsample_bytree': 0.8,        # Standard feature sampling
            'reg_alpha': 0.1,               # Standard L1 regularization
            'reg_lambda': 1.0,              # Standard L2 regularization
            'min_child_weight': 1,          # Standard setting
            'gamma': 0,                     # No minimum split loss for speed
            'early_stopping_rounds': 30,    # Reduced patience for speed
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'random_state': 42,
            'n_jobs': -1,                   # Use all CPU cores
            'nthread': -1,                  # Explicitly use all threads
            'tree_method': 'hist',          # Fastest training method
            'grow_policy': 'depthwise',     # Optimized growth policy
            'max_leaves': 256,              # Reduced for faster training
            'verbosity': 0,                 # Reduce output
            'enable_categorical': False,    # Optimize for numerical features
            'predictor': 'cpu_predictor'    # Optimized CPU prediction
        }
        
        print("🚀 Hybrid LSTM + XGBoost Model Trainer with Walk-Forward Analysis Initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🤖 Models directory: {self.models_dir}")
        print(f"💰 Symbols: {', '.join(self.symbols)}")
        print(f"📊 Walk-Forward Config: Train {self.train_months}m → Test {self.test_months}m (Step: {self.step_months}m)")
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load OHLCV data from SQLite database
        """
        db_path = f"{self.data_dir}/{symbol.lower()}_15m.db"
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"📊 Loaded {len(df):,} candles for {symbol}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators with advanced features
        """
        data = df.copy()
        
        # Enhanced Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['price_change_1h'] = data['close'].pct_change(4)  # 4 * 15min = 1h
        data['price_change_4h'] = data['close'].pct_change(16) # 16 * 15min = 4h
        data['price_change_24h'] = data['close'].pct_change(96) # 96 * 15min = 24h
        
        # Price normalization features
        data['price_zscore_20'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        data['price_zscore_50'] = (data['close'] - data['close'].rolling(50).mean()) / data['close'].rolling(50).std()
        
        # Lag features for returns (important for prediction)
        for lag in [1, 2, 3, 5, 10, 20]:
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
            data[f'log_returns_lag_{lag}'] = data['log_returns'].shift(lag)
        
        # Rolling statistics for returns
        data['returns_mean_10'] = data['returns'].rolling(10).mean()
        data['returns_std_10'] = data['returns'].rolling(10).std()
        data['returns_skew_20'] = data['returns'].rolling(20).skew()
        data['returns_kurt_20'] = data['returns'].rolling(20).kurt()
        
        # Enhanced Volume features
        data['volume_sma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        data['volume_change'] = data['volume'].pct_change()
        data['volume_zscore'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
        
        # Volume-Price relationship
        data['volume_price_trend'] = data['volume'] * data['returns']
        data['volume_weighted_price'] = (data['volume'] * data['close']).rolling(20).sum() / data['volume'].rolling(20).sum()
        
        # Market microstructure features
        data['spread'] = (data['high'] - data['low']) / data['close']
        data['spread_ma'] = data['spread'].rolling(20).mean()
        data['spread_ratio'] = data['spread'] / data['spread_ma']
        
        # Order flow approximation
        data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        data['selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
        data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
        
        # Enhanced Volatility features
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['volatility_50'] = data['returns'].rolling(50).std()
        data['volatility_ratio'] = data['volatility_20'] / data['volatility_50']
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
        data['atr_ratio'] = data['atr'] / data['close']
        
        # Realized volatility (different timeframes)
        data['realized_vol_5'] = data['returns'].rolling(5).std() * np.sqrt(5)
        data['realized_vol_20'] = data['returns'].rolling(20).std() * np.sqrt(20)
        
        # Volatility clustering
        data['vol_regime'] = (data['volatility_20'] > data['volatility_20'].rolling(100).quantile(0.75)).astype(int)
        
        # Enhanced Moving Averages
        data['ema_9'] = ta.ema(data['close'], length=9)
        data['ema_21'] = ta.ema(data['close'], length=21)
        data['ema_50'] = ta.ema(data['close'], length=50)
        data['ema_100'] = ta.ema(data['close'], length=100)
        data['sma_200'] = ta.sma(data['close'], length=200)
        
        # Price relative to MAs
        data['price_vs_ema9'] = (data['close'] - data['ema_9']) / data['ema_9']
        data['price_vs_ema21'] = (data['close'] - data['ema_21']) / data['ema_21']
        data['price_vs_ema50'] = (data['close'] - data['ema_50']) / data['ema_50']
        data['price_vs_sma200'] = (data['close'] - data['sma_200']) / data['sma_200']
        
        # MA crossovers and trends
        data['ema9_vs_ema21'] = (data['ema_9'] - data['ema_21']) / data['ema_21']
        data['ema21_vs_ema50'] = (data['ema_21'] - data['ema_50']) / data['ema_50']
        data['ema50_vs_ema100'] = (data['ema_50'] - data['ema_100']) / data['ema_100']
        
        # Trend strength indicators
        data['ma_alignment'] = ((data['ema_9'] > data['ema_21']) & 
                               (data['ema_21'] > data['ema_50']) & 
                               (data['ema_50'] > data['ema_100'])).astype(int)
        data['ma_slope_9'] = data['ema_9'].pct_change(5)
        data['ma_slope_21'] = data['ema_21'].pct_change(5)
        
        # Enhanced Oscillators
        data['rsi'] = ta.rsi(data['close'], length=14)
        data['rsi_9'] = ta.rsi(data['close'], length=9)
        data['rsi_21'] = ta.rsi(data['close'], length=21)
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
        data['rsi_divergence'] = data['rsi'].diff(5) * data['close'].pct_change(5)
        
        # Stochastic oscillator
        stoch = ta.stoch(data['high'], data['low'], data['close'])
        data['stoch_k'] = stoch['STOCHk_14_3_3']
        data['stoch_d'] = stoch['STOCHd_14_3_3']
        data['stoch_oversold'] = (data['stoch_k'] < 20).astype(int)
        data['stoch_overbought'] = (data['stoch_k'] > 80).astype(int)
        
        # Williams %R
        data['williams_r'] = ta.willr(data['high'], data['low'], data['close'], length=14)
        
        # MACD
        macd_data = ta.macd(data['close'])
        data['macd'] = macd_data['MACD_12_26_9']
        data['macd_signal'] = macd_data['MACDs_12_26_9']
        data['macd_histogram'] = macd_data['MACDh_12_26_9']
        data['macd_bullish'] = (data['macd'] > data['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_data = ta.bbands(data['close'], length=20, std=2)
        data['bb_upper'] = bb_data['BBU_20_2.0']
        data['bb_lower'] = bb_data['BBL_20_2.0']
        data['bb_middle'] = bb_data['BBM_20_2.0']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # VWAP
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['price_vs_vwap'] = (data['close'] - data['vwap']) / data['vwap']
        
        # Candle patterns
        data['candle_body'] = abs(data['close'] - data['open']) / data['open']
        data['upper_wick'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['open']
        data['lower_wick'] = (np.minimum(data['open'], data['close']) - data['low']) / data['open']
        
        # Time-based features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # Momentum indicators
        data['momentum_10'] = ta.mom(data['close'], length=10)
        data['roc_10'] = ta.roc(data['close'], length=10)
        
        # Support/Resistance levels
        data['high_20'] = data['high'].rolling(20).max()
        data['low_20'] = data['low'].rolling(20).min()
        data['near_resistance'] = (data['close'] / data['high_20'] > 0.98).astype(int)
        data['near_support'] = (data['close'] / data['low_20'] < 1.02).astype(int)
        
        # Enhanced Feature Interactions (combined signals)
        data['rsi_macd_combo'] = data['rsi'] * data['macd_signal']
        data['volatility_ema_ratio'] = data['volatility_20'] / data['ema_21']
        data['volume_price_momentum'] = data['volume_ratio'] * data['momentum_10']
        data['bb_rsi_signal'] = data['bb_position'] * data['rsi']
        data['trend_strength'] = data['price_vs_ema9'] * data['price_vs_ema21']
        data['volatility_breakout'] = data['atr'] * data['bb_width']
        
        # Advanced interaction features
        data['momentum_vol_signal'] = data['momentum_10'] * data['volume_ratio'] * data['volatility_ratio']
        data['trend_momentum_align'] = data['ma_alignment'] * data['momentum_10']
        data['pressure_volume_signal'] = data['net_pressure'] * data['volume_zscore']
        data['volatility_regime_signal'] = data['vol_regime'] * data['rsi']
        data['multi_timeframe_signal'] = data['price_change_1h'] * data['price_change_4h'] * data['price_change_24h']
        data['oscillator_consensus'] = (data['rsi_oversold'] + data['stoch_oversold']) - (data['rsi_overbought'] + data['stoch_overbought'])
        
        # Market regime features
        data['trend_regime'] = ((data['ma_alignment'] == 1) & (data['price_vs_sma200'] > 0)).astype(int)
        data['consolidation_regime'] = ((data['bb_width'] < data['bb_width'].rolling(50).quantile(0.3)) & 
                                       (data['atr_ratio'] < data['atr_ratio'].rolling(50).quantile(0.3))).astype(int)
        
        print(f"✅ Created {len([col for col in data.columns if col not in df.columns])} technical features (including interactions)")
        
        return data
    
    def generate_walk_forward_windows(self, df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate walk-forward time windows for training and testing
        Returns list of (train_start, train_end, test_end) tuples
        """
        windows = []
        start_date = df.index.min()
        end_date = df.index.max()
        
        # Start from train_months after the beginning to have enough training data
        current_date = start_date + pd.DateOffset(months=self.train_months)
        
        while current_date + pd.DateOffset(months=self.test_months) <= end_date:
            train_start = current_date - pd.DateOffset(months=self.train_months)
            train_end = current_date
            test_end = current_date + pd.DateOffset(months=self.test_months)
            
            windows.append((train_start, train_end, test_end))
            current_date += pd.DateOffset(months=self.step_months)
        
        print(f"📅 Generated {len(windows)} walk-forward windows")
        return windows
    
    def prepare_lstm_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare enhanced sequences for LSTM training with multiple features
        """
        # Enhanced feature set for LSTM (normalized and scaled)
        lstm_features = [
            'close', 'volume', 'returns', 'log_returns',
            'volatility_20', 'atr_ratio', 'rsi', 'macd', 'bb_position',
            'volume_ratio', 'price_vs_ema9', 'price_vs_ema21',
            'buying_pressure', 'selling_pressure', 'spread_ratio',
            'momentum_10', 'price_zscore_20'
        ]
        
        # Ensure all features exist
        available_features = [f for f in lstm_features if f in df.columns]
        if len(available_features) < len(lstm_features):
            missing = set(lstm_features) - set(available_features)
            print(f"⚠️  Missing LSTM features: {missing}")
        
        # Use available features
        feature_data = df[available_features].values
        
        # Create target: next period price change % with validation
        targets = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon).values
        
        # Validate targets before creating sequences
        valid_target_mask = ~(np.isnan(targets) | np.isinf(targets))
        print(f"📊 Target validation: {valid_target_mask.sum()}/{len(targets)} valid targets")
        
        # Create sequences and track valid indices
        X, y, valid_indices = [], [], []
        
        for i in range(self.lstm_sequence_length, len(feature_data) - self.prediction_horizon):
            # Check if current sequence and target are valid
            sequence = feature_data[i-self.lstm_sequence_length:i]
            target = targets[i]
            
            # Validate sequence (no NaN or inf)
            if not (np.isnan(sequence).any() or np.isinf(sequence).any()) and not (np.isnan(target) or np.isinf(target)):
                X.append(sequence)
                y.append(target)
                valid_indices.append(i)
        
        X = np.array(X)
        y = np.array(y)
        valid_indices = np.array(valid_indices)
        
        # Additional validation and clipping for extreme values
        if len(y) > 0:
            # Clip extreme target values to prevent training instability
            y_std = np.std(y)
            y_mean = np.mean(y)
            clip_threshold = 3 * y_std  # 3 standard deviations
            
            original_count = len(y)
            extreme_mask = np.abs(y - y_mean) <= clip_threshold
            X = X[extreme_mask]
            y = y[extreme_mask]
            valid_indices = valid_indices[extreme_mask]
            
            if len(y) < original_count:
                print(f"📉 Removed {original_count - len(y)} extreme target values (>{clip_threshold:.6f})")
            
            print(f"📊 Target statistics: mean={y_mean:.6f}, std={y_std:.6f}, range=[{y.min():.6f}, {y.max():.6f}]")
        
        # Final validation
        if len(X) == 0 or len(y) == 0:
            print("⚠️  Warning: No valid sequences created for LSTM training")
            return np.array([]), np.array([]), np.array([])
        
        # Get corresponding timestamps for alignment using valid indices
        timestamps = df.index[valid_indices]
        
        print(f"📊 LSTM sequences: {X.shape}, targets: {y.shape}")
        
        return X, y, timestamps
    
    def create_attention_layer(self, lstm_output, attention_units=64):
        """
        Create attention mechanism for LSTM
        """
        from tensorflow.keras.layers import Dense, Activation, Dot, Concatenate
        
        # Attention mechanism
        attention = Dense(attention_units, activation='tanh')(lstm_output)
        attention = Dense(1, activation='softmax')(attention)
        
        # Apply attention weights
        context = Dot(axes=1)([attention, lstm_output])
        
        return context
    
    def directional_loss(self, y_true, y_pred):
        """
        Custom loss function that penalizes wrong directional predictions more
        """
        import tensorflow as tf
        
        # Standard MSE
        mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        
        # Directional penalty
        direction_true = tf.sign(y_true)
        direction_pred = tf.sign(y_pred)
        direction_penalty = tf.where(
            tf.equal(direction_true, direction_pred),
            0.0,
            tf.abs(y_true - y_pred) * 2.0  # Double penalty for wrong direction
        )
        
        return mse + tf.reduce_mean(direction_penalty)
    
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """
        Train enhanced LSTM model with attention mechanism
        """
        print(f"🧠 Training Enhanced LSTM model with attention...")
        
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Add
        from tensorflow.keras.models import Model
        try:
            from tensorflow.keras.optimizers import AdamW
        except ImportError:
            # Fallback to Adam if AdamW is not available
            from tensorflow.keras.optimizers import Adam as AdamW
        # CosineRestartScheduler is not available in standard TensorFlow
        # Using LearningRateScheduler instead for cosine annealing
        
        # Input layer
        inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
        
        # Multi-layer LSTM with residual connections
        x = inputs
        
        # First LSTM layer
        lstm1 = LSTM(self.lstm_units[0], return_sequences=True, dropout=self.dropout_rate, 
                    recurrent_dropout=self.dropout_rate)(x)
        lstm1 = BatchNormalization()(lstm1)
        
        # Second LSTM layer with residual connection
        lstm2 = LSTM(self.lstm_units[1], return_sequences=True, dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Third LSTM layer
        lstm3 = LSTM(self.lstm_units[2], return_sequences=True, dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate)(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        
        # Attention mechanism (if enabled)
        if self.use_attention:
            # Self-attention
            attention_output = self.create_attention_layer(lstm3, self.attention_units)
            x = attention_output
        else:
            # Global average pooling
            from tensorflow.keras.layers import GlobalAveragePooling1D
            x = GlobalAveragePooling1D()(lstm3)
        
        # Dense layers with residual connections
        dense1 = Dense(64, activation='relu')(x)
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense1 = BatchNormalization()(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(self.dropout_rate)(dense2)
        dense2 = BatchNormalization()(dense2)
        
        # Output layer (using float32 for maximum performance)
        outputs = Dense(1, activation='linear')(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Enhanced optimizer with weight decay (if available)
        try:
            optimizer = AdamW(
                learning_rate=0.001,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999
            )
        except TypeError:
            # Fallback to Adam without weight_decay if AdamW is not available
            optimizer = AdamW(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999
            )
        
        # Compile with custom loss
        model.compile(
            optimizer=optimizer,
            loss=self.directional_loss,
            metrics=['mae', 'mse']
        )
        
        # Optimized callbacks for faster training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        ]
        
        # Train model with conservative memory management to prevent crashes
        # Start with moderate batch size and implement robust fallback
        batch_size = 1024  # Conservative starting point to prevent allocation failures
        
        # Enable XLA compilation for additional performance
        model.compile(
            optimizer=optimizer,
            loss=self.directional_loss,
            metrics=['mae', 'mse'],
            jit_compile=True  # Enable XLA compilation
        )
        
        # Implement robust batch size fallback with memory clearing
        batch_sizes = [1024, 512, 256, 128, 64]  # Conservative progression
        history = None
        
        for batch_size in batch_sizes:
            try:
                # Clear any existing GPU memory
                tf.keras.backend.clear_session()
                
                # Create optimized TensorFlow datasets
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                print(f"🚀 Attempting training with batch size: {batch_size}")
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=100,  # Reduced epochs for faster training
                    callbacks=callbacks,
                    verbose=0
                )
                print(f"✅ LSTM training completed successfully with batch size {batch_size}")
                break  # Success, exit the loop
                
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                print(f"⚠️ Memory error with batch_size={batch_size}: {str(e)[:100]}...")
                # Clear memory before trying next batch size
                tf.keras.backend.clear_session()
                if batch_size == batch_sizes[-1]:  # Last attempt
                    raise RuntimeError(f"Unable to train LSTM model - all batch sizes failed. Last error: {e}")
                continue
        
        if history is None:
            raise RuntimeError("LSTM training failed - no successful batch size found")
        
        print(f"✅ LSTM training completed. Best val_loss: {min(history.history['val_loss']):.6f}")
        
        return model
    
    
def generate_lstm_predictions(self, model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    print("🚀 Generating predictions with batch_size=1024")
    preds = model.predict(X, batch_size=1024, verbose=0)
    return preds.flatten()
    main()