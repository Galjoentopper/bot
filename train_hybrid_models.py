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
import random

# Disable CUDA graphs to prevent CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
# Commenting out XLA flag that causes 'Unknown flags' error
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_cuda_graphs=false"

import sqlite3
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta
import pickle
import json
import argparse
from typing import Dict, Tuple, List, Optional

# ML Libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    BatchNormalization,
    GlobalAveragePooling1D,
    Dot,
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import traceback

# Configure GPU for maximum utilization and performance
try:
    # Disable CUDA graphs and set conservative memory allocator
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    # Remove problematic XLA flag that causes 'Unknown flags' error
    # os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_graph_level=0'

    # Enable GPU memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Remove memory limit for maximum GPU utilization
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)]  # 15GB limit removed
            # )
        print(
            f"🚀 GPU Configuration: Found {len(gpus)} GPU(s), memory growth enabled with CUDA graphs DISABLED"
        )

        # Disable mixed precision to reduce overhead and improve speed
        # from tensorflow.keras import mixed_precision
        # policy = mixed_precision.Policy('mixed_float16')
        # mixed_precision.set_global_policy(policy)
        print("⚡ Mixed precision DISABLED for maximum speed (using float32)")

        # Conservative GPU optimizations (disable XLA to prevent graph errors)
        # tf.config.optimizer.set_jit(True)  # Disable XLA compilation to prevent CUDA graph errors

        # Enable TF32 for better performance (if available)
        try:
            tf.config.experimental.enable_tensor_float_32()
            print(
                "🔥 Conservative GPU optimizations: CUDA graphs disabled, TF32 enabled"
            )
        except AttributeError:
            # TF32 not available in this TensorFlow version
            print(
                "🔥 Conservative GPU optimizations: CUDA graphs disabled (TF32 not available)"
            )
        except Exception as tf32_error:
            print(
                f"🔥 Conservative GPU optimizations: CUDA graphs disabled (TF32 error: {tf32_error})"
            )
    else:
        print("💻 No GPU detected, using CPU")
except Exception as e:
    print(f"⚠️ GPU configuration warning: {e}")

# Technical Analysis
import pandas_ta as ta

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Quantile loss for probabilistic forecasting
from tensorflow.keras import backend as K


def quantile_loss(q):
    """Return a quantile loss function configured for quantile ``q``."""

    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

    return loss


# Standalone directional loss function for proper serialization
@tf.keras.utils.register_keras_serializable(package="Custom", name="directional_loss")
def directional_loss(y_true, y_pred):
    """
    Custom loss function that penalizes wrong directional predictions more
    """
    # Standard MSE
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

    # Directional penalty
    direction_true = tf.sign(y_true)
    direction_pred = tf.sign(y_pred)
    direction_penalty = tf.where(
        tf.equal(direction_true, direction_pred),
        0.0,
        tf.abs(y_true - y_pred) * 2.0,  # Double penalty for wrong direction
    )

    return mse + tf.reduce_mean(direction_penalty)


class HybridModelTrainer:
    """
    Hybrid LSTM + XGBoost Model Trainer for Cryptocurrency Trading
    """

    def __init__(
        self,
        data_dir: str = "data",
        models_dir: str = "models",
        train_months: int = 12,
        test_months: int = 1,
        step_months: int = 1,
        symbols: List[str] | None = None,
        seed: int = 42,
        warm_start: bool = False,
    ):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.symbols = symbols or ["BTCEUR", "ETHEUR", "ADAEUR", "SOLEUR", "XRPEUR"]
        self.seed = seed
        self.warm_start = warm_start

        # Ensure reproducibility
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Walk-forward parameters
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.min_training_samples = 5000  # Minimum samples for training

        # Purged walk-forward parameters
        self.purge_candles = 200
        self.embargo_candles = 96

        # Quantile for quantile loss
        self.quantile = 0.5

        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(f"{self.models_dir}/lstm", exist_ok=True)
        os.makedirs(f"{self.models_dir}/xgboost", exist_ok=True)
        os.makedirs(f"{self.models_dir}/scalers", exist_ok=True)
        os.makedirs(f"{self.models_dir}/feature_columns", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/feature_importance", exist_ok=True)

        # Enhanced Model parameters
        self.lstm_sequence_length = 96  # 24 hours of 15-min candles (increased from 60)
        self.prediction_horizon = 1  # Next 15-min candle
        self.price_change_threshold = (
            0.001  # 0.1% for binary classification (more sensitive)
        )

        # Advanced LSTM parameters (optimized for better GPU utilization)
        self.lstm_units = [256, 128, 64]  # Increased units for better GPU utilization
        self.dropout_rate = 0.3  # Increased dropout for better regularization
        self.attention_units = 128  # Increased attention units for more computation
        self.use_attention = True  # Enable attention mechanism

        # Optimized XGBoost parameters for speed and performance balance
        self.xgb_params = {
            "n_estimators": 300,  # Reduced for faster training
            "max_depth": 6,  # Reduced depth for faster training
            "learning_rate": 0.1,  # Increased learning rate for faster convergence
            "subsample": 0.8,  # Standard subsampling
            "colsample_bytree": 0.8,  # Standard feature sampling
            "reg_alpha": 0.1,  # Standard L1 regularization
            "reg_lambda": 1.0,  # Standard L2 regularization
            "min_child_weight": 1,  # Standard setting
            "gamma": 0,  # No minimum split loss for speed
            "early_stopping_rounds": 30,  # Reduced patience for speed
            "eval_metric": "logloss",
            "objective": "binary:logistic",
            "random_state": 42,
            "n_jobs": -1,  # Use all CPU cores
            "nthread": -1,  # Explicitly use all threads
            "tree_method": "hist",  # Fastest training method
            "grow_policy": "depthwise",  # Optimized growth policy
            "max_leaves": 256,  # Reduced for faster training
            "verbosity": 0,  # Reduce output
            "enable_categorical": False,  # Optimize for numerical features
            "predictor": "cpu_predictor",  # Optimized CPU prediction
        }

        print(
            "🚀 Hybrid LSTM + XGBoost Model Trainer with Walk-Forward Analysis Initialized"
        )
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🤖 Models directory: {self.models_dir}")
        print(f"💰 Symbols: {', '.join(self.symbols)}")
        print(
            f"📊 Walk-Forward Config: Train {self.train_months}m → Test {self.test_months}m (Step: {self.step_months}m)"
        )

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
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        print(f"📊 Loaded {len(df):,} candles for {symbol}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")

        return df

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators with advanced features
        """
        data = df.copy()

        # Enhanced Price-based features
        data["returns"] = data["close"].pct_change()
        data["log_returns"] = np.log(data["close"] / data["close"].shift(1))
        data["price_change_1h"] = data["close"].pct_change(4)  # 4 * 15min = 1h
        data["price_change_4h"] = data["close"].pct_change(16)  # 16 * 15min = 4h
        data["price_change_24h"] = data["close"].pct_change(96)  # 96 * 15min = 24h

        # Price normalization features
        data["price_zscore_20"] = (
            data["close"] - data["close"].rolling(20).mean()
        ) / data["close"].rolling(20).std()
        data["price_zscore_50"] = (
            data["close"] - data["close"].rolling(50).mean()
        ) / data["close"].rolling(50).std()

        # Lag features for returns (important for prediction)
        for lag in [1, 2, 3, 5, 10, 20]:
            data[f"returns_lag_{lag}"] = data["returns"].shift(lag)
            data[f"log_returns_lag_{lag}"] = data["log_returns"].shift(lag)

        # Rolling statistics for returns
        data["returns_mean_10"] = data["returns"].rolling(10).mean()
        data["returns_std_10"] = data["returns"].rolling(10).std()
        data["returns_skew_20"] = data["returns"].rolling(20).skew()
        data["returns_kurt_20"] = data["returns"].rolling(20).kurt()

        # Enhanced Volume features
        data["volume_sma_20"] = data["volume"].rolling(20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma_20"]
        data["volume_change"] = data["volume"].pct_change()
        data["volume_zscore"] = (
            data["volume"] - data["volume"].rolling(20).mean()
        ) / data["volume"].rolling(20).std()

        # Volume-Price relationship
        data["volume_price_trend"] = data["volume"] * data["returns"]
        data["volume_weighted_price"] = (data["volume"] * data["close"]).rolling(
            20
        ).sum() / data["volume"].rolling(20).sum()

        # Market microstructure features
        data["spread"] = (data["high"] - data["low"]) / data["close"]
        data["spread_ma"] = data["spread"].rolling(20).mean()
        data["spread_ratio"] = data["spread"] / data["spread_ma"]

        # Order flow approximation
        data["buying_pressure"] = (data["close"] - data["low"]) / (
            data["high"] - data["low"]
        )
        data["selling_pressure"] = (data["high"] - data["close"]) / (
            data["high"] - data["low"]
        )
        data["net_pressure"] = data["buying_pressure"] - data["selling_pressure"]

        # Enhanced Volatility features
        data["volatility_20"] = data["returns"].rolling(20).std()
        data["volatility_50"] = data["returns"].rolling(50).std()
        data["volatility_ratio"] = data["volatility_20"] / data["volatility_50"]
        data["atr"] = ta.atr(data["high"], data["low"], data["close"], length=14)
        data["atr_ratio"] = data["atr"] / data["close"]

        # Realized volatility (different timeframes)
        data["realized_vol_5"] = data["returns"].rolling(5).std() * np.sqrt(5)
        data["realized_vol_20"] = data["returns"].rolling(20).std() * np.sqrt(20)

        # Volatility clustering
        data["vol_regime"] = (
            data["volatility_20"] > data["volatility_20"].rolling(100).quantile(0.75)
        ).astype(int)

        # Enhanced Moving Averages
        data["ema_9"] = ta.ema(data["close"], length=9)
        data["ema_21"] = ta.ema(data["close"], length=21)
        data["ema_50"] = ta.ema(data["close"], length=50)
        data["ema_100"] = ta.ema(data["close"], length=100)
        data["sma_200"] = ta.sma(data["close"], length=200)

        # Price relative to MAs
        data["price_vs_ema9"] = (data["close"] - data["ema_9"]) / data["ema_9"]
        data["price_vs_ema21"] = (data["close"] - data["ema_21"]) / data["ema_21"]
        data["price_vs_ema50"] = (data["close"] - data["ema_50"]) / data["ema_50"]
        data["price_vs_sma200"] = (data["close"] - data["sma_200"]) / data["sma_200"]

        # MA crossovers and trends
        data["ema9_vs_ema21"] = (data["ema_9"] - data["ema_21"]) / data["ema_21"]
        data["ema21_vs_ema50"] = (data["ema_21"] - data["ema_50"]) / data["ema_50"]
        data["ema50_vs_ema100"] = (data["ema_50"] - data["ema_100"]) / data["ema_100"]

        # Trend strength indicators
        data["ma_alignment"] = (
            (data["ema_9"] > data["ema_21"])
            & (data["ema_21"] > data["ema_50"])
            & (data["ema_50"] > data["ema_100"])
        ).astype(int)
        data["ma_slope_9"] = data["ema_9"].pct_change(5)
        data["ma_slope_21"] = data["ema_21"].pct_change(5)

        # Enhanced Oscillators
        data["rsi"] = ta.rsi(data["close"], length=14)
        data["rsi_9"] = ta.rsi(data["close"], length=9)
        data["rsi_21"] = ta.rsi(data["close"], length=21)
        data["rsi_oversold"] = (data["rsi"] < 30).astype(int)
        data["rsi_overbought"] = (data["rsi"] > 70).astype(int)
        data["rsi_divergence"] = data["rsi"].diff(5) * data["close"].pct_change(5)

        # Stochastic oscillator
        stoch = ta.stoch(data["high"], data["low"], data["close"])
        data["stoch_k"] = stoch["STOCHk_14_3_3"]
        data["stoch_d"] = stoch["STOCHd_14_3_3"]
        data["stoch_oversold"] = (data["stoch_k"] < 20).astype(int)
        data["stoch_overbought"] = (data["stoch_k"] > 80).astype(int)

        # Williams %R
        data["williams_r"] = ta.willr(
            data["high"], data["low"], data["close"], length=14
        )

        # MACD
        macd_data = ta.macd(data["close"])
        data["macd"] = macd_data["MACD_12_26_9"]
        data["macd_signal"] = macd_data["MACDs_12_26_9"]
        data["macd_histogram"] = macd_data["MACDh_12_26_9"]
        data["macd_bullish"] = (data["macd"] > data["macd_signal"]).astype(int)

        # Bollinger Bands
        bb_data = ta.bbands(data["close"], length=20, std=2)
        data["bb_upper"] = bb_data["BBU_20_2.0"]
        data["bb_lower"] = bb_data["BBL_20_2.0"]
        data["bb_middle"] = bb_data["BBM_20_2.0"]
        data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]
        data["bb_position"] = (data["close"] - data["bb_lower"]) / (
            data["bb_upper"] - data["bb_lower"]
        )

        # VWAP
        data["vwap"] = (data["close"] * data["volume"]).cumsum() / data[
            "volume"
        ].cumsum()
        data["price_vs_vwap"] = (data["close"] - data["vwap"]) / data["vwap"]

        # Candle patterns
        data["candle_body"] = abs(data["close"] - data["open"]) / data["open"]
        data["upper_wick"] = (
            data["high"] - np.maximum(data["open"], data["close"])
        ) / data["open"]
        data["lower_wick"] = (
            np.minimum(data["open"], data["close"]) - data["low"]
        ) / data["open"]

        # Time-based features
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

        # Momentum indicators
        data["momentum_10"] = ta.mom(data["close"], length=10)
        data["roc_10"] = ta.roc(data["close"], length=10)

        # Support/Resistance levels
        data["high_20"] = data["high"].rolling(20).max()
        data["low_20"] = data["low"].rolling(20).min()
        data["near_resistance"] = (data["close"] / data["high_20"] > 0.98).astype(int)
        data["near_support"] = (data["close"] / data["low_20"] < 1.02).astype(int)

        # Enhanced Feature Interactions (combined signals)
        data["rsi_macd_combo"] = data["rsi"] * data["macd_signal"]
        data["volatility_ema_ratio"] = data["volatility_20"] / data["ema_21"]
        data["volume_price_momentum"] = data["volume_ratio"] * data["momentum_10"]
        data["bb_rsi_signal"] = data["bb_position"] * data["rsi"]
        data["trend_strength"] = data["price_vs_ema9"] * data["price_vs_ema21"]
        data["volatility_breakout"] = data["atr"] * data["bb_width"]

        # Advanced interaction features
        data["momentum_vol_signal"] = (
            data["momentum_10"] * data["volume_ratio"] * data["volatility_ratio"]
        )
        data["trend_momentum_align"] = data["ma_alignment"] * data["momentum_10"]
        data["pressure_volume_signal"] = data["net_pressure"] * data["volume_zscore"]
        data["volatility_regime_signal"] = data["vol_regime"] * data["rsi"]
        data["multi_timeframe_signal"] = (
            data["price_change_1h"] * data["price_change_4h"] * data["price_change_24h"]
        )
        data["oscillator_consensus"] = (
            data["rsi_oversold"] + data["stoch_oversold"]
        ) - (data["rsi_overbought"] + data["stoch_overbought"])

        # Market regime features
        data["trend_regime"] = (
            (data["ma_alignment"] == 1) & (data["price_vs_sma200"] > 0)
        ).astype(int)
        data["consolidation_regime"] = (
            (data["bb_width"] < data["bb_width"].rolling(50).quantile(0.3))
            & (data["atr_ratio"] < data["atr_ratio"].rolling(50).quantile(0.3))
        ).astype(int)

        print(
            f"✅ Created {len([col for col in data.columns if col not in df.columns])} technical features (including interactions)"
        )

        return data

    def generate_walk_forward_windows(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate purged walk-forward windows with embargo."""

        windows = []
        start = df.index.min()
        end = df.index.max()

        current_start = start
        purge = timedelta(minutes=15 * self.purge_candles)
        embargo = timedelta(minutes=15 * self.embargo_candles)

        while True:
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end + purge
            test_end = test_start + pd.DateOffset(months=self.test_months)

            if test_end > end:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current_start = test_end + embargo

        print(
            f"📅 Generated {len(windows)} walk-forward windows with purging and embargo"
        )
        return windows

    def prepare_lstm_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare enhanced sequences for LSTM training with multiple features
        """
        # Enhanced feature set for LSTM (normalized and scaled)
        lstm_features = [
            "close",
            "volume",
            "returns",
            "log_returns",
            "volatility_20",
            "atr_ratio",
            "rsi",
            "macd",
            "bb_position",
            "volume_ratio",
            "price_vs_ema9",
            "price_vs_ema21",
            "buying_pressure",
            "selling_pressure",
            "spread_ratio",
            "momentum_10",
            "price_zscore_20",
        ]

        # Ensure all features exist
        available_features = [f for f in lstm_features if f in df.columns]
        if len(available_features) < len(lstm_features):
            missing = set(lstm_features) - set(available_features)
            print(f"⚠️  Missing LSTM features: {missing}")

        # Use available features
        feature_data = df[available_features].values

        # Create target: next period price change % with validation
        targets = (
            df["close"]
            .pct_change(self.prediction_horizon)
            .shift(-self.prediction_horizon)
            .values
        )

        # Validate targets before creating sequences
        valid_target_mask = ~(np.isnan(targets) | np.isinf(targets))
        print(
            f"📊 Target validation: {valid_target_mask.sum()}/{len(targets)} valid targets"
        )

        # Create sequences and track valid indices
        X, y, valid_indices = [], [], []

        for i in range(
            self.lstm_sequence_length, len(feature_data) - self.prediction_horizon
        ):
            # Check if current sequence and target are valid
            sequence = feature_data[i - self.lstm_sequence_length : i]
            target = targets[i]

            # Validate sequence (no NaN or inf)
            if not (np.isnan(sequence).any() or np.isinf(sequence).any()) and not (
                np.isnan(target) or np.isinf(target)
            ):
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
                print(
                    f"📉 Removed {original_count - len(y)} extreme target values (>{clip_threshold:.6f})"
                )

            print(
                f"📊 Target statistics: mean={y_mean:.6f}, std={y_std:.6f}, range=[{y.min():.6f}, {y.max():.6f}]"
            )

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
        attention = Dense(attention_units, activation="tanh")(lstm_output)
        attention = Dense(1, activation="softmax")(attention)

        # Apply attention weights
        context = Dot(axes=1)([attention, lstm_output])

        return context

    def boruta_feature_selection(self, df: pd.DataFrame) -> List[str]:
        """Select important features using BorutaPy."""
        X = df.drop("target", axis=1)
        y = df["target"]
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=self.seed)
        selector = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=self.seed)
        selector.fit(X.values, y.values)
        selected = X.columns[selector.support_].tolist()
        print(f"✅ Boruta selected {len(selected)}/{X.shape[1]} features")
        return selected

    def get_directional_loss(self):
        """
        Return the standalone directional loss function for model compilation
        """
        return directional_loss

    def train_lstm_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tf.keras.Model:
        """
        Train enhanced LSTM model with attention mechanism
        """
        print(f"🧠 Training Enhanced LSTM model with attention...")

        from tensorflow.keras.layers import (
            Input,
            LSTM,
            Dense,
            Dropout,
            BatchNormalization,
            Add,
            Conv1D,
            MaxPooling1D,
        )
        from tensorflow.keras.models import Model

        try:
            from tensorflow.keras.optimizers import AdamW
        except ImportError:
            # Fallback to Adam if AdamW is not available
            from tensorflow.keras.optimizers import Adam as AdamW

        # Use a cosine decay with restarts learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001,
            first_decay_steps=10,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-4,
        )

        # Input layer
        inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

        # Convolutional feature extractor
        x = Conv1D(64, kernel_size=3, activation="relu", padding="causal")(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        # Multi-layer LSTM with residual connections
        lstm1 = LSTM(
            self.lstm_units[0],
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
        )(x)
        lstm1 = BatchNormalization()(lstm1)

        # Second LSTM layer with residual connection
        lstm2 = LSTM(
            self.lstm_units[1],
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
        )(lstm1)
        lstm2 = BatchNormalization()(lstm2)

        # Third LSTM layer
        lstm3 = LSTM(
            self.lstm_units[2],
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
        )(lstm2)
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
        dense1 = Dense(64, activation="relu")(x)
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense1 = BatchNormalization()(dense1)

        dense2 = Dense(32, activation="relu")(dense1)
        dense2 = Dropout(self.dropout_rate)(dense2)
        dense2 = BatchNormalization()(dense2)

        # Output layer (using float32 for maximum performance)
        outputs = Dense(1, activation="linear")(dense2)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Enhanced optimizer with weight decay (if available)
        try:
            optimizer = AdamW(
                learning_rate=lr_schedule,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
            )
        except TypeError:
            # Fallback to Adam without weight_decay if AdamW is not available
            optimizer = AdamW(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
            )

        # Compile with quantile loss
        model.compile(
            optimizer=optimizer,
            loss=quantile_loss(self.quantile),
            metrics=["mae", "mse"],
        )

        # Optimized callbacks for faster training
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                min_delta=1e-5,
            )
        ]

        # Only add ReduceLROnPlateau when the optimizer's learning rate is settable
        lr_param = getattr(optimizer, "_learning_rate", optimizer.learning_rate)
        if not isinstance(lr_param, tf.keras.optimizers.schedules.LearningRateSchedule):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
                )
            )

        # Train model with conservative memory management to prevent crashes
        # Start with moderate batch size and implement robust fallback
        batch_size = 1024  # Conservative starting point to prevent allocation failures

        # Disable XLA compilation to prevent CUDA graph conflicts
        model.compile(
            optimizer=optimizer,
            loss=quantile_loss(self.quantile),
            metrics=["mae", "mse"],
            jit_compile=False,
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
                train_dataset = train_dataset.batch(batch_size).prefetch(
                    tf.data.AUTOTUNE
                )

                val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

                print(f"🚀 Attempting training with batch size: {batch_size}")
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=100,  # Reduced epochs for faster training
                    callbacks=callbacks,
                    verbose=0,
                )
                print(
                    f"✅ LSTM training completed successfully with batch size {batch_size}"
                )
                break  # Success, exit the loop

            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                print(f"⚠️ Memory error with batch_size={batch_size}: {str(e)[:100]}...")
                # Clear memory before trying next batch size
                tf.keras.backend.clear_session()
                if batch_size == batch_sizes[-1]:  # Last attempt
                    raise RuntimeError(
                        f"Unable to train LSTM model - all batch sizes failed. Last error: {e}"
                    )
                continue

        if history is None:
            raise RuntimeError("LSTM training failed - no successful batch size found")

        print(
            f"✅ LSTM training completed. Best val_loss: {min(history.history['val_loss']):.6f}"
        )

        return model

    def generate_lstm_predictions(
        self, model: tf.keras.Model, X: np.ndarray
    ) -> np.ndarray:
        """
        Generate lstm_delta predictions with conservative memory management
        """
        # Use conservative batch sizes to prevent memory allocation failures
        batch_sizes = [2048, 1024, 512, 256, 128]  # Conservative progression
        predictions = None

        for batch_size in batch_sizes:
            try:
                print(f"🚀 Generating predictions with batch size: {batch_size}")
                predictions = model.predict(X, batch_size=batch_size, verbose=0)
                print(
                    f"✅ Predictions generated successfully with batch size {batch_size}"
                )
                break  # Success, exit the loop

            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                print(
                    f"⚠️ Memory error during prediction with batch_size={batch_size}: {str(e)[:100]}..."
                )
                # Clear memory before trying next batch size
                tf.keras.backend.clear_session()
                if batch_size == batch_sizes[-1]:  # Last attempt
                    raise RuntimeError(
                        f"Unable to generate predictions - all batch sizes failed. Last error: {e}"
                    )
                continue

        if predictions is None:
            raise RuntimeError(
                "Prediction generation failed - no successful batch size found"
            )

        return predictions.flatten()

    def prepare_xgboost_features(
        self,
        df: pd.DataFrame,
        lstm_delta: np.ndarray,
        timestamps: np.ndarray,
        symbol: str,
        window_idx: int,
        is_train: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare feature matrix for XGBoost including lstm_delta.

        When ``is_train`` is True the Boruta feature selection is performed and
        the resulting feature list is persisted.  When ``is_train`` is False the
        previously saved feature list for the given window is loaded so that the
        test set uses exactly the same feature ordering as the training data.
        """
        # Align lstm_delta with dataframe
        lstm_df = pd.DataFrame({"lstm_delta": lstm_delta}, index=timestamps)

        # Merge with technical features
        features_df = df.join(lstm_df, how="inner")

        # Enhanced feature set for XGBoost (comprehensive technical analysis)
        feature_columns = [
            # Enhanced Price features
            "returns",
            "log_returns",
            "price_change_1h",
            "price_change_4h",
            "price_change_24h",
            "price_zscore_20",
            "price_zscore_50",
            # Lag features (important for time series)
            "returns_lag_1",
            "returns_lag_2",
            "returns_lag_3",
            "returns_lag_5",
            "returns_lag_10",
            "log_returns_lag_1",
            "log_returns_lag_2",
            "log_returns_lag_3",
            # Rolling statistics
            "returns_mean_10",
            "returns_std_10",
            "returns_skew_20",
            "returns_kurt_20",
            # Enhanced Volume features
            "volume_ratio",
            "volume_change",
            "volume_zscore",
            "volume_price_trend",
            "volume_weighted_price",
            # Market microstructure
            "spread",
            "spread_ratio",
            "buying_pressure",
            "selling_pressure",
            "net_pressure",
            # Enhanced Volatility
            "volatility_20",
            "volatility_50",
            "volatility_ratio",
            "atr",
            "atr_ratio",
            "realized_vol_5",
            "realized_vol_20",
            "vol_regime",
            # Enhanced Moving averages
            "price_vs_ema9",
            "price_vs_ema21",
            "price_vs_ema50",
            "price_vs_sma200",
            "ema9_vs_ema21",
            "ema21_vs_ema50",
            "ema50_vs_ema100",
            "ma_alignment",
            "ma_slope_9",
            "ma_slope_21",
            # Enhanced Oscillators
            "rsi",
            "rsi_9",
            "rsi_21",
            "rsi_oversold",
            "rsi_overbought",
            "rsi_divergence",
            "stoch_k",
            "stoch_d",
            "stoch_oversold",
            "stoch_overbought",
            "williams_r",
            # MACD
            "macd",
            "macd_signal",
            "macd_histogram",
            "macd_bullish",
            # Bollinger Bands
            "bb_width",
            "bb_position",
            # VWAP
            "price_vs_vwap",
            # Candle patterns
            "candle_body",
            "upper_wick",
            "lower_wick",
            # Time features
            "hour",
            "day_of_week",
            "is_weekend",
            # Momentum
            "momentum_10",
            "roc_10",
            # Support/Resistance
            "near_resistance",
            "near_support",
            # Enhanced Feature Interactions
            "rsi_macd_combo",
            "volatility_ema_ratio",
            "volume_price_momentum",
            "bb_rsi_signal",
            "trend_strength",
            "volatility_breakout",
            "momentum_vol_signal",
            "trend_momentum_align",
            "pressure_volume_signal",
            "volatility_regime_signal",
            "multi_timeframe_signal",
            "oscillator_consensus",
            # Market regime features
            "trend_regime",
            "consolidation_regime",
            # LSTM prediction
            "lstm_delta",
        ]

        # Persist feature column order for inference
        feature_dir = os.path.join(self.models_dir, "feature_columns")
        os.makedirs(feature_dir, exist_ok=True)
        columns_path = os.path.join(
            feature_dir, f"{symbol.lower()}_window_{window_idx}.pkl"
        )
        selected_path = os.path.join(
            feature_dir, f"{symbol.lower()}_window_{window_idx}_selected.pkl"
        )

        try:
            with open(columns_path, "wb") as f:
                pickle.dump(feature_columns, f)
        except Exception as e:
            print(f"⚠️  Failed to save feature columns: {e}")

        # Filter available columns
        available_features = [
            col for col in feature_columns if col in features_df.columns
        ]

        # Create target: binary classification
        features_df["target"] = (
            features_df["close"]
            .pct_change(self.prediction_horizon)
            .shift(-self.prediction_horizon)
            > self.price_change_threshold
        ).astype(int)

        # Select final dataset
        final_df = features_df[available_features + ["target"]].copy()

        # Pre-cleaning: remove NaN/infinite values before Boruta
        pre_clean_df = final_df.replace([np.inf, -np.inf], np.nan)
        rows_before = len(pre_clean_df)
        pre_clean_df = pre_clean_df.dropna()
        rows_dropped_pre = rows_before - len(pre_clean_df)
        if rows_dropped_pre > 0:
            print(
                f"📉 Dropped {rows_dropped_pre} rows with NaN before feature selection"
            )

        # Feature selection using Boruta on cleaned data
        if is_train:
            selected = self.boruta_feature_selection(pre_clean_df)
            try:
                with open(selected_path, "wb") as f:
                    pickle.dump(selected, f)
            except Exception as e:
                print(f"⚠️  Failed to save selected features: {e}")
        else:
            try:
                with open(selected_path, "rb") as f:
                    selected = pickle.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load selected features: {e}")
                selected = [c for c in pre_clean_df.columns if c != "target"]

        # Ensure selected features exist in the dataframe
        selected = [c for c in selected if c in pre_clean_df.columns]
        final_df = pre_clean_df[selected + ["target"]]

        # Data cleaning: handle NaN and infinite values intelligently
        print(f"🧹 Data cleaning: {len(final_df)} samples before cleaning")

        # Replace infinite values with NaN
        final_df = final_df.replace([np.inf, -np.inf], np.nan)

        # Check for NaN values
        nan_counts = final_df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️  Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")

        # Intelligent NaN handling strategy
        # 1. For features with high NaN percentage (>50%), fill with median/mode
        # 2. For features with low NaN percentage (<10%), drop rows
        # 3. For medium NaN percentage (10-50%), use forward fill then median

        total_samples = len(final_df)

        for col in final_df.columns:
            if col == "target":
                continue

            nan_pct = final_df[col].isnull().sum() / total_samples

            if nan_pct > 0.5:  # High NaN percentage - fill with median
                if final_df[col].dtype in ["int64", "float64"]:
                    fill_value = final_df[col].median()
                    final_df[col] = final_df[col].fillna(fill_value)
                    print(
                        f"📊 Filled {col} ({nan_pct:.1%} NaN) with median: {fill_value:.6f}"
                    )
                else:
                    fill_value = (
                        final_df[col].mode().iloc[0]
                        if not final_df[col].mode().empty
                        else 0
                    )
                    final_df[col] = final_df[col].fillna(fill_value)
                    print(
                        f"📊 Filled {col} ({nan_pct:.1%} NaN) with mode: {fill_value}"
                    )

            elif nan_pct > 0.1:  # Medium NaN percentage - forward fill then median
                if final_df[col].dtype in ["int64", "float64"]:
                    final_df[col] = final_df[col].ffill()
                    remaining_nan = final_df[col].isnull().sum()
                    if remaining_nan > 0:
                        fill_value = final_df[col].median()
                        final_df[col] = final_df[col].fillna(fill_value)
                    print(
                        f"📊 Forward filled {col} ({nan_pct:.1%} NaN), then median for remaining"
                    )
                else:
                    final_df[col] = final_df[col].ffill()
                    remaining_nan = final_df[col].isnull().sum()
                    if remaining_nan > 0:
                        fill_value = (
                            final_df[col].mode().iloc[0]
                            if not final_df[col].mode().empty
                            else 0
                        )
                        final_df[col] = final_df[col].fillna(fill_value)
                    print(
                        f"📊 Forward filled {col} ({nan_pct:.1%} NaN), then mode for remaining"
                    )

        # After intelligent filling, drop remaining rows with NaN (should be minimal)
        rows_before_final_drop = len(final_df)
        final_df = final_df.dropna()
        rows_dropped = rows_before_final_drop - len(final_df)

        if rows_dropped > 0:
            print(f"📉 Dropped {rows_dropped} rows with remaining NaN values")

        # Additional validation: ensure no infinite values remain
        if (
            len(final_df) > 0
            and np.isinf(final_df.select_dtypes(include=[np.number]).values).any()
        ):
            print("⚠️  Warning: Infinite values still present after cleaning")
            # Force remove any remaining infinite values
            numeric_cols = final_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                final_df = final_df[np.isfinite(final_df[col])]

        print(
            f"📊 XGBoost features: {len(available_features)} features, {len(final_df)} samples after cleaning"
        )
        print(f"🎯 Target distribution: {final_df['target'].value_counts().to_dict()}")

        return final_df

    def focal_loss_objective(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        """
        Focal loss for XGBoost to handle class imbalance
        """
        import numpy as np

        # Convert to probabilities
        p = 1 / (1 + np.exp(-y_pred))

        # Calculate focal loss components
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        p_t = p * y_true + (1 - p) * (1 - y_true)

        # Focal loss gradient
        grad = alpha_t * (gamma * p_t * np.log(p_t + 1e-8) + p_t - y_true)

        # Focal loss hessian
        hess = alpha_t * gamma * p_t * (1 - p_t) * (2 * np.log(p_t + 1e-8) + 1)

        return grad, hess

    def train_xgboost_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        warm_start_model: Optional[str] = None,
    ) -> xgb.XGBClassifier:
        """
        Train enhanced XGBoost model with advanced configuration
        """
        print(f"🌲 Training Enhanced XGBoost model...")

        # Prepare training data
        X_train = train_df.drop("target", axis=1)
        y_train = train_df["target"]
        X_val = val_df.drop("target", axis=1)
        y_val = val_df["target"]

        # Calculate class distribution and balancing
        class_counts = y_train.value_counts()
        total_samples = len(y_train)

        # Calculate scale_pos_weight for class balancing
        neg_samples = class_counts[0] if 0 in class_counts else 0
        pos_samples = class_counts[1] if 1 in class_counts else 1
        scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0

        print(f"📊 Class Distribution:")
        print(
            f"   Negative (0): {neg_samples:,} ({neg_samples/total_samples*100:.1f}%)"
        )
        print(
            f"   Positive (1): {pos_samples:,} ({pos_samples/total_samples*100:.1f}%)"
        )
        print(f"⚖️  Scale Pos Weight: {scale_pos_weight:.3f}")

        # Enhanced XGBoost with improved parameters
        model = xgb.XGBClassifier(
            **self.xgb_params,
            scale_pos_weight=scale_pos_weight,  # Dynamic class balancing
        )

        if warm_start_model and os.path.exists(warm_start_model):
            try:
                model.load_model(warm_start_model)
                print("♻️  Warm starting XGBoost from previous window")
            except Exception as e:
                print(f"⚠️  Failed to load previous XGBoost model: {e}")

        # Fit with validation and enhanced monitoring
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # Print training summary
        best_iteration = (
            model.best_iteration
            if hasattr(model, "best_iteration")
            else self.xgb_params["n_estimators"]
        )
        print(f"✅ XGBoost training completed. Best iteration: {best_iteration}")

        return model

    def evaluate_window(
        self,
        window_idx: int,
        lstm_model: tf.keras.Model,
        xgb_model: xgb.XGBClassifier,
        test_data: Dict,
    ) -> Dict:
        """
        Comprehensive evaluation with all classification metrics
        """
        # LSTM evaluation
        lstm_pred = self.generate_lstm_predictions(lstm_model, test_data["X_lstm"])
        lstm_mae = mean_absolute_error(test_data["y_lstm"], lstm_pred)
        lstm_rmse = np.sqrt(mean_squared_error(test_data["y_lstm"], lstm_pred))

        # XGBoost evaluation
        X_test = test_data["xgb_df"].drop("target", axis=1)
        y_test = test_data["xgb_df"]["target"]

        xgb_pred = xgb_model.predict(X_test)
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

        # Comprehensive XGBoost metrics
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
        xgb_recall = recall_score(y_test, xgb_pred, zero_division=0)
        xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
        xgb_auc = (
            roc_auc_score(y_test, xgb_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        )

        # Confidence-based predictions (alternative thresholds)
        high_confidence_buys = (xgb_pred_proba > 0.7).astype(int)
        conservative_pred = (xgb_pred_proba > 0.6).astype(int)

        # Calculate metrics for different confidence levels
        conf_precision_70 = precision_score(
            y_test, high_confidence_buys, zero_division=0
        )
        conf_recall_70 = recall_score(y_test, high_confidence_buys, zero_division=0)
        conf_f1_70 = f1_score(y_test, high_confidence_buys, zero_division=0)

        conf_precision_60 = precision_score(y_test, conservative_pred, zero_division=0)
        conf_recall_60 = recall_score(y_test, conservative_pred, zero_division=0)
        conf_f1_60 = f1_score(y_test, conservative_pred, zero_division=0)

        # Class distribution in test set
        test_class_dist = pd.Series(y_test).value_counts()

        print(f"📊 Window {window_idx} Detailed Results:")
        print(f"   🧠 LSTM: MAE={lstm_mae:.6f}, RMSE={lstm_rmse:.6f}")
        print(f"   🌲 XGBoost Metrics (Default 0.5 threshold):")
        print(f"      Accuracy:  {xgb_accuracy:.4f}")
        print(f"      Precision: {xgb_precision:.4f}")
        print(f"      Recall:    {xgb_recall:.4f}")
        print(f"      F1-Score:  {xgb_f1:.4f}")
        print(f"      AUC:       {xgb_auc:.4f}")
        print(f"   🎯 Confidence-Based Metrics:")
        print(
            f"      70% Threshold - P: {conf_precision_70:.4f}, R: {conf_recall_70:.4f}, F1: {conf_f1_70:.4f}"
        )
        print(
            f"      60% Threshold - P: {conf_precision_60:.4f}, R: {conf_recall_60:.4f}, F1: {conf_f1_60:.4f}"
        )
        print(f"   📈 Test Distribution: {dict(test_class_dist)}")

        return {
            "window": window_idx,
            "lstm_mae": float(lstm_mae),
            "lstm_rmse": float(lstm_rmse),
            "xgb_accuracy": float(xgb_accuracy),
            "xgb_precision": float(xgb_precision),
            "xgb_recall": float(xgb_recall),
            "xgb_f1": float(xgb_f1),
            "xgb_auc": float(xgb_auc),
            "conf_precision_70": float(conf_precision_70),
            "conf_recall_70": float(conf_recall_70),
            "conf_f1_70": float(conf_f1_70),
            "conf_precision_60": float(conf_precision_60),
            "conf_recall_60": float(conf_recall_60),
            "conf_f1_60": float(conf_f1_60),
            "test_samples": len(test_data["xgb_df"]),
            "test_positive_ratio": float(test_class_dist.get(1, 0) / len(y_test)),
        }

    def evaluate_models(
        self,
        symbol: str,
        lstm_model: tf.keras.Model,
        xgb_model: xgb.XGBClassifier,
        test_data: Dict,
    ) -> Dict:
        """
        Comprehensive model evaluation
        """
        print(f"📊 Evaluating models for {symbol}...")

        results = {}

        # LSTM Evaluation
        lstm_pred = self.generate_lstm_predictions(lstm_model, test_data["X_lstm"])
        lstm_mae = mean_absolute_error(test_data["y_lstm"], lstm_pred)
        lstm_rmse = np.sqrt(mean_squared_error(test_data["y_lstm"], lstm_pred))

        results["lstm"] = {"mae": lstm_mae, "rmse": lstm_rmse, "predictions": lstm_pred}

        # XGBoost Evaluation
        X_test = test_data["xgb_df"].drop("target", axis=1)
        y_test = test_data["xgb_df"]["target"]

        xgb_pred = xgb_model.predict(X_test)
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_auc = roc_auc_score(y_test, xgb_pred_proba)

        results["xgboost"] = {
            "accuracy": xgb_accuracy,
            "auc": xgb_auc,
            "predictions": xgb_pred,
            "probabilities": xgb_pred_proba,
        }

        # Save results
        with open(f"results/{symbol.lower()}_evaluation.json", "w") as f:
            json.dump(
                {
                    "lstm_mae": float(lstm_mae),
                    "lstm_rmse": float(lstm_rmse),
                    "xgb_accuracy": float(xgb_accuracy),
                    "xgb_auc": float(xgb_auc),
                },
                f,
                indent=2,
            )

        print(f"📈 LSTM - MAE: {lstm_mae:.6f}, RMSE: {lstm_rmse:.6f}")
        print(f"🎯 XGBoost - Accuracy: {xgb_accuracy:.4f}, AUC: {xgb_auc:.4f}")

        return results

    def log_window_results(self, symbol: str, window_results: Dict):
        """
        Log detailed results to CSV for analysis
        """
        csv_path = f"logs/{symbol.lower()}_metrics.csv"

        # Create DataFrame from results
        df_row = pd.DataFrame([window_results])

        # Append to CSV (create if doesn't exist)
        if os.path.exists(csv_path):
            df_row.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(csv_path, mode="w", header=True, index=False)

        print(f"📝 Results logged to {csv_path}")

    def plot_feature_importance(
        self, model: xgb.XGBClassifier, symbol: str, window_idx: int
    ):
        """
        Generate and save feature importance plot
        """
        importance_df = (
            pd.DataFrame(
                {
                    "feature": model.feature_names_in_,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(20)
        )

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title(f"{symbol} - Window {window_idx} - Top 20 Feature Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()

        plot_path = f"logs/feature_importance/{symbol.lower()}_window_{window_idx}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Feature importance plot saved: {plot_path}")

    def get_last_completed_window(self, symbol: str) -> int:
        """
        Check the metrics CSV file to determine the last completed window for resume functionality
        Returns the window number to start from (0-based indexing)
        """
        metrics_file = f"logs/{symbol.lower()}_metrics.csv"

        if not os.path.exists(metrics_file):
            print(
                f"📄 No existing metrics file found for {symbol}, starting from window 1"
            )
            return 0

        try:
            metrics_df = pd.read_csv(metrics_file)
            if len(metrics_df) == 0:
                print(f"📄 Empty metrics file for {symbol}, starting from window 1")
                return 0

            last_window = metrics_df["window"].max()
            print(f"📊 Found existing metrics for {symbol} up to window {last_window}")
            print(f"🔄 Resuming training from window {last_window + 1}")
            return last_window  # Return last completed window (will start from last_window + 1)

        except Exception as e:
            print(f"⚠️  Error reading metrics file for {symbol}: {e}")
            print(f"📄 Starting from window 1")
            return 0

    def train_symbol_walkforward(self, symbol: str) -> List[Dict]:
        """
        Walk-forward training pipeline for a single symbol with resume capability
        """
        print(f"\n{'='*60}")
        print(f"🚀 Walk-Forward Training for {symbol}")
        print(f"{'='*60}")

        # Load and prepare data
        print("\n📊 Data Preparation")
        df = self.load_data(symbol)
        df_features = self.create_technical_features(df)

        # Generate walk-forward windows
        windows = self.generate_walk_forward_windows(df_features)

        if not windows:
            print(f"⚠️  No valid windows found for {symbol}")
            return []

        # Check for resume capability
        last_completed_window = self.get_last_completed_window(symbol)
        start_window_idx = last_completed_window  # 0-based index

        if start_window_idx > 0:
            print(
                f"\n🔄 RESUME MODE: Skipping first {start_window_idx} completed windows"
            )
            print(
                f"📊 Total windows: {len(windows)}, Starting from window: {start_window_idx + 1}"
            )
        else:
            print(f"\n🆕 FRESH START: Training all {len(windows)} windows")

        results = []
        start_time = time.time()  # Initialize start_time for progress tracking

        # Start from the determined window index (resume functionality)
        for i, (train_start, train_end, test_start, test_end) in enumerate(
            windows[start_window_idx:], start=start_window_idx
        ):
            window_start_time = time.time()
            print(
                f"\n🔄 Window {i+1}/{len(windows)}: {train_start.date()} - {train_end.date()} | Test {test_start.date()} to {test_end.date()}"
            )
            print(f"⏰ Window started at: {datetime.now().strftime('%H:%M:%S')}")

            # Split data for this window
            train_data = df_features[train_start:train_end]
            test_data = df_features[test_start:test_end]

            if len(train_data) < self.min_training_samples:
                print(
                    f"⚠️  Skipping window {i+1}: insufficient training data ({len(train_data)} samples)"
                )
                continue

            # Prepare LSTM data for training window
            X_lstm, y_lstm, timestamps = self.prepare_lstm_data(train_data)

            # Validate LSTM data preparation
            if len(X_lstm) == 0 or len(y_lstm) == 0:
                print(f"⚠️  Skipping window {i+1}: no valid LSTM sequences created")
                continue

            if len(X_lstm) < 1000:
                print(
                    f"⚠️  Skipping window {i+1}: insufficient LSTM sequences ({len(X_lstm)})"
                )
                continue

            # Additional validation for LSTM targets
            if np.isnan(y_lstm).any() or np.isinf(y_lstm).any():
                print(f"⚠️  Skipping window {i+1}: invalid LSTM targets detected")
                continue

            # Split LSTM data (80% train, 20% val)
            split_idx = int(0.8 * len(X_lstm))
            X_train_lstm = X_lstm[:split_idx]
            y_train_lstm = y_lstm[:split_idx]
            X_val_lstm = X_lstm[split_idx:]
            y_val_lstm = y_lstm[split_idx:]

            # Scale LSTM data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(
                X_train_lstm.reshape(-1, X_train_lstm.shape[-1])
            ).reshape(X_train_lstm.shape)
            X_val_scaled = scaler.transform(
                X_val_lstm.reshape(-1, X_val_lstm.shape[-1])
            ).reshape(X_val_lstm.shape)

            # Train LSTM with timing
            print(f"🧠 Starting LSTM training for window {i+1}...")
            lstm_start = time.time()
            lstm_model = None
            if self.warm_start and i > 0:
                prev_path = f"{self.models_dir}/lstm/{symbol.lower()}_window_{i}.keras"
                if os.path.exists(prev_path):
                    try:
                        lstm_model = tf.keras.models.load_model(
                            prev_path,
                            compile=False,
                            custom_objects={"directional_loss": directional_loss},
                        )
                        lr = tf.keras.optimizers.schedules.ExponentialDecay(
                            1e-4, 1000, 0.96
                        )
                        lstm_model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                            loss=quantile_loss(self.quantile),
                            metrics=["mae", "mse"],
                        )
                        lstm_model.fit(
                            X_train_scaled,
                            y_train_lstm,
                            validation_data=(X_val_scaled, y_val_lstm),
                            epochs=50,
                            callbacks=[
                                EarlyStopping(patience=5, restore_best_weights=True)
                            ],
                            verbose=0,
                        )
                        print("♻️ Warm started from previous window")
                    except Exception as e:
                        print(f"⚠️ Warm start failed: {e}. Training from scratch.")
                        lstm_model = None

            if lstm_model is None:
                lstm_model = self.train_lstm_model(
                    X_train_scaled,
                    y_train_lstm,
                    X_val_scaled,
                    y_val_lstm,
                )
            lstm_time = (time.time() - lstm_start) / 60
            print(f"✅ LSTM training completed in {lstm_time:.1f} minutes")

            # Generate LSTM predictions for full training period
            X_full_scaled = scaler.transform(
                X_lstm.reshape(-1, X_lstm.shape[-1])
            ).reshape(X_lstm.shape)
            lstm_delta_full = self.generate_lstm_predictions(lstm_model, X_full_scaled)

            # Prepare XGBoost data for training
            xgb_df = self.prepare_xgboost_features(
                train_data,
                lstm_delta_full,
                timestamps,
                symbol,
                i + 1,
                is_train=True,
            )

            if len(xgb_df) < 500:
                print(
                    f"⚠️  Skipping window {i+1}: insufficient XGBoost data ({len(xgb_df)} samples)"
                )
                continue

            # Split XGBoost data (80% train, 20% val)
            split_idx_xgb = int(0.8 * len(xgb_df))
            train_df_xgb = xgb_df.iloc[:split_idx_xgb]
            val_df_xgb = xgb_df.iloc[split_idx_xgb:]

            # Train XGBoost with timing
            print(f"🌳 Starting XGBoost training for window {i+1}...")
            xgb_start = time.time()
            prev_xgb_path = f"{self.models_dir}/xgboost/{symbol.lower()}_window_{i}.pkl"
            warm_start_path = (
                prev_xgb_path
                if (self.warm_start and i > 0 and os.path.exists(prev_xgb_path))
                else None
            )
            xgb_model = self.train_xgboost_model(
                train_df_xgb,
                val_df_xgb,
                warm_start_model=warm_start_path,
            )
            xgb_time = (time.time() - xgb_start) / 60
            print(f"✅ XGBoost training completed in {xgb_time:.1f} minutes")

            # Prepare test data
            X_test_lstm, y_test_lstm, test_timestamps = self.prepare_lstm_data(
                test_data
            )

            # Validate test data preparation
            if len(X_test_lstm) == 0 or len(y_test_lstm) == 0:
                print(f"⚠️  Skipping window {i+1}: no valid test LSTM sequences created")
                continue

            # Additional validation for test LSTM targets
            if np.isnan(y_test_lstm).any() or np.isinf(y_test_lstm).any():
                print(f"⚠️  Skipping window {i+1}: invalid test LSTM targets detected")
                continue

            X_test_scaled = scaler.transform(
                X_test_lstm.reshape(-1, X_test_lstm.shape[-1])
            ).reshape(X_test_lstm.shape)
            lstm_delta_test = self.generate_lstm_predictions(lstm_model, X_test_scaled)

            xgb_test_df = self.prepare_xgboost_features(
                test_data,
                lstm_delta_test,
                test_timestamps,
                symbol,
                i + 1,
                is_train=False,
            )

            if len(xgb_test_df) == 0:
                print(f"⚠️  Skipping window {i+1}: no XGBoost test data")
                continue

            # Evaluate
            test_data_dict = {
                "X_lstm": X_test_scaled,
                "y_lstm": y_test_lstm,
                "xgb_df": xgb_test_df,
            }

            window_results = self.evaluate_window(
                i + 1, lstm_model, xgb_model, test_data_dict
            )
            window_results["symbol"] = symbol
            window_results["train_start"] = train_start.strftime("%Y-%m-%d")
            window_results["train_end"] = train_end.strftime("%Y-%m-%d")
            window_results["test_end"] = test_end.strftime("%Y-%m-%d")

            results.append(window_results)

            # Calculate and display window timing
            window_time = (time.time() - window_start_time) / 60
            print(f"⏱️  Window {i+1} completed in {window_time:.1f} minutes")
            print(
                f"📊 Progress: {i+1}/{len(windows)} windows ({(i+1)/len(windows)*100:.1f}%)"
            )

            # Estimate remaining time (account for resumed training)
            windows_completed_this_session = (i - start_window_idx) + 1
            if windows_completed_this_session > 0:
                avg_time_per_window = (
                    (time.time() - start_time) / windows_completed_this_session / 60
                )
                remaining_windows = len(windows) - (i + 1)
                estimated_remaining = avg_time_per_window * remaining_windows
                print(f"🕐 Estimated remaining time: {estimated_remaining:.1f} minutes")
                if start_window_idx > 0:
                    print(
                        f"📈 Session progress: {windows_completed_this_session}/{len(windows) - start_window_idx} windows in this session"
                    )

            # Log results to CSV
            self.log_window_results(symbol, window_results)

            # Skip feature importance plot for speed (only save for last window)
            if i == len(windows) - 1:  # Only plot for last window
                self.plot_feature_importance(xgb_model, symbol, i + 1)

            # Save models per window
            lstm_model.save(
                f"{self.models_dir}/lstm/{symbol.lower()}_window_{i+1}.keras"
            )
            with open(
                f"{self.models_dir}/xgboost/{symbol.lower()}_window_{i+1}.pkl", "wb"
            ) as f:
                pickle.dump(xgb_model, f)
            with open(
                f"{self.models_dir}/scalers/{symbol.lower()}_window_{i+1}_scaler.pkl",
                "wb",
            ) as f:
                pickle.dump(scaler, f)

            # Save best models from last window
            if i == len(windows) - 1:  # Last window
                # Save final models
                lstm_model.save(f"{self.models_dir}/lstm/{symbol.lower()}_lstm.h5")
                with open(
                    f"{self.models_dir}/xgboost/{symbol.lower()}_xgboost.pkl", "wb"
                ) as f:
                    pickle.dump(xgb_model, f)
                with open(
                    f"{self.models_dir}/scalers/{symbol.lower()}_scaler.pkl", "wb"
                ) as f:
                    pickle.dump(scaler, f)

                # Save feature importance
                X_features = train_df_xgb.drop("target", axis=1)
                importance_df = pd.DataFrame(
                    {
                        "feature": X_features.columns,
                        "importance": xgb_model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                importance_df.to_csv(
                    f"results/{symbol.lower()}_feature_importance.csv", index=False
                )

                print(f"✅ Final models saved for {symbol}")
                print(f"🔍 Top 5 features: {importance_df.head()['feature'].tolist()}")

            # Clear GPU memory between windows to prevent accumulation and CUDA graph errors
            tf.keras.backend.clear_session()
            print(f"🧹 Memory cleared after window {i+1}")

        # Training completion summary
        total_time = (time.time() - start_time) / 60
        if start_window_idx > 0:
            print(f"\n✅ RESUME TRAINING COMPLETED for {symbol}")
            print(
                f"📊 Resumed from window {start_window_idx + 1}, completed {len(results)} additional windows"
            )
            print(f"⏱️  Session time: {total_time:.1f} minutes")
        else:
            print(f"\n✅ FULL TRAINING COMPLETED for {symbol}")
            print(f"📊 Completed all {len(results)} windows")
            print(f"⏱️  Total time: {total_time:.1f} minutes")

        return results

    def train_all_models(self):
        """
        Train models for all symbols using walk-forward validation and save summary
        """
        print(f"\n{'='*80}")
        print(f"���0 WALK-FORWARD HYBRID LSTM + XGBOOST TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(
            f"📅 Walk-Forward Config: {self.train_months}M train → {self.test_months}M test (step: {self.step_months}M)"
        )
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        all_results = {}
        start_time = time.time()

        for i, symbol in enumerate(self.symbols, 1):
            symbol_start = time.time()
            print(
                f"\n🚀 Starting training for symbol {i}/{len(self.symbols)}: {symbol}"
            )
            print(
                f"⏰ Symbol training started at: {datetime.now().strftime('%H:%M:%S')}"
            )

            try:
                symbol_results = self.train_symbol_walkforward(symbol)
                all_results[symbol] = symbol_results

                symbol_time = (time.time() - symbol_start) / 60
                print(f"\n🎯 {symbol} Training Summary:")
                print(f"⏱️  Total time: {symbol_time:.1f} minutes")
                print(
                    f"📊 Windows processed: {len(symbol_results)}/{len(windows) if 'windows' in locals() else 'N/A'}"
                )

                if symbol_results:
                    # Calculate aggregated metrics
                    avg_lstm_mae = np.mean([r["lstm_mae"] for r in symbol_results])
                    avg_lstm_rmse = np.mean([r["lstm_rmse"] for r in symbol_results])
                    avg_xgb_accuracy = np.mean(
                        [r["xgb_accuracy"] for r in symbol_results]
                    )
                    avg_xgb_precision = np.mean(
                        [r["xgb_precision"] for r in symbol_results]
                    )
                    avg_xgb_recall = np.mean([r["xgb_recall"] for r in symbol_results])
                    avg_xgb_f1 = np.mean([r["xgb_f1"] for r in symbol_results])
                    avg_xgb_auc = np.mean([r["xgb_auc"] for r in symbol_results])

                    print(f"\n📊 {symbol} Summary ({len(symbol_results)} windows):")
                    print(f"   LSTM: MAE={avg_lstm_mae:.6f}, RMSE={avg_lstm_rmse:.6f}")
                    print(
                        f"   XGBoost: Acc={avg_xgb_accuracy:.4f}, Prec={avg_xgb_precision:.4f}, Rec={avg_xgb_recall:.4f}, F1={avg_xgb_f1:.4f}, AUC={avg_xgb_auc:.4f}"
                    )
                    print(
                        f"⏱️  Average time per window: {symbol_time/len(symbol_results):.1f} minutes"
                    )
                else:
                    print(f"⚠️  {symbol}: No valid windows processed")

            except Exception as e:
                print(f"\n❌ Error training {symbol}: {str(e)}")
                traceback.print_exc()
                all_results[symbol] = {"error": str(e)}

        total_time = (time.time() - start_time) / 60

        # Calculate overall statistics
        successful_symbols = [
            s
            for s in all_results
            if isinstance(all_results[s], list) and all_results[s]
        ]
        failed_symbols = [
            s
            for s in all_results
            if "error" in str(all_results[s]) or not all_results[s]
        ]

        # Aggregate metrics across all symbols
        all_window_results = []
        for symbol_results in all_results.values():
            if isinstance(symbol_results, list):
                all_window_results.extend(symbol_results)

        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/walkforward_results_{timestamp}.json"

        summary = {
            "timestamp": timestamp,
            "training_type": "walk_forward",
            "config": {
                "train_months": self.train_months,
                "test_months": self.test_months,
                "step_months": self.step_months,
                "min_training_samples": self.min_training_samples,
            },
            "total_time_minutes": total_time,
            "symbols_trained": len(successful_symbols),
            "symbols_failed": len(failed_symbols),
            "total_windows": len(all_window_results),
            "results_by_symbol": all_results,
        }

        # Add aggregated metrics if we have results
        if all_window_results:
            summary["aggregated_metrics"] = {
                "avg_lstm_mae": float(
                    np.mean([r["lstm_mae"] for r in all_window_results])
                ),
                "avg_lstm_rmse": float(
                    np.mean([r["lstm_rmse"] for r in all_window_results])
                ),
                "avg_xgb_accuracy": float(
                    np.mean([r["xgb_accuracy"] for r in all_window_results])
                ),
                "avg_xgb_precision": float(
                    np.mean([r["xgb_precision"] for r in all_window_results])
                ),
                "avg_xgb_recall": float(
                    np.mean([r["xgb_recall"] for r in all_window_results])
                ),
                "avg_xgb_f1": float(np.mean([r["xgb_f1"] for r in all_window_results])),
                "avg_xgb_auc": float(
                    np.mean([r["xgb_auc"] for r in all_window_results])
                ),
                "std_lstm_mae": float(
                    np.std([r["lstm_mae"] for r in all_window_results])
                ),
                "std_xgb_accuracy": float(
                    np.std([r["xgb_accuracy"] for r in all_window_results])
                ),
                "std_xgb_f1": float(np.std([r["xgb_f1"] for r in all_window_results])),
            }

        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Update consolidated training summary
        summary_path = "results/training_summary.json"
        existing_summary = {}
        if os.path.exists(summary_path) and os.path.getsize(summary_path) > 0:
            try:
                with open(summary_path, "r") as f:
                    existing_summary = json.load(f)
            except Exception:
                existing_summary = {}

        if isinstance(existing_summary, dict):
            existing_summary[summary["timestamp"]] = summary
        elif isinstance(existing_summary, list):
            existing_summary.append(summary)
        else:
            existing_summary = {summary["timestamp"]: summary}

        with open(summary_path, "w") as f:
            json.dump(existing_summary, f, indent=2, default=str)

        # Print final summary
        print(f"\n{'='*80}")
        print(f"🎉 WALK-FORWARD TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"⏰ Total time: {total_time:.1f} minutes")
        print(f"✅ Successful: {len(successful_symbols)}/{len(self.symbols)} symbols")
        print(f"📊 Total windows processed: {len(all_window_results)}")

        if all_window_results:
            agg = summary["aggregated_metrics"]
            print(f"\n📈 Overall Performance:")
            print(f"   LSTM: MAE={agg['avg_lstm_mae']:.6f}±{agg['std_lstm_mae']:.6f}")
            print(
                f"   XGBoost: Acc={agg['avg_xgb_accuracy']:.4f}±{agg['std_xgb_accuracy']:.4f}, F1={agg['avg_xgb_f1']:.4f}±{agg['std_xgb_f1']:.4f}, AUC={agg['avg_xgb_auc']:.4f}"
            )
            print(
                f"   Precision={agg['avg_xgb_precision']:.4f}, Recall={agg['avg_xgb_recall']:.4f}"
            )

        print(f"📁 Results saved to: {results_file}")

        if failed_symbols:
            print(f"❌ Failed symbols: {', '.join(failed_symbols)}")

        return summary


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Hybrid LSTM + XGBoost Model Training with Walk-Forward Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_hybrid_models.py                    # Train all symbols
  python train_hybrid_models.py --symbols BTCEUR   # Train only BTCEUR
  python train_hybrid_models.py --symbols BTCEUR ETHEUR  # Train BTCEUR and ETHEUR
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to train (e.g., BTCEUR ETHEUR). If not specified, trains all symbols.",
        choices=["BTCEUR", "ETHEUR", "ADAEUR", "SOLEUR", "XRPEUR"],
        default=None,
    )

    parser.add_argument(
        "--train-months",
        type=int,
        default=3,
        help="Training window size in months (default: 3)",
    )

    parser.add_argument(
        "--test-months",
        type=int,
        default=1,
        help="Test window size in months (default: 1)",
    )

    parser.add_argument(
        "--step-months",
        type=int,
        default=1,
        help="Step size for rolling window in months (default: 1)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing price data (default: data)",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)",
    )

    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Warm start models by fine-tuning from previous window",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)",
    )

    return parser.parse_args()


def main():
    """
    Main training execution
    """
    # Parse command line arguments
    args = parse_arguments()

    # Initialize trainer with specified symbols
    trainer = HybridModelTrainer(
        symbols=args.symbols,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        seed=args.seed,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        warm_start=args.warm_start,
    )

    # Train all models
    results = trainer.train_all_models()

    print("\n🎯 Training pipeline completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Review model performance in results/")
    print("   2. Analyze feature importance files")
    print("   3. Implement trading strategy using trained models")
    print("   4. Set up paper trading for validation")


if __name__ == "__main__":
    main()
