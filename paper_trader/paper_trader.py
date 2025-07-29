import os
import json
import time
import logging
import pandas as pd
import numpy as np
import websocket
import threading
import hmac
import hashlib
import base64
import requests
from datetime import datetime, timedelta
import pickle
from collections import deque
import joblib
import telegram
import asyncio
import ta
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

# Focal loss implementation for LSTM model loading compatibility
@tf.keras.utils.register_keras_serializable(package="Custom", name="FocalLoss")
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal loss for handling class imbalance in binary classification.
    
    This loss focuses learning on hard negatives by down-weighting easy examples.
    It's particularly effective for imbalanced datasets like price jump detection.
    """

    def __init__(self, alpha=0.25, gamma=2.0, name="focal_loss", **kwargs):
        """
        Initialize FocalLoss.

        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma: Focusing parameter to down-weight easy examples (default: 2.0)
            name: Name of the loss function
        """
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Calculate the focal loss.

        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities (0 to 1)
        """
        # Ensure y_pred is clipped to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        focal_loss = -alpha_t * tf.pow(1 - p_t, self.gamma) * tf.math.log(p_t)
        
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        """Get the config for serialization."""
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
        })
        return config

class PaperTrader:
    def __init__(self, api_key, api_secret, telegram_token=None, telegram_chat_id=None, symbols=None, initial_balance=10000):
        """
        Initialize the paper trader
        
        :param api_key: Bitvavo API key
        :param api_secret: Bitvavo API secret
        :param telegram_token: Telegram bot token
        :param telegram_chat_id: Telegram chat ID for notifications
        :param symbols: List of symbols to trade (e.g., ['BTCEUR', 'ETHEUR'])
        :param initial_balance: Initial balance in EUR
        """
        if symbols is None:
            self.symbols = ['BTCEUR', 'ETHEUR', 'SOLEUR', 'XRPEUR', 'ADAEUR']
        else:
            self.symbols = symbols
            
        self.api_key = api_key
        self.api_secret = api_secret
        self.models = {}
        self.lstm_models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.current_prices = {}
        self.positions = {}
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.trade_history = []
        self.websocket = None
        self.running = False
        self.last_prediction_time = {}
        self.historical_data = {}
        self.feature_sequences = {}  # Store historical features for LSTM models
        self.connection_status = "disconnected"
        self.last_websocket_message_time = datetime.now()
        self.api_fallback_active = False
        
        # Telegram notifications
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.telegram_bot = None
        if telegram_token and telegram_chat_id:
            self.telegram_bot = telegram.Bot(token=telegram_token)
            
        # Initialize positions and data containers
        for symbol in self.symbols:
            self.positions[symbol] = {
                'amount': 0,
                'entry_price': 0,
                'take_profit': 0,
                'stop_loss': 0
            }
            self.last_prediction_time[symbol] = datetime.now() - timedelta(minutes=5)
            self.historical_data[symbol] = deque(maxlen=500)  # Store enough data for feature creation
            self.feature_sequences[symbol] = deque(maxlen=120)  # Store last 120 feature vectors for LSTM
            
        # Load models and feature columns
        self.load_models()
        
        # Setup API fallback monitor
        self.api_monitor_thread = None
        
        # Symbol conversions for Bitvavo API format
        self.conversions = {
            'BTCEUR': 'BTC-EUR',
            'ETHEUR': 'ETH-EUR',
            'SOLEUR': 'SOL-EUR',
            'XRPEUR': 'XRP-EUR',
            'ADAEUR': 'ADA-EUR'
        }
        self.reverse_conversions = {v: k for k, v in self.conversions.items()}
    
    def convert_symbol_to_bitvavo_format(self, symbol):
        """Convert symbol from BTCEUR format to BTC-EUR format for Bitvavo API"""
        return self.conversions.get(symbol, symbol)
    
    def convert_symbol_from_bitvavo_format(self, api_symbol):
        """Convert symbol from BTC-EUR format back to BTCEUR format for internal use"""
        return self.reverse_conversions.get(api_symbol, api_symbol)
        
    def load_models(self):
        """Load all available trained models from the models directory"""
        logger.info("Loading all available models...")
        models_dir = "models"
        
        # Load training results to determine best performing models
        training_results = self.load_training_results()
        
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            logger.info(f"Loading models for {symbol}...")
            
            # Load XGBoost models
            self.load_xgboost_models(symbol, symbol_lower, models_dir, training_results)
            
            # Load LSTM models
            self.load_lstm_models(symbol, symbol_lower, models_dir)
            
            # Load scalers
            self.load_scalers(symbol, symbol_lower, models_dir)
            
            # Load feature columns
            self.load_feature_columns(symbol, symbol_lower, models_dir, training_results)
    
    def load_training_results(self):
        """Load training results to determine best performing models"""
        try:
            results_path = os.path.join("train_hybrid_models", "results", "training_summary.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training results: {e}")
        return {}
    
    def get_best_window_for_symbol(self, symbol, training_results):
        """Determine the best window size for a symbol based on training results"""
        symbol_lower = symbol.lower()
        best_window = 6  # Default fallback
        best_score = 0
        
        if symbol_lower in training_results:
            for window_key, metrics in training_results[symbol_lower].items():
                if 'window_' in window_key and isinstance(metrics, dict):
                    # Calculate composite score (accuracy + f1 + auc) / 3
                    try:
                        accuracy = metrics.get('accuracy', 0)
                        f1 = metrics.get('f1', 0)
                        auc = metrics.get('auc', 0)
                        composite_score = (accuracy + f1 + auc) / 3
                        
                        if composite_score > best_score:
                            best_score = composite_score
                            window_num = window_key.replace('window_', '')
                            best_window = int(window_num)
                    except (ValueError, TypeError):
                        continue
        
        logger.info(f"Best window for {symbol}: {best_window} (score: {best_score:.3f})")
        return best_window
    
    def load_xgboost_models(self, symbol, symbol_lower, models_dir, training_results):
        """Load XGBoost models for a symbol"""
        # Load main XGBoost model
        main_model_path = os.path.join(models_dir, "xgboost", f"{symbol_lower}_xgboost.pkl")
        if os.path.exists(main_model_path):
            try:
                self.models[symbol] = joblib.load(main_model_path)
                logger.info(f"Loaded main XGBoost model for {symbol}")
            except Exception as e:
                logger.error(f"Error loading main XGBoost model for {symbol}: {e}")
        
        # Load window-specific XGBoost models
        xgboost_dir = os.path.join(models_dir, "xgboost")
        if os.path.exists(xgboost_dir):
            for file in os.listdir(xgboost_dir):
                if file.startswith(f"{symbol_lower}_window_") and file.endswith(".pkl"):
                    window_num = file.replace(f"{symbol_lower}_window_", "").replace(".pkl", "")
                    model_path = os.path.join(xgboost_dir, file)
                    try:
                        model_key = f"{symbol}_window_{window_num}"
                        self.models[model_key] = joblib.load(model_path)
                        logger.info(f"Loaded XGBoost model for {symbol} window {window_num}")
                    except Exception as e:
                        logger.error(f"Error loading XGBoost model {file}: {e}")
    
    def load_lstm_models(self, symbol, symbol_lower, models_dir):
        """Load LSTM models for a symbol"""
        lstm_dir = os.path.join(models_dir, "lstm")
        if os.path.exists(lstm_dir):
            # Load main LSTM model
            main_lstm_path = os.path.join(lstm_dir, f"{symbol_lower}_lstm.h5")
            if os.path.exists(main_lstm_path):
                try:
                    self.lstm_models[symbol] = keras.models.load_model(main_lstm_path)
                    logger.info(f"Loaded main LSTM model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading main LSTM model for {symbol}: {e}")
            
            # Load window-specific LSTM models
            for file in os.listdir(lstm_dir):
                if file.startswith(f"{symbol_lower}_window_") and file.endswith(".keras"):
                    window_num = file.replace(f"{symbol_lower}_window_", "").replace(".keras", "")
                    model_path = os.path.join(lstm_dir, file)
                    try:
                        model_key = f"{symbol}_window_{window_num}"
                        self.lstm_models[model_key] = keras.models.load_model(model_path)
                        logger.info(f"Loaded LSTM model for {symbol} window {window_num}")
                    except Exception as e:
                        logger.error(f"Error loading LSTM model {file}: {e}")
    
    def load_scalers(self, symbol, symbol_lower, models_dir):
        """Load scalers for a symbol"""
        scalers_dir = os.path.join(models_dir, "scalers")
        if os.path.exists(scalers_dir):
            # Load main scaler
            main_scaler_path = os.path.join(scalers_dir, f"{symbol_lower}_scaler.pkl")
            if os.path.exists(main_scaler_path):
                try:
                    self.scalers[symbol] = joblib.load(main_scaler_path)
                    logger.info(f"Loaded main scaler for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading main scaler for {symbol}: {e}")
            
            # Load window-specific scalers
            for file in os.listdir(scalers_dir):
                if file.startswith(f"{symbol_lower}_window_") and file.endswith("_scaler.pkl"):
                    window_num = file.replace(f"{symbol_lower}_window_", "").replace("_scaler.pkl", "")
                    scaler_path = os.path.join(scalers_dir, file)
                    try:
                        scaler_key = f"{symbol}_window_{window_num}"
                        self.scalers[scaler_key] = joblib.load(scaler_path)
                        logger.info(f"Loaded scaler for {symbol} window {window_num}")
                    except Exception as e:
                        logger.error(f"Error loading scaler {file}: {e}")
    
    def load_feature_columns(self, symbol, symbol_lower, models_dir, training_results):
        """Load feature columns for a symbol"""
        feature_dir = os.path.join(models_dir, "feature_columns")
        if os.path.exists(feature_dir):
            # Determine best window for this symbol
            best_window = self.get_best_window_for_symbol(symbol, training_results)
            
            # Try to load the best window's selected features first
            best_feature_file = f"{symbol_lower}_window_{best_window}_selected.pkl"
            best_feature_path = os.path.join(feature_dir, best_feature_file)
            
            if os.path.exists(best_feature_path):
                try:
                    self.feature_columns[symbol] = joblib.load(best_feature_path)
                    logger.info(f"Loaded best feature columns for {symbol} (window {best_window}, {len(self.feature_columns[symbol])} features)")
                except Exception as e:
                    logger.error(f"Error loading best feature columns for {symbol}: {e}")
            else:
                # Fallback to hardcoded mapping for compatibility
                feature_file_map = {
                    'ADAEUR': 'adaeur_window_6_selected.pkl',  # 21 features
                    'BTCEUR': 'btceur_window_15_selected.pkl',
                    'ETHEUR': 'etheur_window_15_selected.pkl', 
                    'SOLEUR': 'soleur_window_15_selected.pkl',
                    'XRPEUR': 'xrpeur_window_15_selected.pkl'
                }
                
                feature_file = feature_file_map.get(symbol, f"{symbol_lower}_window_15_selected.pkl")
                feature_path = os.path.join(feature_dir, feature_file)
                
                if os.path.exists(feature_path):
                    try:
                        self.feature_columns[symbol] = joblib.load(feature_path)
                        logger.info(f"Loaded fallback feature columns for {symbol} ({len(self.feature_columns[symbol])} features)")
                    except Exception as e:
                        logger.error(f"Error loading fallback feature columns for {symbol}: {e}")
                else:
                    logger.warning(f"No feature columns found for {symbol}")
            
            # Load all window-specific feature columns for ensemble predictions
            # Prioritize selected feature files over regular feature files
            for file in os.listdir(feature_dir):
                if file.startswith(f"{symbol_lower}_window_") and file.endswith("_selected.pkl"):
                    window_info = file.replace(f"{symbol_lower}_window_", "").replace("_selected.pkl", "")
                    feature_path = os.path.join(feature_dir, file)
                    try:
                        window_num = window_info
                        feature_key = f"{symbol}_window_{window_num}"
                        features = joblib.load(feature_path)
                        self.feature_columns[feature_key] = features
                        logger.info(f"Loaded selected feature columns for {symbol} window {window_num} ({len(features)} features)")
                    except Exception as e:
                        logger.error(f"Error loading selected feature columns {file}: {e}")
            
            # Create a consistent 37-feature set for LSTM models based on scaler expectations
            # This addresses the feature count mismatch where scalers expect 37 features
            # but selected features have different counts
            self.create_lstm_feature_set(symbol)
    
    def create_lstm_feature_set(self, symbol):
        """Create a consistent 37-feature set for LSTM models to match scaler expectations"""
        # Define the core 37 features that should be used for LSTM models
        # This is based on the most commonly used features from the training pipeline
        lstm_features = [
            'returns', 'log_returns', 'price_change_1h', 'price_change_4h', 'price_change_24h',
            'volatility_20', 'volatility_50', 'atr_ratio', 'volume_ratio', 'volume_change',
            'spread', 'buying_pressure', 'selling_pressure', 'ema9_vs_ema21', 'ema21_vs_ema50',
            'price_vs_ema9', 'price_vs_ema21', 'price_vs_sma200', 'rsi', 'stoch_k',
            'macd', 'macd_signal', 'macd_histogram', 'bb_width', 'bb_position', 'lstm_delta',
            'returns_lag_1', 'returns_lag_2', 'log_returns_lag_1', 'returns_mean_10', 'returns_std_10',
            'realized_vol_5', 'vol_ratio_15min_30min', 'price_vs_ema50', 'volatility_breakout',
            'vol_regime', 'trend_regime'
        ]
        
        # Store LSTM feature set for each symbol with a special key
        lstm_key = f"{symbol}_lstm"
        self.feature_columns[lstm_key] = lstm_features
        logger.info(f"Created LSTM feature set for {symbol} ({len(lstm_features)} features)")
    
    def get_lstm_sequence(self, symbol):
        """Get the LSTM sequence data for a symbol"""
        if len(self.feature_sequences[symbol]) >= 120:
            # We have enough historical data, use last 120 timesteps
            sequence_data = list(self.feature_sequences[symbol])[-120:]
            return np.array(sequence_data).reshape(1, 120, -1)
        elif len(self.feature_sequences[symbol]) > 0:
            # Not enough historical data, pad with the first available feature
            available_data = list(self.feature_sequences[symbol])
            sequence_length = len(available_data)
            feature_length = len(available_data[0])
            
            # Pad with the first feature vector repeated
            padding_needed = 120 - sequence_length
            padded_sequence = [available_data[0]] * padding_needed + available_data
            return np.array(padded_sequence).reshape(1, 120, feature_length)
        else:
            # No historical data available
            return None
    
    def get_lstm_feature_array(self, symbol, features):
        """Get properly filtered and ordered feature array for LSTM models"""
        lstm_key = f"{symbol}_lstm"
        if lstm_key in self.feature_columns:
            model_features = self.feature_columns[lstm_key]
            # Filter and order features according to LSTM expectations
            filtered_features = {}
            for feature in model_features:
                filtered_features[feature] = features.get(feature, 0.0)
            
            # Convert to array in correct order
            feature_array = np.array([filtered_features[f] for f in model_features]).reshape(1, -1)
            return feature_array, model_features
        else:
            # Fallback to using all features (old behavior)
            feature_array = np.array(list(features.values())).reshape(1, -1)
            return feature_array, list(features.keys())
    
    def fetch_historical_data(self):
        """Fetch initial historical data for all symbols"""
        logger.info("Fetching initial historical data...")
        
        for symbol in self.symbols:
            try:
                # Convert symbol to Bitvavo API format
                api_symbol = self.convert_symbol_to_bitvavo_format(symbol)
                
                # Get 15-minute candles for the past 500 periods (enough for feature creation)
                url = f"https://api.bitvavo.com/v2/{api_symbol}/candles?interval=15m&limit=500"
                response = requests.get(url)
                
                # Handle API errors
                if response.status_code != 200:
                    logger.error(f"Error fetching historical data for {symbol}: HTTP {response.status_code} - {response.text}")
                    continue
                
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    logger.error(f"Error fetching historical data for {symbol}: Invalid JSON response")
                    continue
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                # Store in historical data
                for _, row in df.iterrows():
                    self.historical_data[symbol].append(row.to_dict())
                
                logger.info(f"Fetched {len(df)} historical candles for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                
    def create_features(self, symbol):
        """Create features for the model based on historical data"""
        if not self.historical_data[symbol]:
            logger.warning(f"No historical data available for {symbol}")
            return None
            
        try:
            # Convert deque to DataFrame
            df = pd.DataFrame(list(self.historical_data[symbol]))
            
            # Ensure data is sorted by timestamp
            df = df.sort_values('timestamp')
            df = df.reset_index(drop=True)
            
            # Calculate returns
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            
            # Price change features
            df["price_change_1h"] = df["close"].pct_change(4)  # 4 * 15min = 1h
            df["price_change_4h"] = df["close"].pct_change(16)  # 16 * 15min = 4h
            df["price_change_24h"] = df["close"].pct_change(96)  # 96 * 15min = 24h
            df["price_change_30min"] = df["close"].pct_change(2)  # 2 * 15min = 30min
            
            # Volatility features
            df["volatility_15min"] = df["returns"].rolling(4).std()
            df["volatility_30min"] = df["returns"].rolling(8).std()
            df["volatility_1h"] = df["returns"].rolling(16).std()
            df["volatility_4h"] = df["returns"].rolling(64).std()
            df["volatility_20"] = df["returns"].rolling(20).std()
            df["volatility_50"] = df["returns"].rolling(50).std()
            df["volatility_ratio"] = np.where(df["volatility_50"] == 0, np.nan, df["volatility_20"] / df["volatility_50"])
            
            # ATR
            # Use a simple ATR calculation if ta doesn't have it directly
            df["true_range"] = np.maximum(df["high"] - df["low"], 
                                        np.maximum(abs(df["high"] - df["close"].shift(1)), 
                                                 abs(df["low"] - df["close"].shift(1))))
            df["atr"] = df["true_range"].rolling(14).mean()
            df["atr_ratio"] = df["atr"] / df["close"]
            
            # Volume features
            df["volume_sma_20"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
            df["volume_change"] = df["volume"].pct_change()
            df["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
            df["volume_weighted_price"] = (df["volume"] * df["close"]).rolling(20).sum() / df["volume"].rolling(20).sum()
            
            # Spread features
            df["spread"] = (df["high"] - df["low"]) / df["close"]
            df["spread_ma"] = df["spread"].rolling(20).mean()
            df["spread_ratio"] = np.where(df["spread_ma"] == 0, np.nan, df["spread"] / df["spread_ma"])
            
            # Order flow approximation
            df["buying_pressure"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
            df["selling_pressure"] = (df["high"] - df["close"]) / (df["high"] - df["low"])
            df["net_pressure"] = df["buying_pressure"] - df["selling_pressure"]
            
            # Moving averages
            df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
            df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
            df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
            df["ema_100"] = ta.trend.EMAIndicator(df["close"], window=100).ema_indicator()
            df["sma_200"] = ta.trend.SMAIndicator(df["close"], window=200).sma_indicator()
            
            # EMA relationships
            df["ema9_vs_ema21"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"]
            df["ema21_vs_ema50"] = (df["ema_21"] - df["ema_50"]) / df["ema_50"]
            df["ema50_vs_ema100"] = (df["ema_50"] - df["ema_100"]) / df["ema_100"]
            df["price_vs_ema9"] = (df["close"] - df["ema_9"]) / df["ema_9"]
            df["price_vs_ema21"] = (df["close"] - df["ema_21"]) / df["ema_21"]
            df["price_vs_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
            df["price_vs_vwap"] = (df["close"] - df["volume_weighted_price"]) / df["volume_weighted_price"]
            
            # RSI and other oscillators
            df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_histogram"] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            
            # Add placeholder lstm_delta feature (constant value since we don't have LSTM predictions)
            df["lstm_delta"] = 0  # Placeholder value
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
                df[f"log_returns_lag_{lag}"] = df["log_returns"].shift(lag)
            
            # Rolling statistics - missing features
            df["returns_mean_10"] = df["returns"].rolling(10).mean()
            df["returns_std_10"] = df["returns"].rolling(10).std()
            df["returns_skew_20"] = df["returns"].rolling(20).skew()
            df["returns_kurt_20"] = df["returns"].rolling(20).kurt()
            
            # Realized volatility features
            df["realized_vol_5"] = df["returns"].rolling(5).std() * np.sqrt(5)
            df["realized_vol_20"] = df["returns"].rolling(20).std() * np.sqrt(20)
            
            # Volatility ratios
            df["vol_ratio_15min_30min"] = np.where(df["volatility_30min"] == 0, np.nan, df["volatility_15min"] / df["volatility_30min"])
            df["vol_ratio_30min_1h"] = np.where(df["volatility_1h"] == 0, np.nan, df["volatility_30min"] / df["volatility_1h"])
            df["vol_ratio_1h_4h"] = np.where(df["volatility_4h"] == 0, np.nan, df["volatility_1h"] / df["volatility_4h"])
            
            # Additional price vs EMA features
            df["price_vs_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
            
            # Feature interactions
            epsilon = 1e-8  # Small value to handle division by zero
            df["volatility_ema_ratio"] = np.where(df["ema_21"] == 0, 0, df["volatility_20"] / df["ema_21"])
            df["volume_price_momentum"] = df["volume_ratio"] * df["returns"]
            df["bb_rsi_signal"] = df["bb_position"] * df["rsi"]
            df["trend_strength"] = df["price_vs_ema9"] * df["price_vs_ema21"]
            
            # Volatility breakout indicator
            df["volatility_breakout"] = (df["volatility_20"] > df["volatility_20"].rolling(20).quantile(0.8)).astype(int)
            
            # Time features
            df["hour"] = pd.to_datetime(df["timestamp"], unit='ms').dt.hour
            df["day_of_week"] = pd.to_datetime(df["timestamp"], unit='ms').dt.dayofweek
            df["is_weekend"] = (pd.to_datetime(df["timestamp"], unit='ms').dt.dayofweek >= 5).astype(int)
            
            # Volume price trend
            df["volume_price_trend"] = df["volume_ratio"] * df["returns"]
            
            # Market regime indicators
            df["vol_regime"] = (df["volatility_20"] > df["volatility_20"].rolling(50).median()).astype(int)
            df["trend_regime"] = ((df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])).astype(int)
            df["ma_alignment"] = ((df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"]) & (df["ema_50"] > df["ema_100"])).astype(int)
            
            # Price normalization
            df["price_zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
            df["price_zscore_50"] = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(50).std()
            
            # Additional oscillator features
            df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
            df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
            df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)
            df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
            
            # Momentum features
            df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
            df["roc_10"] = df["close"].pct_change(10)
            
            # MACD bullish signal
            df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)
            
            # Candle pattern features
            df["candle_body"] = abs(df["close"] - df["open"]) / df["open"]
            df["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]
            df["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]
            
            # Additional volume features
            df["volume_surge_5"] = (df["volume"] > df["volume"].rolling(5).quantile(0.8)).astype(int)
            
            # Momentum acceleration
            df["momentum_acceleration"] = df["momentum_10"] - df["momentum_10"].shift(1)
            
            # Market momentum alignment
            df["market_momentum_alignment"] = ((df["momentum_10"] > 0) & (df["rsi"] > 50) & (df["macd"] > df["macd_signal"])).astype(int)
            
            # Add missing features identified in analysis
            # MA slopes
            df["ma_slope_9"] = df["ema_9"].diff()
            df["ma_slope_21"] = df["ema_21"].diff()
            
            # Additional momentum features
            df["momentum_1h"] = df["close"].pct_change(4)  # 4 * 15min = 1h
            df["momentum_2h"] = df["close"].pct_change(8)  # 8 * 15min = 2h
            df["momentum_4h"] = df["close"].pct_change(16)  # 16 * 15min = 4h
            
            # Momentum ratios
            df["momentum_ratio_1h_2h"] = df["momentum_1h"] / (df["momentum_2h"] + 1e-8)
            df["momentum_ratio_2h_4h"] = df["momentum_2h"] / (df["momentum_4h"] + 1e-8)
            
            # Price vs EMA for different timeframes
            df["price_vs_ema_30min"] = (df["close"] - df["ema_9"].rolling(2).mean()) / df["ema_9"].rolling(2).mean()
            df["price_vs_ema_1h"] = (df["close"] - df["ema_9"].rolling(4).mean()) / df["ema_9"].rolling(4).mean()
            df["price_vs_ema_2h"] = (df["close"] - df["ema_9"].rolling(8).mean()) / df["ema_9"].rolling(8).mean()
            df["price_vs_ema_4h"] = (df["close"] - df["ema_9"].rolling(16).mean()) / df["ema_9"].rolling(16).mean()
            
            # RSI divergence (simplified)
            df["rsi_divergence"] = df["rsi"].diff()
            
            # RSI MACD combo
            df["rsi_macd_combo"] = df["rsi"] * df["macd"]
            
            # Trend strength
            df["trend_strength"] = abs(df["ema_9"] - df["ema_21"]) / df["ema_21"]
            
            # Additional volatility ratios
            df["vol_ratio_15min_30min"] = df["volatility_15min"] / (df["volatility_30min"] + 1e-8)
            df["vol_ratio_30min_1h"] = df["volatility_30min"] / (df["volatility_1h"] + 1e-8)
            
            # Volatility EMA ratio
            df["volatility_ema_ratio"] = df["volatility_20"] / (df["ema_21"] + 1e-8)
            
            # Drop NaN values
            df = df.dropna()
            
            if df.empty:
                logger.warning(f"Empty dataframe after feature creation for {symbol}")
                return None
                
            # Return the latest row as a dictionary
            latest_features = df.iloc[-1].to_dict()
            
            # Return all created features - filtering will be done per model in get_ensemble_prediction
            return latest_features
                
        except Exception as e:
            logger.error(f"Error creating features for {symbol}: {e}")
            return None
    
    def start_websocket(self):
        """Start the websocket connection to Bitvavo"""
        logger.info("Starting websocket connection...")
        
        # Define websocket callbacks
        def on_message(ws, message):
            try:
                self.last_websocket_message_time = datetime.now()
                self.connection_status = "connected"
                self.api_fallback_active = False
                
                data = json.loads(message)
                
                # Handle different message types
                if isinstance(data, list):
                    for event in data:
                        self.process_event(event)
                else:
                    self.process_event(data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
        def on_error(ws, error):
            logger.error(f"Websocket error: {error}")
            self.connection_status = "error"
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Websocket closed: {close_status_code} - {close_msg}")
            self.connection_status = "disconnected"
            
            if self.running:
                logger.info("Attempting to reconnect...")
                time.sleep(5)
                self.start_websocket()
                
        def on_open(ws):
            logger.info("Websocket connection established")
            self.connection_status = "connected"
            self.send_telegram_message("🤖 Paper trader bot started and connected to Bitvavo!")
            
            # Subscribe to ticker24h and candle data for all symbols
            markets = []
            for symbol in self.symbols:
                api_symbol = self.convert_symbol_to_bitvavo_format(symbol)
                markets.append(api_symbol)
            
            # Subscribe to ticker24h data
            ticker_message = {
                "action": "subscribe",
                "channels": [
                    {
                        "name": "ticker24h",
                        "markets": markets
                    }
                ]
            }
            ws.send(json.dumps(ticker_message))
            logger.info(f"Subscribed to ticker24h for markets: {markets}")
            
            # Subscribe to candle data
            candle_message = {
                "action": "subscribe",
                "channels": [
                    {
                        "name": "candles",
                        "interval": ["15m"],
                        "markets": markets
                    }
                ]
            }
            ws.send(json.dumps(candle_message))
            logger.info(f"Subscribed to candles for markets: {markets}")
            
        # Create and start websocket
        self.websocket = websocket.WebSocketApp("wss://ws.bitvavo.com/v2/",
                                                on_message=on_message,
                                                on_error=on_error,  
                                                on_close=on_close)
        self.websocket.on_open = on_open
        
        # Start the websocket in a separate thread
        wst = threading.Thread(target=self.websocket.run_forever)
        wst.daemon = True
        wst.start()
        
        # Start connection monitor
        self.start_connection_monitor()
    
    def start_connection_monitor(self):
        """Start a thread to monitor the websocket connection and fallback to API if needed"""
        def monitor_connection():
            while self.running:
                now = datetime.now()
                # If we haven't received a websocket message in 30 seconds, use API fallback
                if (now - self.last_websocket_message_time).total_seconds() > 30 and not self.api_fallback_active:
                    logger.warning("Websocket connection seems inactive, falling back to API")
                    self.api_fallback_active = True
                    self.update_data_via_api()
                
                # Check every 15 seconds
                time.sleep(15)
        
        self.api_monitor_thread = threading.Thread(target=monitor_connection)
        self.api_monitor_thread.daemon = True
        self.api_monitor_thread.start()
    
    def update_data_via_api(self):
        """Update market data using REST API instead of websocket"""
        try:
            logger.info("Updating data via API fallback")
            
            # Update ticker data for all symbols
            for symbol in self.symbols:
                try:
                    # Convert symbol to Bitvavo API format
                    api_symbol = self.convert_symbol_to_bitvavo_format(symbol)
                    
                    # Get current ticker
                    url = f"https://api.bitvavo.com/v2/ticker/price?market={api_symbol}"
                    response = requests.get(url)
                    
                    # Handle API errors
                    if response.status_code != 200:
                        logger.error(f"Error updating API data for {symbol}: HTTP {response.status_code} - {response.text}")
                        continue
                    
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        logger.error(f"Error updating API data for {symbol}: Invalid JSON response")
                        continue
                    
                    if 'price' in data:
                        self.current_prices[symbol] = float(data['price'])
                        logger.debug(f"API fallback: Updated price for {symbol}: {self.current_prices[symbol]}")
                        
                        # Check if it's time to make a prediction
                        now = datetime.now()
                        if (now - self.last_prediction_time[symbol]).total_seconds() >= 60:
                            self.make_prediction(symbol)
                            self.last_prediction_time[symbol] = now
                    
                    # Get latest candle
                    url = f"https://api.bitvavo.com/v2/{api_symbol}/candles?interval=15m&limit=1"
                    response = requests.get(url)
                    
                    # Handle API errors
                    if response.status_code != 200:
                        logger.error(f"Error updating candle data for {symbol}: HTTP {response.status_code} - {response.text}")
                        continue
                    
                    try:
                        candles = response.json()
                    except json.JSONDecodeError:
                        logger.error(f"Error updating candle data for {symbol}: Invalid JSON response")
                        continue
                    
                    if candles and len(candles) > 0:
                        candle = candles[0]
                        candle_data = {
                            'timestamp': pd.to_datetime(candle[0], unit='ms'),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        }
                        
                        # Only append if it's a new candle
                        if not self.historical_data[symbol] or candle_data['timestamp'] > self.historical_data[symbol][-1]['timestamp']:
                            self.historical_data[symbol].append(candle_data)
                            logger.debug(f"API fallback: Updated candle data for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error updating API data for {symbol}: {e}")
            
            # Schedule next API update if fallback is still active
            if self.api_fallback_active and self.running:
                threading.Timer(60, self.update_data_via_api).start()
                
        except Exception as e:
            logger.error(f"Error in API fallback update: {e}")
    
    def process_event(self, event):
        """Process websocket events"""
        try:
            # Handle ticker24h events (new format with data array)
            if event.get('event') == 'ticker24h':
                data_array = event.get('data', [])
                for ticker_data in data_array:
                    api_symbol = ticker_data.get('market')
                    symbol = self.convert_symbol_from_bitvavo_format(api_symbol)
                    if symbol in self.symbols:
                        last_price = ticker_data.get('last')
                        if last_price is not None:
                            self.current_prices[symbol] = float(last_price)
                            logger.debug(f"Updated price for {symbol}: {self.current_prices[symbol]}")
                            
                            # Check if it's time to make a prediction
                            now = datetime.now()
                            if (now - self.last_prediction_time[symbol]).total_seconds() >= 60:  # Predict every minute
                                self.make_prediction(symbol)
                                self.last_prediction_time[symbol] = now
                        else:
                            logger.warning(f"Received None price for {symbol}, skipping update")
                        
            # Handle candle events
            elif event.get('event') == 'candle':
                api_symbol = event.get('market')
                symbol = self.convert_symbol_from_bitvavo_format(api_symbol)
                if symbol in self.symbols:
                    # Check for None values in OHLCV data
                    open_price = event.get('open')
                    high_price = event.get('high')
                    low_price = event.get('low')
                    close_price = event.get('close')
                    volume = event.get('volume')
                    timestamp = event.get('timestamp')
                    
                    if all(x is not None for x in [open_price, high_price, low_price, close_price, volume, timestamp]):
                        candle_data = {
                            'timestamp': pd.to_datetime(timestamp, unit='ms'),
                            'open': float(open_price),
                            'high': float(high_price),
                            'low': float(low_price),
                            'close': float(close_price),
                            'volume': float(volume)
                        }
                        
                        # Update historical data
                        self.historical_data[symbol].append(candle_data)
                        logger.debug(f"Updated candle data for {symbol}")
                    else:
                        logger.warning(f"Received incomplete candle data for {symbol}, skipping update")
                    
        except Exception as e:
            logger.error(f"Error processing event: {e}")
    
    def make_prediction(self, symbol):
        """Make ensemble prediction for a symbol using all available models"""
        try:
            # Create features for prediction
            features = self.create_features(symbol)
            if features is None:
                logger.warning(f"Could not create features for {symbol}")
                return
            
            # Store feature vector for LSTM sequence (using the 37-feature LSTM set)
            lstm_feature_array, _ = self.get_lstm_feature_array(symbol, features)
            self.feature_sequences[symbol].append(lstm_feature_array.flatten())  # Store as 1D array
            
            # Get ensemble prediction
            ensemble_result = self.get_ensemble_prediction(symbol, features)
            if ensemble_result is None:
                logger.warning(f"No models available for prediction for {symbol}")
                return
            
            prediction, prediction_proba, model_count = ensemble_result
            
            logger.info(f"Ensemble prediction for {symbol}: {prediction} (from {model_count} models)")
            if prediction_proba is not None:
                logger.info(f"Ensemble prediction probabilities: {prediction_proba}")
                
            # Execute trade based on prediction
            self.execute_trade(symbol, prediction, prediction_proba)
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
    
    def get_ensemble_prediction(self, symbol, features):
        """Get ensemble prediction from all available models for a symbol"""
        predictions = []
        probabilities = []
        model_count = 0
        
        # Convert features to DataFrame
        X = pd.DataFrame([features])
        
        # Get XGBoost predictions
        xgb_predictions = self.get_xgboost_predictions(symbol, X, features)
        predictions.extend(xgb_predictions['predictions'])
        probabilities.extend(xgb_predictions['probabilities'])
        model_count += xgb_predictions['count']
        
        # Get LSTM predictions
        lstm_predictions = self.get_lstm_predictions(symbol, features)
        predictions.extend(lstm_predictions['predictions'])
        probabilities.extend(lstm_predictions['probabilities'])
        model_count += lstm_predictions['count']
        
        if model_count == 0:
            return None
        
        # Ensemble the predictions (majority vote for classification)
        final_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
        
        # Average the probabilities if available
        final_proba = None
        if probabilities:
            # Filter out None probabilities
            valid_probas = [p for p in probabilities if p is not None]
            if valid_probas:
                # Average probabilities across models
                final_proba = np.mean(valid_probas, axis=0)
        
        return final_prediction, final_proba, model_count
    
    def get_xgboost_predictions(self, symbol, X, features):
        """Get predictions from all XGBoost models for a symbol"""
        predictions = []
        probabilities = []
        count = 0
        
        # Check main XGBoost model
        if symbol in self.models:
            try:
                # Use symbol-specific features if available
                if symbol in self.feature_columns:
                    model_features = self.feature_columns[symbol]
                    missing_features = [f for f in model_features if f not in features]
                    if missing_features:
                        logger.warning(f"Missing features for main {symbol} model: {missing_features}")
                        # Fill missing features with default values
                        filtered_features = {}
                        for feature in model_features:
                            filtered_features[feature] = features.get(feature, 0.0)
                        X_main = pd.DataFrame([filtered_features])
                    else:
                        filtered_features = {f: features[f] for f in model_features}
                        X_main = pd.DataFrame([filtered_features])
                else:
                    X_main = X
                
                pred = self.models[symbol].predict(X_main)[0]
                predictions.append(pred)
                count += 1
                
                if hasattr(self.models[symbol], 'predict_proba'):
                    proba = self.models[symbol].predict_proba(X_main)[0]
                    probabilities.append(proba)
                else:
                    probabilities.append(None)
                    
                logger.debug(f"XGBoost main model prediction for {symbol}: {pred}")
            except Exception as e:
                logger.error(f"Error with main XGBoost model for {symbol}: {e}")
        
        # Check window-specific XGBoost models
        for model_key in self.models.keys():
            if model_key.startswith(f"{symbol}_window_"):
                try:
                    # Debug: Check what feature columns are available
                    logger.debug(f"Checking model {model_key}, feature_columns keys: {list(self.feature_columns.keys())[:5]}...")
                    
                    # Get corresponding feature columns
                    if model_key in self.feature_columns:
                        # Filter features to match this model's requirements
                        model_features = self.feature_columns[model_key]
                        
                        # Check if all required features are available
                        missing_features = [f for f in model_features if f not in features]
                        if missing_features:
                            logger.warning(f"Missing features for {model_key}: {missing_features}")
                            # Fill missing features with default values
                            filtered_features = {}
                            for feature in model_features:
                                filtered_features[feature] = features.get(feature, 0.0)
                        else:
                            filtered_features = {f: features[f] for f in model_features}
                        
                        # Create DataFrame with exact features in correct order
                        X_filtered = pd.DataFrame([filtered_features])[model_features]
                        
                        # Debug logging
                        logger.debug(f"Model {model_key} expects {len(model_features)} features, got {X_filtered.shape[1]}")
                        logger.debug(f"Expected features: {model_features[:5]}...")  # Show first 5 features
                        logger.debug(f"Actual features: {list(X_filtered.columns)[:5]}...")  # Show first 5 features
                        
                        # Ensure we have the exact number of features expected
                        if X_filtered.shape[1] != len(model_features):
                            logger.error(f"Feature shape mismatch for {model_key}: expected {len(model_features)}, got {X_filtered.shape[1]}")
                            continue
                        
                        pred = self.models[model_key].predict(X_filtered)[0]
                        predictions.append(pred)
                        count += 1
                        
                        if hasattr(self.models[model_key], 'predict_proba'):
                            proba = self.models[model_key].predict_proba(X_filtered)[0]
                            probabilities.append(proba)
                        else:
                            probabilities.append(None)
                            
                        logger.debug(f"XGBoost {model_key} prediction: {pred} (features: {X_filtered.shape[1]})")
                    else:
                        logger.warning(f"No feature columns found for {model_key}, using original X with {X.shape[1]} features")
                        # Fallback to using original features
                        pred = self.models[model_key].predict(X)[0]
                        predictions.append(pred)
                        count += 1
                        
                        if hasattr(self.models[model_key], 'predict_proba'):
                            proba = self.models[model_key].predict_proba(X)[0]
                            probabilities.append(proba)
                        else:
                            probabilities.append(None)
                            
                        logger.debug(f"XGBoost {model_key} prediction (fallback): {pred}")
                except Exception as e:
                    logger.error(f"Error with XGBoost model {model_key}: {e}")
        
        return {'predictions': predictions, 'probabilities': probabilities, 'count': count}
    
    def get_lstm_predictions(self, symbol, features):
        """Get predictions from all LSTM models for a symbol"""
        predictions = []
        probabilities = []
        count = 0
        
        # Check main LSTM model
        if symbol in self.lstm_models:
            try:
                # Get sequence data for LSTM
                lstm_sequence = self.get_lstm_sequence(symbol)
                if lstm_sequence is None:
                    logger.warning(f"No sequence data available for LSTM {symbol} model")
                else:
                    # Scale the sequence if scaler is available
                    scaled_sequence = lstm_sequence
                    if symbol in self.scalers:
                        expected_features = self.scalers[symbol].n_features_in_
                        if lstm_sequence.shape[2] == expected_features:
                            # Scale each timestep
                            sequence_2d = lstm_sequence.reshape(-1, lstm_sequence.shape[2])
                            scaled_2d = self.scalers[symbol].transform(sequence_2d)
                            scaled_sequence = scaled_2d.reshape(lstm_sequence.shape)
                        else:
                            logger.warning(f"Feature count mismatch for {symbol} scaler: expected {expected_features}, got {lstm_sequence.shape[2]}")
                    
                    pred_proba = self.lstm_models[symbol].predict(scaled_sequence, verbose=0)[0]
                    pred = 1 if pred_proba[0] > 0.5 else 0
                    predictions.append(pred)
                    probabilities.append([1-pred_proba[0], pred_proba[0]])
                    count += 1
                    
                    logger.debug(f"LSTM main model prediction for {symbol}: {pred} (proba: {pred_proba[0]:.3f})")
            except Exception as e:
                logger.error(f"Error with main LSTM model for {symbol}: {e}")
        
        # Check window-specific LSTM models
        for model_key in self.lstm_models.keys():
            if model_key.startswith(f"{symbol}_window_"):
                try:
                    # Get sequence data for LSTM
                    lstm_sequence = self.get_lstm_sequence(symbol)
                    if lstm_sequence is None:
                        logger.warning(f"No sequence data available for LSTM {model_key}")
                        continue
                    
                    # Use corresponding scaler if available
                    scaler_key = model_key
                    scaled_sequence = lstm_sequence
                    
                    if scaler_key in self.scalers:
                        expected_features = self.scalers[scaler_key].n_features_in_
                        if lstm_sequence.shape[2] == expected_features:
                            # Scale each timestep
                            sequence_2d = lstm_sequence.reshape(-1, lstm_sequence.shape[2])
                            scaled_2d = self.scalers[scaler_key].transform(sequence_2d)
                            scaled_sequence = scaled_2d.reshape(lstm_sequence.shape)
                        else:
                            logger.warning(f"Feature count mismatch for {scaler_key}: expected {expected_features}, got {lstm_sequence.shape[2]}")
                    
                    pred_proba = self.lstm_models[model_key].predict(scaled_sequence, verbose=0)[0]
                    pred = 1 if pred_proba[0] > 0.5 else 0
                    predictions.append(pred)
                    probabilities.append([1-pred_proba[0], pred_proba[0]])
                    count += 1
                    
                    logger.debug(f"LSTM {model_key} prediction: {pred} (proba: {pred_proba[0]:.3f})")
                except Exception as e:
                    logger.error(f"Error with LSTM model {model_key}: {e}")
        
        return {'predictions': predictions, 'probabilities': probabilities, 'count': count}
    
    def execute_trade(self, symbol, prediction, prediction_proba=None):
        """Execute a paper trade based on the prediction"""
        try:
            current_price = self.current_prices.get(symbol)
            if not current_price:
                logger.warning(f"No current price available for {symbol}")
                return
                
            # Check if we already have a position
            current_position = self.positions[symbol]
            
            # Logic for opening a position
            if current_position['amount'] == 0:
                # For classification models: prediction is class (0 or 1)
                should_buy = prediction == 1
                    
                # If we have probabilities, we can be more selective
                if prediction_proba is not None and len(prediction_proba) > 1:
                    # Only buy if probability of positive class is high enough
                    should_buy = should_buy and prediction_proba[1] > 0.6  # Adjust threshold as needed
                
                if should_buy:
                    # Calculate position size (10% of balance)
                    position_size = self.balance * 0.1
                    amount = position_size / current_price
                    
                    # Calculate take profit and stop loss levels
                    take_profit = current_price * 1.005  # 0.5% profit
                    stop_loss = current_price * 0.995   # 0.5% loss
                    
                    # Record the trade
                    trade = {
                        'symbol': symbol,
                        'action': 'buy',
                        'amount': amount,
                        'price': current_price,
                        'value': amount * current_price,
                        'take_profit': take_profit,
                        'stop_loss': stop_loss,
                        'timestamp': datetime.now(),
                        'fees': amount * current_price * 0.003  # 0.3% fees
                    }
                    
                    # Update balance and position
                    trade_value = amount * current_price
                    trade_fee = trade_value * 0.003
                    self.balance -= (trade_value + trade_fee)
                    
                    self.positions[symbol] = {
                        'amount': amount,
                        'entry_price': current_price,
                        'take_profit': take_profit,
                        'stop_loss': stop_loss
                    }
                    
                    self.trade_history.append(trade)
                    logger.info(f"OPENED position for {symbol}: {amount:.6f} @ {current_price} EUR")
                    
                    # Send notification
                    self.send_trade_notification(trade)
                    
            # Logic for closing a position
            else:
                # Check if take profit or stop loss hit
                if current_price >= current_position['take_profit']:
                    # Take profit hit
                    close_reason = "take_profit"
                    self.close_position(symbol, current_price, close_reason)
                    
                elif current_price <= current_position['stop_loss']:
                    # Stop loss hit
                    close_reason = "stop_loss"
                    self.close_position(symbol, current_price, close_reason)
                    
                # Alternatively, close based on model prediction
                elif prediction == 0:
                    close_reason = "model_signal"
                    self.close_position(symbol, current_price, close_reason)
                    
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def close_position(self, symbol, current_price, reason):
        """Close an open position"""
        position = self.positions[symbol]
        if position['amount'] == 0:
            return
            
        amount = position['amount']
        entry_price = position['entry_price']
        
        # Calculate value and P&L
        exit_value = amount * current_price
        entry_value = amount * entry_price
        pnl_before_fees = exit_value - entry_value
        fees = exit_value * 0.003  # 0.3% fees
        pnl = pnl_before_fees - fees
        
        # Record the trade
        trade = {
            'symbol': symbol,
            'action': 'sell',
            'amount': amount,
            'entry_price': entry_price,
            'exit_price': current_price,
            'entry_value': entry_value,
            'exit_value': exit_value,
            'pnl_before_fees': pnl_before_fees,
            'fees': fees,
            'pnl': pnl,
            'reason': reason,
            'timestamp': datetime.now()
        }
        
        # Update balance and position
        self.balance += (exit_value - fees)
        self.positions[symbol] = {
            'amount': 0,
            'entry_price': 0,
            'take_profit': 0,
            'stop_loss': 0
        }
        
        self.trade_history.append(trade)
        logger.info(f"CLOSED position for {symbol}: {amount:.6f} @ {current_price} EUR, PnL: {pnl:.2f} EUR, Reason: {reason}")
        
        # Send notification
        self.send_trade_notification(trade)
    
    def send_telegram_message(self, message):
        """Send a message via Telegram"""
        if not self.telegram_bot or not self.telegram_chat_id:
            return
            
        try:
            # Use asyncio to send the message
            async def send_async():
                async with self.telegram_bot:
                    await self.telegram_bot.send_message(
                        chat_id=self.telegram_chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )
            
            # Run the async function in a new event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # If we're already in an event loop, run the coroutine thread-safely
                asyncio.run_coroutine_threadsafe(send_async(), loop)
            else:
                loop.run_until_complete(send_async())
                
            logger.debug(f"Sent Telegram message: {message}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def send_trade_notification(self, trade):
        """Send a notification about a trade"""
        if not self.telegram_bot:
            return
            
        try:
            if trade['action'] == 'buy':
                message = (
                    f"🟢 *OPENED POSITION*\n\n"
                    f"Symbol: {trade['symbol']}\n"
                    f"Amount: {trade['amount']:.6f}\n"
                    f"Price: €{trade['price']:.4f}\n"
                    f"Value: €{trade['value']:.2f}\n"
                    f"Take Profit: €{trade['take_profit']:.4f}\n"
                    f"Stop Loss: €{trade['stop_loss']:.4f}\n"
                    f"Fees: €{trade['fees']:.2f}\n"
                    f"Time: {trade['timestamp'].strftime('%H:%M:%S')}"
                )
            else:
                pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                message = (
                    f"{pnl_emoji} *CLOSED POSITION*\n\n"
                    f"Symbol: {trade['symbol']}\n"
                    f"Amount: {trade['amount']:.6f}\n"
                    f"Entry: €{trade['entry_price']:.4f}\n"
                    f"Exit: €{trade['exit_price']:.4f}\n"
                    f"P&L: €{trade['pnl']:.2f}\n"
                    f"Reason: {trade['reason']}\n"
                    f"Fees: €{trade['fees']:.2f}\n"
                    f"Time: {trade['timestamp'].strftime('%H:%M:%S')}"
                )
            
            self.send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {e}")
    
    def get_portfolio_summary(self):
        """Get current portfolio summary"""
        total_position_value = 0
        open_positions = 0
        
        for symbol, position in self.positions.items():
            if position['amount'] > 0:
                current_price = self.current_prices.get(symbol, position['entry_price'])
                position_value = position['amount'] * current_price
                total_position_value += position_value
                open_positions += 1
        
        total_portfolio_value = self.balance + total_position_value
        total_pnl = total_portfolio_value - self.initial_balance
        
        return {
            'balance': self.balance,
            'position_value': total_position_value,
            'total_value': total_portfolio_value,
            'total_pnl': total_pnl,
            'pnl_pct': (total_pnl / self.initial_balance) * 100,
            'open_positions': open_positions,
            'total_trades': len(self.trade_history)
        }
    
    def send_hourly_summary(self):
        """Send hourly portfolio summary"""
        if not self.telegram_bot:
            return
            
        try:
            summary = self.get_portfolio_summary()
            pnl_emoji = "🟢" if summary['total_pnl'] > 0 else "🔴"
            
            message = (
                f"📊 *HOURLY SUMMARY*\n\n"
                f"Balance: €{summary['balance']:.2f}\n"
                f"Positions: €{summary['position_value']:.2f}\n"
                f"Total: €{summary['total_value']:.2f}\n"
                f"{pnl_emoji} P&L: €{summary['total_pnl']:.2f} ({summary['pnl_pct']:.2f}%)\n\n"
                f"Open Positions: {summary['open_positions']}\n"
                f"Total Trades: {summary['total_trades']}"
            )
            
            self.send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending hourly summary: {e}")

    def run(self):
        """Run the paper trader"""
        self.running = True
        
        # Fetch initial historical data
        self.fetch_historical_data()
        
        # Start websocket connection
        self.start_websocket()
        
        # Set up hourly summary timer
        def hourly_summary():
            while self.running:
                time.sleep(3600)  # 1 hour
                self.send_hourly_summary()
        
        summary_thread = threading.Thread(target=hourly_summary)
        summary_thread.daemon = True
        summary_thread.start()
        
        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down paper trader...")
            self.running = False
            if self.websocket:
                self.websocket.close()