#!/usr/bin/env python3
"""
Paper Trading Bot using Hybrid LSTM + XGBoost Models
====================================================

This paper trader implements the trained hybrid models from train_hybrid_models.py
to make trading decisions on cryptocurrency pairs. It loads the models, processes
real-time data, and simulates trading with performance tracking.

Features:
- Loads trained LSTM and XGBoost models for each symbol
- Processes 15-minute market data with technical indicators
- Makes buy/sell decisions based on model predictions
- Tracks portfolio performance and statistics
- Supports multiple confidence thresholds
- Simple and focused implementation
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ML imports
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    timestamp: datetime
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    confidence: float
    lstm_prediction: float
    xgb_probability: float
    
@dataclass
class Position:
    """Represents a current position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    entry_confidence: float
    
@dataclass
class PortfolioStats:
    """Portfolio performance statistics"""
    total_value: float
    cash: float
    positions_value: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float

class ModelLoader:
    """Handles loading and managing trained models"""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        
    def load_model_safe(self, filepath: str):
        """Safely load XGBoost or calibrated models"""
        try:
            # Try different extensions if the exact path doesn't exist
            if not os.path.exists(filepath):
                # Try with different extensions
                for ext in ['.pkl', '.json']:
                    alt_path = f"{filepath}{ext}"
                    if os.path.exists(alt_path):
                        filepath = alt_path
                        break
                else:
                    logger.error(f"Model file not found: {filepath}")
                    return None
            
            if filepath.endswith('.json'):
                # XGBoost model
                model = xgb.XGBClassifier()
                model.load_model(filepath)
                return model
            elif filepath.endswith('.pkl'):
                # Calibrated/sklearn model
                return joblib.load(filepath)
            else:
                # Try both methods
                try:
                    model = xgb.XGBClassifier()
                    model.load_model(filepath)
                    return model
                except:
                    return joblib.load(filepath)
                    
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            return None
    
    def load_models_for_symbol(self, symbol: str) -> bool:
        """Load all models for a specific symbol"""
        symbol_lower = symbol.lower()
        
        # Load LSTM model (try multiple approaches)
        lstm_path = os.path.join(self.models_dir, "lstm", f"{symbol_lower}_lstm.h5")
        if not os.path.exists(lstm_path):
            # Try with .keras extension
            lstm_path = os.path.join(self.models_dir, "lstm", f"{symbol_lower}_lstm.keras")
        
        if os.path.exists(lstm_path):
            try:
                # Try loading without custom objects first (for inference only)
                try:
                    self.models[f"{symbol}_lstm"] = tf.keras.models.load_model(lstm_path, compile=False)
                    logger.info(f"✅ Loaded LSTM model for {symbol}")
                except Exception as e1:
                    logger.warning(f"⚠️ Could not load LSTM model for {symbol}: {e1}")
                    logger.info(f"ℹ️ Skipping LSTM model for {symbol} - will use XGBoost only")
                    # Continue without LSTM model for now
            except Exception as e:
                logger.error(f"❌ Failed to load LSTM model for {symbol}: {e}")
                logger.info(f"ℹ️ Continuing without LSTM model for {symbol}")
        else:
            logger.warning(f"❌ LSTM model not found for {symbol}")
            logger.info(f"ℹ️ Continuing without LSTM model for {symbol}")
        
        # Load XGBoost model
        xgb_path = os.path.join(self.models_dir, "xgboost", f"{symbol_lower}_xgboost")
        xgb_model = self.load_model_safe(xgb_path)
        if xgb_model is not None:
            self.models[f"{symbol}_xgb"] = xgb_model
            logger.info(f"✅ Loaded XGBoost model for {symbol}")
        else:
            logger.error(f"❌ Failed to load XGBoost model for {symbol}")
            return False
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, "scalers", f"{symbol_lower}_scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scalers[symbol] = pickle.load(f)
                logger.info(f"✅ Loaded scaler for {symbol}")
            except Exception as e:
                logger.error(f"❌ Failed to load scaler for {symbol}: {e}")
                if f"{symbol}_lstm" not in self.models:  # If no LSTM, scaler is critical
                    return False
        else:
            logger.error(f"❌ Scaler not found for {symbol}")
            if f"{symbol}_lstm" not in self.models:  # If no LSTM, scaler is critical
                return False
        
        # Load feature columns (try to find the correct one for the main model)
        feature_dir = os.path.join(self.models_dir, "feature_columns")
        if os.path.exists(feature_dir):
            # For main models (not window-specific), try to find the matching feature file
            # First, try to find a feature file that matches the expected number of features
            feature_files = [f for f in os.listdir(feature_dir) if f.startswith(f"{symbol_lower}_window") and f.endswith("_selected.pkl")]
            
            if feature_files:
                # Sort files and try to find the right one
                feature_files.sort()
                
                # If we have the XGBoost model loaded, try to match the feature count
                xgb_model = self.models.get(f"{symbol}_xgb")
                if xgb_model is not None:
                    expected_features = None
                    if hasattr(xgb_model, 'n_features_in_'):
                        expected_features = xgb_model.n_features_in_
                    elif hasattr(xgb_model, 'base_estimator') and hasattr(xgb_model.base_estimator, 'n_features_in_'):
                        expected_features = xgb_model.base_estimator.n_features_in_
                    
                    if expected_features:
                        logger.info(f"🔍 Looking for feature file with {expected_features} features for {symbol}")
                        
                        # Find feature file with matching count
                        for feature_file in feature_files:
                            try:
                                feature_path = os.path.join(feature_dir, feature_file)
                                with open(feature_path, 'rb') as f:
                                    features = pickle.load(f)
                                if len(features) == expected_features:
                                    self.feature_columns[symbol] = features
                                    logger.info(f"✅ Loaded feature columns for {symbol} from {feature_file} ({len(features)} features)")
                                    return True
                            except Exception as e:
                                logger.debug(f"Could not load {feature_file}: {e}")
                                continue
                
                # If no exact match found, use the most recent window
                feature_path = os.path.join(feature_dir, feature_files[-1])
                try:
                    with open(feature_path, 'rb') as f:
                        self.feature_columns[symbol] = pickle.load(f)
                    logger.info(f"✅ Loaded feature columns for {symbol} from {feature_files[-1]} (fallback)")
                except Exception as e:
                    logger.error(f"❌ Failed to load feature columns for {symbol}: {e}")
                    return False
            else:
                logger.error(f"❌ Feature columns not found for {symbol}")
                return False
        else:
            logger.error(f"❌ Feature columns directory not found")
            return False
        
        return True
    
    def load_all_models(self, symbols: List[str]) -> bool:
        """Load models for all symbols"""
        success_count = 0
        for symbol in symbols:
            if self.load_models_for_symbol(symbol):
                success_count += 1
            else:
                logger.warning(f"⚠️ Failed to load models for {symbol}")
        
        logger.info(f"📊 Loaded models for {success_count}/{len(symbols)} symbols")
        return success_count > 0

class TechnicalIndicators:
    """Technical indicators calculation (simplified version from train_hybrid_models.py)"""
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features needed for the models"""
        data = df.copy()
        
        # Basic price features
        data["returns"] = data["close"].pct_change()
        data["log_returns"] = np.log(data["close"] / data["close"].shift(1))
        data["price_change_1h"] = data["close"].pct_change(4)  # 4 * 15min = 1h
        data["price_change_4h"] = data["close"].pct_change(16)  # 16 * 15min = 4h
        data["price_change_24h"] = data["close"].pct_change(96)  # 96 * 15min = 24h
        data["price_change_30min"] = data["close"].pct_change(2)  # 2 * 15min = 30min
        
        # Volatility features
        data["volatility_15min"] = data["returns"].rolling(4).std()
        data["volatility_30min"] = data["returns"].rolling(8).std()
        data["volatility_1h"] = data["returns"].rolling(16).std()
        data["volatility_4h"] = data["returns"].rolling(64).std()
        data["volatility_20"] = data["returns"].rolling(20).std()
        data["volatility_50"] = data["returns"].rolling(50).std()
        data["volatility_ratio"] = data["volatility_20"] / data["volatility_50"]
        
        # Volatility ratios
        data["vol_ratio_15min_30min"] = data["volatility_15min"] / data["volatility_30min"]
        data["vol_ratio_30min_1h"] = data["volatility_30min"] / data["volatility_1h"]
        data["vol_ratio_1h_4h"] = data["volatility_1h"] / data["volatility_4h"]
        
        # Price normalization
        data["price_zscore_20"] = (data["close"] - data["close"].rolling(20).mean()) / data["close"].rolling(20).std()
        data["price_zscore_50"] = (data["close"] - data["close"].rolling(50).mean()) / data["close"].rolling(50).std()
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            data[f"returns_lag_{lag}"] = data["returns"].shift(lag)
            data[f"log_returns_lag_{lag}"] = data["log_returns"].shift(lag)
        
        # Rolling statistics
        data["returns_mean_10"] = data["returns"].rolling(10).mean()
        data["returns_std_10"] = data["returns"].rolling(10).std()
        data["returns_skew_20"] = data["returns"].rolling(20).skew()
        data["returns_kurt_20"] = data["returns"].rolling(20).kurt()
        
        # Volume features
        data["volume_sma_20"] = data["volume"].rolling(20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma_20"]
        data["volume_change"] = data["volume"].pct_change()
        data["volume_zscore"] = (data["volume"] - data["volume"].rolling(20).mean()) / data["volume"].rolling(20).std()
        data["volume_price_trend"] = data["volume"] * data["returns"]
        data["volume_weighted_price"] = (data["volume"] * data["close"]).rolling(20).sum() / data["volume"].rolling(20).sum()
        
        # Market microstructure
        data["spread"] = (data["high"] - data["low"]) / data["close"]
        data["spread_ma"] = data["spread"].rolling(20).mean()
        data["spread_ratio"] = data["spread"] / data["spread_ma"]
        data["buying_pressure"] = (data["close"] - data["low"]) / (data["high"] - data["low"])
        data["selling_pressure"] = (data["high"] - data["close"]) / (data["high"] - data["low"])
        data["net_pressure"] = data["buying_pressure"] - data["selling_pressure"]
        
        # Moving averages
        data["ema_9"] = data["close"].ewm(span=9).mean()
        data["ema_21"] = data["close"].ewm(span=21).mean()
        data["ema_50"] = data["close"].ewm(span=50).mean()
        data["ema_100"] = data["close"].ewm(span=100).mean()
        data["sma_200"] = data["close"].rolling(200).mean()
        
        # Multi-timeframe EMAs
        data["ema_30min"] = data["close"].ewm(span=2).mean()
        data["ema_1h"] = data["close"].ewm(span=4).mean()
        data["ema_2h"] = data["close"].ewm(span=8).mean()
        data["ema_4h"] = data["close"].ewm(span=16).mean()
        
        # Price vs EMAs
        data["price_vs_ema9"] = (data["close"] - data["ema_9"]) / data["ema_9"]
        data["price_vs_ema21"] = (data["close"] - data["ema_21"]) / data["ema_21"]
        data["price_vs_ema50"] = (data["close"] - data["ema_50"]) / data["ema_50"]
        data["price_vs_sma200"] = (data["close"] - data["sma_200"]) / data["sma_200"]
        data["price_vs_ema_30min"] = (data["close"] - data["ema_30min"]) / data["ema_30min"]
        data["price_vs_ema_1h"] = (data["close"] - data["ema_1h"]) / data["ema_1h"]
        data["price_vs_ema_2h"] = (data["close"] - data["ema_2h"]) / data["ema_2h"]
        data["price_vs_ema_4h"] = (data["close"] - data["ema_4h"]) / data["ema_4h"]
        
        # MA relationships
        data["ema9_vs_ema21"] = (data["ema_9"] - data["ema_21"]) / data["ema_21"]
        data["ema21_vs_ema50"] = (data["ema_21"] - data["ema_50"]) / data["ema_50"]
        data["ema50_vs_ema100"] = (data["ema_50"] - data["ema_100"]) / data["ema_100"]
        
        # Trend features
        data["ma_alignment"] = (
            (data["ema_9"] > data["ema_21"]) & 
            (data["ema_21"] > data["ema_50"]) & 
            (data["ema_50"] > data["ema_100"])
        ).astype(int)
        data["ma_slope_9"] = data["ema_9"].pct_change(5)
        data["ma_slope_21"] = data["ema_21"].pct_change(5)
        
        # RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))
        data["rsi_9"] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(9).mean() / (-delta.where(delta < 0, 0)).rolling(9).mean()))
        data["rsi_21"] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(21).mean() / (-delta.where(delta < 0, 0)).rolling(21).mean()))
        data["rsi_oversold"] = (data["rsi"] < 30).astype(int)
        data["rsi_overbought"] = (data["rsi"] > 70).astype(int)
        data["rsi_divergence"] = data["rsi"].diff(5) * data["close"].pct_change(5)
        
        # Stochastic
        low_14 = data["low"].rolling(14).min()
        high_14 = data["high"].rolling(14).max()
        data["stoch_k"] = 100 * ((data["close"] - low_14) / (high_14 - low_14))
        data["stoch_d"] = data["stoch_k"].rolling(3).mean()
        data["stoch_oversold"] = (data["stoch_k"] < 20).astype(int)
        data["stoch_overbought"] = (data["stoch_k"] > 80).astype(int)
        
        # Williams %R
        data["williams_r"] = -100 * ((high_14 - data["close"]) / (high_14 - low_14))
        
        # MACD
        ema_12 = data["close"].ewm(span=12).mean()
        ema_26 = data["close"].ewm(span=26).mean()
        data["macd"] = ema_12 - ema_26
        data["macd_signal"] = data["macd"].ewm(span=9).mean()
        data["macd_histogram"] = data["macd"] - data["macd_signal"]
        data["macd_bullish"] = (data["macd"] > data["macd_signal"]).astype(int)
        
        # Bollinger Bands
        bb_ma = data["close"].rolling(20).mean()
        bb_std = data["close"].rolling(20).std()
        data["bb_upper"] = bb_ma + (bb_std * 2)
        data["bb_lower"] = bb_ma - (bb_std * 2)
        data["bb_middle"] = bb_ma
        data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]
        data["bb_position"] = (data["close"] - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])
        
        # VWAP
        data["vwap"] = (data["close"] * data["volume"]).cumsum() / data["volume"].cumsum()
        data["price_vs_vwap"] = (data["close"] - data["vwap"]) / data["vwap"]
        
        # Candle patterns
        data["candle_body"] = abs(data["close"] - data["open"]) / data["open"]
        data["upper_wick"] = (data["high"] - np.maximum(data["open"], data["close"])) / data["open"]
        data["lower_wick"] = (np.minimum(data["open"], data["close"]) - data["low"]) / data["open"]
        
        # Time features
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)
        
        # Momentum
        data["momentum_10"] = data["close"].diff(10)
        data["momentum_30min"] = data["close"].diff(2)
        data["momentum_1h"] = data["close"].diff(4)
        data["momentum_2h"] = data["close"].diff(8)
        data["momentum_4h"] = data["close"].diff(16)
        data["momentum_alignment_short"] = ((data["momentum_30min"] > 0) & (data["momentum_1h"] > 0)).astype(int)
        data["momentum_alignment_all"] = (
            (data["momentum_30min"] > 0) & 
            (data["momentum_1h"] > 0) & 
            (data["momentum_2h"] > 0) & 
            (data["momentum_4h"] > 0)
        ).astype(int)
        data["momentum_ratio_30min_1h"] = data["momentum_30min"] / (data["momentum_1h"] + 1e-8)
        data["momentum_ratio_1h_2h"] = data["momentum_1h"] / (data["momentum_2h"] + 1e-8)
        data["momentum_ratio_2h_4h"] = data["momentum_2h"] / (data["momentum_4h"] + 1e-8)
        
        # ROC
        data["roc_10"] = ((data["close"] - data["close"].shift(10)) / data["close"].shift(10)) * 100
        
        # Support/Resistance
        data["high_20"] = data["high"].rolling(20).max()
        data["low_20"] = data["low"].rolling(20).min()
        data["near_resistance"] = (data["close"] / data["high_20"] > 0.98).astype(int)
        data["near_support"] = (data["close"] / data["low_20"] < 1.02).astype(int)
        
        # ATR
        tr1 = data["high"] - data["low"]
        tr2 = abs(data["high"] - data["close"].shift())
        tr3 = abs(data["low"] - data["close"].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr"] = true_range.rolling(14).mean()
        data["atr_ratio"] = data["atr"] / data["close"]
        data["realized_vol_5"] = data["returns"].rolling(5).std() * np.sqrt(5)
        data["realized_vol_20"] = data["returns"].rolling(20).std() * np.sqrt(20)
        data["vol_regime"] = (data["volatility_20"] > data["volatility_20"].rolling(100).quantile(0.75)).astype(int)
        
        # Feature interactions
        data["rsi_macd_combo"] = data["rsi"] * data["macd_signal"]
        data["volatility_ema_ratio"] = data["volatility_20"] / data["ema_21"]
        data["volume_price_momentum"] = data["volume_ratio"] * data["momentum_10"]
        data["bb_rsi_signal"] = data["bb_position"] * data["rsi"]
        data["trend_strength"] = data["price_vs_ema9"] * data["price_vs_ema21"]
        data["volatility_breakout"] = data["atr"] * data["bb_width"]
        data["volume_surge_5"] = data["volume"] / data["volume"].rolling(5).mean()
        data["momentum_acceleration"] = data["momentum_10"].diff(5)
        data["market_momentum_alignment"] = (
            (data["momentum_30min"] > 0) & 
            (data["momentum_1h"] > 0) & 
            (data["momentum_2h"] > 0) & 
            (data["momentum_4h"] > 0) &
            (data["ma_alignment"] == 1)
        ).astype(int)
        data["momentum_vol_signal"] = data["momentum_10"] * data["volume_ratio"] * data["volatility_ratio"]
        data["trend_momentum_align"] = data["ma_alignment"] * data["momentum_10"]
        data["pressure_volume_signal"] = data["net_pressure"] * data["volume_zscore"]
        data["volatility_regime_signal"] = data["vol_regime"] * data["rsi"]
        data["multi_timeframe_signal"] = data["price_change_1h"] * data["price_change_4h"] * data["price_change_24h"]
        data["oscillator_consensus"] = (data["rsi_oversold"] + data["stoch_oversold"]) - (data["rsi_overbought"] + data["stoch_overbought"])
        data["trend_regime"] = ((data["ma_alignment"] == 1) & (data["price_vs_sma200"] > 0)).astype(int)
        data["consolidation_regime"] = (
            (data["bb_width"] < data["bb_width"].rolling(50).quantile(0.3)) &
            (data["atr_ratio"] < data["atr_ratio"].rolling(50).quantile(0.3))
        ).astype(int)
        
        return data

class PaperTrader:
    """Main paper trading bot"""
    
    def __init__(self, 
                 symbols: List[str] = None,
                 data_dir: str = "data",
                 models_dir: str = "models",
                 initial_cash: float = 10000.0,
                 position_size: float = 0.1,  # 10% of portfolio per trade
                 confidence_threshold: float = 0.6,
                 min_confidence: float = 0.55,
                 stop_loss: float = 0.05,  # 5% stop loss
                 take_profit: float = 0.10,  # 10% take profit
                 max_positions: int = 3):
        
        self.symbols = symbols or ["BTCEUR", "ETHEUR", "ADAEUR", "SOLEUR", "XRPEUR"]
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.confidence_threshold = confidence_threshold
        self.min_confidence = min_confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_positions = max_positions
        
        # Trading state
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        
        # Components
        self.model_loader = ModelLoader(models_dir)
        self.technical_indicators = TechnicalIndicators()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.last_update = None
        
        logger.info(f"🤖 Paper Trader initialized with {len(self.symbols)} symbols")
        logger.info(f"💰 Initial cash: €{initial_cash:,.2f}")
        logger.info(f"📊 Position size: {position_size*100:.1f}% of portfolio")
        logger.info(f"🎯 Confidence threshold: {confidence_threshold:.1%}")
    
    def load_models(self) -> bool:
        """Load all trading models"""
        logger.info("📚 Loading trading models...")
        return self.model_loader.load_all_models(self.symbols)
    
    def get_latest_data(self, symbol: str, lookback_hours: int = 48) -> pd.DataFrame:
        """Get latest market data for a symbol"""
        db_path = os.path.join(self.data_dir, f"{symbol.lower()}_15m.db")
        
        if not os.path.exists(db_path):
            logger.error(f"❌ Database not found: {db_path}")
            return pd.DataFrame()
        
        # Get data from the last lookback_hours
        lookback_minutes = lookback_hours * 60
        lookback_periods = lookback_minutes // 15  # 15-minute intervals
        
        conn = sqlite3.connect(db_path)
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        ORDER BY timestamp DESC
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(lookback_periods,))
        conn.close()
        
        if df.empty:
            logger.warning(f"⚠️ No data found for {symbol}")
            return pd.DataFrame()
        
        # Sort by timestamp ascending
        df = df.sort_values('timestamp')
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def prepare_lstm_sequence(self, df: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
        """Prepare LSTM input sequence from dataframe"""
        if len(df) < 120:  # Need at least 120 periods for LSTM
            logger.warning(f"⚠️ Insufficient data for LSTM sequence: {len(df)} < 120")
            return None
        
        # Check if scaler is available
        scaler = self.model_loader.scalers.get(symbol)
        if scaler is None:
            logger.warning(f"⚠️ No scaler found for {symbol}")
            return None
        
        # Calculate technical features
        df_features = self.technical_indicators.calculate_all_features(df)
        
        # LSTM features (same as in train_hybrid_models.py)
        lstm_features = [
            "close", "volume", "returns", "log_returns", "volume_ratio",
            "price_change_1h", "price_change_4h", "price_change_24h",
            "volatility_20", "volatility_1h", "volatility_4h", "atr_ratio",
            "rsi", "rsi_9", "stoch_k", "williams_r",
            "macd", "macd_histogram", "momentum_10", "price_vs_ema9",
            "price_vs_ema21", "price_vs_ema50", "bb_position", "bb_width",
            "buying_pressure", "selling_pressure", "spread_ratio", "volume_price_trend",
            "vol_regime", "trend_regime", "ma_alignment", "price_zscore_20", "price_zscore_50",
            "volume_surge_5", "volatility_breakout", "momentum_acceleration", "market_momentum_alignment"
        ]
        
        # Check available features
        available_features = [f for f in lstm_features if f in df_features.columns]
        if len(available_features) < len(lstm_features) * 0.8:  # At least 80% of features
            logger.warning(f"⚠️ Only {len(available_features)}/{len(lstm_features)} LSTM features available")
        
        # Get feature data and handle missing values
        feature_data = df_features[available_features].ffill().fillna(0).values
        
        # Get the last sequence
        sequence = feature_data[-120:]  # Last 120 periods
        
        if np.isnan(sequence).any() or np.isinf(sequence).any():
            logger.warning(f"⚠️ Invalid values in LSTM sequence for {symbol}")
            # Replace invalid values
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=1e6, neginf=-1e6)
        
        try:
            sequence_scaled = scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
        except Exception as e:
            logger.warning(f"⚠️ Scaling failed for {symbol}: {e}")
            return None
        
        # Add batch dimension
        return sequence_scaled.reshape(1, 120, -1)  # (1, timesteps, features)
    
    def prepare_xgboost_features(self, df: pd.DataFrame, lstm_prediction: float, symbol: str) -> Optional[np.ndarray]:
        """Prepare XGBoost features including LSTM delta"""
        if len(df) < 100:  # Reduced from 200 to 100 periods
            logger.warning(f"⚠️ Insufficient data for XGBoost features: {len(df)} < 100")
            return None
        
        # Calculate technical features
        df_features = self.technical_indicators.calculate_all_features(df)
        
        # Add LSTM prediction as feature
        df_features['lstm_delta'] = lstm_prediction
        
        # Get feature columns for this symbol
        feature_columns = self.model_loader.feature_columns.get(symbol)
        if feature_columns is None:
            logger.error(f"❌ No feature columns found for {symbol}")
            return None
        
        # Get the latest row of features
        latest_features = df_features.iloc[-1]
        
        # Select only the required features in the exact order
        try:
            feature_values = []
            missing_features = []
            for feature in feature_columns:
                if feature in latest_features.index:
                    value = latest_features[feature]
                    # Handle NaN values
                    if pd.isna(value):
                        value = 0.0
                    feature_values.append(float(value))
                else:
                    missing_features.append(feature)
                    feature_values.append(0.0)  # Use 0 for missing features
            
            if missing_features:
                logger.warning(f"⚠️ Missing features for {symbol}: {missing_features}")
            
            feature_array = np.array(feature_values)
            
            # Handle invalid values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
            
            logger.debug(f"✅ {symbol}: Prepared {len(feature_array)} features")
            
            return feature_array.reshape(1, -1)  # (1, features)
            
        except Exception as e:
            logger.error(f"❌ Feature preparation failed for {symbol}: {e}")
            return None
    
    def make_prediction(self, symbol: str) -> Optional[Tuple[float, float, float]]:
        """Make prediction for a symbol using hybrid models"""
        # Get latest data
        df = self.get_latest_data(symbol, lookback_hours=48)
        if df.empty:
            return None
        
        current_price = df['close'].iloc[-1]
        lstm_pred = 0.0  # Default LSTM prediction
        
        # Try LSTM prediction if model is available
        lstm_model = self.model_loader.models.get(f"{symbol}_lstm")
        if lstm_model is not None:
            # Prepare LSTM sequence
            lstm_sequence = self.prepare_lstm_sequence(df, symbol)
            if lstm_sequence is not None:
                try:
                    lstm_pred = lstm_model.predict(lstm_sequence, verbose=0)[0][0]
                except Exception as e:
                    logger.warning(f"⚠️ LSTM prediction failed for {symbol}: {e}")
                    lstm_pred = 0.0
            else:
                logger.warning(f"⚠️ Could not prepare LSTM sequence for {symbol}")
        else:
            logger.debug(f"ℹ️ No LSTM model available for {symbol}, using XGBoost only")
        
        # Prepare XGBoost features
        xgb_features = self.prepare_xgboost_features(df, lstm_pred, symbol)
        if xgb_features is None:
            return None
        
        # Get XGBoost prediction
        xgb_model = self.model_loader.models.get(f"{symbol}_xgb")
        if xgb_model is None:
            logger.error(f"❌ No XGBoost model found for {symbol}")
            return None
        
        try:
            xgb_proba = xgb_model.predict_proba(xgb_features)[0][1]  # Probability of positive class
        except Exception as e:
            logger.error(f"❌ XGBoost prediction failed for {symbol}: {e}")
            return None
        
        return current_price, lstm_pred, xgb_proba
    
    def should_buy(self, symbol: str, xgb_probability: float) -> bool:
        """Determine if we should buy based on prediction and current state"""
        # Check if we already have a position
        if symbol in self.positions:
            return False
        
        # Check if we have too many positions
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check if we have enough cash
        position_value = self.cash * self.position_size
        if position_value < 100:  # Minimum position size
            return False
        
        # Check confidence threshold
        return xgb_probability >= self.confidence_threshold
    
    def should_sell(self, symbol: str, current_price: float, xgb_probability: float) -> bool:
        """Determine if we should sell based on prediction and position"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        price_change = (current_price - position.entry_price) / position.entry_price
        
        # Stop loss
        if price_change <= -self.stop_loss:
            logger.info(f"🛑 Stop loss triggered for {symbol}: {price_change:.2%}")
            return True
        
        # Take profit
        if price_change >= self.take_profit:
            logger.info(f"🎯 Take profit triggered for {symbol}: {price_change:.2%}")
            return True
        
        # Model says sell (low confidence)
        if xgb_probability < self.min_confidence:
            logger.info(f"📉 Model confidence low for {symbol}: {xgb_probability:.2%}")
            return True
        
        return False
    
    def execute_buy(self, symbol: str, price: float, lstm_pred: float, xgb_proba: float) -> bool:
        """Execute a buy order"""
        position_value = self.cash * self.position_size
        quantity = position_value / price
        
        if quantity * price > self.cash:
            logger.warning(f"⚠️ Insufficient cash for {symbol} buy order")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            entry_confidence=xgb_proba
        )
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            timestamp=datetime.now(),
            action="BUY",
            price=price,
            quantity=quantity,
            confidence=xgb_proba,
            lstm_prediction=lstm_pred,
            xgb_probability=xgb_proba
        )
        
        # Update state
        self.cash -= quantity * price
        self.positions[symbol] = position
        self.trades.append(trade)
        
        logger.info(f"✅ BUY {symbol}: {quantity:.6f} @ €{price:.4f} (conf: {xgb_proba:.2%})")
        return True
    
    def execute_sell(self, symbol: str, price: float, lstm_pred: float, xgb_proba: float) -> bool:
        """Execute a sell order"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        trade_value = position.quantity * price
        pnl = trade_value - (position.quantity * position.entry_price)
        pnl_pct = pnl / (position.quantity * position.entry_price)
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            timestamp=datetime.now(),
            action="SELL",
            price=price,
            quantity=position.quantity,
            confidence=xgb_proba,
            lstm_prediction=lstm_pred,
            xgb_probability=xgb_proba
        )
        
        # Update state
        self.cash += trade_value
        del self.positions[symbol]
        self.trades.append(trade)
        
        logger.info(f"✅ SELL {symbol}: {position.quantity:.6f} @ €{price:.4f} (PnL: €{pnl:.2f}, {pnl_pct:.2%})")
        return True
    
    def update_portfolio_stats(self):
        """Update portfolio statistics"""
        # Calculate current positions value
        positions_value = 0.0
        for symbol, position in self.positions.items():
            current_data = self.get_latest_data(symbol, lookback_hours=1)
            if not current_data.empty:
                current_price = current_data['close'].iloc[-1]
                positions_value += position.quantity * current_price
        
        total_value = self.cash + positions_value
        
        # Calculate trade statistics
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        
        # Group trades by symbol to calculate PnL
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i] if self.trades[i].action == "BUY" else self.trades[i + 1]
                sell_trade = self.trades[i + 1] if self.trades[i + 1].action == "SELL" else self.trades[i]
                
                if buy_trade.action == "BUY" and sell_trade.action == "SELL":
                    pnl = (sell_trade.price - buy_trade.price) * buy_trade.quantity
                    total_pnl += pnl
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        total_completed_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_completed_trades if total_completed_trades > 0 else 0.0
        
        # Calculate max drawdown (simplified)
        portfolio_values = [record['total_value'] for record in self.portfolio_history]
        if portfolio_values:
            peak = max(portfolio_values)
            current_dd = (peak - total_value) / peak if peak > 0 else 0.0
            max_drawdown = max([record.get('drawdown', 0) for record in self.portfolio_history] + [current_dd])
        else:
            max_drawdown = 0.0
        
        # Update portfolio history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'num_positions': len(self.positions),
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'drawdown': max_drawdown
        })
        
        self.last_update = datetime.now()
    
    def get_portfolio_stats(self) -> PortfolioStats:
        """Get current portfolio statistics"""
        # Calculate current positions value
        positions_value = 0.0
        for symbol, position in self.positions.items():
            current_data = self.get_latest_data(symbol, lookback_hours=1)
            if not current_data.empty:
                current_price = current_data['close'].iloc[-1]
                positions_value += position.quantity * current_price
        
        total_value = self.cash + positions_value
        
        # Calculate trade statistics
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        wins = []
        losses = []
        
        # Group trades by symbol to calculate PnL
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i] if self.trades[i].action == "BUY" else self.trades[i + 1]
                sell_trade = self.trades[i + 1] if self.trades[i + 1].action == "SELL" else self.trades[i]
                
                if buy_trade.action == "BUY" and sell_trade.action == "SELL":
                    pnl = (sell_trade.price - buy_trade.price) * buy_trade.quantity
                    total_pnl += pnl
                    
                    if pnl > 0:
                        winning_trades += 1
                        wins.append(pnl)
                    else:
                        losing_trades += 1
                        losses.append(pnl)
        
        total_completed_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_completed_trades if total_completed_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Calculate max drawdown
        portfolio_values = [record['total_value'] for record in self.portfolio_history]
        if portfolio_values:
            peak = max(portfolio_values)
            max_drawdown = max([(peak - val) / peak for val in portfolio_values])
        else:
            max_drawdown = 0.0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.portfolio_history) > 1:
            returns = [(self.portfolio_history[i]['total_value'] / self.portfolio_history[i-1]['total_value'] - 1) 
                      for i in range(1, len(self.portfolio_history))]
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return PortfolioStats(
            total_value=total_value,
            cash=self.cash,
            positions_value=positions_value,
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )
    
    def run_trading_cycle(self):
        """Run one complete trading cycle for all symbols"""
        logger.info("🔄 Running trading cycle...")
        
        for symbol in self.symbols:
            try:
                # Make prediction
                prediction = self.make_prediction(symbol)
                if prediction is None:
                    logger.warning(f"⚠️ Could not make prediction for {symbol}")
                    continue
                
                current_price, lstm_pred, xgb_proba = prediction
                
                logger.info(f"📊 {symbol}: Price=€{current_price:.4f}, LSTM={lstm_pred:.4f}, XGB={xgb_proba:.2%}")
                
                # Trading logic
                if self.should_buy(symbol, xgb_proba):
                    self.execute_buy(symbol, current_price, lstm_pred, xgb_proba)
                elif self.should_sell(symbol, current_price, xgb_proba):
                    self.execute_sell(symbol, current_price, lstm_pred, xgb_proba)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        # Update portfolio statistics
        self.update_portfolio_stats()
        
        # Log current status
        stats = self.get_portfolio_stats()
        logger.info(f"💰 Portfolio: €{stats.total_value:.2f} (Cash: €{stats.cash:.2f}, Positions: €{stats.positions_value:.2f})")
        logger.info(f"📈 Trades: {stats.total_trades} total, {stats.winning_trades} wins, {stats.losing_trades} losses (WR: {stats.win_rate:.1%})")
        if stats.total_pnl != 0:
            logger.info(f"💵 P&L: €{stats.total_pnl:.2f} ({(stats.total_value/self.initial_cash-1):.2%} return)")
    
    def save_results(self, filename_prefix: str = "paper_trading_results"):
        """Save trading results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades
        trades_data = [asdict(trade) for trade in self.trades]
        trades_df = pd.DataFrame(trades_data)
        if not trades_df.empty:
            trades_file = f"{filename_prefix}_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"💾 Trades saved to {trades_file}")
        
        # Save portfolio history
        portfolio_df = pd.DataFrame(self.portfolio_history)
        if not portfolio_df.empty:
            portfolio_file = f"{filename_prefix}_portfolio_{timestamp}.csv"
            portfolio_df.to_csv(portfolio_file, index=False)
            logger.info(f"💾 Portfolio history saved to {portfolio_file}")
        
        # Save summary statistics
        stats = self.get_portfolio_stats()
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'initial_cash': self.initial_cash,
            'final_value': stats.total_value,
            'total_return': (stats.total_value / self.initial_cash - 1),
            'total_trades': stats.total_trades,
            'winning_trades': stats.winning_trades,
            'losing_trades': stats.losing_trades,
            'win_rate': stats.win_rate,
            'total_pnl': stats.total_pnl,
            'max_drawdown': stats.max_drawdown,
            'sharpe_ratio': stats.sharpe_ratio,
            'avg_win': stats.avg_win,
            'avg_loss': stats.avg_loss,
            'symbols': self.symbols,
            'position_size': self.position_size,
            'confidence_threshold': self.confidence_threshold,
            'min_confidence': self.min_confidence,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
        
        summary_file = f"{filename_prefix}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"💾 Summary saved to {summary_file}")
        
        return trades_file if not trades_df.empty else None, portfolio_file if not portfolio_df.empty else None, summary_file

def main():
    """Main function for paper trading"""
    print("🤖 Hybrid LSTM + XGBoost Paper Trading Bot")
    print("=========================================")
    
    # Initialize paper trader
    trader = PaperTrader(
        symbols=["BTCEUR", "ETHEUR", "ADAEUR", "SOLEUR", "XRPEUR"],
        initial_cash=10000.0,
        position_size=0.2,  # 20% per position
        confidence_threshold=0.65,  # 65% confidence to buy
        min_confidence=0.55,  # Below 55% confidence to sell
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.10,  # 10% take profit
        max_positions=3  # Max 3 concurrent positions
    )
    
    # Load models
    if not trader.load_models():
        logger.error("❌ Failed to load models. Exiting.")
        return
    
    logger.info("✅ All models loaded successfully!")
    
    try:
        # Run a single trading cycle for testing
        trader.run_trading_cycle()
        
        # Display results
        stats = trader.get_portfolio_stats()
        
        print("\n📊 Trading Results Summary:")
        print("===========================")
        print(f"💰 Total Portfolio Value: €{stats.total_value:,.2f}")
        print(f"💵 Cash: €{stats.cash:,.2f}")
        print(f"📈 Positions Value: €{stats.positions_value:,.2f}")
        print(f"🔄 Total Trades: {stats.total_trades}")
        print(f"✅ Winning Trades: {stats.winning_trades}")
        print(f"❌ Losing Trades: {stats.losing_trades}")
        print(f"🎯 Win Rate: {stats.win_rate:.1%}")
        print(f"💵 Total P&L: €{stats.total_pnl:,.2f}")
        print(f"📊 Total Return: {(stats.total_value/trader.initial_cash-1):.2%}")
        
        if stats.max_drawdown > 0:
            print(f"📉 Max Drawdown: {stats.max_drawdown:.2%}")
        
        # Show current positions
        if trader.positions:
            print(f"\n🏦 Current Positions ({len(trader.positions)}):")
            for symbol, position in trader.positions.items():
                current_data = trader.get_latest_data(symbol, lookback_hours=1)
                if not current_data.empty:
                    current_price = current_data['close'].iloc[-1]
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    unrealized_pct = (current_price - position.entry_price) / position.entry_price
                    print(f"  {symbol}: {position.quantity:.6f} @ €{position.entry_price:.4f} "
                          f"(Current: €{current_price:.4f}, P&L: €{unrealized_pnl:+.2f} {unrealized_pct:+.2%})")
        
        # Save results
        trader.save_results()
        
        print(f"\n🎉 Paper trading completed successfully!")
        print(f"📁 Results saved to CSV and JSON files")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Trading interrupted by user")
        trader.save_results()
    except Exception as e:
        logger.error(f"❌ Trading error: {e}")
        raise

if __name__ == "__main__":
    main()