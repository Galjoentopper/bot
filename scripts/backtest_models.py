import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

warnings.filterwarnings('ignore')

@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_time: datetime
    entry_price: float
    direction: str  # 'BUY' or 'SELL'
    confidence: float
    position_size: float
    stop_loss: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL_EXIT', 'WINDOW_END'
    pnl: Optional[float] = None
    fees: float = 0.0
    slippage: float = 0.0

class BacktestConfig:
    """Configuration for backtesting parameters"""
    def __init__(self):
        # Trading parameters
        self.initial_capital = 10000.0
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_positions = 10
        self.max_trades_per_hour = 3
        
        # Execution parameters
        self.trading_fee = 0.002  # 0.2% per trade
        self.slippage = 0.001  # 0.1% slippage
        self.stop_loss_pct = 0.03  # 3% stop loss
        
        # Signal thresholds - EXTREMELY aggressive for 20+ trades/month target
        self.buy_threshold = 0.505   # Almost neutral for maximum trades
        self.sell_threshold = 0.495  # Almost neutral for maximum trades 
        self.lstm_delta_threshold = 0.0005  # Extremely sensitive
        
        # Adaptive thresholds for when trades are scarce
        self.min_trades_per_month = 20
        self.adaptive_mode = True
        
        # Walk-forward parameters
        self.train_months = 4
        self.test_months = 1
        self.slide_months = 1
        
        # Data parameters
        # Sequence length must match the value used during model training.
        # Models in this repository were trained with 96 timesteps, so use the
        # same length here to avoid shape mismatches at inference time.
        self.sequence_length = 96
        self.price_change_threshold = 0.002
        
        # Output control
        self.verbose = True  # Control detailed output during backtesting

class TechnicalIndicators:
    """Calculate technical indicators for features"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

class ModelBacktester:
    """Main backtesting engine"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = defaultdict(list)
        self.trades_history = []
        self.equity_curve = []
        self.daily_returns = []
        self.xgb_feature_columns: Dict[int, List[str]] = {}
        
        # Create directories
        os.makedirs('backtests', exist_ok=True)
        for symbol in ['ADAEUR', 'BTCEUR', 'ETHEUR', 'SOLEUR', 'XRPEUR']:
            os.makedirs(f'backtests/{symbol}', exist_ok=True)
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data from SQLite database"""
        db_path = f'data/{symbol.lower()}_15m.db'
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM market_data ORDER BY timestamp"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def load_models(self, symbol: str, window_num: int):
        """Load LSTM and XGBoost models for a specific window - FIXED VERSION"""
        try:
            # Load LSTM model
            lstm_path = f'models/lstm/{symbol.lower()}_window_{window_num}.keras'
            lstm_model = None
            if os.path.exists(lstm_path):
                try:
                    import tensorflow as tf
                    from tensorflow.keras.models import load_model

                    # Try different loading approaches for compatibility
                    try:
                        # First try: standard loading
                        from train_hybrid_models import directional_loss
                        lstm_model = load_model(
                            lstm_path,
                            custom_objects={"directional_loss": directional_loss}
                        )
                    except Exception as e1:
                        try:
                            # Second try: with compile=False for compatibility
                            lstm_model = load_model(lstm_path, compile=False, 
                                                  custom_objects={"directional_loss": directional_loss})
                            # Recompile with current TensorFlow version
                            lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                        except Exception as e2:
                            if self.config.verbose:
                                print(f"    LSTM model loading failed for window {window_num}: {e1}")
                                print(f"    Alternative loading also failed: {e2}")
                            lstm_model = None

                    if lstm_model is not None and self.config.verbose:
                        print(f"    LSTM model loaded successfully for window {window_num}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"    LSTM model loading failed for window {window_num}: {e}")
                    lstm_model = None
            else:
                if self.config.verbose:
                    print(f"    LSTM model file not found: {lstm_path}")
                # Try to find any available LSTM model for this symbol
                lstm_fallback_path = self._find_fallback_model(symbol, 'lstm', '.keras')
                if lstm_fallback_path:
                    try:
                        if self.config.verbose:
                            print(f"    Trying fallback LSTM model: {lstm_fallback_path}")
                        import tensorflow as tf
                        from tensorflow.keras.models import load_model
                        from train_hybrid_models import directional_loss
                        lstm_model = load_model(
                            lstm_fallback_path,
                            custom_objects={"directional_loss": directional_loss}
                        )
                        if self.config.verbose:
                            print(f"    ✅ Fallback LSTM model loaded successfully")
                    except Exception as e:
                        if self.config.verbose:
                            print(f"    ❌ Fallback LSTM model loading failed: {e}")
                        lstm_model = None

            # Load XGBoost model
            xgb_path = f'models/xgboost/{symbol.lower()}_window_{window_num}.json'
            xgb_model = None
            if os.path.exists(xgb_path):
                try:
                    xgb_model = xgb.XGBClassifier()
                    xgb_model.load_model(xgb_path)
                    if self.config.verbose:
                        print(f"    XGBoost model loaded successfully for window {window_num}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"    XGBoost model loading failed for window {window_num}: {e}")
                    xgb_model = None
            else:
                if self.config.verbose:
                    print(f"    XGBoost model file not found: {xgb_path}")
                # Try to find any available XGBoost model for this symbol
                xgb_fallback_path = self._find_fallback_model(symbol, 'xgboost', '.json')
                if xgb_fallback_path:
                    try:
                        if self.config.verbose:
                            print(f"    Trying fallback XGBoost model: {xgb_fallback_path}")
                        xgb_model = xgb.XGBClassifier()
                        xgb_model.load_model(xgb_fallback_path)
                        if self.config.verbose:
                            print(f"    ✅ Fallback XGBoost model loaded successfully")
                    except Exception as e:
                        if self.config.verbose:
                            print(f"    ❌ Fallback XGBoost model loading failed: {e}")
                        xgb_model = None

            # Load or create fitted scaler - THIS IS THE KEY FIX
            scaler = self._load_or_create_scaler(symbol, window_num)

            # Load feature column list for XGBoost
            fc_path = f"models/feature_columns/{symbol.lower()}_window_{window_num}_selected.pkl"
            if os.path.exists(fc_path):
                try:
                    with open(fc_path, 'rb') as f:
                        self.xgb_feature_columns[window_num] = pickle.load(f)
                    if self.config.verbose:
                        print(f"    ✅ Loaded feature columns for window {window_num}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"    ❌ Failed to load feature columns {fc_path}: {e}")
            else:
                if self.config.verbose:
                    print(f"    ⚠️ Feature columns file not found: {fc_path}")
                # Try to find fallback feature columns
                fc_fallback_path = self._find_fallback_feature_columns(symbol)
                if fc_fallback_path:
                    try:
                        if self.config.verbose:
                            print(f"    Trying fallback feature columns: {fc_fallback_path}")
                        with open(fc_fallback_path, 'rb') as f:
                            self.xgb_feature_columns[window_num] = pickle.load(f)
                        if self.config.verbose:
                            print(f"    ✅ Fallback feature columns loaded successfully")
                    except Exception as e:
                        if self.config.verbose:
                            print(f"    ❌ Fallback feature columns loading failed: {e}")

            return lstm_model, xgb_model, scaler

        except Exception as e:
            print(f"Error loading models for {symbol} window {window_num}: {e}")
            # Even in error case, return a fitted scaler
            scaler = self._load_or_create_scaler(symbol, window_num)
            return None, None, scaler

    def _load_or_create_scaler(self, symbol: str, window_num: int) -> StandardScaler:
        """Load existing scaler or create a fitted one"""

        scaler_path = f'models/scalers/{symbol.lower()}_window_{window_num}_scaler.pkl'

        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

                if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                    if self.config.verbose:
                        print(f"    ✅ Loaded fitted scaler for {symbol} window {window_num}")
                    return scaler
                else:
                    if self.config.verbose:
                        print(f"    ⚠️ Loaded scaler is not fitted, will create new one")

            except Exception as e:
                if self.config.verbose:
                    print(f"    ❌ Failed to load scaler {scaler_path}: {e}")
        else:
            if self.config.verbose:
                print(f"    ⚠️ Scaler file not found: {scaler_path}")

        return self._create_fitted_scaler(symbol)

    def _create_fitted_scaler(self, symbol: str) -> StandardScaler:
        """Create a fitted scaler using available historical data"""

        try:
            import sqlite3
            db_path = f"data/{symbol.lower()}_15m.db"

            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                query = """
                SELECT * FROM market_data
                ORDER BY timestamp
                LIMIT 5000
                """
                df = pd.read_sql_query(query, conn)
                conn.close()

                if len(df) > 100:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    df = df.fillna(method="ffill").dropna()
                    df = self.calculate_features(df)

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

                    feature_data = df[lstm_features].dropna()

                    if len(feature_data) > 50:
                        scaler = StandardScaler()
                        scaler.fit(feature_data.values)
                        if self.config.verbose:
                            print(
                                f"    ✅ Created fitted scaler using {len(feature_data)} historical samples"
                            )
                        return scaler

        except Exception as e:
            print(f"    ❌ Could not load historical data for {symbol}: {e}")

        print(f"    ⚠️ Creating pass-through scaler for {symbol}")
        return self._create_passthrough_scaler()

    def _create_passthrough_scaler(self) -> StandardScaler:
        """Create a 'scaler' that doesn't actually scale (identity transformation)"""

        scaler = StandardScaler()
        scaler.mean_ = np.zeros(17)
        scaler.scale_ = np.ones(17)
        scaler.var_ = np.ones(17)
        scaler.n_samples_seen_ = 1000

        return scaler

    def check_scalers(self, symbol: str):
        """Check status of all scalers for a symbol"""

        print(f"\n🔍 Checking scalers for {symbol}:")

        import glob
        pattern = f"models/scalers/{symbol.lower()}_window_*_scaler.pkl"
        scaler_files = glob.glob(pattern)

        if not scaler_files:
            print(f"  ❌ No scaler files found")
            return

        for file in scaler_files:
            try:
                window = file.split('_window_')[1].split('_scaler')[0]
                with open(file, 'rb') as f:
                    scaler = pickle.load(f)

                is_fitted = hasattr(scaler, 'scale_') and scaler.scale_ is not None
                status = "✅ Fitted" if is_fitted else "❌ Not fitted"
                print(f"  Window {window}: {status}")

            except Exception as e:
                print(f"  Window {window}: ❌ Error - {e}")

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and features."""
        from paper_trader.models.feature_engineer import FeatureEngineer

        fe = FeatureEngineer()
        engineered = fe.engineer_features(df)
        if engineered is None:
            raise ValueError("Feature engineering failed due to insufficient data")
        return engineered
    
    def create_lstm_sequences(self, data: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
        """Create sequences for LSTM prediction."""

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

        feature_data = data[lstm_features].fillna(method="ffill").dropna()

        if len(feature_data) == 0:
            if self.config.verbose:
                print("    ❌ No valid data for LSTM sequences")
            return np.array([])

        if not hasattr(scaler, 'scale_') or scaler.scale_ is None:
            if self.config.verbose:
                print("    ⚠️ Scaler not fitted, fitting on available data...")
            scaler.fit(feature_data.values)

        try:
            scaled_data = scaler.transform(feature_data.values)
            if self.config.verbose:
                print(f"    ✅ Successfully scaled {len(feature_data)} data points")
        except Exception as e:
            if self.config.verbose:
                print(f"    ❌ Scaling failed: {e}, using unscaled data")
            scaled_data = feature_data.values

        sequences = []
        for i in range(self.config.sequence_length, len(scaled_data)):
            sequences.append(scaled_data[i-self.config.sequence_length:i])

        if len(sequences) == 0:
            if self.config.verbose:
                print(f"    ❌ Not enough data for sequences (need {self.config.sequence_length + 1}+)")
            return np.array([])

        if self.config.verbose:
            print(f"    ✅ Created {len(sequences)} LSTM sequences")
        return np.array(sequences)
    
    def get_xgb_features(self, data: pd.DataFrame, lstm_delta: float, window_num: int) -> np.ndarray:
        """Get feature vector for XGBoost prediction"""
        feature_columns = self.xgb_feature_columns.get(window_num)
        if not feature_columns:
            # Fallback: Use a basic set of features when specific window features are not available
            basic_features = ['close', 'volume'] if 'close' in data.columns and 'volume' in data.columns else []
            if basic_features:
                features = [float(data[col].iloc[-1]) for col in basic_features]
                features.append(float(lstm_delta))
                return np.array(features, dtype=float).reshape(1, -1)
            else:
                raise ValueError(f"Feature columns for window {window_num} not loaded and no basic features available")
        
        # Add lstm_delta to features
        features = []
        last_row = data.iloc[-1]
        
        # Check which columns are available
        available_cols = set(data.columns)
        missing_cols = [col for col in feature_columns if col not in available_cols]
        
        if missing_cols:
            print(f"      ⚠️ Missing feature columns: {missing_cols[:5]}... ({len(missing_cols)} total)")
            # Use only available columns
            feature_columns = [col for col in feature_columns if col in available_cols]
            print(f"      Using {len(feature_columns)} available features instead")
        
        for col in feature_columns:
            val = last_row[col]
            if isinstance(val, (list, tuple, np.ndarray)):
                # Flatten any accidental sequences and take the first element
                val = np.asarray(val).flatten()[0]
            features.append(float(val))

        # append lstm_delta ensuring it is numeric
        features.append(float(lstm_delta))

        result = np.array(features, dtype=float).reshape(1, -1)
        return result
    
    def generate_signal(self, xgb_prob: float, lstm_delta: float) -> str:
        """Generate trading signal based on model outputs with multiple fallback methods"""
        
        # PRIMARY: Both models must agree (original logic but more relaxed)
        primary_buy = (xgb_prob > self.config.buy_threshold and 
                      lstm_delta > self.config.lstm_delta_threshold)
        primary_sell = (xgb_prob < self.config.sell_threshold and 
                       lstm_delta < -self.config.lstm_delta_threshold)
        
        if primary_buy:
            return 'BUY'
        elif primary_sell:
            return 'SELL'
        
        # SECONDARY: Either model can trigger (relaxed thresholds)
        secondary_buy_threshold = max(0.51, self.config.buy_threshold - 0.05)
        secondary_sell_threshold = min(0.49, self.config.sell_threshold + 0.05)
        secondary_lstm_threshold = max(0.003, self.config.lstm_delta_threshold * 0.4)
        
        secondary_buy = (xgb_prob > secondary_buy_threshold or 
                        lstm_delta > secondary_lstm_threshold)
        secondary_sell = (xgb_prob < secondary_sell_threshold or 
                         lstm_delta < -secondary_lstm_threshold)
        
        if secondary_buy:
            return 'BUY'
        elif secondary_sell:
            return 'SELL'
        
        # TERTIARY: Strong momentum-based signals (more sensitive)
        strong_lstm_threshold = 0.005  # Much lower for more trades
        if abs(lstm_delta) > strong_lstm_threshold:
            if lstm_delta > 0:
                return 'BUY'
            else:
                return 'SELL'
        
        # QUATERNARY: Very loose XGBoost signals
        if xgb_prob > 0.505:  # Just slightly above neutral
            return 'BUY'
        elif xgb_prob < 0.495:  # Just slightly below neutral
            return 'SELL'
        
        # QUINARY: Any detectable movement
        if abs(lstm_delta) > 0.0005:  # Extremely sensitive
            if lstm_delta > 0:
                return 'BUY'
            else:
                return 'SELL'
        
        return 'HOLD'
    
    def generate_signal_with_adaptive_stats(self, xgb_prob: float, lstm_delta: float, 
                                          stats: dict, aggressiveness_multiplier: float = 1.0) -> str:
        """Generate trading signal with adaptive aggressiveness based on trade frequency"""
        
        # Apply aggressiveness multiplier to make thresholds more permissive when behind target
        adaptive_buy_threshold = max(0.505, self.config.buy_threshold - (0.02 * aggressiveness_multiplier))
        adaptive_sell_threshold = min(0.495, self.config.sell_threshold + (0.02 * aggressiveness_multiplier))
        adaptive_lstm_threshold = max(0.0005, self.config.lstm_delta_threshold - (0.003 * aggressiveness_multiplier))
        
        # PRIMARY: Both models must agree (adaptive thresholds)
        primary_buy = (xgb_prob > adaptive_buy_threshold and 
                      lstm_delta > adaptive_lstm_threshold)
        primary_sell = (xgb_prob < adaptive_sell_threshold and 
                       lstm_delta < -adaptive_lstm_threshold)
        
        if primary_buy:
            stats['BUY'] += 1
            stats['primary_signals'] += 1
            return 'BUY'
        elif primary_sell:
            stats['SELL'] += 1
            stats['primary_signals'] += 1
            return 'SELL'
        
        # SECONDARY: Either model can trigger (even more relaxed with aggressiveness)
        secondary_buy_threshold = max(0.505, adaptive_buy_threshold - 0.03)
        secondary_sell_threshold = min(0.495, adaptive_sell_threshold + 0.03)
        secondary_lstm_threshold = max(0.0005, adaptive_lstm_threshold * 0.3)
        
        secondary_buy = (xgb_prob > secondary_buy_threshold or 
                        lstm_delta > secondary_lstm_threshold)
        secondary_sell = (xgb_prob < secondary_sell_threshold or 
                         lstm_delta < -secondary_lstm_threshold)
        
        if secondary_buy:
            stats['BUY'] += 1
            stats['secondary_signals'] += 1
            return 'BUY'
        elif secondary_sell:
            stats['SELL'] += 1
            stats['secondary_signals'] += 1
            return 'SELL'
        
        # TERTIARY: Strong momentum-based signals (adaptive sensitivity)
        tertiary_threshold = max(0.0005, 0.005 - (0.002 * aggressiveness_multiplier))
        if abs(lstm_delta) > tertiary_threshold:
            if lstm_delta > 0:
                stats['BUY'] += 1
                stats['tertiary_signals'] += 1
                return 'BUY'
            else:
                stats['SELL'] += 1
                stats['tertiary_signals'] += 1
                return 'SELL'
        
        # QUATERNARY: Very loose XGBoost signals (adaptive)
        quaternary_buy_threshold = max(0.502, 0.505 - (0.001 * aggressiveness_multiplier))
        quaternary_sell_threshold = min(0.498, 0.495 + (0.001 * aggressiveness_multiplier))
        
        if xgb_prob > quaternary_buy_threshold:
            stats['BUY'] += 1
            stats.setdefault('quaternary_signals', 0)
            stats['quaternary_signals'] += 1
            return 'BUY'
        elif xgb_prob < quaternary_sell_threshold:
            stats['SELL'] += 1
            stats.setdefault('quaternary_signals', 0)
            stats['quaternary_signals'] += 1
            return 'SELL'
        
        # QUINARY: Any detectable movement (extremely aggressive when behind)
        quinary_threshold = max(0.0001, 0.0005 - (0.0002 * aggressiveness_multiplier))
        if abs(lstm_delta) > quinary_threshold:
            if lstm_delta > 0:
                stats['BUY'] += 1
                stats.setdefault('quinary_signals', 0)
                stats['quinary_signals'] += 1
                return 'BUY'
            else:
                stats['SELL'] += 1
                stats.setdefault('quinary_signals', 0)
                stats['quinary_signals'] += 1
                return 'SELL'
        
        # DESPERATION MODE: When very aggressive, any tiny movement triggers trades
        if aggressiveness_multiplier > 2.0:
            # Essentially random signal generation when far behind target
            if xgb_prob > 0.5001:  # Extremely tiny threshold
                stats['BUY'] += 1
                stats.setdefault('desperation_signals', 0)
                stats['desperation_signals'] += 1
                return 'BUY'
            elif xgb_prob < 0.4999:
                stats['SELL'] += 1
                stats.setdefault('desperation_signals', 0)
                stats['desperation_signals'] += 1
                return 'SELL'
        
        stats['HOLD'] += 1
        return 'HOLD'
        """Generate trading signal and update statistics with adaptive frequency targeting"""
        
        # PRIMARY: Both models must agree (relaxed thresholds)
        primary_buy = (xgb_prob > self.config.buy_threshold and 
                      lstm_delta > self.config.lstm_delta_threshold)
        primary_sell = (xgb_prob < self.config.sell_threshold and 
                       lstm_delta < -self.config.lstm_delta_threshold)
        
        if primary_buy:
            stats['BUY'] += 1
            stats['primary_signals'] += 1
            return 'BUY'
        elif primary_sell:
            stats['SELL'] += 1
            stats['primary_signals'] += 1
            return 'SELL'
        
        # SECONDARY: Either model can trigger (relaxed thresholds)
        secondary_buy_threshold = max(0.51, self.config.buy_threshold - 0.05)
        secondary_sell_threshold = min(0.49, self.config.sell_threshold + 0.05)
        secondary_lstm_threshold = max(0.003, self.config.lstm_delta_threshold * 0.4)
        
        secondary_buy = (xgb_prob > secondary_buy_threshold or 
                        lstm_delta > secondary_lstm_threshold)
        secondary_sell = (xgb_prob < secondary_sell_threshold or 
                         lstm_delta < -secondary_lstm_threshold)
        
        if secondary_buy:
            stats['BUY'] += 1
            stats['secondary_signals'] += 1
            return 'BUY'
        elif secondary_sell:
            stats['SELL'] += 1
            stats['secondary_signals'] += 1
            return 'SELL'
        
        # TERTIARY: Strong momentum-based signals (more sensitive)
        strong_lstm_threshold = 0.005  # Much lower for more trades
        if abs(lstm_delta) > strong_lstm_threshold:
            if lstm_delta > 0:
                stats['BUY'] += 1
                stats['tertiary_signals'] += 1
                return 'BUY'
            else:
                stats['SELL'] += 1
                stats['tertiary_signals'] += 1
                return 'SELL'
        
        # QUATERNARY: Very loose XGBoost signals
        if xgb_prob > 0.505:  # Just slightly above neutral
            stats['BUY'] += 1
            stats.setdefault('quaternary_signals', 0)
            stats['quaternary_signals'] += 1
            return 'BUY'
        elif xgb_prob < 0.495:  # Just slightly below neutral
            stats['SELL'] += 1
            stats.setdefault('quaternary_signals', 0)
            stats['quaternary_signals'] += 1
            return 'SELL'
        
        # QUINARY: Any detectable movement (extremely aggressive)
        if abs(lstm_delta) > 0.0005:  # Extremely sensitive
            if lstm_delta > 0:
                stats['BUY'] += 1
                stats.setdefault('quinary_signals', 0)
                stats['quinary_signals'] += 1
                return 'BUY'
            else:
                stats['SELL'] += 1
                stats.setdefault('quinary_signals', 0)
                stats['quinary_signals'] += 1
                return 'SELL'
        
        stats['HOLD'] += 1
        return 'HOLD'
    
    def calculate_position_size(self, capital: float, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = capital * self.config.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        if price_risk > 0:
            position_size = risk_amount / price_risk
            return min(position_size, capital * 0.1)  # Max 10% of capital per trade
        return 0
    
    def apply_slippage_and_fees(self, price: float, direction: str) -> Tuple[float, float, float]:
        """Apply slippage and fees to trade execution"""
        slippage_amount = price * self.config.slippage
        if direction == 'BUY':
            execution_price = price * (1 + self.config.slippage)
        else:
            execution_price = price * (1 - self.config.slippage)
        
        fees = execution_price * self.config.trading_fee
        
        return execution_price, slippage_amount, fees
    
    def backtest_symbol(self, symbol: str, progress_callback=None) -> Dict:
        """Backtest a single symbol using walk-forward validation"""
        print(f"\nBacktesting {symbol}...")
        
        # Load data
        data = self.load_data(symbol)
        data = self.calculate_features(data)
        
        print(f"  Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
        
        # Initialize portfolio
        capital = self.config.initial_capital
        positions = []
        trades = []
        equity_history = []
        
        # Walk-forward validation
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Calculate total expected windows for progress tracking
        total_months = ((end_date - start_date).days // 30)
        estimated_windows = max(1, (total_months - self.config.train_months) // self.config.slide_months)
        
        window_num = 1
        current_date = start_date
        
        while current_date < end_date:
            # Update progress callback if available
            if progress_callback:
                progress_callback(symbol, window_num, estimated_windows, None, None)
            
            # Define training and testing periods
            train_start = current_date
            train_end = train_start + timedelta(days=30 * self.config.train_months)
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.config.test_months)
            
            if test_end > end_date:
                break
            
            print(f"  Window {window_num}: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
            
            # Get test data
            test_data = data[(data.index >= test_start) & (data.index < test_end)].copy()
            
            if len(test_data) < self.config.sequence_length + 1:
                current_date += timedelta(days=30 * self.config.slide_months)
                window_num += 1
                continue
            
            # Load models for this window
            lstm_model, xgb_model, scaler = self.load_models(symbol, window_num)
            
            if xgb_model is None:
                print(f"    XGBoost model not found for window {window_num}, skipping...")
                current_date += timedelta(days=30 * self.config.slide_months)
                window_num += 1
                continue
            
            # Allow running with just XGBoost if LSTM is not available
            if lstm_model is None and self.config.verbose:
                print(f"    ⚠️ LSTM model not available for window {window_num}, using XGBoost-only mode...")
            
            # Simulate trading for this window
            window_trades, window_capital = self.simulate_trading_window(
                test_data, lstm_model, xgb_model, scaler,
                symbol, capital, positions, window_num, progress_callback, estimated_windows
            )
            
            trades.extend(window_trades)
            capital = window_capital
            equity_history.append({
                'date': test_end,
                'capital': capital,
                'window': window_num
            })
            
            # Move to next window
            current_date += timedelta(days=30 * self.config.slide_months)
            window_num += 1
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(trades, equity_history, symbol)
        
        return {
            'symbol': symbol,
            'trades': trades,
            'equity_history': equity_history,
            'performance': performance,
            'final_capital': capital
        }
    
    def simulate_trading_window(self, data: pd.DataFrame, lstm_model, xgb_model, scaler,
                              symbol: str, initial_capital: float, positions: List[Trade],
                              window_num: int, progress_callback=None, total_windows=None) -> Tuple[List[Trade], float]:
        """Simulate trading for a single window with adaptive frequency targeting"""
        capital = initial_capital
        window_trades = []
        trades_this_hour = 0
        last_trade_hour = None
        
        # Signal generation counters for debugging
        signal_stats = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'primary_signals': 0, 
                       'secondary_signals': 0, 'tertiary_signals': 0}
        
        total_steps = len(data) - self.config.sequence_length
        progress_update_frequency = max(1, total_steps // 20)  # Update 20 times per window
        
        # Adaptive frequency tracking - target 20+ trades per month
        days_in_window = len(data) / (24 * 4)  # 15-min intervals: 4 per hour, 24 hours
        target_trades_this_window = max(1, int((self.config.min_trades_per_month * days_in_window) / 30))
        trades_generated = 0
        
        if self.config.verbose:
            print(f"        🎯 Target trades for this window ({days_in_window:.1f} days): {target_trades_this_window}")
        
        for i in range(self.config.sequence_length, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Update progress periodically
            current_step = i - self.config.sequence_length + 1
            if progress_callback and (current_step % progress_update_frequency == 0 or current_step == total_steps):
                progress_callback(symbol, window_num, total_windows, current_step, total_steps)
            
            # Reset hourly trade counter
            if last_trade_hour is None or current_time.hour != last_trade_hour:
                trades_this_hour = 0
                last_trade_hour = current_time.hour
            
            # Check for exits first
            positions, exit_trades = self.check_exits(positions, current_time, current_price)
            window_trades.extend(exit_trades)
            trades_generated += len(exit_trades)
            
            # Update capital from closed trades
            for trade in exit_trades:
                capital += trade.pnl
            
            # Skip if we've hit trade limits
            if (len(positions) >= self.config.max_positions or 
                trades_this_hour >= self.config.max_trades_per_hour):
                continue
            
            # Adaptive thresholds based on trade frequency
            progress_ratio = current_step / total_steps
            expected_trades_by_now = target_trades_this_window * progress_ratio
            trade_deficit = max(0, expected_trades_by_now - trades_generated)
            
            # Become more aggressive if behind target
            aggressiveness_multiplier = 1.0 + (trade_deficit * 0.1)  # 10% more aggressive per missing trade
            
            # Generate predictions
            try:
                # LSTM prediction (if model is available)
                if lstm_model is not None:
                    lstm_sequence = self.create_lstm_sequences(
                        data.iloc[i-self.config.sequence_length:i+1], scaler
                    )[-1:]
                    lstm_pred = lstm_model.predict(lstm_sequence, verbose=0)[0][0]
                    lstm_delta = lstm_pred  # LSTM already outputs percentage change directly
                else:
                    # Use neutral lstm_delta when LSTM model is not available
                    lstm_delta = 0.0
                
                # XGBoost prediction
                try:
                    xgb_features = self.get_xgb_features(data.iloc[:i+1], lstm_delta, window_num)
                    xgb_prob = xgb_model.predict_proba(xgb_features)[0][1]
                except Exception as xgb_error:
                    # Fallback: Use simple price-based signal when XGBoost features don't match
                    if "Feature shape mismatch" in str(xgb_error) or "expected" in str(xgb_error):
                        # Simple momentum-based probability
                        returns = data['returns'].iloc[-10:].mean() if 'returns' in data.columns else 0
                        xgb_prob = 0.5 + (returns * 10)  # Convert return to probability
                        xgb_prob = max(0, min(1, xgb_prob))  # Clamp to [0, 1]
                    else:
                        raise xgb_error
                
                # Generate signal with adaptive statistics tracking
                signal = self.generate_signal_with_adaptive_stats(
                    xgb_prob, lstm_delta, signal_stats, aggressiveness_multiplier
                )
                
                # Enhanced debug signal generation (print more frequently for debugging)
                if i % 500 == 0 and self.config.verbose:  # More frequent updates
                    print(f"      Debug at {current_time}: XGB={xgb_prob:.3f}, LSTM={lstm_delta:.6f}, Signal={signal}")
                    print(f"        Trades: {trades_generated}/{target_trades_this_window} (deficit={trade_deficit:.1f}, aggr={aggressiveness_multiplier:.2f})")
                    
                if signal in ['BUY', 'SELL']:
                    # Calculate stop loss
                    atr = data['atr'].iloc[i]
                    if signal == 'BUY':
                        stop_loss = current_price - (atr * 2)
                    else:
                        stop_loss = current_price + (atr * 2)
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(capital, current_price, stop_loss)
                    
                    if position_size > 0:
                        # Execute trade
                        execution_price, slippage, fees = self.apply_slippage_and_fees(current_price, signal)
                        
                        trade = Trade(
                            symbol=symbol,
                            entry_time=current_time,
                            entry_price=execution_price,
                            direction=signal,
                            confidence=max(xgb_prob, 1-xgb_prob),
                            position_size=position_size,
                            stop_loss=stop_loss,
                            fees=fees,
                            slippage=slippage
                        )
                        
                        positions.append(trade)
                        capital -= (position_size * execution_price + fees)
                        trades_this_hour += 1
                        trades_generated += 1
                        
            except Exception as e:
                print(f"    Error generating signal at {current_time}: {e}")
                continue
        
        # Close remaining positions at window end
        final_time = data.index[-1]
        final_price = data['close'].iloc[-1]
        positions, final_trades = self.check_exits(positions, final_time, final_price, force_exit=True)
        window_trades.extend(final_trades)
        trades_generated += len(final_trades)
        
        # Update capital from final trades
        for trade in final_trades:
            capital += trade.pnl
        
        # Debug output for trade generation with enhanced signal statistics
        if self.config.verbose:
            total_signals = signal_stats['BUY'] + signal_stats['SELL'] + signal_stats['HOLD']
            print(f"        🎯 Trade target: {target_trades_this_window}, Generated: {trades_generated}")
            
            if trades_generated == 0:
                print(f"        ⚠️  No trades generated in window {window_num} (TARGET MISSED)")
                print(f"        Signal stats: BUY={signal_stats['BUY']}, SELL={signal_stats['SELL']}, HOLD={signal_stats['HOLD']} (total={total_signals})")
            elif trades_generated < target_trades_this_window:
                print(f"        ⚠️  Generated {trades_generated} trades in window {window_num} (BELOW TARGET)")
                print(f"        Signal stats: BUY={signal_stats['BUY']}, SELL={signal_stats['SELL']}, HOLD={signal_stats['HOLD']} (total={total_signals})")
            else:
                print(f"        ✅ Generated {trades_generated} trades in window {window_num} (TARGET MET)")
                print(f"        Signal stats: BUY={signal_stats['BUY']}, SELL={signal_stats['SELL']}, HOLD={signal_stats['HOLD']} (total={total_signals})")
            
            # Show signal tier breakdown
            tier_stats = []
            for tier in ['primary_signals', 'secondary_signals', 'tertiary_signals', 'quaternary_signals', 'quinary_signals']:
                count = signal_stats.get(tier, 0)
                if count > 0:
                    tier_stats.append(f"{tier.replace('_signals', '').title()}={count}")
            if tier_stats:
                print(f"        Signal tiers: {', '.join(tier_stats)}")
        
        return window_trades, capital
    
    def check_exits(self, positions: List[Trade], current_time: datetime, 
                   current_price: float, force_exit: bool = False) -> Tuple[List[Trade], List[Trade]]:
        """Check for trade exits (stop loss, take profit, or forced exit)"""
        remaining_positions = []
        closed_trades = []
        
        for trade in positions:
            should_exit = False
            exit_reason = None
            
            if force_exit:
                should_exit = True
                exit_reason = 'WINDOW_END'
            elif trade.direction == 'BUY':
                if current_price <= trade.stop_loss:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'
            elif trade.direction == 'SELL':
                if current_price >= trade.stop_loss:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'
            
            if should_exit:
                # Execute exit
                exit_price, slippage, fees = self.apply_slippage_and_fees(
                    current_price, 'SELL' if trade.direction == 'BUY' else 'BUY'
                )
                
                # Calculate PnL
                if trade.direction == 'BUY':
                    pnl = (exit_price - trade.entry_price) * trade.position_size
                else:
                    pnl = (trade.entry_price - exit_price) * trade.position_size
                
                pnl -= (trade.fees + fees)  # Subtract entry and exit fees
                
                trade.exit_time = current_time
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl = pnl
                trade.fees += fees
                trade.slippage += slippage
                
                closed_trades.append(trade)
            else:
                remaining_positions.append(trade)
        
        return remaining_positions, closed_trades
    
    def calculate_performance_metrics(self, trades: List[Trade], equity_history: List[Dict], symbol: str) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {}
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = sum(t.pnl for t in trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        
        # Equity curve analysis
        if equity_history:
            equity_values = [e['capital'] for e in equity_history]
            returns = np.diff(equity_values) / equity_values[:-1]
            
            # Sharpe ratio (assuming 252 trading days, but we have fewer data points)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Total return
            total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            total_return = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }
    
    def save_results(self, results: Dict):
        """Save backtest results to files"""
        symbol = results['symbol']
        
        # Save trades to CSV
        if results['trades']:
            trades_df = pd.DataFrame([
                {
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'position_size': t.position_size,
                    'pnl': t.pnl,
                    'confidence': t.confidence,
                    'exit_reason': t.exit_reason,
                    'fees': t.fees,
                    'slippage': t.slippage
                } for t in results['trades']
            ])
            trades_df.to_csv(f'backtests/{symbol}/trades.csv', index=False)
        
        # Save equity curve
        if results['equity_history']:
            equity_df = pd.DataFrame(results['equity_history'])
            equity_df.to_csv(f'backtests/{symbol}/equity_curve.csv', index=False)
        
        # Save performance metrics
        with open(f'backtests/{symbol}/performance_metrics.json', 'w') as f:
            json.dump(results['performance'], f, indent=2)
        
        print(f"Results saved for {symbol}")
    
    def plot_results(self, results: Dict):
        """Create visualization plots"""
        symbol = results['symbol']
        
        if not results['equity_history']:
            return
        
        # Create equity curve plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{symbol} Backtest Results', fontsize=16)
        
        # Equity curve
        equity_df = pd.DataFrame(results['equity_history'])
        ax1.plot(equity_df['date'], equity_df['capital'])
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital')
        ax1.grid(True)
        
        # Trade distribution
        if results['trades']:
            pnls = [t.pnl for t in results['trades']]
            ax2.hist(pnls, bins=30, alpha=0.7)
            ax2.set_title('Trade PnL Distribution')
            ax2.set_xlabel('PnL')
            ax2.set_ylabel('Frequency')
            ax2.axvline(x=0, color='red', linestyle='--')
            ax2.grid(True)
            
            # Monthly returns
            trades_df = pd.DataFrame([
                {'date': t.exit_time, 'pnl': t.pnl} for t in results['trades'] if t.exit_time
            ])
            if not trades_df.empty:
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                monthly_pnl = trades_df.groupby(trades_df['date'].dt.to_period('M'))['pnl'].sum()
                ax3.bar(range(len(monthly_pnl)), monthly_pnl.values)
                ax3.set_title('Monthly PnL')
                ax3.set_xlabel('Month')
                ax3.set_ylabel('PnL')
                ax3.grid(True)
            
            # Win/Loss ratio by confidence
            confidences = [t.confidence for t in results['trades']]
            pnls = [t.pnl for t in results['trades']]
            colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
            ax4.scatter(confidences, pnls, c=colors, alpha=0.6)
            ax4.set_title('PnL vs Confidence')
            ax4.set_xlabel('Confidence')
            ax4.set_ylabel('PnL')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'backtests/{symbol}/performance_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved for {symbol}")
    
    def _find_fallback_model(self, symbol: str, model_type: str, extension: str) -> Optional[str]:
        """Find any available model file for the given symbol and model type"""
        import glob
        pattern = f'models/{model_type}/{symbol.lower()}_window_*{extension}'
        model_files = glob.glob(pattern)
        if model_files:
            return model_files[0]  # Return the first available model
        return None
    
    def _find_fallback_feature_columns(self, symbol: str) -> Optional[str]:
        """Find any available feature columns file for the given symbol"""
        import glob
        pattern = f'models/feature_columns/{symbol.lower()}_window_*_selected.pkl'
        fc_files = glob.glob(pattern)
        if fc_files:
            return fc_files[0]  # Return the first available feature columns
        return None
    
    def run_backtest(self, symbols: List[str] = None, progress_callback=None) -> Dict:
        """Run backtest for all specified symbols"""
        if symbols is None:
            symbols = ['ADAEUR', 'BTCEUR', 'ETHEUR', 'SOLEUR']
        
        all_results = {}
        
        for symbol_idx, symbol in enumerate(symbols):
            try:
                if progress_callback:
                    # Estimate total windows for this symbol (rough estimate)
                    progress_callback(symbol, 1, "~60", None, None)
                
                results = self.backtest_symbol(symbol, progress_callback)
                all_results[symbol] = results
                
                # Save and plot results
                self.save_results(results)
                self.plot_results(results)
                
                # Print summary
                perf = results['performance']
                print(f"\n{symbol} Summary:")
                print(f"  Total Trades: {perf.get('total_trades', 0)}")
                print(f"  Win Rate: {perf.get('win_rate', 0):.2%}")
                print(f"  Total PnL: {perf.get('total_pnl', 0):.2f}")
                print(f"  Total Return: {perf.get('total_return', 0):.2%}")
                print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
                
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
                continue
        
        return all_results

if __name__ == "__main__":
    # Initialize configuration
    config = BacktestConfig()
    
    # Create backtester
    backtester = ModelBacktester(config)
    
    # Run backtest
    print("Starting backtest...")
    results = backtester.run_backtest()
    
    print("\nBacktest completed!")
    print(f"Results saved in 'backtests/' directory")