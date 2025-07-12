import pandas as pd
import numpy as np
import sqlite3
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import json
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
        
        # Signal thresholds
        self.buy_threshold = 0.6  # Lowered from 0.7 for more trades
        self.sell_threshold = 0.4  # Raised from 0.3 for more trades
        self.lstm_delta_threshold = 0.02  # Realistic for % changes (was 0.5)
        
        # Walk-forward parameters
        self.train_months = 4
        self.test_months = 1
        self.slide_months = 1
        
        # Data parameters
        self.sequence_length = 60
        self.price_change_threshold = 0.002

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
        """Load LSTM and XGBoost models for a specific window"""
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
                        lstm_model = load_model(lstm_path)
                    except Exception as e1:
                        try:
                            # Second try: with compile=False for compatibility
                            lstm_model = load_model(lstm_path, compile=False)
                            # Recompile with current TensorFlow version
                            lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                        except Exception as e2:
                            print(f"    LSTM model loading failed for window {window_num}: {e1}")
                            print(f"    Alternative loading also failed: {e2}")
                            lstm_model = None
                            
                    if lstm_model is not None:
                        print(f"    LSTM model loaded successfully for window {window_num}")
                except Exception as e:
                    print(f"    LSTM model loading failed for window {window_num}: {e}")
                    lstm_model = None
            else:
                print(f"    LSTM model file not found: {lstm_path}")
            
            # Load XGBoost model
            xgb_path = f'models/xgboost/{symbol.lower()}_window_{window_num}.pkl'
            xgb_model = None
            if os.path.exists(xgb_path):
                try:
                    with open(xgb_path, 'rb') as f:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            xgb_model = pickle.load(f)
                    print(f"    XGBoost model loaded successfully for window {window_num}")
                except Exception as e:
                    print(f"    XGBoost model loading failed for window {window_num}: {e}")
                    xgb_model = None
            else:
                print(f"    XGBoost model file not found: {xgb_path}")
            
            # Load scalers
            scaler_path = f'models/scalers/{symbol.lower()}_window_{window_num}_scaler.pkl'
            scaler = MinMaxScaler()
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    print(f"    Scaler loaded successfully for window {window_num}")
                except Exception as e:
                    print(f"    Scaler loading failed for window {window_num}: {e}")
            else:
                print(f"    Scaler file not found: {scaler_path}")
            
            return lstm_model, xgb_model, scaler
        except Exception as e:
            print(f"Error loading models for {symbol} window {window_num}: {e}")
            return None, None, MinMaxScaler()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and features"""
        data = df.copy()
        
        # Price features
        data['returns'] = data['close'].pct_change()
        data['price_change_1h'] = data['close'].pct_change(4)  # 4 * 15min = 1h
        data['price_change_4h'] = data['close'].pct_change(16)  # 16 * 15min = 4h
        
        # Volume features
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        data['volume_change'] = data['volume'].pct_change()
        
        # Volatility features
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['atr'] = TechnicalIndicators.calculate_atr(data['high'], data['low'], data['close'])
        
        # Moving averages
        data['sma_200'] = data['close'].rolling(200).mean()
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_21'] = data['close'].ewm(span=21).mean()
        
        # Price vs moving averages
        data['price_vs_sma200'] = (data['close'] - data['sma_200']) / data['sma_200']
        data['price_vs_ema9'] = (data['close'] - data['ema_9']) / data['ema_9']
        data['price_vs_ema21'] = (data['close'] - data['ema_21']) / data['ema_21']
        
        # RSI
        data['rsi'] = TechnicalIndicators.calculate_rsi(data['close'])
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.calculate_macd(data['close'])
        data['macd'] = macd
        data['macd_histogram'] = histogram
        data['macd_bullish'] = (macd > signal).astype(int)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = TechnicalIndicators.calculate_bollinger_bands(data['close'])
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        data['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # VWAP
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['price_vs_vwap'] = (data['close'] - data['vwap']) / data['vwap']
        
        # Momentum
        data['roc_10'] = data['close'].pct_change(10)
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        
        # Candlestick patterns
        data['candle_body'] = abs(data['close'] - data['open']) / data['open']
        data['upper_wick'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['open']
        data['lower_wick'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['open']
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # Support/Resistance (simplified)
        data['near_support'] = 0  # Placeholder
        data['near_resistance'] = 0  # Placeholder
        
        # Feature interactions (from our enhanced model)
        data['rsi_macd_combo'] = data['rsi'] * data['macd']
        data['volatility_ema_ratio'] = data['volatility_20'] / data['ema_21']
        data['volume_price_momentum'] = data['volume_ratio'] * data['momentum_10']
        data['bb_rsi_signal'] = data['bb_position'] * data['rsi']
        data['trend_strength'] = abs(data['price_vs_ema9']) * data['volume_ratio']
        data['volatility_breakout'] = (data['volatility_20'] > data['volatility_20'].rolling(20).mean()).astype(int)
        
        # Create target for validation
        data['target'] = (data['close'].pct_change().shift(-1) > self.config.price_change_threshold).astype(int)
        
        return data
    
    def create_lstm_sequences(self, data: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
        """Create sequences for LSTM prediction"""
        # Select features for LSTM (must match training: close and volume only)
        lstm_features = ['close', 'volume']
        
        # Scale the data (use transform only, scaler is already fitted)
        scaled_data = scaler.transform(data[lstm_features].fillna(method='ffill'))
        
        sequences = []
        for i in range(self.config.sequence_length, len(scaled_data)):
            sequences.append(scaled_data[i-self.config.sequence_length:i])
        
        return np.array(sequences)
    
    def get_xgb_features(self, data: pd.DataFrame, lstm_delta: float) -> np.ndarray:
        """Get feature vector for XGBoost prediction"""
        feature_columns = [
            'returns', 'price_change_1h', 'price_change_4h', 'volume_ratio', 'volume_change',
            'volatility_20', 'atr', 'price_vs_sma200', 'price_vs_ema9', 'price_vs_ema21',
            'rsi', 'rsi_overbought', 'rsi_oversold', 'macd', 'macd_histogram', 'macd_bullish',
            'bb_position', 'bb_width', 'price_vs_vwap', 'roc_10', 'momentum_10',
            'candle_body', 'upper_wick', 'lower_wick', 'hour', 'day_of_week', 'is_weekend',
            'near_support', 'near_resistance', 'rsi_macd_combo', 'volatility_ema_ratio',
            'volume_price_momentum', 'bb_rsi_signal', 'trend_strength', 'volatility_breakout'
        ]
        
        # Add lstm_delta to features
        features = data[feature_columns].iloc[-1].values.tolist()
        features.append(lstm_delta)
        
        return np.array(features).reshape(1, -1)
    
    def generate_signal(self, xgb_prob: float, lstm_delta: float) -> str:
        """Generate trading signal based on model outputs"""
        if (xgb_prob > self.config.buy_threshold and 
            lstm_delta > self.config.lstm_delta_threshold):
            return 'BUY'
        elif (xgb_prob < self.config.sell_threshold and 
              lstm_delta < -self.config.lstm_delta_threshold):
            return 'SELL'
        else:
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
    
    def backtest_symbol(self, symbol: str) -> Dict:
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
        
        window_num = 1
        current_date = start_date
        
        while current_date < end_date:
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
            
            # Skip window if either model fails to load (both models required)
            if lstm_model is None:
                print(f"    LSTM model failed to load for window {window_num}, skipping window...")
                current_date += timedelta(days=30 * self.config.slide_months)
                window_num += 1
                continue
            
            # Simulate trading for this window
            window_trades, window_capital = self.simulate_trading_window(
                test_data, lstm_model, xgb_model, scaler, symbol, capital, positions
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
                              symbol: str, initial_capital: float, positions: List[Trade]) -> Tuple[List[Trade], float]:
        """Simulate trading for a single window"""
        capital = initial_capital
        window_trades = []
        trades_this_hour = 0
        last_trade_hour = None
        
        for i in range(self.config.sequence_length, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Reset hourly trade counter
            if last_trade_hour is None or current_time.hour != last_trade_hour:
                trades_this_hour = 0
                last_trade_hour = current_time.hour
            
            # Check for exits first
            positions, exit_trades = self.check_exits(positions, current_time, current_price)
            window_trades.extend(exit_trades)
            
            # Update capital from closed trades
            for trade in exit_trades:
                capital += trade.pnl
            
            # Skip if we've hit trade limits
            if (len(positions) >= self.config.max_positions or 
                trades_this_hour >= self.config.max_trades_per_hour):
                continue
            
            # Generate predictions
            try:
                # LSTM prediction (both models are required at this point)
                lstm_sequence = self.create_lstm_sequences(
                    data.iloc[i-self.config.sequence_length:i+1], scaler
                )[-1:]
                lstm_pred = lstm_model.predict(lstm_sequence, verbose=0)[0][0]
                lstm_delta = lstm_pred  # LSTM already outputs percentage change directly
                
                # XGBoost prediction
                xgb_features = self.get_xgb_features(data.iloc[:i+1], lstm_delta)
                xgb_prob = xgb_model.predict_proba(xgb_features)[0][1]
                
                # Generate signal
                signal = self.generate_signal(xgb_prob, lstm_delta)
                
                # Debug signal generation (only print occasionally to avoid spam)
                if i % 500 == 0:  # Print every 500 iterations
                    print(f"      Debug at {current_time}: XGB_prob={xgb_prob:.3f}, LSTM_delta={lstm_delta:.6f}, Signal={signal}")
                    print(f"        Thresholds: buy={self.config.buy_threshold}, sell={self.config.sell_threshold}, lstm_delta={self.config.lstm_delta_threshold}")
                
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
                        
            except Exception as e:
                print(f"    Error generating signal at {current_time}: {e}")
                continue
        
        # Close remaining positions at window end
        final_time = data.index[-1]
        final_price = data['close'].iloc[-1]
        positions, final_trades = self.check_exits(positions, final_time, final_price, force_exit=True)
        window_trades.extend(final_trades)
        
        # Update capital from final trades
        for trade in final_trades:
            capital += trade.pnl
        
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
    
    def run_backtest(self, symbols: List[str] = None) -> Dict:
        """Run backtest for all specified symbols"""
        if symbols is None:
            symbols = ['ADAEUR', 'BTCEUR', 'ETHEUR', 'SOLEUR']
        
        all_results = {}
        
        for symbol in symbols:
            try:
                results = self.backtest_symbol(symbol)
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