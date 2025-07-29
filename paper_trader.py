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

# Configure logging
logger = logging.getLogger(__name__)

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
            
        # Load models and feature columns
        self.load_models()
        
        # Setup API fallback monitor
        self.api_monitor_thread = None
        
    def load_models(self):
        """Load trained models from the models directory"""
        logger.info("Loading models...")
        models_dir = "models"
        
        for symbol in self.symbols:
            # Try to load XGBoost model
            model_path = os.path.join(models_dir, "xgboost", f"{symbol.lower()}_xgboost.pkl")
            feature_path = os.path.join(models_dir, "feature_columns", f"{symbol.lower()}_window_15_selected.pkl")
            
            if os.path.exists(model_path):
                try:
                    self.models[symbol] = joblib.load(model_path)
                    logger.info(f"Loaded XGBoost model for {symbol}")
                    
                    # Load feature columns if available
                    if os.path.exists(feature_path):
                        self.feature_columns[symbol] = joblib.load(feature_path)
                        logger.info(f"Loaded feature columns for {symbol} ({len(self.feature_columns[symbol])} features)")
                    else:
                        logger.warning(f"No feature columns found for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error loading model for {symbol}: {e}")
            else:
                logger.warning(f"No XGBoost model found for {symbol} at {model_path}")
    
    def fetch_historical_data(self):
        """Fetch initial historical data for all symbols"""
        logger.info("Fetching initial historical data...")
        
        for symbol in self.symbols:
            try:
                # Get 15-minute candles for the past 500 periods (enough for feature creation)
                url = f"https://api.bitvavo.com/v2/candles?market={symbol}&interval=15m&limit=500"
                response = requests.get(url)
                data = response.json()
                
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
            
            # Drop NaN values
            df = df.dropna()
            
            if df.empty:
                logger.warning(f"Empty dataframe after feature creation for {symbol}")
                return None
                
            # Return the latest row as a dictionary
            latest_features = df.iloc[-1].to_dict()
            
            # If we have specific feature columns for this symbol, filter to those
            if symbol in self.feature_columns:
                filtered_features = {}
                for feature in self.feature_columns[symbol]:
                    if feature in latest_features:
                        filtered_features[feature] = latest_features[feature]
                    else:
                        logger.warning(f"Feature {feature} not found in created features for {symbol}")
                        filtered_features[feature] = 0.0  # Default value
                return filtered_features
            else:
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
            
            # Subscribe to ticker data for all symbols
            subscribe_message = {
                "action": "subscribe",
                "channels": ["ticker"]
            }
            
            markets = []
            for symbol in self.symbols:
                markets.append(symbol)
                
            subscribe_message["markets"] = markets
            ws.send(json.dumps(subscribe_message))
            
            # Also subscribe to candle data
            for symbol in self.symbols:
                candle_message = {
                    "action": "subscribe",
                    "channels": ["candles"],
                    "markets": [symbol],
                    "interval": ["15m"]
                }
                ws.send(json.dumps(candle_message))
            
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
                    # Get current ticker
                    url = f"https://api.bitvavo.com/v2/ticker/price?market={symbol}"
                    response = requests.get(url)
                    data = response.json()
                    
                    if 'price' in data:
                        self.current_prices[symbol] = float(data['price'])
                        logger.debug(f"API fallback: Updated price for {symbol}: {self.current_prices[symbol]}")
                        
                        # Check if it's time to make a prediction
                        now = datetime.now()
                        if (now - self.last_prediction_time[symbol]).total_seconds() >= 60:
                            self.make_prediction(symbol)
                            self.last_prediction_time[symbol] = now
                    
                    # Get latest candle
                    url = f"https://api.bitvavo.com/v2/candles?market={symbol}&interval=15m&limit=1"
                    response = requests.get(url)
                    candles = response.json()
                    
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
            # Handle ticker events
            if event.get('event') == 'ticker24h':
                symbol = event.get('market')
                if symbol in self.symbols:
                    self.current_prices[symbol] = float(event.get('last'))
                    logger.debug(f"Updated price for {symbol}: {self.current_prices[symbol]}")
                    
                    # Check if it's time to make a prediction
                    now = datetime.now()
                    if (now - self.last_prediction_time[symbol]).total_seconds() >= 60:  # Predict every minute
                        self.make_prediction(symbol)
                        self.last_prediction_time[symbol] = now
                        
            # Handle candle events
            elif event.get('event') == 'candle':
                symbol = event.get('market')
                if symbol in self.symbols:
                    candle_data = {
                        'timestamp': pd.to_datetime(event.get('timestamp'), unit='ms'),
                        'open': float(event.get('open')),
                        'high': float(event.get('high')),
                        'low': float(event.get('low')),
                        'close': float(event.get('close')),
                        'volume': float(event.get('volume'))
                    }
                    
                    # Update historical data
                    self.historical_data[symbol].append(candle_data)
                    logger.debug(f"Updated candle data for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error processing event: {e}")
    
    def make_prediction(self, symbol):
        """Make prediction for a symbol using the loaded model"""
        try:
            if symbol not in self.models:
                logger.warning(f"No model available for {symbol}")
                return
                
            # Create features for prediction
            features = self.create_features(symbol)
            if features is None:
                logger.warning(f"Could not create features for {symbol}")
                return
                
            # Convert to DataFrame for prediction
            X = pd.DataFrame([features])
            
            # Make prediction
            prediction = self.models[symbol].predict(X)[0]
            prediction_proba = None
            
            # If the model has predict_proba method (for classification models)
            if hasattr(self.models[symbol], 'predict_proba'):
                prediction_proba = self.models[symbol].predict_proba(X)[0]
                
            logger.info(f"Prediction for {symbol}: {prediction}")
            if prediction_proba is not None:
                logger.info(f"Prediction probabilities: {prediction_proba}")
                
            # Execute trade based on prediction
            self.execute_trade(symbol, prediction, prediction_proba)
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
    
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