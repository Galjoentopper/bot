"""Bitvavo data collector for real-time market data."""

import asyncio
import json
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import httpx
import websockets
import requests
from collections import deque

# Import settings for configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import TradingSettings


class FeatureCache:
    def __init__(self, settings: TradingSettings = None):
        if settings is None:
            settings = TradingSettings()
        self.settings = settings
        self.cache = {}
        self.timestamps = deque(maxlen=settings.data_cache_max_size)
        self.data_buffers = {}
        self.last_update = {}
        self.logger = logging.getLogger(__name__)
        
        # Constants from settings
        self.MIN_DATA_LENGTH = settings.min_data_length

    def get_buffer_data(self, symbol: str, min_length: int = None) -> Optional[pd.DataFrame]:
        """
        Retrieves validated buffer data for a symbol, ensuring minimum length requirements.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-EUR')
            min_length: Minimum required data points (default: class MIN_DATA_LENGTH)
        
        Returns:
            Copy of validated DataFrame or None if requirements can't be met
        """
        try:
            min_length = min_length or self.MIN_DATA_LENGTH
            
            # Validate buffer existence and structure
            if not self._validate_buffer(symbol):
                self.logger.warning(f"Initializing new buffer for {symbol}")
                if not self.ensure_sufficient_data(symbol, min_length):
                    return None
            
            buffer = self.data_buffers[symbol]
            
            # Check data sufficiency
            if len(buffer) < min_length:
                self.logger.warning(f"Insufficient data for {symbol} ({len(buffer)}/{min_length})")
                if not self.ensure_sufficient_data(symbol, min_length):
                    return None
            
            # Return safe copy with latest validation
            return self._validate_buffer(symbol, update=True).copy()
            
        except Exception as e:
            self.logger.error(f"Buffer retrieval failed for {symbol}: {str(e)}", exc_info=True)
            return None

    def get_detailed_buffer_status(self) -> Dict[str, dict]:
        """Get detailed status of all data buffers for debugging."""
        status = {}
        for symbol, buffer in self.data_buffers.items():
            try:
                if isinstance(buffer, pd.DataFrame) and not buffer.empty:
                    last_update = self.last_update.get(symbol)
                    latest_timestamp = buffer.index[-1] if len(buffer) > 0 else None
                    oldest_timestamp = buffer.index[0] if len(buffer) > 0 else None
                    
                    status[symbol] = {
                        "buffer_size": len(buffer),
                        "last_update": last_update.isoformat() if last_update else None,
                        "last_update_ago_minutes": (
                            (datetime.now() - last_update).total_seconds() / 60
                        ) if last_update else None,
                        "latest_price": float(buffer['close'].iloc[-1]) if 'close' in buffer.columns else None,
                        "latest_timestamp": latest_timestamp.isoformat() if latest_timestamp else None,
                        "oldest_timestamp": oldest_timestamp.isoformat() if oldest_timestamp else None,
                        "data_span_hours": (
                            (latest_timestamp - oldest_timestamp).total_seconds() / 3600
                        ) if latest_timestamp and oldest_timestamp else None,
                        "columns": list(buffer.columns),
                        "null_counts": buffer.isnull().sum().to_dict() if len(buffer) > 0 else {},
                        "status": "healthy" if len(buffer) >= self.settings.healthy_buffer_threshold else "insufficient_data"
                    }
                else:
                    status[symbol] = {
                        "buffer_size": 0,
                        "status": "empty",
                        "last_update": None,
                        "error": "No data in buffer"
                    }
            except Exception as e:
                status[symbol] = {
                    "buffer_size": 0,
                    "status": "error",
                    "error": str(e)
                }
        return status

    def ensure_sufficient_data(self, symbol: str, min_length: int = None) -> bool:
        """Ensure buffer has sufficient data for feature engineering."""
        if min_length is None:
            min_length = self.settings.min_data_length
        try:
            if symbol not in self.data_buffers:
                self.logger.info(f"No buffer for {symbol}, initializing...")
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._initialize_single_buffer(symbol, min_length * self.settings.sufficient_data_multiplier))
                    return False
                else:
                    loop.run_until_complete(self._initialize_single_buffer(symbol, min_length * self.settings.sufficient_data_multiplier))
                    return len(self.data_buffers.get(symbol, pd.DataFrame())) >= min_length
            current_length = len(self.data_buffers[symbol])
            if current_length < min_length:
                self.logger.info(f"Fetching more data for {symbol}: current={current_length}, needed={min_length}")
                new_data = self._get_historical_data_sync(symbol, self.interval, max(self.settings.max_buffer_size, min_length * self.settings.sufficient_data_multiplier))
                if new_data is not None and len(new_data) >= min_length:
                    self.data_buffers[symbol] = new_data
                    self.logger.info(f"Updated buffer for {symbol} with {len(new_data)} candles")
                    return True
                else:
                    self.logger.warning(f"Could not fetch sufficient data for {symbol}")
                    return False
            return current_length >= min_length
        except Exception as e:
            self.logger.error(f"Error ensuring sufficient data for {symbol}: {e}")
            return False

    async def _initialize_single_buffer(self, symbol: str, limit: int = None):
        """Helper method to initialize a single buffer."""
        if limit is None:
            limit = self.settings.buffer_initialization_limit
        try:
            historical_data = await self.get_historical_data(symbol, self.interval, limit)
            if historical_data is not None and len(historical_data) >= self.settings.healthy_buffer_threshold:
                self.data_buffers[symbol] = historical_data.copy()
                # Track the timestamp of the latest candle rather than the
                # current time so that downstream logic only triggers when a
                # truly new candle arrives
                self.last_update[symbol] = historical_data.index[-1]
                self.logger.info(f"Initialized buffer for {symbol} with {len(historical_data)} candles")
            else:
                self.data_buffers[symbol] = pd.DataFrame()
                self.logger.warning(f"Failed to initialize buffer for {symbol}")
        except Exception as e:
            self.logger.error(f"Error initializing buffer for {symbol}: {e}")
            self.data_buffers[symbol] = pd.DataFrame()



    async def start_periodic_updates(self, symbols: List[str], interval_minutes: int = 15):
        """Start periodic data updates via REST API."""
        self.logger.info(f"Starting periodic updates for {len(symbols)} symbols every {interval_minutes} minutes")
        while True:
            try:
                for symbol in symbols:
                    last_update = self.last_update.get(symbol)
                    if (last_update is None or 
                        datetime.now() - last_update > timedelta(minutes=interval_minutes + 1)):
                        latest_data = await self.get_historical_data(symbol, self.interval, 1)
                        if latest_data is not None and len(latest_data) > 0:
                            latest_candle = latest_data.iloc[-1]
                            new_timestamp = latest_candle.name
                            if (symbol in self.data_buffers and 
                                not self.data_buffers[symbol].empty and
                                new_timestamp <= self.data_buffers[symbol].index[-1]):
                                continue
                            new_row = pd.DataFrame([latest_candle.values], 
                                                 columns=latest_candle.index, 
                                                 index=[new_timestamp])
                            # Update buffer with new data
                            if symbol in self.data_buffers and not self.data_buffers[symbol].empty:
                                self.data_buffers[symbol] = pd.concat([self.data_buffers[symbol], new_row])
                                if len(self.data_buffers[symbol]) > self.settings.max_buffer_size:
                                    self.data_buffers[symbol] = self.data_buffers[symbol].tail(self.settings.max_buffer_size)
                            else:
                                self.data_buffers[symbol] = new_row
                            # Use the candle timestamp as the last update marker
                            self.last_update[symbol] = new_timestamp
                            self.logger.info(f"Updated {symbol} buffer via API: {latest_candle['close']:.2f}")
                await asyncio.sleep(self.settings.websocket_sleep_seconds)
            except Exception as e:
                self.logger.error(f"Error in periodic data update: {e}")
                await asyncio.sleep(self.settings.websocket_sleep_seconds)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        try:
            if symbol in self.data_buffers and not self.data_buffers[symbol].empty:
                return float(self.data_buffers[symbol]['close'].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def get_buffer_status(self) -> Dict[str, dict]:
        """Get status of all data buffers."""
        status = {}
        for symbol, buffer in self.data_buffers.items():
            if not buffer.empty:
                last_update = self.last_update.get(symbol)
                status[symbol] = {
                    "buffer_size": len(buffer),
                    "last_update": last_update.isoformat() if last_update else None,
                    "latest_price": float(buffer['close'].iloc[-1]) if len(buffer) > 0 else None,
                    "latest_timestamp": buffer.index[-1].isoformat() if len(buffer) > 0 else None
                }
            else:
                status[symbol] = {
                    "buffer_size": 0,
                    "last_update": None,
                    "latest_price": None,
                    "latest_timestamp": None
                }
        return status



    async def _ensure_data_async(self, symbol: str, min_length: int):
        """Helper method to ensure data asynchronously."""
        try:
            historical_data = await self.get_historical_data(symbol, self.interval, min_length * 2)
            if historical_data is not None and len(historical_data) >= min_length:
                self.data_buffers[symbol] = historical_data.copy()
                # Use the candle timestamp of the fetched data as last_update
                self.last_update[symbol] = historical_data.index[-1]
                self.logger.info(f"Initialized buffer for {symbol} with {len(historical_data)} candles")
            else:
                self.data_buffers[symbol] = pd.DataFrame()
                self.logger.warning(f"Failed to initialize buffer for {symbol}")
        except Exception as e:
            self.logger.error(f"Error initializing buffer for {symbol}: {e}")
            self.data_buffers[symbol] = pd.DataFrame()

    def get_detailed_buffer_status(self) -> Dict[str, dict]:
        """Get detailed status of all data buffers for debugging."""
        status = {}
        for symbol, buffer in self.data_buffers.items():
            try:
                if not buffer.empty:
                    last_update = self.last_update.get(symbol)
                    latest_timestamp = buffer.index[-1] if len(buffer) > 0 else None
                    oldest_timestamp = buffer.index[0] if len(buffer) > 0 else None
                    
                    status[symbol] = {
                        "buffer_size": len(buffer),
                        "last_update": last_update.isoformat() if last_update else None,
                        "last_update_ago_minutes": ((datetime.now() - last_update).total_seconds() / 60) if last_update else None,
                        "latest_price": float(buffer['close'].iloc[-1]) if len(buffer) > 0 else None,
                        "latest_timestamp": latest_timestamp.isoformat() if latest_timestamp else None,
                        "oldest_timestamp": oldest_timestamp.isoformat() if oldest_timestamp else None,
                        "data_span_hours": ((latest_timestamp - oldest_timestamp).total_seconds() / 3600) if latest_timestamp and oldest_timestamp else None,
                        "columns": list(buffer.columns),
                        "null_counts": buffer.isnull().sum().to_dict() if len(buffer) > 0 else {},
                        "status": "healthy" if len(buffer) >= 100 else "insufficient_data"
                    }
                else:
                    status[symbol] = {
                        "buffer_size": 0,
                        "status": "empty",
                        "last_update": None,
                        "error": "No data in buffer"
                    }
            except Exception as e:
                status[symbol] = {
                    "buffer_size": 0,
                    "status": "error",
                    "error": str(e)
                }
        return status



    def get_features(self, symbol: str, timestamp: str) -> Optional[Dict]:
        return self.cache.get(f"{symbol}_{timestamp}")


class BitvavoDataCollector:
    """Collects real-time and historical data from Bitvavo API."""

    def __init__(self, api_key: str, api_secret: str, interval: str = "15m", settings: TradingSettings = None):
        if settings is None:
            settings = TradingSettings()
        self.settings = settings
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = settings.bitvavo_base_url
        self.ws_url = settings.bitvavo_ws_url

        # Candle interval used for both REST and WebSocket calls
        self.interval = interval
        
        # Add headers to avoid Cloudflare detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = httpx.AsyncClient(headers=headers, follow_redirects=True)
        
        # Data buffers for each symbol (store last 100 candles) as pandas DataFrames
        self.data_buffers: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        
        self.logger = logging.getLogger(__name__)

    async def initialize_buffers(self, symbols: List[str], limit: int = 100):
        """Initialize data buffers for multiple symbols asynchronously."""
        tasks = [self.get_historical_data(symbol, limit=limit) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        for symbol, data in zip(symbols, results):
            if data is not None and not data.empty:
                self.data_buffers[symbol] = data
                self.logger.info(f"Initialized buffer for {symbol} with {len(data)} candles")
            else:
                self.logger.warning(f"Failed to initialize buffer for {symbol}")

    def get_buffer_data(self, symbol: str, min_length: int = 250) -> pd.DataFrame:
        """Get buffer data for feature engineering, ensuring minimum length."""
        if symbol not in self.data_buffers or len(self.data_buffers[symbol]) < min_length:
            self.logger.warning(f"Insufficient data for {symbol}: have {len(self.data_buffers.get(symbol, []))}, need {min_length}")
            return pd.DataFrame()
        return self.data_buffers[symbol].tail(min_length).copy()

    async def get_historical_data(self, symbol: str, interval: str | None = None, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical candle data for a symbol."""
        interval = interval or self.interval
        endpoint = f"{self.base_url}/{symbol}/candles"
        
        try:
            await asyncio.sleep(random.uniform(self.settings.api_retry_delay_min, self.settings.api_retry_delay_max))  # Random delay
            params = {'interval': interval, 'limit': limit}
            self.logger.debug(f"Fetching data from {endpoint} for symbol: {symbol}")
            
            response = await self.session.get(endpoint, params=params, timeout=self.settings.api_timeout_seconds)
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    if not df.empty:
                        self.logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
                        return df
                    else:
                        self.logger.warning(f"No valid data after processing for {symbol}")
                else:
                    self.logger.warning(f"Empty or invalid data from API for {symbol}")
            elif response.status_code == 404:
                self.logger.error(f"Market {symbol} not found on Bitvavo API")
            else:
                self.logger.error(f"HTTP {response.status_code} from Bitvavo API for {symbol}: {response.text[:150]}")

        except httpx.RequestError as e:
            self.logger.error(f"Request failed for {symbol}: {e}")
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error parsing JSON for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred for {symbol}: {e}")

        self.logger.error(f"Failed to fetch historical data for {symbol}")
        return None


    async def get_latest_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get latest data from buffer or fetch from API."""
        try:
            # Check if we have data in buffer
            if symbol in self.data_buffers and not self.data_buffers[symbol].empty:
                # Convert buffer to DataFrame
                data_list = list(self.data_buffers[symbol])
                df = pd.DataFrame(data_list)
                
                # Ensure we have enough data
                if len(df) >= min(limit, 50):  # At least 50 candles
                    return df.tail(limit)
            
            # Fallback to API call
            return await self.get_historical_data(symbol, self.interval, limit)
            
        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Prefer price from buffer if it was updated recently (within 15 minutes for 15-min candles)
            if (
                symbol in self.data_buffers
                and not self.data_buffers[symbol].empty
            ):
                latest_ts = self.data_buffers[symbol].index[-1]
                if datetime.now() - latest_ts < timedelta(minutes=15):
                    return float(self.data_buffers[symbol]['close'].iloc[-1])

            url = f"{self.base_url}/ticker/price"
            params = {'market': symbol}

            response = requests.get(url, params=params, timeout=self.settings.price_api_timeout_seconds)
            response.raise_for_status()

            data = response.json()
            return float(data['price'])

        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    async def get_current_price_for_trading(self, symbol: str) -> Optional[float]:
        """Get the most current price for trading decisions, always using live API."""
        try:
            url = f"{self.base_url}/ticker/price"
            params = {'market': symbol}

            response = requests.get(url, params=params, timeout=self.settings.price_api_timeout_seconds)
            response.raise_for_status()

            data = response.json()
            current_price = float(data['price'])
            
            # Log price comparison for monitoring
            if (symbol in self.data_buffers and not self.data_buffers[symbol].empty):
                buffer_price = float(self.data_buffers[symbol]['close'].iloc[-1])
                price_diff = abs(current_price - buffer_price) / buffer_price if buffer_price else 0
                latest_ts = self.data_buffers[symbol].index[-1]
                age_minutes = (datetime.now() - latest_ts).total_seconds() / 60
                
                self.logger.debug(
                    f"Trading price for {symbol}: live={current_price:.4f}, buffer={buffer_price:.4f}, "
                    f"diff={price_diff*100:.2f}%, buffer_age={age_minutes:.1f}min"
                )
                
                # Warn if there's a significant difference
                if price_diff > 0.02:  # 2% difference
                    self.logger.warning(
                        f"⚠️ Price discrepancy for {symbol}: live API={current_price:.4f}, "
                        f"buffer={buffer_price:.4f} (diff: {price_diff*100:.2f}%, age: {age_minutes:.1f}min)"
                    )
            
            return current_price

        except Exception as e:
            self.logger.error(f"Error fetching current trading price for {symbol}: {e}")
            # Fallback to buffer price if API fails
            if (symbol in self.data_buffers and not self.data_buffers[symbol].empty):
                self.logger.warning(f"Falling back to buffer price for {symbol}")
                return float(self.data_buffers[symbol]['close'].iloc[-1])
            return None

    async def refresh_latest_price(self, symbol: str) -> None:
        """Update the latest candle close with the most recent price."""
        try:
            price = await self.get_current_price(symbol)
            if (
                price is not None
                and symbol in self.data_buffers
                and not self.data_buffers[symbol].empty
            ):
                last_close = float(self.data_buffers[symbol]['close'].iloc[-1])
                price_diff = abs(price - last_close) / last_close if last_close else 0
                
                # More sophisticated price validation logic
                # Check how old the last candle is to determine appropriate threshold
                latest_candle_time = self.data_buffers[symbol].index[-1]
                time_since_candle = (datetime.now() - latest_candle_time).total_seconds() / 60  # minutes
                
                # Dynamic threshold based on time since last candle update
                # Allow larger differences for older candles (up to 15% for very old data)
                max_threshold = min(0.15, 0.02 + (time_since_candle / 60) * 0.10)  # 2% base + up to 10% for age
                
                self.logger.debug(
                    f"Price validation for {symbol}: current={price:.4f}, buffer={last_close:.4f}, "
                    f"diff={price_diff:.4f} ({price_diff*100:.2f}%), threshold={max_threshold:.4f} ({max_threshold*100:.2f}%), "
                    f"age={time_since_candle:.1f}min"
                )
                
                if price_diff <= max_threshold:
                    self.data_buffers[symbol].iloc[-1, self.data_buffers[symbol].columns.get_loc('close')] = price
                    # Keep last_update in sync with the candle timestamp to
                    # avoid triggering multiple predictions for interim updates
                    self.last_update[symbol] = self.data_buffers[symbol].index[-1]
                    self.logger.debug(f"✅ Refreshed {symbol} price: {last_close:.4f} -> {price:.4f} (diff: {price_diff*100:.2f}%)")
                else:
                    # Log detailed warning but still update if the data is very old (>30 min)
                    if time_since_candle > 30:
                        self.logger.warning(
                            f"⚠️ Large price difference for {symbol} but updating due to stale data (>{time_since_candle:.1f}min old): "
                            f"{last_close:.4f} -> {price:.4f} (diff: {price_diff*100:.2f}%)"
                        )
                        self.data_buffers[symbol].iloc[-1, self.data_buffers[symbol].columns.get_loc('close')] = price
                        self.last_update[symbol] = self.data_buffers[symbol].index[-1]
                    else:
                        self.logger.warning(
                            f"❌ Rejected price update for {symbol} due to large difference: "
                            f"{last_close:.4f} -> {price:.4f} (diff: {price_diff*100:.2f}% > {max_threshold*100:.2f}%)"
                        )
        except Exception as e:
            self.logger.warning(f"Failed to refresh price for {symbol}: {e}")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return the most recent close price stored for a symbol."""
        try:
            if symbol in self.data_buffers and not self.data_buffers[symbol].empty:
                return float(self.data_buffers[symbol]["close"].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    async def start_websocket_feed(self, symbols: List[str]):
        """Start WebSocket feed for real-time data updates."""
        while True:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    # Subscribe to candle updates for all symbols
                    for symbol in symbols:
                        subscribe_msg = {
                            "action": "subscribe",
                            "channels": [{
                                "name": "candles",
                                "markets": [symbol],
                                "interval": self.interval
                            }]
                        }
                        await websocket.send(json.dumps(subscribe_msg))
                    
                    self.logger.info(f"Subscribed to WebSocket feed for {len(symbols)} symbols")
                    
                    # Listen for updates
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._process_websocket_message(data)
                        except Exception as e:
                            self.logger.error(f"Error processing WebSocket message: {e}")
                            
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting

    async def validate_candle_data(self, candle: dict) -> bool:
        """Validate incoming candle data from the WebSocket."""
        try:
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(k in candle for k in required):
                return False

            numeric_fields = ["open", "high", "low", "close", "volume"]
            for field in numeric_fields:
                if candle.get(field) is None:
                    return False

            return True
        except Exception as e:
            self.logger.warning(f"Invalid candle data: {e}")
            return False
    
    async def _process_websocket_message(self, data: dict):
        """Process incoming WebSocket messages."""
        try:
            if data.get('event') == 'candle':
                symbol = data.get('market')
                candle_data = data.get('candle')
                
                if symbol and candle_data and symbol in self.data_buffers:
                    # Convert candle data
                    candle = {
                        'timestamp': pd.to_datetime(candle_data[0], unit='ms'),
                        'open': float(candle_data[1]),
                        'high': float(candle_data[2]),
                        'low': float(candle_data[3]),
                        'close': float(candle_data[4]),
                        'volume': float(candle_data[5])
                    }
                    if await self.validate_candle_data(candle):
                        # Create dataframe row
                        new_row = pd.DataFrame([candle]).set_index("timestamp")

                        # Replace existing candle if we already have this timestamp
                        if (
                            symbol in self.data_buffers
                            and not self.data_buffers[symbol].empty
                            and new_row.index[0] in self.data_buffers[symbol].index
                        ):
                            self.data_buffers[symbol].loc[new_row.index[0]] = new_row.iloc[0]
                        else:
                            self.data_buffers[symbol] = pd.concat(
                                [self.data_buffers[symbol], new_row]
                            )
                            if len(self.data_buffers[symbol]) > self.settings.max_buffer_size:
                                self.data_buffers[symbol] = self.data_buffers[symbol].tail(self.settings.max_buffer_size)

                        # Update last_update using candle timestamp so repeated
                        # updates for the same candle don't trigger extra predictions
                        self.last_update[symbol] = new_row.index[0]
                        self.logger.debug(
                            f"Updated {symbol} candle: {candle['close']}"
                        )
                    
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    async def ensure_sufficient_data(self, symbol: str, min_length: int = 250) -> bool:
        """Ensure buffer has sufficient data for feature engineering."""
        try:
            if symbol not in self.data_buffers:
                self.logger.info(f"No buffer for {symbol}, initializing...")
                await self.initialize_buffers([symbol], min_length * self.settings.sufficient_data_multiplier)
                return len(self.data_buffers.get(symbol, pd.DataFrame())) >= min_length

            current_length = len(self.data_buffers[symbol])
            if current_length < min_length:
                self.logger.info(f"Fetching more data for {symbol}: current={current_length}, needed={min_length}")
                new_data = await self.get_historical_data(symbol, self.interval, max(self.settings.max_buffer_size, min_length * self.settings.sufficient_data_multiplier))

                if new_data is not None and len(new_data) >= min_length:
                    self.data_buffers[symbol] = new_data
                    self.logger.info(f"Updated buffer for {symbol} with {len(new_data)} candles")
                    return True
                else:
                    self.logger.warning(f"Could not fetch sufficient data for {symbol}")
                    return False

            return current_length >= min_length

        except Exception as e:
            self.logger.error(f"Error ensuring sufficient data for {symbol}: {e}")
            return False

    async def update_data_periodically(self, symbols: List[str], interval_minutes: int = 15):
        """Periodically update data buffers via API calls.""" 
        while True: 
            try: 
                for symbol in symbols: 
                    # Check if we need to update 
                    last_update = self.last_update.get(symbol) 
                    if (last_update is None or 
                        datetime.now() - last_update > timedelta(minutes=interval_minutes + 1)): 
                        
                        # Fetch latest candle 
                        latest_data = await self.get_historical_data(symbol, self.interval, 1)
                        if latest_data is not None and len(latest_data) > 0: 
                            latest_timestamp = latest_data.index[-1] 
                            
                            # Check if this is a new candle 
                            if symbol in self.data_buffers and not self.data_buffers[symbol].empty: 
                                current_latest = self.data_buffers[symbol].index[-1] 
                                if latest_timestamp <= current_latest: 
                                    self.logger.debug(f"No new candle for {symbol}, skipping update") 
                                    continue 
                            
                            # Append new data to buffer 
                            if symbol in self.data_buffers and not self.data_buffers[symbol].empty: 
                                # Concatenate the new data 
                                self.data_buffers[symbol] = pd.concat([ 
                                    self.data_buffers[symbol], 
                                    latest_data 
                                ]) 
                                # Remove duplicates and sort by index 
                                self.data_buffers[symbol] = self.data_buffers[symbol][ 
                                    ~self.data_buffers[symbol].index.duplicated(keep='last') 
                                ].sort_index() 
                            else: 
                                # Initialize buffer with the latest data 
                                self.data_buffers[symbol] = latest_data 
                            
                            # Trim buffer to max size 
                            if len(self.data_buffers[symbol]) > self.settings.max_buffer_size: 
                                self.data_buffers[symbol] = self.data_buffers[symbol].tail(self.settings.max_buffer_size) 
                            
                            # Record the timestamp of the candle rather than
                            # the current time so main loop only triggers on
                            # new candles
                            self.last_update[symbol] = latest_timestamp
                            self.logger.info(f"Updated {symbol} buffer: latest price {latest_data['close'].iloc[-1]:.2f}") 
                        else: 
                            self.logger.warning(f"Failed to fetch latest candle for {symbol}") 
                            
            except Exception as e: 
                self.logger.error(f"Error in periodic data update: {e}") 
                
            await asyncio.sleep(self.settings.websocket_sleep_seconds)  # Wait before next check