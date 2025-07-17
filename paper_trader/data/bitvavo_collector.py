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


class FeatureCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.timestamps = deque(maxlen=max_size)
        self.data_buffers = {}
        self.last_update = {}
        self.logger = logging.getLogger(__name__)

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

    def ensure_sufficient_data(self, symbol: str, min_length: int = 250) -> bool:
        """Ensure buffer has sufficient data for feature engineering."""
        try:
            if symbol not in self.data_buffers:
                self.logger.info(f"No buffer for {symbol}, initializing...")
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._initialize_single_buffer(symbol, min_length * 2))
                    return False
                else:
                    loop.run_until_complete(self._initialize_single_buffer(symbol, min_length * 2))
                    return len(self.data_buffers.get(symbol, pd.DataFrame())) >= min_length
            current_length = len(self.data_buffers[symbol])
            if current_length < min_length:
                self.logger.info(f"Fetching more data for {symbol}: current={current_length}, needed={min_length}")
                new_data = self._get_historical_data_sync(symbol, self.interval, max(500, min_length * 2))
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

    async def _initialize_single_buffer(self, symbol: str, limit: int = 300):
        """Helper method to initialize a single buffer."""
        try:
            historical_data = await self.get_historical_data(symbol, self.interval, limit)
            if historical_data is not None and len(historical_data) >= 100:
                self.data_buffers[symbol] = historical_data.copy()
                self.last_update[symbol] = datetime.now()
                self.logger.info(f"Initialized buffer for {symbol} with {len(historical_data)} candles")
            else:
                self.data_buffers[symbol] = pd.DataFrame()
                self.logger.warning(f"Failed to initialize buffer for {symbol}")
        except Exception as e:
            self.logger.error(f"Error initializing buffer for {symbol}: {e}")
            self.data_buffers[symbol] = pd.DataFrame()

    def get_buffer_data(self, symbol: str, min_length: int = 250) -> Optional[pd.DataFrame]:
        """Get buffer data for feature engineering with minimum length validation."""
        try:
            if symbol not in self.data_buffers:
                self.logger.warning(f"No buffer data for {symbol}")
                return None
            buffer_data = self.data_buffers[symbol].copy()
            if len(buffer_data) < min_length:
                self.logger.warning(f"Insufficient buffer data for {symbol}: {len(buffer_data)} < {min_length}")
                if self.ensure_sufficient_data(symbol, min_length):
                    buffer_data = self.data_buffers[symbol].copy()
                else:
                    return None
            return buffer_data
        except Exception as e:
            self.logger.error(f"Error getting buffer data for {symbol}: {e}")
            return None

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
                            # To:
                            if symbol in self.data_buffers and not self.data_buffers[symbol].empty:
                                    self.data_buffers[symbol] = pd.concat([self.data_buffers[symbol], new_row])
                            if len(self.data_buffers[symbol]) > 500:
                                    self.data_buffers[symbol] = self.data_buffers[symbol].tail(500)
                            else:
                                self.data_buffers[symbol] = new_row
                            self.last_update[symbol] = datetime.now()
                            self.logger.info(f"Updated {symbol} buffer via API: {latest_candle['close']:.2f}")
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error in periodic data update: {e}")
                await asyncio.sleep(60)

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
                self.last_update[symbol] = datetime.now()
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

    def __init__(self, api_key: str, api_secret: str, interval: str = "15m"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.bitvavo.com/v2"
        self.ws_url = "wss://ws.bitvavo.com/v2"

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
        endpoints_to_try = [
            f"{self.base_url}/candles",
            f"{self.base_url}/{symbol}/candles",
        ]
        
        alt_symbols = [
            symbol,
            symbol.replace('-', ''),
            symbol.lower(),
        ]

        for endpoint in endpoints_to_try:
            for s in alt_symbols:
                try:
                    await asyncio.sleep(random.uniform(0.2, 0.6))  # Random delay
                    params = {'market': s, 'interval': interval, 'limit': limit}
                    self.logger.debug(f"Trying endpoint: {endpoint} with symbol: {s}")
                    
                    response = await self.session.get(endpoint, params=params, timeout=20)
                    
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
                            self.logger.debug(f"Empty or invalid data from {endpoint} for {s}")
                    elif response.status_code == 404:
                        self.logger.debug(f"Endpoint {endpoint} not found for symbol {s}")
                    else:
                        self.logger.warning(f"HTTP {response.status_code} from {endpoint} for {s}: {response.text[:150]}")

                except httpx.RequestError as e:
                    self.logger.warning(f"Request failed for {endpoint} with symbol {s}: {e}")
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error parsing JSON for {s}: {e}")
                except Exception as e:
                    self.logger.error(f"An unexpected error occurred for {s}: {e}")

        self.logger.error(f"Failed to fetch historical data for {symbol} after all attempts.")
        return None

    # ... rest of the class remains unchanged



    
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
            # Prefer price from buffer if it was updated recently
            if (
                symbol in self.data_buffers
                and not self.data_buffers[symbol].empty
            ):
                latest_ts = self.data_buffers[symbol].index[-1]
                if datetime.now() - latest_ts < timedelta(minutes=1):
                    return float(self.data_buffers[symbol]['close'].iloc[-1])

            url = f"{self.base_url}/ticker/price"
            params = {'market': symbol}

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()
            return float(data['price'])

        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
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
                if price_diff < 0.05:  # avoid overwriting with stale data
                    self.data_buffers[symbol].iloc[-1, self.data_buffers[symbol].columns.get_loc('close')] = price
                    self.last_update[symbol] = datetime.now()
                    self.logger.debug(f"Refreshed {symbol} price to {price}")
                else:
                    self.logger.warning(
                        f"Skipped refreshing {symbol} price due to large difference: {last_close} -> {price}"
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
                        # Add to buffer using DataFrame concatenation
                        new_row = (
                            pd.DataFrame([candle])
                            .set_index("timestamp")
                        )
                        self.data_buffers[symbol] = pd.concat(
                            [self.data_buffers[symbol], new_row]
                        )

                        # Trim buffer to last 500 candles
                        if len(self.data_buffers[symbol]) > 500:
                            self.data_buffers[symbol] = self.data_buffers[symbol].tail(500)

                        self.last_update[symbol] = datetime.now()
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
                await self.initialize_buffers([symbol], min_length * 2)
                return len(self.data_buffers.get(symbol, pd.DataFrame())) >= min_length

            current_length = len(self.data_buffers[symbol])
            if current_length < min_length:
                self.logger.info(f"Fetching more data for {symbol}: current={current_length}, needed={min_length}")
                new_data = await self.get_historical_data(symbol, self.interval, max(500, min_length * 2))

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
                            if len(self.data_buffers[symbol]) > 500: 
                                self.data_buffers[symbol] = self.data_buffers[symbol].tail(500) 
                            
                            self.last_update[symbol] = datetime.now() 
                            self.logger.info(f"Updated {symbol} buffer: latest price {latest_data['close'].iloc[-1]:.2f}") 
                        else: 
                            self.logger.warning(f"Failed to fetch latest candle for {symbol}") 
                            
            except Exception as e: 
                self.logger.error(f"Error in periodic data update: {e}") 
                
            await asyncio.sleep(60)  # Wait for 60 seconds before next check