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
from collections import deque


class FeatureCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.timestamps = deque(maxlen=max_size)

    def get_features(self, symbol, timestamp):
        return self.cache.get(f"{symbol}_{timestamp}")


class BitvavoDataCollector:
    """Collects real-time and historical data from Bitvavo API."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.bitvavo.com/v2"
        self.ws_url = "wss://ws.bitvavo.com/v2"
        
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
        
        # Data buffers for each symbol (store last 100 candles)
        self.data_buffers: Dict[str, deque] = {}
        self.last_update: Dict[str, datetime] = {}
        
        self.logger = logging.getLogger(__name__)

    async def validate_candle_data(self, candle_data: dict) -> bool:
        """Validate incoming candle data."""
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return all(field in candle_data for field in required_fields)
        
    async def initialize_buffers(self, symbols: List[str]):
        """Initialize data buffers with historical data."""
        for symbol in symbols:
            try:
                # Get last 100 15-minute candles
                historical_data = await self.get_historical_data(symbol, '15m', 100)
                if historical_data is not None:
                    self.data_buffers[symbol] = deque(maxlen=100)
                    for _, row in historical_data.iterrows():
                        self.data_buffers[symbol].append({
                            'timestamp': row['timestamp'],
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume']
                        })
                    self.last_update[symbol] = datetime.now()
                    self.logger.info(f"Initialized buffer for {symbol} with {len(self.data_buffers[symbol])} candles")
                else:
                    self.data_buffers[symbol] = deque(maxlen=100)
                    self.logger.warning(f"Failed to initialize buffer for {symbol}")
            except Exception as e:
                self.logger.error(f"Error initializing buffer for {symbol}: {e}")
                self.data_buffers[symbol] = deque(maxlen=100)
    
    async def get_historical_data(self, symbol: str, interval: str = "15m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch historical candle data with anti-Cloudflare measures and robust error handling."""
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
    
    async def get_latest_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get latest data from buffer or fetch from API."""
        try:
            # Check if we have data in buffer
            if symbol in self.data_buffers and len(self.data_buffers[symbol]) > 0:
                # Convert buffer to DataFrame
                data_list = list(self.data_buffers[symbol])
                df = pd.DataFrame(data_list)
                
                # Ensure we have enough data
                if len(df) >= min(limit, 50):  # At least 50 candles
                    return df.tail(limit)
            
            # Fallback to API call
            return await self.get_historical_data(symbol, '15m', limit)
            
        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            url = f"{self.base_url}/ticker/price"
            params = {'market': symbol}
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
            
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
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
                                "interval": "15m"
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
                        # Add to buffer
                        self.data_buffers[symbol].append(candle)
                        self.last_update[symbol] = datetime.now()
                        self.logger.debug(f"Updated {symbol} candle: {candle['close']}")
                    
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
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
                        latest_data = await self.get_historical_data(symbol, '15m', 1)
                        if latest_data is not None and len(latest_data) > 0:
                            latest_candle = latest_data.iloc[-1]
                            
                            candle = {
                                'timestamp': latest_candle['timestamp'],
                                'open': latest_candle['open'],
                                'high': latest_candle['high'],
                                'low': latest_candle['low'],
                                'close': latest_candle['close'],
                                'volume': latest_candle['volume']
                            }
                            
                            # Add to buffer if it's newer than the last candle
                            if (symbol not in self.data_buffers or
                                len(self.data_buffers[symbol]) == 0 or
                                candle['timestamp'] > self.data_buffers[symbol][-1]['timestamp']):

                                if symbol not in self.data_buffers:
                                    self.data_buffers[symbol] = deque(maxlen=100)

                                if await self.validate_candle_data(candle):
                                    self.data_buffers[symbol].append(candle)
                                    self.last_update[symbol] = datetime.now()
                                    self.logger.info(f"Updated {symbol} via API: {candle['close']}")
                
                # Wait for next update cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in periodic data update: {e}")
                await asyncio.sleep(60)
    
    def get_buffer_status(self) -> Dict[str, dict]:
        """Get status of all data buffers."""
        status = {}
        for symbol, buffer in self.data_buffers.items():
            last_update = self.last_update.get(symbol)
            status[symbol] = {
                'buffer_size': len(buffer),
                'last_update': last_update.isoformat() if last_update else None,
                'latest_price': buffer[-1]['close'] if buffer else None
            }
        return status