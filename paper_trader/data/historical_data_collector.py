"""Historical data collector for feature engineering, separate from real-time pricing."""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import httpx

from ..config.settings import TradingSettings


class HistoricalDataCollector:
    """Collects and manages historical data for feature engineering."""
    
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Historical data buffers for feature engineering
        self.historical_buffers: Dict[str, pd.DataFrame] = {}
        self.last_historical_update: Dict[str, datetime] = {}
        
        # HTTP client for API calls
        self.http_client = None
        
        # Control flags
        self.is_running = False
        self.update_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the historical data collector."""
        self.logger.info("Initializing historical data collector...")
        
        # Initialize HTTP client
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        self.http_client = httpx.AsyncClient(headers=headers, follow_redirects=True)
        
        # Initialize historical buffers for all symbols
        await self._initialize_historical_buffers()
        
        self.logger.info(f"Historical data collector initialized for {len(self.settings.symbols)} symbols")
        
    async def start_periodic_updates(self):
        """Start periodic updates of historical data."""
        if self.is_running:
            self.logger.warning("Historical data updates are already running")
            return
            
        self.is_running = True
        self.update_task = asyncio.create_task(self._periodic_update_loop())
        self.logger.info("Started periodic historical data updates")
        
    async def stop(self):
        """Stop the historical data collector."""
        self.logger.info("Stopping historical data collector...")
        self.is_running = False
        
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
                
        if self.http_client:
            await self.http_client.aclose()
            
        self.logger.info("Historical data collector stopped")
        
    async def get_historical_data_for_features(self, symbol: str, min_length: int = 500) -> Optional[pd.DataFrame]:
        """Get historical data for feature engineering."""
        try:
            # Ensure we have sufficient data
            if not await self.ensure_sufficient_historical_data(symbol, min_length):
                self.logger.warning(f"Could not ensure sufficient historical data for {symbol}")
                return None
                
            # Return copy of historical buffer
            buffer = self.historical_buffers.get(symbol)
            if buffer is None or len(buffer) < min_length:
                self.logger.warning(
                    f"Insufficient historical data for {symbol}: "
                    f"{len(buffer) if buffer is not None else 0}/{min_length}"
                )
                return None
                
            return buffer.tail(min_length).copy()
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
            
    async def ensure_sufficient_historical_data(self, symbol: str, min_length: int) -> bool:
        """Ensure we have sufficient historical data for a symbol."""
        try:
            current_length = len(self.historical_buffers.get(symbol, pd.DataFrame()))
            
            if current_length >= min_length:
                return True
                
            self.logger.info(
                f"Fetching more historical data for {symbol}: "
                f"current={current_length}, needed={min_length}"
            )
            
            # Fetch more data
            limit = max(self.settings.max_buffer_size, min_length * 2)
            new_data = await self._fetch_historical_candles(symbol, limit)
            
            if new_data is not None and len(new_data) >= min_length:
                self.historical_buffers[symbol] = new_data
                self.last_historical_update[symbol] = datetime.now()
                self.logger.info(f"Updated historical buffer for {symbol} with {len(new_data)} candles")
                return True
            else:
                self.logger.warning(f"Could not fetch sufficient historical data for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error ensuring sufficient historical data for {symbol}: {e}")
            return False
            
    async def refresh_latest_historical_data(self, symbol: str):
        """Refresh the latest historical data for a symbol."""
        try:
            # Fetch latest candle
            latest_data = await self._fetch_historical_candles(symbol, 1)
            if latest_data is None or len(latest_data) == 0:
                return
                
            latest_candle = latest_data.iloc[-1]
            latest_timestamp = latest_candle.name
            
            # Check if this is new data
            if symbol in self.historical_buffers and not self.historical_buffers[symbol].empty:
                current_latest = self.historical_buffers[symbol].index[-1]
                if latest_timestamp <= current_latest:
                    return  # No new data
                    
                # Append new data
                new_row = pd.DataFrame([latest_candle.values], 
                                     columns=latest_candle.index, 
                                     index=[latest_timestamp])
                
                self.historical_buffers[symbol] = pd.concat([
                    self.historical_buffers[symbol], 
                    new_row
                ])
                
                # Trim buffer to max size
                if len(self.historical_buffers[symbol]) > self.settings.max_buffer_size:
                    self.historical_buffers[symbol] = self.historical_buffers[symbol].tail(
                        self.settings.max_buffer_size
                    )
            else:
                # Initialize buffer with latest data
                self.historical_buffers[symbol] = latest_data
                
            self.last_historical_update[symbol] = datetime.now()
            self.logger.debug(f"Refreshed historical data for {symbol}: {latest_candle['close']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error refreshing historical data for {symbol}: {e}")
            
    async def _initialize_historical_buffers(self):
        """Initialize historical data buffers for all symbols."""
        self.logger.info("Initializing historical data buffers...")
        
        tasks = []
        for symbol in self.settings.symbols:
            task = asyncio.create_task(self._initialize_symbol_buffer(symbol))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = 0
        for symbol, result in zip(self.settings.symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to initialize buffer for {symbol}: {result}")
            else:
                successful += 1
                
        self.logger.info(f"Initialized historical buffers for {successful}/{len(self.settings.symbols)} symbols")
        
    async def _initialize_symbol_buffer(self, symbol: str):
        """Initialize historical buffer for a single symbol."""
        try:
            # Fetch initial historical data
            limit = self.settings.buffer_initialization_limit
            data = await self._fetch_historical_candles(symbol, limit)
            
            if data is not None and len(data) >= self.settings.healthy_buffer_threshold:
                self.historical_buffers[symbol] = data
                self.last_historical_update[symbol] = datetime.now()
                self.logger.info(f"Initialized historical buffer for {symbol} with {len(data)} candles")
            else:
                self.historical_buffers[symbol] = pd.DataFrame()
                self.logger.warning(f"Failed to initialize historical buffer for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error initializing historical buffer for {symbol}: {e}")
            self.historical_buffers[symbol] = pd.DataFrame()
            
    async def _fetch_historical_candles(self, symbol: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch historical candle data from API."""
        try:
            # Add random delay to avoid rate limiting
            await asyncio.sleep(random.uniform(
                self.settings.api_retry_delay_min, 
                self.settings.api_retry_delay_max
            ))
            
            endpoint = f"{self.settings.bitvavo_base_url}/{symbol}/candles"
            params = {
                'interval': self.settings.candle_interval,
                'limit': limit
            }
            
            self.logger.debug(f"Fetching historical data for {symbol} (limit={limit})")
            
            response = await self.http_client.get(
                endpoint, 
                params=params, 
                timeout=self.settings.api_timeout_seconds
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        data, 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    
                    # Convert to numeric
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove any invalid data
                    df = df.dropna()
                    
                    if not df.empty:
                        self.logger.debug(f"Successfully fetched {len(df)} historical candles for {symbol}")
                        return df
                    else:
                        self.logger.warning(f"No valid historical data after processing for {symbol}")
                else:
                    self.logger.warning(f"Empty historical data response for {symbol}")
            elif response.status_code == 404:
                self.logger.error(f"Market {symbol} not found on Bitvavo API")
            else:
                self.logger.error(
                    f"HTTP {response.status_code} from Bitvavo API for {symbol}: "
                    f"{response.text[:150]}"
                )
                
        except httpx.RequestError as e:
            self.logger.error(f"Request failed for {symbol}: {e}")
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error parsing JSON for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching historical data for {symbol}: {e}")
            
        return None
        
    async def _periodic_update_loop(self):
        """Periodic update loop for historical data."""
        self.logger.info("Starting periodic historical data update loop")
        
        while self.is_running:
            try:
                # Update historical data for all symbols
                for symbol in self.settings.symbols:
                    if not self.is_running:
                        break
                        
                    # Check if update is needed
                    last_update = self.last_historical_update.get(symbol)
                    if (last_update is None or 
                        datetime.now() - last_update > timedelta(
                            minutes=self.settings.api_update_interval_minutes
                        )):
                        
                        await self.refresh_latest_historical_data(symbol)
                        
                        # Small delay between symbols
                        await asyncio.sleep(1)
                        
                # Wait before next update cycle
                if self.is_running:
                    await asyncio.sleep(self.settings.websocket_sleep_seconds)
                    
            except Exception as e:
                self.logger.error(f"Error in periodic historical data update: {e}")
                await asyncio.sleep(30)  # Wait before retrying
                
    def get_buffer_status(self) -> Dict[str, dict]:
        """Get status of all historical data buffers."""
        status = {}
        current_time = datetime.now()
        
        for symbol in self.settings.symbols:
            buffer = self.historical_buffers.get(symbol, pd.DataFrame())
            last_update = self.last_historical_update.get(symbol)
            
            if not buffer.empty:
                latest_timestamp = buffer.index[-1]
                oldest_timestamp = buffer.index[0]
                
                status[symbol] = {
                    'buffer_size': len(buffer),
                    'last_update': last_update.isoformat() if last_update else None,
                    'minutes_since_update': (
                        (current_time - last_update).total_seconds() / 60
                        if last_update else None
                    ),
                    'latest_candle_price': float(buffer['close'].iloc[-1]),
                    'latest_candle_timestamp': latest_timestamp.isoformat(),
                    'oldest_candle_timestamp': oldest_timestamp.isoformat(),
                    'data_span_hours': (
                        (latest_timestamp - oldest_timestamp).total_seconds() / 3600
                    ),
                    'null_counts': buffer.isnull().sum().to_dict(),
                    'status': 'healthy' if len(buffer) >= self.settings.healthy_buffer_threshold else 'insufficient'
                }
            else:
                status[symbol] = {
                    'buffer_size': 0,
                    'status': 'empty',
                    'last_update': last_update.isoformat() if last_update else None,
                    'error': 'No historical data in buffer'
                }
                
        return status
        
    def get_latest_candle_price(self, symbol: str) -> Optional[float]:
        """Get the latest candle close price for a symbol."""
        try:
            buffer = self.historical_buffers.get(symbol)
            if buffer is not None and not buffer.empty:
                return float(buffer['close'].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest candle price for {symbol}: {e}")
            return None