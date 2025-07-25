"""Real-time price collector using dedicated WebSocket connections for each symbol."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque

import websockets
import httpx

from ..config.settings import TradingSettings


@dataclass
class PriceUpdate:
    """Represents a real-time price update."""
    symbol: str
    price: float
    timestamp: datetime
    volume: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None


class RealtimePriceCollector:
    """Collects real-time prices using dedicated WebSocket connections for each symbol."""
    
    def __init__(self, settings: TradingSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Store current prices for each symbol
        self.current_prices: Dict[str, PriceUpdate] = {}
        
        # Price history for volatility calculations (last 60 updates per symbol)
        self.price_history: Dict[str, deque] = {}
        
        # WebSocket connections per symbol
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Callback functions for price updates
        self.price_update_callbacks: List[Callable[[PriceUpdate], None]] = []
        
        # Track connection health
        self.connection_health: Dict[str, datetime] = {}
        self.reconnect_delays: Dict[str, float] = {}
        
        # HTTP client for fallback API calls
        self.http_client = None
        
        # Control flags
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the real-time price collector."""
        self.logger.info("Initializing real-time price collector...")
        
        # Initialize HTTP client for fallback
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        self.http_client = httpx.AsyncClient(headers=headers, follow_redirects=True)
        
        # Initialize price history for each symbol
        for symbol in self.settings.symbols:
            self.price_history[symbol] = deque(maxlen=60)  # Keep last 60 price updates
            self.reconnect_delays[symbol] = 1.0  # Start with 1 second delay
            
        self.logger.info(f"Initialized for {len(self.settings.symbols)} symbols")
        
    async def start(self):
        """Start real-time price collection for all symbols."""
        if self.is_running:
            self.logger.warning("Real-time price collector is already running")
            return
            
        self.is_running = True
        self.logger.info("Starting real-time price collection...")
        
        # Start individual WebSocket connections for each symbol
        for symbol in self.settings.symbols:
            task = asyncio.create_task(self._start_symbol_websocket(symbol))
            self.tasks.append(task)
            
        # Start health monitoring task
        health_task = asyncio.create_task(self._monitor_connection_health())
        self.tasks.append(health_task)
        
        self.logger.info(f"Started WebSocket connections for {len(self.settings.symbols)} symbols")
        
    async def stop(self):
        """Stop all WebSocket connections and cleanup."""
        self.logger.info("Stopping real-time price collector...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        # Close WebSocket connections
        for symbol, ws in list(self.websocket_connections.items()):
            if ws and not ws.closed:
                await ws.close()
                
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
            
        self.tasks.clear()
        self.websocket_connections.clear()
        self.logger.info("Real-time price collector stopped")
        
    def add_price_update_callback(self, callback: Callable[[PriceUpdate], None]):
        """Add a callback function to be called on each price update."""
        self.price_update_callbacks.append(callback)
        
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current price for a symbol."""
        price_update = self.current_prices.get(symbol)
        return price_update.price if price_update else None
        
    def get_current_price_update(self, symbol: str) -> Optional[PriceUpdate]:
        """Get the full price update object for a symbol."""
        return self.current_prices.get(symbol)
        
    def get_price_age_seconds(self, symbol: str) -> Optional[float]:
        """Get the age of the current price in seconds."""
        price_update = self.current_prices.get(symbol)
        if not price_update:
            return None
        return (datetime.now() - price_update.timestamp).total_seconds()
        
    def calculate_volatility(self, symbol: str, periods: int = 20) -> float:
        """Calculate recent price volatility for a symbol."""
        history = self.price_history.get(symbol, deque())
        if len(history) < 2:
            return 0.5  # Default volatility
            
        # Use last N periods or all available if less
        recent_prices = list(history)[-min(periods, len(history)):]
        prices = [update.price for update in recent_prices]
        
        if len(prices) < 2:
            return 0.5
            
        # Calculate standard deviation relative to mean
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        
        return std_dev / mean_price if mean_price > 0 else 0.5
        
    async def _start_symbol_websocket(self, symbol: str):
        """Start WebSocket connection for a specific symbol."""
        while self.is_running:
            try:
                self.logger.info(f"Connecting to WebSocket for {symbol}...")
                
                # Connect to Bitvavo WebSocket
                async with websockets.connect(
                    self.settings.bitvavo_ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    self.websocket_connections[symbol] = websocket
                    
                    # Subscribe to ticker updates for this symbol
                    subscribe_msg = {
                        "action": "subscribe",
                        "channels": [{
                            "name": "ticker",
                            "markets": [symbol]
                        }]
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    self.logger.info(f"Subscribed to ticker updates for {symbol}")
                    self.connection_health[symbol] = datetime.now()
                    self.reconnect_delays[symbol] = 1.0  # Reset delay on successful connection
                    
                    # Listen for updates
                    async for message in websocket:
                        if not self.is_running:
                            break
                            
                        try:
                            await self._process_ticker_message(symbol, message)
                            self.connection_health[symbol] = datetime.now()
                        except Exception as e:
                            self.logger.error(f"Error processing message for {symbol}: {e}")
                            
            except Exception as e:
                self.logger.error(f"WebSocket error for {symbol}: {e}")
                
                # Remove from active connections
                if symbol in self.websocket_connections:
                    del self.websocket_connections[symbol]
                    
                if self.is_running:
                    # Exponential backoff for reconnection
                    delay = min(self.reconnect_delays[symbol], 60.0)  # Max 60 seconds
                    self.logger.info(f"Reconnecting to {symbol} in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    self.reconnect_delays[symbol] *= 1.5  # Increase delay for next attempt
                    
    async def _process_ticker_message(self, symbol: str, message: str):
        """Process ticker message from WebSocket."""
        try:
            data = json.loads(message)
            
            # Handle ticker events
            if data.get('event') == 'ticker' and data.get('market') == symbol:
                ticker_data = data.get('data', {})
                
                # Extract price information
                price = float(ticker_data.get('lastPrice', 0))
                if price <= 0:
                    return
                    
                volume = float(ticker_data.get('volume', 0))
                bid = ticker_data.get('bid')
                ask = ticker_data.get('ask')
                
                # Create price update
                price_update = PriceUpdate(
                    symbol=symbol,
                    price=price,
                    timestamp=datetime.now(),
                    volume=volume,
                    bid=float(bid) if bid else None,
                    ask=float(ask) if ask else None
                )
                
                # Update current price
                old_price = self.current_prices.get(symbol)
                self.current_prices[symbol] = price_update
                
                # Add to price history
                self.price_history[symbol].append(price_update)
                
                # Log significant price changes
                if old_price:
                    price_change_pct = abs(price - old_price.price) / old_price.price * 100
                    if price_change_pct > 0.1:  # Log changes > 0.1%
                        self.logger.debug(
                            f"Price update {symbol}: {old_price.price:.4f} -> {price:.4f} "
                            f"({price_change_pct:+.2f}%)"
                        )
                
                # Call registered callbacks
                for callback in self.price_update_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(price_update))
                        else:
                            callback(price_update)
                    except Exception as e:
                        self.logger.error(f"Error in price update callback: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error processing ticker message for {symbol}: {e}")
            
    async def _monitor_connection_health(self):
        """Monitor WebSocket connection health and handle reconnections."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for symbol in self.settings.symbols:
                    last_update = self.connection_health.get(symbol)
                    
                    if not last_update:
                        continue
                        
                    # Check if connection is stale (no updates for 2 minutes)
                    age = (current_time - last_update).total_seconds()
                    if age > 120:  # 2 minutes
                        self.logger.warning(
                            f"Stale connection for {symbol} (no updates for {age:.0f}s)"
                        )
                        
                        # Try to get fallback price from API
                        await self._get_fallback_price(symbol)
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in connection health monitoring: {e}")
                await asyncio.sleep(30)
                
    async def _get_fallback_price(self, symbol: str):
        """Get fallback price from REST API when WebSocket is stale."""
        try:
            if not self.http_client:
                return
                
            url = f"{self.settings.bitvavo_base_url}/ticker/price"
            params = {'market': symbol}
            
            response = await self.http_client.get(
                url, 
                params=params, 
                timeout=self.settings.price_api_timeout_seconds
            )
            
            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])
                
                # Create fallback price update
                price_update = PriceUpdate(
                    symbol=symbol,
                    price=price,
                    timestamp=datetime.now()
                )
                
                # Update current price
                old_price = self.current_prices.get(symbol)
                self.current_prices[symbol] = price_update
                self.price_history[symbol].append(price_update)
                
                self.logger.info(f"Fallback price for {symbol}: {price:.4f}")
                
                # Call callbacks
                for callback in self.price_update_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(price_update))
                        else:
                            callback(price_update)
                    except Exception as e:
                        self.logger.error(f"Error in fallback price callback: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error getting fallback price for {symbol}: {e}")
            
    def get_connection_status(self) -> Dict[str, dict]:
        """Get status of all WebSocket connections."""
        status = {}
        current_time = datetime.now()
        
        for symbol in self.settings.symbols:
            ws = self.websocket_connections.get(symbol)
            last_update = self.connection_health.get(symbol)
            current_price = self.current_prices.get(symbol)
            
            status[symbol] = {
                'connected': ws is not None and not ws.closed if ws else False,
                'last_update': last_update.isoformat() if last_update else None,
                'seconds_since_update': (
                    (current_time - last_update).total_seconds() 
                    if last_update else None
                ),
                'current_price': current_price.price if current_price else None,
                'price_age_seconds': self.get_price_age_seconds(symbol),
                'price_history_count': len(self.price_history.get(symbol, [])),
                'volatility': self.calculate_volatility(symbol)
            }
            
        return status