import os
import sqlite3
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Data fetcher for retrieving historical and live market data.
    
    This class interfaces with the existing data collection system
    and provides a clean API for the feature factory and models.
    """
    
    def __init__(self, symbol: str, data_dir: str = None):
        """
        Initialize the DataFetcher.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'BTCEUR')
            data_dir: Directory containing the data files
        """
        self.symbol = symbol.upper()
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
        self.base_url = "https://api.binance.com/api/v3"
        
        # Determine database file name
        self.db_file = os.path.join(self.data_dir, f"{symbol.lower()}_15m.db")
        
    def get_historical_data(self, limit: int = 1000, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical data from the local database or API.
        
        Args:
            limit: Maximum number of records to retrieve
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try to get data from local database first
        if os.path.exists(self.db_file):
            try:
                return self._get_data_from_db(limit, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to get data from database: {e}")
        
        # Fallback to API
        logger.info(f"Getting data from API for {self.symbol}")
        return self._get_data_from_api(limit)
    
    def _get_data_from_db(self, limit: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get data from local SQLite database."""
        conn = sqlite3.connect(self.db_file)
        
        query = """
        SELECT open_time, open, high, low, close, volume
        FROM klines
        WHERE 1=1
        """
        params = []
        
        if start_date:
            # Convert date to timestamp
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            query += " AND open_time >= ?"
            params.append(start_ts)
        
        if end_date:
            # Convert date to timestamp
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            query += " AND open_time <= ?"
            params.append(end_ts)
        
        query += " ORDER BY open_time DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            raise ValueError("No data found in database")
        
        # Reverse to get chronological order
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Ensure numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def _get_data_from_api(self, limit: int = 1000) -> pd.DataFrame:
        """Get data from Binance API."""
        endpoint = f"{self.base_url}/klines"
        params = {
            'symbol': self.symbol,
            'interval': '15m',
            'limit': min(limit, 1000)  # API limit
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise ValueError("No data received from API")
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            logger.error(f"Failed to get data from API: {e}")
            raise
    
    def get_latest_data(self, count: int = 1) -> pd.DataFrame:
        """
        Get the most recent market data.
        
        Args:
            count: Number of latest candles to retrieve
            
        Returns:
            DataFrame with latest OHLCV data
        """
        return self.get_historical_data(limit=count)
    
    def get_current_price(self) -> Dict[str, float]:
        """
        Get current price information.
        
        Returns:
            Dictionary with current price information
        """
        endpoint = f"{self.base_url}/ticker/price"
        params = {'symbol': self.symbol}
        
        try:
            response = requests.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': data['symbol'],
                'price': float(data['price']),
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            # Fallback to latest historical data
            latest = self.get_latest_data(1)
            if not latest.empty:
                return {
                    'symbol': self.symbol,
                    'price': float(latest.iloc[-1]['close']),
                    'timestamp': latest.iloc[-1]['timestamp']
                }
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the data is suitable for feature engineering.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Expected: {required_columns}")
            return False
        
        # Check for sufficient data
        if len(df) < 200:  # Need at least 200 periods for indicators
            logger.error(f"Insufficient data: {len(df)} rows (need at least 200)")
            return False
        
        # Check for null values
        if df[required_columns].isnull().any().any():
            logger.warning("Data contains null values")
        
        # Check for zero prices
        price_columns = ['open', 'high', 'low', 'close']
        if (df[price_columns] <= 0).any().any():
            logger.error("Data contains zero or negative prices")
            return False
        
        # Check for logical consistency (high >= low, etc.)
        if not (df['high'] >= df['low']).all():
            logger.error("Data inconsistency: high < low")
            return False
        
        if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
            logger.error("Data inconsistency: high < open/close")
            return False
        
        if not ((df['low'] <= df['open']) & (df['low'] <= df['close'])).all():
            logger.error("Data inconsistency: low > open/close")
            return False
        
        return True