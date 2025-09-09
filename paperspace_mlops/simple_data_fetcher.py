#!/usr/bin/env python3
"""
Simple Data Fetcher
==================

Direct data fetching for Paperspace that bypasses database lookups.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SimpleDataFetcher:
    """Simple data fetcher that gets fresh data directly from sources"""
    
    def fetch_symbol_data(self, symbol: str, interval: str = "1h", days: int = 180) -> Optional[pd.DataFrame]:
        """Fetch data directly from yfinance"""
        
        logger.info(f"📊 Fetching {symbol} data ({interval}, {days} days)")
        
        try:
            import yfinance as yf
            
            # Try yfinance first
            ticker = yf.Ticker(symbol)
            
            # Calculate period
            if days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"  
            elif days <= 180:
                period = "6mo"
            else:
                period = "1y"
            
            # Fetch data
            hist = ticker.history(period=period, interval=interval)
            
            if len(hist) > 50:
                logger.info(f"✅ yfinance {symbol}: {len(hist)} samples")
                return hist
            else:
                logger.warning(f"⚠️ yfinance {symbol}: Only {len(hist)} samples")
                
        except Exception as e:
            logger.error(f"❌ yfinance {symbol}: {e}")
        
        # Try alternative approach with requests
        try:
            logger.info(f"🔄 Trying alternative data source for {symbol}...")
            return self._fetch_alternative_data(symbol, interval, days)
            
        except Exception as e:
            logger.error(f"❌ Alternative fetch {symbol}: {e}")
            
        return None
    
    def _fetch_alternative_data(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Alternative data fetching method"""
        
        try:
            import requests
            from datetime import datetime
            
            # Convert EUR pairs to USDT for some APIs
            alt_symbol = symbol.replace("EUR", "USDT")
            
            # Try simple HTTP request to binance
            url = "https://api.binance.com/api/v3/klines"
            
            # Convert interval
            binance_interval = interval
            if interval == "30m":
                binance_interval = "30m"
            elif interval == "1h":
                binance_interval = "1h"
            
            params = {
                "symbol": alt_symbol,
                "interval": binance_interval,
                "limit": min(days * 24 if interval == "1h" else days * 48, 1000)
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if len(data) > 10:
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert types
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'timestamp': 'Datetime',
                        'open': 'Open',
                        'high': 'High', 
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                    
                    df.set_index('Datetime', inplace=True)
                    
                    logger.info(f"✅ Alternative {symbol}: {len(df)} samples")
                    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    
        except Exception as e:
            logger.error(f"❌ Alternative method: {e}")
            
        return None
    
    def build_simple_dataset(self, symbol: str, interval: str = "1h") -> Optional[Tuple]:
        """Build a simple dataset with basic features"""
        
        logger.info(f"🏗️ Building simple dataset for {symbol}")
        
        # Fetch raw data
        data = self.fetch_symbol_data(symbol, interval, days=90)
        
        if data is None or len(data) < 50:
            logger.error(f"❌ Insufficient data for {symbol}")
            return None
        
        try:
            # Create basic features
            df = data.copy()
            
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['high_low_ratio'] = df['High'] / df['Low']
            df['price_change'] = df['Close'] - df['Open']
            df['volume_ma'] = df['Volume'].rolling(window=10).mean()
            
            # Technical indicators
            df['sma_5'] = df['Close'].rolling(window=5).mean()
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['rsi'] = self._calculate_rsi(df['Close'], window=14)
            
            # Target (next period return)
            df['target'] = df['returns'].shift(-1)
            
            # Drop NaN values
            df = df.dropna()
            
            if len(df) < 30:
                logger.error(f"❌ Insufficient clean data for {symbol}: {len(df)}")
                return None
            
            # Prepare features and target
            feature_cols = ['returns', 'high_low_ratio', 'price_change', 'volume_ma', 'sma_5', 'sma_20', 'rsi']
            X = df[feature_cols].values
            y = df['target'].values
            timestamps = df.index
            
            logger.info(f"✅ {symbol}: {len(X)} samples, {X.shape[1]} features")
            
            return X, y, timestamps, feature_cols, {'symbol': symbol, 'samples': len(X)}
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi