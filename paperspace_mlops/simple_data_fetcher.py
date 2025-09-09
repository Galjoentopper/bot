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
        """Fetch data from multiple sources with aggressive fallback for geo-restricted environments"""
        
        logger.info(f"📊 Fetching {symbol} data ({interval}, {days} days)")
        
        # Prioritize geo-restriction friendly sources first
        methods = [
            ("yfinance", self._fetch_yfinance),
            ("coingecko", self._fetch_coingecko),
            ("yahoo_crypto", self._fetch_yahoo_crypto),
            ("coinbase", self._fetch_coinbase),
            ("kraken", self._fetch_kraken),
            ("binance_proxy", self._fetch_binance_proxy),
            ("binance_eur", self._fetch_binance_eur),
            ("binance_usdt", self._fetch_binance_usdt),
            ("alternative_api", self._fetch_alternative_data)
        ]
        
        for method_name, method in methods:
            try:
                logger.info(f"  🔄 Trying {method_name} for {symbol}...")
                data = method(symbol, interval, days)
                
                if data is not None and len(data) > 15:  # Very low threshold
                    logger.info(f"  ✅ {method_name} {symbol}: {len(data)} samples")
                    return data
                else:
                    logger.warning(f"  ⚠️ {method_name} {symbol}: Insufficient data ({len(data) if data is not None else 0} samples)")
                    
            except Exception as e:
                logger.warning(f"  ❌ {method_name} {symbol}: {e}")
        
        logger.error(f"❌ All methods failed for {symbol}")
        return None
    
    def _fetch_yfinance(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from yfinance"""
        import yfinance as yf
        
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
        
        # Try different intervals if the requested one fails
        intervals_to_try = [interval, "1h", "1d", "30m"]
        
        for try_interval in intervals_to_try:
            try:
                hist = ticker.history(period=period, interval=try_interval)
                if len(hist) > 15:
                    return hist
            except Exception:
                continue
                
        return None
    
    def _fetch_binance_eur(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Try binance with EUR pairs"""
        return self._fetch_binance_api(symbol, interval, days)
    
    def _fetch_binance_usdt(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Try binance with USDT pairs"""
        usdt_symbol = symbol.replace("EUR", "USDT")
        return self._fetch_binance_api(usdt_symbol, interval, days)
    
    def _fetch_binance_api(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Generic binance API fetch"""
        import requests
        
        url = "https://api.binance.com/api/v3/klines"
        
        # Convert interval
        binance_interval = interval
        if interval == "30m":
            binance_interval = "30m"
        elif interval == "1h":
            binance_interval = "1h"
        elif interval == "1d":
            binance_interval = "1d"
        elif interval == "4h":
            binance_interval = "4h"
        elif interval == "2h":
            binance_interval = "2h"
        
        params = {
            "symbol": symbol,
            "interval": binance_interval,
            "limit": min(days * 24 if "h" in interval else days, 1000)
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
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
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
    
    def _fetch_coingecko(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from CoinGecko API (geo-restriction friendly)"""
        try:
            import requests
            
            # Map symbols to CoinGecko IDs
            symbol_map = {
                "BTCEUR": "bitcoin",
                "ETHEUR": "ethereum", 
                "ADAEUR": "cardano",
                "DOTEUR": "polkadot",
                "LINKEUR": "chainlink"
            }
            
            if symbol not in symbol_map:
                return None
                
            coin_id = symbol_map[symbol]
            
            # CoinGecko free tier allows up to 30 days of hourly data
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "eur",
                "days": min(days, 30),
                "interval": "hourly"
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])
                volumes = data.get("total_volumes", [])
                
                if len(prices) > 15:
                    # Convert to DataFrame
                    df_data = []
                    for i, (timestamp, price) in enumerate(prices):
                        volume = volumes[i][1] if i < len(volumes) else 0
                        df_data.append({
                            'Datetime': pd.to_datetime(timestamp, unit='ms'),
                            'Open': price,
                            'High': price * 1.002,  # Estimate high/low from price
                            'Low': price * 0.998,
                            'Close': price,
                            'Volume': volume
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('Datetime', inplace=True)
                    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {e}")
        return None
    
    def _fetch_yahoo_crypto(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch crypto data from Yahoo Finance with different symbol format"""
        try:
            import yfinance as yf
            
            # Convert to Yahoo crypto format
            yahoo_symbol = symbol.replace("EUR", "-EUR") + "="
            if symbol == "BTCEUR":
                yahoo_symbol = "BTC-EUR"
            elif symbol == "ETHEUR": 
                yahoo_symbol = "ETH-EUR"
            elif symbol == "ADAEUR":
                yahoo_symbol = "ADA-EUR"
            elif symbol == "DOTEUR":
                yahoo_symbol = "DOT-EUR" 
            elif symbol == "LINKEUR":
                yahoo_symbol = "LINK-EUR"
                
            ticker = yf.Ticker(yahoo_symbol)
            
            # Calculate period
            if days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            else:
                period = "6mo"
            
            hist = ticker.history(period=period, interval=interval)
            if len(hist) > 15:
                return hist[['Open', 'High', 'Low', 'Close', 'Volume']]
                
        except Exception as e:
            logger.warning(f"Yahoo crypto fetch failed: {e}")
        return None
    
    def _fetch_coinbase(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from Coinbase Pro API"""
        try:
            import requests
            from datetime import datetime, timedelta
            
            # Map to Coinbase product IDs
            coinbase_map = {
                "BTCEUR": "BTC-EUR",
                "ETHEUR": "ETH-EUR",
                "ADAEUR": "ADA-EUR", 
                "DOTEUR": "DOT-EUR",
                "LINKEUR": "LINK-EUR"
            }
            
            if symbol not in coinbase_map:
                return None
                
            product_id = coinbase_map[symbol]
            
            # Coinbase granularity (seconds)
            granularity_map = {
                "1h": 3600,
                "4h": 14400,
                "1d": 86400
            }
            granularity = granularity_map.get(interval, 3600)
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
            params = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(), 
                "granularity": granularity
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 15:
                    # Coinbase format: [timestamp, low, high, open, close, volume]
                    df_data = []
                    for candle in data:
                        df_data.append({
                            'Datetime': pd.to_datetime(candle[0], unit='s'),
                            'Open': candle[3],
                            'High': candle[2], 
                            'Low': candle[1],
                            'Close': candle[4],
                            'Volume': candle[5]
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('Datetime', inplace=True)
                    df = df.sort_index()
                    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    
        except Exception as e:
            logger.warning(f"Coinbase fetch failed: {e}")
        return None
    
    def _fetch_kraken(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from Kraken API"""
        try:
            import requests
            
            # Map to Kraken pairs
            kraken_map = {
                "BTCEUR": "XBTEUR",
                "ETHEUR": "ETHEUR",
                "ADAEUR": "ADAEUR",
                "DOTEUR": "DOTEUR", 
                "LINKEUR": "LINKEUR"
            }
            
            if symbol not in kraken_map:
                return None
                
            pair = kraken_map[symbol]
            
            # Kraken interval mapping
            interval_map = {
                "1h": 60,
                "4h": 240,
                "1d": 1440
            }
            kraken_interval = interval_map.get(interval, 60)
            
            url = "https://api.kraken.com/0/public/OHLC"
            params = {
                "pair": pair,
                "interval": kraken_interval
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and pair in data["result"]:
                    ohlc_data = data["result"][pair]
                    if len(ohlc_data) > 15:
                        df_data = []
                        for candle in ohlc_data:
                            df_data.append({
                                'Datetime': pd.to_datetime(candle[0], unit='s'),
                                'Open': float(candle[1]),
                                'High': float(candle[2]),
                                'Low': float(candle[3]), 
                                'Close': float(candle[4]),
                                'Volume': float(candle[6])
                            })
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('Datetime', inplace=True)
                        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        
        except Exception as e:
            logger.warning(f"Kraken fetch failed: {e}")
        return None
    
    def _fetch_binance_proxy(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch from Binance using proxy servers to bypass geo-restrictions"""
        try:
            import requests
            
            # Free proxy list - rotate through different ones
            proxy_urls = [
                "https://api.binance.us/api/v3/klines",  # US version
                "https://dapi.binance.com/dapi/v1/klines",  # Futures API (different endpoint)
                "https://api1.binance.com/api/v3/klines",  # Alternative mirror
                "https://api2.binance.com/api/v3/klines",  # Alternative mirror
                "https://api3.binance.com/api/v3/klines"   # Alternative mirror
            ]
            
            for proxy_url in proxy_urls:
                try:
                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "limit": min(days * 24 if "h" in interval else days, 1000)
                    }
                    
                    # Try with different headers to avoid detection
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'application/json',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Cache-Control': 'no-cache'
                    }
                    
                    response = requests.get(proxy_url, params=params, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if len(data) > 15:
                            return self._process_binance_data(data)
                            
                except Exception as e:
                    logger.debug(f"Proxy {proxy_url} failed: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Binance proxy fetch failed: {e}")
        return None
    
    def _process_binance_data(self, data) -> pd.DataFrame:
        """Process Binance API response into DataFrame"""
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
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
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