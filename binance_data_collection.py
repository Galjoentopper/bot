#!/usr/bin/env python3
"""
Binance 15-Minute Data Collector
Collects historical OHLCV data for crypto pairs without API keys
Stores each coin in separate SQLite database files
"""

import os
import sqlite3
import requests
import pandas as pd
import time
import zipfile
import io
import random
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RateLimitManager:
    """
    Token bucket rate limiter for managing API requests
    Ensures we stay well under Binance's rate limits
    """
    
    def __init__(self, max_requests_per_minute: int = 50, burst_capacity: int = 10):
        """
        Initialize rate limiter with conservative limits
        
        Args:
            max_requests_per_minute: Maximum requests per minute (well under Binance's 1200)
            burst_capacity: Maximum burst requests allowed
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity  # Start with full burst capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        # Calculate token refill rate (tokens per second)
        self.refill_rate = max_requests_per_minute / 60.0
        
        logger.info(f"RateLimitManager initialized: {max_requests_per_minute} req/min, burst: {burst_capacity}")
    
    def acquire(self, tokens_needed: int = 1) -> bool:
        """
        Acquire tokens for making requests
        
        Args:
            tokens_needed: Number of tokens needed
            
        Returns:
            True if tokens acquired, False if need to wait
        """
        with self.lock:
            current_time = time.time()
            
            # Refill tokens based on time elapsed
            time_elapsed = current_time - self.last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = current_time
            
            # Check if we have enough tokens
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            else:
                return False
    
    def wait_for_token(self, tokens_needed: int = 1) -> None:
        """
        Wait until tokens are available
        
        Args:
            tokens_needed: Number of tokens needed
        """
        while not self.acquire(tokens_needed):
            # Calculate wait time
            with self.lock:
                tokens_deficit = tokens_needed - self.tokens
                wait_time = tokens_deficit / self.refill_rate
                
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {tokens_needed} tokens")
            time.sleep(min(wait_time + 0.1, 2.0))  # Cap wait time at 2 seconds
    
    def get_status(self) -> Dict:
        """Get current rate limiter status"""
        with self.lock:
            current_time = time.time()
            time_elapsed = current_time - self.last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            current_tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            
            return {
                "current_tokens": current_tokens,
                "max_tokens": self.burst_capacity,
                "refill_rate": self.refill_rate,
                "requests_per_minute": self.max_requests_per_minute
            }

class BinanceDataCollector:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize Binance Data Collector with improved rate limiting
        
        Args:
            data_dir: Directory to store SQLite database files
        """
        self.base_url = "https://api.binance.com/api/v3"
        self.bulk_data_url = "https://data.binance.vision/data/spot"
        self.data_dir = data_dir
        self.symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "SOLEUR", "XRPEUR"]
        self.interval = "15m"  # 15-minute candles
        self.start_date = "2020-01-01"  # Start from 2020
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize rate limiter with very conservative settings
        self.rate_limiter = RateLimitManager(
            max_requests_per_minute=6,  # Much more conservative (vs Binance's 1200 limit)
            burst_capacity=2  # Very small burst capacity
        )
        
        # Bulk download settings (more aggressive since these don't count against API limits)
        self.bulk_request_delay = 0.5  # Reduced delay for bulk downloads
        self.bulk_max_retries = 5  # More retries for bulk downloads
        
        # API request settings (extremely conservative)
        self.api_max_retries = 3  # Fewer retries
        self.base_backoff_delay = 10.0  # Longer base delay
        self.max_backoff_delay = 600.0  # Up to 10 minutes max backoff
        
        # Very conservative batch sizing for API requests
        self.api_batch_size = 50  # Very small batches to reduce load
        self.api_min_batch_size = 10  # Minimum batch size
        
        # Bulk data cutoff - minimize API usage even more
        self.api_only_days = 3  # Only use API for last 3 days of data
        
        # Circuit breaker for API failures
        self.api_failure_count = 0
        self.max_api_failures = 3  # Stop API calls after 3 consecutive failures
        self.api_circuit_open = False
        
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with improved retry logic, rate limiting, and circuit breaker
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result or None if all retries failed
        """
        # Check circuit breaker
        if self.api_circuit_open:
            logger.warning("API circuit breaker is open, skipping request")
            return None
        
        for attempt in range(self.api_max_retries):
            try:
                # Wait for rate limit token before making request
                self.rate_limiter.wait_for_token()
                
                result = func(*args, **kwargs)
                
                if result is not None:
                    logger.debug(f"Request successful on attempt {attempt + 1}")
                    # Reset failure count on success
                    self.api_failure_count = 0
                    return result
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [429, 451, 418]:  # Rate limit errors
                    self.api_failure_count += 1
                    
                    if self.api_failure_count >= self.max_api_failures:
                        self.api_circuit_open = True
                        logger.error(f"API circuit breaker opened after {self.api_failure_count} failures")
                        return None
                    
                    if attempt < self.api_max_retries - 1:
                        # Much longer backoff for rate limits
                        backoff_delay = min(
                            self.base_backoff_delay * (3 ** attempt),  # Faster exponential growth
                            self.max_backoff_delay
                        )
                        # Larger jitter for rate limits
                        jitter = random.uniform(0.8, 1.2) * backoff_delay
                        total_delay = backoff_delay + jitter
                        
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{self.api_max_retries}), "
                            f"waiting {total_delay:.1f}s before retry. Failures: {self.api_failure_count}/{self.max_api_failures}"
                        )
                        time.sleep(total_delay)
                    else:
                        logger.error(f"All {self.api_max_retries} retry attempts failed due to rate limiting")
                        self.api_failure_count += 1
                        if self.api_failure_count >= self.max_api_failures:
                            self.api_circuit_open = True
                            logger.error("API circuit breaker opened due to repeated failures")
                        return None
                        
                elif e.response.status_code >= 500:  # Server errors
                    self.api_failure_count += 1
                    if attempt < self.api_max_retries - 1:
                        backoff_delay = min(5.0 * (2 ** attempt), 60.0)
                        logger.warning(f"Server error {e.response.status_code} (attempt {attempt + 1}/{self.api_max_retries}), retrying in {backoff_delay}s")
                        time.sleep(backoff_delay)
                    else:
                        logger.error(f"Server error {e.response.status_code} after {self.api_max_retries} attempts")
                        return None
                else:
                    # Client errors (4xx) - don't retry
                    logger.error(f"Client error {e.response.status_code}: {e}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.api_failure_count += 1
                if attempt < self.api_max_retries - 1:
                    backoff_delay = min(5.0 * (2 ** attempt), 60.0)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.api_max_retries}): {e}, retrying in {backoff_delay}s")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"All {self.api_max_retries} retry attempts failed: {e}")
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected error in retry mechanism: {e}")
                return None
        
        return None
        
    def create_database(self, symbol: str) -> str:
        """
        Create SQLite database for a specific symbol
        
        Args:
            symbol: Trading pair symbol (e.g., BTCEUR)
            
        Returns:
            Path to the database file
        """
        db_path = os.path.join(self.data_dir, f"{symbol.lower()}_15m.db")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table for OHLCV data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                datetime TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                trades INTEGER NOT NULL,
                taker_buy_base REAL NOT NULL,
                taker_buy_quote REAL NOT NULL,
                UNIQUE(timestamp)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data(timestamp)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database created/verified: {db_path}")
        return db_path
    
    def get_server_time(self) -> int:
        """
        Get Binance server time to sync requests
        
        Returns:
            Server timestamp in milliseconds
        """
        try:
            response = requests.get(f"{self.base_url}/time")
            response.raise_for_status()
            return response.json()["serverTime"]
        except Exception as e:
            logger.warning(f"Could not get server time: {e}")
            return int(time.time() * 1000)
    
    def download_bulk_file(self, symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        Download bulk historical data file from data.binance.vision with improved error handling
        
        Args:
            symbol: Trading pair symbol
            year: Year to download
            month: Month to download
            
        Returns:
            DataFrame with kline data or None if failed
        """
        filename = f"{symbol}-{self.interval}-{year:04d}-{month:02d}.zip"
        url = f"{self.bulk_data_url}/monthly/klines/{symbol}/{self.interval}/{filename}"
        
        for attempt in range(self.bulk_max_retries):
            try:
                logger.info(f"Downloading bulk file: {filename} (attempt {attempt + 1})")
                
                response = requests.get(url, timeout=60)  # Longer timeout for bulk files
                response.raise_for_status()
                
                # Extract CSV from ZIP
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    csv_filename = filename.replace('.zip', '.csv')
                    with zip_file.open(csv_filename) as csv_file:
                        df = pd.read_csv(csv_file, header=None, names=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ])

                # Normalize timestamps (Binance switched to microseconds in 2025)
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                if df['timestamp'].abs().max() > 1e14:
                    df['timestamp'] = (df['timestamp'] // 1000).astype('int64')

                # Add human-readable datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Select relevant columns
                df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 
                        'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
                
                logger.info(f"Bulk file downloaded successfully: {len(df)} candles for {symbol} {year}-{month:02d}")
                return df
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.info(f"Bulk file not available: {symbol} {year}-{month:02d} (404)")
                    return None  # Don't retry for 404s
                elif attempt < self.bulk_max_retries - 1:
                    backoff_delay = (2 ** attempt) * 2.0  # Exponential backoff
                    logger.warning(f"Bulk download HTTP error {e.response.status_code} (attempt {attempt + 1}), retrying in {backoff_delay}s")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Bulk download failed after {self.bulk_max_retries} attempts: HTTP {e.response.status_code}")
                    return None
                    
            except Exception as e:
                if attempt < self.bulk_max_retries - 1:
                    backoff_delay = (2 ** attempt) * 2.0
                    logger.warning(f"Bulk download failed (attempt {attempt + 1}): {e}, retrying in {backoff_delay}s")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Bulk download failed after {self.bulk_max_retries} attempts: {e}")
                    return None
            
            # Add delay between retries and bulk downloads
            if attempt < self.bulk_max_retries - 1:
                time.sleep(self.bulk_request_delay)
        
        return None
    
    def get_klines_api(self, symbol: str, start_time: int, end_time: int = None, limit: int = None) -> Optional[List]:
        """
        Get historical klines using API method with conservative batch sizing and rate limiting
        
        Args:
            symbol: Trading pair symbol
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds (optional)
            limit: Number of candles to fetch (uses conservative default if None)
            
        Returns:
            List of kline data or None if failed
        """
        # Use conservative batch size if limit not specified
        if limit is None:
            limit = self.api_batch_size
            
        def _make_api_request():
            current_time = int(time.time() * 1000)
            
            # Don't try to fetch future data
            if start_time >= current_time:
                logger.debug(f"Skipping future data request for {symbol}: start_time {start_time} >= current_time {current_time}")
                return []
                
            # Prepare parameters
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": start_time,
                "limit": limit
            }
            
            # Add end time if specified and valid
            if end_time:
                adjusted_end_time = min(end_time, current_time)
                params["endTime"] = adjusted_end_time
            
            logger.debug(f"API request: {symbol} from {start_time} limit {limit}")
            
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"API fetch successful: {len(data)} candles for {symbol}")
            return data
        
        return self._retry_with_backoff(_make_api_request)
    
    def get_database_gaps(self, symbol: str, db_path: str) -> List[tuple]:
        """
        Identify gaps in the database that need to be filled
        
        Args:
            symbol: Trading pair symbol
            db_path: Path to database file
            
        Returns:
            List of (start_timestamp, end_timestamp) tuples for gaps
        """
        conn = sqlite3.connect(db_path)
        
        # Get all timestamps and find gaps
        df = pd.read_sql_query("SELECT timestamp FROM market_data ORDER BY timestamp", conn)
        conn.close()
        
        if df.empty:
            # No data, entire range is a gap
            start_timestamp = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
            end_timestamp = int(time.time() * 1000)
            return [(start_timestamp, end_timestamp)]
        
        gaps = []
        expected_interval = 15 * 60 * 1000  # 15 minutes in milliseconds
        
        timestamps = df['timestamp'].tolist()
        
        # Check for gaps between consecutive timestamps
        for i in range(len(timestamps) - 1):
            current_ts = timestamps[i]
            next_ts = timestamps[i + 1]
            
            # If gap is larger than expected interval, there's missing data
            if next_ts - current_ts > expected_interval * 1.5:  # Allow some tolerance
                gap_start = current_ts + expected_interval
                gap_end = next_ts - expected_interval
                gaps.append((gap_start, gap_end))
        
        # Check if we need data before the first timestamp
        first_timestamp = timestamps[0]
        start_timestamp = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
        if first_timestamp > start_timestamp + expected_interval:
            gaps.insert(0, (start_timestamp, first_timestamp - expected_interval))
        
        # Check if we need recent data after the last timestamp
        last_timestamp = timestamps[-1]
        current_time = int(time.time() * 1000)
        # Only fetch data up to current time minus a safety buffer (1 hour)
        safe_current_time = current_time - (60 * 60 * 1000)  # 1 hour ago
        
        if last_timestamp < safe_current_time - expected_interval:
            gaps.append((last_timestamp + expected_interval, safe_current_time))
        
        return gaps

    def process_klines_data(self, klines_data: List) -> pd.DataFrame:
        """
        Process raw klines data into pandas DataFrame
        
        Args:
            klines_data: Raw klines data from Binance API
            
        Returns:
            Processed DataFrame
        """
        if not klines_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(klines_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['trades'] = pd.to_numeric(df['trades'], errors='coerce')

        # Normalize timestamps (microseconds handling)
        if df['timestamp'].abs().max() > 1e14:
            df['timestamp'] = (df['timestamp'] // 1000).astype('int64')

        # Add human-readable datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Select relevant columns
        df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 
                'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        return df
    
    def save_to_database(self, df: pd.DataFrame, db_path: str) -> int:
        """
        Save DataFrame to SQLite database
        
        Args:
            df: DataFrame with OHLCV data
            db_path: Path to SQLite database
            
        Returns:
            Number of new records inserted
        """
        if df.empty:
            return 0
        
        conn = sqlite3.connect(db_path)
        
        # Get existing timestamps to avoid duplicates
        existing_timestamps = pd.read_sql_query(
            "SELECT timestamp FROM market_data", conn
        )['timestamp'].tolist()
        
        # Filter out existing data
        new_data = df[~df['timestamp'].isin(existing_timestamps)]
        
        if not new_data.empty:
            # Insert new data
            new_data.to_sql('market_data', conn, if_exists='append', index=False)
            logger.info(f"Inserted {len(new_data)} new records")
        else:
            logger.info("No new data to insert")
        
        conn.close()
        return len(new_data)

    def collect_symbol_data(self, symbol: str) -> bool:
        """
        Collect all historical data for a specific symbol using optimized bulk-first approach
        
        This method prioritizes bulk downloads and only uses API for very recent data
        to minimize rate limiting issues.

        Args:
            symbol: Trading pair symbol

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting optimized data collection for {symbol}")
        
        # Create database
        db_path = self.create_database(symbol)
        
        # Calculate date ranges
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        current_dt = datetime.now()
        current_timestamp = int(time.time() * 1000)
        
        # Calculate cutoff for API-only data (last N days)
        api_cutoff_dt = current_dt - timedelta(days=self.api_only_days)
        api_cutoff_timestamp = int(api_cutoff_dt.timestamp() * 1000)
        
        total_records = 0
        
        logger.info(f"{symbol}: Using bulk downloads until {api_cutoff_dt.strftime('%Y-%m-%d')}, API for recent data")
        
        # Phase 1: Aggressive bulk downloading for historical data
        logger.info(f"Phase 1: Bulk downloading historical data for {symbol}")
        
        # Download ALL available monthly files from start date to API cutoff
        current_month_dt = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        while current_month_dt <= api_cutoff_dt:
            year = current_month_dt.year
            month = current_month_dt.month
            
            logger.info(f"{symbol}: Downloading bulk data for {year}-{month:02d}")
            
            df = self.download_bulk_file(symbol, year, month)
            
            if df is not None and not df.empty:
                new_records = self.save_to_database(df, db_path)
                total_records += new_records
                logger.info(f"{symbol}: Bulk {year}-{month:02d} - {new_records} new records")
            else:
                logger.info(f"{symbol}: No bulk data available for {year}-{month:02d}")
            
            # Rate limiting for bulk downloads
            time.sleep(self.bulk_request_delay)
            
            # Move to next month
            if current_month_dt.month == 12:
                current_month_dt = current_month_dt.replace(year=current_month_dt.year + 1, month=1)
            else:
                current_month_dt = current_month_dt.replace(month=current_month_dt.month + 1)
        
        # Phase 2: Use API only for recent data and critical gaps
        logger.info(f"Phase 2: Using API for recent data for {symbol}")
        
        # Check remaining gaps after bulk downloads
        remaining_gaps = self.get_database_gaps(symbol, db_path)
        
        # Filter gaps to only work on recent data (last api_only_days)
        recent_gaps = []
        for gap_start, gap_end in remaining_gaps:
            # Only process gaps that are in the recent data range
            if gap_start >= api_cutoff_timestamp:
                recent_gaps.append((gap_start, gap_end))
            elif gap_end >= api_cutoff_timestamp:
                # Partial gap in recent range
                recent_gaps.append((max(gap_start, api_cutoff_timestamp), gap_end))
        
        logger.info(f"{symbol}: Found {len(recent_gaps)} gaps in recent data requiring API calls")
        
        for gap_start, gap_end in recent_gaps:
            # Check circuit breaker before processing gaps
            if self.api_circuit_open:
                logger.warning(f"{symbol}: API circuit breaker is open, skipping remaining gaps")
                break
                
            # Ensure we don't try to fetch future data
            gap_end = min(gap_end, current_timestamp)
            
            if gap_start >= gap_end:
                continue  # Skip if gap is invalid
            
            logger.info(f"{symbol}: Filling recent gap from {datetime.fromtimestamp(gap_start/1000)} to {datetime.fromtimestamp(gap_end/1000)}")
            
            current_ts = gap_start
            
            while current_ts < gap_end and not self.api_circuit_open:
                # Skip if we're trying to fetch future data
                if current_ts >= current_timestamp:
                    logger.debug(f"{symbol}: Reached current time, stopping API requests")
                    break
                
                # Use conservative API method
                klines_data = self.get_klines_api(symbol, current_ts, gap_end)
                
                if klines_data is None:
                    if self.api_circuit_open:
                        logger.warning(f"{symbol}: API circuit breaker opened, stopping gap filling")
                        break
                    
                    logger.error(f"{symbol}: API request failed at timestamp {current_ts}")
                    # Skip ahead by one day on failure to avoid getting stuck
                    current_ts += (24 * 60 * 60 * 1000)  # Skip 1 day
                    continue
                
                # Process and save data
                df = self.process_klines_data(klines_data)
                
                if not df.empty:
                    new_records = self.save_to_database(df, db_path)
                    total_records += new_records
                    
                    # Update current timestamp to last candle + 1 interval
                    current_ts = int(df['timestamp'].max()) + (15 * 60 * 1000)
                    
                    # Show progress
                    if gap_end > gap_start:
                        progress = ((current_ts - gap_start) / (gap_end - gap_start)) * 100
                        logger.info(f"{symbol}: Recent gap filling {progress:.1f}% complete")
                else:
                    # If no data returned, skip ahead by batch size
                    current_ts += (self.api_batch_size * 15 * 60 * 1000)
                
                # Important: Small delay to respect rate limits (in addition to rate limiter)
                time.sleep(0.5)  # 500ms delay between API calls
        
        # Phase 3: Validation and summary
        final_gaps = self.get_database_gaps(symbol, db_path)
        
        # Only report significant gaps (larger than 1 day)
        significant_gaps = [
            gap for gap in final_gaps 
            if (gap[1] - gap[0]) > (24 * 60 * 60 * 1000)  # > 1 day
        ]
        
        if significant_gaps:
            logger.warning(f"{symbol}: {len(significant_gaps)} significant gaps remain (>1 day each)")
            for gap_start, gap_end in significant_gaps[:3]:  # Show first 3 gaps
                logger.warning(f"  Gap: {datetime.fromtimestamp(gap_start/1000)} to {datetime.fromtimestamp(gap_end/1000)}")
        else:
            logger.info(f"{symbol}: Data collection complete, no significant gaps detected")
        
        logger.info(f"Completed {symbol}: {total_records} total new records collected")
        return True
    
    def collect_all_data(self) -> Dict[str, bool]:
        """
        Collect data for all configured symbols
        
        Returns:
            Dictionary with symbol: success status
        """
        results = {}
        
        logger.info(f"Starting data collection for {len(self.symbols)} symbols")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Timeframe: {self.interval}")
        logger.info(f"Start date: {self.start_date}")
        logger.info(f"Data directory: {self.data_dir}")
        
        for symbol in self.symbols:
            try:
                success = self.collect_symbol_data(symbol)
                results[symbol] = success
                
                if success:
                    logger.info(f"✅ {symbol} collection completed successfully")
                else:
                    logger.error(f"❌ {symbol} collection failed")
                    
            except Exception as e:
                logger.error(f"❌ {symbol} collection failed with exception: {e}")
                results[symbol] = False
            
            # Brief pause between symbols
            time.sleep(1)
        
        return results
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics for collected data
        
        Returns:
            Dictionary with summary for each symbol
        """
        summary = {}
        
        for symbol in self.symbols:
            db_path = os.path.join(self.data_dir, f"{symbol.lower()}_15m.db")
            
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                
                # Get basic stats
                stats = pd.read_sql_query("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(datetime) as start_date,
                        MAX(datetime) as end_date,
                        MIN(close) as min_price,
                        MAX(close) as max_price,
                        AVG(volume) as avg_volume
                    FROM market_data
                """, conn).iloc[0].to_dict()
                
                conn.close()
                summary[symbol] = stats
            else:
                summary[symbol] = {"status": "No database file found"}
        
        return summary

def main():
    """
    Main function to run the data collector
    """
    print("🚀 Binance 15-Minute Data Collector")
    print("====================================")
    
    # Initialize collector
    collector = BinanceDataCollector()
    
    # Collect all data
    results = collector.collect_all_data()
    
    # Print results
    print("\n📊 Collection Results:")
    print("=====================")
    for symbol, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{symbol}: {status}")
    
    # Print summary
    print("\n📈 Data Summary:")
    print("===============")
    summary = collector.get_data_summary()
    
    for symbol, stats in summary.items():
        print(f"\n{symbol}:")
        if "status" in stats:
            print(f"  {stats['status']}")
        else:
            print(f"  Records: {stats['total_records']:,}")
            print(f"  Date range: {stats['start_date']} to {stats['end_date']}")
            print(f"  Price range: €{stats['min_price']:.4f} - €{stats['max_price']:.4f}")
            print(f"  Avg volume: {stats['avg_volume']:,.2f}")
    
    print("\n🎉 Data collection completed!")
    print(f"📁 Data stored in: {collector.data_dir}/")
    print("💡 Each coin has its own SQLite database file")

if __name__ == "__main__":
    main()