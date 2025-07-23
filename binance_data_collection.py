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
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceDataCollector:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize Binance Data Collector
        
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
        
        # Rate limiting
        self.request_delay = 0.1  # 100ms between requests
        self.bulk_request_delay = 0.5  # 500ms between bulk file downloads
        
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
        Download bulk historical data file from data.binance.vision
        
        Args:
            symbol: Trading pair symbol
            year: Year to download
            month: Month to download
            
        Returns:
            DataFrame with kline data or None if failed
        """
        try:
            # Format: BTCEUR-15m-2020-01.zip
            filename = f"{symbol}-{self.interval}-{year:04d}-{month:02d}.zip"
            url = f"{self.bulk_data_url}/monthly/klines/{symbol}/{self.interval}/{filename}"
            
            logger.info(f"Downloading bulk file: {filename}")
            response = requests.get(url, timeout=30)
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
            
            logger.info(f"Bulk file downloaded: {len(df)} candles for {symbol} {year}-{month:02d}")
            return df
            
        except Exception as e:
            logger.warning(f"Bulk file download failed for {symbol} {year}-{month:02d}: {e}")
            return None
    
    def get_klines_api(self, symbol: str, start_time: int, end_time: int, limit: int = 1000) -> Optional[List]:
        """
        Get historical klines using API method (fallback)
        
        Args:
            symbol: Trading pair symbol
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of candles to fetch (max 1000)
            
        Returns:
            List of kline data or None if failed
        """
        try:
            # Don't try to fetch future data
            current_time = int(time.time() * 1000)
            if start_time >= current_time:
                logger.warning(f"Skipping future data request for {symbol}: start_time {start_time} >= current_time {current_time}")
                return None
                
            # Adjust end_time if it's in the future
            end_time = min(end_time, current_time)
            
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            }
            
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"API fetch: {len(data)} candles for {symbol}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 451:
                logger.error(f"API fetch failed for {symbol}: 451 error (rate limit or data unavailable) for timestamp {start_time}")
            else:
                logger.error(f"API fetch failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"API fetch failed for {symbol}: {e}")
            return None
    
    def get_klines_regular(self, symbol: str, start_time: int, limit: int = 500) -> Optional[List]:
        """
        Get historical klines using regular method (fallback)
        
        Args:
            symbol: Trading pair symbol
            start_time: Start timestamp in milliseconds
            limit: Number of candles to fetch (max 500 for regular)
            
        Returns:
            List of kline data or None if failed
        """
        try:
            # Don't try to fetch future data
            current_time = int(time.time() * 1000)
            if start_time >= current_time:
                logger.warning(f"Skipping future data request for {symbol}: start_time {start_time} >= current_time {current_time}")
                return None
                
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": start_time,
                "limit": limit
            }
            
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Regular fetch: {len(data)} candles for {symbol}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 451:
                logger.error(f"Regular fetch failed for {symbol}: 451 error (rate limit or data unavailable) for timestamp {start_time}")
            else:
                logger.error(f"Regular fetch failed for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Regular fetch failed for {symbol}: {e}")
            return None
    
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
        # Only fetch data up to current time minus a safety buffer (2 hours)
        # This prevents trying to fetch very recent data that might not be available
        safe_current_time = current_time - (2 * 60 * 60 * 1000)  # 2 hours ago
        
        if last_timestamp < safe_current_time - expected_interval:
            # Make sure the gap doesn't extend into very recent time
            gap_end = min(safe_current_time, last_timestamp + (7 * 24 * 60 * 60 * 1000))  # Limit to 7 days from last data
            if gap_end > last_timestamp + expected_interval:
                gaps.append((last_timestamp + expected_interval, gap_end))
        
        return gaps

    def collect_symbol_data(self, symbol: str) -> bool:
        """
        Collect all historical data for a specific symbol using improved hybrid approach

        Args:
            symbol: Trading pair symbol

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting data collection for {symbol}")
        
        # Create database
        db_path = self.create_database(symbol)
        
        # Calculate start and end dates
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        current_dt = datetime.now()
        current_timestamp = int(time.time() * 1000)
        
        total_records = 0
        
        # First, identify what data we already have and what gaps exist
        gaps = self.get_database_gaps(symbol, db_path)
        logger.info(f"{symbol}: Found {len(gaps)} data gaps to fill")
        
        if not gaps:
            logger.info(f"{symbol}: Database is up to date, no gaps found")
            return True
        
        # Phase 1: Download bulk monthly files for large gaps (much faster)
        logger.info(f"Phase 1: Downloading bulk monthly files for {symbol}")
        
        # Organize gaps by month for bulk downloading
        monthly_gaps = {}
        for gap_start, gap_end in gaps:
            gap_start_dt = datetime.fromtimestamp(gap_start / 1000)
            gap_end_dt = datetime.fromtimestamp(gap_end / 1000)
            
            # Process each month in the gap
            current_month_dt = gap_start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current_month_dt <= gap_end_dt:
                year_month = (current_month_dt.year, current_month_dt.month)
                if year_month not in monthly_gaps:
                    monthly_gaps[year_month] = []
                monthly_gaps[year_month].append((gap_start, gap_end))
                
                # Move to next month
                if current_month_dt.month == 12:
                    current_month_dt = current_month_dt.replace(year=current_month_dt.year + 1, month=1)
                else:
                    current_month_dt = current_month_dt.replace(month=current_month_dt.month + 1)
        
        # Download bulk files for identified months
        for (year, month), _ in monthly_gaps.items():
            # Only download if the month is in the past (not current or future months)
            month_end = datetime(year, month, 1)
            if month < 12:
                month_end = month_end.replace(month=month + 1)
            else:
                month_end = month_end.replace(year=year + 1, month=1)
            
            if month_end < current_dt:
                df = self.download_bulk_file(symbol, year, month)
                
                if df is not None and not df.empty:
                    new_records = self.save_to_database(df, db_path)
                    total_records += new_records
                    logger.info(f"{symbol}: Bulk {year}-{month:02d} - {new_records} new records")
                else:
                    logger.warning(f"{symbol}: No bulk data for {year}-{month:02d}")
                
                # Rate limiting for bulk downloads
                time.sleep(self.bulk_request_delay)
        
        # Phase 2: Use API for recent data and remaining gaps
        logger.info(f"Phase 2: Using API for recent data and remaining gaps for {symbol}")
        
        # Recalculate gaps after bulk downloads
        remaining_gaps = self.get_database_gaps(symbol, db_path)
        
        for gap_start, gap_end in remaining_gaps:
            # Ensure we don't try to fetch future data
            gap_end = min(gap_end, current_timestamp)
            
            if gap_start >= gap_end:
                continue  # Skip if gap is invalid
            
            logger.info(f"{symbol}: Filling gap from {datetime.fromtimestamp(gap_start/1000)} to {datetime.fromtimestamp(gap_end/1000)}")
            
            current_ts = gap_start
            
            while current_ts < gap_end:
                # Calculate batch end time (don't exceed gap end or current time)
                batch_end = min(current_ts + (1000 * 15 * 60 * 1000), gap_end, current_timestamp)
                
                # Skip if we're trying to fetch future data
                if current_ts >= current_timestamp:
                    logger.warning(f"{symbol}: Skipping future timestamp {current_ts} (current: {current_timestamp})")
                    break
                
                # Use API method
                klines_data = self.get_klines_api(symbol, current_ts, batch_end, 1000)
                
                # Fallback to smaller batches if needed
                if klines_data is None:
                    logger.warning(f"Large API batch failed for {symbol}, trying smaller batch")
                    klines_data = self.get_klines_regular(symbol, current_ts, 500)
                    
                    if klines_data is None:
                        logger.error(f"API methods failed for {symbol} at timestamp {current_ts}")
                        # Skip this batch and continue
                        current_ts += (100 * 15 * 60 * 1000)  # Skip ahead by 100 candles
                        continue
                
                # Process and save data
                df = self.process_klines_data(klines_data)
                
                if not df.empty:
                    new_records = self.save_to_database(df, db_path)
                    total_records += new_records
                    
                    # Update current timestamp to last candle + 1
                    current_ts = int(df['timestamp'].max()) + (15 * 60 * 1000)  # Add 15 minutes
                else:
                    # If no data, move forward by a smaller amount
                    current_ts += (50 * 15 * 60 * 1000)  # Skip ahead by 50 candles
                
                # Rate limiting
                time.sleep(self.request_delay)
                
                # Progress update
                if gap_end > gap_start:
                    progress = ((current_ts - gap_start) / (gap_end - gap_start)) * 100
                    logger.info(f"{symbol}: Gap filling {progress:.1f}% complete")
        
        logger.info(f"Completed {symbol}: {total_records} total records collected")
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