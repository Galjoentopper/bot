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
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            }
            
            response = requests.get(f"{self.base_url}/klines", params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"API fetch: {len(data)} candles for {symbol}")
            return data
            
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
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": start_time,
                "limit": limit
            }
            
            response = requests.get(f"{self.base_url}/klines", params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Regular fetch: {len(data)} candles for {symbol}")
            return data
            
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
    
    def collect_symbol_data(self, symbol: str) -> bool:
        """
        Collect all historical data for a specific symbol using hybrid approach

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
        
        total_records = 0
        
        # Phase 1: Download bulk monthly files (much faster)
        logger.info(f"Phase 1: Downloading bulk monthly files for {symbol}")
        
        year = start_dt.year
        month = start_dt.month
        
        while year < current_dt.year or (year == current_dt.year and month < current_dt.month):
            # Try to download bulk file for this month
            df = self.download_bulk_file(symbol, year, month)
            
            if df is not None and not df.empty:
                new_records = self.save_to_database(df, db_path)
                total_records += new_records
                logger.info(f"{symbol}: Bulk {year}-{month:02d} - {new_records} new records")
            else:
                logger.warning(f"{symbol}: No bulk data for {year}-{month:02d}, will use API later")
            
            # Rate limiting for bulk downloads
            time.sleep(self.bulk_request_delay)
            
            # Move to next month
            month += 1
            if month > 12:
                month = 1
                year += 1
        
        # Phase 2: Use API for recent data (current month and any missing data)
        logger.info(f"Phase 2: Using API for recent data for {symbol}")
        
        # Get the latest timestamp in database
        conn = sqlite3.connect(db_path)
        latest_timestamp_query = "SELECT MAX(timestamp) FROM market_data"
        latest_result = pd.read_sql_query(latest_timestamp_query, conn)
        conn.close()
        
        if latest_result.iloc[0, 0] is not None:
            current_timestamp = int(latest_result.iloc[0, 0]) + (15 * 60 * 1000)  # Start from next 15min
        else:
            current_timestamp = int(start_dt.timestamp() * 1000)
        
        end_timestamp = int(time.time() * 1000)
        
        # Fill gaps with API calls
        while current_timestamp < end_timestamp:
            # Calculate batch end time
            batch_end = min(current_timestamp + (1000 * 15 * 60 * 1000), end_timestamp)  # 1000 candles worth
            
            # Use API method
            klines_data = self.get_klines_api(symbol, current_timestamp, batch_end, 1000)
            
            # Fallback to smaller batches if needed
            if klines_data is None:
                logger.warning(f"Large API batch failed for {symbol}, trying smaller batch")
                klines_data = self.get_klines_regular(symbol, current_timestamp, 500)
                
                if klines_data is None:
                    logger.error(f"API methods failed for {symbol} at timestamp {current_timestamp}")
                    break
            
            # Process and save data
            df = self.process_klines_data(klines_data)
            
            if not df.empty:
                new_records = self.save_to_database(df, db_path)
                total_records += new_records
                
                # Update current timestamp to last candle + 1
                current_timestamp = int(df['timestamp'].max()) + (15 * 60 * 1000)  # Add 15 minutes
            else:
                # If no data, move forward by a day
                current_timestamp += (24 * 60 * 60 * 1000)
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            # Progress update
            progress = ((current_timestamp - int(start_dt.timestamp() * 1000)) / (end_timestamp - int(start_dt.timestamp() * 1000))) * 100
            logger.info(f"{symbol}: API phase {progress:.1f}% complete, {total_records} total records")
        
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