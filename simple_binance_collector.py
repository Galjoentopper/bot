#!/usr/bin/env python3
"""
Simple Binance Data Collector
=============================

A simplified version of the binance data collection script focused on:
1. 15-minute interval data collection only
2. Bulk download first, API fallback
3. Simple, robust implementation
4. Databases saved in same folder as script
5. Easy to understand and maintain

Supports the same symbols: BTCEUR, ETHEUR, ADAEUR, SOLEUR, XRPEUR
"""

import os
import sqlite3
import requests
import zipfile
import io
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleBinanceCollector:
    """Simple Binance data collector with bulk download + API fallback"""
    
    def __init__(self, symbols: List[str] = None, data_dir: str = None):
        """Initialize the collector
        
        Args:
            symbols: List of symbols to collect (default: major EUR pairs)
            data_dir: Directory to save databases (default: same as script)
        """
        self.symbols = symbols or ["BTCEUR", "ETHEUR", "ADAEUR", "SOLEUR", "XRPEUR"]
        
        # Set data directory to same folder as script if not specified
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = script_dir
        else:
            self.data_dir = data_dir
        
        self.interval = "15m"  # Fixed to 15-minute intervals
        self.base_api_url = "https://api.binance.com/api/v3"
        self.bulk_base_url = "https://data.binance.vision/data/spot/monthly/klines"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Simple state tracking
        self.state_file = os.path.join(self.data_dir, "collection_state.json")
        self.state = self._load_state()
        
        logger.info(f"🚀 Simple Binance Collector initialized")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"⏰ Interval: {self.interval}")
        logger.info(f"📁 Data directory: {self.data_dir}")
    
    def _load_state(self) -> Dict:
        """Load collection state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return {}
    
    def _save_state(self):
        """Save collection state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def get_db_path(self, symbol: str) -> str:
        """Get database path for symbol"""
        return os.path.join(self.data_dir, f"{symbol.lower()}_{self.interval}.db")
    
    def create_database(self, symbol: str):
        """Create database table for storing market data"""
        db_path = self.get_db_path(symbol)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL UNIQUE,
                datetime TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                trades INTEGER NOT NULL,
                taker_buy_base REAL NOT NULL,
                taker_buy_quote REAL NOT NULL
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data(timestamp)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Database created/verified for {symbol}")
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate that symbol exists on Binance"""
        try:
            url = f"{self.base_api_url}/exchangeInfo"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                symbols = [s['symbol'] for s in data.get('symbols', [])]
                return symbol in symbols
            else:
                logger.warning(f"Could not validate {symbol}: HTTP {response.status_code}")
                return True  # Assume valid if we can't check
                
        except Exception as e:
            logger.warning(f"Symbol validation failed for {symbol}: {e}")
            return True  # Assume valid if validation fails
    
    def download_bulk_month(self, symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """Download bulk data for a specific month"""
        filename = f"{symbol}-{self.interval}-{year:04d}-{month:02d}.zip"
        url = f"{self.bulk_base_url}/{symbol}/{self.interval}/{filename}"
        
        try:
            logger.info(f"📥 Downloading bulk: {filename}")
            
            response = self.session.get(url, timeout=60)
            
            if response.status_code == 404:
                logger.info(f"ℹ️ Bulk file not available: {symbol} {year}-{month:02d}")
                return None
            
            response.raise_for_status()
            
            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                csv_filename = filename.replace('.zip', '.csv')
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
            
            # Set column names
            df.columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
            
            # Convert timestamp (handle both ms and s)
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            if df['timestamp'].max() > 1e12:  # Already in milliseconds
                pass
            else:  # Convert from seconds to milliseconds
                df['timestamp'] = df['timestamp'] * 1000
            
            # Convert to proper datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
            df['timestamp'] = df['timestamp'].astype('int64')
            
            # Remove invalid rows
            df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Select final columns
            df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 
                    'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
            
            logger.info(f"✅ Downloaded {len(df):,} candles for {symbol} {year}-{month:02d}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to download bulk {symbol} {year}-{month:02d}: {e}")
            return None
    
    def get_api_data(self, symbol: str, start_time: int, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get data using Binance API"""
        try:
            params = {
                'symbol': symbol,
                'interval': self.interval,
                'startTime': start_time,
                'limit': limit
            }
            
            url = f"{self.base_api_url}/klines"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('int64')
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
            
            # Remove invalid rows
            df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Select final columns
            df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 
                    'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
            
            logger.info(f"✅ API fetch: {len(df):,} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ API request failed for {symbol}: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, symbol: str) -> int:
        """Save DataFrame to database, avoiding duplicates"""
        if df.empty:
            return 0
        
        db_path = self.get_db_path(symbol)
        conn = sqlite3.connect(db_path)
        
        try:
            # Get existing timestamps to avoid duplicates
            existing_timestamps = set()
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM market_data WHERE timestamp BETWEEN ? AND ?", 
                          (int(df['timestamp'].min()), int(df['timestamp'].max())))
            existing_timestamps = {row[0] for row in cursor.fetchall()}
            
            # Filter out existing data
            new_data = df[~df['timestamp'].isin(existing_timestamps)]
            
            if new_data.empty:
                logger.info(f"ℹ️ No new data to save for {symbol}")
                return 0
            
            # Prepare records for insertion
            records = []
            for _, row in new_data.iterrows():
                records.append((
                    int(row['timestamp']),
                    str(row['datetime']),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    float(row['quote_volume']),
                    int(row['trades']),
                    float(row['taker_buy_base']),
                    float(row['taker_buy_quote'])
                ))
            
            # Batch insert
            cursor.executemany("""
                INSERT OR IGNORE INTO market_data 
                (timestamp, datetime, open, high, low, close, volume, 
                 quote_volume, trades, taker_buy_base, taker_buy_quote)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            conn.commit()
            inserted_count = cursor.rowcount
            
            logger.info(f"💾 Saved {inserted_count:,} new records for {symbol}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to save data for {symbol}: {e}")
            return 0
        finally:
            conn.close()
    
    def get_last_timestamp(self, symbol: str) -> Optional[int]:
        """Get the last timestamp in the database"""
        db_path = self.get_db_path(symbol)
        
        if not os.path.exists(db_path):
            return None
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT MAX(timestamp) FROM market_data")
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"❌ Failed to get last timestamp for {symbol}: {e}")
            return None
        finally:
            conn.close()
    
    def collect_symbol_bulk(self, symbol: str, start_date: str = "2020-01-01") -> bool:
        """Collect bulk historical data for a symbol"""
        logger.info(f"📊 Starting bulk collection for {symbol}")
        
        # Validate symbol first
        if not self.validate_symbol(symbol):
            logger.error(f"❌ Invalid symbol: {symbol}")
            return False
        
        # Create database
        self.create_database(symbol)
        
        # Determine start date
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        current_dt = datetime.now()
        
        # Stop bulk collection 3 days ago (use API for recent data)
        bulk_end_dt = current_dt - timedelta(days=3)
        
        # Check state for resume capability
        state_key = f"{symbol}_bulk_progress"
        if state_key in self.state:
            resume_date = self.state[state_key]
            resume_dt = datetime.strptime(resume_date, "%Y-%m-%d")
            if resume_dt > start_dt:
                start_dt = resume_dt
                logger.info(f"🔄 Resuming bulk collection from {resume_date}")
        
        total_records = 0
        current_month_dt = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        while current_month_dt <= bulk_end_dt:
            year = current_month_dt.year
            month = current_month_dt.month
            
            # Download bulk data for this month
            df = self.download_bulk_month(symbol, year, month)
            
            if df is not None and not df.empty:
                # Save to database
                saved_count = self.save_data(df, symbol)
                total_records += saved_count
                
                # Update state
                self.state[state_key] = f"{year}-{month:02d}-01"
                self._save_state()
            
            # Rate limiting
            time.sleep(0.5)  # 500ms between requests
            
            # Move to next month
            if current_month_dt.month == 12:
                current_month_dt = current_month_dt.replace(year=current_month_dt.year + 1, month=1)
            else:
                current_month_dt = current_month_dt.replace(month=current_month_dt.month + 1)
        
        logger.info(f"✅ Bulk collection completed for {symbol}: {total_records:,} total records")
        return True
    
    def collect_symbol_recent(self, symbol: str) -> bool:
        """Collect recent data using API"""
        logger.info(f"🔄 Collecting recent data for {symbol}")
        
        # Get last timestamp from database
        last_timestamp = self.get_last_timestamp(symbol)
        
        if last_timestamp is None:
            # No data in database, start from 30 days ago
            start_dt = datetime.now() - timedelta(days=30)
            start_timestamp = int(start_dt.timestamp() * 1000)
        else:
            # Start from last timestamp + 15 minutes
            start_timestamp = last_timestamp + (15 * 60 * 1000)
        
        current_timestamp = int(datetime.now().timestamp() * 1000)
        
        # Check if we need to collect anything
        if start_timestamp >= current_timestamp:
            logger.info(f"ℹ️ {symbol} is already up to date")
            return True
        
        total_records = 0
        current_ts = start_timestamp
        
        # Collect in batches
        while current_ts < current_timestamp:
            # Get data batch
            df = self.get_api_data(symbol, current_ts, limit=1000)
            
            if df is None:
                logger.warning(f"⚠️ API request failed for {symbol}, skipping batch")
                current_ts += (1000 * 15 * 60 * 1000)  # Skip 1000 periods
                continue
            
            if df.empty:
                logger.info(f"ℹ️ No more data available for {symbol}")
                break
            
            # Save data
            saved_count = self.save_data(df, symbol)
            total_records += saved_count
            
            # Update current timestamp
            current_ts = int(df['timestamp'].max()) + (15 * 60 * 1000)
            
            # Rate limiting
            time.sleep(0.1)  # 100ms between API requests
        
        logger.info(f"✅ Recent collection completed for {symbol}: {total_records:,} new records")
        return True
    
    def collect_symbol(self, symbol: str) -> bool:
        """Collect all data for a symbol (bulk + recent)"""
        logger.info(f"🚀 Starting full collection for {symbol}")
        
        try:
            # Phase 1: Bulk historical data
            if not self.collect_symbol_bulk(symbol):
                logger.error(f"❌ Bulk collection failed for {symbol}")
                return False
            
            # Phase 2: Recent data via API  
            if not self.collect_symbol_recent(symbol):
                logger.error(f"❌ Recent collection failed for {symbol}")
                return False
            
            logger.info(f"✅ Full collection completed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Collection failed for {symbol}: {e}")
            return False
    
    def collect_all(self) -> Dict[str, bool]:
        """Collect data for all symbols"""
        logger.info(f"🚀 Starting collection for {len(self.symbols)} symbols")
        
        results = {}
        
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Symbol {i}/{len(self.symbols)}: {symbol}")
            logger.info(f"{'='*50}")
            
            try:
                results[symbol] = self.collect_symbol(symbol)
                
                if results[symbol]:
                    logger.info(f"✅ {symbol} collection successful")
                else:
                    logger.error(f"❌ {symbol} collection failed")
                    
            except Exception as e:
                logger.error(f"❌ {symbol} collection error: {e}")
                results[symbol] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        logger.info(f"\n🎉 Collection Summary:")
        logger.info(f"✅ Successful: {successful}/{len(self.symbols)} symbols")
        
        for symbol, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {symbol}")
        
        return results
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """Get summary of collected data"""
        summary = {}
        
        for symbol in self.symbols:
            db_path = self.get_db_path(symbol)
            
            if not os.path.exists(db_path):
                summary[symbol] = {"status": "No database file"}
                continue
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(datetime) as start_date,
                        MAX(datetime) as end_date,
                        MIN(close) as min_price,
                        MAX(close) as max_price,
                        AVG(volume) as avg_volume
                    FROM market_data
                """)
                
                row = cursor.fetchone()
                if row and row[0] > 0:
                    summary[symbol] = {
                        "total_records": row[0],
                        "start_date": row[1],
                        "end_date": row[2],
                        "min_price": row[3],
                        "max_price": row[4],
                        "avg_volume": row[5],
                        "database_size_mb": os.path.getsize(db_path) / (1024 * 1024)
                    }
                else:
                    summary[symbol] = {"status": "No data in database"}
                    
            except Exception as e:
                summary[symbol] = {"status": f"Error: {e}"}
            finally:
                conn.close()
        
        return summary

def main():
    """Main function"""
    print("🚀 Simple Binance Data Collector")
    print("================================")
    print("Collecting 15-minute data for EUR cryptocurrency pairs")
    print()
    
    # Initialize collector (databases will be saved in same directory as script)
    collector = SimpleBinanceCollector()
    
    try:
        # Collect all data
        results = collector.collect_all()
        
        # Show summary
        print("\n📊 Data Summary:")
        print("================")
        
        summary = collector.get_data_summary()
        
        for symbol, stats in summary.items():
            print(f"\n{symbol}:")
            if "status" in stats:
                print(f"  Status: {stats['status']}")
            else:
                print(f"  Records: {stats['total_records']:,}")
                print(f"  Date range: {stats['start_date']} to {stats['end_date']}")
                print(f"  Price range: €{stats['min_price']:.4f} - €{stats['max_price']:.4f}")
                print(f"  Avg volume: {stats['avg_volume']:,.2f}")
                print(f"  Database size: {stats['database_size_mb']:.1f} MB")
        
        print(f"\n✅ Collection completed!")
        print(f"📁 Databases saved in: {collector.data_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        raise

if __name__ == "__main__":
    main()