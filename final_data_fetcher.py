#!/usr/bin/env python3
"""
Final Data Fetcher - Ultimate 1-Year Database Builder
=====================================================

This is the definitive script for building 1-year trading databases.
It combines the best approaches from all previous fetchers:

- Uses your original collector's bulk + API strategy
- Incorporates the improved chunking methods
- Handles 365 days of 30-minute candles (17,520 samples expected)
- Built-in retry logic and fallbacks
- Single script to replace all others

Usage:
    python final_data_fetcher.py                    # Fetch all symbols + push to GitHub
    python final_data_fetcher.py BTCEUR ETHEUR     # Fetch specific symbols + push to GitHub
    python final_data_fetcher.py --dry-run         # Test without saving
    python final_data_fetcher.py --no-push         # Fetch without pushing to GitHub
"""

import argparse
import io
import json
import logging
import os
import sqlite3
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FinalDataFetcher:
    """Ultimate data fetcher for 1-year trading databases"""

    def __init__(self):
        self.load_config()
        self.setup_directories()

    def load_config(self):
        """Load configuration from training_config.yaml"""
        config_path = "training_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self.data_config = config.get("data_acquisition", {})
        else:
            logger.warning("training_config.yaml not found, using defaults")
            self.data_config = {}

        # Configuration with proper 1-year defaults
        self.symbols = self.data_config.get(
            "symbols", ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]
        )
        self.interval = self.data_config.get("interval", "30m")
        self.lookback_days = 365  # Always 1 year
        self.data_dir = Path(self.data_config.get("output_directory", "./data"))

        # Calculate expected samples for validation
        if self.interval == "30m":
            self.expected_samples = 365 * 48  # 17,520
        elif self.interval == "1h":
            self.expected_samples = 365 * 24  # 8,760
        elif self.interval == "1d":
            self.expected_samples = 365  # 365
        else:
            self.expected_samples = 10000  # Conservative estimate

        logger.info(
            f"Config: {len(self.symbols)} symbols, {self.interval} interval, {self.lookback_days} days"
        )
        logger.info(f"Expected samples per symbol: {self.expected_samples:,}")

    def setup_directories(self):
        """Setup data directory"""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch 1 year of data using best available method"""

        logger.info(f"📊 Fetching {symbol} data ({self.interval}, {self.lookback_days} days)")

        # Method priority based on testing results
        methods = [
            ("binance_chunked", self._fetch_binance_chunked),
            ("bulk_historical", self._fetch_bulk_historical),
            ("yfinance_standard", self._fetch_yfinance_standard),
            ("yfinance_chunked", self._fetch_yfinance_chunked),
            ("coinbase", self._fetch_coinbase),
            ("kraken", self._fetch_kraken),
        ]

        for method_name, method in methods:
            try:
                logger.info(f"  🔄 Trying {method_name}...")
                data = method(symbol)

                if data is not None and len(data) > 100:
                    coverage = len(data) / self.expected_samples * 100
                    logger.info(
                        f"  ✅ {method_name}: {len(data):,} samples ({coverage:.1f}% coverage)"
                    )

                    if len(data) >= self.expected_samples * 0.8:  # 80% coverage is excellent
                        logger.info(f"  🎉 Excellent coverage with {method_name}!")
                        return data
                    elif len(data) >= self.expected_samples * 0.5:  # 50% coverage is good
                        logger.info(f"  ✅ Good coverage with {method_name}")
                        return data
                    elif len(data) >= self.expected_samples * 0.2:  # 20% coverage is acceptable
                        logger.info(f"  ⚠️  Acceptable coverage with {method_name}")
                        return data

            except Exception as e:
                logger.warning(f"  ❌ {method_name} failed: {e}")

        logger.error(f"❌ All methods failed for {symbol}")
        return None

    def _fetch_binance_chunked(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from Binance API in chunks (best method - gets 17,520 samples)"""
        try:
            url = "https://api.binance.com/api/v3/klines"

            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            # Chunk sizes based on API limits
            if self.interval == "30m":
                chunk_days = 20  # 1000 candles ≈ 20.8 days
            elif self.interval == "1h":
                chunk_days = 40  # 1000 candles ≈ 41.6 days
            else:
                chunk_days = 1000

            chunks = []
            current_end = end_date

            while current_end > start_date:
                current_start = max(current_end - timedelta(days=chunk_days), start_date)

                params = {
                    "symbol": symbol,
                    "interval": self.interval,
                    "startTime": int(current_start.timestamp() * 1000),
                    "endTime": int(current_end.timestamp() * 1000),
                    "limit": 1000,
                }

                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json()

                    if len(data) > 0:
                        df = pd.DataFrame(
                            data,
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "close_time",
                                "quote_asset_volume",
                                "number_of_trades",
                                "taker_buy_base_asset_volume",
                                "taker_buy_quote_asset_volume",
                                "ignore",
                            ],
                        )

                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        for col in ["open", "high", "low", "close", "volume"]:
                            df[col] = pd.to_numeric(df[col])

                        chunks.append(df)

                current_end = current_start
                time.sleep(0.2)  # Rate limiting

            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

                # Format consistently
                df = df.rename(
                    columns={
                        "timestamp": "Datetime",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )

                df.set_index("Datetime", inplace=True)
                return df[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.debug(f"Binance chunked failed: {e}")

        return None

    def _fetch_bulk_historical(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch bulk historical data from Binance data download"""
        try:
            base_url = "https://data.binance.vision/data/spot/monthly/klines"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            all_data = []
            current_date = start_date.replace(day=1)

            while current_date <= end_date:
                year = current_date.year
                month = current_date.month

                filename = f"{symbol}-{self.interval}-{year:04d}-{month:02d}.zip"
                url = f"{base_url}/{symbol}/{self.interval}/{filename}"

                try:
                    response = requests.get(url, timeout=30)

                    if response.status_code == 200:
                        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                            csv_filename = filename.replace(".zip", ".csv")
                            with zip_file.open(csv_filename) as csv_file:
                                df = pd.read_csv(
                                    csv_file,
                                    header=None,
                                    names=[
                                        "timestamp",
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                        "close_time",
                                        "quote_asset_volume",
                                        "number_of_trades",
                                        "taker_buy_base_asset_volume",
                                        "taker_buy_quote_asset_volume",
                                        "ignore",
                                    ],
                                )

                                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                                all_data.append(df)

                except Exception:
                    pass  # Skip failed months

                # Next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)

                time.sleep(0.1)

            if all_data:
                df = pd.concat(all_data, ignore_index=True)

                # Filter to requested date range
                start_ts = pd.to_datetime(start_date)
                end_ts = pd.to_datetime(end_date)
                df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

                df = df.rename(
                    columns={
                        "timestamp": "Datetime",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )

                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col])

                df.set_index("Datetime", inplace=True)
                df = df.sort_index()

                return df[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.debug(f"Bulk historical failed: {e}")

        return None

    def _fetch_yfinance_standard(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from YFinance (convert symbol format)"""
        try:
            import yfinance as yf

            # Convert symbol format
            if symbol.endswith("EUR"):
                yf_symbol = symbol.replace("EUR", "-EUR")
            else:
                yf_symbol = symbol

            ticker = yf.Ticker(yf_symbol)

            hist = ticker.history(period="1y", interval=self.interval)
            if len(hist) > 100:
                return hist[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.debug(f"YFinance standard failed: {e}")

        return None

    def _fetch_yfinance_chunked(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from YFinance in chunks"""
        try:
            import yfinance as yf

            if symbol.endswith("EUR"):
                yf_symbol = symbol.replace("EUR", "-EUR")
            else:
                yf_symbol = symbol

            ticker = yf.Ticker(yf_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            chunks = []
            chunk_days = 180
            current_start = start_date

            while current_start < end_date:
                current_end = min(current_start + timedelta(days=chunk_days), end_date)

                try:
                    hist = ticker.history(
                        start=current_start.strftime("%Y-%m-%d"),
                        end=current_end.strftime("%Y-%m-%d"),
                        interval=self.interval,
                    )
                    if len(hist) > 0:
                        chunks.append(hist)
                except Exception:
                    pass

                current_start = current_end
                time.sleep(1)

            if chunks:
                df = pd.concat(chunks)
                df = df.sort_index()
                return df[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.debug(f"YFinance chunked failed: {e}")

        return None

    def _fetch_coinbase(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from Coinbase (placeholder - implement if needed)"""
        return None

    def _fetch_kraken(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from Kraken (placeholder - implement if needed)"""
        return None

    def create_database(self, symbol: str, data: pd.DataFrame, dry_run: bool = False) -> int:
        """Create SQLite database for symbol"""

        if dry_run:
            logger.info(f"  🔍 DRY RUN: Would create database with {len(data)} rows")
            return len(data)

        db_path = self.data_dir / f"{symbol.lower()}_{self.interval}.db"

        # Backup existing database
        if db_path.exists():
            backup_path = (
                self.data_dir
                / f"backups/{symbol.lower()}_{self.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            )
            backup_path.parent.mkdir(exist_ok=True)
            import shutil

            shutil.copy2(db_path, backup_path)
            logger.info(f"  💾 Backed up existing database to {backup_path}")

        # Create new database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table
        # Drop existing table if it exists to ensure clean schema
        cursor.execute("DROP TABLE IF EXISTS market_data")

        cursor.execute(
            """
            CREATE TABLE market_data (
                datetime TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Insert data
        rows = []
        for idx, row in data.iterrows():
            rows.append(
                (
                    idx.isoformat(),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    float(row["Volume"]),
                )
            )

        cursor.executemany(
            "INSERT OR REPLACE INTO market_data (datetime, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

        conn.commit()
        conn.close()

        logger.info(f"  💾 Created database: {db_path} ({len(rows)} rows)")
        return len(rows)

    def _push_to_github(self, results: Dict[str, Any]) -> None:
        """Push updated databases to GitHub (like the original /database command)"""
        import os
        import subprocess

        logger.info(f"\n🔄 Pushing updated databases to GitHub...")

        try:
            # Change to the correct directory
            os.chdir(self.data_dir.parent)

            # Git pull first
            logger.info("📥 Pulling latest changes...")
            subprocess.run(["git", "pull"], check=True, capture_output=True)

            # Add database files
            successful_symbols = results["success"]
            for symbol in successful_symbols:
                db_file = f"data/{symbol.lower()}_{self.interval}.db"
                subprocess.run(["git", "add", db_file], check=True, capture_output=True)
                logger.info(f"  ✅ Added {db_file}")

            # Check if there are changes to commit
            result = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True
            )
            if result.stdout.strip():
                # Create commit message
                symbol_list = ", ".join(successful_symbols)
                commit_msg = f"Refresh databases via final_data_fetcher: {symbol_list}\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"

                # Commit changes
                logger.info("💾 Committing changes...")
                subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)

                # Push to GitHub
                logger.info("🚀 Pushing to GitHub...")
                subprocess.run(["git", "push"], check=True, capture_output=True)

                logger.info(
                    f"✅ Successfully pushed {len(successful_symbols)} database(s) to GitHub"
                )
            else:
                logger.info("ℹ️  No changes to commit - databases already up to date")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git operation failed: {e}")
            # Try to get more details from stderr
            if e.stderr:
                logger.error(f"Git error details: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"❌ GitHub push failed: {e}")
            raise

    def process_symbols(
        self, symbols: Optional[List[str]] = None, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Process all symbols and create databases"""

        if symbols is None:
            symbols = self.symbols

        results = {"success": [], "failed": [], "stats": {}}
        total_samples = 0

        logger.info(f"🚀 Starting data collection for {len(symbols)} symbols")
        logger.info(
            f"📊 Target: {self.expected_samples:,} samples per symbol ({self.interval} candles)"
        )

        for symbol in symbols:
            try:
                logger.info(f"\n📈 Processing {symbol}...")

                data = self.fetch_symbol_data(symbol)

                if data is not None:
                    rows = self.create_database(symbol, data, dry_run)
                    total_samples += rows

                    coverage = rows / self.expected_samples * 100
                    results["success"].append(symbol)
                    results["stats"][symbol] = {
                        "rows": rows,
                        "coverage": coverage,
                        "date_range": f"{data.index[0]} to {data.index[-1]}",
                    }

                    if coverage >= 80:
                        logger.info(f"  🎉 {symbol}: EXCELLENT ({rows:,} samples, {coverage:.1f}%)")
                    elif coverage >= 50:
                        logger.info(f"  ✅ {symbol}: GOOD ({rows:,} samples, {coverage:.1f}%)")
                    elif coverage >= 20:
                        logger.info(
                            f"  ⚠️  {symbol}: ACCEPTABLE ({rows:,} samples, {coverage:.1f}%)"
                        )
                    else:
                        logger.info(f"  ❌ {symbol}: POOR ({rows:,} samples, {coverage:.1f}%)")

                else:
                    results["failed"].append(symbol)
                    logger.error(f"  ❌ {symbol}: FAILED")

            except Exception as e:
                results["failed"].append(symbol)
                logger.error(f"  ❌ {symbol}: ERROR - {e}")

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successful: {len(results['success'])}")
        logger.info(f"❌ Failed: {len(results['failed'])}")
        logger.info(f"📈 Total samples: {total_samples:,}")
        logger.info(f"🎯 Expected total: {self.expected_samples * len(symbols):,}")
        logger.info(
            f"📊 Overall coverage: {total_samples / (self.expected_samples * len(symbols)) * 100:.1f}%"
        )

        if not dry_run:
            logger.info(f"💾 Databases saved in: {self.data_dir}")

            # Push to GitHub (like the original system)
            if results["success"] and not getattr(self, "no_push", False):
                try:
                    self._push_to_github(results)
                except Exception as e:
                    logger.warning(f"⚠️ GitHub push failed: {e}")
            elif getattr(self, "no_push", False):
                logger.info("⏭️  Skipping GitHub push (--no-push flag set)")

        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Final Data Fetcher - Build 1-year trading databases"
    )
    parser.add_argument(
        "symbols", nargs="*", help="Specific symbols to fetch (default: all from config)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Test run without saving databases")
    parser.add_argument("--interval", default="30m", help="Time interval (default: 30m)")
    parser.add_argument(
        "--no-push", action="store_true", help="Skip GitHub push after updating databases"
    )

    args = parser.parse_args()

    fetcher = FinalDataFetcher()

    if args.interval != "30m":
        fetcher.interval = args.interval

    # Set the no_push flag
    fetcher.no_push = args.no_push

    symbols = args.symbols if args.symbols else None

    try:
        results = fetcher.process_symbols(symbols, dry_run=args.dry_run)

        if results["failed"]:
            sys.exit(1)  # Exit with error if any symbols failed
        else:
            sys.exit(0)  # Success

    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
