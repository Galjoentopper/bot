#!/usr/bin/env python3
"""
Load Training Data from SQLite Databases
========================================

This script loads real trading data from the SQLite databases in the data folder.
"""

import logging
import os
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data_from_db(symbol: str, data_dir: str = "data") -> pd.DataFrame:
    """Load trading data from SQLite database."""
    db_path = os.path.join(data_dir, f"{symbol.lower()}_30m.db")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info(f"📊 Loading {symbol} data from {db_path}")

    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)

        # Try to find the correct table name
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            raise ValueError(f"No tables found in {db_path}")

        table_name = tables[0][0]  # Use first table
        logger.info(f"   Using table: {table_name}")

        # Load data
        query = f"SELECT * FROM {table_name} ORDER BY timestamp"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Check columns
        logger.info(f"   Columns: {list(df.columns)}")

        # Handle duplicate timestamp columns - keep only required ones
        required_cols = ["open", "high", "low", "close", "volume"]

        # Add timestamp column (prefer datetime over timestamp if both exist)
        if "datetime" in df.columns:
            df["timestamp"] = df["datetime"]
        elif "timestamp" in df.columns:
            # Keep existing timestamp
            pass

        # Select only the columns we need
        available_cols = ["timestamp"] + [col for col in required_cols if col in df.columns]
        df = df[available_cols]

        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp if needed
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except:
                pass  # Keep as is if conversion fails

        # Remove any NaN values
        df = df.dropna()

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"✅ Loaded {len(df):,} rows for {symbol}")
        if len(df) > 0:
            logger.info(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
            if "timestamp" in df.columns:
                logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    except Exception as e:
        logger.error(f"❌ Failed to load data for {symbol}: {e}")
        raise


def prepare_training_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split data into training and evaluation sets."""
    logger.info(f"🔄 Preparing training data for {symbol}")

    # Load data
    df = load_data_from_db(symbol)

    if len(df) < 1000:
        raise ValueError(f"Insufficient data for {symbol}: {len(df)} rows")

    # Split into train/eval (80/20)
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx].copy()
    eval_data = df.iloc[split_idx:].copy()

    logger.info(f"📊 Data split for {symbol}:")
    logger.info(f"   Training: {len(train_data):,} rows")
    logger.info(f"   Evaluation: {len(eval_data):,} rows")

    return train_data, eval_data


def test_data_loading():
    """Test loading data for all available symbols."""
    data_dir = "data"

    # Find all database files
    db_files = [f for f in os.listdir(data_dir) if f.endswith("_30m.db")]
    symbols = [f.replace("_30m.db", "").upper() for f in db_files]

    logger.info(f"🔍 Found databases for symbols: {symbols}")

    for symbol in symbols:
        try:
            df = load_data_from_db(symbol)
            logger.info(f"✅ {symbol}: {len(df):,} rows available")
        except Exception as e:
            logger.error(f"❌ {symbol}: Failed to load - {e}")

    return symbols


if __name__ == "__main__":
    # Test data loading
    available_symbols = test_data_loading()

    logger.info(f"\n🎯 Available symbols for training: {available_symbols}")
    logger.info(f"\nTo train with real data:")
    logger.info(f"python run_superior_training.py --symbol BTCEUR --timesteps 200000")
