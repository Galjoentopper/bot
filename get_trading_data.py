#!/usr/bin/env python3
"""
Get Trading Data for Superior PPO Training
==========================================

This script fetches real BTCEUR trading data for training the superior PPO model.
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def fetch_crypto_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch cryptocurrency data from Yahoo Finance."""
    logger.info(f"📊 Fetching {symbol} data for {days} days...")

    # Convert crypto symbol to Yahoo Finance format
    if symbol == "BTCEUR":
        yf_symbol = "BTC-EUR"
    elif symbol == "ETHEUR":
        yf_symbol = "ETH-EUR"
    elif symbol == "ADAEUR":
        yf_symbol = "ADA-EUR"
    elif symbol == "DOTEUR":
        yf_symbol = "DOT-EUR"
    elif symbol == "LINKEUR":
        yf_symbol = "LINK-EUR"
    else:
        yf_symbol = symbol

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        # Fetch data with 30-minute intervals
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval="30m",  # 30-minute intervals for multi-timeframe analysis
            auto_adjust=True,
            prepost=True,
        )

        if data.empty:
            logger.warning(f"No data found for {yf_symbol}")
            return None

        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]

        # Reset index to have timestamp as column
        data = data.reset_index()
        data.rename(columns={"datetime": "timestamp"}, inplace=True)

        # Keep only OHLCV columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        available_cols = [col for col in required_cols if col in data.columns]
        data = data[available_cols]

        # Remove any rows with NaN values
        data = data.dropna()

        logger.info(f"✅ Fetched {len(data)} rows for {symbol}")
        logger.info(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        logger.info(f"   Price range: €{data['close'].min():.2f} - €{data['close'].max():.2f}")

        return data

    except Exception as e:
        logger.error(f"❌ Failed to fetch data for {yf_symbol}: {e}")
        return None


def save_data(data: pd.DataFrame, symbol: str, output_dir: str = "data"):
    """Save data to parquet format."""
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f"{symbol}_30m.parquet")
    data.to_parquet(filepath, index=False)

    logger.info(f"💾 Saved data to {filepath}")
    return filepath


def main():
    """Main function to fetch and save trading data."""
    symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]

    logger.info("🚀 Starting trading data collection...")

    for symbol in symbols:
        logger.info(f"\n📈 Processing {symbol}...")

        # Fetch data
        data = fetch_crypto_data(symbol, days=365)

        if data is not None and len(data) > 0:
            # Save data
            filepath = save_data(data, symbol)

            # Show summary
            logger.info(f"✅ {symbol} data ready:")
            logger.info(f"   File: {filepath}")
            logger.info(f"   Rows: {len(data):,}")
            logger.info(f"   Columns: {list(data.columns)}")

        else:
            logger.warning(f"⚠️  Skipping {symbol} - no data available")

    logger.info("\n🎉 Data collection complete!")
    logger.info("\nNext steps:")
    logger.info("1. python run_superior_training.py --symbol BTCEUR --timesteps 200000")
    logger.info("2. Watch the superior model train without OOM kills!")


if __name__ == "__main__":
    main()
