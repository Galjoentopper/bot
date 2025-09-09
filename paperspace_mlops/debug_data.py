#!/usr/bin/env python3
"""
Debug Data Issues
================

Test data fetching for all 5 symbols to identify why datasets aren't being built.
"""

import logging
import os
import sys
from pathlib import Path

# Add bot path for imports
bot_path = "/notebooks/bot" if os.environ.get("PAPERSPACE_JOB_ID") else str(Path(__file__).parent.parent)
if bot_path not in sys.path:
    sys.path.insert(0, bot_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_data_sources():
    """Test different data sources for all symbols"""
    
    symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]
    
    logger.info("🔍 Testing data sources...")
    
    # Test 1: Direct yfinance
    logger.info("📊 Testing yfinance...")
    try:
        import yfinance as yf
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="30d", interval="1h")
                logger.info(f"✅ yfinance {symbol}: {len(hist)} samples")
            except Exception as e:
                logger.error(f"❌ yfinance {symbol}: {e}")
    except ImportError:
        logger.error("❌ yfinance not available")
    
    # Test 2: Binance (if available)
    logger.info("📊 Testing binance...")
    try:
        import requests
        for symbol in symbols:
            try:
                # Try Binance API
                url = f"https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": symbol.replace("EUR", "USDT"),  # Convert to USDT pairs
                    "interval": "1h",
                    "limit": 100
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ binance {symbol}USDT: {len(data)} samples")
                else:
                    logger.warning(f"⚠️ binance {symbol}USDT: HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"❌ binance {symbol}: {e}")
    except ImportError:
        logger.error("❌ requests not available")


def test_dataset_builder():
    """Test DatasetBuilder directly"""
    
    logger.info("🏗️ Testing DatasetBuilder...")
    
    try:
        from src.data_pipeline.dataset_builder import DatasetBuilder
        
        # Create builder
        builder = DatasetBuilder(
            data_dir="./data",
            cache_dir="./data/cache",
            config={}
        )
        
        # Test with each symbol
        symbols = ["BTCEUR", "ETHEUR"]  # Start with just 2
        
        for symbol in symbols:
            try:
                logger.info(f"🔨 Building dataset for {symbol}...")
                
                dataset = builder.build_dataset(
                    symbol=symbol,
                    interval="1h",  # Try hourly first
                    use_cache=False  # Force fresh data
                )
                
                if dataset:
                    if isinstance(dataset, tuple) and len(dataset) >= 2:
                        X, y = dataset[0], dataset[1]
                        logger.info(f"✅ {symbol}: {len(X)} samples, {X.shape[1] if hasattr(X, 'shape') else 'unknown'} features")
                    else:
                        logger.warning(f"⚠️ {symbol}: Unexpected format: {type(dataset)}")
                else:
                    logger.error(f"❌ {symbol}: No dataset returned")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
    except ImportError as e:
        logger.error(f"❌ DatasetBuilder import failed: {e}")


def test_data_loader():
    """Test DataLoader directly"""
    
    logger.info("📥 Testing DataLoader...")
    
    try:
        from src.data_pipeline.loader import DataLoader
        
        loader = DataLoader()
        symbols = ["BTCEUR", "ETHEUR"]
        
        for symbol in symbols:
            try:
                logger.info(f"📊 Loading data for {symbol}...")
                
                # Try different intervals
                for interval in ["1h", "30m"]:
                    try:
                        data = loader.fetch_data(
                            symbol=symbol,
                            interval=interval,
                            lookback_days=30
                        )
                        
                        if data is not None and len(data) > 0:
                            logger.info(f"✅ {symbol} {interval}: {len(data)} samples")
                            break
                        else:
                            logger.warning(f"⚠️ {symbol} {interval}: No data")
                    except Exception as e:
                        logger.error(f"❌ {symbol} {interval}: {e}")
                        
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
                
    except ImportError as e:
        logger.error(f"❌ DataLoader import failed: {e}")


def main():
    """Run all tests"""
    
    logger.info("🚀 Data Debugging Session")
    logger.info("=" * 50)
    
    test_data_sources()
    logger.info("-" * 30)
    test_data_loader() 
    logger.info("-" * 30)
    test_dataset_builder()
    
    logger.info("✅ Debug session complete!")


if __name__ == "__main__":
    main()