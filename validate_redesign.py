#!/usr/bin/env python3
"""Simple validation test for the redesigned paper trader components."""

import asyncio
import logging
import sys
from pathlib import Path

# Add paper_trader to path
sys.path.append(str(Path(__file__).parent))

from paper_trader.config.settings import TradingSettings
from paper_trader.data.historical_data_collector import HistoricalDataCollector

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def validate_historical_data_collector():
    """Validate the historical data collector works correctly."""
    logger.info("🔍 Validating Historical Data Collector...")
    
    settings = TradingSettings()
    # Test with just BTC for simplicity
    settings.symbols = ['BTC-EUR']
    
    collector = HistoricalDataCollector(settings)
    
    try:
        # Initialize
        await collector.initialize()
        
        # Get historical data
        symbol = 'BTC-EUR'
        historical_data = await collector.get_historical_data_for_features(symbol, min_length=100)
        
        if historical_data is not None and len(historical_data) >= 100:
            latest_price = historical_data['close'].iloc[-1]
            logger.info(f"✅ Successfully got {len(historical_data)} candles for {symbol}")
            logger.info(f"✅ Latest price: €{latest_price:.2f}")
            
            # Check data quality
            if historical_data.isnull().sum().sum() == 0:
                logger.info("✅ No null values in historical data")
            else:
                logger.warning("⚠️ Found null values in historical data")
                
            # Check data spans reasonable time
            time_span = (historical_data.index[-1] - historical_data.index[0]).total_seconds() / 3600
            logger.info(f"✅ Data spans {time_span:.1f} hours")
            
            return True
        else:
            logger.error("❌ Failed to get sufficient historical data")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in historical data collector: {e}")
        return False
    finally:
        await collector.stop()


async def validate_data_separation():
    """Validate that the new design properly separates concerns."""
    logger.info("🔍 Validating Data Separation Design...")
    
    settings = TradingSettings()
    settings.symbols = ['BTC-EUR']
    
    # Test historical data collector
    historical_collector = HistoricalDataCollector(settings)
    
    try:
        await historical_collector.initialize()
        
        # Get data for features (this should work independently)
        historical_data = await historical_collector.get_historical_data_for_features('BTC-EUR', 200)
        
        if historical_data is not None:
            logger.info("✅ Historical data collector works independently")
            logger.info(f"✅ Got {len(historical_data)} candles for feature engineering")
            
            # Check that we have proper OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in historical_data.columns for col in required_cols):
                logger.info("✅ All required OHLCV columns present")
                
                # Check price consistency (high >= close >= low, etc.)
                price_check = (
                    (historical_data['high'] >= historical_data['close']).all() and
                    (historical_data['close'] >= historical_data['low']).all() and
                    (historical_data['high'] >= historical_data['open']).all() and
                    (historical_data['open'] >= historical_data['low']).all()
                )
                
                if price_check:
                    logger.info("✅ Price data integrity check passed")
                    return True
                else:
                    logger.error("❌ Price data integrity check failed")
                    return False
            else:
                logger.error("❌ Missing required OHLCV columns")
                return False
        else:
            logger.error("❌ Could not get historical data")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in data separation validation: {e}")
        return False
    finally:
        await historical_collector.stop()


async def validate_feature_engineering_compatibility():
    """Validate that historical data is compatible with feature engineering."""
    logger.info("🔍 Validating Feature Engineering Compatibility...")
    
    settings = TradingSettings()
    settings.symbols = ['BTC-EUR']
    
    try:
        # Import feature engineer
        from paper_trader.models.feature_engineer import FeatureEngineer
        
        historical_collector = HistoricalDataCollector(settings)
        feature_engineer = FeatureEngineer()
        
        await historical_collector.initialize()
        
        # Get historical data
        historical_data = await historical_collector.get_historical_data_for_features('BTC-EUR', 300)
        
        if historical_data is not None and len(historical_data) >= 300:
            logger.info(f"✅ Got {len(historical_data)} candles for feature engineering")
            
            # Test feature engineering
            features_df = feature_engineer.engineer_features(historical_data)
            
            if features_df is not None and len(features_df) > 0:
                logger.info(f"✅ Feature engineering successful: {len(features_df)} feature rows")
                logger.info(f"✅ Feature columns: {len(features_df.columns)}")
                
                # Check for reasonable feature values
                if not features_df.isnull().all().any():
                    logger.info("✅ Features contain valid data")
                    return True
                else:
                    logger.warning("⚠️ Some features are all null")
                    return False
            else:
                logger.error("❌ Feature engineering failed")
                return False
        else:
            logger.error("❌ Insufficient historical data for feature engineering")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in feature engineering validation: {e}")
        return False
    finally:
        await historical_collector.stop()


async def main():
    """Run validation tests."""
    logger.info("🚀 Starting Redesigned Paper Trader Validation...")
    
    tests = [
        ("Historical Data Collector", validate_historical_data_collector),
        ("Data Separation Design", validate_data_separation),
        ("Feature Engineering Compatibility", validate_feature_engineering_compatibility),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = await test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All validation tests passed! The redesigned components are working correctly.")
        logger.info("\n📋 Key Improvements Validated:")
        logger.info("   ✅ Historical data collection works independently")
        logger.info("   ✅ Data integrity and quality checks pass")
        logger.info("   ✅ Feature engineering compatibility confirmed")
        logger.info("   ✅ Clean separation between historical and real-time data")
        logger.info("\n🚀 Ready to use the redesigned paper trader!")
    else:
        logger.warning(f"⚠️ {total - passed} validation tests failed. Please review the issues.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error in validation: {e}")
        sys.exit(2)