#!/usr/bin/env python3
"""
Test script to verify bot fixes for trading issues and XRP-EUR model loading.
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add paper_trader to path
sys.path.append(str(Path(__file__).parent))

def test_xrp_eur_configuration():
    """Test that XRP-EUR is properly configured in environment."""
    print("Testing XRP-EUR configuration...")
    
    # Check .env file
    with open('.env', 'r') as f:
        env_content = f.read()
    
    if 'XRP-EUR' in env_content and 'SYMBOLS=BTC-EUR,ETH-EUR,ADA-EUR,SOL-EUR,XRP-EUR' in env_content:
        print("✅ XRP-EUR is properly configured in SYMBOLS")
        return True
    else:
        print("❌ XRP-EUR is missing from SYMBOLS configuration")
        return False

def test_xrp_eur_models():
    """Test that XRP-EUR models exist."""
    print("Testing XRP-EUR model files...")
    
    model_types = {
        'LSTM': 'models/lstm/xrpeur_window_10.keras',
        'XGBoost': 'models/xgboost/xrpeur_window_10.json',
        'Scaler': 'models/scalers/xrpeur_window_10_scaler.pkl',
        'Features': 'models/feature_columns/xrpeur_window_10.pkl'
    }
    
    all_exist = True
    for model_type, path in model_types.items():
        if os.path.exists(path):
            print(f"✅ {model_type} model exists: {path}")
        else:
            print(f"❌ {model_type} model missing: {path}")
            all_exist = False
    
    return all_exist

def test_local_database():
    """Test that local SQLite database exists and has data."""
    print("Testing local database fallback...")
    
    db_path = 'data/xrpeur_15m.db'
    if not os.path.exists(db_path):
        print(f"❌ Local database missing: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if market_data table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
        if not cursor.fetchone():
            print("❌ market_data table not found in database")
            return False
        
        # Check row count
        cursor.execute("SELECT COUNT(*) FROM market_data")
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"✅ Local database exists with {count} records")
            
            # Check recent data
            cursor.execute("SELECT datetime, close FROM market_data ORDER BY timestamp DESC LIMIT 1")
            latest = cursor.fetchone()
            if latest:
                print(f"✅ Latest data point: {latest[0]} at price {latest[1]}")
            
            conn.close()
            return True
        else:
            print("❌ Local database is empty")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return False

def test_database_fallback_functionality():
    """Test the database fallback functionality."""
    print("Testing database fallback functionality...")
    
    try:
        from paper_trader.data.bitvavo_collector import BitvavoDataCollector
        from paper_trader.config.settings import TradingSettings
        
        settings = TradingSettings()
        # Use dummy credentials for testing
        collector = BitvavoDataCollector('dummy_key', 'dummy_secret', settings)
        
        # Test the new fallback method
        import asyncio
        
        async def test_fallback():
            data = await collector._load_from_local_database('XRP-EUR', 10)
            if data is not None and not data.empty:
                print(f"✅ Database fallback working: loaded {len(data)} records")
                print(f"✅ Data columns: {list(data.columns)}")
                print(f"✅ Latest price: {data['close'].iloc[-1]}")
                return True
            else:
                print("❌ Database fallback returned no data")
                return False
        
        result = asyncio.run(test_fallback())
        return result
        
    except Exception as e:
        print(f"❌ Error testing database fallback: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING BOT FIXES FOR TRADING AND XRP-EUR ISSUES")
    print("=" * 60)
    
    tests = [
        test_xrp_eur_configuration,
        test_xrp_eur_models,
        test_local_database,
        test_database_fallback_functionality
    ]
    
    results = []
    for test in tests:
        print()
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = all(results)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ XRP-EUR configuration is correct")
        print("✅ XRP-EUR models are available")
        print("✅ Local database fallback is working")
        print("✅ Bot should now trade with all symbols including XRP-EUR")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)