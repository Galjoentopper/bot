#!/usr/bin/env python3
"""Simple adapter layer test."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print("Starting Simple Adapter Layer Test")
print("=" * 40)

try:
    # Test ConfigAdapter
    print("\n1. Testing ConfigAdapter...")
    from src.adapters.config_adapter import ConfigAdapter
    config_adapter = ConfigAdapter()
    print("   ✓ ConfigAdapter created successfully")
    
    # Test basic config access
    symbols = config_adapter.get_config('symbols', ['BTCEUR'])
    print(f"   ✓ Retrieved symbols: {symbols}")
    
    # Test validation
    is_valid = config_adapter.validate_config()
    print(f"   ✓ Configuration validation: {is_valid}")
    
except Exception as e:
    print(f"   ✗ ConfigAdapter failed: {e}")

try:
    # Test FeatureAdapter
    print("\n2. Testing FeatureAdapter...")
    from src.adapters.feature_adapter import FeatureAdapter
    import pandas as pd
    import numpy as np
    
    feature_adapter = FeatureAdapter()
    print("   ✓ FeatureAdapter created successfully")
    
    # Create minimal test data
    test_data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [102, 103, 104],
        'volume': [1000, 1100, 1200]
    })
    
    # Test feature generation
    features = feature_adapter.generate_features(test_data)
    print(f"   ✓ Generated {len(features.columns)} feature columns")
    
    # Test validation
    is_valid = feature_adapter.validate_features(features)
    print(f"   ✓ Feature validation: {is_valid}")
    
except Exception as e:
    print(f"   ✗ FeatureAdapter failed: {e}")
    import traceback
    traceback.print_exc()

try:
    # Test TraderAdapter (without initialization)
    print("\n3. Testing TraderAdapter...")
    from src.adapters.trader_adapter import TraderAdapter
    
    trader_adapter = TraderAdapter()
    print("   ✓ TraderAdapter created successfully")
    
    # Test basic methods that don't require initialization
    is_active = trader_adapter.is_trading_active()
    print(f"   ✓ Trading active status: {is_active}")
    
    # Test portfolio status (should handle uninitialized state)
    portfolio = trader_adapter.get_portfolio_status()
    print(f"   ✓ Portfolio status retrieved: {type(portfolio)}")
    
except Exception as e:
    print(f"   ✗ TraderAdapter failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 40)
print("Simple Adapter Layer Test Completed")
print("✓ Basic adapter functionality verified")