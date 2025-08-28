#!/usr/bin/env python3
"""Test script for adapter layer validation."""

import sys
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.core.container import DIContainer, resolve
from src.core.interfaces import IConfigurationManager, IFeatureManager, ITradingEngine
from src.adapters.config_adapter import ConfigAdapter
from src.adapters.feature_adapter import FeatureAdapter
from src.adapters.trader_adapter import TraderAdapter


def test_config_adapter():
    """Test ConfigAdapter functionality."""
    print("\n=== Testing ConfigAdapter ===")
    
    try:
        # Register and resolve ConfigAdapter
        container = DIContainer()
        container.register_singleton(IConfigurationManager, ConfigAdapter)
        
        config_manager = resolve(IConfigurationManager)
        
        # Test basic functionality
        print(f"✓ ConfigAdapter created successfully")
        
        # Test configuration access
        symbols = config_manager.get_config('symbols', ['BTCEUR'])
        print(f"✓ Retrieved symbols: {symbols}")
        
        # Test nested key access
        initial_balance = config_manager.get_config('trading.initial_balance', 10000)
        print(f"✓ Retrieved initial balance: {initial_balance}")
        
        # Test validation
        is_valid = config_manager.validate_config()
        print(f"✓ Configuration validation: {is_valid}")
        
        # Test has_config
        has_symbols = config_manager.has_config('symbols')
        print(f"✓ Has symbols config: {has_symbols}")
        
        print("✓ ConfigAdapter tests passed")
        return True
        
    except Exception as e:
        print(f"✗ ConfigAdapter test failed: {e}")
        return False


def test_feature_adapter():
    """Test FeatureAdapter functionality."""
    print("\n=== Testing FeatureAdapter ===")
    
    try:
        # Register and resolve FeatureAdapter
        container = DIContainer()
        container.register_singleton(IFeatureManager, FeatureAdapter)
        
        feature_manager = resolve(IFeatureManager)
        
        print(f"✓ FeatureAdapter created successfully")
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(50000, 55000, 100),
            'low': np.random.uniform(35000, 40000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Ensure high >= low and other OHLC relationships
        sample_data['high'] = np.maximum(sample_data[['open', 'close']].max(axis=1), sample_data['high'])
        sample_data['low'] = np.minimum(sample_data[['open', 'close']].min(axis=1), sample_data['low'])
        
        print(f"✓ Created sample data with {len(sample_data)} records")
        
        # Test feature generation
        features_df = feature_manager.generate_features(sample_data)
        print(f"✓ Generated features: {len(features_df.columns)} columns")
        
        # Test feature validation
        is_valid = feature_manager.validate_features(features_df)
        print(f"✓ Feature validation: {is_valid}")
        
        # Test schema operations
        schema = feature_manager.get_feature_schema()
        print(f"✓ Retrieved feature schema with {len(schema.get('features', {}))} features")
        
        # Test drift detection with same data (should show no drift)
        drift_results = feature_manager.detect_drift(features_df[:50], features_df[50:])
        print(f"✓ Drift detection completed: {drift_results.get('summary', {})}")
        
        print("✓ FeatureAdapter tests passed")
        return True
        
    except Exception as e:
        print(f"✗ FeatureAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trader_adapter():
    """Test TraderAdapter functionality."""
    print("\n=== Testing TraderAdapter ===")
    
    try:
        # Register and resolve TraderAdapter
        container = DIContainer()
        container.register_singleton(ITradingEngine, TraderAdapter)
        
        trading_engine = resolve(ITradingEngine)
        
        print(f"✓ TraderAdapter created successfully")
        
        # Test initialization
        await trading_engine.initialize()
        print(f"✓ TraderAdapter initialized")
        
        # Test portfolio status
        portfolio = trading_engine.get_portfolio_status()
        print(f"✓ Retrieved portfolio status: {portfolio.get('total_balance', 'N/A')}")
        
        # Test trading status
        is_active = trading_engine.is_trading_active()
        print(f"✓ Trading active status: {is_active}")
        
        # Test trade execution (simulation)
        trade_result = trading_engine.execute_trade('BTCEUR', 'buy', 0.001, 45000)
        print(f"✓ Trade execution result: {trade_result.get('success', False)}")
        
        # Test trading history
        history = trading_engine.get_trading_history(10)
        print(f"✓ Retrieved trading history: {len(history)} trades")
        
        # Test market data
        market_data = trading_engine.get_market_data('BTCEUR', '1h', 10)
        print(f"✓ Retrieved market data: {len(market_data)} records")
        
        # Test shutdown
        await trading_engine.shutdown()
        print(f"✓ TraderAdapter shutdown completed")
        
        print("✓ TraderAdapter tests passed")
        return True
        
    except Exception as e:
        print(f"✗ TraderAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between adapters."""
    print("\n=== Testing Adapter Integration ===")
    
    try:
        # Register all adapters
        container = DIContainer()
        container.register_singleton(IConfigurationManager, ConfigAdapter)
        container.register_singleton(IFeatureManager, FeatureAdapter)
        container.register_singleton(ITradingEngine, TraderAdapter)
        
        # Resolve all services
        config_manager = resolve(IConfigurationManager)
        feature_manager = resolve(IFeatureManager)
        trading_engine = resolve(ITradingEngine)
        
        print(f"✓ All adapters resolved successfully")
        
        # Test configuration sharing
        symbols = config_manager.get_config('symbols', ['BTCEUR', 'ETHEUR'])
        print(f"✓ Configuration shared: symbols = {symbols}")
        
        # Test feature generation with config
        feature_config = {
            'sma_periods': [5, 10, 20],
            'rsi_period': 14
        }
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [45000, 45100, 45200],
            'high': [45500, 45600, 45700],
            'low': [44500, 44600, 44700],
            'close': [45200, 45300, 45400],
            'volume': [100, 150, 120]
        })
        
        features_df = feature_manager.generate_features(sample_data)
        print(f"✓ Feature generation integrated: {len(features_df.columns)} features")
        
        # Test portfolio configuration
        portfolio = trading_engine.get_portfolio_status()
        print(f"✓ Trading engine integrated: {len(portfolio.get('symbols', []))} symbols")
        
        print("✓ Adapter integration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Adapter integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all adapter tests."""
    print("Starting Adapter Layer Validation Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run individual adapter tests
    test_results.append(test_config_adapter())
    test_results.append(test_feature_adapter())
    test_results.append(await test_trader_adapter())
    test_results.append(test_integration())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "=" * 50)
    print(f"Adapter Layer Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All adapter layer tests passed! Phase 2 adapter layer is ready.")
        return True
    else:
        print(f"✗ {total - passed} tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)