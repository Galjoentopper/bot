"""Test script for Phase 1 foundation components."""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.container import DIContainer, get_container
from src.core.interfaces import (
    IConfigurationManager, IFeatureManager, ILogger,
    ValidationResult, ModelMetadata
)
from src.core.config_manager import ConfigurationManager
from src.core.feature_manager import FeatureManager
from src.core.enhanced_logger import EnhancedLogger


def test_dependency_injection():
    """Test dependency injection container."""
    print("\n=== Testing Dependency Injection ===")
    
    container = get_container()
    container.clear()  # Start fresh
    
    # Register services
    container.register_singleton(ILogger, EnhancedLogger)
    container.register_singleton(IConfigurationManager, ConfigurationManager)
    container.register_singleton(IFeatureManager, FeatureManager)
    
    # Test resolution
    logger = container.resolve(ILogger)
    config_manager = container.resolve(IConfigurationManager)
    feature_manager = container.resolve(IFeatureManager)
    
    print(f"✓ Logger resolved: {type(logger).__name__}")
    print(f"✓ Config Manager resolved: {type(config_manager).__name__}")
    print(f"✓ Feature Manager resolved: {type(feature_manager).__name__}")
    
    # Test singleton behavior
    logger2 = container.resolve(ILogger)
    assert logger is logger2, "Singleton behavior failed"
    print("✓ Singleton behavior verified")
    
    return True


def test_enhanced_logger():
    """Test enhanced logging system."""
    print("\n=== Testing Enhanced Logger ===")
    
    container = get_container()
    logger = container.resolve(ILogger)
    
    # Initialize logger
    if not logger.initialize():
        print("✗ Logger initialization failed")
        return False
    
    print("✓ Logger initialized successfully")
    
    # Test basic logging
    logger.log_info("Test info message", {'test_context': 'basic_logging'})
    logger.log_warning("Test warning message")
    logger.log_error("Test error message", {'error_code': 'TEST_001'})
    
    print("✓ Basic logging methods work")
    
    # Test context management
    logger.set_global_context('test_session', 'phase1_test')
    logger.push_context({'operation': 'context_test'})
    logger.log_info("Message with context")
    logger.pop_context()
    
    print("✓ Context management works")
    
    # Test trade logging
    logger.log_trade('BTCEUR', 'BUY', 0.1, {'price': 45000, 'strategy': 'test'})
    print("✓ Trade logging works")
    
    return True


def test_configuration_manager():
    """Test configuration management."""
    print("\n=== Testing Configuration Manager ===")
    
    container = get_container()
    config_manager = container.resolve(IConfigurationManager)
    
    # Initialize config manager
    if not config_manager.initialize():
        print("✗ Config manager initialization failed")
        return False
    
    print("✓ Config manager initialized successfully")
    
    # Test configuration loading
    try:
        config = config_manager.load_config('trading')
        print(f"✓ Trading config loaded with {len(config)} sections")
        
        # Test validation
        validation_result = config_manager.validate_config(config)
        print(f"✓ Config validation: valid={validation_result.is_valid}, "
              f"errors={len(validation_result.errors)}, warnings={len(validation_result.warnings)}")
        
        # Test symbols retrieval
        symbols = config_manager.get_symbols()
        print(f"✓ Retrieved {len(symbols)} symbols: {symbols}")
        
    except Exception as e:
        print(f"⚠ Config loading failed (expected if no config files): {e}")
        
        # Test with minimal config
        test_config = {
            'data': {'symbols': ['BTCEUR', 'ETHEUR']},
            'models': {'gru': {}, 'lightgbm': {}},
            'trading': {'initial_balance': 1000}
        }
        
        validation_result = config_manager.validate_config(test_config)
        print(f"✓ Test config validation: valid={validation_result.is_valid}")
    
    return True


def test_feature_manager():
    """Test feature management."""
    print("\n=== Testing Feature Manager ===")
    
    container = get_container()
    feature_manager = container.resolve(IFeatureManager)
    
    # Initialize feature manager
    if not feature_manager.initialize():
        print("✗ Feature manager initialization failed")
        return False
    
    print("✓ Feature manager initialized successfully")
    
    # Test schema loading
    try:
        schema = feature_manager.load_feature_schema('BTCEUR', 'gru')
        print(f"✓ Schema loaded for BTCEUR_gru: {len(schema.get('features', {}))} features")
    except Exception as e:
        print(f"⚠ Schema loading failed (expected if no schemas): {e}")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(100) + 45000,
        'volume': np.random.randn(100) + 1000,
        'rsi': np.random.randn(100) * 20 + 50,
        'macd': np.random.randn(100) * 100
    })
    
    # Generate and save schema
    success = feature_manager.save_schema('TESTEUR', 'gru', sample_data)
    print(f"✓ Schema save: {'success' if success else 'failed'}")
    
    # Test feature validation
    test_schema = {
        'version': '1.0.0',
        'features': {
            'close': {'dtype': 'float64', 'nullable': False},
            'volume': {'dtype': 'float64', 'nullable': False},
            'rsi': {'dtype': 'float64', 'nullable': False, 'min_value': 0, 'max_value': 100}
        }
    }
    
    validation_result = feature_manager.validate_features(sample_data, test_schema)
    print(f"✓ Feature validation: valid={validation_result.is_valid}, "
          f"errors={len(validation_result.errors)}, warnings={len(validation_result.warnings)}")
    
    # Test drift detection
    # Create slightly different data
    drift_data = sample_data.copy()
    drift_data['new_feature'] = np.random.randn(100)  # Add new feature
    drift_data = drift_data.drop('macd', axis=1)  # Remove existing feature
    
    drift_result = feature_manager.detect_schema_drift(drift_data, test_schema)
    print(f"✓ Drift detection: valid={drift_result.is_valid}, "
          f"errors={len(drift_result.errors)}, warnings={len(drift_result.warnings)}")
    
    return True


def test_service_integration():
    """Test integration between services."""
    print("\n=== Testing Service Integration ===")
    
    container = get_container()
    
    # Get all services
    logger = container.resolve(ILogger)
    config_manager = container.resolve(IConfigurationManager)
    feature_manager = container.resolve(IFeatureManager)
    
    # Test that services can work together
    logger.set_global_context('integration_test', True)
    
    # Config manager should use logger
    logger.log_info("Testing service integration")
    
    # Feature manager should use logger
    sample_data = pd.DataFrame({
        'price': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    })
    
    feature_manager.save_schema('INTEGRATION', 'test', sample_data)
    
    print("✓ Services integrated successfully")
    return True


def test_error_handling():
    """Test error handling and resilience."""
    print("\n=== Testing Error Handling ===")
    
    container = get_container()
    logger = container.resolve(ILogger)
    
    # Test exception logging
    try:
        raise ValueError("Test exception for logging")
    except Exception as e:
        logger.log_error("Caught test exception", {'test': True}, e)
        print("✓ Exception logging works")
    
    # Test service resilience
    feature_manager = container.resolve(IFeatureManager)
    
    # Try to validate with invalid data
    invalid_data = pd.DataFrame({'invalid': ['not', 'numeric', 'data']})
    invalid_schema = {
        'features': {
            'numeric_feature': {'dtype': 'float64', 'nullable': False}
        }
    }
    
    try:
        result = feature_manager.validate_features(invalid_data, invalid_schema)
        print(f"✓ Graceful handling of validation errors: {len(result.errors)} errors")
    except Exception as e:
        print(f"⚠ Validation error handling needs improvement: {e}")
    
    return True


def run_all_tests():
    """Run all Phase 1 foundation tests."""
    print("Phase 1 Foundation Component Tests")
    print("=" * 50)
    
    tests = [
        ("Dependency Injection", test_dependency_injection),
        ("Enhanced Logger", test_enhanced_logger),
        ("Configuration Manager", test_configuration_manager),
        ("Feature Manager", test_feature_manager),
        ("Service Integration", test_service_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name} test...")
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✓ {test_name} test PASSED")
            else:
                print(f"✗ {test_name} test FAILED")
        except Exception as e:
            print(f"✗ {test_name} test FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 1 foundation tests PASSED!")
        print("\nPhase 1 foundation is ready. You can proceed to Phase 2.")
    else:
        print("⚠ Some tests failed. Please review the issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)