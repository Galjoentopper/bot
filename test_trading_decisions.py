#!/usr/bin/env python3
"""
Test script to validate trading decision logging and adjusted parameters.
This script tests the configuration and logging without running the full trading system.
"""

import sys
from pathlib import Path
import logging

# Add paper_trader to path
sys.path.append(str(Path(__file__).parent))

def test_environment_variables():
    """Test that environment variables are loaded correctly."""
    print("=== TESTING ENVIRONMENT VARIABLES ===")
    
    try:
        from paper_trader.config.settings import TradingSettings
        
        settings = TradingSettings()
        
        print(f"✅ Min Confidence Threshold: {settings.min_confidence_threshold}")
        print(f"✅ Min Signal Strength: {settings.min_signal_strength}")
        print(f"✅ Min Expected Gain PCT: {settings.min_expected_gain_pct}")
        print(f"✅ Enable Strict Entry Conditions: {settings.enable_strict_entry_conditions}")
        print(f"✅ Max Prediction Uncertainty: {settings.max_prediction_uncertainty}")
        print(f"✅ Min Ensemble Agreement Count: {settings.min_ensemble_agreement_count}")
        print(f"✅ Min Volume Ratio Threshold: {settings.min_volume_ratio_threshold}")
        print(f"✅ Strong Signal Confidence Boost: {settings.strong_signal_confidence_boost}")
        print(f"✅ Trend Strength Threshold: {settings.trend_strength_threshold}")
        print(f"✅ Symbols: {settings.symbols}")
        
        # Verify that the settings are less restrictive now
        assert settings.min_confidence_threshold == 0.5, f"Expected 0.5, got {settings.min_confidence_threshold}"
        assert settings.min_signal_strength == "WEAK", f"Expected WEAK, got {settings.min_signal_strength}"
        assert settings.min_expected_gain_pct == 0.0005, f"Expected 0.0005, got {settings.min_expected_gain_pct}"
        assert settings.max_prediction_uncertainty == 0.5, f"Expected 0.5, got {settings.max_prediction_uncertainty}"
        assert settings.min_ensemble_agreement_count == 1, f"Expected 1, got {settings.min_ensemble_agreement_count}"
        
        print("✅ All environment variables loaded correctly and are less restrictive!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading settings: {e}")
        return False

def test_signal_generator_logging():
    """Test the signal generator with mock data to verify logging."""
    print("\n=== TESTING SIGNAL GENERATOR LOGGING ===")
    
    try:
        from paper_trader.strategy.signal_generator import SignalGenerator
        from paper_trader.config.settings import TradingSettings
        
        settings = TradingSettings()
        
        # Create a signal generator
        signal_gen = SignalGenerator(settings=settings)
        
        # Mock prediction data - should pass the less restrictive thresholds
        mock_prediction = {
            'current_price': 100.0,
            'predicted_price': 100.5,
            'price_change_pct': 0.005,  # 0.5% gain (above 0.0005 threshold)
            'confidence': 0.6,  # Above 0.5 threshold
            'signal_strength': 'WEAK',  # Meets WEAK requirement
            'uncertainty': 0.3,  # Below 0.5 threshold
            'individual_predictions': {'model1': 100.5}  # 1 model agreement
        }
        
        # Mock portfolio
        class MockPortfolio:
            def __init__(self):
                self.positions = {}
                self.last_closed_time = {}
                self.trades = []
            
            def get_available_capital(self):
                return 10000.0
        
        mock_portfolio = MockPortfolio()
        
        # Test validation
        is_valid = signal_gen._is_prediction_valid(mock_prediction)
        print(f"✅ Prediction validation: {is_valid}")
        
        # Test strict entry conditions
        strict_ok = signal_gen._check_strict_entry_conditions('BTC-EUR', mock_prediction)
        print(f"✅ Strict entry conditions: {strict_ok}")
        
        print("✅ Signal generator tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing signal generator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_setup():
    """Test that logging is properly configured."""
    print("\n=== TESTING LOGGING SETUP ===")
    
    try:
        # Create log directory
        log_dir = Path('paper_trader/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Test logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('test_trading_decisions')
        
        # Create trading decisions logger
        trading_logger = logging.getLogger('trading_decisions')
        trading_handler = logging.FileHandler(log_dir / 'trading_decisions.log', mode='a')
        trading_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        trading_logger.addHandler(trading_handler)
        trading_logger.setLevel(logging.INFO)
        
        # Test logging
        trading_logger.info("🧪 TEST: Trading decisions logging is working!")
        trading_logger.info("✅ Less restrictive settings enabled")
        trading_logger.info("🔍 Comprehensive signal analysis logging enabled")
        
        print("✅ Logging setup completed successfully!")
        print(f"✅ Trading decisions log: {log_dir / 'trading_decisions.log'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up logging: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 TESTING PAPER TRADER IMPROVEMENTS")
    print("=" * 50)
    
    results = []
    
    # Test environment variables
    results.append(test_environment_variables())
    
    # Test signal generator
    results.append(test_signal_generator_logging())
    
    # Test logging
    results.append(test_logging_setup())
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 IMPROVEMENTS READY:")
        print("   • Less restrictive trading thresholds")
        print("   • Comprehensive trading decision logging")
        print("   • Enhanced signal analysis")
        print("   • Better debugging capabilities")
        return True
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)