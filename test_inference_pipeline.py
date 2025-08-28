#!/usr/bin/env python3
"""
Test script to verify the complete inference pipeline with alignment fixes.
This script tests that all models (GRU, LightGBM, PPO) can predict successfully.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.enhanced_trader import EnhancedUnifiedPaperTrader
from src.utils.logger import Logger

def create_test_data():
    """Create sample market data for testing."""
    # Generate 100 rows of sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Create realistic price data
    np.random.seed(42)
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, 100)  # 2% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Minimum price of 1000
    
    data = {
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100),
        'quote_volume': np.random.uniform(1000000, 10000000, 100),
        'trades': np.random.randint(50, 500, 100)
    }
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def test_model_predictions(trader, test_data, symbol='BTCEUR'):
    """Test predictions for all model types."""
    logger = Logger("test_inference")
    results = {}
    
    # Test each model type
    model_types = ['gru', 'lightgbm', 'ppo']
    
    for model_type in model_types:
        logger.logger.info(f"Testing {model_type.upper()} model for {symbol}...")
        
        try:
            if model_type == 'gru':
                prediction = trader._get_gru_prediction(test_data, symbol)
            elif model_type == 'lightgbm':
                prediction = trader._get_lightgbm_prediction(test_data, symbol)
            elif model_type == 'ppo':
                prediction = trader._get_ppo_prediction(test_data, symbol)
            
            results[model_type] = {
                'success': True,
                'prediction': prediction,
                'error': None
            }
            
            logger.logger.info(f"{model_type.upper()} prediction: {prediction}")
            
        except Exception as e:
            results[model_type] = {
                'success': False,
                'prediction': None,
                'error': str(e)
            }
            
            logger.logger.error(f"{model_type.upper()} prediction failed: {e}")
    
    return results

def main():
    """Main test function."""
    logger = Logger("test_main")
    logger.logger.info("Starting inference pipeline test...")
    
    try:
        # Initialize trader
        logger.logger.info("Initializing EnhancedUnifiedPaperTrader...")
        trader = EnhancedUnifiedPaperTrader()
        
        # Load models
        logger.logger.info("Loading models...")
        trader.load_all_models()
        
        # Create test data
        logger.logger.info("Creating test data...")
        test_data = create_test_data()
        
        # Test predictions for each symbol
        symbols = ['BTCEUR', 'ETHEUR', 'ADAEUR']
        all_results = {}
        
        for symbol in symbols:
            logger.logger.info(f"\n=== Testing {symbol} ===")
            
            # Check if models are loaded for this symbol
            if symbol not in trader.models:
                logger.logger.warning(f"No models loaded for {symbol}, skipping...")
                continue
            
            results = test_model_predictions(trader, test_data, symbol)
            all_results[symbol] = results
            
            # Print results for this symbol
            logger.logger.info(f"Results for {symbol}:")
            for model_type, result in results.items():
                status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
                logger.logger.info(f"  {model_type.upper()}: {status}")
                if not result['success']:
                    logger.logger.error(f"    Error: {result['error']}")
        
        # Summary
        logger.logger.info("\n=== SUMMARY ===")
        total_tests = 0
        successful_tests = 0
        
        for symbol, symbol_results in all_results.items():
            for model_type, result in symbol_results.items():
                total_tests += 1
                if result['success']:
                    successful_tests += 1
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        logger.logger.info(f"Success rate: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate == 100:
            logger.logger.info("🎉 All tests passed! Inference pipeline is working correctly.")
        elif success_rate >= 80:
            logger.logger.warning(f"⚠️  Most tests passed ({success_rate:.1f}%), but some issues remain.")
        else:
            logger.logger.error(f"❌ Many tests failed ({success_rate:.1f}%), significant issues detected.")
        
        return success_rate == 100
        
    except Exception as e:
        logger.logger.error(f"Test failed with exception: {e}")
        import traceback
        logger.logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)