#!/usr/bin/env python3
"""
Quick test script to validate trading logic and model predictions.
This script tests just a few windows to quickly identify if trades can be generated.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add scripts directory to path
sys.path.append('scripts')
from backtest_models import ModelBacktester, BacktestConfig
from train_hybrid_models import HybridModelTrainer

def quick_validation_test(symbol='BTCEUR', max_windows=3):
    """Run a quick validation test with just a few windows"""
    print(f"\n{'='*50}")
    print(f"QUICK VALIDATION TEST FOR {symbol}")
    print(f"Testing {max_windows} windows only")
    print(f"{'='*50}")
    
    # Create very aggressive config for testing
    config = BacktestConfig()
    config.buy_threshold = 0.50  # Very low threshold
    config.sell_threshold = 0.50
    config.lstm_delta_threshold = 0.0001  # Very low threshold
    config.train_months = 3
    config.test_months = 1
    config.slide_months = 1
    config.initial_capital = 10000.0
    config.risk_per_trade = 0.02
    
    print(f"Configuration:")
    print(f"  Buy threshold: {config.buy_threshold}")
    print(f"  Sell threshold: {config.sell_threshold}")
    print(f"  LSTM delta threshold: {config.lstm_delta_threshold}")
    
    # Create backtester
    backtester = ModelBacktester(config)
    
    # Load data
    print(f"\nLoading data for {symbol}...")
    data = backtester.load_data(symbol)
    print(f"Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    # Generate windows
    trainer = HybridModelTrainer()
    windows = trainer.generate_walk_forward_windows(data)
    print(f"Total windows available: {len(windows)}")
    
    # Test only first few windows
    test_windows = windows[:max_windows]
    print(f"Testing windows: {len(test_windows)}")
    
    total_predictions = 0
    total_buy_signals = 0
    total_sell_signals = 0
    
    for i, (train_start, train_end, test_end) in enumerate(test_windows):
        print(f"\n--- Window {i+1}/{len(test_windows)} ---")
        print(f"Train: {train_start} to {train_end}")
        print(f"Test: {train_end} to {test_end}")
        
        # Try to load models
        try:
            models = backtester.load_models(symbol, i+1)
            if models['lstm_model'] is None or models['xgb_model'] is None:
                print(f"  ⚠️  Models not found for window {i+1}")
                continue
            print(f"  ✓ Models loaded successfully")
        except Exception as e:
            print(f"  ❌ Error loading models: {e}")
            continue
        
        # Get test data
        test_data = data[train_end:test_end].copy()
        if len(test_data) == 0:
            print(f"  ⚠️  No test data for this window")
            continue
        
        print(f"  Test data: {len(test_data)} rows")
        
        # Test a few predictions
        sample_size = min(100, len(test_data))  # Test first 100 rows
        sample_data = test_data.head(sample_size)
        
        buy_signals = 0
        sell_signals = 0
        predictions_made = 0
        
        for idx in range(len(sample_data)):
            try:
                # Get current row
                current_data = sample_data.iloc[idx:idx+1]
                
                # Generate prediction
                signal = backtester.generate_signal(
                    current_data, models, symbol, debug=False
                )
                
                predictions_made += 1
                
                if signal == 'BUY':
                    buy_signals += 1
                elif signal == 'SELL':
                    sell_signals += 1
                    
            except Exception as e:
                # Skip problematic predictions
                continue
        
        print(f"  Predictions made: {predictions_made}")
        print(f"  Buy signals: {buy_signals}")
        print(f"  Sell signals: {sell_signals}")
        print(f"  Hold signals: {predictions_made - buy_signals - sell_signals}")
        
        total_predictions += predictions_made
        total_buy_signals += buy_signals
        total_sell_signals += sell_signals
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total predictions: {total_predictions}")
    print(f"Total buy signals: {total_buy_signals} ({total_buy_signals/max(total_predictions,1)*100:.1f}%)")
    print(f"Total sell signals: {total_sell_signals} ({total_sell_signals/max(total_predictions,1)*100:.1f}%)")
    print(f"Total hold signals: {total_predictions-total_buy_signals-total_sell_signals} ({(total_predictions-total_buy_signals-total_sell_signals)/max(total_predictions,1)*100:.1f}%)")
    
    if total_buy_signals > 0 or total_sell_signals > 0:
        print(f"\n✅ SUCCESS: Trading signals are being generated!")
        print(f"   The models are working and can generate trades.")
        print(f"   You can now run the full backtest with confidence.")
    else:
        print(f"\n❌ ISSUE: No trading signals generated")
        print(f"   This suggests the thresholds are still too conservative")
        print(f"   or there might be an issue with the models.")
    
    return total_buy_signals + total_sell_signals > 0

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick validation test')
    parser.add_argument('--symbol', default='BTCEUR', help='Symbol to test')
    parser.add_argument('--windows', type=int, default=3, help='Number of windows to test')
    
    args = parser.parse_args()
    
    try:
        success = quick_validation_test(args.symbol, args.windows)
        
        if success:
            print(f"\n🚀 Ready to run full backtest!")
            print(f"   Try: python run_backtest_optimized.py --symbol {args.symbol} --config very_aggressive")
        else:
            print(f"\n🔧 Need to adjust thresholds further")
            print(f"   Consider lowering buy_threshold to 0.49 or lstm_delta_threshold to 0.00005")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())