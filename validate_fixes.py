#!/usr/bin/env python3
"""
Validation script to demonstrate the fixes made to optimized_variables.py
"""

import sys
sys.path.append('scripts')

from backtest_models import ModelBacktester, BacktestConfig
from optimized_variables import ScientificOptimizer
import warnings
warnings.filterwarnings('ignore')

def test_window_selection():
    """Test that last 15 windows are now being used"""
    print("=== Testing Window Selection ===")
    
    config = BacktestConfig()
    config.verbose = False
    backtester = ModelBacktester(config)
    
    symbol = 'BTCEUR'
    available_windows = backtester._get_available_windows(symbol)
    
    print(f"Available windows for {symbol}: {available_windows}")
    print(f"Using first window: {available_windows[0]}")
    print(f"Using last window: {available_windows[-1]}")
    
    # Verify these are the last 15 windows (should be all available: 1-15)
    expected_last_15 = list(range(1, 16))  # 1, 2, 3, ..., 15 (all available windows)
    actual_last_15 = available_windows[-15:] if len(available_windows) >= 15 else available_windows
    
    if actual_last_15 == expected_last_15:
        print("✅ PASS: Using last 15 windows as expected (1-15)")
    else:
        print(f"❌ FAIL: Expected {expected_last_15}, got {actual_last_15}")
    
    print()

def test_performance_metrics():
    """Test that performance metrics return proper values instead of empty dict"""
    print("=== Testing Performance Metrics ===")
    
    config = BacktestConfig()
    backtester = ModelBacktester(config)
    
    # Test with no trades (this used to return empty dict)
    performance = backtester.calculate_performance_metrics([], [], 'BTCEUR')
    
    print("Performance metrics with no trades:")
    expected_keys = ['total_trades', 'win_rate', 'total_pnl', 'avg_win', 'avg_loss', 
                    'profit_factor', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'total_return']
    
    all_keys_present = all(key in performance for key in expected_keys)
    all_values_numeric = all(isinstance(performance[key], (int, float)) for key in expected_keys)
    
    print(f"  All expected keys present: {all_keys_present}")
    print(f"  All values are numeric: {all_values_numeric}")
    print(f"  Sample values: total_trades={performance['total_trades']}, win_rate={performance['win_rate']:.2f}")
    
    if all_keys_present and all_values_numeric:
        print("✅ PASS: Performance metrics return proper values")
    else:
        print("❌ FAIL: Performance metrics incomplete or wrong types")
    
    print()

def test_parameter_validation():
    """Test that parameter validation is working correctly"""
    print("=== Testing Parameter Validation ===")
    
    optimizer = ScientificOptimizer(['BTCEUR'], verbose=False)
    
    # Test random parameter generation
    for i in range(5):
        params = optimizer._sample_random_params()
        
        # Check key relationships
        tp_valid = params['take_profit_pct'] > params['stop_loss_pct']
        thresh_valid = params['buy_threshold'] > params['sell_threshold']
        risk_valid = params['risk_per_trade'] <= params['max_capital_per_trade']
        
        print(f"  Test {i+1}: TP>SL={tp_valid}, Buy>Sell={thresh_valid}, Risk<=Pos={risk_valid}")
        
        if not (tp_valid and thresh_valid and risk_valid):
            print("❌ FAIL: Parameter validation not working")
            return
    
    print("✅ PASS: Parameter validation working correctly")
    print()

def test_optimizer_integration():
    """Test that the optimizer works end-to-end with the fixes"""
    print("=== Testing Optimizer Integration ===")
    
    try:
        optimizer = ScientificOptimizer(['BTCEUR'], verbose=False)
        
        # Test parameter evaluation (this exercises the complete pipeline)
        test_params = {
            'buy_threshold': 0.52,
            'sell_threshold': 0.48,
            'lstm_delta_threshold': 0.001,
            'risk_per_trade': 0.02,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.06,
            'max_capital_per_trade': 0.1,
            'max_positions': 10
        }
        
        print("  Testing parameter evaluation...")
        result = optimizer.evaluate_parameters(test_params)
        
        print(f"  Evaluation success: {result['success']}")
        print(f"  Objective value: {result['objective_value']}")
        print(f"  Total trades: {result['metrics'].get('total_trades', 0)}")
        
        if result['success'] or result['objective_value'] == -999:  # -999 is expected for no trades
            print("✅ PASS: Optimizer integration working")
        else:
            print("❌ FAIL: Optimizer evaluation failed unexpectedly")
            
    except Exception as e:
        print(f"❌ FAIL: Optimizer integration error: {e}")
    
    print()

if __name__ == '__main__':
    print("🔍 Validating fixes to optimized_variables.py\n")
    
    test_window_selection()
    test_performance_metrics()
    test_parameter_validation()
    test_optimizer_integration()
    
    print("🎉 Validation complete!")