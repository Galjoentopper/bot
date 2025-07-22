#!/usr/bin/env python3
"""
Debug script to test signal generation logic
"""

import sys
sys.path.append('scripts')
from backtest_models import BacktestConfig, ModelBacktester
import numpy as np

def test_signal_generation():
    """Test the signal generation with various inputs"""
    config = BacktestConfig()
    backtester = ModelBacktester(config)
    
    print("🔍 Testing signal generation logic...")
    print(f"Current thresholds:")
    print(f"  buy_threshold: {config.buy_threshold}")
    print(f"  sell_threshold: {config.sell_threshold}")
    print(f"  lstm_delta_threshold: {config.lstm_delta_threshold}")
    print()
    
    # Test cases with different xgb_prob and lstm_delta values
    test_cases = [
        # (xgb_prob, lstm_delta, description)
        (0.5, 0.0, "Exactly neutral"),
        (0.51, 0.001, "Slightly bullish"),
        (0.49, -0.001, "Slightly bearish"),
        (0.6, 0.01, "Strongly bullish"),
        (0.4, -0.01, "Strongly bearish"),
        (0.502, 0.0001, "Just above buy threshold"),
        (0.498, -0.0001, "Just above sell threshold"),
        (0.5001, 0.00000001, "Minimal bias toward buy"),
        (0.4999, -0.00000001, "Minimal bias toward sell"),
    ]
    
    signal_stats = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'primary_signals': 0, 
                   'secondary_signals': 0, 'tertiary_signals': 0}
    
    for xgb_prob, lstm_delta, description in test_cases:
        signal = backtester.generate_signal_with_adaptive_stats(
            xgb_prob, lstm_delta, signal_stats, aggressiveness_multiplier=1.0
        )
        print(f"{description:20} -> XGB={xgb_prob:.6f}, LSTM={lstm_delta:.8f} -> {signal}")
    
    print(f"\nSignal statistics: {signal_stats}")
    print(f"Total signals generated: {signal_stats['BUY'] + signal_stats['SELL']}")
    print(f"HOLD signals: {signal_stats['HOLD']}")
    
    # Test with higher aggressiveness
    print(f"\n🚀 Testing with higher aggressiveness (2.0)...")
    signal_stats_aggressive = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'primary_signals': 0, 
                              'secondary_signals': 0, 'tertiary_signals': 0}
    
    for xgb_prob, lstm_delta, description in test_cases:
        signal = backtester.generate_signal_with_adaptive_stats(
            xgb_prob, lstm_delta, signal_stats_aggressive, aggressiveness_multiplier=2.0
        )
        print(f"{description:20} -> XGB={xgb_prob:.6f}, LSTM={lstm_delta:.8f} -> {signal}")
    
    print(f"\nAggressive signal statistics: {signal_stats_aggressive}")
    print(f"Total signals generated: {signal_stats_aggressive['BUY'] + signal_stats_aggressive['SELL']}")
    print(f"HOLD signals: {signal_stats_aggressive['HOLD']}")

if __name__ == '__main__':
    test_signal_generation()