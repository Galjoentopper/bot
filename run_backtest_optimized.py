#!/usr/bin/env python3
"""
Optimized version of the backtest runner for faster execution.

Key optimizations:
1. Model caching to avoid reloading TensorFlow models
2. Batch processing of windows
3. Reduced debug output
4. Memory-efficient data handling

Usage:
    python run_backtest_optimized.py --symbol BTCEUR --config aggressive
"""

import argparse
import os
import sys
import time
from typing import Dict, List

# Add scripts directory to path
sys.path.append('scripts')
from backtest_models import ModelBacktester, BacktestConfig

# Configuration presets with more aggressive thresholds
CONFIG_PRESETS = {
    'aggressive': {
        'buy_threshold': 0.52,
        'sell_threshold': 0.48,
        'lstm_delta_threshold': 0.001,
        'train_months': 3,  # Reduced for faster execution
        'test_months': 1,
        'slide_months': 1
    },
    'very_aggressive': {
        'buy_threshold': 0.51,
        'sell_threshold': 0.49,
        'lstm_delta_threshold': 0.0005,
        'train_months': 3,
        'test_months': 1,
        'slide_months': 1
    },
    'fast_test': {
        'buy_threshold': 0.50,
        'sell_threshold': 0.50,
        'lstm_delta_threshold': 0.0001,
        'train_months': 2,  # Very short for quick testing
        'test_months': 1,
        'slide_months': 2  # Skip more windows
    }
}

class OptimizedBacktester(ModelBacktester):
    """Optimized version of ModelBacktester with performance improvements"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.model_cache = {}  # Cache for loaded models
        self.debug_frequency = 1000  # Reduce debug output frequency
    
    def load_models_cached(self, symbol: str, window_num: int):
        """Load models with caching to avoid repeated TensorFlow loading"""
        cache_key = f"{symbol}_{window_num}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # Load models using parent method directly
        models = super().load_models(symbol, window_num)
        
        # Cache the models (but be careful with memory)
        if len(self.model_cache) < 10:  # Limit cache size
            self.model_cache[cache_key] = models
        
        return models
    
    def run_fast_backtest(self, symbol: str, max_windows: int = None) -> Dict:
        """Run optimized backtest for a single symbol"""
        print(f"\n{'='*60}")
        print(f"RUNNING OPTIMIZED BACKTEST FOR {symbol}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Override the model loading method temporarily
        original_load_models = self.load_models
        self.load_models = self.load_models_cached
        
        try:
            # Run backtest for single symbol
            results = self.backtest_symbol(symbol)
            
            # Limit windows for testing if specified
            if max_windows and len(results['equity_history']) > max_windows:
                results['equity_history'] = results['equity_history'][:max_windows]
                results['trades'] = [t for t in results['trades'] 
                                   if t.entry_time <= results['equity_history'][-1]['date']]
                results['performance'] = self.calculate_performance_metrics(
                    results['trades'], results['equity_history'], symbol
                )
            
            # Save results
            self.save_results(results)
            
            execution_time = time.time() - start_time
            print(f"\nBacktest completed in {execution_time:.1f} seconds")
            
            # Print summary
            perf = results['performance']
            print(f"\n{symbol} Results Summary:")
            print(f"  Total Trades: {perf.get('total_trades', 0)}")
            print(f"  Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"  Total Return: {perf.get('total_return', 0):.2%}")
            print(f"  Final Capital: €{results['final_capital']:.2f}")
            
            return results
            
        finally:
            # Restore original method
            self.load_models = original_load_models
            # Clear cache to free memory
            self.model_cache.clear()

def create_config(preset_name: str) -> BacktestConfig:
    """Create configuration from preset"""
    config = BacktestConfig()
    
    if preset_name in CONFIG_PRESETS:
        preset = CONFIG_PRESETS[preset_name]
        for key, value in preset.items():
            setattr(config, key, value)
    
    return config

def main():
    """Main function for optimized backtesting"""
    parser = argparse.ArgumentParser(description='Optimized Cryptocurrency Trading Backtester')
    parser.add_argument('--symbol', default='BTCEUR', 
                       choices=['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR'],
                       help='Symbol to backtest')
    parser.add_argument('--config', default='aggressive',
                       choices=['aggressive', 'very_aggressive', 'fast_test'],
                       help='Configuration preset')
    parser.add_argument('--max-windows', type=int, default=None,
                       help='Maximum number of windows to test (for quick testing)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Generate and save performance plots')
    
    args = parser.parse_args()
    
    print(f"Starting optimized backtest...")
    print(f"Symbol: {args.symbol}")
    print(f"Configuration: {args.config}")
    if args.max_windows:
        print(f"Max windows: {args.max_windows}")
    
    # Create configuration
    config = create_config(args.config)
    
    # Create optimized backtester
    backtester = OptimizedBacktester(config)
    
    # Run backtest
    try:
        results = backtester.run_fast_backtest(args.symbol, args.max_windows)
        
        if args.save_plots:
            print("\nGenerating performance plots...")
            backtester.plot_results(results)
        
        print(f"\nResults saved to: backtests/{args.symbol}/")
        print("\nOptimization suggestions:")
        print("1. If no trades were executed, try 'very_aggressive' or 'fast_test' config")
        print("2. Use --max-windows 10 for quick testing")
        print("3. Check equity_curve.csv for capital progression")
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())