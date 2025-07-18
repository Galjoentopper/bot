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
import random
from datetime import timedelta
from typing import Dict, List

# Add scripts directory to path
sys.path.append('scripts')
from backtest_models import ModelBacktester, BacktestConfig, Trade

# Configuration presets with more aggressive thresholds
CONFIG_PRESETS = {
    'balanced': {
        'buy_threshold': 0.45,
        'sell_threshold': 0.55,
        'lstm_delta_threshold': 0.0001,
        'risk_per_trade': 0.01,
        'stop_loss_pct': 0.02,
        'train_months': 4,
        'test_months': 1,
        'slide_months': 1
    },
    'aggressive': {
        'buy_threshold': 0.42,
        'sell_threshold': 0.58,
        'lstm_delta_threshold': 0.00005,
        'risk_per_trade': 0.015,
        'stop_loss_pct': 0.015,
        'train_months': 3,
        'test_months': 1,
        'slide_months': 1
    },
    'conservative': {
        'buy_threshold': 0.48,
        'sell_threshold': 0.52,
        'lstm_delta_threshold': 0.0005,
        'risk_per_trade': 0.005,
        'stop_loss_pct': 0.025,
        'train_months': 4,
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
    
    def __init__(self, config: BacktestConfig, debug_mode: bool = False):
        super().__init__(config)
        self.model_cache = {}  # Cache for loaded models (unlimited size)
        self.debug_frequency = 2000  # Reduce debug output frequency
        self.debug_mode = debug_mode
        self.debug_data = []  # Store debug information for CSV export
    
    def load_models_cached(self, symbol: str, window_num: int):
        """Load models with caching to avoid repeated TensorFlow loading"""
        cache_key = f"{symbol}_{window_num}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # Load models using parent method directly
        models = super().load_models(symbol, window_num)
        
        # Cache the models (unlimited cache for better performance)
        self.model_cache[cache_key] = models
        
        return models
    
    def simulate_trading_window(self, data, lstm_model, xgb_model, scaler,
                                symbol, initial_capital, positions, window_num):
        """Override simulate_trading_window to add debug data collection"""
        if not self.debug_mode:
            # Use parent method if debug mode is disabled
            return super().simulate_trading_window(
                data, lstm_model, xgb_model, scaler,
                symbol, initial_capital, positions, window_num
            )
        
        # Debug mode: collect detailed information
        import csv
        import pandas as pd
        
        capital = initial_capital
        window_trades = []
        trades_this_hour = 0
        last_trade_hour = None
        
        for i in range(self.config.sequence_length, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Reset hourly trade counter
            if last_trade_hour is None or current_time.hour != last_trade_hour:
                trades_this_hour = 0
                last_trade_hour = current_time.hour
            
            # Check for exits first
            positions, exit_trades = self.check_exits(positions, current_time, current_price)
            window_trades.extend(exit_trades)
            
            # Update capital from closed trades
            for trade in exit_trades:
                capital += trade.pnl
            
            # Skip if we've hit trade limits
            position_limit_hit = len(positions) >= self.config.max_positions
            hourly_limit_hit = trades_this_hour >= self.config.max_trades_per_hour
            
            # Generate predictions for debug data (even if we can't trade)
            try:
                # LSTM prediction
                lstm_sequence = self.create_lstm_sequences(
                    data.iloc[i-self.config.sequence_length:i+1], scaler
                )[-1:]
                lstm_pred = lstm_model.predict(lstm_sequence, verbose=0)[0][0]
                lstm_delta = lstm_pred
                
                # XGBoost prediction
                xgb_features = self.get_xgb_features(
                    data.iloc[:i+1], lstm_delta, window_num
                )
                xgb_prob = xgb_model.predict_proba(xgb_features)[0][1]
                
                # Generate signal
                signal = self.generate_signal(xgb_prob, lstm_delta)
                
                # Collect debug data
                debug_row = {
                    'timestamp': current_time,
                    'symbol': symbol,
                    'price': current_price,
                    'lstm_prediction': lstm_pred,
                    'lstm_delta': lstm_delta,
                    'xgb_probability': xgb_prob,
                    'signal': signal,
                    'buy_threshold': self.config.buy_threshold,
                    'sell_threshold': self.config.sell_threshold,
                    'lstm_delta_threshold': self.config.lstm_delta_threshold,
                    'capital': capital,
                    'open_positions': len(positions),
                    'position_limit_hit': position_limit_hit,
                    'hourly_limit_hit': hourly_limit_hit,
                    'atr': data['atr'].iloc[i] if 'atr' in data.columns else 0,
                    'volume': data['volume'].iloc[i] if 'volume' in data.columns else 0,
                    'rsi': data['rsi'].iloc[i] if 'rsi' in data.columns else 0,
                    'macd': data['macd'].iloc[i] if 'macd' in data.columns else 0,
                    'bb_upper': data['bb_upper'].iloc[i] if 'bb_upper' in data.columns else 0,
                    'bb_lower': data['bb_lower'].iloc[i] if 'bb_lower' in data.columns else 0,
                    'trade_executed': False,
                    'exit_trades_count': len(exit_trades)
                }
                
                # Execute trade if conditions are met
                if signal in ['BUY', 'SELL'] and not position_limit_hit and not hourly_limit_hit:
                    # Calculate stop loss
                    atr = data['atr'].iloc[i]
                    if signal == 'BUY':
                        stop_loss = current_price - (atr * 2)
                    else:
                        stop_loss = current_price + (atr * 2)
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(capital, current_price, stop_loss)
                    
                    if position_size > 0:
                        # Execute trade
                        execution_price, slippage, fees = self.apply_slippage_and_fees(current_price, signal)
                        
                        trade = Trade(
                            symbol=symbol,
                            entry_time=current_time,
                            entry_price=execution_price,
                            direction=signal,
                            confidence=max(xgb_prob, 1-xgb_prob),
                            position_size=position_size,
                            stop_loss=stop_loss,
                            fees=fees,
                            slippage=slippage
                        )
                        
                        positions.append(trade)
                        capital -= (position_size * execution_price + fees)
                        trades_this_hour += 1
                        
                        # Update debug data for executed trade
                        debug_row.update({
                            'trade_executed': True,
                            'execution_price': execution_price,
                            'position_size': position_size,
                            'stop_loss': stop_loss,
                            'fees': fees,
                            'slippage': slippage,
                            'confidence': trade.confidence
                        })
                
                self.debug_data.append(debug_row)
                
            except Exception as e:
                print(f"    Error generating signal at {current_time}: {e}")
                # Still add debug row with error info
                debug_row = {
                    'timestamp': current_time,
                    'symbol': symbol,
                    'price': current_price,
                    'error': str(e),
                    'capital': capital,
                    'open_positions': len(positions)
                }
                self.debug_data.append(debug_row)
                continue
        
        # Close remaining positions at window end
        final_time = data.index[-1]
        final_price = data['close'].iloc[-1]
        positions, final_trades = self.check_exits(positions, final_time, final_price, force_exit=True)
        window_trades.extend(final_trades)
        
        # Update capital from final trades
        for trade in final_trades:
            capital += trade.pnl
        
        return window_trades, capital
    
    def save_debug_csv(self, symbol: str):
        """Save debug data to CSV file"""
        if not self.debug_data:
            print("No debug data to save.")
            return
        
        import pandas as pd
        import os
        
        # Create debug directory
        debug_dir = f"backtests/{symbol}"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Convert debug data to DataFrame
        df = pd.DataFrame(self.debug_data)
        
        # Save to CSV
        debug_file = f"{debug_dir}/debug_detailed.csv"
        df.to_csv(debug_file, index=False)
        
        print(f"Debug data saved to: {debug_file}")
        print(f"Debug CSV contains {len(df)} rows with detailed trading information")
        
        # Print summary statistics
        if 'signal' in df.columns:
            signal_counts = df['signal'].value_counts()
            print(f"Signal distribution: {dict(signal_counts)}")
        
        if 'trade_executed' in df.columns:
            executed_trades = df['trade_executed'].sum()
            print(f"Total trades executed: {executed_trades}")
    
    def backtest_symbol(self, symbol: str, max_windows: int = None, random_start: bool = False) -> Dict:
        """Override backtest_symbol to support max_windows limit and random starting date"""
        print(f"\nBacktesting {symbol}...")
        if max_windows:
            print(f"  Limited to {max_windows} windows")
        if random_start:
            print(f"  Using random starting date")
        
        # Load data
        data = self.load_data(symbol)
        data = self.calculate_features(data)
        
        print(f"  Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
        
        # Initialize portfolio
        capital = self.config.initial_capital
        positions = []
        trades = []
        equity_history = []
        
        # Walk-forward validation
        original_start_date = data.index[0]
        end_date = data.index[-1]
        
        # Calculate random starting date if requested
        if random_start:
            # Calculate minimum required data for training + testing
            min_required_days = (self.config.train_months + self.config.test_months) * 30
            if max_windows:
                # Add extra days for additional windows
                min_required_days += (max_windows - 1) * self.config.slide_months * 30
            
            # Find the latest possible start date that still allows for required data
            latest_start = end_date - timedelta(days=min_required_days)
            
            if latest_start > original_start_date:
                # Generate random date between original start and latest possible start
                time_diff = latest_start - original_start_date
                random_days = random.randint(0, time_diff.days)
                start_date = original_start_date + timedelta(days=random_days)
                print(f"  Random start date selected: {start_date.date()}")
            else:
                start_date = original_start_date
                print(f"  Dataset too small for random start, using original start: {start_date.date()}")
        else:
            start_date = original_start_date
        
        # Calculate the correct window number for random start date
        if random_start and start_date != original_start_date:
            # Calculate how many months from original start to random start
            months_diff = (start_date.year - original_start_date.year) * 12 + (start_date.month - original_start_date.month)
            
            # Calculate which window this corresponds to
            # Windows start at train_months after original start, then advance by slide_months
            # Window 1 starts at: original_start + train_months
            # Window 2 starts at: original_start + train_months + slide_months
            # Window N starts at: original_start + train_months + (N-1) * slide_months
            
            # Find the window that would be testing at our random start date
            # Test period starts at: train_start + train_months
            # So we need: original_start + train_months + (N-1) * slide_months + train_months = start_date
            # Solving for N: N = 1 + (start_date - original_start - 2*train_months) / slide_months
            
            if months_diff >= 2 * self.config.train_months:
                window_num = 1 + max(0, (months_diff - 2 * self.config.train_months) // self.config.slide_months)
                print(f"  Calculated starting window: {window_num} (based on {months_diff} months offset)")
            else:
                window_num = 1
                print(f"  Using window 1 (insufficient offset: {months_diff} months)")
        else:
            window_num = 1
            
        # For random start, automatically set a reasonable max_windows if not specified
        starting_window = window_num
        if random_start and max_windows is None:
            # Limit to 10 windows from the starting window to keep execution time reasonable
            max_windows = 10
            print(f"  Auto-setting max windows to {max_windows} for random start (starting from window {starting_window})")
        
        current_date = start_date
        windows_processed = 0
        
        while current_date < end_date:
            # Check max_windows limit (count windows processed, not absolute window number)
            if max_windows and windows_processed >= max_windows:
                print(f"  Reached maximum windows limit ({max_windows}), stopping...")
                break
                
            # Define training and testing periods
            train_start = current_date
            train_end = train_start + timedelta(days=30 * self.config.train_months)
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.config.test_months)
            
            if test_end > end_date:
                break
            
            print(f"  Window {window_num}: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
            
            # Get test data
            test_data = data[(data.index >= test_start) & (data.index < test_end)].copy()
            
            if len(test_data) < self.config.sequence_length + 1:
                current_date += timedelta(days=30 * self.config.slide_months)
                window_num += 1
                windows_processed += 1
                continue
            
            # Load models for this window
            lstm_model, xgb_model, scaler = self.load_models(symbol, window_num)
            
            if xgb_model is None:
                print(f"    XGBoost model not found for window {window_num}, skipping...")
                current_date += timedelta(days=30 * self.config.slide_months)
                window_num += 1
                windows_processed += 1
                continue
            
            # Skip window if either model fails to load (both models required)
            if lstm_model is None:
                print(f"    LSTM model failed to load for window {window_num}, skipping window...")
                current_date += timedelta(days=30 * self.config.slide_months)
                window_num += 1
                windows_processed += 1
                continue
            
            # Simulate trading for this window
            window_trades, window_capital = self.simulate_trading_window(
                test_data, lstm_model, xgb_model, scaler,
                symbol, capital, positions, window_num
            )
            
            trades.extend(window_trades)
            capital = window_capital
            equity_history.append({
                'date': test_end,
                'capital': capital,
                'window': window_num
            })
            
            # Move to next window
            current_date += timedelta(days=30 * self.config.slide_months)
            window_num += 1
            windows_processed += 1
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(trades, equity_history, symbol)
        
        return {
            'symbol': symbol,
            'trades': trades,
            'equity_history': equity_history,
            'performance': performance,
            'final_capital': capital
        }
    
    def run_fast_backtest(self, symbol: str, max_windows: int = None, random_start: bool = False) -> Dict:
        """Run optimized backtest for a single symbol"""
        print(f"\n{'='*60}")
        print(f"RUNNING OPTIMIZED BACKTEST FOR {symbol}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Override the model loading method temporarily
        original_load_models = self.load_models
        self.load_models = self.load_models_cached
        
        try:
            # Run backtest for single symbol with max_windows limit and random start
            results = self.backtest_symbol(symbol, max_windows, random_start)
            
            # Save results
            self.save_results(results)
            
            # Save debug CSV if debug mode is enabled
            if self.debug_mode:
                self.save_debug_csv(symbol)
            
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
    parser.add_argument('--config', default='balanced',
                       choices=['balanced', 'aggressive', 'conservative', 'very_aggressive', 'fast_test'],
                       help='Configuration preset')
    parser.add_argument('--max-windows', type=int, default=None,
                       help='Maximum number of windows to test (for quick testing)')
    parser.add_argument('--random', action='store_true',
                       help='Randomly select starting date from dataset')
    parser.add_argument('--save-plots', action='store_true',
                       help='Generate and save performance plots')
    parser.add_argument('--debug', action='store_true',
                       help='Generate detailed debug CSV file with trading signals and model outputs')
    
    args = parser.parse_args()
    
    print(f"Starting optimized backtest...")
    print(f"Symbol: {args.symbol}")
    print(f"Configuration: {args.config}")
    if args.max_windows:
        print(f"Max windows: {args.max_windows}")
    if args.random:
        print(f"Random start: enabled")
    if args.debug:
        print(f"Debug mode: enabled (will generate detailed CSV)")
    
    # Create configuration
    config = create_config(args.config)
    
    # Create optimized backtester
    backtester = OptimizedBacktester(config, debug_mode=args.debug)
    
    # Run backtest
    try:
        results = backtester.run_fast_backtest(args.symbol, args.max_windows, args.random)
        
        if args.save_plots:
            print("\nGenerating performance plots...")
            backtester.plot_results(results)
        
        print(f"\nResults saved to: backtests/{args.symbol}/")
        print("\nOptimization suggestions:")
        print("1. If no trades were executed, try 'very_aggressive' or 'fast_test' config")
        print("2. Use --max-windows 10 for quick testing (auto-limited for --random)")
        print("3. Use --random for testing different market conditions")
        print("4. Check equity_curve.csv for capital progression")
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())