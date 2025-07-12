#!/usr/bin/env python3
"""
Main script to run comprehensive backtesting of cryptocurrency trading models.

This script provides multiple testing modes:
1. Standard backtest with default configuration
2. Multi-configuration testing (conservative, balanced, aggressive)
3. Symbol-specific optimization
4. Bootstrap robustness testing
5. Sensitivity analysis

Usage:
    python run_backtest.py --mode standard
    python run_backtest.py --mode multi-config
    python run_backtest.py --mode bootstrap
    python run_backtest.py --mode sensitivity
    python run_backtest.py --mode all
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Dict
import json
import pandas as pd

# Import our modules
from scripts.backtest_models import ModelBacktester
from scripts.backtest_config import BacktestConfig, CONFIG_PRESETS, get_symbol_config
from scripts.backtest_analysis import BacktestAnalyzer
from scripts.bootstrap_backtest import BootstrapBacktester

class BacktestRunner:
    """Main class to orchestrate backtesting operations"""
    
    def __init__(self):
        self.symbols = ['ADAEUR', 'BTCEUR', 'ETHEUR', 'SOLEUR']
        self.results_dir = 'backtests'
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.results_dir,
            f'{self.results_dir}/configs',
            f'{self.results_dir}/bootstrap',
            f'{self.results_dir}/sensitivity',
            f'{self.results_dir}/multi_config'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_standard_backtest(self, symbols: List[str] = None, config_name: str = 'balanced') -> Dict:
        """Run standard backtest with specified configuration"""
        print(f"\n{'='*60}")
        print(f"RUNNING STANDARD BACKTEST - {config_name.upper()} CONFIG")
        print(f"{'='*60}")
        
        if symbols is None:
            symbols = self.symbols
        
        # Get configuration
        if config_name in CONFIG_PRESETS:
            config = CONFIG_PRESETS[config_name]
        else:
            config = BacktestConfig()
        
        # Save configuration
        config_file = f'{self.results_dir}/configs/{config_name}_config.json'
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"Configuration saved to: {config_file}")
        print(f"Testing symbols: {symbols}")
        
        # Run backtest
        backtester = ModelBacktester(config)
        start_time = time.time()
        
        results = backtester.run_backtest(symbols)
        
        end_time = time.time()
        print(f"\nBacktest completed in {end_time - start_time:.1f} seconds")
        
        return results
    
    def run_multi_config_backtest(self, symbols: List[str] = None) -> Dict:
        """Run backtest with multiple configurations for comparison"""
        print(f"\n{'='*60}")
        print("RUNNING MULTI-CONFIGURATION BACKTEST")
        print(f"{'='*60}")
        
        if symbols is None:
            symbols = self.symbols
        
        all_results = {}
        
        for config_name, config in CONFIG_PRESETS.items():
            print(f"\nTesting {config_name} configuration...")
            
            # Create subdirectory for this config
            config_dir = f'{self.results_dir}/multi_config/{config_name}'
            os.makedirs(config_dir, exist_ok=True)
            
            # Temporarily change results directory
            original_results_dir = self.results_dir
            self.results_dir = config_dir
            
            try:
                # Run backtest
                backtester = ModelBacktester(config)
                results = backtester.run_backtest(symbols)
                all_results[config_name] = results
                
                # Save configuration
                config_file = f'{config_dir}/config.json'
                with open(config_file, 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)
                
            except Exception as e:
                print(f"Error running {config_name} configuration: {e}")
                all_results[config_name] = {'error': str(e)}
            
            finally:
                # Restore original results directory
                self.results_dir = original_results_dir
        
        # Generate comparison report
        self.generate_multi_config_report(all_results)
        
        return all_results
    
    def run_symbol_specific_backtest(self, symbols: List[str] = None) -> Dict:
        """Run backtest with symbol-specific optimized configurations"""
        print(f"\n{'='*60}")
        print("RUNNING SYMBOL-SPECIFIC OPTIMIZED BACKTEST")
        print(f"{'='*60}")
        
        if symbols is None:
            symbols = self.symbols
        
        results = {}
        
        for symbol in symbols:
            print(f"\nTesting {symbol} with optimized configuration...")
            
            # Get symbol-specific configuration
            base_config = BacktestConfig()
            symbol_config = get_symbol_config(symbol, base_config)
            
            # Save configuration
            config_file = f'{self.results_dir}/configs/{symbol}_optimized_config.json'
            with open(config_file, 'w') as f:
                json.dump(symbol_config.to_dict(), f, indent=2)
            
            # Run backtest for this symbol only
            backtester = ModelBacktester(symbol_config)
            symbol_results = backtester.run_backtest([symbol])
            results[symbol] = symbol_results.get(symbol, {})
        
        return results
    
    def run_bootstrap_analysis(self, symbols: List[str] = None, n_bootstrap: int = 10) -> Dict:
        """Run bootstrap robustness analysis"""
        print(f"\n{'='*60}")
        print(f"RUNNING BOOTSTRAP ROBUSTNESS ANALYSIS ({n_bootstrap} runs)")
        print(f"{'='*60}")
        
        if symbols is None:
            # Use best performing symbols for bootstrap (computationally intensive)
            symbols = ['BTCEUR', 'ETHEUR']
        
        config = BacktestConfig()
        bootstrap_tester = BootstrapBacktester(config, n_bootstrap)
        
        results = bootstrap_tester.run_bootstrap_validation(symbols)
        
        # Generate plots and report
        bootstrap_tester.plot_bootstrap_results(results)
        report = bootstrap_tester.generate_robustness_report(results)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'{self.results_dir}/bootstrap/robustness_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Bootstrap analysis completed. Report saved to: {report_file}")
        
        return results
    
    def run_sensitivity_analysis(self, symbols: List[str] = None) -> Dict:
        """Run sensitivity analysis on key parameters"""
        print(f"\n{'='*60}")
        print("RUNNING SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        
        if symbols is None:
            symbols = ['BTCEUR']  # Use best performer for sensitivity analysis
        
        # Define parameter ranges to test
        sensitivity_params = {
            'buy_threshold': [0.6, 0.65, 0.7, 0.75, 0.8],
            'risk_per_trade': [0.01, 0.015, 0.02, 0.025, 0.03],
            'stop_loss_pct': [0.02, 0.025, 0.03, 0.035, 0.04],
            'max_positions': [5, 8, 10, 12, 15]
        }
        
        results = {}
        base_config = BacktestConfig()
        
        for param_name, param_values in sensitivity_params.items():
            print(f"\nTesting sensitivity for {param_name}...")
            param_results = []
            
            for value in param_values:
                print(f"  Testing {param_name} = {value}")
                
                # Create modified configuration
                config_dict = base_config.to_dict()
                config_dict[param_name] = value
                test_config = BacktestConfig.from_dict(config_dict)
                
                # Run backtest
                backtester = ModelBacktester(test_config)
                test_results = backtester.run_backtest(symbols)
                
                # Extract key metrics
                if symbols[0] in test_results:
                    performance = test_results[symbols[0]]['performance']
                    param_results.append({
                        'parameter_value': value,
                        'total_return': performance.get('total_return', 0),
                        'sharpe_ratio': performance.get('sharpe_ratio', 0),
                        'max_drawdown': performance.get('max_drawdown', 0),
                        'win_rate': performance.get('win_rate', 0),
                        'total_trades': performance.get('total_trades', 0)
                    })
            
            results[param_name] = param_results
        
        # Save sensitivity results
        self.save_sensitivity_results(results, symbols[0])
        
        return results
    
    def save_sensitivity_results(self, results: Dict, symbol: str):
        """Save sensitivity analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = f'{self.results_dir}/sensitivity/sensitivity_{symbol}_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary CSV for each parameter
        for param_name, param_data in results.items():
            if param_data:
                df = pd.DataFrame(param_data)
                csv_file = f'{self.results_dir}/sensitivity/{param_name}_sensitivity_{symbol}_{timestamp}.csv'
                df.to_csv(csv_file, index=False)
        
        print(f"Sensitivity results saved to: {results_file}")
    
    def generate_multi_config_report(self, results: Dict):
        """Generate comparison report for multi-configuration backtest"""
        print(f"\n{'='*60}")
        print("MULTI-CONFIGURATION COMPARISON REPORT")
        print(f"{'='*60}")
        
        # Create comparison table
        comparison_data = []
        
        for config_name, config_results in results.items():
            if 'error' in config_results:
                print(f"\n{config_name}: ERROR - {config_results['error']}")
                continue
            
            # Aggregate metrics across all symbols
            total_return = 0
            total_trades = 0
            avg_sharpe = 0
            avg_win_rate = 0
            max_drawdown = 0
            symbol_count = 0
            
            for symbol, symbol_data in config_results.items():
                if 'performance' in symbol_data:
                    perf = symbol_data['performance']
                    total_return += perf.get('total_return', 0)
                    total_trades += perf.get('total_trades', 0)
                    avg_sharpe += perf.get('sharpe_ratio', 0)
                    avg_win_rate += perf.get('win_rate', 0)
                    max_drawdown = min(max_drawdown, perf.get('max_drawdown', 0))
                    symbol_count += 1
            
            if symbol_count > 0:
                comparison_data.append({
                    'Configuration': config_name,
                    'Avg_Return': total_return / symbol_count,
                    'Total_Trades': total_trades,
                    'Avg_Sharpe': avg_sharpe / symbol_count,
                    'Avg_Win_Rate': avg_win_rate / symbol_count,
                    'Worst_Drawdown': max_drawdown,
                    'Symbols_Tested': symbol_count
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Sort by average return
            comparison_df = comparison_df.sort_values('Avg_Return', ascending=False)
            
            print("\nConfiguration Performance Ranking:")
            print(comparison_df.to_string(index=False, float_format='%.4f'))
            
            # Save comparison
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            comparison_file = f'{self.results_dir}/multi_config/comparison_{timestamp}.csv'
            comparison_df.to_csv(comparison_file, index=False)
            
            print(f"\nComparison saved to: {comparison_file}")
            
            # Recommendations
            best_config = comparison_df.iloc[0]['Configuration']
            print(f"\nRecommendation: {best_config} configuration shows best overall performance")
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis of existing results"""
        print(f"\n{'='*60}")
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print(f"{'='*60}")
        
        analyzer = BacktestAnalyzer(self.results_dir)
        
        if not analyzer.symbols:
            print("No backtest results found. Please run a backtest first.")
            return
        
        print(f"Analyzing results for: {analyzer.symbols}")
        
        # Generate summary report
        summary = analyzer.generate_summary_report()
        print("\nPerformance Summary:")
        print(summary.to_string(index=False))
        
        # Analyze trade patterns
        analyzer.analyze_trade_patterns()
        
        # Create visualizations
        analyzer.plot_performance_comparison()
        
        # Generate detailed reports
        analyzer.generate_detailed_report()
        analyzer.export_results_to_excel()
        
        print("\nComprehensive analysis completed!")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Model Backtester')
    parser.add_argument('--mode', choices=['standard', 'multi-config', 'symbol-specific', 
                                          'bootstrap', 'sensitivity', 'analysis', 'all'],
                       default='standard', help='Backtesting mode')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to test (default: all available)')
    parser.add_argument('--config', default='balanced',
                       choices=['conservative', 'balanced', 'aggressive', 'high_frequency'],
                       help='Configuration preset for standard mode')
    parser.add_argument('--bootstrap-runs', type=int, default=10,
                       help='Number of bootstrap runs (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BacktestRunner()
    
    print(f"Cryptocurrency Trading Model Backtester")
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        if args.mode == 'standard':
            results = runner.run_standard_backtest(args.symbols, args.config)
            
        elif args.mode == 'multi-config':
            results = runner.run_multi_config_backtest(args.symbols)
            
        elif args.mode == 'symbol-specific':
            results = runner.run_symbol_specific_backtest(args.symbols)
            
        elif args.mode == 'bootstrap':
            results = runner.run_bootstrap_analysis(args.symbols, args.bootstrap_runs)
            
        elif args.mode == 'sensitivity':
            results = runner.run_sensitivity_analysis(args.symbols)
            
        elif args.mode == 'analysis':
            runner.run_comprehensive_analysis()
            results = {}
            
        elif args.mode == 'all':
            print("Running comprehensive backtesting suite...")
            
            # 1. Standard backtest
            runner.run_standard_backtest(args.symbols, 'balanced')
            
            # 2. Multi-configuration
            runner.run_multi_config_backtest(args.symbols)
            
            # 3. Symbol-specific
            runner.run_symbol_specific_backtest(args.symbols)
            
            # 4. Bootstrap (limited runs for time)
            runner.run_bootstrap_analysis(['BTCEUR'], 5)
            
            # 5. Sensitivity analysis
            runner.run_sensitivity_analysis(['BTCEUR'])
            
            # 6. Comprehensive analysis
            runner.run_comprehensive_analysis()
            
            results = {}
        
        end_time = time.time()
        print(f"\n{'='*60}")
        print(f"BACKTESTING COMPLETED")
        print(f"{'='*60}")
        print(f"Total execution time: {end_time - start_time:.1f} seconds")
        print(f"Results saved in: {runner.results_dir}")
        
        # Final recommendations
        if args.mode in ['standard', 'multi-config', 'symbol-specific']:
            print("\nNext steps:")
            print("1. Review the generated plots and reports")
            print("2. Run 'python run_backtest.py --mode analysis' for detailed analysis")
            print("3. Consider bootstrap testing for robustness validation")
            print("4. Implement the best-performing configuration in live trading")
        
    except KeyboardInterrupt:
        print("\nBacktesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during backtesting: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()