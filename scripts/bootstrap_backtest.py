import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
from .backtest_models import ModelBacktester, Trade
from .backtest_config import BacktestConfig
import multiprocessing as mp

warnings.filterwarnings('ignore')

class BootstrapBacktester:
    """Bootstrap validation for backtesting robustness"""
    
    def __init__(self, config: BacktestConfig, n_bootstrap: int = 10):
        self.config = config
        self.n_bootstrap = n_bootstrap
        self.results = {}
        
        # Create bootstrap results directory
        os.makedirs('backtests/bootstrap', exist_ok=True)
    
    def add_noise_to_data(self, data: pd.DataFrame, noise_level: float = 0.001, seed: int = None) -> pd.DataFrame:
        """Add random noise to price data for bootstrap validation"""
        if seed is not None:
            np.random.seed(seed)
        
        noisy_data = data.copy()
        
        # Add noise to OHLC prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in noisy_data.columns:
                noise = np.random.normal(0, noise_level, len(noisy_data))
                noisy_data[col] = noisy_data[col] * (1 + noise)
        
        # Ensure OHLC consistency after adding noise
        noisy_data['high'] = noisy_data[['open', 'high', 'low', 'close']].max(axis=1)
        noisy_data['low'] = noisy_data[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Add noise to volume (smaller amount)
        if 'volume' in noisy_data.columns:
            volume_noise = np.random.normal(0, noise_level * 0.5, len(noisy_data))
            noisy_data['volume'] = noisy_data['volume'] * (1 + volume_noise)
            noisy_data['volume'] = noisy_data['volume'].clip(lower=0)  # Ensure positive volume
        
        return noisy_data
    
    def run_single_bootstrap(self, symbol: str, run_id: int, noise_level: float) -> Dict:
        """Run a single bootstrap iteration"""
        try:
            print(f"  Bootstrap run {run_id + 1}/{self.n_bootstrap} for {symbol}")
            
            # Create a new backtester instance for this run
            backtester = ModelBacktester(self.config)
            
            # Load original data
            original_data = backtester.load_data(symbol)
            
            # Add noise
            noisy_data = self.add_noise_to_data(original_data, noise_level, seed=run_id)
            
            # Temporarily replace the load_data method to return noisy data
            def load_noisy_data(sym):
                if sym == symbol:
                    return noisy_data
                return backtester.load_data(sym)
            
            backtester.load_data = load_noisy_data
            
            # Run backtest
            result = backtester.backtest_symbol(symbol)
            
            # Extract key metrics
            performance = result['performance']
            
            return {
                'run_id': run_id,
                'symbol': symbol,
                'total_return': performance.get('total_return', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'win_rate': performance.get('win_rate', 0),
                'total_trades': performance.get('total_trades', 0),
                'profit_factor': performance.get('profit_factor', 0),
                'total_pnl': performance.get('total_pnl', 0)
            }
            
        except Exception as e:
            print(f"    Error in bootstrap run {run_id} for {symbol}: {e}")
            return {
                'run_id': run_id,
                'symbol': symbol,
                'error': str(e)
            }
    
    def run_bootstrap_validation(self, symbols: List[str] = None, noise_level: float = 0.001) -> Dict:
        """Run bootstrap validation for robustness testing"""
        if symbols is None:
            symbols = ['ADAEUR', 'BTCEUR', 'ETHEUR', 'SOLEUR']
        
        print(f"Starting bootstrap validation with {self.n_bootstrap} runs...")
        print(f"Noise level: {noise_level * 100:.1f}%")
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\nRunning bootstrap for {symbol}...")
            
            symbol_results = []
            
            # Run bootstrap iterations
            for run_id in range(self.n_bootstrap):
                result = self.run_single_bootstrap(symbol, run_id, noise_level)
                symbol_results.append(result)
            
            # Filter out error results
            valid_results = [r for r in symbol_results if 'error' not in r]
            error_results = [r for r in symbol_results if 'error' in r]
            
            if error_results:
                print(f"  {len(error_results)} runs failed for {symbol}")
            
            if valid_results:
                # Calculate statistics
                stats = self.calculate_bootstrap_statistics(valid_results)
                all_results[symbol] = {
                    'individual_runs': valid_results,
                    'statistics': stats,
                    'error_count': len(error_results)
                }
                
                # Print summary
                print(f"  Completed {len(valid_results)} successful runs")
                print(f"  Mean return: {stats['total_return']['mean']:.2%} ± {stats['total_return']['std']:.2%}")
                print(f"  Mean Sharpe: {stats['sharpe_ratio']['mean']:.2f} ± {stats['sharpe_ratio']['std']:.2f}")
            else:
                print(f"  No successful runs for {symbol}")
                all_results[symbol] = {
                    'individual_runs': [],
                    'statistics': {},
                    'error_count': len(error_results)
                }
        
        # Save results
        self.save_bootstrap_results(all_results, noise_level)
        
        return all_results
    
    def calculate_bootstrap_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics from bootstrap results"""
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                  'total_trades', 'profit_factor', 'total_pnl']
        
        stats = {}
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and r[metric] is not None]
            
            if values:
                # Handle infinite values
                finite_values = [v for v in values if np.isfinite(v)]
                
                if finite_values:
                    stats[metric] = {
                        'mean': np.mean(finite_values),
                        'std': np.std(finite_values),
                        'min': np.min(finite_values),
                        'max': np.max(finite_values),
                        'median': np.median(finite_values),
                        'q25': np.percentile(finite_values, 25),
                        'q75': np.percentile(finite_values, 75),
                        'count': len(finite_values)
                    }
                else:
                    stats[metric] = {'count': 0}
            else:
                stats[metric] = {'count': 0}
        
        return stats
    
    def save_bootstrap_results(self, results: Dict, noise_level: float):
        """Save bootstrap results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = f'backtests/bootstrap/bootstrap_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self.convert_numpy_types(results)
            json.dump({
                'timestamp': timestamp,
                'noise_level': noise_level,
                'n_bootstrap': self.n_bootstrap,
                'results': json_results
            }, f, indent=2)
        
        # Save summary statistics
        summary_data = []
        for symbol, data in results.items():
            if data['statistics']:
                stats = data['statistics']
                summary_data.append({
                    'Symbol': symbol,
                    'Successful_Runs': stats.get('total_return', {}).get('count', 0),
                    'Error_Count': data['error_count'],
                    'Mean_Return': stats.get('total_return', {}).get('mean', 0),
                    'Std_Return': stats.get('total_return', {}).get('std', 0),
                    'Mean_Sharpe': stats.get('sharpe_ratio', {}).get('mean', 0),
                    'Std_Sharpe': stats.get('sharpe_ratio', {}).get('std', 0),
                    'Mean_Drawdown': stats.get('max_drawdown', {}).get('mean', 0),
                    'Std_Drawdown': stats.get('max_drawdown', {}).get('std', 0),
                    'Mean_WinRate': stats.get('win_rate', {}).get('mean', 0),
                    'Std_WinRate': stats.get('win_rate', {}).get('std', 0)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = f'backtests/bootstrap/bootstrap_summary_{timestamp}.csv'
            summary_df.to_csv(summary_file, index=False)
            
            print(f"\nBootstrap results saved:")
            print(f"  Detailed: {results_file}")
            print(f"  Summary: {summary_file}")
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def plot_bootstrap_results(self, results: Dict, save_path: str = None):
        """Create visualization of bootstrap results"""
        symbols = list(results.keys())
        n_symbols = len(symbols)
        
        if n_symbols == 0:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bootstrap Validation Results', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # 1. Return Distribution
        ax1 = axes[0]
        return_data = []
        labels = []
        
        for symbol in symbols:
            if results[symbol]['individual_runs']:
                returns = [r['total_return'] * 100 for r in results[symbol]['individual_runs'] 
                          if 'total_return' in r and np.isfinite(r['total_return'])]
                if returns:
                    return_data.append(returns)
                    labels.append(symbol)
        
        if return_data:
            ax1.boxplot(return_data, labels=labels)
            ax1.set_title('Total Return Distribution (%)')
            ax1.set_ylabel('Return (%)')
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio Distribution
        ax2 = axes[1]
        sharpe_data = []
        sharpe_labels = []
        
        for symbol in symbols:
            if results[symbol]['individual_runs']:
                sharpes = [r['sharpe_ratio'] for r in results[symbol]['individual_runs'] 
                          if 'sharpe_ratio' in r and np.isfinite(r['sharpe_ratio'])]
                if sharpes:
                    sharpe_data.append(sharpes)
                    sharpe_labels.append(symbol)
        
        if sharpe_data:
            ax2.boxplot(sharpe_data, labels=sharpe_labels)
            ax2.set_title('Sharpe Ratio Distribution')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
        
        # 3. Win Rate Distribution
        ax3 = axes[2]
        winrate_data = []
        winrate_labels = []
        
        for symbol in symbols:
            if results[symbol]['individual_runs']:
                winrates = [r['win_rate'] * 100 for r in results[symbol]['individual_runs'] 
                           if 'win_rate' in r and np.isfinite(r['win_rate'])]
                if winrates:
                    winrate_data.append(winrates)
                    winrate_labels.append(symbol)
        
        if winrate_data:
            ax3.boxplot(winrate_data, labels=winrate_labels)
            ax3.set_title('Win Rate Distribution (%)')
            ax3.set_ylabel('Win Rate (%)')
            ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3)
        
        # 4. Stability Analysis (Coefficient of Variation)
        ax4 = axes[3]
        stability_data = []
        stability_labels = []
        
        for symbol in symbols:
            stats = results[symbol]['statistics']
            if 'total_return' in stats and stats['total_return'].get('count', 0) > 0:
                mean_return = stats['total_return']['mean']
                std_return = stats['total_return']['std']
                if mean_return != 0:
                    cv = abs(std_return / mean_return)  # Coefficient of variation
                    stability_data.append(cv)
                    stability_labels.append(symbol)
        
        if stability_data:
            bars = ax4.bar(stability_labels, stability_data, alpha=0.7)
            ax4.set_title('Return Stability (Lower = More Stable)')
            ax4.set_ylabel('Coefficient of Variation')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, stability_data):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height * 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f'backtests/bootstrap/bootstrap_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        print("Bootstrap plots saved")
    
    def generate_robustness_report(self, results: Dict) -> str:
        """Generate a robustness analysis report"""
        report = []
        report.append("=" * 60)
        report.append("BOOTSTRAP ROBUSTNESS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Number of bootstrap runs: {self.n_bootstrap}")
        report.append(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for symbol, data in results.items():
            report.append(f"\n{symbol} Analysis:")
            report.append("-" * 30)
            
            if not data['statistics']:
                report.append("  No successful runs")
                continue
            
            stats = data['statistics']
            
            # Return analysis
            if 'total_return' in stats and stats['total_return'].get('count', 0) > 0:
                ret_stats = stats['total_return']
                report.append(f"  Total Return:")
                report.append(f"    Mean: {ret_stats['mean']:.2%} ± {ret_stats['std']:.2%}")
                report.append(f"    Range: [{ret_stats['min']:.2%}, {ret_stats['max']:.2%}]")
                report.append(f"    Median: {ret_stats['median']:.2%}")
                
                # Stability assessment
                cv = abs(ret_stats['std'] / ret_stats['mean']) if ret_stats['mean'] != 0 else float('inf')
                if cv < 0.5:
                    stability = "High"
                elif cv < 1.0:
                    stability = "Medium"
                else:
                    stability = "Low"
                report.append(f"    Stability: {stability} (CV: {cv:.2f})")
            
            # Sharpe ratio analysis
            if 'sharpe_ratio' in stats and stats['sharpe_ratio'].get('count', 0) > 0:
                sharpe_stats = stats['sharpe_ratio']
                report.append(f"  Sharpe Ratio:")
                report.append(f"    Mean: {sharpe_stats['mean']:.2f} ± {sharpe_stats['std']:.2f}")
                report.append(f"    Range: [{sharpe_stats['min']:.2f}, {sharpe_stats['max']:.2f}]")
            
            # Win rate analysis
            if 'win_rate' in stats and stats['win_rate'].get('count', 0) > 0:
                wr_stats = stats['win_rate']
                report.append(f"  Win Rate:")
                report.append(f"    Mean: {wr_stats['mean']:.1%} ± {wr_stats['std']:.1%}")
                report.append(f"    Range: [{wr_stats['min']:.1%}, {wr_stats['max']:.1%}]")
            
            # Success rate
            total_runs = data['statistics'].get('total_return', {}).get('count', 0) + data['error_count']
            success_rate = (data['statistics'].get('total_return', {}).get('count', 0) / total_runs * 100) if total_runs > 0 else 0
            report.append(f"  Success Rate: {success_rate:.1f}% ({data['statistics'].get('total_return', {}).get('count', 0)}/{total_runs} runs)")
        
        # Overall assessment
        report.append("\n" + "=" * 60)
        report.append("OVERALL ROBUSTNESS ASSESSMENT")
        report.append("=" * 60)
        
        # Calculate overall metrics
        all_returns = []
        all_sharpes = []
        success_rates = []
        
        for symbol, data in results.items():
            if data['individual_runs']:
                returns = [r['total_return'] for r in data['individual_runs'] 
                          if 'total_return' in r and np.isfinite(r['total_return'])]
                sharpes = [r['sharpe_ratio'] for r in data['individual_runs'] 
                          if 'sharpe_ratio' in r and np.isfinite(r['sharpe_ratio'])]
                
                all_returns.extend(returns)
                all_sharpes.extend(sharpes)
                
                total_runs = len(data['individual_runs']) + data['error_count']
                if total_runs > 0:
                    success_rates.append(len(data['individual_runs']) / total_runs)
        
        if all_returns:
            positive_returns = sum(1 for r in all_returns if r > 0)
            report.append(f"Positive return rate: {positive_returns/len(all_returns):.1%}")
            report.append(f"Average return across all runs: {np.mean(all_returns):.2%}")
        
        if all_sharpes:
            positive_sharpes = sum(1 for s in all_sharpes if s > 0)
            report.append(f"Positive Sharpe rate: {positive_sharpes/len(all_sharpes):.1%}")
        
        if success_rates:
            report.append(f"Average success rate: {np.mean(success_rates):.1%}")
        
        return "\n".join(report)

def run_bootstrap_analysis(symbols: List[str] = None, n_bootstrap: int = 10, noise_level: float = 0.001):
    """Main function to run bootstrap analysis"""
    config = BacktestConfig()
    bootstrap_tester = BootstrapBacktester(config, n_bootstrap)
    
    print("Starting bootstrap robustness analysis...")
    results = bootstrap_tester.run_bootstrap_validation(symbols, noise_level)
    
    # Generate plots
    bootstrap_tester.plot_bootstrap_results(results)
    
    # Generate report
    report = bootstrap_tester.generate_robustness_report(results)
    print("\n" + report)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'backtests/bootstrap/robustness_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nRobustness report saved to: {report_file}")
    
    return results

if __name__ == "__main__":
    # Run bootstrap analysis
    results = run_bootstrap_analysis(
        symbols=['BTCEUR', 'ETHEUR'],  # Start with best performers
        n_bootstrap=5,  # Reduced for testing
        noise_level=0.001
    )