import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

class BacktestAnalyzer:
    """Comprehensive analysis of backtest results"""
    
    def __init__(self, results_dir: str = 'backtests'):
        self.results_dir = results_dir
        self.symbols = self._get_available_symbols()
        self.results = self._load_all_results()
    
    def _get_available_symbols(self) -> List[str]:
        """Get list of symbols with backtest results"""
        symbols = []
        if os.path.exists(self.results_dir):
            for item in os.listdir(self.results_dir):
                symbol_dir = os.path.join(self.results_dir, item)
                if os.path.isdir(symbol_dir):
                    # Check if required files exist
                    if (os.path.exists(os.path.join(symbol_dir, 'performance_metrics.json')) and
                        os.path.exists(os.path.join(symbol_dir, 'trades.csv'))):
                        symbols.append(item)
        return symbols
    
    def _load_all_results(self) -> Dict:
        """Load all backtest results"""
        results = {}
        
        for symbol in self.symbols:
            try:
                symbol_dir = os.path.join(self.results_dir, symbol)
                
                # Load performance metrics
                with open(os.path.join(symbol_dir, 'performance_metrics.json'), 'r') as f:
                    performance = json.load(f)
                
                # Load trades
                trades_file = os.path.join(symbol_dir, 'trades.csv')
                if os.path.exists(trades_file):
                    trades = pd.read_csv(trades_file)
                    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
                    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
                else:
                    trades = pd.DataFrame()
                
                # Load equity curve
                equity_file = os.path.join(symbol_dir, 'equity_curve.csv')
                if os.path.exists(equity_file):
                    equity = pd.read_csv(equity_file)
                    equity['date'] = pd.to_datetime(equity['date'])
                else:
                    equity = pd.DataFrame()
                
                results[symbol] = {
                    'performance': performance,
                    'trades': trades,
                    'equity': equity
                }
                
            except Exception as e:
                print(f"Error loading results for {symbol}: {e}")
                continue
        
        return results
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate summary report comparing all symbols"""
        summary_data = []
        
        for symbol, data in self.results.items():
            perf = data['performance']
            trades = data['trades']
            
            # Calculate additional metrics
            if not trades.empty:
                avg_trade_duration = self._calculate_avg_trade_duration(trades)
                monthly_trades = self._calculate_monthly_trade_frequency(trades)
                largest_win = trades['pnl'].max() if 'pnl' in trades.columns else 0
                largest_loss = trades['pnl'].min() if 'pnl' in trades.columns else 0
                consecutive_wins = self._calculate_max_consecutive_wins(trades)
                consecutive_losses = self._calculate_max_consecutive_losses(trades)
            else:
                avg_trade_duration = 0
                monthly_trades = 0
                largest_win = 0
                largest_loss = 0
                consecutive_wins = 0
                consecutive_losses = 0
            
            summary_data.append({
                'Symbol': symbol,
                'Total Trades': perf.get('total_trades', 0),
                'Win Rate (%)': perf.get('win_rate', 0) * 100,
                'Total PnL': perf.get('total_pnl', 0),
                'Total Return (%)': perf.get('total_return', 0) * 100,
                'Sharpe Ratio': perf.get('sharpe_ratio', 0),
                'Sortino Ratio': perf.get('sortino_ratio', 0),
                'Max Drawdown (%)': perf.get('max_drawdown', 0) * 100,
                'Profit Factor': perf.get('profit_factor', 0),
                'Avg Win': perf.get('avg_win', 0),
                'Avg Loss': perf.get('avg_loss', 0),
                'Largest Win': largest_win,
                'Largest Loss': largest_loss,
                'Avg Trade Duration (hours)': avg_trade_duration,
                'Monthly Trade Frequency': monthly_trades,
                'Max Consecutive Wins': consecutive_wins,
                'Max Consecutive Losses': consecutive_losses
            })
        
        return pd.DataFrame(summary_data)
    
    def _calculate_avg_trade_duration(self, trades: pd.DataFrame) -> float:
        """Calculate average trade duration in hours"""
        if trades.empty or 'entry_time' not in trades.columns or 'exit_time' not in trades.columns:
            return 0
        
        valid_trades = trades.dropna(subset=['entry_time', 'exit_time'])
        if valid_trades.empty:
            return 0
        
        durations = (valid_trades['exit_time'] - valid_trades['entry_time']).dt.total_seconds() / 3600
        return durations.mean()
    
    def _calculate_monthly_trade_frequency(self, trades: pd.DataFrame) -> float:
        """Calculate average number of trades per month"""
        if trades.empty or 'entry_time' not in trades.columns:
            return 0
        
        trades_with_time = trades.dropna(subset=['entry_time'])
        if trades_with_time.empty:
            return 0
        
        start_date = trades_with_time['entry_time'].min()
        end_date = trades_with_time['entry_time'].max()
        
        if pd.isna(start_date) or pd.isna(end_date):
            return 0
        
        months = (end_date - start_date).days / 30.44  # Average days per month
        if months == 0:
            return len(trades_with_time)
        
        return len(trades_with_time) / months
    
    def _calculate_max_consecutive_wins(self, trades: pd.DataFrame) -> int:
        """Calculate maximum consecutive winning trades"""
        if trades.empty or 'pnl' not in trades.columns:
            return 0
        
        wins = (trades['pnl'] > 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for win in wins:
            if win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_losses(self, trades: pd.DataFrame) -> int:
        """Calculate maximum consecutive losing trades"""
        if trades.empty or 'pnl' not in trades.columns:
            return 0
        
        losses = (trades['pnl'] <= 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def plot_performance_comparison(self, save_path: str = None):
        """Create comprehensive performance comparison plots"""
        if not self.results:
            print("No results to plot")
            return
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Equity Curves Comparison
        ax1 = plt.subplot(3, 3, 1)
        for symbol, data in self.results.items():
            if not data['equity'].empty:
                equity = data['equity']
                ax1.plot(equity['date'], equity['capital'], label=symbol, linewidth=2)
        ax1.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Total Returns Comparison
        ax2 = plt.subplot(3, 3, 2)
        symbols = list(self.results.keys())
        returns = [self.results[s]['performance'].get('total_return', 0) * 100 for s in symbols]
        colors = ['green' if r > 0 else 'red' for r in returns]
        bars = ax2.bar(symbols, returns, color=colors, alpha=0.7)
        ax2.set_title('Total Returns Comparison (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 3. Sharpe Ratio Comparison
        ax3 = plt.subplot(3, 3, 3)
        sharpe_ratios = [self.results[s]['performance'].get('sharpe_ratio', 0) for s in symbols]
        colors = ['green' if r > 0 else 'red' for r in sharpe_ratios]
        bars = ax3.bar(symbols, sharpe_ratios, color=colors, alpha=0.7)
        ax3.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 4. Win Rate Comparison
        ax4 = plt.subplot(3, 3, 4)
        win_rates = [self.results[s]['performance'].get('win_rate', 0) * 100 for s in symbols]
        bars = ax4.bar(symbols, win_rates, color='skyblue', alpha=0.7)
        ax4.set_title('Win Rate Comparison (%)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_ylim(0, 100)
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add value labels
        for bar, value in zip(bars, win_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 5. Max Drawdown Comparison
        ax5 = plt.subplot(3, 3, 5)
        drawdowns = [abs(self.results[s]['performance'].get('max_drawdown', 0)) * 100 for s in symbols]
        bars = ax5.bar(symbols, drawdowns, color='red', alpha=0.7)
        ax5.set_title('Maximum Drawdown (%)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Max Drawdown (%)')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, drawdowns):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 6. Trade Count Comparison
        ax6 = plt.subplot(3, 3, 6)
        trade_counts = [self.results[s]['performance'].get('total_trades', 0) for s in symbols]
        bars = ax6.bar(symbols, trade_counts, color='orange', alpha=0.7)
        ax6.set_title('Total Trades Comparison', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Number of Trades')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, trade_counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + max(trade_counts) * 0.01,
                    f'{int(value)}', ha='center', va='bottom')
        
        # 7. Profit Factor Comparison
        ax7 = plt.subplot(3, 3, 7)
        profit_factors = [self.results[s]['performance'].get('profit_factor', 0) for s in symbols]
        # Cap extremely high values for better visualization
        profit_factors = [min(pf, 10) if pf != float('inf') else 10 for pf in profit_factors]
        colors = ['green' if pf > 1 else 'red' for pf in profit_factors]
        bars = ax7.bar(symbols, profit_factors, color=colors, alpha=0.7)
        ax7.set_title('Profit Factor Comparison', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Profit Factor')
        ax7.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Breakeven')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # Add value labels
        for bar, value in zip(bars, profit_factors):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 8. Risk-Return Scatter Plot
        ax8 = plt.subplot(3, 3, 8)
        returns_data = [self.results[s]['performance'].get('total_return', 0) * 100 for s in symbols]
        drawdowns_data = [abs(self.results[s]['performance'].get('max_drawdown', 0)) * 100 for s in symbols]
        
        scatter = ax8.scatter(drawdowns_data, returns_data, s=100, alpha=0.7, c=range(len(symbols)), cmap='viridis')
        
        # Add symbol labels
        for i, symbol in enumerate(symbols):
            ax8.annotate(symbol, (drawdowns_data[i], returns_data[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax8.set_title('Risk vs Return', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Max Drawdown (%)')
        ax8.set_ylabel('Total Return (%)')
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax8.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax8.grid(True, alpha=0.3)
        
        # 9. Monthly PnL Distribution (for best performer)
        ax9 = plt.subplot(3, 3, 9)
        best_symbol = max(symbols, key=lambda s: self.results[s]['performance'].get('total_return', 0))
        best_trades = self.results[best_symbol]['trades']
        
        if not best_trades.empty and 'exit_time' in best_trades.columns and 'pnl' in best_trades.columns:
            best_trades_clean = best_trades.dropna(subset=['exit_time', 'pnl'])
            if not best_trades_clean.empty:
                best_trades_clean['month'] = pd.to_datetime(best_trades_clean['exit_time']).dt.to_period('M')
                monthly_pnl = best_trades_clean.groupby('month')['pnl'].sum()
                
                colors = ['green' if pnl > 0 else 'red' for pnl in monthly_pnl.values]
                bars = ax9.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors, alpha=0.7)
                ax9.set_title(f'Monthly PnL - {best_symbol} (Best Performer)', fontsize=14, fontweight='bold')
                ax9.set_xlabel('Month')
                ax9.set_ylabel('PnL')
                ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax9.grid(True, alpha=0.3)
                
                # Set x-axis labels
                ax9.set_xticks(range(len(monthly_pnl)))
                ax9.set_xticklabels([str(period) for period in monthly_pnl.index], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to {save_path}")
        else:
            plt.savefig('backtests/performance_comparison.png', dpi=300, bbox_inches='tight')
            print("Performance comparison plot saved to backtests/performance_comparison.png")
        
        plt.show()
    
    def analyze_trade_patterns(self, symbol: str = None):
        """Analyze trading patterns for a specific symbol or all symbols"""
        if symbol and symbol in self.results:
            symbols_to_analyze = [symbol]
        else:
            symbols_to_analyze = list(self.results.keys())
        
        for sym in symbols_to_analyze:
            print(f"\n=== Trade Pattern Analysis for {sym} ===")
            
            trades = self.results[sym]['trades']
            if trades.empty:
                print("No trades found")
                continue
            
            # Time-based analysis
            if 'entry_time' in trades.columns:
                trades_with_time = trades.dropna(subset=['entry_time'])
                if not trades_with_time.empty:
                    trades_with_time['hour'] = pd.to_datetime(trades_with_time['entry_time']).dt.hour
                    trades_with_time['day_of_week'] = pd.to_datetime(trades_with_time['entry_time']).dt.dayofweek
                    
                    print("\nTrades by Hour of Day:")
                    hourly_trades = trades_with_time.groupby('hour').size()
                    for hour, count in hourly_trades.items():
                        print(f"  {hour:02d}:00 - {count} trades")
                    
                    print("\nTrades by Day of Week:")
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    daily_trades = trades_with_time.groupby('day_of_week').size()
                    for day, count in daily_trades.items():
                        print(f"  {day_names[day]} - {count} trades")
            
            # Direction analysis
            if 'direction' in trades.columns:
                direction_counts = trades['direction'].value_counts()
                print(f"\nTrade Direction Distribution:")
                for direction, count in direction_counts.items():
                    print(f"  {direction}: {count} trades ({count/len(trades)*100:.1f}%)")
            
            # PnL analysis
            if 'pnl' in trades.columns:
                pnl_data = trades['pnl'].dropna()
                if not pnl_data.empty:
                    print(f"\nPnL Statistics:")
                    print(f"  Mean PnL: {pnl_data.mean():.2f}")
                    print(f"  Median PnL: {pnl_data.median():.2f}")
                    print(f"  Std Dev: {pnl_data.std():.2f}")
                    print(f"  Skewness: {stats.skew(pnl_data):.2f}")
                    print(f"  Kurtosis: {stats.kurtosis(pnl_data):.2f}")
                    
                    # Percentiles
                    percentiles = [10, 25, 75, 90]
                    print(f"\nPnL Percentiles:")
                    for p in percentiles:
                        print(f"  {p}th percentile: {np.percentile(pnl_data, p):.2f}")
            
            # Confidence analysis
            if 'confidence' in trades.columns:
                conf_data = trades['confidence'].dropna()
                if not conf_data.empty:
                    print(f"\nConfidence Statistics:")
                    print(f"  Mean Confidence: {conf_data.mean():.3f}")
                    print(f"  Median Confidence: {conf_data.median():.3f}")
                    
                    # Confidence vs PnL correlation
                    if 'pnl' in trades.columns:
                        valid_data = trades.dropna(subset=['confidence', 'pnl'])
                        if len(valid_data) > 1:
                            correlation = valid_data['confidence'].corr(valid_data['pnl'])
                            print(f"  Confidence-PnL Correlation: {correlation:.3f}")
    
    def generate_detailed_report(self, output_file: str = 'backtests/detailed_analysis_report.html'):
        """Generate a detailed HTML report"""
        summary_df = self.generate_summary_report()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .neutral {{ color: #666; }}
                .summary-box {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Cryptocurrency Trading Model Backtest Analysis</h1>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary-box">
                <h2>Executive Summary</h2>
                <p>This report analyzes the performance of hybrid LSTM-XGBoost trading models across {len(self.symbols)} cryptocurrency pairs.</p>
                <ul>
                    <li><strong>Best Performer:</strong> {summary_df.loc[summary_df['Total Return (%)'].idxmax(), 'Symbol'] if not summary_df.empty else 'N/A'}</li>
                    <li><strong>Highest Sharpe Ratio:</strong> {summary_df.loc[summary_df['Sharpe Ratio'].idxmax(), 'Symbol'] if not summary_df.empty else 'N/A'}</li>
                    <li><strong>Most Active:</strong> {summary_df.loc[summary_df['Total Trades'].idxmax(), 'Symbol'] if not summary_df.empty else 'N/A'}</li>
                </ul>
            </div>
            
            <h2>Performance Summary</h2>
            {summary_df.to_html(classes='summary-table', escape=False, index=False)}
            
            <h2>Key Insights</h2>
        """
        
        # Add insights based on data
        if not summary_df.empty:
            best_return = summary_df['Total Return (%)'].max()
            worst_return = summary_df['Total Return (%)'].min()
            avg_win_rate = summary_df['Win Rate (%)'].mean()
            
            html_content += f"""
            <ul>
                <li>Best performing model achieved {best_return:.1f}% return</li>
                <li>Worst performing model had {worst_return:.1f}% return</li>
                <li>Average win rate across all models: {avg_win_rate:.1f}%</li>
                <li>Models with Sharpe ratio > 1.0: {len(summary_df[summary_df['Sharpe Ratio'] > 1.0])}</li>
                <li>Models with positive returns: {len(summary_df[summary_df['Total Return (%)'] > 0])}</li>
            </ul>
            """
        
        html_content += """
            <h2>Recommendations</h2>
            <ul>
                <li>Focus on models with consistent positive Sharpe ratios</li>
                <li>Consider ensemble approaches combining best performers</li>
                <li>Monitor drawdown levels for risk management</li>
                <li>Implement position sizing based on model confidence</li>
            </ul>
            
        </body>
        </html>
        """
        
        # Save HTML report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Detailed report saved to {output_file}")
    
    def export_results_to_excel(self, output_file: str = 'backtests/backtest_results.xlsx'):
        """Export all results to Excel file with multiple sheets"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = self.generate_summary_report()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual symbol sheets
            for symbol, data in self.results.items():
                if not data['trades'].empty:
                    data['trades'].to_excel(writer, sheet_name=f'{symbol}_Trades', index=False)
                
                if not data['equity'].empty:
                    data['equity'].to_excel(writer, sheet_name=f'{symbol}_Equity', index=False)
        
        print(f"Results exported to {output_file}")

def main():
    """Main function to run analysis"""
    print("Starting backtest analysis...")
    
    # Initialize analyzer
    analyzer = BacktestAnalyzer()
    
    if not analyzer.symbols:
        print("No backtest results found. Please run backtest_models.py first.")
        return
    
    print(f"Found results for symbols: {analyzer.symbols}")
    
    # Generate summary report
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    summary = analyzer.generate_summary_report()
    print(summary.to_string(index=False))
    
    # Analyze trade patterns
    analyzer.analyze_trade_patterns()
    
    # Create visualizations
    print("\nGenerating performance comparison plots...")
    analyzer.plot_performance_comparison()
    
    # Generate detailed report
    print("\nGenerating detailed HTML report...")
    analyzer.generate_detailed_report()
    
    # Export to Excel
    print("\nExporting results to Excel...")
    analyzer.export_results_to_excel()
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()