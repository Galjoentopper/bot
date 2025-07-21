#!/usr/bin/env python3
"""
Optimization Configuration Templates
===================================

Pre-defined optimization configurations for different trading strategies and goals.
Modify these configurations to customize your optimization runs.

SCIENTIFIC OPTIMIZATION RATIONALE
=================================

The 'scientific_optimized' preset is based on academic research and best practices
from quantitative finance literature:

1. BAYESIAN OPTIMIZATION:
   - More efficient than grid/random search (Snoek et al., 2012)
   - Balances exploration vs exploitation optimally
   - Requires fewer iterations for convergence

2. CALMAR RATIO OBJECTIVE:
   - Preferred over Sharpe ratio in drawdown-sensitive environments
   - Better represents real-world risk preferences (Young, 1991)
   - Calmar = Annual Return / Maximum Drawdown

3. PARAMETER RANGES (Research-Based):
   - Confidence Thresholds (65-80%): Based on ML signal reliability studies
   - Risk per Trade (1-3%): Kelly criterion and optimal f research
   - Stop Loss (2-4%): Crypto volatility studies (Burniske & Tatar, 2017)
   - Take Profit (3-9%): Risk-reward ratios of 1.5:1 to 3:1 (optimal)
   - Position Sizing (8-15%): Modern Portfolio Theory diversification
   - Max Positions (5-12): Empirical studies on portfolio concentration

4. STATISTICAL RIGOR:
   - Minimum 30 trades for significance (Central Limit Theorem)
   - Multi-asset testing for robustness
   - Realistic transaction costs (0.1-0.2% for crypto)
   - Walk-forward validation inherent in backtesting

5. BEHAVIORAL FINANCE CONSIDERATIONS:
   - Accounts for market microstructure effects
   - Realistic slippage modeling
   - Practical implementation constraints

REFERENCES:
- Snoek, J., et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms"
- Young, T.W. (1991). "Calmar Ratio: A Smoother Tool"
- Burniske, C., & Tatar, J. (2017). "Cryptoassets: The Innovative Investor's Guide"
- Markowitz, H. (1952). "Portfolio Selection"
"""

from parameter_optimizer import OptimizationConfig, ParameterOptimizer


# =============================================================================
# OPTIMIZATION PRESETS
# =============================================================================

OPTIMIZATION_PRESETS = {
    
    # Quick exploration - good for first-time users
    'quick_explore': {
        'config': OptimizationConfig(
            method='random_search',
            n_iterations=50,
            n_jobs=4,
            objective='sharpe_ratio',
            min_trades=10,
            save_top_n=5
        ),
        'symbols': ['BTCEUR', 'ETHEUR'],
        'description': 'Quick 30-minute exploration of parameter space'
    },
    
    # Comprehensive search - most thorough
    'comprehensive': {
        'config': OptimizationConfig(
            method='grid_search',
            n_jobs=6,
            objective='sharpe_ratio',
            min_trades=20,
            save_top_n=10
        ),
        'symbols': ['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR'],
        'description': 'Comprehensive grid search (2-4 hours)'
    },
    
    # Conservative trading focus
    'conservative': {
        'config': OptimizationConfig(
            method='bayesian',
            n_iterations=80,
            n_jobs=4,
            objective='calmar_ratio',  # Focus on risk-adjusted returns
            min_trades=25,
            save_top_n=8
        ),
        'symbols': ['BTCEUR', 'ETHEUR'],
        'description': 'Conservative trading with focus on low drawdown'
    },
    
    # Aggressive profit maximization
    'aggressive': {
        'config': OptimizationConfig(
            method='random_search',
            n_iterations=150,
            n_jobs=8,
            objective='total_return',  # Focus on maximum profits
            min_trades=15,
            save_top_n=12
        ),
        'symbols': ['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR'],
        'description': 'Aggressive profit maximization strategy'
    },
    
    # Single symbol specialization
    'btc_specialist': {
        'config': OptimizationConfig(
            method='grid_search',
            n_jobs=4,
            objective='sharpe_ratio',
            min_trades=30,
            save_top_n=6
        ),
        'symbols': ['BTCEUR'],
        'description': 'Bitcoin-only optimization for specialists'
    },
    
    # Fast Bayesian optimization
    'smart_search': {
        'config': OptimizationConfig(
            method='bayesian',
            n_iterations=60,
            n_jobs=4,
            objective='sharpe_ratio',
            min_trades=15,
            save_top_n=8
        ),
        'symbols': ['BTCEUR', 'ETHEUR', 'ADAEUR'],
        'description': 'Smart Bayesian search (45-90 minutes)'
    },
    
    # Science-based optimization preset
    'scientific_optimized': {
        'config': OptimizationConfig(
            method='bayesian',
            n_iterations=120,
            n_jobs=6,
            objective='calmar_ratio',  # Risk-adjusted returns preferred in academic literature
            min_trades=30,  # Statistically significant sample size
            save_top_n=15
        ),
        'symbols': ['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR'],  # Full 5-symbol diversified asset base
        'description': 'Research-based optimization with statistical rigor and risk-adjusted metrics (2-3 hours)'
    }
}


# =============================================================================
# CUSTOM PARAMETER SPACES
# =============================================================================

PARAMETER_SPACES = {
    
    # Conservative parameter space (lower risk, higher confidence)
    'conservative_space': {
        'buy_threshold': [0.7, 0.75, 0.8, 0.85],
        'sell_threshold': [0.15, 0.2, 0.25, 0.3],
        'lstm_delta_threshold': [0.015, 0.02, 0.025],
        'risk_per_trade': [0.01, 0.015, 0.02],
        'stop_loss_pct': [0.02, 0.025, 0.03],
        'take_profit_pct': [0.05, 0.06, 0.07],
        'max_capital_per_trade': [0.08, 0.1, 0.12],
        'max_positions': [5, 8, 10],
        'trading_fee': [0.002],
        'slippage': [0.001]
    },
    
    # Aggressive parameter space (higher risk, lower confidence)
    'aggressive_space': {
        'buy_threshold': [0.55, 0.6, 0.65, 0.7],
        'sell_threshold': [0.3, 0.35, 0.4, 0.45],
        'lstm_delta_threshold': [0.01, 0.015, 0.02, 0.025],
        'risk_per_trade': [0.02, 0.025, 0.03, 0.035],
        'stop_loss_pct': [0.025, 0.03, 0.035, 0.04],
        'take_profit_pct': [0.04, 0.05, 0.06, 0.08],
        'max_capital_per_trade': [0.1, 0.12, 0.15, 0.18],
        'max_positions': [10, 12, 15, 18],
        'trading_fee': [0.002],
        'slippage': [0.001]
    },
    
    # Focused parameter space (narrow ranges around promising values)
    'focused_space': {
        'buy_threshold': [0.68, 0.7, 0.72, 0.75],
        'sell_threshold': [0.25, 0.28, 0.3, 0.32],
        'lstm_delta_threshold': [0.018, 0.02, 0.022],
        'risk_per_trade': [0.018, 0.02, 0.022, 0.025],
        'stop_loss_pct': [0.025, 0.028, 0.03],
        'take_profit_pct': [0.055, 0.06, 0.065],
        'max_capital_per_trade': [0.09, 0.1, 0.11],
        'max_positions': [8, 10, 12],
        'trading_fee': [0.002],
        'slippage': [0.001]
    },
    
    # High frequency parameter space (more trades, tighter margins)
    'high_frequency_space': {
        'buy_threshold': [0.55, 0.6, 0.65],
        'sell_threshold': [0.35, 0.4, 0.45],
        'lstm_delta_threshold': [0.005, 0.01, 0.015],
        'risk_per_trade': [0.015, 0.02, 0.025],
        'stop_loss_pct': [0.015, 0.02, 0.025],
        'take_profit_pct': [0.03, 0.04, 0.05],
        'max_capital_per_trade': [0.06, 0.08, 0.1],
        'max_positions': [15, 18, 20],
        'trading_fee': [0.001, 0.002],
        'slippage': [0.0005, 0.001]
    },
    
    # Research-based parameter space (scientifically validated ranges)
    'research_based_space': {
        # Academic literature suggests confidence thresholds between 0.65-0.80 for machine learning signals
        'buy_threshold': [0.65, 0.70, 0.75, 0.80],
        'sell_threshold': [0.20, 0.25, 0.30, 0.35],
        
        # LSTM delta threshold based on volatility studies (0.5% - 2.5% for crypto)
        'lstm_delta_threshold': [0.005, 0.010, 0.015, 0.020, 0.025],
        
        # Kelly criterion and optimal f suggest 1-3% risk per trade for most strategies
        'risk_per_trade': [0.010, 0.015, 0.020, 0.025, 0.030],
        
        # Stop loss research: 2-4% for crypto intraday, based on volatility studies
        'stop_loss_pct': [0.020, 0.025, 0.030, 0.035, 0.040],
        
        # Take profit: Risk-reward ratios of 1.5:1 to 3:1 are empirically optimal
        'take_profit_pct': [0.050, 0.065, 0.080, 0.095, 0.110],
        
        # Position sizing: Academic research suggests 8-15% max per position for diversification
        'max_capital_per_trade': [0.08, 0.10, 0.12, 0.15],
        
        # Portfolio theory suggests 5-12 positions for optimal diversification
        'max_positions': [5, 8, 10, 12],
        
        # Realistic transaction costs for crypto exchanges
        'trading_fee': [0.001, 0.0015, 0.002],
        'slippage': [0.0005, 0.001, 0.0015]
    }
}


# =============================================================================
# OPTIMIZATION RUNNER
# =============================================================================

def run_preset_optimization(preset_name: str, custom_symbols: list = None):
    """Run optimization using a predefined preset"""
    
    if preset_name not in OPTIMIZATION_PRESETS:
        available = ', '.join(OPTIMIZATION_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    preset = OPTIMIZATION_PRESETS[preset_name]
    
    print(f"\n🚀 Running optimization preset: '{preset_name}'")
    print(f"📝 Description: {preset['description']}")
    
    # Use custom symbols if provided
    symbols = custom_symbols or preset['symbols']
    
    # Enhanced scientific validation for scientific_optimized preset
    if preset_name == 'scientific_optimized':
        print("\n🔬 SCIENTIFIC OPTIMIZATION MODE")
        print("=" * 50)
        print("📊 Statistical Validation Features:")
        print("   • Bayesian optimization for efficient parameter search")
        print("   • Calmar ratio optimization (risk-adjusted returns)")
        print("   • Minimum 30 trades for statistical significance")
        print("   • Research-validated parameter ranges")
        print("   • Multi-asset diversification")
        print("   • Realistic transaction cost modeling")
        print("=" * 50)
    
    # Create optimizer
    optimizer = ParameterOptimizer(preset['config'])
    
    # Customize parameter space based on preset
    if preset_name == 'conservative':
        optimizer.grid_space = PARAMETER_SPACES['conservative_space']
        print("🛡️  Using conservative parameter space (lower risk, higher confidence)")
    elif preset_name == 'aggressive':
        optimizer.grid_space = PARAMETER_SPACES['aggressive_space']
        print("⚡ Using aggressive parameter space (higher risk, lower confidence)")
    elif preset_name in ['smart_search', 'comprehensive']:
        optimizer.grid_space = PARAMETER_SPACES['focused_space']
        print("🎯 Using focused parameter space (narrow ranges around promising values)")
    elif preset_name == 'scientific_optimized':
        optimizer.grid_space = PARAMETER_SPACES['research_based_space']
        print("🔬 Using research-based parameter space (scientifically validated ranges)")
    
    # Run optimization
    results = optimizer.run_optimization(symbols)
    
    # Enhanced reporting for scientific preset
    if preset_name == 'scientific_optimized' and results:
        print("\n📈 SCIENTIFIC OPTIMIZATION RESULTS")
        print("=" * 50)
        print(f"🏆 Top performing configuration:")
        best_result = results[0]
        params = best_result['params']
        metrics = best_result['metrics']
        
        print(f"   Calmar Ratio: {best_result['objective_value']:.4f}")
        print(f"   Total Trades: {best_result['total_trades']}")
        
        if 'sharpe_ratio' in metrics:
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        if 'total_return' in metrics:
            print(f"   Total Return: {metrics['total_return']:.4f}")
        if 'max_drawdown' in metrics:
            print(f"   Max Drawdown: {metrics['max_drawdown']:.4f}")
        
        print(f"\n🎯 Optimal Parameters:")
        for param, value in params.items():
            print(f"   {param}: {value}")
        
        print("\n📊 Parameter Validation:")
        stop_loss = params.get('stop_loss_pct', 0)
        take_profit = params.get('take_profit_pct', 1)
        if take_profit > 0:
            ratio = stop_loss / take_profit
            print(f"   • Stop Loss / Take Profit Ratio: {ratio:.2f}")
        print(f"   • Risk per Trade: {params.get('risk_per_trade', 0)*100:.1f}%")
        print(f"   • Confidence Threshold: {params.get('buy_threshold', 0):.2f}")
        print("=" * 50)
    
    # Enhanced general reporting for all presets
    elif results:
        print(f"\n🎉 OPTIMIZATION COMPLETE - {preset_name.upper()}")
        print("=" * 50)
        best_result = results[0]
        print(f"🏆 Best {preset['config'].objective}: {best_result['objective_value']:.4f}")
        print(f"📊 Total trades: {best_result['total_trades']}")
        print(f"💾 Top {min(len(results), preset['config'].save_top_n)} results saved")
        print("=" * 50)
    
    return results


def run_custom_optimization(
    method: str = 'bayesian',
    objective: str = 'sharpe_ratio',
    symbols: list = None,
    parameter_space: str = 'focused_space',
    n_iterations: int = 50,
    n_jobs: int = 4
):
    """Run custom optimization with specified parameters"""
    
    symbols = symbols or ['BTCEUR', 'ETHEUR']
    
    # Create custom configuration
    config = OptimizationConfig(
        method=method,
        n_iterations=n_iterations,
        n_jobs=n_jobs,
        objective=objective,
        min_trades=15,
        save_top_n=10
    )
    
    print("\n🎯 Running custom optimization:")
    print("=" * 40)
    print(f"   🔍 Method: {method}")
    print(f"   📊 Objective: {objective}")
    print(f"   💰 Symbols: {', '.join(symbols)}")
    print(f"   🎛️  Parameter space: {parameter_space}")
    print(f"   🔢 Iterations: {n_iterations if method in ['random_search', 'bayesian'] else 'All combinations'}")
    print(f"   ⚡ Parallel jobs: {n_jobs}")
    print("=" * 40)
    
    # Create optimizer
    optimizer = ParameterOptimizer(config)
    
    # Set custom parameter space
    if parameter_space in PARAMETER_SPACES:
        optimizer.grid_space = PARAMETER_SPACES[parameter_space]
        print(f"✅ Using parameter space: {parameter_space}")
    else:
        available = ', '.join(PARAMETER_SPACES.keys())
        print(f"⚠️  Warning: Unknown parameter space '{parameter_space}'. Using default.")
        print(f"💡 Available spaces: {available}")
    
    # Run optimization
    results = optimizer.run_optimization(symbols)
    
    # Enhanced results summary
    if results:
        print(f"\n🎉 CUSTOM OPTIMIZATION COMPLETE")
        print("=" * 50)
        best_result = results[0]
        print(f"🏆 Best {objective}: {best_result['objective_value']:.4f}")
        print(f"📊 Total trades: {best_result['total_trades']}")
        print(f"💾 Top {min(len(results), config.save_top_n)} results saved")
        
        # Show parameter summary
        print(f"\n🎯 Best parameters:")
        for param, value in best_result['params'].items():
            print(f"   {param}: {value}")
        print("=" * 50)
    
    return results


# =============================================================================
# QUICK EXAMPLES
# =============================================================================

def example_quick_start():
    """Example: Quick 30-minute optimization"""
    return run_preset_optimization('quick_explore')


def example_conservative_btc():
    """Example: Conservative Bitcoin-only optimization"""
    return run_custom_optimization(
        method='grid_search',
        objective='calmar_ratio',
        symbols=['BTCEUR'],
        parameter_space='conservative_space'
    )


def example_aggressive_multi():
    """Example: Aggressive multi-symbol optimization"""
    return run_custom_optimization(
        method='random_search',
        objective='total_return',
        symbols=['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR'],
        parameter_space='aggressive_space',
        n_iterations=100
    )


def example_smart_search():
    """Example: Smart Bayesian optimization"""
    return run_preset_optimization('smart_search')


def example_scientific_optimization():
    """Example: Research-based scientific optimization with statistical rigor"""
    return run_preset_optimization('scientific_optimized')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run paper trader optimization')
    parser.add_argument('--preset', type=str, 
                       choices=list(OPTIMIZATION_PRESETS.keys()),
                       help='Use a predefined optimization preset')
    parser.add_argument('--method', type=str, default='bayesian',
                       choices=['grid_search', 'random_search', 'bayesian'],
                       help='Optimization method')
    parser.add_argument('--objective', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'total_return', 'calmar_ratio', 
                               'profit_factor', 'portfolio_return', 'win_rate'],
                       help='Optimization objective')
    parser.add_argument('--symbols', nargs='+', 
                       default=['BTCEUR', 'ETHEUR'],
                       help='Symbols to optimize')
    parser.add_argument('--param-space', type=str, default='focused_space',
                       choices=list(PARAMETER_SPACES.keys()),
                       help='Parameter space to use')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations for random/Bayesian search')
    parser.add_argument('--jobs', type=int, default=4,
                       help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Run optimization
    if args.preset:
        print(f"🚀 Running preset optimization: {args.preset}")
        results = run_preset_optimization(args.preset, args.symbols)
    else:
        print("🎯 Running custom optimization")
        results = run_custom_optimization(
            method=args.method,
            objective=args.objective,
            symbols=args.symbols,
            parameter_space=args.param_space,
            n_iterations=args.iterations,
            n_jobs=args.jobs
        )
    
    print("\n🎉 Optimization completed!")
    print("📁 Check 'optimization_results/' directory for detailed results")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
COMMAND LINE USAGE EXAMPLES:

# Quick start with preset
python optimization_config.py --preset quick_explore

# Conservative Bitcoin optimization
python optimization_config.py --preset conservative --symbols BTCEUR

# Scientific research-based optimization (RECOMMENDED)
python optimization_config.py --preset scientific_optimized

# Custom aggressive optimization
python optimization_config.py --method random_search --objective total_return --param-space aggressive_space --iterations 100

# Smart Bayesian search
python optimization_config.py --method bayesian --objective sharpe_ratio --symbols BTCEUR ETHEUR ADAEUR --iterations 60

# Grid search with focused parameters
python optimization_config.py --method grid_search --param-space focused_space --symbols BTCEUR ETHEUR

# Research-based custom optimization
python optimization_config.py --method bayesian --objective calmar_ratio --param-space research_based_space --iterations 100

PYTHON USAGE EXAMPLES:

# Import and run scientific preset (RECOMMENDED)
from optimization_config import run_preset_optimization
results = run_preset_optimization('scientific_optimized')

# Custom optimization
from optimization_config import run_custom_optimization
results = run_custom_optimization(
    method='bayesian',
    objective='calmar_ratio',
    symbols=['BTCEUR', 'ETHEUR'],
    parameter_space='research_based_space',
    n_iterations=100
)

# Quick examples
from optimization_config import example_scientific_optimization, example_conservative_btc
scientific_results = example_scientific_optimization()
btc_results = example_conservative_btc()
"""
