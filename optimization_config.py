#!/usr/bin/env python3
"""
Optimization Configuration Templates
===================================

Pre-defined optimization configurations for different trading strategies and goals.
Modify these configurations to customize your optimization runs.
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
        'symbols': ['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR'],
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
        'symbols': ['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR'],
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
    
    print(f"🚀 Running optimization preset: '{preset_name}'")
    print(f"📝 Description: {preset['description']}")
    
    # Use custom symbols if provided
    symbols = custom_symbols or preset['symbols']
    
    # Create optimizer
    optimizer = ParameterOptimizer(preset['config'])
    
    # Customize parameter space based on preset
    if preset_name == 'conservative':
        optimizer.grid_space = PARAMETER_SPACES['conservative_space']
    elif preset_name == 'aggressive':
        optimizer.grid_space = PARAMETER_SPACES['aggressive_space']
    elif preset_name in ['smart_search', 'comprehensive']:
        optimizer.grid_space = PARAMETER_SPACES['focused_space']
    
    # Run optimization
    results = optimizer.run_optimization(symbols)
    
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
    
    print("🎯 Running custom optimization:")
    print(f"   Method: {method}")
    print(f"   Objective: {objective}")
    print(f"   Symbols: {symbols}")
    print(f"   Parameter space: {parameter_space}")
    
    # Create optimizer
    optimizer = ParameterOptimizer(config)
    
    # Set custom parameter space
    if parameter_space in PARAMETER_SPACES:
        optimizer.grid_space = PARAMETER_SPACES[parameter_space]
    
    # Run optimization
    results = optimizer.run_optimization(symbols)
    
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
        symbols=['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR'],
        parameter_space='aggressive_space',
        n_iterations=100
    )


def example_smart_search():
    """Example: Smart Bayesian optimization"""
    return run_preset_optimization('smart_search')


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

# Custom aggressive optimization
python optimization_config.py --method random_search --objective total_return --param-space aggressive_space --iterations 100

# Smart Bayesian search
python optimization_config.py --method bayesian --objective sharpe_ratio --symbols BTCEUR ETHEUR ADAEUR --iterations 60

# Grid search with focused parameters
python optimization_config.py --method grid_search --param-space focused_space --symbols BTCEUR ETHEUR

PYTHON USAGE EXAMPLES:

# Import and run preset
from optimization_config import run_preset_optimization
results = run_preset_optimization('smart_search')

# Custom optimization
from optimization_config import run_custom_optimization
results = run_custom_optimization(
    method='bayesian',
    objective='sharpe_ratio',
    symbols=['BTCEUR', 'ETHEUR'],
    n_iterations=80
)

# Quick examples
from optimization_config import example_quick_start, example_conservative_btc
quick_results = example_quick_start()
btc_results = example_conservative_btc()
"""
