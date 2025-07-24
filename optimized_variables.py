#!/usr/bin/env python3
"""
Scientific Parameter Optimization for Paper Trading Bot
======================================================

This script uses advanced scientific optimization methods to find the best parameter
combinations for the paper trading bot. It leverages existing trained models and
historical data to maximize profit potential.

Features:
- Bayesian optimization for efficient parameter search
- Support for single or multiple symbols via command line
- Scientific optimization metrics (Sharpe ratio, Calmar ratio, Total return)
- Automated .env file generation with symbol and date naming
- Integration with existing model and data infrastructure

Usage:
    python optimized_variables.py --symbols BTCEUR
    python optimized_variables.py --symbols BTCEUR ETHEUR
    python optimized_variables.py --symbols BTCEUR ETHEUR ADAEUR --method bayesian --iterations 100
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Scientific optimization imports
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import random

# Add scripts directory to path for importing backtest framework
sys.path.append('scripts')
from backtest_models import ModelBacktester, BacktestConfig, Trade


class ParameterSpace:
    """Defines the parameter search space based on scientific research"""
    
    def __init__(self, optimization_mode: str = 'balanced'):
        """
        Initialize parameter space based on optimization mode
        
        Args:
            optimization_mode: 'conservative', 'balanced', 'aggressive', 'profit_focused', 'high_frequency'
        """
        # Base parameter ranges optimized for different trading frequencies
        if optimization_mode == 'conservative':
            self.param_bounds = {
                'buy_threshold': (0.502, 0.58),         # Nearly neutral for maximum trades
                'sell_threshold': (0.42, 0.498),         # Nearly neutral for maximum trades
                'lstm_delta_threshold': (0.0001, 0.005), # Ultra-sensitive
                'risk_per_trade': (0.005, 0.015),        # Lower risk
                'stop_loss_pct': (0.015, 0.035),         # Tighter stop loss
                'take_profit_pct': (0.03, 0.08),         # Conservative targets
                'max_capital_per_trade': (0.05, 0.12),   # Smaller positions
                'max_positions': (3, 8),                 # Fewer positions
            }
        elif optimization_mode == 'aggressive':
            self.param_bounds = {
                'buy_threshold': (0.501, 0.55),         # Nearly neutral for maximum trades
                'sell_threshold': (0.45, 0.499),         # Nearly neutral for maximum trades
                'lstm_delta_threshold': (0.0001, 0.003), # Ultra-sensitive
                'risk_per_trade': (0.015, 0.035),        # Higher risk
                'stop_loss_pct': (0.02, 0.05),           # Wider stop loss
                'take_profit_pct': (0.04, 0.12),         # Aggressive targets
                'max_capital_per_trade': (0.08, 0.18),   # Larger positions
                'max_positions': (8, 15),                # More positions
            }
        elif optimization_mode == 'high_frequency':
            # ULTRA-aggressive parameters for 5+ trades per day target - near-neutral thresholds for maximum trades
            self.param_bounds = {
                'buy_threshold': (0.5001, 0.505),      # Near-neutral for maximum trades (any slight bullish bias)
                'sell_threshold': (0.495, 0.4999),     # Near-neutral for maximum trades (any slight bearish bias) 
                'lstm_delta_threshold': (0.0000001, 0.0001), # ULTRA-sensitive - trade on minimal signals
                'risk_per_trade': (0.02, 0.05),        # Smaller risk for more frequent trades (2-5%)
                'stop_loss_pct': (0.002, 0.015),       # Tighter stop loss (0.2-1.5%)
                'take_profit_pct': (0.003, 0.02),      # Smaller take profit (0.3-2%) but > stop_loss
                'max_capital_per_trade': (0.02, 0.05), # Smaller positions (2-5%) to enable more trades
                'max_positions': (12, 20),             # High position limit for frequent trading
            }
        elif optimization_mode == 'profit_focused':
            self.param_bounds = {
                'buy_threshold': (0.502, 0.58),         # Nearly neutral for maximum trades
                'sell_threshold': (0.42, 0.498),         # Nearly neutral for maximum trades
                'lstm_delta_threshold': (0.0001, 0.005), # Ultra-sensitive
                'risk_per_trade': (0.01, 0.025),         # Moderate risk
                'stop_loss_pct': (0.018, 0.04),          # Balanced stop loss
                'take_profit_pct': (0.035, 0.10),        # Profit-focused targets
                'max_capital_per_trade': (0.06, 0.15),   # Balanced positions
                'max_positions': (5, 12),                # Optimal diversification
            }
        else:  # balanced
            self.param_bounds = {
                'buy_threshold': (0.501, 0.57),         # Nearly neutral for maximum trades
                'sell_threshold': (0.43, 0.499),         # Nearly neutral for maximum trades
                'lstm_delta_threshold': (0.0001, 0.005), # Ultra-sensitive
                'risk_per_trade': (0.01, 0.025),         # Moderate risk
                'stop_loss_pct': (0.02, 0.04),           # Balanced stop loss
                'take_profit_pct': (0.04, 0.09),         # Balanced targets
                'max_capital_per_trade': (0.06, 0.15),   # Balanced positions
                'max_positions': (5, 12),                # Balanced diversification
            }
        
        # Fixed parameters optimized for high-frequency trading
        self.fixed_params = {
            'trading_fee': 0.002,        # Realistic crypto trading fees
            'slippage': 0.001,           # Conservative slippage estimate  
            'max_trades_per_hour': 15,   # Allow frequent trades for high frequency (up to 1 every 4 minutes)
            'initial_capital': 10000.0,  # Standard starting capital
        }
        
        self.param_names = list(self.param_bounds.keys())
        self.bounds = [self.param_bounds[name] for name in self.param_names]


class ScientificOptimizer:
    """
    Advanced scientific optimization engine using Bayesian optimization
    and other sophisticated methods for parameter tuning
    """
    
    def __init__(self, symbols: List[str], optimization_mode: str = 'balanced', 
                 objective: str = 'profit_factor', verbose: bool = True):
        """
        Initialize the scientific optimizer
        
        Args:
            symbols: List of symbols to optimize for
            optimization_mode: Parameter space mode ('conservative', 'balanced', 'aggressive', 'profit_focused')
            objective: Optimization objective ('sharpe_ratio', 'total_return', 'calmar_ratio', 'profit_factor')
            verbose: Enable detailed logging
        """
        self.symbols = symbols
        self.optimization_mode = optimization_mode
        self.objective = objective
        self.verbose = verbose
        
        # Initialize parameter space
        self.param_space = ParameterSpace(optimization_mode)
        
        # Initialize backtest configuration
        self.base_config = BacktestConfig()
        self.base_config.verbose = False  # Suppress backtest output during optimization
        
        # Optimization state
        self.evaluated_params = []
        self.evaluation_results = []
        self.best_result = None
        self.iteration_count = 0
        self.zero_trades_detected = False
        self.aggressive_mode_enabled = False
        
        # Gaussian Process for Bayesian optimization
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42,
            alpha=1e-6,
            normalize_y=True
        )
        
        # Parameter scaling for GP
        self.param_scaler = StandardScaler()
        
        if self.verbose:
            print(f"🔬 Scientific Optimizer initialized")
            print(f"📊 Target symbols: {', '.join(symbols)}")
            print(f"🎯 Optimization mode: {optimization_mode}")
            print(f"📈 Objective: {objective}")
            print(f"🔧 Parameter space: {len(self.param_space.param_names)} dimensions")
    
    def switch_to_aggressive_mode(self):
        """Switch to more aggressive parameter ranges when zero trades are detected"""
        if self.aggressive_mode_enabled:
            return  # Already in aggressive mode
            
        if self.verbose:
            print(f"\n⚠️  Zero trades detected across multiple evaluations!")
            print(f"🔄 Switching to AGGRESSIVE mode for better trade generation...")
        
        # Store original parameters for reference
        self.original_param_bounds = self.param_space.param_bounds.copy()
        
        # Switch to ultra-aggressive parameter ranges that trade on any signal
        self.param_space.param_bounds = {
            'buy_threshold': (0.5001, 0.5005),        # ULTRA-near neutral for maximum trades
            'sell_threshold': (0.4995, 0.4999),        # ULTRA-near neutral for maximum trades  
            'lstm_delta_threshold': (0.00000001, 0.0001), # EXTREMELY sensitive - trade on tiny signals
            'risk_per_trade': (0.01, 0.03),           # Lower risk for more trades
            'stop_loss_pct': (0.001, 0.01),           # Very tight stop loss for quick exits
            'take_profit_pct': (0.002, 0.015),        # Small targets but > stop_loss
            'max_capital_per_trade': (0.015, 0.04),   # Very small positions
            'max_positions': (15, 25),                # Many positions allowed
        }
        
        # Update bounds for optimization
        self.param_space.bounds = [self.param_space.param_bounds[name] for name in self.param_space.param_names]
        
        self.aggressive_mode_enabled = True
        
        if self.verbose:
            print(f"✅ Aggressive mode enabled - using very permissive thresholds")
            print(f"   Buy threshold: {self.param_space.param_bounds['buy_threshold']}")
            print(f"   Sell threshold: {self.param_space.param_bounds['sell_threshold']}")
            print(f"   LSTM delta threshold: {self.param_space.param_bounds['lstm_delta_threshold']}")
    
    def check_for_zero_trades_and_adapt(self):
        """Check if we're getting zero trades and adapt if needed - now more responsive"""
        if len(self.evaluation_results) >= 1:  # Check after just 1 evaluation for immediate adaptation
            # Count recent evaluations with zero trades (objective = -999)
            recent_results = self.evaluation_results[-1:]  # Check just the most recent result
            zero_trade_count = sum(1 for result in recent_results if result <= -990)
            
            # Also check for very low trade frequency in recent results
            low_frequency_count = 0
            if len(self.evaluation_results) >= 2:
                recent_more = self.evaluation_results[-2:]
                for result in recent_more:
                    # Consider anything below -500 as likely low frequency (partial penalty)
                    if -900 <= result <= -500:
                        low_frequency_count += 1
            
            # Switch to aggressive mode if we see zero trades OR multiple low frequency results
            if (zero_trade_count >= 1 or low_frequency_count >= 2) and not self.aggressive_mode_enabled:
                self.switch_to_aggressive_mode()
                return True
        return False
    
    def evaluate_parameters(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate a parameter combination using backtesting
        
        Args:
            params: Parameter dictionary to evaluate
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        try:
            # Create backtest configuration with these parameters
            config = self._create_config_from_params(params)
            
            # Initialize backtester
            backtester = ModelBacktester(config)
            
            # Run backtest for all symbols
            start_time = time.time()
            results = backtester.run_backtest(self.symbols)
            duration = time.time() - start_time
            
            # Aggregate results across symbols
            metrics = self._aggregate_results(results)
            
            # Calculate objective value
            objective_value = self._calculate_objective(metrics)
            
            if self.verbose and objective_value > 0:
                print(f"✅ Evaluation complete: {self.objective}={objective_value:.4f} "
                      f"(trades={metrics.get('total_trades', 0)}, time={duration:.1f}s)")
            
            return {
                'params': params.copy(),
                'metrics': metrics,
                'objective_value': objective_value,
                'evaluation_time': duration,
                'success': True
            }
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Evaluation failed: {str(e)[:100]}")
            return {
                'params': params.copy(),
                'metrics': {},
                'objective_value': -999,  # Penalty for failed evaluations
                'evaluation_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def bayesian_optimize(self, n_iterations: int = 50, n_initial: int = 10) -> Dict[str, Any]:
        """
        Perform Bayesian optimization to find optimal parameters
        
        Args:
            n_iterations: Total number of optimization iterations
            n_initial: Number of initial random evaluations
            
        Returns:
            Best parameter configuration found
        """
        if self.verbose:
            print(f"\n🚀 Starting Bayesian optimization")
            print(f"🔄 Iterations: {n_iterations} (including {n_initial} initial random)")
            print(f"⏰ Estimated time: {n_iterations * 2:.0f}-{n_iterations * 5:.0f} minutes")
            print("-" * 60)
        
        # Phase 1: Initial random exploration
        if self.verbose:
            print(f"📊 Phase 1: Initial random exploration ({n_initial} evaluations)")
        
        for i in range(n_initial):
            params = self._sample_random_params()
            result = self.evaluate_parameters(params)
            
            self.evaluated_params.append(self._params_to_array(params))
            self.evaluation_results.append(result['objective_value'])
            
            if result['success'] and (self.best_result is None or 
                                    result['objective_value'] > self.best_result['objective_value']):
                self.best_result = result
                if self.verbose:
                    print(f"🏆 New best result: {self.objective}={result['objective_value']:.4f}")
            
            # Check for zero trades and adapt parameters if needed
            adapted = self.check_for_zero_trades_and_adapt()
            if adapted:
                if self.verbose:
                    print(f"🔄 Re-sampling with aggressive parameters...")
                # Sample a few more with aggressive parameters
                for j in range(2):
                    aggressive_params = self._sample_random_params()
                    aggressive_result = self.evaluate_parameters(aggressive_params)
                    self.evaluated_params.append(self._params_to_array(aggressive_params))
                    self.evaluation_results.append(aggressive_result['objective_value'])
                    if aggressive_result['success'] and (self.best_result is None or 
                                                    aggressive_result['objective_value'] > self.best_result['objective_value']):
                        self.best_result = aggressive_result
                        if self.verbose:
                            print(f"🏆 New aggressive best result: {self.objective}={aggressive_result['objective_value']:.4f}")
            
            self.iteration_count += 1
            
            if self.verbose and (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{n_initial} initial evaluations complete")
        
        # Fit initial GP model
        if len(self.evaluated_params) >= 3:
            X = np.array(self.evaluated_params)
            y = np.array(self.evaluation_results)
            
            # Scale parameters for GP
            X_scaled = self.param_scaler.fit_transform(X)
            self.gp.fit(X_scaled, y)
            
            if self.verbose:
                print(f"🧠 Gaussian Process model fitted with {len(X)} samples")
        else:
            # Not enough samples yet for GP - fit scaler on current data
            if len(self.evaluated_params) > 0:
                X = np.array(self.evaluated_params)
                self.param_scaler.fit(X)
        
        # Phase 2: Bayesian optimization
        if self.verbose:
            print(f"\n📈 Phase 2: Bayesian optimization ({n_iterations - n_initial} iterations)")
        
        for i in range(n_initial, n_iterations):
            # Acquisition function optimization
            next_params = self._optimize_acquisition()
            
            # Evaluate the suggested parameters
            result = self.evaluate_parameters(next_params)
            
            # Update GP model
            self.evaluated_params.append(self._params_to_array(next_params))
            self.evaluation_results.append(result['objective_value'])
            
            X = np.array(self.evaluated_params)
            y = np.array(self.evaluation_results)
            
            # Refit scaler and GP model with all data
            X_scaled = self.param_scaler.fit_transform(X)
            self.gp.fit(X_scaled, y)
            
            # Update best result
            if result['success'] and (self.best_result is None or 
                                    result['objective_value'] > self.best_result['objective_value']):
                self.best_result = result
                if self.verbose:
                    print(f"🏆 New best result: {self.objective}={result['objective_value']:.4f}")
            
            self.iteration_count += 1
            
            # Progress reporting
            if self.verbose and (i - n_initial + 1) % 5 == 0:
                current_best = max(self.evaluation_results) if self.evaluation_results else 0
                print(f"Progress: {i + 1}/{n_iterations} iterations, "
                      f"best {self.objective}: {current_best:.4f}")
        
        if self.verbose:
            print(f"\n🎉 Bayesian optimization complete!")
            if self.best_result:
                print(f"🏆 Best {self.objective}: {self.best_result['objective_value']:.4f}")
                print(f"📊 Total evaluations: {len(self.evaluation_results)}")
                print(f"✅ Successful evaluations: {sum(1 for r in self.evaluation_results if r > -900)}")
        
        return self.best_result or {}
    
    def grid_search(self, n_points_per_dim: int = 3) -> Dict[str, Any]:
        """
        Perform grid search optimization (for comparison with Bayesian)
        
        Args:
            n_points_per_dim: Number of points per dimension for grid
            
        Returns:
            Best parameter configuration found
        """
        if self.verbose:
            total_combinations = n_points_per_dim ** len(self.param_space.param_names)
            print(f"\n🔍 Starting grid search")
            print(f"📊 Grid points per dimension: {n_points_per_dim}")
            print(f"🔢 Total combinations: {total_combinations}")
            print(f"⏰ Estimated time: {total_combinations * 2:.0f}-{total_combinations * 5:.0f} minutes")
            print("-" * 60)
        
        # Generate grid
        grid_points = []
        for param_name in self.param_space.param_names:
            low, high = self.param_space.param_bounds[param_name]
            points = np.linspace(low, high, n_points_per_dim)
            grid_points.append(points)
        
        # Evaluate all combinations
        from itertools import product
        all_combinations = list(product(*grid_points))
        
        for i, combination in enumerate(all_combinations):
            params = dict(zip(self.param_space.param_names, combination))
            params.update(self.param_space.fixed_params)
            
            result = self.evaluate_parameters(params)
            
            if result['success'] and (self.best_result is None or 
                                    result['objective_value'] > self.best_result['objective_value']):
                self.best_result = result
                if self.verbose:
                    print(f"🏆 New best result: {self.objective}={result['objective_value']:.4f}")
            
            self.iteration_count += 1
            
            if self.verbose and (i + 1) % 10 == 0:
                progress = (i + 1) / len(all_combinations) * 100
                print(f"Progress: {i + 1}/{len(all_combinations)} ({progress:.1f}%)")
        
        if self.verbose:
            print(f"\n🎉 Grid search complete!")
            if self.best_result:
                print(f"🏆 Best {self.objective}: {self.best_result['objective_value']:.4f}")
        
        return self.best_result or {}
    
    def _create_config_from_params(self, params: Dict[str, float]) -> BacktestConfig:
        """Create a BacktestConfig from parameter dictionary"""
        # Start with base config to preserve verbose=False and other settings
        config = BacktestConfig()
        
        # Copy all settings from base_config
        for attr_name in dir(self.base_config):
            if not attr_name.startswith('_'):
                setattr(config, attr_name, getattr(self.base_config, attr_name))
        
        # Apply all parameters
        all_params = {**self.param_space.fixed_params, **params}
        for param_name, value in all_params.items():
            if hasattr(config, param_name):
                setattr(config, param_name, value)
        
        return config
    
    def _aggregate_results(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Aggregate backtest results across symbols"""
        metrics = {}
        
        # Collect all valid performance metrics and trades
        valid_performances = []
        all_trades = []
        for symbol, result in results.items():
            # Check if we have a valid result with performance metrics
            if (result and 'performance' in result and 
                isinstance(result['performance'], dict) and 
                'total_trades' in result['performance']):
                valid_performances.append(result['performance'])
            if result and 'trades' in result and result['trades']:
                all_trades.extend(result['trades'])
        
        if not valid_performances:
            return {'total_trades': 0, 'trades_per_day': 0, 'total_return': -1, 'sharpe_ratio': -1, 'max_drawdown': -0.5, 'win_rate': 0, 'profit_factor': 0}
        
        # Aggregate basic metrics
        metrics['total_trades'] = sum(p.get('total_trades', 0) for p in valid_performances)
        metrics['total_return'] = np.mean([p.get('total_return', 0) for p in valid_performances])
        metrics['sharpe_ratio'] = np.mean([p.get('sharpe_ratio', 0) for p in valid_performances])
        metrics['max_drawdown'] = np.mean([p.get('max_drawdown', 0) for p in valid_performances])
        metrics['win_rate'] = np.mean([p.get('win_rate', 0) for p in valid_performances])
        metrics['profit_factor'] = np.mean([p.get('profit_factor', 0) for p in valid_performances])
        
        # Calculate trades per day based on actual trade data
        if all_trades:
            # Find the time span of all trades to calculate trades per day accurately
            trade_times = []
            for trade in all_trades:
                if hasattr(trade, 'entry_time') and trade.entry_time:
                    if isinstance(trade.entry_time, str):
                        try:
                            trade_times.append(pd.to_datetime(trade.entry_time))
                        except Exception:
                            pass
                    elif hasattr(trade.entry_time, 'date'):  # datetime object
                        trade_times.append(trade.entry_time)
            
            if trade_times and len(trade_times) > 1:
                time_span_days = (max(trade_times) - min(trade_times)).days + 1  # Add 1 for inclusive days
                if time_span_days > 0:
                    metrics['trades_per_day'] = len(all_trades) / time_span_days
                else:
                    # Single day trading
                    metrics['trades_per_day'] = len(all_trades)
            elif trade_times:
                # Only one trade or all on same timestamp
                metrics['trades_per_day'] = len(all_trades)
            else:
                # No valid timestamps - use conservative estimate based on window assumption
                # Assume each window represents roughly 1-2 weeks of data
                estimated_days = max(7, len(set(t.symbol for t in all_trades)) * 14)  # 2 weeks per symbol
                metrics['trades_per_day'] = len(all_trades) / estimated_days
        else:
            metrics['trades_per_day'] = 0
        
        return metrics
    
    def _calculate_objective(self, metrics: Dict[str, float]) -> float:
        """
        Calculate objective value from metrics with explicit trades-per-day optimization
        and improved win rate weighting for high-frequency trading
        """
        total_trades = metrics.get('total_trades', 0)
        trades_per_day = metrics.get('trades_per_day', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # For high frequency trading, we need minimum trades per day
        # Target: 5+ trades per day as specified in requirements
        min_trades_per_day = 5
        
        if total_trades == 0:
            if self.verbose:
                print(f"      ❌ Zero trades - returning maximum penalty")
            return -999
        
        if trades_per_day < 0.5:  # Less than 0.5 trades per day is very poor
            if self.verbose:
                print(f"      ❌ Very low frequency ({trades_per_day:.2f} trades/day) - returning penalty")
            return -999
        
        # Calculate base objective based on selected optimization metric
        base_objective = 0
        if self.objective == 'sharpe_ratio':
            base_objective = metrics.get('sharpe_ratio', -999)
        elif self.objective == 'total_return':
            base_objective = metrics.get('total_return', -999)
        elif self.objective == 'calmar_ratio':
            total_return = metrics.get('total_return', 0)
            max_drawdown = abs(metrics.get('max_drawdown', 0.01))
            # Handle edge cases for Calmar ratio calculation
            if total_trades == 0 or total_return <= 0 or max_drawdown <= 0:
                base_objective = -999
            else:
                base_objective = total_return / max_drawdown
        elif self.objective == 'profit_factor':
            base_objective = metrics.get('profit_factor', -999)
        else:
            base_objective = metrics.get('total_return', -999)  # Default to total return
        
        if base_objective <= -990:
            return base_objective
        
        # High-frequency trading bonuses and penalties
        frequency_multiplier = 1.0
        
        # Major bonus for achieving target frequency (5+ trades/day)
        if trades_per_day >= min_trades_per_day:
            frequency_multiplier = 1.3  # 30% bonus for hitting target
            if self.verbose and trades_per_day >= 8:
                print(f"      🚀 High frequency achieved ({trades_per_day:.1f} trades/day) - applying 30% bonus")
        elif trades_per_day >= 3:  # Good frequency
            frequency_multiplier = 1.2  # 20% bonus
            if self.verbose:
                print(f"      ✅ Good frequency ({trades_per_day:.1f} trades/day) - applying 20% bonus")
        elif trades_per_day >= 1:  # Acceptable frequency  
            frequency_multiplier = 1.1  # 10% bonus
            if self.verbose:
                print(f"      📈 Acceptable frequency ({trades_per_day:.1f} trades/day) - applying 10% bonus")
        else:  # Low frequency penalty
            frequency_multiplier = 0.8  # 20% penalty for low frequency
            if self.verbose:
                print(f"      ⚠️  Low frequency ({trades_per_day:.1f} trades/day) - applying 20% penalty")
        
        # Win rate bonus/penalty - critical for sustainable high-frequency trading
        win_rate_multiplier = 1.0
        if win_rate >= 0.6:  # Excellent win rate
            win_rate_multiplier = 1.25  # 25% bonus
        elif win_rate >= 0.5:  # Good win rate
            win_rate_multiplier = 1.15  # 15% bonus
        elif win_rate >= 0.4:  # Acceptable win rate
            win_rate_multiplier = 1.0   # No penalty/bonus
        elif win_rate >= 0.2:  # Poor win rate
            win_rate_multiplier = 0.9   # 10% penalty
        else:  # Very poor win rate (including 0%)
            win_rate_multiplier = 0.7   # 30% penalty
        
        # Combined multiplier
        final_multiplier = frequency_multiplier * win_rate_multiplier
        
        # Apply multiplier to base objective
        final_objective = base_objective * final_multiplier
        
        if self.verbose and final_multiplier != 1.0:
            print(f"      📊 Final multiplier: {final_multiplier:.2f} (freq: {frequency_multiplier:.2f}, win_rate: {win_rate_multiplier:.2f})")
            print(f"      🎯 Final objective: {final_objective:.4f} (base: {base_objective:.4f})")
        
        return final_objective
    
    def _sample_random_params(self) -> Dict[str, float]:
        """Sample random parameters from the parameter space with validation"""
        params = {}
        max_attempts = 50  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            # Sample all parameters
            for param_name, (low, high) in self.param_space.param_bounds.items():
                params[param_name] = random.uniform(low, high)
            
            # Add fixed parameters
            params.update(self.param_space.fixed_params)
            
            # Validate parameter relationships for trading viability
            is_valid = True
            
            # 1. Take-profit > stop-loss for positive expectancy (key requirement)
            if params.get('take_profit_pct', 0) <= params.get('stop_loss_pct', 0):
                is_valid = False
            
            # 2. Buy threshold should be reasonable relative to sell threshold
            buy_thresh = params.get('buy_threshold', 0.5)
            sell_thresh = params.get('sell_threshold', 0.5)
            if buy_thresh <= sell_thresh:  # Buy should be > sell for logical trading
                is_valid = False
                
            # 3. Risk per trade should be reasonable relative to position size
            risk_per_trade = params.get('risk_per_trade', 0.02)
            max_capital_per_trade = params.get('max_capital_per_trade', 0.1)
            if risk_per_trade > max_capital_per_trade:  # Risk shouldn't exceed position size
                is_valid = False
            
            if is_valid:
                return params  # Valid parameters
            
        # If we couldn't find valid params after many attempts, force fix the last attempt
        if 'take_profit_pct' in params and 'stop_loss_pct' in params:
            # Ensure take_profit is at least 20% larger than stop_loss
            if params['take_profit_pct'] <= params['stop_loss_pct']:
                params['take_profit_pct'] = params['stop_loss_pct'] * 1.2
        
        # Fix buy/sell threshold relationship
        if 'buy_threshold' in params and 'sell_threshold' in params:
            if params['buy_threshold'] <= params['sell_threshold']:
                # Make buy threshold slightly higher than sell threshold
                params['buy_threshold'] = params['sell_threshold'] + 0.001
                
        # Fix risk/position size relationship  
        if 'risk_per_trade' in params and 'max_capital_per_trade' in params:
            if params['risk_per_trade'] > params['max_capital_per_trade']:
                params['risk_per_trade'] = params['max_capital_per_trade'] * 0.8
                
        return params
    
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to numpy array for GP"""
        return np.array([params[name] for name in self.param_space.param_names])
    
    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to parameter dictionary with validation"""
        params = dict(zip(self.param_space.param_names, arr))
        params.update(self.param_space.fixed_params)
        
        # Apply parameter validation and corrections
        
        # 1. Validate take-profit > stop-loss for positive expectancy
        if 'take_profit_pct' in params and 'stop_loss_pct' in params:
            if params['take_profit_pct'] <= params['stop_loss_pct']:
                # Adjust take_profit to be at least 20% larger than stop_loss
                params['take_profit_pct'] = params['stop_loss_pct'] * 1.2
                # Ensure it stays within bounds
                _, max_tp = self.param_space.param_bounds['take_profit_pct']
                if params['take_profit_pct'] > max_tp:
                    # If take_profit would exceed bounds, reduce stop_loss instead
                    params['stop_loss_pct'] = params['take_profit_pct'] / 1.2
        
        # 2. Ensure buy threshold > sell threshold for logical trading
        if 'buy_threshold' in params and 'sell_threshold' in params:
            if params['buy_threshold'] <= params['sell_threshold']:
                # Make buy threshold slightly higher than sell threshold
                params['buy_threshold'] = params['sell_threshold'] + 0.001
                # Ensure it stays within bounds
                _, max_buy = self.param_space.param_bounds['buy_threshold']
                if params['buy_threshold'] > max_buy:
                    params['sell_threshold'] = params['buy_threshold'] - 0.001
        
        # 3. Ensure risk per trade <= position size
        if 'risk_per_trade' in params and 'max_capital_per_trade' in params:
            if params['risk_per_trade'] > params['max_capital_per_trade']:
                params['risk_per_trade'] = params['max_capital_per_trade'] * 0.8
                    
        return params
    
    def _optimize_acquisition(self) -> Dict[str, float]:
        """Optimize acquisition function to find next parameter combination"""
        # Ensure we have enough data and the scaler is fitted
        if len(self.evaluated_params) < 3:
            return self._sample_random_params()
        
        def acquisition_function(x):
            """Expected Improvement acquisition function"""
            try:
                x_scaled = self.param_scaler.transform(x.reshape(1, -1))
                mu, sigma = self.gp.predict(x_scaled, return_std=True)
                
                if sigma == 0:
                    return 0
                
                # Expected Improvement
                current_best = max(self.evaluation_results) if self.evaluation_results else 0
                xi = 0.01  # Exploration parameter
                
                with np.errstate(divide='ignore'):
                    improvement = mu - current_best - xi
                    Z = improvement / sigma
                    ei = improvement * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
                    
                return -ei[0]  # Minimize negative EI
            except Exception:
                return 1000  # Large penalty for invalid evaluations
        
        # Random restart optimization
        best_x = None
        best_acquisition = float('inf')
        
        for _ in range(10):  # Multiple random starts
            # Random starting point
            x0 = np.array([random.uniform(low, high) for low, high in self.param_space.bounds])
            
            # Optimize acquisition function
            result = minimize(
                acquisition_function,
                x0,
                bounds=self.param_space.bounds,
                method='L-BFGS-B'
            )
            
            if result.success and result.fun < best_acquisition:
                best_acquisition = result.fun
                best_x = result.x
        
        if best_x is None:
            # Fallback to random sampling
            return self._sample_random_params()
        
        return self._array_to_params(best_x)
    
    def _normal_cdf(self, x):
        """Standard normal CDF"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _normal_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def debug_model_outputs(symbol, days=7):
    """Debug raw model outputs before thresholds are applied"""
    from paper_trader.config.settings import TradingSettings
    from paper_trader.data.bitvavo_collector import BitvavoDataCollector
    from paper_trader.models.feature_engineer import FeatureEngineer
    from paper_trader.models.model_loader import WindowBasedModelLoader, WindowBasedEnsemblePredictor
    import pandas as pd
    import asyncio

    settings = TradingSettings()

    # Initialize data collector with proper API credentials
    data_collector = BitvavoDataCollector(
        api_key=settings.bitvavo_api_key,
        api_secret=settings.bitvavo_api_secret,
        interval=settings.candle_interval,
        settings=settings
    )

    feature_engineer = FeatureEngineer()
    model_loader = WindowBasedModelLoader(settings.model_path)
    predictor = WindowBasedEnsemblePredictor(
        model_loader=model_loader,
        settings=settings
    )

    # Get historical data
    print(f"Getting {days} days of historical data for {symbol}...")
    data = asyncio.run(data_collector.get_historical_data(symbol, limit=days*1440))  # days * minutes per day

    if data is None or len(data) < 100:
        print(f"❌ Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
        return

    # Engineer features
    features_df = feature_engineer.engineer_features(data)
    if features_df is None:
        print("❌ Feature engineering failed")
        return

    # Make predictions
    predictions = []
    confidences = []
    signals = []

    for i in range(min(100, len(features_df) - settings.sequence_length)):
        window = features_df.iloc[i:i+settings.sequence_length]
        pred_result = predictor.predict(window)
        if pred_result:
            predictions.append(pred_result)
            confidences.append(pred_result.get('confidence', 0))
            signals.append(pred_result.get('signal_strength', 'NONE'))

    # Analyze distribution
    if confidences:
        print("\n====== MODEL OUTPUT DISTRIBUTION ======")
        print(f"Total predictions: {len(confidences)}")
        print(f"Confidence min: {min(confidences):.4f}")
        print(f"Confidence max: {max(confidences):.4f}")
        print(f"Confidence mean: {sum(confidences)/len(confidences):.4f}")
        print(f"Confidence median: {sorted(confidences)[len(confidences)//2]:.4f}")
        print("\nConfidence histogram:")
        
        # Simple histogram
        bins = {'0.45-0.47': 0, '0.47-0.49': 0, '0.49-0.50': 0, '0.50-0.51': 0, '0.51-0.53': 0, '0.53-0.55': 0, '0.55+': 0}
        for c in confidences:
            if c < 0.47:
                bins['0.45-0.47'] += 1
            elif c < 0.49:
                bins['0.47-0.49'] += 1
            elif c < 0.50:
                bins['0.49-0.50'] += 1
            elif c < 0.51:
                bins['0.50-0.51'] += 1
            elif c < 0.53:
                bins['0.51-0.53'] += 1
            elif c < 0.55:
                bins['0.53-0.55'] += 1
            else:
                bins['0.55+'] += 1
        
        for bin_name, count in bins.items():
            print(f"{bin_name}: {'#' * (count // 2)} ({count})")
        
        # Signal strength distribution
        print("\nSignal strength distribution:")
        signal_counts = {}
        for s in signals:
            if s not in signal_counts:
                signal_counts[s] = 0
            signal_counts[s] += 1
        
        for signal, count in signal_counts.items():
            print(f"{signal}: {count} ({count/len(signals)*100:.1f}%)")

    else:
        print("❌ No predictions generated")

def create_optimized_env_file(symbols: List[str], best_params: Dict[str, float], 
                            optimization_results: Dict[str, Any]) -> str:
    """
    Create an optimized .env file with the best parameters found
    
    Args:
        symbols: List of symbols optimized for
        best_params: Best parameter configuration
        optimization_results: Full optimization results
        
    Returns:
        Path to the created .env file
    """
    # Create filename with symbols and date
    symbols_str = "_".join(symbols)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f".env_{symbols_str}_{date_str}"
    
    # Load base .env.example as template
    base_env_path = Path(".env.example")
    if base_env_path.exists():
        with open(base_env_path, 'r') as f:
            env_content = f.read()
    else:
        env_content = "# Optimized trading configuration\n"
    
    # Create optimized configuration content
    optimization_comment = f"""
# =============================================================================
# SCIENTIFICALLY OPTIMIZED TRADING CONFIGURATION
# =============================================================================
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Symbols: {', '.join(symbols)}
# Optimization method: {optimization_results.get('method', 'bayesian')}
# Objective: {optimization_results.get('objective', 'profit_factor')}
# Best objective value: {optimization_results.get('best_objective', 'N/A')}
# Total trades in backtest: {optimization_results.get('total_trades', 'N/A')}
# =============================================================================

"""
    
    # Map our optimization parameters to .env variables
    env_mappings = {
        'buy_threshold': 'MIN_CONFIDENCE_THRESHOLD',
        'risk_per_trade': 'BASE_POSITION_SIZE',
        'stop_loss_pct': 'STOP_LOSS_PCT',
        'take_profit_pct': 'TAKE_PROFIT_PCT',
        'max_capital_per_trade': 'MAX_POSITION_SIZE',
        'max_positions': 'MAX_POSITIONS',
        'trading_fee': 'TRADING_FEE',
        'max_trades_per_hour': 'MAX_DAILY_TRADES_PER_SYMBOL',
    }
    
    # Update environment variables with optimized values
    updated_env_content = optimization_comment + env_content
    
    for param_name, param_value in best_params.items():
        if param_name in env_mappings:
            env_var = env_mappings[param_name]
            
            # Handle special cases
            if param_name == 'buy_threshold':
                # Convert buy threshold to confidence threshold
                confidence_value = param_value
            elif param_name == 'max_trades_per_hour':
                # Convert to daily trades (multiply by reasonable factor)
                daily_trades = int(param_value * 8)  # Assume 8 trading hours per day
                param_value = daily_trades
            
            # Update or add the environment variable
            pattern = rf'^{env_var}=.*$'
            replacement = f'{env_var}={param_value}'
            
            if re.search(pattern, updated_env_content, re.MULTILINE):
                updated_env_content = re.sub(pattern, replacement, updated_env_content, flags=re.MULTILINE)
            else:
                updated_env_content += f'\n{replacement}'
    
    # Update symbols list
    symbols_env = ','.join([s.replace('EUR', '-EUR') for s in symbols])
    pattern = r'^SYMBOLS=.*$'
    replacement = f'SYMBOLS={symbols_env}'
    
    if re.search(pattern, updated_env_content, re.MULTILINE):
        updated_env_content = re.sub(pattern, replacement, updated_env_content, flags=re.MULTILINE)
    else:
        updated_env_content += f'\nSYMBOLS={symbols_env}'
    
    # Write the optimized .env file
    with open(filename, 'w') as f:
        f.write(updated_env_content)
    
    return filename


def main():
    """Main entry point for the optimization script"""
    parser = argparse.ArgumentParser(
        description='Scientific Parameter Optimization for Paper Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbols BTCEUR
  %(prog)s --symbols BTCEUR ETHEUR
  %(prog)s --symbols BTCEUR ETHEUR ADAEUR --method bayesian --iterations 100
  %(prog)s --symbols ETHEUR --mode aggressive --objective total_return
        """
    )
    
    parser.add_argument('--symbols', nargs='+', required=True,
                       choices=['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR'],
                       help='Cryptocurrency symbols to optimize for')
    
    parser.add_argument('--method', choices=['bayesian', 'grid'], default='bayesian',
                       help='Optimization method (default: bayesian)')
    
    parser.add_argument('--mode', choices=['conservative', 'balanced', 'aggressive', 'profit_focused', 'high_frequency'],
                       default='high_frequency',
                       help='Optimization mode determining parameter ranges (default: high_frequency)')
    
    parser.add_argument('--objective', choices=['sharpe_ratio', 'total_return', 'calmar_ratio', 'profit_factor'],
                       default='profit_factor',
                       help='Optimization objective (default: profit_factor)')
    
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of optimization iterations (default: 50)')
    
    parser.add_argument('--initial-random', type=int, default=10,
                       help='Number of initial random evaluations for Bayesian optimization (default: 10)')
    
    parser.add_argument('--grid-points', type=int, default=3,
                       help='Number of grid points per dimension for grid search (default: 3)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    parser.add_argument('--debug_model_output', nargs=2, metavar=('SYMBOL', 'DAYS'),
                       help='Debug model outputs for a symbol. Usage: --debug_model_output BTC-EUR 7')
    
    args = parser.parse_args()
    
    # Handle debug mode
    if args.debug_model_output:
        symbol = args.debug_model_output[0]
        days = int(args.debug_model_output[1])
        print(f"🔍 Debug mode: Analyzing {symbol} for {days} days")
        debug_model_outputs(symbol, days)
        return 0
    
    # Validate arguments
    if args.method == 'bayesian' and args.iterations < args.initial_random:
        parser.error(f"--iterations ({args.iterations}) must be >= --initial-random ({args.initial_random})")
    
    # Print header
    if not args.quiet:
        print("=" * 80)
        print("🔬 SCIENTIFIC PARAMETER OPTIMIZATION FOR PAPER TRADING BOT")
        print("=" * 80)
        print(f"📊 Symbols: {', '.join(args.symbols)}")
        print(f"🔬 Method: {args.method}")
        print(f"🎯 Mode: {args.mode}")
        print(f"📈 Objective: {args.objective}")
        if args.method == 'bayesian':
            print(f"🔄 Iterations: {args.iterations} (including {args.initial_random} random)")
        else:
            total_combinations = args.grid_points ** 8  # Approximate for 8 parameters
            print(f"🔄 Grid combinations: ~{total_combinations}")
        print()
    
    # Check data availability
    missing_data = []
    for symbol in args.symbols:
        data_file = f"data/{symbol.lower()}_15m.db"
        if not os.path.exists(data_file):
            missing_data.append(symbol)
    
    if missing_data:
        print(f"❌ Missing data files for symbols: {', '.join(missing_data)}")
        print("Please ensure the data files exist in the 'data/' directory.")
        return 1
    
    # Check model availability
    missing_models = []
    for symbol in args.symbols:
        lstm_models = f"models/lstm/{symbol.lower()}_window_*.keras"
        xgb_models = f"models/xgboost/{symbol.lower()}_window_*.json"
        
        import glob
        if not glob.glob(lstm_models) or not glob.glob(xgb_models):
            missing_models.append(symbol)
    
    if missing_models:
        print(f"⚠️  Warning: Limited models available for symbols: {', '.join(missing_models)}")
        print("Optimization will continue but results may be limited.")
        print()
    
    # Initialize optimizer
    try:
        optimizer = ScientificOptimizer(
            symbols=args.symbols,
            optimization_mode=args.mode,
            objective=args.objective,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"❌ Failed to initialize optimizer: {e}")
        return 1
    
    # Run optimization
    start_time = time.time()
    
    try:
        if args.method == 'bayesian':
            best_result = optimizer.bayesian_optimize(
                n_iterations=args.iterations,
                n_initial=args.initial_random
            )
        else:  # grid search
            best_result = optimizer.grid_search(
                n_points_per_dim=args.grid_points
            )
        
        optimization_time = time.time() - start_time
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        optimization_time = time.time() - start_time
        best_result = optimizer.best_result or {}
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Report results
    if not args.quiet:
        print("\n" + "=" * 80)
        print("🎉 OPTIMIZATION COMPLETED")
        print("=" * 80)
        print(f"⏰ Total time: {optimization_time/60:.1f} minutes")
        print(f"🔄 Evaluations completed: {optimizer.iteration_count}")
    
    if best_result and best_result.get('success'):
        if not args.quiet:
            print(f"🏆 Best {args.objective}: {best_result['objective_value']:.4f}")
            print(f"📊 Total trades: {best_result['metrics'].get('total_trades', 0)}")
            print(f"📈 Performance metrics:")
            for metric, value in best_result['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
            
            print(f"\n🎯 Optimal parameters:")
            for param, value in best_result['params'].items():
                if param not in optimizer.param_space.fixed_params:
                    print(f"   {param}: {value:.4f}")
        
        # Create optimized .env file
        try:
            optimization_results = {
                'method': args.method,
                'objective': args.objective,
                'best_objective': best_result['objective_value'],
                'total_trades': best_result['metrics'].get('total_trades', 0)
            }
            
            env_file = create_optimized_env_file(
                args.symbols,
                best_result['params'],
                optimization_results
            )
            
            print(f"\n📁 Optimized configuration saved to: {env_file}")
            print(f"🚀 Use this file with main_paper_trader.py for optimal performance!")
            
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to create .env file: {e}")
    
    else:
        print("❌ No valid optimization results found")
        print("This may be due to:")
        print("- Insufficient data or models")
        print("- Parameter ranges too restrictive")
        print("- All backtests failing due to configuration issues")
        return 1
    
    if not args.quiet:
        print("\n" + "=" * 80)
        print("Happy trading! 🚀")
        print("=" * 80)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
