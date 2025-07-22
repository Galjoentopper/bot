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
            optimization_mode: 'conservative', 'balanced', 'aggressive', 'profit_focused'
        """
        # Base parameter ranges - EXTREMELY aggressive for 20+ trades per month target
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
        
        # Fixed parameters based on research and practical constraints
        self.fixed_params = {
            'trading_fee': 0.002,        # Realistic crypto trading fees
            'slippage': 0.001,           # Conservative slippage estimate
            'max_trades_per_hour': 3,    # Prevent overtrading
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
        
        # Switch to very aggressive parameter ranges
        self.param_space.param_bounds = {
            'buy_threshold': (0.502, 0.55),          # Extremely low for maximum trades
            'sell_threshold': (0.45, 0.498),          # Extremely high for maximum trades
            'lstm_delta_threshold': (0.0001, 0.005),  # Ultra-sensitive
            'risk_per_trade': (0.01, 0.03),          # Moderate risk
            'stop_loss_pct': (0.015, 0.04),          # Balanced stop loss
            'take_profit_pct': (0.03, 0.08),         # Conservative targets for reliability
            'max_capital_per_trade': (0.05, 0.15),   # Balanced positions
            'max_positions': (8, 15),                # More positions allowed
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
        """Check if we're getting zero trades and adapt if needed"""
        if len(self.evaluation_results) >= 2:  # Check after just 2 evaluations for faster adaptation
            # Count recent evaluations with zero trades (objective = -999)
            recent_results = self.evaluation_results[-2:]
            zero_trade_count = sum(1 for result in recent_results if result <= -990)
            
            if zero_trade_count >= 1 and not self.aggressive_mode_enabled:  # Switch after just 1 zero-trade result
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
        
        # Collect all valid performance metrics
        valid_performances = []
        for symbol, result in results.items():
            if result and 'performance' in result and result['performance']:
                valid_performances.append(result['performance'])
        
        if not valid_performances:
            return {'total_trades': 0, 'total_return': -1, 'sharpe_ratio': -1, 'max_drawdown': -0.5}
        
        # Aggregate metrics
        metrics['total_trades'] = sum(p.get('total_trades', 0) for p in valid_performances)
        metrics['total_return'] = np.mean([p.get('total_return', 0) for p in valid_performances])
        metrics['sharpe_ratio'] = np.mean([p.get('sharpe_ratio', 0) for p in valid_performances])
        metrics['max_drawdown'] = np.mean([p.get('max_drawdown', 0) for p in valid_performances])
        metrics['win_rate'] = np.mean([p.get('win_rate', 0) for p in valid_performances])
        metrics['profit_factor'] = np.mean([p.get('profit_factor', 0) for p in valid_performances])
        
        return metrics
    
    def _calculate_objective(self, metrics: Dict[str, float]) -> float:
        """Calculate objective value from metrics"""
        total_trades = metrics.get('total_trades', 0)
        
        # Very permissive trade requirement - just need at least 1 trade for valid evaluation
        if total_trades < 1:
            if self.verbose:
                print(f"      ❌ No trades generated - returning penalty")
            return -999
        elif total_trades < 3:
            if self.verbose:
                print(f"      ⚠️  Few trades ({total_trades}) but allowing evaluation with penalty")
            # Small penalty for few trades but still allow evaluation
            penalty_factor = 0.5  # 50% penalty for having very few trades
        else:
            penalty_factor = 1.0  # No penalty for sufficient trades
        
        base_objective = 0
        if self.objective == 'sharpe_ratio':
            base_objective = metrics.get('sharpe_ratio', -999)
        elif self.objective == 'total_return':
            base_objective = metrics.get('total_return', -999)
        elif self.objective == 'calmar_ratio':
            total_return = metrics.get('total_return', 0)
            max_drawdown = abs(metrics.get('max_drawdown', 0.01))
            base_objective = total_return / max_drawdown if max_drawdown > 0 else -999
        elif self.objective == 'profit_factor':
            base_objective = metrics.get('profit_factor', -999)
        else:
            base_objective = metrics.get('total_return', -999)  # Default to total return
        
        # Apply penalty factor for few trades
        return base_objective * penalty_factor if base_objective > -990 else base_objective
    
    def _sample_random_params(self) -> Dict[str, float]:
        """Sample random parameters from the parameter space"""
        params = {}
        for param_name, (low, high) in self.param_space.param_bounds.items():
            params[param_name] = random.uniform(low, high)
        
        # Add fixed parameters
        params.update(self.param_space.fixed_params)
        return params
    
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to numpy array for GP"""
        return np.array([params[name] for name in self.param_space.param_names])
    
    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to parameter dictionary"""
        params = dict(zip(self.param_space.param_names, arr))
        params.update(self.param_space.fixed_params)
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

# Add this to your optimization script to see raw model outputs
def debug_model_outputs(symbol, days=7):
    """Debug raw model outputs before thresholds are applied"""
    from paper_trader.config.settings import TradingSettings
    from paper_trader.data.bitvavo_collector import BitvavoDataCollector
    from paper_trader.models.feature_engineer import FeatureEngineer
    from paper_trader.models.model_loader import WindowBasedModelLoader, WindowBasedEnsemblePredictor
    import pandas as pd
    import asyncio
    
    settings = TradingSettings()
    data_collector = BitvavoDataCollector(settings)
    feature_engineer = FeatureEngineer()
    model_loader = WindowBasedModelLoader(settings)
    predictor = WindowBasedEnsemblePredictor(model_loader, settings)
    
    # Get historical data
    print(f"Getting {days} days of historical data for {symbol}...")
    asyncio.run(data_collector.initialize())
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
    
    parser.add_argument('--mode', choices=['conservative', 'balanced', 'aggressive', 'profit_focused'],
                       default='profit_focused',
                       help='Optimization mode determining parameter ranges (default: profit_focused)')
    
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
