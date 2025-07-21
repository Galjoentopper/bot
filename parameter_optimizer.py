#!/usr/bin/env python3
"""Parameter Optimization Engine for backtesting configurations."""
from __future__ import annotations

import itertools
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm
import psutil

# Import the backtesting engine
from scripts.backtest_models import ModelBacktester, BacktestConfig


@dataclass
class OptimizationConfig:
    """Configuration options for the optimization engine."""

    method: str = "grid_search"  # grid_search, random_search, bayesian
    n_iterations: int = 50  # used for random/bayesian search
    n_jobs: int = 1
    objective: str = "sharpe_ratio"
    min_trades: int = 10
    save_top_n: int = 5


class ParameterOptimizer:
    """Runs backtests for many parameter combinations to find the best settings."""

    def __init__(self, config: OptimizationConfig, base_config: BacktestConfig | None = None):
        self.config = config
        self.base_config = base_config or BacktestConfig()
        
        # Performance tracking
        self.start_time = None
        self.current_best = None
        self.best_score = float('-inf')
        self.total_evaluations = 0
        self.successful_evaluations = 0
        
        # Memory tracking
        self.process = psutil.Process()
        self.initial_memory = None

        # Default search space
        self.grid_space: Dict[str, List[Any]] = {
            "buy_threshold": [0.6, 0.7, 0.8],
            "sell_threshold": [0.2, 0.3, 0.4],
            "lstm_delta_threshold": [0.01, 0.02],
            "risk_per_trade": [0.01, 0.02],
            "stop_loss_pct": [0.02, 0.03],
        }

    # ------------------------------------------------------------------
    def run_optimization(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Run optimization over the defined parameter space."""
        print("🤖 Starting Parameter Optimization")
        print("=" * 60)
        
        # Initialize performance tracking
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Display optimization setup
        self._display_optimization_setup(symbols)
        
        # Generate parameter combinations
        print("\n📊 Generating parameter combinations...")
        combos = self._generate_parameter_sets()
        total_combos = len(combos)
        
        print(f"✅ Generated {total_combos:,} parameter combinations")
        print(f"🎯 Optimization method: {self.config.method}")
        print(f"🎪 Target objective: {self.config.objective}")
        print(f"🔢 Minimum trades required: {self.config.min_trades}")
        
        # Initialize results tracking
        results = []
        self.total_evaluations = 0
        self.successful_evaluations = 0
        
        # Create progress bar
        progress_bar = tqdm(
            combos, 
            desc="🔍 Optimizing parameters",
            unit="combo",
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        print(f"\n🚀 Starting optimization of {total_combos:,} combinations...")
        print("💡 Real-time updates will appear below:")
        print("-" * 60)
        
        for i, params in enumerate(progress_bar):
            self.total_evaluations += 1
            
            # Update progress bar with current best
            if self.current_best:
                progress_desc = f"🔍 Best {self.config.objective}: {self.best_score:.4f}"
            else:
                progress_desc = "🔍 Searching for optimal parameters"
            progress_bar.set_description(progress_desc)
            
            # Evaluate parameters
            result = self._evaluate_params(params, symbols)
            
            if result:
                self.successful_evaluations += 1
                results.append(result)
                
                # Check if this is a new best result
                if result["objective_value"] > self.best_score:
                    self.best_score = result["objective_value"]
                    self.current_best = result
                    self._display_new_best_result(result, i + 1, total_combos)
            
            # Display progress updates every 10% or significant milestones
            if (i + 1) % max(1, total_combos // 10) == 0 or (i + 1) in [1, 5, 10]:
                self._display_progress_update(i + 1, total_combos)
        
        progress_bar.close()
        
        # Sort by objective descending
        results.sort(key=lambda x: x["objective_value"], reverse=True)
        
        # Display final results
        self._display_final_results(results, total_combos)
        
        # Save results
        self._save_results(results)
        
        return results

    # ------------------------------------------------------------------
    def _generate_parameter_sets(self) -> List[Dict[str, Any]]:
        keys = list(self.grid_space.keys())
        values = [self.grid_space[k] for k in keys]
        all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if self.config.method == "grid_search":
            return all_combos

        if self.config.method in {"random_search", "bayesian"}:
            random.shuffle(all_combos)
            return all_combos[: self.config.n_iterations]

        # fallback
        return all_combos

    # ------------------------------------------------------------------
    def _evaluate_params(self, params: Dict[str, Any], symbols: List[str]) -> Dict[str, Any] | None:
        """Evaluate a parameter combination with enhanced error handling."""
        cfg = deepcopy(self.base_config)
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        backtester = ModelBacktester(cfg)
        try:
            # Track evaluation time
            eval_start = time.time()
            run_results = backtester.run_backtest(symbols)
            eval_time = time.time() - eval_start
            
        except Exception as exc:
            # Enhanced error reporting
            error_msg = f"❌ Error evaluating parameters: {str(exc)}"
            if hasattr(exc, '__class__'):
                error_msg += f" (Type: {exc.__class__.__name__})"
            print(f"\n{error_msg}")
            print(f"   Parameters: {params}")
            return None

        metrics = [res["performance"] for res in run_results.values() if res.get("performance")]
        if not metrics:
            print(f"⚠️  No performance metrics found for parameters: {params}")
            return None

        # Aggregate metrics across symbols
        agg = {k: np.mean([m.get(k, 0) for m in metrics]) for k in metrics[0].keys()}
        objective_value = self._extract_objective(agg)
        total_trades = sum(m.get("total_trades", 0) for m in metrics)

        if total_trades < self.config.min_trades:
            return None

        return {
            "params": params,
            "metrics": agg,
            "objective_value": objective_value,
            "total_trades": total_trades,
            "evaluation_time": eval_time,
            "symbols_tested": len(symbols)
        }

    # ------------------------------------------------------------------
    def _extract_objective(self, metrics: Dict[str, Any]) -> float:
        obj = self.config.objective
        if obj == "calmar_ratio":
            dd = metrics.get("max_drawdown", 0)
            tr = metrics.get("total_return", 0)
            return tr / abs(dd) if dd else 0
        if obj == "portfolio_return":
            return metrics.get("total_return", 0)
        return metrics.get(obj, 0)

    # ------------------------------------------------------------------
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save optimization results with enhanced formatting."""
        if not results:
            print("⚠️  No results to save!")
            return
            
        os.makedirs("optimization_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("optimization_results", f"optimization_{timestamp}.json")
        
        # Prepare enhanced results for saving
        top_n = results[: self.config.save_top_n]
        
        # Add optimization metadata
        optimization_summary = {
            "optimization_metadata": {
                "timestamp": timestamp,
                "method": self.config.method,
                "objective": self.config.objective,
                "total_evaluations": self.total_evaluations,
                "successful_evaluations": self.successful_evaluations,
                "success_rate": self.successful_evaluations / self.total_evaluations if self.total_evaluations > 0 else 0,
                "optimization_duration_seconds": time.time() - self.start_time if self.start_time else 0,
                "min_trades_required": self.config.min_trades
            },
            "top_results": top_n
        }
        
        with open(file_path, "w") as f:
            json.dump(optimization_summary, f, indent=2)
        
        print(f"\n💾 Saved top {len(top_n)} results to: {file_path}")
    
    # ------------------------------------------------------------------
    def _display_optimization_setup(self, symbols: List[str]):
        """Display the optimization setup information."""
        print(f"🎯 Symbols to optimize: {', '.join(symbols)}")
        print(f"🔧 Optimization method: {self.config.method}")
        print(f"📊 Objective function: {self.config.objective}")
        print(f"🧮 Parameter space dimensions: {len(self.grid_space)} parameters")
        
        # Display parameter ranges
        print("\n📋 Parameter ranges:")
        for param, values in self.grid_space.items():
            if isinstance(values, list) and len(values) > 0:
                if len(values) <= 3:
                    print(f"   • {param}: {values}")
                else:
                    print(f"   • {param}: [{values[0]} ... {values[-1]}] ({len(values)} values)")
        
        # Display system info
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        print(f"\n💻 System status:")
        print(f"   • CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
        print(f"   • Memory usage: {memory_mb:.1f} MB")
        print(f"   • Available CPU cores: {psutil.cpu_count()}")
    
    # ------------------------------------------------------------------
    def _display_new_best_result(self, result: Dict[str, Any], current_eval: int, total_evals: int):
        """Display information about a new best result."""
        params = result['params']
        metrics = result['metrics']
        
        print(f"\n🏆 NEW BEST RESULT! (Evaluation {current_eval}/{total_evals})")
        print(f"   📈 {self.config.objective}: {result['objective_value']:.6f}")
        print(f"   📊 Total trades: {result['total_trades']}")
        
        # Display key parameters in a compact format
        key_params = ['buy_threshold', 'sell_threshold', 'risk_per_trade', 'stop_loss_pct']
        param_str = ", ".join([f"{k}={params.get(k, 'N/A')}" for k in key_params if k in params])
        print(f"   🎛️  Key params: {param_str}")
        
        # Display key metrics if available
        if 'total_return' in metrics:
            print(f"   💰 Total return: {metrics['total_return']:.4f}")
        if 'max_drawdown' in metrics:
            print(f"   📉 Max drawdown: {metrics['max_drawdown']:.4f}")
    
    # ------------------------------------------------------------------
    def _display_progress_update(self, current: int, total: int):
        """Display periodic progress updates."""
        elapsed = time.time() - self.start_time
        progress_pct = (current / total) * 100
        
        # Calculate ETA
        if current > 0:
            avg_time_per_eval = elapsed / current
            remaining_evals = total - current
            eta_seconds = avg_time_per_eval * remaining_evals
            eta_str = f"{eta_seconds/60:.1f} min" if eta_seconds > 60 else f"{eta_seconds:.0f} sec"
        else:
            eta_str = "calculating..."
        
        # Memory usage
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_delta = current_memory - self.initial_memory if self.initial_memory else 0
        
        print(f"\n📊 Progress Update ({progress_pct:.1f}% complete)")
        print(f"   ⏱️  Elapsed: {elapsed/60:.1f} min | ETA: {eta_str}")
        print(f"   ✅ Successful evaluations: {self.successful_evaluations}/{self.total_evaluations}")
        print(f"   🧠 Memory: {current_memory:.1f} MB ({memory_delta:+.1f} MB)")
        if self.current_best:
            print(f"   🏆 Current best {self.config.objective}: {self.best_score:.6f}")
    
    # ------------------------------------------------------------------
    def _display_final_results(self, results: List[Dict[str, Any]], total_combos: int):
        """Display comprehensive final results."""
        elapsed = time.time() - self.start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - self.initial_memory if self.initial_memory else 0
        
        print("\n" + "="*80)
        print("🎉 OPTIMIZATION COMPLETED!")
        print("="*80)
        
        # Summary statistics
        print(f"⏱️  Total time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
        print(f"📊 Evaluations: {self.successful_evaluations}/{self.total_evaluations} successful")
        print(f"📈 Success rate: {(self.successful_evaluations/self.total_evaluations)*100:.1f}%")
        print(f"🧠 Memory used: {memory_used:+.1f} MB")
        print(f"⚡ Avg time per evaluation: {elapsed/max(1, self.total_evaluations):.2f} seconds")
        
        if not results:
            print("\n⚠️  No successful results found!")
            print("💡 Consider:")
            print("   • Reducing min_trades requirement")
            print("   • Adjusting parameter ranges")
            print("   • Checking data availability")
            return
        
        # Top results summary
        print(f"\n🏆 TOP {min(len(results), 5)} RESULTS:")
        print("-" * 80)
        
        for i, result in enumerate(results[:5], 1):
            params = result['params']
            metrics = result['metrics']
            
            print(f"\n#{i} - {self.config.objective}: {result['objective_value']:.6f}")
            print(f"    Trades: {result['total_trades']} | Eval time: {result.get('evaluation_time', 0):.2f}s")
            
            # Key parameters
            key_params = ['buy_threshold', 'sell_threshold', 'risk_per_trade', 'stop_loss_pct', 'take_profit_pct']
            for param in key_params:
                if param in params:
                    print(f"    {param}: {params[param]}")
            
            # Key metrics
            key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            metrics_str = []
            for metric in key_metrics:
                if metric in metrics:
                    metrics_str.append(f"{metric}: {metrics[metric]:.4f}")
            if metrics_str:
                print(f"    Metrics: {' | '.join(metrics_str)}")
        
        print("\n" + "="*80)
        print("📁 Check 'optimization_results/' directory for detailed JSON results")
        print("="*80)
