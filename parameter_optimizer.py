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
import psutil
from tqdm import tqdm

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
        
        # Performance monitoring
        self.start_time = None
        self.best_score = float('-inf')
        self.best_params = None
        self.evaluation_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'evaluation_times': []
        }

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
        # Initialize monitoring
        self.start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n🚀 Starting optimization with {self.config.method}")
        print(f"📊 Optimization target: {self.config.objective}")
        print(f"🎯 Symbols: {', '.join(symbols)}")
        print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
        print("=" * 50)
        
        # Generate parameter combinations
        combos = self._generate_parameter_sets()
        total_combos = len(combos)
        
        print(f"🔢 Total parameter combinations to evaluate: {total_combos}")
        print(f"⚡ Parallel jobs: {self.config.n_jobs}")
        print(f"🎯 Minimum trades required: {self.config.min_trades}")
        print("=" * 50)
        
        results = []
        
        # Create progress bar
        pbar = tqdm(
            combos, 
            desc="🔍 Optimizing parameters",
            unit="combo",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Best: {postfix}"
        )
        
        for i, params in enumerate(pbar):
            iteration_start = time.time()
            
            try:
                result = self._evaluate_params(params, symbols)
                iteration_time = time.time() - iteration_start
                self.evaluation_stats['evaluation_times'].append(iteration_time)
                self.evaluation_stats['total_evaluations'] += 1
                
                if result:
                    self.evaluation_stats['successful_evaluations'] += 1
                    results.append(result)
                    
                    # Update best score if improved
                    if result['objective_value'] > self.best_score:
                        self.best_score = result['objective_value']
                        self.best_params = result['params']
                        
                        # Update progress bar with best score
                        pbar.set_postfix_str(f"{self.best_score:.4f}")
                else:
                    self.evaluation_stats['failed_evaluations'] += 1
                    
            except Exception as e:
                self.evaluation_stats['failed_evaluations'] += 1
                iteration_time = time.time() - iteration_start
                self.evaluation_stats['evaluation_times'].append(iteration_time)
                
                # Log error but continue optimization
                tqdm.write(f"⚠️  Warning: Evaluation failed for params {params}: {str(e)}")
                continue
            
            # Memory monitoring every 10 iterations
            if (i + 1) % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_change = current_memory - initial_memory
                
                # Update description with memory info
                success_rate = (self.evaluation_stats['successful_evaluations'] / 
                              self.evaluation_stats['total_evaluations'] * 100)
                avg_time = np.mean(self.evaluation_stats['evaluation_times'])
                
                pbar.set_description(
                    f"🔍 Optimizing | Mem: +{memory_change:+.1f}MB | "
                    f"Success: {success_rate:.1f}% | Avg: {avg_time:.2f}s"
                )
        
        pbar.close()
        
        # Sort by objective descending
        results.sort(key=lambda x: x["objective_value"], reverse=True)
        
        # Generate comprehensive results summary
        self._print_optimization_summary(results, symbols, total_combos, initial_memory)
        
        # Save top results
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
        """Evaluate a single parameter combination with enhanced error handling."""
        cfg = deepcopy(self.base_config)
        
        # Apply parameters to configuration
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                tqdm.write(f"⚠️  Warning: Unknown parameter '{k}' with value {v}")

        backtester = ModelBacktester(cfg)
        
        try:
            run_results = backtester.run_backtest(symbols)
        except FileNotFoundError as e:
            tqdm.write(f"❌ File not found error: {str(e)}")
            tqdm.write("💡 Suggestion: Ensure all required model files and data are present")
            return None
        except ValueError as e:
            tqdm.write(f"❌ Value error in parameters: {str(e)}")
            tqdm.write(f"💡 Suggestion: Check parameter ranges - {params}")
            return None
        except MemoryError:
            tqdm.write("❌ Out of memory error")
            tqdm.write("💡 Suggestion: Reduce batch size or number of parallel jobs")
            return None
        except Exception as exc:
            tqdm.write(f"❌ Unexpected error evaluating params {params}: {type(exc).__name__}: {str(exc)}")
            tqdm.write("💡 Suggestion: Check logs for detailed error information")
            return None

        metrics = [res["performance"] for res in run_results.values() if res.get("performance")]
        if not metrics:
            tqdm.write(f"⚠️  No performance metrics generated for params: {params}")
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
        }

    # ------------------------------------------------------------------
    def _print_optimization_summary(self, results: List[Dict[str, Any]], symbols: List[str], 
                                   total_combos: int, initial_memory: float):
        """Print comprehensive optimization results summary."""
        end_time = time.time()
        total_time = end_time - self.start_time
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_change = current_memory - initial_memory
        
        print("\n" + "=" * 60)
        print("🎉 OPTIMIZATION COMPLETED!")
        print("=" * 60)
        
        # Performance Statistics
        print("📊 PERFORMANCE STATISTICS:")
        print(f"   ⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   💾 Memory usage change: {memory_change:+.1f} MB")
        print(f"   🔢 Total evaluations: {self.evaluation_stats['total_evaluations']}")
        print(f"   ✅ Successful evaluations: {self.evaluation_stats['successful_evaluations']}")
        print(f"   ❌ Failed evaluations: {self.evaluation_stats['failed_evaluations']}")
        
        success_rate = (self.evaluation_stats['successful_evaluations'] / 
                       max(self.evaluation_stats['total_evaluations'], 1) * 100)
        print(f"   📈 Success rate: {success_rate:.1f}%")
        
        if self.evaluation_stats['evaluation_times']:
            avg_time = np.mean(self.evaluation_stats['evaluation_times'])
            min_time = np.min(self.evaluation_stats['evaluation_times'])
            max_time = np.max(self.evaluation_stats['evaluation_times'])
            print(f"   ⚡ Average evaluation time: {avg_time:.2f}s")
            print(f"   ⚡ Fastest evaluation: {min_time:.2f}s")
            print(f"   ⚡ Slowest evaluation: {max_time:.2f}s")
        
        print()
        
        # Optimization Configuration
        print("⚙️  OPTIMIZATION CONFIGURATION:")
        print(f"   🎯 Method: {self.config.method}")
        print(f"   📊 Objective: {self.config.objective}")
        print(f"   💰 Symbols: {', '.join(symbols)}")
        print(f"   🔢 Parameter combinations: {total_combos}")
        print(f"   ⚡ Parallel jobs: {self.config.n_jobs}")
        print(f"   📈 Minimum trades: {self.config.min_trades}")
        print()
        
        # Results Summary
        if results:
            print("🏆 TOP RESULTS:")
            top_results = results[:min(3, len(results))]
            
            for i, result in enumerate(top_results, 1):
                print(f"   #{i} - {self.config.objective}: {result['objective_value']:.4f}")
                print(f"       Trades: {result['total_trades']}")
                
                # Show key parameters
                key_params = ['buy_threshold', 'sell_threshold', 'risk_per_trade', 'stop_loss_pct']
                param_str = ", ".join([f"{k}: {result['params'].get(k, 'N/A')}" 
                                     for k in key_params if k in result['params']])
                print(f"       Key params: {param_str}")
                print()
            
            print(f"💾 Saved top {min(self.config.save_top_n, len(results))} results to optimization_results/")
        else:
            print("❌ NO VALID RESULTS FOUND")
            print("💡 Suggestions:")
            print("   - Check if all required data files are present")
            print("   - Verify parameter ranges are reasonable")
            print("   - Consider lowering min_trades requirement")
            print("   - Check model files and dependencies")
        
        print("=" * 60)

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
        """Save optimization results with enhanced metadata."""
        if not results:
            print("⚠️  No results to save")
            return
            
        os.makedirs("optimization_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("optimization_results", f"optimization_{timestamp}.json")
        
        # Prepare enhanced results with metadata
        enhanced_results = {
            "metadata": {
                "timestamp": timestamp,
                "optimization_method": self.config.method,
                "objective": self.config.objective,
                "total_evaluations": self.evaluation_stats['total_evaluations'],
                "successful_evaluations": self.evaluation_stats['successful_evaluations'],
                "failed_evaluations": self.evaluation_stats['failed_evaluations'],
                "success_rate_percent": (self.evaluation_stats['successful_evaluations'] / 
                                       max(self.evaluation_stats['total_evaluations'], 1) * 100),
                "execution_time_seconds": time.time() - self.start_time if self.start_time else 0,
                "min_trades_required": self.config.min_trades,
                "parallel_jobs": self.config.n_jobs
            },
            "performance_stats": {
                "average_evaluation_time": np.mean(self.evaluation_stats['evaluation_times']) if self.evaluation_stats['evaluation_times'] else 0,
                "fastest_evaluation": np.min(self.evaluation_stats['evaluation_times']) if self.evaluation_stats['evaluation_times'] else 0,
                "slowest_evaluation": np.max(self.evaluation_stats['evaluation_times']) if self.evaluation_stats['evaluation_times'] else 0
            },
            "top_results": results[:self.config.save_top_n]
        }
        
        try:
            with open(file_path, "w") as f:
                json.dump(enhanced_results, f, indent=2, default=str)
            print(f"💾 Saved top {len(enhanced_results['top_results'])} results to {file_path}")
            print(f"📊 Results include performance statistics and metadata")
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            print("💡 Suggestion: Check write permissions for optimization_results/ directory")
