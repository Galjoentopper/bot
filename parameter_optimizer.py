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
        # Initialize performance tracking
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n🚀 Starting optimization with {self.config.method}")
        print(f"📊 Target symbols: {', '.join(symbols)}")
        print(f"🎯 Optimization objective: {self.config.objective}")
        print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
        
        combos = self._generate_parameter_sets()
        total_combos = len(combos)
        
        print(f"🔍 Total parameter combinations to evaluate: {total_combos}")
        print(f"⚡ Minimum trades required: {self.config.min_trades}")
        
        results = []
        best_score = float('-inf')
        best_params = None
        
        # Create progress bar
        with tqdm(total=total_combos, desc="Optimizing", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for i, params in enumerate(combos):
                # Update progress bar with current parameter info
                param_summary = f"buy_th={params.get('buy_threshold', 'N/A'):.2f}" if 'buy_threshold' in params else "params"
                pbar.set_postfix_str(f"Current: {param_summary}")
                
                try:
                    result = self._evaluate_params(params, symbols)
                    
                    if result:
                        results.append(result)
                        
                        # Track best result for real-time updates
                        if result['objective_value'] > best_score:
                            best_score = result['objective_value']
                            best_params = result['params']
                            
                            # Update progress bar with new best score
                            pbar.set_description(f"Optimizing (Best: {best_score:.4f})")
                    
                    # Memory monitoring every 10 iterations
                    if (i + 1) % 10 == 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_change = current_memory - initial_memory
                        pbar.set_postfix_str(f"{param_summary}, Mem: +{memory_change:.1f}MB")
                        
                except Exception as e:
                    # Enhanced error handling
                    error_msg = f"Error evaluating {param_summary}: {str(e)[:50]}..."
                    pbar.set_postfix_str(error_msg)
                    tqdm.write(f"⚠️  {error_msg}")
                
                pbar.update(1)

        # Final performance summary
        end_time = time.time()
        duration = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"\n📈 Optimization completed!")
        print(f"⏱️  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"💾 Final memory usage: {final_memory:.1f} MB (Δ{final_memory-initial_memory:+.1f} MB)")
        print(f"✅ Valid results: {len(results)}/{total_combos} ({len(results)/total_combos*100:.1f}%)")
        
        if results:
            print(f"🏆 Best {self.config.objective}: {best_score:.4f}")
            if best_params:
                key_params = ['buy_threshold', 'sell_threshold', 'risk_per_trade']
                best_summary = {k: v for k, v in best_params.items() if k in key_params}
                print(f"🎯 Best parameters preview: {best_summary}")

        # Sort by objective descending
        results.sort(key=lambda x: x["objective_value"], reverse=True)

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
        cfg = deepcopy(self.base_config)
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        
        # Suppress verbose output during optimization to avoid spam
        cfg.verbose = False

        backtester = ModelBacktester(cfg)
        try:
            run_results = backtester.run_backtest(symbols)
        except Exception as exc:  # pragma: no cover - runtime failure
            # Enhanced error reporting with context
            param_summary = f"buy_th={params.get('buy_threshold', 'N/A')}, symbols={len(symbols)}"
            error_type = type(exc).__name__
            print(f"❌ {error_type} evaluating params ({param_summary}): {str(exc)[:100]}")
            return None

        metrics = [res["performance"] for res in run_results.values() if res.get("performance")]
        if not metrics:
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
        if not results:
            print("⚠️  No valid results to save")
            return
            
        os.makedirs("optimization_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("optimization_results", f"optimization_{timestamp}.json")
        
        top_n = results[: self.config.save_top_n]
        
        # Add metadata to saved results
        save_data = {
            "metadata": {
                "timestamp": timestamp,
                "method": self.config.method,
                "objective": self.config.objective,
                "total_results": len(results),
                "saved_count": len(top_n),
                "min_trades": self.config.min_trades
            },
            "results": top_n
        }
        
        with open(file_path, "w") as f:
            json.dump(save_data, f, indent=2)
            
        print(f"💾 Saved top {len(top_n)} results to {file_path}")
        
        # Display summary of top results
        if top_n:
            print(f"\n📊 Top {min(3, len(top_n))} Results Summary:")
            print("-" * 60)
            for i, result in enumerate(top_n[:3], 1):
                obj_val = result['objective_value']
                trades = result['total_trades']
                key_params = result['params']
                buy_th = key_params.get('buy_threshold', 'N/A')
                sell_th = key_params.get('sell_threshold', 'N/A')
                risk = key_params.get('risk_per_trade', 'N/A')
                
                print(f"#{i}: {self.config.objective}={obj_val:.4f}, trades={trades}")
                print(f"    buy_th={buy_th}, sell_th={sell_th}, risk={risk}")
                print()
