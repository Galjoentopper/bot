#!/usr/bin/env python3
"""Parameter Optimization Engine for backtesting configurations."""
from __future__ import annotations

import itertools
import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

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
        combos = self._generate_parameter_sets()
        results = []
        for params in combos:
            result = self._evaluate_params(params, symbols)
            if result:
                results.append(result)

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

        backtester = ModelBacktester(cfg)
        try:
            run_results = backtester.run_backtest(symbols)
        except Exception as exc:  # pragma: no cover - runtime failure
            print(f"Error evaluating params {params}: {exc}")
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
            return
        os.makedirs("optimization_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("optimization_results", f"optimization_{timestamp}.json")
        top_n = results[: self.config.save_top_n]
        with open(file_path, "w") as f:
            json.dump(top_n, f, indent=2)
        print(f"Saved top {len(top_n)} results to {file_path}")
