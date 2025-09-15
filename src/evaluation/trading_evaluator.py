"""
Trading Model Evaluation Framework
=================================

Comprehensive evaluation framework for trading models with:
- Multi-metric evaluation
- Cross-validation for time series
- Walk-forward analysis
- Regime-based evaluation
- Risk-adjusted performance measurement
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ..utils.trading_metrics import (
    TradingMetricsCalculator,
    evaluate_trading_performance,
    optimize_trading_threshold,
)

logger = logging.getLogger(__name__)


class TradingModelEvaluator:
    """
    Comprehensive evaluator for trading models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trading model evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Evaluation parameters
        self.walk_forward_splits = self.config.get("walk_forward_splits", 5)
        self.min_train_size = self.config.get("min_train_size", 1000)
        self.test_size_ratio = self.config.get("test_size_ratio", 0.2)
        self.gap_size = self.config.get("gap_size", 0)  # Gap between train and test

        # Trading parameters
        self.initial_balance = self.config.get("initial_balance", 10000.0)
        self.transaction_cost = self.config.get("transaction_cost", 0.001)
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)

        # Regime analysis parameters
        self.volatility_window = self.config.get("volatility_window", 30)
        self.trend_window = self.config.get("trend_window", 20)

        # Metrics calculator
        self.metrics_calc = TradingMetricsCalculator(
            risk_free_rate=self.risk_free_rate,
            trading_days=365,  # Crypto trades 365 days
        )

        logger.info("Trading model evaluator initialized")

    def walk_forward_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        symbol: str = "Unknown",
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation for trading models.

        Args:
            model: Trained model with predict method
            X: Feature matrix
            y: Target values
            prices: Price series
            symbol: Trading symbol

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Starting walk-forward validation for {symbol}")

        if len(X) < self.min_train_size:
            logger.warning(
                f"Insufficient data for walk-forward validation: {len(X)} < {self.min_train_size}"
            )
            return {}

        # Create time series splits
        tscv = TimeSeriesSplit(
            n_splits=self.walk_forward_splits,
            test_size=int(len(X) * self.test_size_ratio),
            gap=self.gap_size,
        )

        fold_results = []
        all_predictions = []
        all_actuals = []
        all_test_indices = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing fold {fold + 1}/{self.walk_forward_splits}")

            try:
                # Make predictions on test set
                X_test = X[test_idx]
                y_test = y[test_idx]
                prices_test = (
                    prices[test_idx] if len(prices) > max(test_idx) else prices[-len(test_idx) :]
                )

                # Get model predictions
                predictions = model.predict(X_test)

                # Ensure predictions and targets are same length
                min_len = min(len(predictions), len(y_test))
                predictions = predictions[:min_len]
                y_test = y_test[:min_len]
                prices_test = prices_test[:min_len]

                # Calculate basic metrics
                fold_metrics = {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "mse": mean_squared_error(y_test, predictions),
                    "mae": mean_absolute_error(y_test, predictions),
                    "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                }

                # Calculate trading metrics
                if len(predictions) > 1 and len(prices_test) > 1:
                    trading_metrics = evaluate_trading_performance(
                        predictions=predictions,
                        actual_prices=prices_test,
                        initial_balance=self.initial_balance,
                        transaction_cost=self.transaction_cost,
                    )
                    fold_metrics.update({f"trading_{k}": v for k, v in trading_metrics.items()})

                fold_results.append(fold_metrics)

                # Store for overall analysis
                all_predictions.extend(predictions)
                all_actuals.extend(y_test)
                all_test_indices.extend(test_idx[:min_len])

            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                continue

        if not fold_results:
            logger.error("No successful folds in walk-forward validation")
            return {}

        # Aggregate results
        results = self._aggregate_fold_results(fold_results)

        # Overall out-of-sample evaluation
        if all_predictions and all_actuals:
            overall_metrics = self._calculate_overall_metrics(
                np.array(all_predictions),
                np.array(all_actuals),
                prices[all_test_indices] if len(all_test_indices) > 0 else prices,
            )
            results["overall"] = overall_metrics

        # Regime-based analysis
        if len(all_predictions) > 0:
            regime_analysis = self._analyze_regime_performance(
                np.array(all_predictions),
                np.array(all_actuals),
                prices[all_test_indices] if len(all_test_indices) > 0 else prices,
            )
            results["regime_analysis"] = regime_analysis

        logger.info(f"Walk-forward validation completed for {symbol}")
        return results

    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across folds."""
        if not fold_results:
            return {}

        # Get all metric names
        all_metrics = set()
        for fold in fold_results:
            all_metrics.update(fold.keys())

        # Exclude non-numeric fields
        numeric_metrics = all_metrics - {"fold", "train_size", "test_size"}

        aggregated = {}

        for metric in numeric_metrics:
            values = [fold.get(metric, np.nan) for fold in fold_results]
            values = [v for v in values if not np.isnan(v)]

            if values:
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                aggregated[f"{metric}_min"] = np.min(values)
                aggregated[f"{metric}_max"] = np.max(values)

        # Add fold statistics
        aggregated["num_folds"] = len(fold_results)
        aggregated["avg_train_size"] = np.mean([f["train_size"] for f in fold_results])
        aggregated["avg_test_size"] = np.mean([f["test_size"] for f in fold_results])

        return aggregated

    def _calculate_overall_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray, prices: np.ndarray
    ) -> Dict[str, float]:
        """Calculate overall out-of-sample metrics."""
        metrics = {}

        # Basic prediction metrics
        metrics["mse"] = float(mean_squared_error(actuals, predictions))
        metrics["mae"] = float(mean_absolute_error(actuals, predictions))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))

        # Correlation
        if np.std(predictions) > 1e-8 and np.std(actuals) > 1e-8:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            metrics["correlation"] = float(correlation) if not np.isnan(correlation) else 0.0
        else:
            metrics["correlation"] = 0.0

        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            metrics["directional_accuracy"] = float(np.mean(pred_direction == actual_direction))

        # Trading performance
        if len(prices) >= len(predictions):
            trading_perf = evaluate_trading_performance(
                predictions=predictions,
                actual_prices=(
                    prices[: len(predictions)] if len(prices) > len(predictions) else prices
                ),
                initial_balance=self.initial_balance,
                transaction_cost=self.transaction_cost,
            )
            metrics.update({f"trading_{k}": v for k, v in trading_perf.items()})

            # Optimize threshold
            if len(predictions) == len(actuals):
                optimal_threshold, threshold_metrics = optimize_trading_threshold(
                    predictions, actuals
                )
                metrics["optimal_threshold"] = optimal_threshold
                metrics.update({f"optimized_{k}": v for k, v in threshold_metrics.items()})

        return metrics

    def _analyze_regime_performance(
        self, predictions: np.ndarray, actuals: np.ndarray, prices: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance across different market regimes."""
        if len(prices) < self.volatility_window:
            return {}

        regime_analysis = {}

        # Calculate market indicators
        returns = np.diff(prices) / prices[:-1]

        # Volatility regime
        if len(returns) >= self.volatility_window:
            rolling_vol = pd.Series(returns).rolling(self.volatility_window).std()
            vol_median = rolling_vol.median()

            # Align with predictions
            if len(rolling_vol) >= len(predictions):
                vol_regime = rolling_vol.iloc[-len(predictions) :].values
            else:
                vol_regime = rolling_vol.values

            # High volatility periods
            high_vol_mask = vol_regime > vol_median
            if np.any(high_vol_mask) and len(predictions) == len(high_vol_mask):
                high_vol_metrics = self._calculate_regime_metrics(
                    predictions[high_vol_mask],
                    (
                        actuals[high_vol_mask]
                        if len(actuals) == len(high_vol_mask)
                        else actuals[: np.sum(high_vol_mask)]
                    ),
                )
                regime_analysis["high_volatility"] = high_vol_metrics

            # Low volatility periods
            low_vol_mask = ~high_vol_mask
            if np.any(low_vol_mask) and len(predictions) == len(low_vol_mask):
                low_vol_metrics = self._calculate_regime_metrics(
                    predictions[low_vol_mask],
                    (
                        actuals[low_vol_mask]
                        if len(actuals) == len(low_vol_mask)
                        else actuals[: np.sum(low_vol_mask)]
                    ),
                )
                regime_analysis["low_volatility"] = low_vol_metrics

        # Trend regime
        if len(returns) >= self.trend_window:
            rolling_return = pd.Series(returns).rolling(self.trend_window).mean()

            if len(rolling_return) >= len(predictions):
                trend_regime = rolling_return.iloc[-len(predictions) :].values
            else:
                trend_regime = rolling_return.values

            # Bull market (positive trend)
            bull_mask = trend_regime > 0
            if np.any(bull_mask) and len(predictions) == len(bull_mask):
                bull_metrics = self._calculate_regime_metrics(
                    predictions[bull_mask],
                    (
                        actuals[bull_mask]
                        if len(actuals) == len(bull_mask)
                        else actuals[: np.sum(bull_mask)]
                    ),
                )
                regime_analysis["bull_market"] = bull_metrics

            # Bear market (negative trend)
            bear_mask = trend_regime <= 0
            if np.any(bear_mask) and len(predictions) == len(bear_mask):
                bear_metrics = self._calculate_regime_metrics(
                    predictions[bear_mask],
                    (
                        actuals[bear_mask]
                        if len(actuals) == len(bear_mask)
                        else actuals[: np.sum(bear_mask)]
                    ),
                )
                regime_analysis["bear_market"] = bear_metrics

        return regime_analysis

    def _calculate_regime_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for a specific regime."""
        if len(predictions) == 0 or len(actuals) == 0:
            return {}

        # Ensure same length
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        metrics = {}

        # Basic metrics
        metrics["count"] = float(len(predictions))
        metrics["mse"] = float(mean_squared_error(actuals, predictions))
        metrics["mae"] = float(mean_absolute_error(actuals, predictions))

        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            metrics["directional_accuracy"] = float(np.mean(pred_direction == actual_direction))

        # Correlation
        if np.std(predictions) > 1e-8 and np.std(actuals) > 1e-8:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            metrics["correlation"] = float(correlation) if not np.isnan(correlation) else 0.0
        else:
            metrics["correlation"] = 0.0

        return metrics

    def evaluate_model_stability(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate model stability across multiple runs.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target values
            prices: Price series
            num_runs: Number of evaluation runs

        Returns:
            Stability analysis results
        """
        logger.info(f"Evaluating model stability across {num_runs} runs")

        run_results = []

        for run in range(num_runs):
            try:
                # Add small noise to inputs to test stability
                noise_scale = 0.001
                X_noisy = X + np.random.normal(0, noise_scale, X.shape)

                # Make predictions
                predictions = model.predict(X_noisy)

                # Calculate metrics
                run_metrics = {
                    "run": run,
                    "mse": mean_squared_error(y, predictions),
                    "mae": mean_absolute_error(y, predictions),
                    "correlation": (
                        np.corrcoef(y, predictions)[0, 1] if np.std(predictions) > 1e-8 else 0.0
                    ),
                }

                # Trading metrics
                if len(prices) >= len(predictions):
                    trading_perf = evaluate_trading_performance(
                        predictions=predictions,
                        actual_prices=prices[: len(predictions)],
                        initial_balance=self.initial_balance,
                        transaction_cost=self.transaction_cost,
                    )
                    run_metrics.update({f"trading_{k}": v for k, v in trading_perf.items()})

                run_results.append(run_metrics)

            except Exception as e:
                logger.error(f"Error in stability run {run}: {e}")
                continue

        if not run_results:
            return {}

        # Calculate stability metrics
        stability_results = {}

        # Get all numeric metrics
        numeric_metrics = set()
        for run in run_results:
            numeric_metrics.update(
                [k for k, v in run.items() if isinstance(v, (int, float)) and k != "run"]
            )

        for metric in numeric_metrics:
            values = [run.get(metric, np.nan) for run in run_results]
            values = [v for v in values if not np.isnan(v)]

            if values:
                stability_results[f"{metric}_mean"] = np.mean(values)
                stability_results[f"{metric}_std"] = np.std(values)
                stability_results[f"{metric}_cv"] = np.std(values) / (
                    np.mean(values) + 1e-8
                )  # Coefficient of variation

        stability_results["num_successful_runs"] = len(run_results)
        stability_results["stability_score"] = 1.0 / (
            1.0 + stability_results.get("mse_cv", 1.0)
        )  # Higher is more stable

        return stability_results

    def benchmark_comparison(
        self,
        model_predictions: np.ndarray,
        actual_prices: np.ndarray,
        benchmark_strategies: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Compare model performance against benchmark strategies.

        Args:
            model_predictions: Model predictions
            actual_prices: Actual price series
            benchmark_strategies: Dictionary of benchmark predictions

        Returns:
            Comparison results
        """
        logger.info("Performing benchmark comparison")

        if len(actual_prices) <= 1:
            return {}

        comparison_results = {}

        # Model performance
        model_perf = evaluate_trading_performance(
            predictions=model_predictions,
            actual_prices=actual_prices,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost,
        )
        comparison_results["model"] = model_perf

        # Buy and hold benchmark
        buy_hold_return = (actual_prices[-1] / actual_prices[0]) - 1
        comparison_results["buy_and_hold"] = {
            "total_return": buy_hold_return,
            "sharpe_ratio": 0.0,  # Simplified
            "max_drawdown": 0.0,  # Simplified
        }

        # Random strategy benchmark
        np.random.seed(42)  # For reproducibility
        random_predictions = np.random.normal(0, np.std(model_predictions), len(model_predictions))
        random_perf = evaluate_trading_performance(
            predictions=random_predictions,
            actual_prices=actual_prices,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost,
        )
        comparison_results["random_strategy"] = random_perf

        # Custom benchmarks
        if benchmark_strategies:
            for name, predictions in benchmark_strategies.items():
                if len(predictions) == len(actual_prices) - 1:  # Align with returns
                    bench_perf = evaluate_trading_performance(
                        predictions=predictions,
                        actual_prices=actual_prices,
                        initial_balance=self.initial_balance,
                        transaction_cost=self.transaction_cost,
                    )
                    comparison_results[name] = bench_perf

        # Calculate relative performance
        model_sharpe = model_perf.get("sharpe_ratio", 0)
        model_return = model_perf.get("net_return", 0)

        comparison_results["relative_performance"] = {
            "vs_buy_hold_return": model_return - buy_hold_return,
            "vs_random_sharpe": model_sharpe - random_perf.get("sharpe_ratio", 0),
            "outperformed_buy_hold": model_return > buy_hold_return,
            "outperformed_random": model_sharpe > random_perf.get("sharpe_ratio", 0),
        }

        return comparison_results

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], symbol: str) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            evaluation_results: Results from various evaluation methods
            symbol: Trading symbol

        Returns:
            Formatted evaluation report
        """
        report = f"""
Trading Model Evaluation Report for {symbol}
{'=' * 50}

"""

        # Walk-forward validation results
        if "walk_forward" in evaluation_results:
            wf_results = evaluation_results["walk_forward"]
            report += f"""
Walk-Forward Validation Results:
- Number of folds: {wf_results.get('num_folds', 'N/A')}
- Average RMSE: {wf_results.get('rmse_mean', 0):.6f} ± {wf_results.get('rmse_std', 0):.6f}
- Average Directional Accuracy: {wf_results.get('trading_directional_accuracy_mean', 0):.3f}
- Average Sharpe Ratio: {wf_results.get('trading_sharpe_ratio_mean', 0):.3f}
- Average Max Drawdown: {wf_results.get('trading_max_drawdown_mean', 0):.3f}
"""

        # Overall performance
        if "overall" in evaluation_results:
            overall = evaluation_results["overall"]
            report += f"""
Overall Out-of-Sample Performance:
- RMSE: {overall.get('rmse', 0):.6f}
- Correlation: {overall.get('correlation', 0):.3f}
- Directional Accuracy: {overall.get('directional_accuracy', 0):.3f}
- Total Return: {overall.get('trading_net_return', 0):.3%}
- Sharpe Ratio: {overall.get('trading_sharpe_ratio', 0):.3f}
- Max Drawdown: {overall.get('trading_max_drawdown', 0):.3%}
"""

        # Regime analysis
        if "regime_analysis" in evaluation_results:
            regime = evaluation_results["regime_analysis"]
            report += "\nRegime-Based Performance:\n"

            for regime_name, regime_metrics in regime.items():
                if regime_metrics:
                    report += f"- {regime_name.replace('_', ' ').title()}:\n"
                    report += f"  - Directional Accuracy: {regime_metrics.get('directional_accuracy', 0):.3f}\n"
                    report += f"  - Correlation: {regime_metrics.get('correlation', 0):.3f}\n"
                    report += f"  - Sample Size: {regime_metrics.get('count', 0):.0f}\n"

        # Stability analysis
        if "stability" in evaluation_results:
            stability = evaluation_results["stability"]
            report += f"""
Model Stability Analysis:
- Stability Score: {stability.get('stability_score', 0):.3f}
- RMSE Coefficient of Variation: {stability.get('mse_cv', 0):.3f}
- Correlation Stability: {stability.get('correlation_mean', 0):.3f} ± {stability.get('correlation_std', 0):.3f}
"""

        # Benchmark comparison
        if "benchmark" in evaluation_results:
            benchmark = evaluation_results["benchmark"]
            relative = benchmark.get("relative_performance", {})
            report += f"""
Benchmark Comparison:
- Model Return: {benchmark.get('model', {}).get('net_return', 0):.3%}
- Buy & Hold Return: {benchmark.get('buy_and_hold', {}).get('total_return', 0):.3%}
- Excess Return vs Buy & Hold: {relative.get('vs_buy_hold_return', 0):.3%}
- Outperformed Buy & Hold: {relative.get('outperformed_buy_hold', False)}
- Outperformed Random Strategy: {relative.get('outperformed_random', False)}
"""

        report += f"\nReport generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return report
