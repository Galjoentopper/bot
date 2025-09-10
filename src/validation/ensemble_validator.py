"""
Advanced Ensemble Validation Framework
======================================

Comprehensive validation framework for trading model ensembles with:
- Cross-validation for ensemble configurations
- Performance attribution analysis
- Robustness testing
- Hyperparameter optimization for ensemble weights
- Real-time validation monitoring
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..ensemble.trading_ensemble import TradingEnsemble
from ..evaluation.trading_evaluator import TradingModelEvaluator
from ..utils.trading_metrics import TradingMetricsCalculator, evaluate_trading_performance

logger = logging.getLogger(__name__)


class EnsembleValidator:
    """
    Advanced validation framework for trading model ensembles.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Validation parameters
        self.n_splits = self.config.get("n_splits", 5)
        self.test_size = self.config.get("test_size", 0.2)
        self.gap_size = self.config.get("gap_size", 0)
        self.min_train_size = self.config.get("min_train_size", 1000)

        # Optimization parameters
        self.n_trials = self.config.get("n_trials", 100)
        self.optimization_timeout = self.config.get("optimization_timeout", 3600)  # 1 hour
        self.optimization_metric = self.config.get("optimization_metric", "sharpe_ratio")

        # Robustness testing
        self.noise_levels = self.config.get("noise_levels", [0.001, 0.005, 0.01])
        self.bootstrap_samples = self.config.get("bootstrap_samples", 100)

        # Performance tracking
        self.validation_history = []
        self.best_configuration = None
        self.best_score = -np.inf

        # Metrics calculator
        self.metrics_calc = TradingMetricsCalculator()

        logger.info("Ensemble validator initialized")

    def validate_ensemble_configuration(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        ensemble_config: Dict[str, Any],
        symbol: str = "Unknown",
    ) -> Dict[str, Any]:
        """
        Validate a specific ensemble configuration using cross-validation.

        Args:
            models: Dictionary of trained models
            X: Feature matrix
            y: Target values
            prices: Price series
            ensemble_config: Ensemble configuration to test
            symbol: Trading symbol

        Returns:
            Validation results
        """
        logger.info(f"Validating ensemble configuration for {symbol}")

        if len(X) < self.min_train_size:
            logger.warning(f"Insufficient data for validation: {len(X)} < {self.min_train_size}")
            return {}

        # Create time series splits
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits, test_size=int(len(X) * self.test_size), gap=self.gap_size
        )

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing validation fold {fold + 1}/{self.n_splits}")

            try:
                # Create ensemble for this fold
                ensemble = TradingEnsemble(ensemble_config)

                # Add models to ensemble
                for name, model in models.items():
                    initial_weight = ensemble_config.get("static_weights", {}).get(name)
                    ensemble.add_model(name, model, initial_weight)

                # Get test data
                X_test = X[test_idx]
                y_test = y[test_idx]
                prices_test = (
                    prices[test_idx] if len(prices) > max(test_idx) else prices[-len(test_idx) :]
                )

                # Make ensemble predictions
                ensemble_pred = ensemble.predict(X_test, prices_test)

                # Ensure same length
                min_len = min(len(ensemble_pred), len(y_test))
                ensemble_pred = ensemble_pred[:min_len]
                y_test = y_test[:min_len]
                prices_test = prices_test[:min_len]

                # Calculate metrics
                fold_metrics = self._calculate_fold_metrics(
                    ensemble_pred, y_test, prices_test, fold
                )

                # Add ensemble-specific metrics
                ensemble_stats = ensemble.get_ensemble_stats()
                fold_metrics.update(
                    {
                        f"ensemble_{k}": v
                        for k, v in ensemble_stats.items()
                        if isinstance(v, (int, float))
                    }
                )

                fold_results.append(fold_metrics)

            except Exception as e:
                logger.error(f"Error in validation fold {fold}: {e}")
                continue

        if not fold_results:
            logger.error("No successful validation folds")
            return {}

        # Aggregate results
        validation_results = self._aggregate_validation_results(fold_results)
        validation_results["ensemble_config"] = ensemble_config
        validation_results["symbol"] = symbol

        # Store in history
        self.validation_history.append(validation_results)

        # Update best configuration
        current_score = validation_results.get(f"{self.optimization_metric}_mean", -np.inf)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_configuration = ensemble_config.copy()
            logger.info(
                f"New best ensemble configuration: {self.optimization_metric} = {current_score:.6f}"
            )

        return validation_results

    def _calculate_fold_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray, prices: np.ndarray, fold: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a validation fold."""
        metrics = {"fold": fold}

        # Basic prediction metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        metrics["mse"] = mean_squared_error(actuals, predictions)
        metrics["mae"] = mean_absolute_error(actuals, predictions)
        metrics["rmse"] = np.sqrt(metrics["mse"])

        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            metrics["directional_accuracy"] = np.mean(pred_direction == actual_direction)

        # Correlation
        if np.std(predictions) > 1e-8 and np.std(actuals) > 1e-8:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics["correlation"] = 0.0

        # Trading performance metrics
        if len(prices) >= len(predictions):
            trading_perf = evaluate_trading_performance(
                predictions=predictions,
                actual_prices=prices[: len(predictions)],
                initial_balance=10000.0,
                transaction_cost=0.001,
            )

            # Add trading metrics with prefix
            for key, value in trading_perf.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value

        return metrics

    def _aggregate_validation_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across validation folds."""
        if not fold_results:
            return {}

        # Get all numeric metrics
        all_metrics = set()
        for fold in fold_results:
            all_metrics.update(
                [k for k, v in fold.items() if isinstance(v, (int, float)) and k != "fold"]
            )

        aggregated = {}

        for metric in all_metrics:
            values = [fold.get(metric, np.nan) for fold in fold_results]
            values = [v for v in values if not np.isnan(v)]

            if values:
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                aggregated[f"{metric}_min"] = np.min(values)
                aggregated[f"{metric}_max"] = np.max(values)
                aggregated[f"{metric}_median"] = np.median(values)

        # Add validation statistics
        aggregated["num_folds"] = len(fold_results)
        aggregated["validation_score"] = aggregated.get(f"{self.optimization_metric}_mean", 0)

        return aggregated

    def optimize_ensemble_weights(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        symbol: str = "Unknown",
    ) -> Dict[str, Any]:
        """
        Optimize ensemble weights using Optuna.

        Args:
            models: Dictionary of trained models
            X: Feature matrix
            y: Target values
            prices: Price series
            symbol: Trading symbol

        Returns:
            Optimization results
        """
        logger.info(f"Optimizing ensemble weights for {symbol}")

        model_names = list(models.keys())

        def objective(trial):
            """Optuna objective function."""
            # Sample ensemble configuration
            config = {
                "weighting_method": trial.suggest_categorical(
                    "weighting_method", ["static", "dynamic", "optimal"]
                ),
                "min_weight": trial.suggest_float("min_weight", 0.01, 0.1),
                "max_weight": trial.suggest_float("max_weight", 0.6, 0.9),
                "decay_factor": trial.suggest_float("decay_factor", 0.9, 0.99),
                "temperature": trial.suggest_float("temperature", 1.0, 5.0),
                "performance_window": trial.suggest_int("performance_window", 24, 96),
                "static_weights": {},
            }

            # Sample static weights
            if config["weighting_method"] == "static":
                weights = []
                for name in model_names:
                    weight = trial.suggest_float(f"weight_{name}", 0.1, 0.8)
                    weights.append(weight)

                # Normalize weights
                total_weight = sum(weights)
                for i, name in enumerate(model_names):
                    config["static_weights"][name] = weights[i] / total_weight

            # Validate configuration
            try:
                validation_results = self.validate_ensemble_configuration(
                    models, X, y, prices, config, symbol
                )

                # Return objective value
                score = validation_results.get(f"{self.optimization_metric}_mean", -np.inf)
                return score if not np.isnan(score) else -np.inf

            except Exception as e:
                logger.error(f"Error in trial: {e}")
                return -np.inf

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Optimize
        study.optimize(
            objective, n_trials=self.n_trials, timeout=self.optimization_timeout, catch=(Exception,)
        )

        # Get best results
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Optimization completed. Best {self.optimization_metric}: {best_value:.6f}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": study,
            "optimization_history": [
                (trial.value, trial.params) for trial in study.trials if trial.value is not None
            ],
        }

    def test_ensemble_robustness(
        self, ensemble: TradingEnsemble, X: np.ndarray, y: np.ndarray, prices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test ensemble robustness to input noise and data variations.

        Args:
            ensemble: Configured ensemble
            X: Feature matrix
            y: Target values
            prices: Price series

        Returns:
            Robustness test results
        """
        logger.info("Testing ensemble robustness")

        robustness_results = {}

        # Test noise robustness
        noise_results = {}
        baseline_pred = ensemble.predict(X, prices)

        for noise_level in self.noise_levels:
            noise_scores = []

            for _ in range(10):  # Multiple noise realizations
                # Add noise to features
                noise = np.random.normal(0, noise_level, X.shape)
                X_noisy = X + noise

                try:
                    noisy_pred = ensemble.predict(X_noisy, prices)

                    # Calculate prediction stability
                    pred_correlation = np.corrcoef(baseline_pred, noisy_pred)[0, 1]
                    noise_scores.append(pred_correlation if not np.isnan(pred_correlation) else 0.0)

                except Exception as e:
                    logger.error(f"Error in noise test: {e}")
                    noise_scores.append(0.0)

            noise_results[f"noise_{noise_level}"] = {
                "mean_stability": np.mean(noise_scores),
                "std_stability": np.std(noise_scores),
                "min_stability": np.min(noise_scores),
                "robustness_score": np.mean(noise_scores),
            }

        robustness_results["noise_robustness"] = noise_results

        # Test bootstrap robustness
        bootstrap_results = []
        n_samples = len(X)

        for _ in range(min(self.bootstrap_samples, 20)):  # Limit for performance
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            prices_boot = prices[indices] if len(prices) > max(indices) else prices

            try:
                boot_pred = ensemble.predict(X_boot, prices_boot)

                # Calculate performance metrics
                if len(boot_pred) == len(y_boot):
                    trading_perf = evaluate_trading_performance(
                        predictions=boot_pred,
                        actual_prices=prices_boot[: len(boot_pred)]
                        if len(prices_boot) > len(boot_pred)
                        else prices_boot,
                        initial_balance=10000.0,
                        transaction_cost=0.001,
                    )

                    bootstrap_results.append(
                        {
                            "sharpe_ratio": trading_perf.get("sharpe_ratio", 0),
                            "total_return": trading_perf.get("net_return", 0),
                            "max_drawdown": trading_perf.get("max_drawdown", 0),
                        }
                    )

            except Exception as e:
                logger.error(f"Error in bootstrap test: {e}")
                continue

        # Aggregate bootstrap results
        if bootstrap_results:
            bootstrap_aggregated = {}
            for metric in ["sharpe_ratio", "total_return", "max_drawdown"]:
                values = [result[metric] for result in bootstrap_results if metric in result]
                if values:
                    bootstrap_aggregated[f"{metric}_mean"] = np.mean(values)
                    bootstrap_aggregated[f"{metric}_std"] = np.std(values)
                    bootstrap_aggregated[f"{metric}_confidence_interval"] = np.percentile(
                        values, [5, 95]
                    ).tolist()

            robustness_results["bootstrap_robustness"] = bootstrap_aggregated

        # Calculate overall robustness score
        noise_score = np.mean([result["robustness_score"] for result in noise_results.values()])

        bootstrap_score = 1.0  # Default if no bootstrap results
        if bootstrap_results:
            sharpe_std = bootstrap_aggregated.get("sharpe_ratio_std", 1.0)
            bootstrap_score = 1.0 / (1.0 + sharpe_std)  # Lower std = higher score

        robustness_results["overall_robustness_score"] = (noise_score + bootstrap_score) / 2

        return robustness_results

    def performance_attribution(
        self, ensemble: TradingEnsemble, X: np.ndarray, y: np.ndarray, prices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze performance attribution of ensemble components.

        Args:
            ensemble: Configured ensemble
            X: Feature matrix
            y: Target values
            prices: Price series

        Returns:
            Performance attribution analysis
        """
        logger.info("Performing performance attribution analysis")

        attribution_results = {}

        # Get ensemble prediction
        ensemble_pred = ensemble.predict(X, prices)

        # Calculate ensemble performance
        ensemble_perf = evaluate_trading_performance(
            predictions=ensemble_pred,
            actual_prices=prices[: len(ensemble_pred)]
            if len(prices) > len(ensemble_pred)
            else prices,
            initial_balance=10000.0,
            transaction_cost=0.001,
        )

        attribution_results["ensemble_performance"] = ensemble_perf

        # Analyze individual model contributions
        model_contributions = {}

        for model_name, model in ensemble.models.items():
            try:
                # Get individual model prediction
                model_pred = model.predict(X)

                # Align lengths
                min_len = min(len(model_pred), len(y), len(prices))
                model_pred = model_pred[:min_len]
                y_aligned = y[:min_len]
                prices_aligned = prices[:min_len]

                # Calculate individual performance
                model_perf = evaluate_trading_performance(
                    predictions=model_pred,
                    actual_prices=prices_aligned,
                    initial_balance=10000.0,
                    transaction_cost=0.001,
                )

                # Calculate contribution metrics
                model_weight = ensemble.model_weights.get(model_name, 0)
                weighted_contribution = model_perf.get("net_return", 0) * model_weight

                model_contributions[model_name] = {
                    "individual_performance": model_perf,
                    "weight": model_weight,
                    "weighted_contribution": weighted_contribution,
                    "performance_rank": 0,  # Will be filled later
                }

            except Exception as e:
                logger.error(f"Error analyzing {model_name}: {e}")
                continue

        # Rank models by performance
        sorted_models = sorted(
            model_contributions.items(),
            key=lambda x: x[1]["individual_performance"].get("sharpe_ratio", 0),
            reverse=True,
        )

        for rank, (model_name, contrib) in enumerate(sorted_models, 1):
            model_contributions[model_name]["performance_rank"] = rank

        attribution_results["model_contributions"] = model_contributions

        # Calculate diversification benefit
        individual_returns = [
            contrib["individual_performance"].get("net_return", 0) * contrib["weight"]
            for contrib in model_contributions.values()
        ]

        weighted_avg_return = sum(individual_returns)
        ensemble_return = ensemble_perf.get("net_return", 0)
        diversification_benefit = ensemble_return - weighted_avg_return

        attribution_results["diversification_benefit"] = diversification_benefit

        # Weight efficiency analysis
        weight_efficiency = {}
        for model_name, contrib in model_contributions.items():
            weight = contrib["weight"]
            performance = contrib["individual_performance"].get("sharpe_ratio", 0)
            efficiency = performance / (weight + 1e-8)  # Performance per unit weight
            weight_efficiency[model_name] = efficiency

        attribution_results["weight_efficiency"] = weight_efficiency

        return attribution_results

    def generate_validation_report(self, symbol: str) -> str:
        """
        Generate comprehensive validation report.

        Args:
            symbol: Trading symbol

        Returns:
            Formatted validation report
        """
        if not self.validation_history:
            return "No validation results available."

        latest_results = self.validation_history[-1]

        report = f"""
Ensemble Validation Report for {symbol}
{'=' * 50}

Best Configuration:
{'-' * 20}
"""

        if self.best_configuration:
            for key, value in self.best_configuration.items():
                if isinstance(value, dict):
                    report += f"{key}:\n"
                    for subkey, subvalue in value.items():
                        report += f"  {subkey}: {subvalue}\n"
                else:
                    report += f"{key}: {value}\n"

        report += f"""

Validation Results:
{'-' * 20}
Number of folds: {latest_results.get('num_folds', 'N/A')}
Validation score ({self.optimization_metric}): {latest_results.get(f'{self.optimization_metric}_mean', 0):.6f} ± {latest_results.get(f'{self.optimization_metric}_std', 0):.6f}

Performance Metrics:
- RMSE: {latest_results.get('rmse_mean', 0):.6f} ± {latest_results.get('rmse_std', 0):.6f}
- Directional Accuracy: {latest_results.get('directional_accuracy_mean', 0):.3f} ± {latest_results.get('directional_accuracy_std', 0):.3f}
- Correlation: {latest_results.get('correlation_mean', 0):.3f} ± {latest_results.get('correlation_std', 0):.3f}

Trading Performance:
- Sharpe Ratio: {latest_results.get('sharpe_ratio_mean', 0):.3f} ± {latest_results.get('sharpe_ratio_std', 0):.3f}
- Total Return: {latest_results.get('net_return_mean', 0):.3%} ± {latest_results.get('net_return_std', 0):.3%}
- Max Drawdown: {latest_results.get('max_drawdown_mean', 0):.3%} ± {latest_results.get('max_drawdown_std', 0):.3%}

Ensemble Statistics:
- Number of models: {latest_results.get('ensemble_num_models', 'N/A')}
- Weighting method: {latest_results.get('ensemble_config', {}).get('weighting_method', 'N/A')}
"""

        report += f"\nReport generated for validation run with {len(self.validation_history)} total configurations tested.\n"

        return report
