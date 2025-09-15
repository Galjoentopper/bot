"""
Advanced Trading Model Ensemble
==============================

Sophisticated ensemble system for cryptocurrency trading models with:
- Dynamic weighting based on market conditions
- Multi-objective optimization
- Risk-aware model selection
- Real-time performance tracking
- Regime-aware ensemble weights
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from ..utils.trading_metrics import TradingMetricsCalculator, evaluate_trading_performance

logger = logging.getLogger(__name__)


class TradingEnsemble:
    """
    Advanced ensemble system for trading models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trading ensemble.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Ensemble parameters
        self.weighting_method = self.config.get(
            "weighting_method", "dynamic"
        )  # static, dynamic, optimal
        self.rebalance_frequency = self.config.get(
            "rebalance_frequency", 48
        )  # 24 hours for 30min data
        self.min_weight = self.config.get("min_weight", 0.05)  # Minimum model weight
        self.max_weight = self.config.get("max_weight", 0.8)  # Maximum model weight

        # Performance tracking
        self.performance_window = self.config.get("performance_window", 96)  # 48 hours
        self.decay_factor = self.config.get("decay_factor", 0.95)  # For exponential weighting

        # Risk management
        self.risk_penalty = self.config.get("risk_penalty", 0.1)
        self.volatility_target = self.config.get("volatility_target", 0.02)  # 2% daily volatility

        # Model storage
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.ensemble_history = []

        # Static weights (fallback)
        self.static_weights = self.config.get(
            "static_weights", {"gru": 0.4, "lightgbm": 0.4, "ppo": 0.2}
        )

        # Metrics calculator
        self.metrics_calc = TradingMetricsCalculator()

        logger.info(f"Trading ensemble initialized with {self.weighting_method} weighting")

    def add_model(self, name: str, model: Any, initial_weight: Optional[float] = None):
        """
        Add a model to the ensemble.

        Args:
            name: Model name
            model: Model object with predict method
            initial_weight: Initial weight for the model
        """
        self.models[name] = model

        if initial_weight is not None:
            self.model_weights[name] = initial_weight
        elif name in self.static_weights:
            self.model_weights[name] = self.static_weights[name]
        else:
            # Equal weighting by default
            self.model_weights[name] = 1.0 / len(self.models)

        # Initialize performance tracking
        self.model_performance[name] = {
            "predictions": [],
            "errors": [],
            "sharpe_ratios": [],
            "returns": [],
            "last_update": 0,
        }

        logger.info(f"Added model '{name}' with weight {self.model_weights[name]:.3f}")

    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_weight

    def _calculate_model_performance(
        self,
        name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        prices: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a model.

        Args:
            name: Model name
            predictions: Model predictions
            actuals: Actual values
            prices: Price series for trading metrics

        Returns:
            Performance metrics dictionary
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return {}

        # Align arrays
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        metrics = {}

        # Basic prediction metrics
        metrics["mse"] = mean_squared_error(actuals, predictions)
        metrics["rmse"] = np.sqrt(metrics["mse"])

        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            metrics["directional_accuracy"] = np.mean(pred_direction == actual_direction)
        else:
            metrics["directional_accuracy"] = 0.5

        # Correlation
        if np.std(predictions) > 1e-8 and np.std(actuals) > 1e-8:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics["correlation"] = 0.0

        # Trading metrics
        if prices is not None and len(prices) >= len(predictions):
            trading_perf = evaluate_trading_performance(
                predictions=predictions,
                actual_prices=(
                    prices[: len(predictions)] if len(prices) > len(predictions) else prices
                ),
                initial_balance=10000.0,
                transaction_cost=0.001,
            )

            metrics["sharpe_ratio"] = trading_perf.get("sharpe_ratio", 0.0)
            metrics["total_return"] = trading_perf.get("net_return", 0.0)
            metrics["max_drawdown"] = trading_perf.get("max_drawdown", 0.0)
            metrics["volatility"] = trading_perf.get("volatility", 0.0)

        # Composite score (higher is better)
        composite_score = (
            metrics["directional_accuracy"] * 0.3
            + (1 - metrics["rmse"] / (metrics["rmse"] + 1)) * 0.2
            + max(0, metrics["correlation"]) * 0.2
            + max(0, metrics.get("sharpe_ratio", 0)) * 0.2
            + max(0, 1 - metrics.get("max_drawdown", 0)) * 0.1
        )
        metrics["composite_score"] = composite_score

        return metrics

    def _update_dynamic_weights(
        self,
        model_predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
        prices: Optional[np.ndarray] = None,
    ):
        """
        Update model weights based on recent performance.

        Args:
            model_predictions: Dictionary of model predictions
            actuals: Actual values
            prices: Price series
        """
        if len(actuals) < self.performance_window:
            logger.warning("Insufficient data for dynamic weight updates")
            return

        # Calculate recent performance for each model
        model_scores = {}

        for name, predictions in model_predictions.items():
            if name in self.models and len(predictions) > 0:
                # Use recent window
                recent_preds = predictions[-self.performance_window :]
                recent_actuals = actuals[-len(recent_preds) :]
                recent_prices = prices[-len(recent_preds) :] if prices is not None else None

                # Calculate performance
                perf_metrics = self._calculate_model_performance(
                    name, recent_preds, recent_actuals, recent_prices
                )

                # Update performance history
                self.model_performance[name].update(perf_metrics)

                # Use composite score for weighting
                model_scores[name] = perf_metrics.get("composite_score", 0.0)

        if not model_scores:
            return

        # Convert scores to weights
        # Use softmax to ensure positive weights that sum to 1
        score_values = np.array(list(model_scores.values()))

        # Temperature parameter for softmax (higher = more uniform)
        temperature = self.config.get("temperature", 2.0)
        softmax_weights = np.exp(score_values / temperature)
        softmax_weights = softmax_weights / np.sum(softmax_weights)

        # Apply constraints
        for i, name in enumerate(model_scores.keys()):
            new_weight = softmax_weights[i]

            # Apply min/max constraints
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))

            # Exponential smoothing with previous weight
            old_weight = self.model_weights.get(name, 1.0 / len(self.models))
            smoothed_weight = self.decay_factor * old_weight + (1 - self.decay_factor) * new_weight

            self.model_weights[name] = smoothed_weight

        # Normalize weights
        self._normalize_weights()

        logger.info(f"Updated dynamic weights: {self.model_weights}")

    def _optimize_weights(
        self,
        model_predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
        prices: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights using numerical optimization.

        Args:
            model_predictions: Dictionary of model predictions
            actuals: Actual values
            prices: Price series

        Returns:
            Optimized weights dictionary
        """
        model_names = list(model_predictions.keys())
        n_models = len(model_names)

        if n_models == 0:
            return {}

        # Stack predictions into matrix
        pred_matrix = np.column_stack([model_predictions[name] for name in model_names])

        # Ensure same length
        min_len = min(len(pred_matrix), len(actuals))
        pred_matrix = pred_matrix[:min_len]
        actuals = actuals[:min_len]

        def objective(weights):
            """Objective function to minimize."""
            # Ensure weights are valid
            weights = np.abs(weights)
            weights = weights / np.sum(weights)

            # Calculate ensemble predictions
            ensemble_pred = np.dot(pred_matrix, weights)

            # Multi-objective loss
            mse_loss = mean_squared_error(actuals, ensemble_pred)

            # Directional accuracy loss
            if len(ensemble_pred) > 1:
                pred_dir = np.sign(ensemble_pred)
                actual_dir = np.sign(actuals)
                directional_loss = 1 - np.mean(pred_dir == actual_dir)
            else:
                directional_loss = 0.0

            # Trading performance loss
            trading_loss = 0.0
            if prices is not None and len(prices) >= len(ensemble_pred):
                try:
                    trading_perf = evaluate_trading_performance(
                        predictions=ensemble_pred,
                        actual_prices=prices[: len(ensemble_pred)],
                        initial_balance=10000.0,
                        transaction_cost=0.001,
                    )
                    sharpe_ratio = trading_perf.get("sharpe_ratio", 0.0)
                    trading_loss = max(0, -sharpe_ratio)  # Negative Sharpe as loss
                except:
                    trading_loss = 0.0

            # Weight concentration penalty (encourage diversification)
            concentration_penalty = np.sum(weights**2)

            # Combined objective
            total_loss = (
                mse_loss * 0.4
                + directional_loss * 0.3
                + trading_loss * 0.2
                + concentration_penalty * 0.1
            )

            return total_loss

        # Constraints: weights sum to 1 and are within bounds
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        bounds = [(self.min_weight, self.max_weight) for _ in range(n_models)]

        # Initial guess (equal weights)
        x0 = np.ones(n_models) / n_models

        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                optimal_weights = result.x / np.sum(result.x)  # Normalize
                return dict(zip(model_names, optimal_weights))
            else:
                logger.warning("Weight optimization failed, using current weights")
                return self.model_weights

        except Exception as e:
            logger.error(f"Error in weight optimization: {e}")
            return self.model_weights

    def _detect_market_regime(self, prices: np.ndarray, window: int = 48) -> str:
        """
        Detect current market regime.

        Args:
            prices: Price series
            window: Window for regime detection

        Returns:
            Market regime string
        """
        if len(prices) < window:
            return "unknown"

        recent_prices = prices[-window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]

        # Calculate regime indicators
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        trend = (recent_prices[-1] / recent_prices[0]) - 1

        # Simple regime classification
        if volatility > 0.03:  # High volatility
            if avg_return > 0:
                return "volatile_bull"
            else:
                return "volatile_bear"
        else:  # Low volatility
            if trend > 0.02:  # Positive trend
                return "stable_bull"
            elif trend < -0.02:  # Negative trend
                return "stable_bear"
            else:
                return "sideways"

    def predict(
        self,
        X: np.ndarray,
        prices: Optional[np.ndarray] = None,
        update_weights: bool = True,
    ) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Input features
            prices: Price series for regime detection
            update_weights: Whether to update weights

        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        # Get predictions from all models
        model_predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                model_predictions[name] = pred
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")
                continue

        if not model_predictions:
            raise ValueError("No valid predictions from ensemble models")

        # Ensure all predictions have the same length
        min_length = min(len(pred) for pred in model_predictions.values())
        for name in model_predictions:
            model_predictions[name] = model_predictions[name][:min_length]

        # Regime-aware weight adjustment
        if prices is not None and len(prices) > 0:
            regime = self._detect_market_regime(prices)

            # Adjust weights based on regime
            regime_adjustments = self.config.get(
                "regime_weights",
                {
                    "volatile_bull": {"gru": 1.2, "lightgbm": 0.9, "ppo": 1.1},
                    "volatile_bear": {"gru": 1.1, "lightgbm": 1.2, "ppo": 0.8},
                    "stable_bull": {"gru": 0.9, "lightgbm": 1.1, "ppo": 1.2},
                    "stable_bear": {"gru": 1.0, "lightgbm": 1.1, "ppo": 0.9},
                    "sideways": {"gru": 1.0, "lightgbm": 1.0, "ppo": 1.0},
                },
            )

            if regime in regime_adjustments:
                adjusted_weights = {}
                for name in self.model_weights:
                    base_weight = self.model_weights[name]
                    adjustment = regime_adjustments[regime].get(name, 1.0)
                    adjusted_weights[name] = base_weight * adjustment

                # Normalize adjusted weights
                total_weight = sum(adjusted_weights.values())
                if total_weight > 0:
                    for name in adjusted_weights:
                        adjusted_weights[name] /= total_weight

                weights_to_use = adjusted_weights
            else:
                weights_to_use = self.model_weights
        else:
            weights_to_use = self.model_weights

        # Calculate ensemble prediction
        ensemble_pred = np.zeros(min_length)
        total_weight = 0

        for name, predictions in model_predictions.items():
            weight = weights_to_use.get(name, 0)
            ensemble_pred += weight * predictions
            total_weight += weight

        # Normalize if needed
        if total_weight > 0:
            ensemble_pred /= total_weight

        # Store ensemble history
        self.ensemble_history.append(
            {
                "predictions": model_predictions.copy(),
                "weights": weights_to_use.copy(),
                "ensemble_prediction": ensemble_pred.copy(),
                "regime": (self._detect_market_regime(prices) if prices is not None else "unknown"),
            }
        )

        return ensemble_pred

    def update_performance(self, actuals: np.ndarray, prices: Optional[np.ndarray] = None):
        """
        Update ensemble performance and weights based on recent results.

        Args:
            actuals: Actual values
            prices: Price series
        """
        if not self.ensemble_history:
            return

        # Get recent predictions from history
        recent_history = self.ensemble_history[-self.performance_window :]

        if len(recent_history) == 0:
            return

        # Extract model predictions from history
        model_predictions = {}
        for name in self.models.keys():
            preds = []
            for entry in recent_history:
                if name in entry["predictions"]:
                    preds.extend(entry["predictions"][name])
            model_predictions[name] = np.array(preds)

        # Update weights based on method
        if self.weighting_method == "dynamic":
            self._update_dynamic_weights(model_predictions, actuals, prices)
        elif self.weighting_method == "optimal":
            optimized_weights = self._optimize_weights(model_predictions, actuals, prices)
            self.model_weights.update(optimized_weights)

        # Log performance
        if len(recent_history) > 0:
            recent_ensemble_preds = np.concatenate(
                [entry["ensemble_prediction"] for entry in recent_history]
            )

            if len(recent_ensemble_preds) > 0 and len(actuals) > 0:
                min_len = min(len(recent_ensemble_preds), len(actuals))
                ensemble_metrics = self._calculate_model_performance(
                    "ensemble",
                    recent_ensemble_preds[:min_len],
                    actuals[:min_len],
                    prices[:min_len] if prices is not None else None,
                )

                logger.info(
                    f"Ensemble performance - Sharpe: {ensemble_metrics.get('sharpe_ratio', 0):.3f}, "
                    f"Dir Acc: {ensemble_metrics.get('directional_accuracy', 0):.3f}"
                )

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics."""
        stats = {
            "num_models": len(self.models),
            "current_weights": self.model_weights.copy(),
            "weighting_method": self.weighting_method,
            "model_performance": self.model_performance.copy(),
            "history_length": len(self.ensemble_history),
        }

        # Weight stability (coefficient of variation)
        if len(self.ensemble_history) > 10:
            weight_history = []
            for entry in self.ensemble_history[-50:]:  # Last 50 predictions
                weight_history.append(list(entry["weights"].values()))

            if weight_history:
                weight_cv = np.std(weight_history, axis=0) / (
                    np.mean(weight_history, axis=0) + 1e-8
                )
                stats["weight_stability"] = np.mean(weight_cv)

        return stats
