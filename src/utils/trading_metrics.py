"""
Advanced Trading Metrics and Loss Functions
==========================================

Comprehensive trading metrics and loss functions optimized for cryptocurrency trading:
- Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
- Trading-specific loss functions
- Multi-objective optimization
- Portfolio performance analytics
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class TradingMetricsCalculator:
    """
    Comprehensive calculator for trading-specific performance metrics.
    """

    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 365):
        """
        Initialize trading metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days: Number of trading days per year (default 365 for crypto)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_returns(
        self, prices: np.ndarray, predictions: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate various types of returns.

        Args:
            prices: Array of prices
            predictions: Optional array of predictions for strategy returns

        Returns:
            Dictionary with different return series
        """
        returns = {}

        # Simple returns
        if len(prices) > 1:
            simple_returns = np.diff(prices) / prices[:-1]
            returns["simple"] = simple_returns

            # Log returns
            log_returns = np.diff(np.log(prices))
            returns["log"] = log_returns

            # Strategy returns (if predictions provided)
            if predictions is not None and len(predictions) == len(simple_returns):
                # Simple strategy: long when prediction > 0, short when < 0
                signals = np.sign(predictions)
                strategy_returns = signals * simple_returns
                returns["strategy"] = strategy_returns

                # Risk-adjusted strategy (scale by confidence)
                confidence = np.abs(predictions) / (np.max(np.abs(predictions)) + 1e-8)
                risk_adjusted_returns = signals * confidence * simple_returns
                returns["risk_adjusted_strategy"] = risk_adjusted_returns

        return returns

    def sharpe_ratio(self, returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (defaults to instance setting)

        Returns:
            Sharpe ratio
        """
        if len(returns) <= 1:
            return 0.0

        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf_rate = rf_rate / self.trading_days

        excess_returns = returns - daily_rf_rate
        return_std = np.std(returns)

        if return_std == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / return_std * np.sqrt(self.trading_days)
        return float(sharpe)

    def sortino_ratio(self, returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside deviation version of Sharpe).

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate

        Returns:
            Sortino ratio
        """
        if len(returns) <= 1:
            return 0.0

        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf_rate = rf_rate / self.trading_days

        excess_returns = returns - daily_rf_rate
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.sqrt(np.mean(downside_returns**2))

        if downside_deviation == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days)
        return float(sortino)

    def calmar_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Calmar ratio (annual return / maximum drawdown).

        Args:
            returns: Array of returns

        Returns:
            Calmar ratio
        """
        if len(returns) <= 1:
            return 0.0

        # Annual return
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (self.trading_days / len(returns)) - 1

        # Maximum drawdown
        max_dd = self.maximum_drawdown(returns)

        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0

        calmar = annual_return / max_dd
        return float(calmar)

    def maximum_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown as positive value
        """
        if len(returns) <= 1:
            return 0.0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdowns
        drawdowns = (running_max - cumulative) / running_max

        return float(np.max(drawdowns))

    def value_at_risk(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array of returns
            confidence_level: Confidence level (default 5%)

        Returns:
            VaR at given confidence level
        """
        if len(returns) <= 1:
            return 0.0

        return float(-np.percentile(returns, confidence_level * 100))

    def conditional_value_at_risk(
        self, returns: np.ndarray, confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Array of returns
            confidence_level: Confidence level

        Returns:
            CVaR at given confidence level
        """
        if len(returns) <= 1:
            return 0.0

        var_threshold = -np.percentile(returns, confidence_level * 100)
        tail_losses = returns[returns <= -var_threshold]

        if len(tail_losses) == 0:
            return var_threshold

        return float(-np.mean(tail_losses))

    def omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Args:
            returns: Array of returns
            threshold: Return threshold

        Returns:
            Omega ratio
        """
        if len(returns) <= 1:
            return 1.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if len(losses) == 0 or np.sum(losses) == 0:
            return np.inf if len(gains) > 0 else 1.0

        if len(gains) == 0:
            return 0.0

        omega = np.sum(gains) / np.sum(losses)
        return float(omega)

    def information_ratio(
        self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate Information ratio.

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Information ratio
        """
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) <= 1:
            return 0.0

        active_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(active_returns)

        if tracking_error == 0:
            return 0.0

        return float(np.mean(active_returns) / tracking_error * np.sqrt(self.trading_days))

    def hit_ratio(
        self, predictions: np.ndarray, actual: np.ndarray, threshold: float = 0.0
    ) -> float:
        """
        Calculate hit ratio (percentage of correct directional predictions).

        Args:
            predictions: Predicted values
            actual: Actual values
            threshold: Threshold for defining direction

        Returns:
            Hit ratio between 0 and 1
        """
        if len(predictions) != len(actual) or len(predictions) == 0:
            return 0.0

        pred_direction = (predictions > threshold).astype(int)
        actual_direction = (actual > threshold).astype(int)

        return float(np.mean(pred_direction == actual_direction))

    def calculate_comprehensive_metrics(
        self,
        prices: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive set of trading metrics.

        Args:
            prices: Price series
            predictions: Optional predictions for strategy evaluation
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary of all calculated metrics
        """
        metrics = {}

        # Calculate returns
        returns_dict = self.calculate_returns(prices, predictions)

        if "simple" in returns_dict:
            simple_returns = returns_dict["simple"]

            # Basic statistics
            metrics["total_return"] = float(np.prod(1 + simple_returns) - 1)
            metrics["annual_return"] = float(
                (1 + metrics["total_return"]) ** (self.trading_days / len(simple_returns)) - 1
            )
            metrics["volatility"] = float(np.std(simple_returns) * np.sqrt(self.trading_days))
            metrics["skewness"] = float(stats.skew(simple_returns))
            metrics["kurtosis"] = float(stats.kurtosis(simple_returns))

            # Risk-adjusted metrics
            metrics["sharpe_ratio"] = self.sharpe_ratio(simple_returns)
            metrics["sortino_ratio"] = self.sortino_ratio(simple_returns)
            metrics["calmar_ratio"] = self.calmar_ratio(simple_returns)
            metrics["omega_ratio"] = self.omega_ratio(simple_returns)

            # Risk metrics
            metrics["max_drawdown"] = self.maximum_drawdown(simple_returns)
            metrics["var_5pct"] = self.value_at_risk(simple_returns, 0.05)
            metrics["cvar_5pct"] = self.conditional_value_at_risk(simple_returns, 0.05)

            # Strategy-specific metrics
            if "strategy" in returns_dict:
                strategy_returns = returns_dict["strategy"]
                metrics["strategy_total_return"] = float(np.prod(1 + strategy_returns) - 1)
                metrics["strategy_sharpe"] = self.sharpe_ratio(strategy_returns)
                metrics["strategy_sortino"] = self.sortino_ratio(strategy_returns)
                metrics["strategy_max_drawdown"] = self.maximum_drawdown(strategy_returns)

                # Information ratio vs buy-and-hold
                metrics["information_ratio"] = self.information_ratio(
                    strategy_returns, simple_returns
                )

            # Prediction accuracy metrics
            if predictions is not None:
                if len(predictions) == len(simple_returns):
                    metrics["hit_ratio"] = self.hit_ratio(predictions, simple_returns)
                    metrics["directional_accuracy"] = self.hit_ratio(
                        predictions, simple_returns, 0.0
                    )

                    # Correlation metrics
                    if np.std(predictions) > 1e-8 and np.std(simple_returns) > 1e-8:
                        correlation = np.corrcoef(predictions, simple_returns)[0, 1]
                        metrics["prediction_correlation"] = (
                            float(correlation) if not np.isnan(correlation) else 0.0
                        )
                    else:
                        metrics["prediction_correlation"] = 0.0

            # Benchmark comparison
            if benchmark_returns is not None and len(benchmark_returns) == len(simple_returns):
                metrics["excess_return"] = metrics["annual_return"] - float(
                    np.mean(benchmark_returns) * self.trading_days
                )
                metrics["tracking_error"] = float(
                    np.std(simple_returns - benchmark_returns) * np.sqrt(self.trading_days)
                )
                metrics["information_ratio_benchmark"] = self.information_ratio(
                    simple_returns, benchmark_returns
                )

        return metrics


class TradingLossFunction(nn.Module):
    """
    Multi-objective loss function optimized for trading performance.
    """

    def __init__(
        self,
        prediction_weight: float = 1.0,
        directional_weight: float = 0.5,
        volatility_weight: float = 0.3,
        sharpe_weight: float = 0.2,
        drawdown_weight: float = 0.1,
        risk_penalty_weight: float = 0.1,
    ):
        """
        Initialize trading loss function.

        Args:
            prediction_weight: Weight for prediction accuracy loss
            directional_weight: Weight for directional accuracy loss
            volatility_weight: Weight for volatility prediction loss
            sharpe_weight: Weight for Sharpe ratio optimization
            drawdown_weight: Weight for drawdown minimization
            risk_penalty_weight: Weight for risk penalty
        """
        super().__init__()

        self.prediction_weight = prediction_weight
        self.directional_weight = directional_weight
        self.volatility_weight = volatility_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.risk_penalty_weight = risk_penalty_weight

        # Standard loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def directional_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate directional prediction loss."""
        if len(predictions) <= 1:
            return torch.tensor(0.0, device=predictions.device)

        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)

        # Penalize wrong directions more heavily
        direction_matches = (pred_direction * target_direction > 0).float()
        directional_accuracy = torch.mean(direction_matches)

        # Loss is 1 - accuracy
        return 1.0 - directional_accuracy

    def volatility_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate volatility prediction loss."""
        if len(predictions) <= 1:
            return torch.tensor(0.0, device=predictions.device)

        pred_vol = torch.std(predictions)
        target_vol = torch.std(targets)

        # Relative volatility error
        vol_error = torch.abs(pred_vol - target_vol) / (target_vol + 1e-8)
        return vol_error

    def sharpe_loss(
        self, predictions: torch.Tensor, prices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate negative Sharpe ratio as loss."""
        if len(predictions) <= 1:
            return torch.tensor(0.0, device=predictions.device)

        # Use predictions as returns if prices not provided
        if prices is not None and len(prices) == len(predictions) + 1:
            # Calculate returns from prices
            returns = (prices[1:] - prices[:-1]) / prices[:-1]
            # Create strategy returns using predictions as signals
            strategy_returns = torch.sign(predictions) * returns
        else:
            # Use predictions directly as returns
            strategy_returns = predictions

        # Calculate Sharpe ratio
        mean_return = torch.mean(strategy_returns)
        std_return = torch.std(strategy_returns)

        if std_return < 1e-8:
            return torch.tensor(0.0, device=predictions.device)

        sharpe = mean_return / std_return

        # Return negative Sharpe (since we want to minimize loss)
        return -sharpe

    def drawdown_loss(
        self, predictions: torch.Tensor, prices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate maximum drawdown loss."""
        if len(predictions) <= 1:
            return torch.tensor(0.0, device=predictions.device)

        # Calculate cumulative returns
        if prices is not None and len(prices) == len(predictions) + 1:
            returns = (prices[1:] - prices[:-1]) / prices[:-1]
            strategy_returns = torch.sign(predictions) * returns
        else:
            strategy_returns = predictions

        # Calculate cumulative portfolio value
        cumulative = torch.cumprod(1 + strategy_returns, dim=0)

        # Calculate running maximum
        running_max = torch.cummax(cumulative, dim=0)[0]

        # Calculate drawdowns
        drawdowns = (running_max - cumulative) / running_max

        # Maximum drawdown
        max_drawdown = torch.max(drawdowns)

        return max_drawdown

    def risk_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate risk penalty for extreme predictions."""
        # Penalize predictions that are too extreme
        extreme_threshold = 3.0  # 3 standard deviations
        pred_std = torch.std(predictions)

        if pred_std < 1e-8:
            return torch.tensor(0.0, device=predictions.device)

        normalized_preds = torch.abs(predictions) / pred_std
        extreme_penalty = torch.mean(torch.clamp(normalized_preds - extreme_threshold, min=0))

        return extreme_penalty

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate multi-objective trading loss.

        Args:
            predictions: Model predictions
            targets: Target values
            prices: Optional price series for trading metrics

        Returns:
            Tuple of (total_loss, loss_components)
        """
        loss_components = {}

        # Prediction accuracy loss
        loss_components["prediction"] = self.mse_loss(predictions, targets)

        # Directional accuracy loss
        loss_components["directional"] = self.directional_loss(predictions, targets)

        # Volatility prediction loss
        loss_components["volatility"] = self.volatility_loss(predictions, targets)

        # Sharpe ratio loss (if we have enough data)
        if len(predictions) > 10:
            loss_components["sharpe"] = self.sharpe_loss(predictions, prices)
        else:
            loss_components["sharpe"] = torch.tensor(0.0, device=predictions.device)

        # Drawdown loss
        if len(predictions) > 10:
            loss_components["drawdown"] = self.drawdown_loss(predictions, prices)
        else:
            loss_components["drawdown"] = torch.tensor(0.0, device=predictions.device)

        # Risk penalty
        loss_components["risk_penalty"] = self.risk_penalty(predictions)

        # Combine losses
        total_loss = (
            self.prediction_weight * loss_components["prediction"]
            + self.directional_weight * loss_components["directional"]
            + self.volatility_weight * loss_components["volatility"]
            + self.sharpe_weight * loss_components["sharpe"]
            + self.drawdown_weight * loss_components["drawdown"]
            + self.risk_penalty_weight * loss_components["risk_penalty"]
        )

        return total_loss, loss_components


def evaluate_trading_performance(
    predictions: np.ndarray,
    actual_prices: np.ndarray,
    initial_balance: float = 10000.0,
    transaction_cost: float = 0.001,
) -> Dict[str, float]:
    """
    Evaluate trading performance using predictions.

    Args:
        predictions: Model predictions (returns or price changes)
        actual_prices: Actual price series
        initial_balance: Starting portfolio value
        transaction_cost: Transaction cost as fraction

    Returns:
        Dictionary of performance metrics
    """
    if len(predictions) == 0 or len(actual_prices) <= 1:
        return {}

    # Calculate actual returns
    actual_returns = np.diff(actual_prices) / actual_prices[:-1]

    # Align predictions with returns
    if len(predictions) > len(actual_returns):
        predictions = predictions[: len(actual_returns)]
    elif len(predictions) < len(actual_returns):
        actual_returns = actual_returns[: len(predictions)]

    # Simple trading strategy: long when prediction > 0, short when < 0
    positions = np.sign(predictions)

    # Calculate strategy returns
    gross_strategy_returns = positions * actual_returns

    # Apply transaction costs (simplified)
    position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    transaction_costs = position_changes * transaction_cost
    net_strategy_returns = gross_strategy_returns - transaction_costs[: len(gross_strategy_returns)]

    # Calculate portfolio value
    portfolio_values = [initial_balance]
    for ret in net_strategy_returns:
        new_value = portfolio_values[-1] * (1 + ret)
        portfolio_values.append(new_value)

    portfolio_values = np.array(portfolio_values)

    # Initialize metrics calculator
    metrics_calc = TradingMetricsCalculator()

    # Calculate comprehensive metrics
    metrics = metrics_calc.calculate_comprehensive_metrics(
        prices=portfolio_values, predictions=predictions
    )

    # Add trading-specific metrics
    metrics["final_balance"] = float(portfolio_values[-1])
    metrics["total_trades"] = int(np.sum(position_changes))
    metrics["total_transaction_costs"] = float(np.sum(transaction_costs))
    metrics["gross_return"] = float(np.prod(1 + gross_strategy_returns) - 1)
    metrics["net_return"] = float(portfolio_values[-1] / initial_balance - 1)

    # Buy and hold comparison
    buy_hold_return = (actual_prices[-1] / actual_prices[0]) - 1
    metrics["excess_return_vs_buy_hold"] = metrics["net_return"] - buy_hold_return

    return metrics


def optimize_trading_threshold(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    threshold_range: Tuple[float, float] = (-0.01, 0.01),
    num_thresholds: int = 100,
) -> Tuple[float, Dict[str, float]]:
    """
    Optimize trading threshold for maximum Sharpe ratio.

    Args:
        predictions: Model predictions
        actual_returns: Actual returns
        threshold_range: Range of thresholds to test
        num_thresholds: Number of thresholds to test

    Returns:
        Tuple of (optimal_threshold, best_metrics)
    """
    if len(predictions) != len(actual_returns) or len(predictions) == 0:
        return 0.0, {}

    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    best_sharpe = -np.inf
    best_threshold = 0.0
    best_metrics = {}

    metrics_calc = TradingMetricsCalculator()

    for threshold in thresholds:
        # Create trading signals
        signals = np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0))

        # Calculate strategy returns
        strategy_returns = signals * actual_returns

        # Skip if no trades
        if np.sum(np.abs(signals)) == 0:
            continue

        # Calculate Sharpe ratio
        sharpe = metrics_calc.sharpe_ratio(strategy_returns)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold

            # Calculate full metrics for best threshold
            best_metrics = {
                "threshold": threshold,
                "sharpe_ratio": sharpe,
                "total_return": float(np.prod(1 + strategy_returns) - 1),
                "volatility": float(np.std(strategy_returns) * np.sqrt(365)),
                "max_drawdown": metrics_calc.maximum_drawdown(strategy_returns),
                "hit_ratio": metrics_calc.hit_ratio(predictions, actual_returns, threshold),
                "num_trades": int(np.sum(np.abs(np.diff(signals)))),
            }

    return best_threshold, best_metrics
