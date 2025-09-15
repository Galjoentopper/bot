#!/usr/bin/env python3
"""
Portfolio Optimizer
==================

Advanced portfolio optimization using Modern Portfolio Theory principles:
- Efficient frontier calculation
- Risk parity allocation
- Dynamic rebalancing
- Multi-objective optimization (return vs risk vs correlation)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Optimal portfolio allocation results."""

    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    diversification_ratio: float
    optimization_method: str


@dataclass
class ConstraintSet:
    """Portfolio optimization constraints."""

    max_weight: float = 0.25  # Max 25% per asset
    min_weight: float = 0.01  # Min 1% per asset
    max_correlation_cluster: float = 0.4  # Max 40% in correlated assets
    target_volatility: Optional[float] = None
    target_return: Optional[float] = None


class PortfolioOptimizer:
    """Advanced portfolio optimization system."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize portfolio optimizer."""
        self.config = config

        portfolio_config = config.get("portfolio_optimization", {})
        self.rebalance_threshold = portfolio_config.get("rebalance_threshold", 0.05)  # 5%
        self.lookback_period = portfolio_config.get("lookback_period", 60)  # 60 periods
        self.risk_free_rate = portfolio_config.get("risk_free_rate", 0.02)  # 2% annual

        # Optimization methods available
        self.optimization_methods = {
            "max_sharpe": self._optimize_max_sharpe,
            "min_volatility": self._optimize_min_volatility,
            "risk_parity": self._optimize_risk_parity,
            "max_diversification": self._optimize_max_diversification,
        }

        logger.info("Portfolio Optimizer initialized with advanced optimization methods")

    def calculate_expected_returns(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate expected returns using multiple methods."""
        try:
            expected_returns = {}

            for symbol, df in price_data.items():
                if len(df) < self.lookback_period or "close" not in df.columns:
                    expected_returns[symbol] = 0.02  # Default 2% expected return
                    continue

                # Calculate returns
                returns = df["close"].pct_change().dropna()

                if len(returns) < 20:
                    expected_returns[symbol] = 0.02
                    continue

                # Method 1: Historical mean
                historical_mean = returns.mean() * 24 * 365  # Annualized

                # Method 2: Exponentially weighted mean (more weight on recent data)
                ewm_mean = returns.ewm(span=20).mean().iloc[-1] * 24 * 365

                # Method 3: CAPM-like adjustment (simplified)
                market_return = 0.08  # Assume 8% market return
                beta = min(2.0, max(0.5, returns.std() / 0.02))  # Simplified beta
                capm_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)

                # Weighted combination
                expected_return = 0.4 * historical_mean + 0.4 * ewm_mean + 0.2 * capm_return

                # Cap returns at reasonable levels
                expected_returns[symbol] = max(-0.5, min(expected_return, 0.5))  # -50% to +50%

            logger.info(f"Expected returns calculated for {len(expected_returns)} assets")
            return expected_returns

        except Exception as e:
            logger.error(f"Expected returns calculation failed: {e}")
            return {symbol: 0.02 for symbol in price_data.keys()}

    def calculate_covariance_matrix(self, price_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Calculate covariance matrix for portfolio optimization."""
        try:
            symbols = list(price_data.keys())
            returns_matrix = []

            # Align all return series
            common_dates = None
            returns_data = {}

            for symbol, df in price_data.items():
                if len(df) >= self.lookback_period and "close" in df.columns:
                    returns = df["close"].pct_change().dropna()
                    returns = returns.tail(self.lookback_period)

                    if common_dates is None:
                        common_dates = returns.index
                    else:
                        common_dates = common_dates.intersection(returns.index)

                    returns_data[symbol] = returns

            # Create aligned returns matrix
            for symbol in symbols:
                if symbol in returns_data:
                    aligned_returns = returns_data[symbol].reindex(common_dates).fillna(0)
                    returns_matrix.append(aligned_returns.values)
                else:
                    # Fill missing symbols with zero returns
                    returns_matrix.append(np.zeros(len(common_dates)))

            returns_matrix = np.array(returns_matrix)

            if returns_matrix.shape[1] < 10:  # Need minimum data
                # Use identity matrix with estimated volatilities
                volatilities = [0.02] * len(symbols)  # Default 2% daily vol
                cov_matrix = np.diag([vol**2 for vol in volatilities])
            else:
                # Calculate sample covariance and apply shrinkage
                sample_cov = np.cov(returns_matrix)

                # Ledoit-Wolf shrinkage towards identity matrix
                n, p = returns_matrix.shape[1], returns_matrix.shape[0]
                shrinkage_target = np.trace(sample_cov) / p * np.eye(p)

                # Simplified shrinkage parameter
                shrinkage_param = min(0.8, max(0.1, 1.0 / n))

                cov_matrix = (1 - shrinkage_param) * sample_cov + shrinkage_param * shrinkage_target

            # Annualize covariance matrix
            cov_matrix *= 24 * 365  # For 30-minute intervals

            return cov_matrix

        except Exception as e:
            logger.error(f"Covariance matrix calculation failed: {e}")
            # Return identity matrix as fallback
            n_assets = len(price_data)
            return np.eye(n_assets) * 0.02**2  # 2% volatility assumption

    def _optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: ConstraintSet,
    ) -> Any:
        """Optimize for maximum Sharpe ratio."""
        n_assets = len(expected_returns)

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / (
                portfolio_vol + 1e-8
            )  # Negative for minimization

        # Constraints
        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]

        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints_list
        )
        return result

    def _optimize_min_volatility(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: ConstraintSet,
    ) -> Any:
        """Optimize for minimum volatility."""
        n_assets = len(expected_returns)

        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        ]

        if constraints.target_return:
            constraints_list.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.dot(x, expected_returns) - constraints.target_return,
                }
            )

        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints_list
        )
        return result

    def _optimize_risk_parity(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: ConstraintSet,
    ) -> Any:
        """Optimize for risk parity (equal risk contribution)."""
        n_assets = len(expected_returns)

        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib

            # Minimize difference between risk contributions
            target_contrib = portfolio_vol**2 / n_assets
            return np.sum((contrib - target_contrib) ** 2)

        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        ]

        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints_list
        )
        return result

    def _optimize_max_diversification(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: ConstraintSet,
    ) -> Any:
        """Optimize for maximum diversification ratio."""
        n_assets = len(expected_returns)

        def objective(weights):
            # Diversification ratio = weighted average volatility / portfolio volatility
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights, individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -weighted_avg_vol / (portfolio_vol + 1e-8)  # Negative for minimization

        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        ]

        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints_list
        )
        return result

    def optimize_portfolio(
        self,
        price_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, float],
        method: str = "max_sharpe",
        constraints: Optional[ConstraintSet] = None,
    ) -> PortfolioAllocation:
        """Optimize portfolio allocation."""
        try:
            logger.info(f"Starting portfolio optimization using {method} method")

            if not price_data:
                return PortfolioAllocation({}, 0, 0, 0, 0, 0, method)

            symbols = list(price_data.keys())

            # Calculate expected returns and covariance matrix
            expected_returns_dict = self.calculate_expected_returns(price_data)
            expected_returns = np.array(
                [expected_returns_dict.get(symbol, 0.02) for symbol in symbols]
            )
            cov_matrix = self.calculate_covariance_matrix(price_data)

            # Set default constraints if none provided
            if constraints is None:
                constraints = ConstraintSet()

            # Optimize using selected method
            if method in self.optimization_methods:
                result = self.optimization_methods[method](
                    expected_returns, cov_matrix, constraints
                )
            else:
                logger.warning(f"Unknown optimization method {method}, using max_sharpe")
                result = self._optimize_max_sharpe(expected_returns, cov_matrix, constraints)

            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                # Return equal weights as fallback
                equal_weight = 1.0 / len(symbols)
                weights = {symbol: equal_weight for symbol in symbols}
            else:
                weights = {symbol: max(0, weight) for symbol, weight in zip(symbols, result.x)}

                # Normalize weights to sum to 1
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

            # Calculate portfolio metrics
            weights_array = np.array([weights[symbol] for symbol in symbols])
            portfolio_return = np.dot(weights_array, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_vol + 1e-8)

            # Calculate diversification ratio
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights_array, individual_vols)
            diversification_ratio = weighted_avg_vol / (portfolio_vol + 1e-8)

            allocation = PortfolioAllocation(
                weights=weights,
                expected_return=portfolio_return,
                expected_volatility=portfolio_vol,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_estimate=portfolio_vol * 2.5,  # Rough estimate
                diversification_ratio=diversification_ratio,
                optimization_method=method,
            )

            logger.info(
                f"Portfolio optimization complete: "
                f"expected_return={portfolio_return:.2%}, "
                f"volatility={portfolio_vol:.2%}, "
                f"sharpe={sharpe_ratio:.2f}"
            )

            return allocation

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Return equal weights as fallback
            equal_weight = 1.0 / len(price_data) if price_data else 0
            return PortfolioAllocation(
                weights={symbol: equal_weight for symbol in price_data.keys()},
                expected_return=0.05,  # 5% default
                expected_volatility=0.15,  # 15% default
                sharpe_ratio=0.2,
                max_drawdown_estimate=0.15,
                diversification_ratio=1.0,
                optimization_method="fallback",
            )

    def should_rebalance(
        self,
        current_positions: Dict[str, float],
        target_allocation: PortfolioAllocation,
    ) -> bool:
        """Determine if portfolio should be rebalanced."""
        try:
            if not current_positions or not target_allocation.weights:
                return True

            # Calculate total current position value
            total_current = sum(abs(pos) for pos in current_positions.values())
            if total_current < 0.01:  # Less than 1% invested
                return True

            # Check deviation from target weights
            max_deviation = 0
            for symbol, target_weight in target_allocation.weights.items():
                current_weight = abs(current_positions.get(symbol, 0)) / total_current
                deviation = abs(current_weight - target_weight)
                max_deviation = max(max_deviation, deviation)

            should_rebalance = max_deviation > self.rebalance_threshold

            if should_rebalance:
                logger.info(
                    f"Rebalancing triggered: max deviation {max_deviation:.2%} > threshold {self.rebalance_threshold:.2%}"
                )

            return should_rebalance

        except Exception as e:
            logger.error(f"Rebalance check failed: {e}")
            return False

    def generate_rebalance_orders(
        self,
        current_positions: Dict[str, float],
        target_allocation: PortfolioAllocation,
        available_capital: float,
    ) -> Dict[str, float]:
        """Generate orders to rebalance portfolio to target allocation."""
        try:
            rebalance_orders = {}

            if not target_allocation.weights:
                return rebalance_orders

            # Calculate target positions in absolute terms
            total_target_capital = available_capital * 0.8  # Use 80% of available capital

            for symbol, target_weight in target_allocation.weights.items():
                target_position = total_target_capital * target_weight
                current_position = abs(current_positions.get(symbol, 0)) * available_capital

                # Calculate required change
                position_change = target_position - current_position

                # Only include significant changes
                if abs(position_change / available_capital) > 0.01:  # >1% of capital
                    rebalance_orders[symbol] = position_change / available_capital

            logger.info(f"Generated {len(rebalance_orders)} rebalance orders")
            return rebalance_orders

        except Exception as e:
            logger.error(f"Rebalance order generation failed: {e}")
            return {}


if __name__ == "__main__":
    # Test the portfolio optimizer
    test_config = {
        "portfolio_optimization": {
            "rebalance_threshold": 0.05,
            "lookback_period": 60,
            "risk_free_rate": 0.02,
        }
    }

    optimizer = PortfolioOptimizer(test_config)
    print("Portfolio Optimizer initialized successfully!")
