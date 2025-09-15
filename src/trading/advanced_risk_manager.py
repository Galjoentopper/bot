#!/usr/bin/env python3
"""
Advanced Risk Management System
==============================

Implements sophisticated risk management strategies:
- Kelly Criterion position sizing
- Dynamic correlation-based portfolio management
- Real-time volatility adjustment
- Advanced stop-loss and take-profit optimization
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics."""

    portfolio_var: float
    portfolio_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    correlation_exposure: float
    leverage_ratio: float
    risk_budget_utilization: float


@dataclass
class PositionRisk:
    """Risk assessment for individual positions."""

    symbol: str
    size_pct: float
    stop_loss: float
    take_profit: float
    max_loss_pct: float
    expected_return: float
    volatility: float
    kelly_fraction: float
    correlation_penalty: float


class AdvancedRiskManager:
    """Advanced risk management with Kelly Criterion and portfolio optimization."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced risk manager."""
        self.config = config

        # Risk management parameters
        risk_config = config.get("risk_management", {})
        self.max_portfolio_risk = risk_config.get(
            "max_portfolio_risk", 0.02
        )  # 2% max portfolio risk
        self.max_position_size = risk_config.get("max_position_pct", 0.15)  # 15% max per position
        self.max_correlation_exposure = risk_config.get(
            "max_correlation_exposure", 0.4
        )  # 40% max correlated positions
        self.kelly_multiplier = risk_config.get(
            "kelly_multiplier", 0.25
        )  # Be conservative with Kelly
        self.volatility_lookback = risk_config.get("volatility_lookback", 20)

        # Dynamic thresholds
        self.base_stop_loss = risk_config.get("stop_loss", 0.02)  # 2%
        self.base_take_profit = risk_config.get("take_profit", 0.04)  # 4%
        self.trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.015)  # 1.5%

        # Portfolio state tracking
        self.position_history = defaultdict(deque)  # Track position performance
        self.correlation_matrix = {}
        self.volatility_estimates = {}
        self.model_performance_history = defaultdict(deque)

        # Risk budget allocation
        self.risk_budget = {
            "trend_following": 0.4,
            "mean_reversion": 0.3,
            "momentum": 0.2,
            "arbitrage": 0.1,
        }

        logger.info(
            f"Advanced Risk Manager initialized with {self.max_portfolio_risk:.1%} portfolio risk limit"
        )

    def calculate_kelly_criterion(
        self,
        symbol: str,
        expected_return: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Calculate Kelly Criterion optimal position size."""
        try:
            if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
                return 0.0

            # Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win/avg_loss), p = win probability, q = loss probability
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
            p = win_rate
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Apply conservative multiplier and cap at reasonable levels
            kelly_fraction = max(
                0, min(kelly_fraction * self.kelly_multiplier, self.max_position_size)
            )

            logger.debug(
                f"Kelly Criterion for {symbol}: {kelly_fraction:.3f} "
                f"(win_rate={win_rate:.2f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f})"
            )

            return kelly_fraction

        except Exception as e:
            logger.warning(f"Kelly Criterion calculation failed for {symbol}: {e}")
            return self.max_position_size * 0.5  # Conservative fallback

    def estimate_position_volatility(self, symbol: str, price_history: pd.DataFrame) -> float:
        """Estimate position volatility using multiple methods."""
        try:
            if len(price_history) < self.volatility_lookback:
                return 0.02  # Default 2% daily volatility

            # Method 1: Historical returns volatility
            returns = price_history["close"].pct_change().dropna()
            historical_vol = returns.std() * np.sqrt(24)  # Annualized for 30m intervals

            # Method 2: Parkinson volatility (uses OHLC)
            if all(col in price_history.columns for col in ["high", "low", "open", "close"]):
                parkinson_vol = np.sqrt(
                    (1 / (4 * len(price_history) * np.log(2)))
                    * np.sum(np.log(price_history["high"] / price_history["low"]) ** 2)
                ) * np.sqrt(365)

                # Weighted average of methods
                volatility = 0.7 * historical_vol + 0.3 * parkinson_vol
            else:
                volatility = historical_vol

            # Cap volatility at reasonable bounds
            volatility = max(0.01, min(volatility, 1.0))  # 1% to 100%

            self.volatility_estimates[symbol] = volatility
            return volatility

        except Exception as e:
            logger.warning(f"Volatility estimation failed for {symbol}: {e}")
            return 0.02

    def calculate_correlation_matrix(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate rolling correlation matrix between symbols."""
        try:
            correlations = {}
            symbols = list(price_data.keys())

            # Extract returns for correlation calculation
            returns_data = {}
            for symbol, df in price_data.items():
                if len(df) >= self.volatility_lookback and "close" in df.columns:
                    returns = df["close"].pct_change().dropna()
                    if len(returns) >= self.volatility_lookback:
                        returns_data[symbol] = returns.tail(self.volatility_lookback)

            # Calculate pairwise correlations
            for i, symbol1 in enumerate(returns_data.keys()):
                for symbol2 in list(returns_data.keys())[i + 1 :]:
                    try:
                        corr = returns_data[symbol1].corr(returns_data[symbol2])
                        if not np.isnan(corr):
                            correlations[(symbol1, symbol2)] = abs(corr)  # Use absolute correlation
                            correlations[(symbol2, symbol1)] = abs(corr)
                    except Exception:
                        correlations[(symbol1, symbol2)] = 0.0
                        correlations[(symbol2, symbol1)] = 0.0

            self.correlation_matrix = correlations
            return correlations

        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return {}

    def calculate_correlation_penalty(
        self, symbol: str, current_positions: Dict[str, float]
    ) -> float:
        """Calculate correlation penalty for position sizing."""
        try:
            penalty = 0.0

            for other_symbol, position_size in current_positions.items():
                if other_symbol != symbol and position_size > 0.01:  # 1% minimum position
                    correlation = self.correlation_matrix.get((symbol, other_symbol), 0.0)

                    # Penalty increases with correlation and position size
                    penalty += correlation * position_size

            # Cap penalty at 50% reduction
            return min(penalty, 0.5)

        except Exception as e:
            logger.warning(f"Correlation penalty calculation failed for {symbol}: {e}")
            return 0.0

    def optimize_position_size(
        self,
        symbol: str,
        signal_strength: float,
        model_predictions: List[float],
        current_positions: Dict[str, float],
        price_history: pd.DataFrame,
    ) -> Tuple[float, PositionRisk]:
        """Optimize position size using multiple risk factors."""
        try:
            # Base size from signal strength (normalized 0-1)
            base_size = abs(signal_strength) * self.max_position_size

            # Calculate volatility
            volatility = self.estimate_position_volatility(symbol, price_history)

            # Volatility adjustment (inverse relationship)
            vol_adjustment = min(1.0, 0.02 / volatility)  # Scale down for higher volatility
            adjusted_size = base_size * vol_adjustment

            # Calculate Kelly Criterion if we have performance history
            kelly_size = self.calculate_kelly_from_history(symbol)

            # Use Kelly if available, otherwise use adjusted size
            optimal_size = min(kelly_size, adjusted_size) if kelly_size > 0 else adjusted_size

            # Apply correlation penalty
            correlation_penalty = self.calculate_correlation_penalty(symbol, current_positions)
            final_size = optimal_size * (1 - correlation_penalty)

            # Portfolio-level constraints
            total_exposure = sum(abs(pos) for pos in current_positions.values())
            if total_exposure + final_size > 0.8:  # 80% max total exposure
                final_size *= 0.8 / (total_exposure + final_size)

            # Calculate risk metrics
            stop_loss = self.calculate_dynamic_stop_loss(symbol, volatility, signal_strength)
            take_profit = self.calculate_dynamic_take_profit(symbol, volatility, signal_strength)

            position_risk = PositionRisk(
                symbol=symbol,
                size_pct=final_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_loss_pct=final_size * stop_loss,
                expected_return=signal_strength * 0.02,  # Estimate 2% max expected return
                volatility=volatility,
                kelly_fraction=kelly_size,
                correlation_penalty=correlation_penalty,
            )

            logger.info(
                f"Optimized position for {symbol}: {final_size:.2%} "
                f"(base={base_size:.2%}, vol_adj={vol_adjustment:.2f}, "
                f"corr_penalty={correlation_penalty:.2f})"
            )

            return final_size, position_risk

        except Exception as e:
            logger.error(f"Position optimization failed for {symbol}: {e}")
            return 0.0, PositionRisk(symbol, 0, 0, 0, 0, 0, 0, 0, 0)

    def calculate_kelly_from_history(self, symbol: str) -> float:
        """Calculate Kelly Criterion from historical performance."""
        try:
            history = self.position_history[symbol]
            if len(history) < 10:  # Need minimum history
                return 0.0

            # Convert history to wins/losses
            wins = [trade for trade in history if trade > 0]
            losses = [abs(trade) for trade in history if trade < 0]

            if not wins or not losses:
                return 0.0

            win_rate = len(wins) / len(history)
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)

            return self.calculate_kelly_criterion(symbol, 0, win_rate, avg_win, avg_loss)

        except Exception as e:
            logger.warning(f"Kelly from history calculation failed for {symbol}: {e}")
            return 0.0

    def calculate_dynamic_stop_loss(
        self, symbol: str, volatility: float, signal_strength: float
    ) -> float:
        """Calculate dynamic stop loss based on volatility and signal strength."""
        try:
            # Base stop loss adjusted for volatility
            vol_multiplier = max(1.0, volatility / 0.02)  # Scale from 2% base volatility
            dynamic_stop = self.base_stop_loss * vol_multiplier

            # Adjust for signal strength (stronger signals get wider stops)
            signal_multiplier = 1 + (abs(signal_strength) * 0.5)
            dynamic_stop *= signal_multiplier

            # Cap stop loss at reasonable levels
            return max(0.01, min(dynamic_stop, 0.05))  # 1% to 5%

        except Exception:
            return self.base_stop_loss

    def calculate_dynamic_take_profit(
        self, symbol: str, volatility: float, signal_strength: float
    ) -> float:
        """Calculate dynamic take profit target."""
        try:
            # Base take profit adjusted for volatility
            vol_multiplier = max(1.0, volatility / 0.02)
            dynamic_tp = self.base_take_profit * vol_multiplier

            # Adjust for signal strength (stronger signals get higher targets)
            signal_multiplier = 1 + (abs(signal_strength) * 0.3)
            dynamic_tp *= signal_multiplier

            # Maintain 2:1 risk-reward minimum
            stop_loss = self.calculate_dynamic_stop_loss(symbol, volatility, signal_strength)
            min_tp = stop_loss * 2

            return max(min_tp, min(dynamic_tp, 0.10))  # Cap at 10%

        except Exception:
            return self.base_take_profit

    def assess_portfolio_risk(
        self, positions: Dict[str, float], position_risks: Dict[str, PositionRisk]
    ) -> RiskMetrics:
        """Comprehensive portfolio risk assessment."""
        try:
            if not positions:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0)

            # Calculate portfolio-level metrics
            total_exposure = sum(abs(pos) for pos in positions.values())
            max_individual_loss = (
                max(risk.max_loss_pct for risk in position_risks.values()) if position_risks else 0
            )

            # Correlation exposure
            correlation_exposure = 0
            symbols = list(positions.keys())
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i + 1 :]:
                    correlation = self.correlation_matrix.get((symbol1, symbol2), 0)
                    exposure = correlation * abs(positions[symbol1]) * abs(positions[symbol2])
                    correlation_exposure += exposure

            # Portfolio volatility estimate (simplified)
            portfolio_vol = (
                np.sqrt(
                    sum(
                        (positions[symbol] * position_risks[symbol].volatility) ** 2
                        for symbol in positions.keys()
                        if symbol in position_risks
                    )
                )
                if position_risks
                else 0
            )

            # Risk budget utilization
            risk_budget_used = min(1.0, total_exposure / 0.8)  # Against 80% max

            risk_metrics = RiskMetrics(
                portfolio_var=portfolio_vol**2,
                portfolio_volatility=portfolio_vol,
                max_drawdown=max_individual_loss,
                sharpe_ratio=0.0,  # Will be calculated from performance
                correlation_exposure=correlation_exposure,
                leverage_ratio=total_exposure,
                risk_budget_utilization=risk_budget_used,
            )

            logger.info(
                f"Portfolio risk assessment: exposure={total_exposure:.1%}, "
                f"max_loss={max_individual_loss:.2%}, vol={portfolio_vol:.2%}"
            )

            return risk_metrics

        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)

    def update_performance_history(self, symbol: str, trade_return: float):
        """Update performance history for Kelly Criterion calculation."""
        try:
            self.position_history[symbol].append(trade_return)

            # Keep only recent history
            max_history = 100
            if len(self.position_history[symbol]) > max_history:
                self.position_history[symbol].popleft()

            logger.debug(f"Updated performance history for {symbol}: {trade_return:.4f}")

        except Exception as e:
            logger.warning(f"Failed to update performance history for {symbol}: {e}")

    def should_reduce_risk(self, recent_performance: List[float]) -> bool:
        """Determine if risk should be reduced based on recent performance."""
        try:
            if len(recent_performance) < 5:
                return False

            # Check for consecutive losses
            recent_returns = recent_performance[-5:]
            consecutive_losses = 0
            for ret in reversed(recent_returns):
                if ret < 0:
                    consecutive_losses += 1
                else:
                    break

            # Reduce risk after 3 consecutive losses
            if consecutive_losses >= 3:
                return True

            # Check for significant drawdown
            cumulative_return = np.prod([1 + r for r in recent_returns]) - 1
            if cumulative_return < -0.05:  # 5% drawdown
                return True

            return False

        except Exception as e:
            logger.warning(f"Risk reduction check failed: {e}")
            return False

    def get_risk_adjusted_signals(
        self,
        raw_signals: Dict[str, int],
        model_predictions: Dict[str, List[float]],
        current_positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Tuple[float, PositionRisk]]:
        """Apply advanced risk management to raw trading signals."""
        try:
            logger.info("Applying advanced risk management to trading signals...")

            # Update correlation matrix
            self.calculate_correlation_matrix(price_data)

            risk_adjusted_positions = {}

            for symbol, signal in raw_signals.items():
                if signal != 0 and symbol in price_data:
                    # Convert signal to signal strength (0-1)
                    signal_strength = abs(signal) * 0.5  # Scale down from {-1, 0, 1}

                    # Get model predictions if available
                    predictions = model_predictions.get(symbol, [signal_strength])

                    # Optimize position size
                    position_size, position_risk = self.optimize_position_size(
                        symbol=symbol,
                        signal_strength=signal_strength,
                        model_predictions=predictions,
                        current_positions=current_positions,
                        price_history=price_data[symbol],
                    )

                    # Apply signal direction
                    if signal < 0:
                        position_size *= -1

                    if abs(position_size) > 0.001:  # Minimum 0.1% position
                        risk_adjusted_positions[symbol] = (position_size, position_risk)

                        logger.info(
                            f"Risk-adjusted signal for {symbol}: "
                            f"size={position_size:.2%}, stop={position_risk.stop_loss:.2%}, "
                            f"target={position_risk.take_profit:.2%}"
                        )

            # Final portfolio-level risk check
            position_risks = {symbol: risk for symbol, (_, risk) in risk_adjusted_positions.items()}
            positions = {symbol: size for symbol, (size, _) in risk_adjusted_positions.items()}

            portfolio_risk = self.assess_portfolio_risk(positions, position_risks)

            if portfolio_risk.leverage_ratio > 0.8:  # Scale down if too leveraged
                scale_factor = 0.8 / portfolio_risk.leverage_ratio
                risk_adjusted_positions = {
                    symbol: (size * scale_factor, risk)
                    for symbol, (size, risk) in risk_adjusted_positions.items()
                }
                logger.warning(
                    f"Scaled down positions by {scale_factor:.2f} due to leverage limits"
                )

            logger.info(
                f"Risk management complete: {len(risk_adjusted_positions)} positions optimized"
            )
            return risk_adjusted_positions

        except Exception as e:
            logger.error(f"Risk management failed: {e}")
            return {}


if __name__ == "__main__":
    # Test the risk manager
    test_config = {
        "risk_management": {
            "max_portfolio_risk": 0.02,
            "max_position_pct": 0.15,
            "stop_loss": 0.02,
            "take_profit": 0.04,
        }
    }

    rm = AdvancedRiskManager(test_config)
    print("Advanced Risk Manager initialized successfully!")
