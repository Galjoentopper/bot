"""
Dynamic Position Sizing System

Implements sophisticated position sizing algorithms including Kelly Criterion,
volatility-based sizing, risk parity, and adaptive sizing based on market conditions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .risk_calculator import RiskCalculator


class SizingMethod(Enum):
    """Position sizing methods"""

    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    ADAPTIVE_VOLATILITY = "adaptive_volatility"
    MAX_DRAWDOWN_ADJUSTED = "max_drawdown_adjusted"


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""

    recommended_size: float  # Recommended position size (as fraction of portfolio)
    max_position_size: float  # Maximum allowed position size
    risk_adjusted_size: float  # Risk-adjusted position size

    # Sizing method used
    sizing_method: SizingMethod

    # Risk metrics that influenced sizing
    estimated_volatility: float
    estimated_var: float
    kelly_fraction: Optional[float] = None
    confidence_score: float = 0.5  # Confidence in sizing recommendation (0-1)

    # Constraints applied
    size_constraints: Dict[str, float] = None
    reasoning: str = ""  # Human-readable explanation

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.size_constraints is None:
            self.size_constraints = {}


class DynamicPositionSizer:
    """Advanced dynamic position sizing system"""

    def __init__(
        self,
        base_position_size: float = 0.05,  # 5% base position size
        max_position_size: float = 0.20,  # 20% maximum position
        min_position_size: float = 0.001,  # 0.1% minimum position
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        target_volatility: float = 0.15,  # 15% target volatility
        kelly_multiplier: float = 0.25,  # Conservative Kelly multiplier
        lookback_days: int = 252,
    ):  # 1 year for calculations
        """
        Initialize dynamic position sizer

        Args:
            base_position_size: Base position size as fraction of portfolio
            max_position_size: Maximum allowed position size
            min_position_size: Minimum position size threshold
            risk_free_rate: Annual risk-free rate
            target_volatility: Target portfolio volatility
            kelly_multiplier: Multiplier for Kelly criterion (for safety)
            lookback_days: Days of historical data to use
        """
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.kelly_multiplier = kelly_multiplier
        self.lookback_days = lookback_days

        self.logger = logging.getLogger(__name__)
        self.risk_calculator = RiskCalculator(lookback_days=lookback_days)

    def calculate_position_size(
        self,
        asset_returns: np.ndarray,
        predicted_return: Optional[float] = None,
        current_portfolio_value: float = 100000,
        current_positions: Optional[Dict[str, float]] = None,
        sizing_method: SizingMethod = SizingMethod.ADAPTIVE_VOLATILITY,
        market_regime: Optional[str] = None,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using specified method

        Args:
            asset_returns: Historical returns for the asset
            predicted_return: Model's predicted return (optional)
            current_portfolio_value: Current portfolio value
            current_positions: Current positions {asset: value}
            sizing_method: Position sizing method to use
            market_regime: Current market regime ('bull', 'bear', 'sideways')
        """

        if len(asset_returns) < 30:
            raise ValueError("Insufficient historical data for position sizing")

        # Calculate risk metrics for the asset
        risk_metrics = self.risk_calculator.calculate_comprehensive_risk(asset_returns)

        # Apply different sizing methods
        if sizing_method == SizingMethod.FIXED_FRACTIONAL:
            result = self._fixed_fractional_sizing(risk_metrics)

        elif sizing_method == SizingMethod.KELLY_CRITERION:
            result = self._kelly_criterion_sizing(asset_returns, predicted_return, risk_metrics)

        elif sizing_method == SizingMethod.VOLATILITY_ADJUSTED:
            result = self._volatility_adjusted_sizing(risk_metrics)

        elif sizing_method == SizingMethod.RISK_PARITY:
            result = self._risk_parity_sizing(asset_returns, current_positions, risk_metrics)

        elif sizing_method == SizingMethod.ADAPTIVE_VOLATILITY:
            result = self._adaptive_volatility_sizing(asset_returns, risk_metrics, market_regime)

        elif sizing_method == SizingMethod.MAX_DRAWDOWN_ADJUSTED:
            result = self._max_drawdown_adjusted_sizing(risk_metrics)

        else:
            raise ValueError(f"Unknown sizing method: {sizing_method}")

        # Apply portfolio-level constraints
        result = self._apply_portfolio_constraints(
            result, current_positions, current_portfolio_value
        )

        # Add confidence scoring
        result.confidence_score = self._calculate_confidence_score(
            result, risk_metrics, asset_returns
        )

        return result

    def _fixed_fractional_sizing(self, risk_metrics) -> PositionSizeResult:
        """Fixed fractional position sizing"""

        return PositionSizeResult(
            recommended_size=self.base_position_size,
            max_position_size=self.max_position_size,
            risk_adjusted_size=self.base_position_size,
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            estimated_volatility=risk_metrics.realized_volatility,
            estimated_var=risk_metrics.var_1d_95,
            reasoning=f"Fixed {self.base_position_size:.1%} position size",
        )

    def _kelly_criterion_sizing(
        self, returns: np.ndarray, predicted_return: Optional[float], risk_metrics
    ) -> PositionSizeResult:
        """Kelly criterion position sizing"""

        if predicted_return is None:
            # Use historical mean return
            predicted_return = np.mean(returns)

        # Annualize predicted return if it's daily
        if abs(predicted_return) < 0.1:  # Assume daily if small
            annual_predicted = predicted_return * 252
        else:
            annual_predicted = predicted_return

        variance = risk_metrics.realized_volatility**2

        if variance <= 0:
            kelly_fraction = 0
        else:
            # Kelly formula: f = (bp - q) / b
            # Where b = odds, p = prob of win, q = prob of loss
            # Simplified: f = (expected_return - risk_free_rate) / variance
            excess_return = annual_predicted - self.risk_free_rate
            kelly_fraction = excess_return / variance

        # Apply safety multiplier and constraints
        conservative_kelly = kelly_fraction * self.kelly_multiplier
        recommended_size = np.clip(conservative_kelly, 0, self.max_position_size)

        confidence = min(abs(kelly_fraction) / 0.5, 1.0)  # Higher for stronger signal

        return PositionSizeResult(
            recommended_size=max(recommended_size, self.min_position_size),
            max_position_size=self.max_position_size,
            risk_adjusted_size=recommended_size,
            sizing_method=SizingMethod.KELLY_CRITERION,
            estimated_volatility=risk_metrics.realized_volatility,
            estimated_var=risk_metrics.var_1d_95,
            kelly_fraction=kelly_fraction,
            confidence_score=confidence,
            reasoning=f"Kelly fraction: {kelly_fraction:.3f}, conservative: {conservative_kelly:.3f}",
        )

    def _volatility_adjusted_sizing(self, risk_metrics) -> PositionSizeResult:
        """Volatility-adjusted position sizing"""

        current_vol = risk_metrics.realized_volatility

        if current_vol <= 0:
            vol_adjustment = 1.0
        else:
            # Scale position inversely with volatility
            vol_adjustment = self.target_volatility / current_vol

        # Apply reasonable bounds to volatility adjustment
        vol_adjustment = np.clip(vol_adjustment, 0.1, 3.0)

        adjusted_size = self.base_position_size * vol_adjustment
        recommended_size = np.clip(adjusted_size, self.min_position_size, self.max_position_size)

        return PositionSizeResult(
            recommended_size=recommended_size,
            max_position_size=self.max_position_size,
            risk_adjusted_size=adjusted_size,
            sizing_method=SizingMethod.VOLATILITY_ADJUSTED,
            estimated_volatility=current_vol,
            estimated_var=risk_metrics.var_1d_95,
            reasoning=f"Volatility adjustment: {vol_adjustment:.2f}x (target: {self.target_volatility:.1%}, current: {current_vol:.1%})",
        )

    def _risk_parity_sizing(
        self,
        returns: np.ndarray,
        current_positions: Optional[Dict[str, float]],
        risk_metrics,
    ) -> PositionSizeResult:
        """Risk parity position sizing"""

        if current_positions is None or len(current_positions) <= 1:
            # Fall back to volatility adjustment for single asset
            return self._volatility_adjusted_sizing(risk_metrics)

        # Calculate risk contribution target
        num_positions = len(current_positions) + 1  # Include new position
        target_risk_contrib = 1.0 / num_positions

        current_vol = risk_metrics.realized_volatility

        if current_vol <= 0:
            recommended_size = self.base_position_size
        else:
            # Size to achieve target risk contribution
            # Simplified: position_size = target_risk / asset_volatility
            recommended_size = target_risk_contrib / current_vol

        recommended_size = np.clip(recommended_size, self.min_position_size, self.max_position_size)

        return PositionSizeResult(
            recommended_size=recommended_size,
            max_position_size=self.max_position_size,
            risk_adjusted_size=recommended_size,
            sizing_method=SizingMethod.RISK_PARITY,
            estimated_volatility=current_vol,
            estimated_var=risk_metrics.var_1d_95,
            reasoning=f"Risk parity sizing for {num_positions} assets, target contribution: {target_risk_contrib:.1%}",
        )

    def _adaptive_volatility_sizing(
        self, returns: np.ndarray, risk_metrics, market_regime: Optional[str]
    ) -> PositionSizeResult:
        """Adaptive volatility-based sizing with regime awareness"""

        current_vol = risk_metrics.realized_volatility

        # Base volatility adjustment
        if current_vol <= 0:
            vol_adjustment = 1.0
        else:
            vol_adjustment = self.target_volatility / current_vol

        # Regime-based adjustments
        regime_multiplier = 1.0
        if market_regime == "bear":
            regime_multiplier = 0.7  # Reduce position sizes in bear market
        elif market_regime == "bull":
            regime_multiplier = 1.2  # Increase in bull market
        elif market_regime == "high_volatility":
            regime_multiplier = 0.6  # Very conservative in high vol

        # Drawdown adjustment
        drawdown_multiplier = 1.0
        if risk_metrics.current_drawdown > 0.10:  # 10% drawdown
            drawdown_multiplier = 0.8
        elif risk_metrics.current_drawdown > 0.20:  # 20% drawdown
            drawdown_multiplier = 0.5

        # Recent performance adjustment
        recent_returns = returns[-20:]  # Last 20 days
        if len(recent_returns) >= 10:
            recent_sharpe = (
                np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                if np.std(recent_returns) > 0
                else 0
            )

            # Adjust based on recent performance
            if recent_sharpe > 1.0:  # Good recent performance
                performance_multiplier = 1.1
            elif recent_sharpe < -0.5:  # Poor recent performance
                performance_multiplier = 0.8
            else:
                performance_multiplier = 1.0
        else:
            performance_multiplier = 1.0

        # Combine all adjustments
        total_adjustment = (
            vol_adjustment * regime_multiplier * drawdown_multiplier * performance_multiplier
        )
        total_adjustment = np.clip(total_adjustment, 0.1, 2.5)

        adjusted_size = self.base_position_size * total_adjustment
        recommended_size = np.clip(adjusted_size, self.min_position_size, self.max_position_size)

        reasoning_parts = [
            f"Vol adj: {vol_adjustment:.2f}x",
            f"Regime ({market_regime}): {regime_multiplier:.2f}x",
            f"Drawdown: {drawdown_multiplier:.2f}x",
            f"Performance: {performance_multiplier:.2f}x",
        ]

        return PositionSizeResult(
            recommended_size=recommended_size,
            max_position_size=self.max_position_size,
            risk_adjusted_size=adjusted_size,
            sizing_method=SizingMethod.ADAPTIVE_VOLATILITY,
            estimated_volatility=current_vol,
            estimated_var=risk_metrics.var_1d_95,
            reasoning=f"Adaptive sizing - {', '.join(reasoning_parts)} = {total_adjustment:.2f}x total",
        )

    def _max_drawdown_adjusted_sizing(self, risk_metrics) -> PositionSizeResult:
        """Position sizing adjusted for maximum drawdown risk"""

        max_dd = risk_metrics.max_drawdown
        current_dd = risk_metrics.current_drawdown

        # Reduce position size based on historical max drawdown
        if max_dd > 0.5:  # More than 50% historical drawdown
            dd_adjustment = 0.5
        elif max_dd > 0.3:  # 30-50% historical drawdown
            dd_adjustment = 0.7
        elif max_dd > 0.2:  # 20-30% historical drawdown
            dd_adjustment = 0.85
        else:
            dd_adjustment = 1.0

        # Further reduce if currently in drawdown
        if current_dd > 0.1:  # Currently in 10%+ drawdown
            current_dd_adjustment = 1 - (current_dd * 0.5)  # Reduce proportionally
        else:
            current_dd_adjustment = 1.0

        total_adjustment = dd_adjustment * current_dd_adjustment

        adjusted_size = self.base_position_size * total_adjustment
        recommended_size = np.clip(adjusted_size, self.min_position_size, self.max_position_size)

        return PositionSizeResult(
            recommended_size=recommended_size,
            max_position_size=self.max_position_size,
            risk_adjusted_size=adjusted_size,
            sizing_method=SizingMethod.MAX_DRAWDOWN_ADJUSTED,
            estimated_volatility=risk_metrics.realized_volatility,
            estimated_var=risk_metrics.var_1d_95,
            reasoning=f"Max DD adjustment: {dd_adjustment:.2f}x, current DD adjustment: {current_dd_adjustment:.2f}x",
        )

    def _apply_portfolio_constraints(
        self,
        result: PositionSizeResult,
        current_positions: Optional[Dict[str, float]],
        portfolio_value: float,
    ) -> PositionSizeResult:
        """Apply portfolio-level position sizing constraints"""

        constraints_applied = {}
        original_size = result.recommended_size

        # Maximum position size constraint (already applied in individual methods)
        if result.recommended_size > self.max_position_size:
            result.recommended_size = self.max_position_size
            constraints_applied["max_position"] = self.max_position_size

        # Minimum position size constraint
        if result.recommended_size < self.min_position_size:
            if original_size > 0:  # Only apply if originally wanted a position
                result.recommended_size = self.min_position_size
                constraints_applied["min_position"] = self.min_position_size
            else:
                result.recommended_size = 0  # Don't force minimum if signal was negative

        # Portfolio concentration constraint
        if current_positions:
            total_positions = sum(abs(pos) for pos in current_positions.values())
            total_exposure = (
                total_positions + result.recommended_size * portfolio_value
            ) / portfolio_value

            max_total_exposure = 0.8  # Maximum 80% of portfolio deployed

            if total_exposure > max_total_exposure:
                # Reduce position to stay within exposure limit
                available_capacity = max_total_exposure - (total_positions / portfolio_value)
                if available_capacity > 0:
                    result.recommended_size = min(result.recommended_size, available_capacity)
                    constraints_applied["portfolio_exposure"] = available_capacity
                else:
                    result.recommended_size = 0
                    constraints_applied["portfolio_exposure"] = 0

        result.size_constraints = constraints_applied

        # Update reasoning if constraints were applied
        if constraints_applied:
            constraint_text = ", ".join([f"{k}: {v:.3f}" for k, v in constraints_applied.items()])
            result.reasoning += f" | Constraints applied: {constraint_text}"

        return result

    def _calculate_confidence_score(
        self, result: PositionSizeResult, risk_metrics, returns: np.ndarray
    ) -> float:
        """Calculate confidence score for position sizing recommendation"""

        confidence_factors = []

        # Data quality factor
        data_points = len(returns)
        if data_points >= 252:  # 1 year+
            data_quality = 1.0
        elif data_points >= 126:  # 6 months+
            data_quality = 0.8
        elif data_points >= 60:  # 2 months+
            data_quality = 0.6
        else:
            data_quality = 0.4

        confidence_factors.append(data_quality)

        # Volatility stability factor
        recent_vol = np.std(returns[-30:]) if len(returns) >= 30 else np.std(returns)
        overall_vol = np.std(returns)

        if overall_vol > 0:
            vol_stability = 1 - abs(recent_vol - overall_vol) / overall_vol
            vol_stability = np.clip(vol_stability, 0.3, 1.0)
        else:
            vol_stability = 0.5

        confidence_factors.append(vol_stability)

        # Constraint factor (lower confidence if heavily constrained)
        if result.size_constraints:
            constraint_factor = 0.7  # Reduced confidence when constrained
        else:
            constraint_factor = 1.0

        confidence_factors.append(constraint_factor)

        # Method-specific factors
        if result.sizing_method == SizingMethod.KELLY_CRITERION:
            # Higher confidence for stronger Kelly signals
            if result.kelly_fraction and abs(result.kelly_fraction) > 0.1:
                method_confidence = min(abs(result.kelly_fraction) * 2, 1.0)
            else:
                method_confidence = 0.3
            confidence_factors.append(method_confidence)

        elif result.sizing_method == SizingMethod.ADAPTIVE_VOLATILITY:
            # Moderate confidence for adaptive method
            confidence_factors.append(0.8)

        # Overall confidence is geometric mean of factors
        overall_confidence = np.prod(confidence_factors) ** (1.0 / len(confidence_factors))

        return float(np.clip(overall_confidence, 0.1, 1.0))

    def calculate_portfolio_sizes(
        self,
        assets_data: Dict[str, np.ndarray],  # {asset: returns}
        predicted_returns: Optional[Dict[str, float]] = None,
        current_portfolio_value: float = 100000,
        current_positions: Optional[Dict[str, float]] = None,
        sizing_method: SizingMethod = SizingMethod.RISK_PARITY,
    ) -> Dict[str, PositionSizeResult]:
        """
        Calculate position sizes for entire portfolio

        Args:
            assets_data: Dictionary of asset returns {asset_name: returns_array}
            predicted_returns: Model predictions {asset_name: predicted_return}
            current_portfolio_value: Current portfolio value
            current_positions: Current positions {asset: value}
            sizing_method: Position sizing method to use
        """

        results = {}

        for asset_name, returns in assets_data.items():
            predicted_return = predicted_returns.get(asset_name) if predicted_returns else None

            try:
                result = self.calculate_position_size(
                    asset_returns=returns,
                    predicted_return=predicted_return,
                    current_portfolio_value=current_portfolio_value,
                    current_positions=current_positions,
                    sizing_method=sizing_method,
                )
                results[asset_name] = result

            except Exception as e:
                self.logger.error(f"Error calculating position size for {asset_name}: {e}")

                # Fallback to minimum position
                results[asset_name] = PositionSizeResult(
                    recommended_size=self.min_position_size,
                    max_position_size=self.max_position_size,
                    risk_adjusted_size=self.min_position_size,
                    sizing_method=sizing_method,
                    estimated_volatility=0.0,
                    estimated_var=0.0,
                    confidence_score=0.1,
                    reasoning=f"Error in calculation, using minimum size: {str(e)}",
                )

        # Normalize position sizes to ensure they don't exceed portfolio capacity
        total_recommended = sum(result.recommended_size for result in results.values())
        max_total_allocation = 0.9  # Maximum 90% of portfolio allocated

        if total_recommended > max_total_allocation:
            normalization_factor = max_total_allocation / total_recommended

            for asset_name, result in results.items():
                result.recommended_size *= normalization_factor
                result.reasoning += f" | Normalized by {normalization_factor:.3f}x"

        return results
