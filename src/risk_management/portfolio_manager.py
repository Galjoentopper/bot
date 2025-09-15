"""
Portfolio Risk Management System

Implements correlation-based portfolio limits, sector concentration limits,
dynamic hedging, and comprehensive portfolio risk monitoring.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .position_sizer import DynamicPositionSizer, PositionSizeResult
from .risk_calculator import RiskCalculator, RiskMetrics


class RiskLimitType(Enum):
    """Types of risk limits"""

    POSITION_SIZE = "position_size"
    SECTOR_CONCENTRATION = "sector_concentration"
    CORRELATION_CLUSTER = "correlation_cluster"
    VAR_LIMIT = "var_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LEVERAGE_LIMIT = "leverage_limit"


@dataclass
class RiskLimit:
    """Risk limit definition"""

    limit_type: RiskLimitType
    limit_value: float
    current_value: float
    utilization_pct: float
    is_breached: bool
    warning_threshold: float = 0.8  # Warning at 80% utilization
    description: str = ""

    @property
    def is_warning(self) -> bool:
        return self.utilization_pct >= self.warning_threshold and not self.is_breached


@dataclass
class PortfolioConstraints:
    """Portfolio-level constraints and limits"""

    # Position limits
    max_single_position: float = 0.20  # 20% max single position
    max_sector_concentration: float = 0.40  # 40% max per sector
    max_correlation_cluster: float = 0.50  # 50% max highly correlated positions

    # Risk limits
    max_portfolio_var_95: float = 0.03  # 3% max daily VaR
    max_portfolio_var_99: float = 0.05  # 5% max daily VaR at 99%
    max_drawdown_limit: float = 0.15  # 15% max drawdown before action
    max_leverage: float = 1.0  # 100% max portfolio deployment

    # Correlation thresholds
    high_correlation_threshold: float = 0.7  # Positions > 70% correlation
    correlation_lookback_days: int = 90  # 3 months for correlation calculation

    # Dynamic adjustments
    volatility_adjustment_enabled: bool = True
    stress_test_enabled: bool = True


@dataclass
class CorrelationCluster:
    """Group of highly correlated assets"""

    cluster_id: str
    assets: List[str]
    avg_correlation: float
    total_allocation: float
    risk_contribution: float


class PortfolioRiskManager:
    """Comprehensive portfolio risk management system"""

    def __init__(
        self,
        constraints: Optional[PortfolioConstraints] = None,
        sector_mappings: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize portfolio risk manager

        Args:
            constraints: Portfolio constraints and limits
            sector_mappings: Asset to sector mapping {asset: sector}
        """
        self.constraints = constraints or PortfolioConstraints()
        self.sector_mappings = sector_mappings or {}

        self.logger = logging.getLogger(__name__)
        self.risk_calculator = RiskCalculator()

        # Risk monitoring state
        self.current_limits = {}
        self.correlation_matrix = None
        self.correlation_clusters = []

    def analyze_portfolio_risk(
        self,
        positions: Dict[str, float],  # {asset: position_value}
        asset_returns: Dict[str, np.ndarray],  # {asset: returns_history}
        portfolio_value: float,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk analysis

        Returns:
            Dictionary with portfolio risk metrics and limit analysis
        """

        if not positions:
            return {"status": "empty_portfolio", "risk_metrics": {}, "limits": {}}

        self.logger.info(
            f"Analyzing portfolio risk for {len(positions)} positions, total value: ${portfolio_value:,.0f}"
        )

        # Calculate portfolio weights
        position_values = np.array(list(positions.values()))
        weights = position_values / portfolio_value
        assets = list(positions.keys())

        # Ensure we have returns data for all positions
        available_assets = [asset for asset in assets if asset in asset_returns]
        if len(available_assets) != len(assets):
            missing_assets = set(assets) - set(available_assets)
            self.logger.warning(f"Missing returns data for assets: {missing_assets}")

        # Calculate correlation matrix
        self.correlation_matrix = self._calculate_correlation_matrix(
            asset_returns, available_assets
        )

        # Identify correlation clusters
        self.correlation_clusters = self._identify_correlation_clusters(
            self.correlation_matrix, available_assets, positions, portfolio_value
        )

        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(
            asset_returns, positions, portfolio_value
        )

        # Calculate comprehensive risk metrics
        portfolio_risk_metrics = self.risk_calculator.calculate_comprehensive_risk(
            returns=portfolio_returns, benchmark_returns=benchmark_returns
        )

        # Analyze individual risk limits
        risk_limits = self._analyze_risk_limits(positions, portfolio_value, portfolio_risk_metrics)

        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(
            positions, asset_returns, portfolio_value
        )

        # Sector analysis
        sector_analysis = self._analyze_sector_concentration(positions, portfolio_value)

        # Stress testing
        stress_test_results = self._run_stress_tests(positions, asset_returns, portfolio_value)

        return {
            "timestamp": datetime.now(),
            "portfolio_value": portfolio_value,
            "num_positions": len(positions),
            "portfolio_weights": dict(zip(assets, weights)),
            "portfolio_risk_metrics": portfolio_risk_metrics,
            "risk_limits": risk_limits,
            "correlation_analysis": {
                "correlation_matrix": (
                    self.correlation_matrix.tolist()
                    if self.correlation_matrix is not None
                    else None
                ),
                "correlation_clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "assets": cluster.assets,
                        "avg_correlation": cluster.avg_correlation,
                        "total_allocation": cluster.total_allocation,
                        "risk_contribution": cluster.risk_contribution,
                    }
                    for cluster in self.correlation_clusters
                ],
            },
            "risk_contributions": risk_contributions,
            "sector_analysis": sector_analysis,
            "stress_test_results": stress_test_results,
            "overall_risk_status": self._determine_overall_risk_status(risk_limits),
            "recommendations": self._generate_risk_recommendations(
                risk_limits, self.correlation_clusters, sector_analysis
            ),
        }

    def _calculate_correlation_matrix(
        self, asset_returns: Dict[str, np.ndarray], assets: List[str]
    ) -> Optional[np.ndarray]:
        """Calculate correlation matrix for portfolio assets"""

        if len(assets) < 2:
            return None

        # Align all return series to same length
        min_length = min(len(asset_returns[asset]) for asset in assets)
        lookback_length = min(min_length, self.constraints.correlation_lookback_days)

        if lookback_length < 30:
            self.logger.warning("Insufficient data for correlation analysis")
            return None

        # Create returns matrix
        returns_matrix = np.column_stack(
            [asset_returns[asset][-lookback_length:] for asset in assets]
        )

        # Handle NaN values
        if np.any(np.isnan(returns_matrix)):
            # Forward fill and backward fill NaN values
            returns_df = pd.DataFrame(returns_matrix, columns=assets)
            returns_df = returns_df.fillna(method="ffill").fillna(method="bfill")
            returns_matrix = returns_df.values

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(returns_matrix.T)

        return correlation_matrix

    def _identify_correlation_clusters(
        self,
        correlation_matrix: Optional[np.ndarray],
        assets: List[str],
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> List[CorrelationCluster]:
        """Identify clusters of highly correlated assets"""

        if correlation_matrix is None or len(assets) < 2:
            return []

        clusters = []
        used_assets = set()

        for i, asset_i in enumerate(assets):
            if asset_i in used_assets:
                continue

            # Find all assets highly correlated with this one
            cluster_assets = [asset_i]
            correlations = []

            for j, asset_j in enumerate(assets):
                if i != j and asset_j not in used_assets:
                    correlation = correlation_matrix[i, j]
                    if abs(correlation) >= self.constraints.high_correlation_threshold:
                        cluster_assets.append(asset_j)
                        correlations.append(abs(correlation))

            if len(cluster_assets) > 1:
                # Calculate cluster metrics
                avg_correlation = np.mean(correlations) if correlations else 0
                total_allocation = (
                    sum(positions.get(asset, 0) for asset in cluster_assets) / portfolio_value
                )

                # Simplified risk contribution (could be more sophisticated)
                risk_contribution = total_allocation * avg_correlation

                cluster = CorrelationCluster(
                    cluster_id=f"cluster_{len(clusters) + 1}",
                    assets=cluster_assets,
                    avg_correlation=avg_correlation,
                    total_allocation=total_allocation,
                    risk_contribution=risk_contribution,
                )
                clusters.append(cluster)

                # Mark assets as used
                used_assets.update(cluster_assets)

        return clusters

    def _calculate_portfolio_returns(
        self,
        asset_returns: Dict[str, np.ndarray],
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> np.ndarray:
        """Calculate historical portfolio returns"""

        # Calculate weights
        weights = {asset: pos_value / portfolio_value for asset, pos_value in positions.items()}

        # Find common time period
        available_assets = [asset for asset in positions.keys() if asset in asset_returns]
        if not available_assets:
            return np.array([])

        min_length = min(len(asset_returns[asset]) for asset in available_assets)

        # Calculate weighted portfolio returns
        portfolio_returns = np.zeros(min_length)

        for asset in available_assets:
            asset_weight = weights.get(asset, 0)
            asset_return_series = asset_returns[asset][-min_length:]
            portfolio_returns += asset_weight * asset_return_series

        return portfolio_returns

    def _analyze_risk_limits(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        portfolio_risk_metrics: RiskMetrics,
    ) -> Dict[str, RiskLimit]:
        """Analyze all portfolio risk limits"""

        limits = {}

        # Position size limits
        for asset, position_value in positions.items():
            position_weight = position_value / portfolio_value

            limits[f"position_{asset}"] = RiskLimit(
                limit_type=RiskLimitType.POSITION_SIZE,
                limit_value=self.constraints.max_single_position,
                current_value=position_weight,
                utilization_pct=position_weight / self.constraints.max_single_position,
                is_breached=position_weight > self.constraints.max_single_position,
                description=f"Position size for {asset}: {position_weight:.1%}",
            )

        # Portfolio VaR limits
        limits["portfolio_var_95"] = RiskLimit(
            limit_type=RiskLimitType.VAR_LIMIT,
            limit_value=self.constraints.max_portfolio_var_95,
            current_value=portfolio_risk_metrics.var_1d_95,
            utilization_pct=portfolio_risk_metrics.var_1d_95
            / self.constraints.max_portfolio_var_95,
            is_breached=portfolio_risk_metrics.var_1d_95 > self.constraints.max_portfolio_var_95,
            description=f"Portfolio 95% VaR: {portfolio_risk_metrics.var_1d_95:.2%}",
        )

        limits["portfolio_var_99"] = RiskLimit(
            limit_type=RiskLimitType.VAR_LIMIT,
            limit_value=self.constraints.max_portfolio_var_99,
            current_value=portfolio_risk_metrics.var_1d_99,
            utilization_pct=portfolio_risk_metrics.var_1d_99
            / self.constraints.max_portfolio_var_99,
            is_breached=portfolio_risk_metrics.var_1d_99 > self.constraints.max_portfolio_var_99,
            description=f"Portfolio 99% VaR: {portfolio_risk_metrics.var_1d_99:.2%}",
        )

        # Drawdown limit
        limits["drawdown"] = RiskLimit(
            limit_type=RiskLimitType.DRAWDOWN_LIMIT,
            limit_value=self.constraints.max_drawdown_limit,
            current_value=portfolio_risk_metrics.current_drawdown,
            utilization_pct=portfolio_risk_metrics.current_drawdown
            / self.constraints.max_drawdown_limit,
            is_breached=portfolio_risk_metrics.current_drawdown
            > self.constraints.max_drawdown_limit,
            description=f"Current drawdown: {portfolio_risk_metrics.current_drawdown:.2%}",
        )

        # Leverage limit
        total_exposure = sum(abs(pos) for pos in positions.values()) / portfolio_value
        limits["leverage"] = RiskLimit(
            limit_type=RiskLimitType.LEVERAGE_LIMIT,
            limit_value=self.constraints.max_leverage,
            current_value=total_exposure,
            utilization_pct=total_exposure / self.constraints.max_leverage,
            is_breached=total_exposure > self.constraints.max_leverage,
            description=f"Portfolio leverage: {total_exposure:.1%}",
        )

        # Correlation cluster limits
        for cluster in self.correlation_clusters:
            limit_key = f"correlation_cluster_{cluster.cluster_id}"
            limits[limit_key] = RiskLimit(
                limit_type=RiskLimitType.CORRELATION_CLUSTER,
                limit_value=self.constraints.max_correlation_cluster,
                current_value=cluster.total_allocation,
                utilization_pct=cluster.total_allocation / self.constraints.max_correlation_cluster,
                is_breached=cluster.total_allocation > self.constraints.max_correlation_cluster,
                description=f"Correlation cluster {cluster.cluster_id} ({len(cluster.assets)} assets): {cluster.total_allocation:.1%}",
            )

        return limits

    def _calculate_risk_contributions(
        self,
        positions: Dict[str, float],
        asset_returns: Dict[str, np.ndarray],
        portfolio_value: float,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate risk contributions of each position"""

        risk_contributions = {}

        for asset, position_value in positions.items():
            if asset not in asset_returns:
                continue

            position_weight = position_value / portfolio_value
            asset_vol = np.std(asset_returns[asset]) * np.sqrt(252)  # Annualized

            # Simplified risk contribution calculation
            # More sophisticated would use marginal VaR
            risk_contrib = position_weight * asset_vol

            risk_contributions[asset] = {
                "position_weight": position_weight,
                "asset_volatility": asset_vol,
                "risk_contribution": risk_contrib,
                "risk_contribution_pct": (
                    risk_contrib
                    / sum(
                        pos_val
                        / portfolio_value
                        * np.std(asset_returns.get(ast, [0]))
                        * np.sqrt(252)
                        for ast, pos_val in positions.items()
                        if ast in asset_returns
                    )
                    if any(ast in asset_returns for ast in positions.keys())
                    else 0
                ),
            }

        return risk_contributions

    def _analyze_sector_concentration(
        self, positions: Dict[str, float], portfolio_value: float
    ) -> Dict[str, Any]:
        """Analyze sector concentration risk"""

        if not self.sector_mappings:
            return {"enabled": False, "reason": "No sector mappings provided"}

        sector_exposures = defaultdict(float)
        unmapped_assets = []

        for asset, position_value in positions.items():
            if asset in self.sector_mappings:
                sector = self.sector_mappings[asset]
                sector_exposures[sector] += position_value / portfolio_value
            else:
                unmapped_assets.append(asset)

        # Check sector limits
        sector_limit_breaches = []
        for sector, exposure in sector_exposures.items():
            if exposure > self.constraints.max_sector_concentration:
                sector_limit_breaches.append(
                    {
                        "sector": sector,
                        "exposure": exposure,
                        "limit": self.constraints.max_sector_concentration,
                        "excess": exposure - self.constraints.max_sector_concentration,
                    }
                )

        return {
            "enabled": True,
            "sector_exposures": dict(sector_exposures),
            "sector_limit_breaches": sector_limit_breaches,
            "unmapped_assets": unmapped_assets,
            "max_sector_exposure": (max(sector_exposures.values()) if sector_exposures else 0),
            "diversification_score": (
                1 - max(sector_exposures.values()) if sector_exposures else 0
            ),
        }

    def _run_stress_tests(
        self,
        positions: Dict[str, float],
        asset_returns: Dict[str, np.ndarray],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Run portfolio stress tests"""

        if not self.constraints.stress_test_enabled:
            return {"enabled": False}

        stress_scenarios = {
            "market_crash_2008": -0.20,  # 20% market drop
            "volatility_spike": 2.0,  # 2x volatility increase
            "correlation_increase": 0.9,  # All correlations → 0.9
            "tail_event_99": -3.0,  # 3 standard deviation event
        }

        results = {}

        # Calculate current portfolio returns for scenarios
        portfolio_returns = self._calculate_portfolio_returns(
            asset_returns, positions, portfolio_value
        )

        if len(portfolio_returns) == 0:
            return {"enabled": True, "error": "Insufficient data for stress testing"}

        for scenario_name, scenario_param in stress_scenarios.items():
            try:
                if scenario_name == "market_crash_2008":
                    # Apply uniform shock to all positions
                    stressed_portfolio_value = portfolio_value * (1 + scenario_param)
                    portfolio_loss = portfolio_value - stressed_portfolio_value

                elif scenario_name == "volatility_spike":
                    # Estimate loss from volatility increase
                    current_vol = np.std(portfolio_returns) * np.sqrt(252)
                    stressed_vol = current_vol * scenario_param
                    # Rough estimate: loss = vol increase * portfolio value * time factor
                    portfolio_loss = (stressed_vol - current_vol) * portfolio_value * 0.1

                elif scenario_name == "tail_event_99":
                    # Calculate loss from N-sigma event
                    portfolio_std = np.std(portfolio_returns)
                    tail_loss = abs(scenario_param) * portfolio_std * portfolio_value
                    portfolio_loss = tail_loss

                else:
                    portfolio_loss = 0

                results[scenario_name] = {
                    "scenario_parameter": scenario_param,
                    "estimated_loss": portfolio_loss,
                    "loss_percentage": (
                        portfolio_loss / portfolio_value if portfolio_value > 0 else 0
                    ),
                    "exceeds_var_limit": portfolio_loss / portfolio_value
                    > self.constraints.max_portfolio_var_99,
                }

            except Exception as e:
                self.logger.warning(f"Error in stress test {scenario_name}: {e}")
                results[scenario_name] = {"error": str(e)}

        return {
            "enabled": True,
            "scenarios": results,
            "max_scenario_loss": (
                max(
                    result.get("loss_percentage", 0)
                    for result in results.values()
                    if "error" not in result
                )
                if results
                else 0
            ),
        }

    def _determine_overall_risk_status(self, risk_limits: Dict[str, RiskLimit]) -> str:
        """Determine overall portfolio risk status"""

        breached_limits = [limit for limit in risk_limits.values() if limit.is_breached]
        warning_limits = [limit for limit in risk_limits.values() if limit.is_warning]

        if any(
            limit.limit_type in [RiskLimitType.VAR_LIMIT, RiskLimitType.DRAWDOWN_LIMIT]
            for limit in breached_limits
        ):
            return "critical"
        elif len(breached_limits) >= 3:
            return "high_risk"
        elif len(breached_limits) > 0:
            return "elevated_risk"
        elif len(warning_limits) >= 3:
            return "moderate_risk"
        elif len(warning_limits) > 0:
            return "low_risk"
        else:
            return "normal"

    def _generate_risk_recommendations(
        self,
        risk_limits: Dict[str, RiskLimit],
        correlation_clusters: List[CorrelationCluster],
        sector_analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate portfolio risk management recommendations"""

        recommendations = []

        # Check for breached limits
        breached_limits = [limit for limit in risk_limits.values() if limit.is_breached]

        if breached_limits:
            critical_breaches = [
                limit
                for limit in breached_limits
                if limit.limit_type in [RiskLimitType.VAR_LIMIT, RiskLimitType.DRAWDOWN_LIMIT]
            ]

            if critical_breaches:
                recommendations.append(
                    "URGENT: Critical risk limits breached. Consider reducing portfolio exposure immediately."
                )

            position_breaches = [
                limit
                for limit in breached_limits
                if limit.limit_type == RiskLimitType.POSITION_SIZE
            ]

            if position_breaches:
                assets_to_reduce = [
                    limit.description.split()[-1].rstrip(":") for limit in position_breaches[:3]
                ]
                recommendations.append(f"Reduce position sizes in: {', '.join(assets_to_reduce)}")

        # Check correlation clusters
        large_clusters = [
            cluster for cluster in correlation_clusters if cluster.total_allocation > 0.3
        ]
        if large_clusters:
            recommendations.append(
                f"Consider diversifying large correlation clusters: {', '.join(c.cluster_id for c in large_clusters)}"
            )

        # Check sector concentration
        if sector_analysis.get("enabled") and sector_analysis.get("sector_limit_breaches"):
            over_concentrated_sectors = [
                breach["sector"] for breach in sector_analysis["sector_limit_breaches"]
            ]
            recommendations.append(
                f"Reduce sector concentration in: {', '.join(over_concentrated_sectors)}"
            )

        # Warning level recommendations
        warning_limits = [limit for limit in risk_limits.values() if limit.is_warning]
        if len(warning_limits) >= 3:
            recommendations.append(
                "Multiple risk limits approaching thresholds. Monitor closely and consider rebalancing."
            )

        # Diversification recommendations
        if (
            len(correlation_clusters) == 0 and len(risk_limits) > 5
        ):  # Many positions but no clusters
            recommendations.append(
                "Portfolio appears well-diversified from correlation perspective."
            )
        elif len(correlation_clusters) > 2:
            recommendations.append(
                "Consider reducing the number of highly correlated asset groups."
            )

        if not recommendations:
            recommendations.append(
                "Portfolio risk profile appears acceptable. Continue monitoring."
            )

        return recommendations

    def check_trade_compliance(
        self,
        proposed_trade: Dict[str, float],  # {asset: trade_size}
        current_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Check if proposed trade complies with portfolio risk limits

        Returns:
            Dictionary with compliance status and recommendations
        """

        # Calculate positions after proposed trade
        new_positions = current_positions.copy()
        for asset, trade_size in proposed_trade.items():
            new_positions[asset] = new_positions.get(asset, 0) + trade_size

        # Remove zero positions
        new_positions = {k: v for k, v in new_positions.items() if abs(v) > 0.01}

        # Analyze risk with new positions
        new_risk_analysis = self.analyze_portfolio_risk(
            positions=new_positions,
            asset_returns=asset_returns,
            portfolio_value=portfolio_value,
        )

        # Check for new limit breaches
        new_breaches = [
            limit for limit in new_risk_analysis["risk_limits"].values() if limit.is_breached
        ]

        current_risk_analysis = self.analyze_portfolio_risk(
            positions=current_positions,
            asset_returns=asset_returns,
            portfolio_value=portfolio_value,
        )

        current_breaches = [
            limit for limit in current_risk_analysis["risk_limits"].values() if limit.is_breached
        ]

        # Determine compliance
        compliance_status = "approved"
        issues = []

        if len(new_breaches) > len(current_breaches):
            compliance_status = "rejected"
            issues.append("Trade would create new risk limit breaches")

        elif any(
            limit.limit_type in [RiskLimitType.VAR_LIMIT, RiskLimitType.DRAWDOWN_LIMIT]
            for limit in new_breaches
        ):
            compliance_status = "rejected"
            issues.append("Trade would breach critical risk limits (VaR/Drawdown)")

        elif len(new_breaches) > 0:
            compliance_status = "review_required"
            issues.append("Trade would breach non-critical risk limits")

        return {
            "compliance_status": compliance_status,
            "issues": issues,
            "current_risk_status": current_risk_analysis["overall_risk_status"],
            "new_risk_status": new_risk_analysis["overall_risk_status"],
            "risk_improvement": len(current_breaches) - len(new_breaches),
            "recommendations": new_risk_analysis["recommendations"],
            "detailed_analysis": {
                "current": current_risk_analysis,
                "proposed": new_risk_analysis,
            },
        }
