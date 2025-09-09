"""
Risk Calculation Engine

Implements VaR, CVaR, volatility estimation, and other risk metrics
using modern portfolio theory and statistical methods.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch package not available. GARCH volatility models will be disabled.")

try:
    import empyrical

    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False

from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a trading position or portfolio"""

    # Value at Risk metrics
    var_1d_95: float  # 1-day VaR at 95% confidence
    var_1d_99: float  # 1-day VaR at 99% confidence
    cvar_1d_95: float  # 1-day CVaR (Expected Shortfall) at 95%
    cvar_1d_99: float  # 1-day CVaR at 99%

    # Volatility metrics
    realized_volatility: float  # Historical volatility
    garch_volatility: Optional[float] = None  # GARCH volatility forecast

    # Drawdown metrics
    current_drawdown: float  # Current drawdown from peak
    max_drawdown: float  # Maximum historical drawdown
    drawdown_duration: int  # Days in current drawdown

    # Portfolio metrics (if applicable)
    portfolio_beta: Optional[float] = None  # Portfolio beta
    diversification_ratio: Optional[float] = None  # Diversification benefit

    # Risk-adjusted returns
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Additional metrics
    skewness: float = 0.0  # Return distribution skewness
    kurtosis: float = 0.0  # Return distribution kurtosis
    tail_ratio: Optional[float] = None  # Ratio of 95th to 5th percentile

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RiskCalculator:
    """Comprehensive risk calculation engine"""

    def __init__(
        self,
        lookback_days: int = 252,  # 1 year of trading days
        confidence_levels: List[float] = [0.95, 0.99],
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        target_return: float = 0.0,
    ):  # Target return for downside deviation
        """
        Initialize risk calculator

        Args:
            lookback_days: Days of historical data to use for calculations
            confidence_levels: Confidence levels for VaR/CVaR calculations
            risk_free_rate: Annual risk-free rate for Sharpe ratio
            target_return: Target return for downside deviation metrics
        """
        self.lookback_days = lookback_days
        self.confidence_levels = sorted(confidence_levels, reverse=True)
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return

        self.logger = logging.getLogger(__name__)

    def calculate_comprehensive_risk(
        self,
        returns: Union[pd.Series, np.ndarray],
        prices: Optional[Union[pd.Series, np.ndarray]] = None,
        portfolio_weights: Optional[np.ndarray] = None,
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics

        Args:
            returns: Historical returns (daily)
            prices: Historical prices (optional, for drawdown calculation)
            portfolio_weights: Portfolio weights (for multi-asset calculations)
            benchmark_returns: Benchmark returns for beta calculation
        """

        # Convert to numpy arrays
        if isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = np.array(returns)

        # Remove any NaN values
        returns_array = returns_array[~np.isnan(returns_array)]

        if len(returns_array) < 30:
            raise ValueError(
                "Insufficient data for risk calculation (minimum 30 observations required)"
            )

        # Use only recent data if we have too much
        if len(returns_array) > self.lookback_days:
            returns_array = returns_array[-self.lookbook_days :]

        # Calculate VaR and CVaR
        var_metrics = self._calculate_var_cvar(returns_array)

        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility(returns_array)

        # Calculate drawdown metrics
        if prices is not None:
            drawdown_metrics = self._calculate_drawdown_metrics(prices)
        else:
            # Estimate from returns
            drawdown_metrics = self._estimate_drawdown_from_returns(returns_array)

        # Calculate distribution metrics
        dist_metrics = self._calculate_distribution_metrics(returns_array)

        # Calculate risk-adjusted returns
        risk_adj_metrics = self._calculate_risk_adjusted_returns(returns_array)

        # Portfolio-specific metrics
        portfolio_metrics = {}
        if portfolio_weights is not None and len(portfolio_weights) > 1:
            portfolio_metrics = self._calculate_portfolio_metrics(returns_array, portfolio_weights)

        # Beta calculation
        beta = None
        if benchmark_returns is not None:
            beta = self._calculate_beta(returns_array, benchmark_returns)

        return RiskMetrics(
            var_1d_95=var_metrics["var_95"],
            var_1d_99=var_metrics["var_99"],
            cvar_1d_95=var_metrics["cvar_95"],
            cvar_1d_99=var_metrics["cvar_99"],
            realized_volatility=volatility_metrics["realized_vol"],
            garch_volatility=volatility_metrics.get("garch_vol"),
            current_drawdown=drawdown_metrics["current_drawdown"],
            max_drawdown=drawdown_metrics["max_drawdown"],
            drawdown_duration=drawdown_metrics["drawdown_duration"],
            portfolio_beta=beta,
            diversification_ratio=portfolio_metrics.get("diversification_ratio"),
            sharpe_ratio=risk_adj_metrics.get("sharpe_ratio"),
            sortino_ratio=risk_adj_metrics.get("sortino_ratio"),
            calmar_ratio=risk_adj_metrics.get("calmar_ratio"),
            skewness=dist_metrics["skewness"],
            kurtosis=dist_metrics["kurtosis"],
            tail_ratio=dist_metrics.get("tail_ratio"),
        )

    def _calculate_var_cvar(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""

        results = {}

        for conf_level in self.confidence_levels:
            # Historical VaR (percentile method)
            var_percentile = (1 - conf_level) * 100
            var_value = np.percentile(returns, var_percentile)

            # CVaR (Expected Shortfall) - average of returns below VaR
            tail_returns = returns[returns <= var_value]
            cvar_value = np.mean(tail_returns) if len(tail_returns) > 0 else var_value

            # Convert to positive values (loss convention)
            var_value = abs(var_value)
            cvar_value = abs(cvar_value)

            conf_key = str(int(conf_level * 100))
            results[f"var_{conf_key}"] = var_value
            results[f"cvar_{conf_key}"] = cvar_value

        # Parametric VaR using normal distribution (for comparison)
        returns_std = np.std(returns)
        returns_mean = np.mean(returns)

        for conf_level in self.confidence_levels:
            z_score = stats.norm.ppf(1 - conf_level)
            parametric_var = abs(returns_mean + z_score * returns_std)

            conf_key = str(int(conf_level * 100))
            results[f"parametric_var_{conf_key}"] = parametric_var

        return results

    def _calculate_volatility(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate various volatility measures"""

        # Realized volatility (annualized)
        realized_vol = np.std(returns) * np.sqrt(252)  # 252 trading days

        results = {"realized_vol": realized_vol, "daily_vol": np.std(returns)}

        # GARCH volatility forecast (if available)
        if ARCH_AVAILABLE and len(returns) >= 100:
            try:
                # Fit GARCH(1,1) model
                returns_pct = returns * 100  # Convert to percentage for better numerical stability

                # Remove extreme outliers that can break GARCH fitting
                q99 = np.percentile(np.abs(returns_pct), 99)
                cleaned_returns = np.clip(returns_pct, -q99, q99)

                am = arch_model(cleaned_returns, vol="Garch", p=1, q=1, dist="normal")
                res = am.fit(disp="off", show_warning=False)

                # Get 1-day ahead volatility forecast
                forecast = res.forecast(horizon=1)
                garch_vol_daily = np.sqrt(
                    forecast.variance.iloc[-1, 0] / 10000
                )  # Convert back from percentage
                garch_vol_annual = garch_vol_daily * np.sqrt(252)

                results["garch_vol"] = garch_vol_annual
                results["garch_vol_daily"] = garch_vol_daily

            except Exception as e:
                self.logger.warning(f"GARCH volatility calculation failed: {e}")

        return results

    def _calculate_drawdown_metrics(self, prices: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Calculate drawdown metrics from price series"""

        if isinstance(prices, pd.Series):
            prices_array = prices.values
        else:
            prices_array = np.array(prices)

        # Calculate running maximum (peak values)
        running_max = np.maximum.accumulate(prices_array)

        # Calculate drawdown
        drawdown = (prices_array - running_max) / running_max

        # Current drawdown
        current_drawdown = abs(drawdown[-1])

        # Maximum drawdown
        max_drawdown = abs(np.min(drawdown))

        # Drawdown duration (days in current drawdown)
        drawdown_duration = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown[i] < -0.001:  # Small threshold to avoid noise
                drawdown_duration += 1
            else:
                break

        return {
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "drawdown_duration": drawdown_duration,
            "drawdown_series": drawdown,
        }

    def _estimate_drawdown_from_returns(self, returns: np.ndarray) -> Dict[str, Any]:
        """Estimate drawdown metrics from returns (when prices not available)"""

        # Simulate cumulative returns to estimate drawdown
        cumulative_returns = np.cumprod(1 + returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)

        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max

        current_drawdown = abs(drawdown[-1])
        max_drawdown = abs(np.min(drawdown))

        # Estimate drawdown duration
        drawdown_duration = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown[i] < -0.001:
                drawdown_duration += 1
            else:
                break

        return {
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "drawdown_duration": drawdown_duration,
        }

    def _calculate_distribution_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate return distribution characteristics"""

        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis

        results = {"skewness": skewness, "kurtosis": kurtosis}

        # Tail ratio (95th percentile / 5th percentile of absolute returns)
        if len(returns) >= 50:
            abs_returns = np.abs(returns)
            p95 = np.percentile(abs_returns, 95)
            p5 = np.percentile(abs_returns, 5)

            if p5 > 0:
                results["tail_ratio"] = p95 / p5

        return results

    def _calculate_risk_adjusted_returns(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics"""

        results = {}

        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Sharpe ratio
        if std_return > 0:
            excess_return = mean_return - (self.risk_free_rate / 252)  # Daily risk-free rate
            sharpe = excess_return / std_return * np.sqrt(252)  # Annualized
            results["sharpe_ratio"] = sharpe

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < self.target_return / 252]  # Daily target
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                sortino = (mean_return - self.target_return / 252) / downside_std * np.sqrt(252)
                results["sortino_ratio"] = sortino

        # Calmar ratio (if we can estimate max drawdown)
        if len(returns) >= 50:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))

            if max_drawdown > 0.001:  # Avoid division by very small numbers
                annual_return = mean_return * 252
                calmar = annual_return / max_drawdown
                results["calmar_ratio"] = calmar

        return results

    def _calculate_portfolio_metrics(
        self, returns_matrix: np.ndarray, weights: np.ndarray
    ) -> Dict[str, float]:
        """Calculate portfolio-specific risk metrics"""

        results = {}

        # Ensure we have a 2D returns matrix for portfolio calculations
        if returns_matrix.ndim == 1:
            # Single asset case - no portfolio metrics to calculate
            return results

        # Calculate covariance matrix
        cov_matrix = np.cov(returns_matrix.T)  # Assets as rows

        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var) * np.sqrt(252)  # Annualized

        # Individual asset volatilities
        asset_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)

        # Weighted average volatility
        weighted_avg_vol = np.dot(weights, asset_vols)

        # Diversification ratio
        if portfolio_vol > 0:
            diversification_ratio = weighted_avg_vol / portfolio_vol
            results["diversification_ratio"] = diversification_ratio

        return results

    def _calculate_beta(
        self, returns: np.ndarray, benchmark_returns: Union[pd.Series, np.ndarray]
    ) -> Optional[float]:
        """Calculate beta relative to benchmark"""

        if isinstance(benchmark_returns, pd.Series):
            benchmark_array = benchmark_returns.values
        else:
            benchmark_array = np.array(benchmark_returns)

        # Align lengths
        min_length = min(len(returns), len(benchmark_array))
        returns_aligned = returns[-min_length:]
        benchmark_aligned = benchmark_array[-min_length:]

        # Remove NaN values
        mask = ~(np.isnan(returns_aligned) | np.isnan(benchmark_aligned))
        returns_clean = returns_aligned[mask]
        benchmark_clean = benchmark_aligned[mask]

        if len(returns_clean) < 30:
            return None

        # Calculate beta using covariance
        covariance = np.cov(returns_clean, benchmark_clean)[0, 1]
        benchmark_var = np.var(benchmark_clean)

        if benchmark_var > 1e-8:  # Avoid division by zero
            beta = covariance / benchmark_var
            return beta

        return None

    def calculate_position_risk(
        self, position_value: float, asset_returns: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate risk metrics for a specific position size"""

        if confidence_level not in self.confidence_levels:
            # Calculate VaR/CVaR for this specific confidence level
            var_percentile = (1 - confidence_level) * 100
            var_value = abs(np.percentile(asset_returns, var_percentile))

            tail_returns = asset_returns[asset_returns <= -var_value]
            cvar_value = abs(np.mean(tail_returns)) if len(tail_returns) > 0 else var_value
        else:
            # Use pre-calculated values
            var_metrics = self._calculate_var_cvar(asset_returns)
            conf_key = str(int(confidence_level * 100))
            var_value = var_metrics[f"var_{conf_key}"]
            cvar_value = var_metrics[f"cvar_{conf_key}"]

        # Position-adjusted risk
        position_var = position_value * var_value
        position_cvar = position_value * cvar_value

        # Daily volatility risk
        daily_vol = np.std(asset_returns)
        position_vol_risk = position_value * daily_vol

        return {
            "position_value": position_value,
            "var_amount": position_var,
            "cvar_amount": position_cvar,
            "daily_vol_risk": position_vol_risk,
            "var_percentage": var_value,
            "cvar_percentage": cvar_value,
            "confidence_level": confidence_level,
        }

    def calculate_portfolio_risk(
        self,
        positions: Dict[str, float],  # {asset: position_value}
        asset_returns: Dict[str, np.ndarray],  # {asset: returns}
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculate risk metrics for entire portfolio"""

        assets = list(positions.keys())
        position_values = np.array([positions[asset] for asset in assets])

        # Create returns matrix
        returns_matrix = np.column_stack([asset_returns[asset] for asset in assets])

        # Calculate correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = np.corrcoef(returns_matrix.T)

        # Portfolio weights (by value)
        total_value = np.sum(position_values)
        weights = (
            position_values / total_value if total_value > 0 else np.zeros_like(position_values)
        )

        # Portfolio returns
        portfolio_returns = np.dot(returns_matrix, weights)

        # Calculate portfolio risk metrics
        portfolio_risk = self.calculate_comprehensive_risk(
            returns=portfolio_returns, portfolio_weights=weights
        )

        # Individual position risks
        position_risks = {}
        for asset in assets:
            pos_risk = self.calculate_position_risk(positions[asset], asset_returns[asset])
            position_risks[asset] = pos_risk

        # Concentration risk
        concentration_risk = self._calculate_concentration_risk(weights)

        return {
            "portfolio_metrics": portfolio_risk,
            "position_risks": position_risks,
            "concentration_risk": concentration_risk,
            "total_portfolio_value": total_value,
            "weights": dict(zip(assets, weights)),
            "correlation_matrix": correlation_matrix.tolist()
            if correlation_matrix is not None
            else None,
        }

    def _calculate_concentration_risk(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio concentration risk metrics"""

        # Herfindahl-Hirschman Index (HHI)
        hhi = np.sum(weights**2)

        # Effective number of assets
        effective_assets = 1 / hhi if hhi > 0 else 0

        # Maximum weight
        max_weight = np.max(weights)

        return {
            "hhi": hhi,
            "effective_assets": effective_assets,
            "max_weight": max_weight,
            "concentration_score": hhi,  # Higher = more concentrated
        }
