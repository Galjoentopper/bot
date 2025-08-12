#!/usr/bin/env python3
"""
Cost-Aware Evaluation Utilities
===============================

This module provides cost-aware metrics and evaluation tools for trading models:
1. Cost models (fees, slippage) for realistic performance estimation
2. Net Sharpe/Sortino ratio computation
3. Optimal threshold selection for classifiers
4. Trading simulation with transaction costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import optimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Trading cost model configuration."""
    fee_bps: float = 10.0  # Trading fees in basis points (0.1%)
    slippage_bps: float = 5.0  # Market slippage in basis points (0.05%)
    min_position_size: float = 10.0  # Minimum position size in USD
    max_position_size: float = 100000.0  # Maximum position size in USD


@dataclass
class TradingMetrics:
    """Container for trading performance metrics."""
    # Returns
    total_return: float
    annual_return: float
    net_return: float
    gross_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    net_sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Trading metrics
    num_trades: int
    win_rate: float
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    
    # Cost breakdown
    total_costs: float
    total_fees: float
    total_slippage: float
    cost_ratio: float  # Costs as % of gross returns


class CostAwareEvaluator:
    """
    Cost-aware evaluation for trading models.
    
    Incorporates realistic trading costs and provides net performance metrics.
    """
    
    def __init__(self, cost_model: Optional[CostModel] = None):
        """
        Initialize evaluator with cost model.
        
        Args:
            cost_model: Trading cost configuration
        """
        self.cost_model = cost_model or CostModel()
        logger.info(f"CostAwareEvaluator initialized: fees={self.cost_model.fee_bps}bps, slippage={self.cost_model.slippage_bps}bps")
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray,
                           y_pred_proba: np.ndarray,
                           returns: np.ndarray,
                           threshold: float = 0.5,
                           position_size: float = 1000.0) -> TradingMetrics:
        """
        Evaluate predictions with trading costs.
        
        Args:
            y_true: True labels (0/1 for binary classification)
            y_pred_proba: Predicted probabilities
            returns: Forward returns for each prediction
            threshold: Decision threshold for classification
            position_size: Position size in USD
            
        Returns:
            TradingMetrics with cost-aware performance
        """
        # Generate trading signals
        signals = (y_pred_proba >= threshold).astype(int)
        
        # Calculate position changes (entries/exits)
        position_changes = np.diff(np.concatenate(([0], signals)))
        trade_mask = position_changes != 0
        
        # Calculate gross returns
        strategy_returns = signals[:-1] * returns[1:]  # Avoid look-ahead bias
        gross_return = np.sum(strategy_returns)
        
        # Calculate trading costs
        num_trades = np.sum(np.abs(position_changes))
        total_fees = self._calculate_fees(num_trades, position_size)
        total_slippage = self._calculate_slippage(num_trades, position_size, returns)
        total_costs = total_fees + total_slippage
        
        # Net returns
        net_return = gross_return - total_costs
        
        # Risk metrics
        volatility = np.std(strategy_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = self._calculate_sharpe(strategy_returns)
        net_sharpe_ratio = self._calculate_sharpe(strategy_returns - total_costs / len(strategy_returns))
        sortino_ratio = self._calculate_sortino(strategy_returns)
        max_drawdown = self._calculate_max_drawdown(np.cumsum(strategy_returns))
        
        # Trading statistics
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        
        win_rate = len(winning_trades) / max(1, num_trades)
        avg_trade_return = np.mean(strategy_returns) if len(strategy_returns) > 0 else 0
        avg_winning_trade = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_losing_trade = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        
        # Cost analysis
        cost_ratio = total_costs / max(abs(gross_return), 1e-8)
        
        return TradingMetrics(
            total_return=net_return,
            annual_return=net_return * 252 / len(strategy_returns) if len(strategy_returns) > 0 else 0,
            net_return=net_return,
            gross_return=gross_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            net_sharpe_ratio=net_sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            num_trades=int(num_trades),
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            total_costs=total_costs,
            total_fees=total_fees,
            total_slippage=total_slippage,
            cost_ratio=cost_ratio
        )
    
    def find_optimal_threshold(self,
                              y_true: np.ndarray,
                              y_pred_proba: np.ndarray, 
                              returns: np.ndarray,
                              position_size: float = 1000.0,
                              metric: str = "net_sharpe") -> Tuple[float, TradingMetrics]:
        """
        Find optimal threshold that maximizes specified metric.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            returns: Forward returns
            position_size: Position size in USD
            metric: Metric to optimize ("net_sharpe", "net_return", "sortino")
            
        Returns:
            Tuple of (optimal_threshold, best_metrics)
        """
        def objective(threshold):
            try:
                metrics = self.evaluate_predictions(
                    y_true, y_pred_proba, returns, threshold[0], position_size
                )
                
                if metric == "net_sharpe":
                    return -metrics.net_sharpe_ratio  # Minimize negative
                elif metric == "net_return":
                    return -metrics.net_return
                elif metric == "sortino":
                    return -metrics.sortino_ratio
                else:
                    return -metrics.net_sharpe_ratio
                    
            except Exception as e:
                logger.warning(f"Optimization failed at threshold {threshold[0]}: {e}")
                return 1e6  # Large penalty for failed evaluation
        
        # Optimize threshold
        result = optimize.minimize_scalar(
            lambda x: objective([x]),
            bounds=(0.1, 0.9),
            method='bounded'
        )
        
        optimal_threshold = result.x
        best_metrics = self.evaluate_predictions(
            y_true, y_pred_proba, returns, optimal_threshold, position_size
        )
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}, {metric}: {getattr(best_metrics, metric):.4f}")
        return optimal_threshold, best_metrics
    
    def evaluate_threshold_range(self,
                                y_true: np.ndarray,
                                y_pred_proba: np.ndarray,
                                returns: np.ndarray,
                                thresholds: np.ndarray = None,
                                position_size: float = 1000.0) -> pd.DataFrame:
        """
        Evaluate performance across a range of thresholds.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            returns: Forward returns
            thresholds: Array of thresholds to evaluate
            position_size: Position size in USD
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        results = []
        
        for threshold in thresholds:
            try:
                metrics = self.evaluate_predictions(
                    y_true, y_pred_proba, returns, threshold, position_size
                )
                
                result = {
                    'threshold': threshold,
                    'net_return': metrics.net_return,
                    'gross_return': metrics.gross_return,
                    'net_sharpe': metrics.net_sharpe_ratio,
                    'sharpe': metrics.sharpe_ratio,
                    'sortino': metrics.sortino_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'num_trades': metrics.num_trades,
                    'win_rate': metrics.win_rate,
                    'total_costs': metrics.total_costs,
                    'cost_ratio': metrics.cost_ratio
                }
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Evaluation failed for threshold {threshold}: {e}")
        
        return pd.DataFrame(results)
    
    def _calculate_fees(self, num_trades: int, position_size: float) -> float:
        """Calculate trading fees."""
        return num_trades * position_size * (self.cost_model.fee_bps / 10000.0)
    
    def _calculate_slippage(self, num_trades: int, position_size: float, returns: np.ndarray) -> float:
        """Calculate market slippage costs."""
        # Simple slippage model - could be made more sophisticated
        avg_volatility = np.std(returns)
        slippage_factor = self.cost_model.slippage_bps / 10000.0
        return num_trades * position_size * slippage_factor * (1 + avg_volatility)
    
    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252
        return excess_returns / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside risk only)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        return excess_returns / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / np.maximum(peak, 1e-8)
        return float(np.max(drawdown))
    
    def generate_performance_report(self, metrics: TradingMetrics) -> str:
        """Generate formatted performance report."""
        report = f"""
Trading Performance Report
=========================

Returns:
  Gross Return:     {metrics.gross_return:8.4f}
  Net Return:       {metrics.net_return:8.4f}
  Annual Return:    {metrics.annual_return:8.2%}

Risk Metrics:
  Volatility:       {metrics.volatility:8.2%}
  Sharpe Ratio:     {metrics.sharpe_ratio:8.4f}
  Net Sharpe:       {metrics.net_sharpe_ratio:8.4f}
  Sortino Ratio:    {metrics.sortino_ratio:8.4f}
  Max Drawdown:     {metrics.max_drawdown:8.2%}

Trading Stats:
  Number of Trades: {metrics.num_trades:8d}
  Win Rate:         {metrics.win_rate:8.2%}
  Avg Trade:        {metrics.avg_trade_return:8.6f}
  Avg Winner:       {metrics.avg_winning_trade:8.6f}
  Avg Loser:        {metrics.avg_losing_trade:8.6f}

Cost Breakdown:
  Total Costs:      {metrics.total_costs:8.4f}
  Total Fees:       {metrics.total_fees:8.4f}
  Total Slippage:   {metrics.total_slippage:8.4f}
  Cost Ratio:       {metrics.cost_ratio:8.2%}
"""
        return report


def compute_portfolio_metrics(returns: pd.Series,
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Compute comprehensive portfolio performance metrics.
    
    Args:
        returns: Portfolio returns time series
        benchmark_returns: Benchmark returns for comparison
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    
    # Risk metrics
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
    
    # Drawdown metrics
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_vol),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_drawdown),
        'win_rate': float((returns > 0).mean()),
        'num_periods': len(returns)
    }
    
    # Benchmark comparison if provided
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        benchmark_total = (1 + aligned_benchmark).prod() - 1
        benchmark_annual = (1 + benchmark_total) ** (252 / len(aligned_benchmark)) - 1
        
        metrics.update({
            'benchmark_return': float(benchmark_total),
            'benchmark_annual': float(benchmark_annual),
            'excess_return': float(total_return - benchmark_total),
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'beta': float(np.cov(aligned_returns, aligned_benchmark)[0, 1] / np.var(aligned_benchmark))
        })
    
    return metrics