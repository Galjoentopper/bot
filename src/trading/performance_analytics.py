"""
Enhanced Performance Analytics Module
===================================

This module provides comprehensive trading performance analysis including:
- Real-time P&L tracking
- Risk metrics calculation
- Sharpe ratio optimization
- Portfolio analysis
- Trade performance evaluation
- Market impact analysis
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    portfolio_value: float
    cash_balance: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float


@dataclass
class TradePerformance:
    """Individual trade performance analysis."""
    symbol: str
    entry_time: str
    exit_time: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float
    pnl_pct: float
    holding_period_hours: Optional[float]
    fees_paid: float
    trade_reason: str
    market_conditions: Dict[str, Any]


class PerformanceAnalyzer:
    """Advanced performance analytics and reporting system."""
    
    def __init__(self, config: Dict[str, Any], initial_capital: float = 10000.0):
        self.config = config
        self.initial_capital = initial_capital
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_snapshots: List[Dict[str, Any]] = []
        self.performance_history: List[PerformanceMetrics] = []
        
        # Configuration
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        self.benchmark_return = config.get('benchmark_return', 0.08)  # 8% annual
        self.report_frequency = config.get('report_frequency', 'daily')
        self.export_path = Path(config.get('export_path', 'reports'))
        
        # Create reports directory
        self.export_path.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.min_sharpe_ratio = config.get('min_sharpe_ratio', 1.0)
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.15)
        self.min_win_rate = config.get('min_win_rate', 0.45)
        
        logger.info(f"PerformanceAnalyzer initialized with capital: €{initial_capital:,.2f}")
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record a trade for performance analysis."""
        try:
            trade_data['recorded_at'] = datetime.now().isoformat()
            self.trade_history.append(trade_data)
            
            logger.debug(f"Trade recorded: {trade_data['symbol']} {trade_data['action']} "
                        f"{trade_data.get('quantity', 0):.6f} @ {trade_data.get('price', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    def record_portfolio_snapshot(self, portfolio_data: Dict[str, Any]) -> None:
        """Record a portfolio snapshot for performance tracking."""
        try:
            portfolio_data['timestamp'] = datetime.now().isoformat()
            self.portfolio_snapshots.append(portfolio_data)
            
            # Keep only last 10000 snapshots to manage memory
            if len(self.portfolio_snapshots) > 10000:
                self.portfolio_snapshots = self.portfolio_snapshots[-10000:]
            
        except Exception as e:
            logger.error(f"Failed to record portfolio snapshot: {e}")
    
    def calculate_comprehensive_metrics(self, current_positions: Dict[str, float],
                                      current_prices: Dict[str, float],
                                      current_balance: float) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            # Calculate current portfolio value
            position_value = sum(
                pos * current_prices.get(symbol, 0.0)
                for symbol, pos in current_positions.items()
            )
            current_portfolio_value = current_balance + position_value
            
            # Basic return metrics
            total_return = (current_portfolio_value / self.initial_capital) - 1
            
            # Calculate returns series for advanced metrics
            returns_series = self._calculate_returns_series()
            
            # Calculate metrics
            metrics = self._calculate_advanced_metrics(
                returns_series, current_portfolio_value, current_balance,
                position_value, current_positions, current_prices
            )
            
            # Store metrics
            self.performance_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate comprehensive metrics: {e}")
            return self._create_default_metrics(current_balance)
    
    def _calculate_returns_series(self) -> pd.Series:
        """Calculate portfolio returns series from snapshots."""
        try:
            if len(self.portfolio_snapshots) < 2:
                return pd.Series([0.0])
            
            # Create DataFrame from snapshots
            df = pd.DataFrame(self.portfolio_snapshots)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate returns
            portfolio_values = df['portfolio_value']
            returns = portfolio_values.pct_change().dropna()
            
            return returns
            
        except Exception as e:
            logger.error(f"Failed to calculate returns series: {e}")
            return pd.Series([0.0])
    
    def _calculate_advanced_metrics(self, returns: pd.Series, portfolio_value: float,
                                  cash_balance: float, position_value: float,
                                  positions: Dict[str, float],
                                  prices: Dict[str, float]) -> PerformanceMetrics:
        """Calculate advanced performance metrics."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Basic metrics
            total_return = (portfolio_value / self.initial_capital) - 1
            
            # Return and risk metrics
            if len(returns) > 1:
                mean_return = returns.mean()
                volatility = returns.std()
                
                # Annualize (assuming 30-minute intervals, 48 periods per day, 365 days)
                annualization_factor = np.sqrt(48 * 365)
                annualized_return = mean_return * 48 * 365
                annualized_volatility = volatility * annualization_factor
                
                # Sharpe ratio
                excess_return = annualized_return - self.risk_free_rate
                sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0.0
                
                # Sortino ratio (downside deviation)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std() * annualization_factor
                    sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0
                else:
                    sortino_ratio = float('inf') if excess_return > 0 else 0.0
                
                # Drawdown analysis
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = abs(drawdowns.min())
                current_drawdown = abs(drawdowns.iloc[-1])
                
            else:
                annualized_return = 0.0
                annualized_volatility = 0.0
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                max_drawdown = 0.0
                current_drawdown = 0.0
            
            # Trade analysis
            trade_metrics = self._analyze_trades()
            
            # P&L calculation
            realized_pnl = sum(
                trade.get('realized_pnl', 0.0) for trade in self.trade_history
                if trade.get('action') == 'SELL'
            )
            
            unrealized_pnl = position_value - sum(
                trade.get('cost', 0.0) for trade in self.trade_history
                if trade.get('action') == 'BUY'
            ) + sum(
                trade.get('proceeds', 0.0) for trade in self.trade_history
                if trade.get('action') == 'SELL'
            )
            
            total_pnl = realized_pnl + unrealized_pnl
            
            return PerformanceMetrics(
                timestamp=timestamp,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=annualized_volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                win_rate=trade_metrics['win_rate'],
                profit_factor=trade_metrics['profit_factor'],
                average_win=trade_metrics['average_win'],
                average_loss=trade_metrics['average_loss'],
                total_trades=trade_metrics['total_trades'],
                winning_trades=trade_metrics['winning_trades'],
                losing_trades=trade_metrics['losing_trades'],
                portfolio_value=portfolio_value,
                cash_balance=cash_balance,
                total_pnl=total_pnl,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate advanced metrics: {e}")
            return self._create_default_metrics(cash_balance)
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade performance."""
        try:
            if not self.trade_history:
                return {
                    'win_rate': 0.0, 'profit_factor': 0.0, 'average_win': 0.0,
                    'average_loss': 0.0, 'total_trades': 0, 'winning_trades': 0,
                    'losing_trades': 0
                }
            
            # Group trades by symbol to calculate P&L
            symbol_trades = {}
            for trade in self.trade_history:
                symbol = trade['symbol']
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = {'buys': [], 'sells': []}
                
                if trade['action'] == 'BUY':
                    symbol_trades[symbol]['buys'].append(trade)
                elif trade['action'] == 'SELL':
                    symbol_trades[symbol]['sells'].append(trade)
            
            # Calculate completed trade P&L
            completed_trades = []
            for symbol, trades in symbol_trades.items():
                buys = sorted(trades['buys'], key=lambda x: x.get('timestamp', 0))
                sells = sorted(trades['sells'], key=lambda x: x.get('timestamp', 0))
                
                # Simple FIFO matching
                buy_idx = 0
                for sell in sells:
                    if buy_idx < len(buys):
                        buy = buys[buy_idx]
                        sell_qty = sell.get('quantity', 0.0)
                        buy_qty = buy.get('quantity', 0.0)
                        
                        if sell_qty > 0 and buy_qty > 0:
                            trade_qty = min(sell_qty, buy_qty)
                            pnl = (sell.get('price', 0.0) - buy.get('price', 0.0)) * trade_qty
                            pnl -= sell.get('fee', 0.0) + buy.get('fee', 0.0)
                            
                            completed_trades.append({
                                'symbol': symbol,
                                'pnl': pnl,
                                'pnl_pct': pnl / (buy.get('price', 1.0) * trade_qty) if buy.get('price', 0.0) > 0 else 0.0,
                                'quantity': trade_qty,
                                'entry_price': buy.get('price', 0.0),
                                'exit_price': sell.get('price', 0.0)
                            })
                            
                            buy_idx += 1
            
            if not completed_trades:
                return {
                    'win_rate': 0.0, 'profit_factor': 0.0, 'average_win': 0.0,
                    'average_loss': 0.0, 'total_trades': 0, 'winning_trades': 0,
                    'losing_trades': 0
                }
            
            # Calculate trade statistics
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] < 0]
            
            total_trades = len(completed_trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0.0
            
            total_wins = sum(t['pnl'] for t in winning_trades)
            total_losses = abs(sum(t['pnl'] for t in losing_trades))
            
            average_win = total_wins / win_count if win_count > 0 else 0.0
            average_loss = total_losses / loss_count if loss_count > 0 else 0.0
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_win': average_win,
                'average_loss': average_loss,
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze trades: {e}")
            return {
                'win_rate': 0.0, 'profit_factor': 0.0, 'average_win': 0.0,
                'average_loss': 0.0, 'total_trades': 0, 'winning_trades': 0,
                'losing_trades': 0
            }
    
    def _create_default_metrics(self, cash_balance: float) -> PerformanceMetrics:
        """Create default metrics when calculation fails."""
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            portfolio_value=cash_balance,
            cash_balance=cash_balance,
            total_pnl=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0
        )
    
    def generate_performance_report(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            # Performance summary
            performance_summary = {
                'current_metrics': asdict(metrics),
                'performance_grade': self._calculate_performance_grade(metrics),
                'risk_assessment': self._assess_risk_levels(metrics),
                'improvement_suggestions': self._generate_improvement_suggestions(metrics),
                'benchmark_comparison': self._compare_to_benchmark(metrics)
            }
            
            # Symbol-level analysis
            symbol_analysis = self._analyze_symbol_performance()
            
            # Time-based analysis
            time_analysis = self._analyze_time_performance()
            
            # Risk analysis
            risk_analysis = self._analyze_risk_metrics(metrics)
            
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'report_period': self._get_report_period(),
                'performance_summary': performance_summary,
                'symbol_analysis': symbol_analysis,
                'time_analysis': time_analysis,
                'risk_analysis': risk_analysis,
                'recommendations': self._generate_recommendations(metrics)
            }
            
            # Export report
            self._export_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance grade (A-F)."""
        try:
            score = 0
            
            # Return score (30%)
            if metrics.annualized_return > 0.2:  # >20%
                score += 30
            elif metrics.annualized_return > 0.1:  # >10%
                score += 25
            elif metrics.annualized_return > 0.05:  # >5%
                score += 20
            elif metrics.annualized_return > 0:  # >0%
                score += 15
            
            # Sharpe ratio score (25%)
            if metrics.sharpe_ratio > 2.0:
                score += 25
            elif metrics.sharpe_ratio > 1.5:
                score += 20
            elif metrics.sharpe_ratio > 1.0:
                score += 15
            elif metrics.sharpe_ratio > 0.5:
                score += 10
            
            # Drawdown score (25%)
            if metrics.max_drawdown < 0.05:  # <5%
                score += 25
            elif metrics.max_drawdown < 0.1:  # <10%
                score += 20
            elif metrics.max_drawdown < 0.15:  # <15%
                score += 15
            elif metrics.max_drawdown < 0.2:  # <20%
                score += 10
            
            # Win rate score (20%)
            if metrics.win_rate > 0.6:  # >60%
                score += 20
            elif metrics.win_rate > 0.5:  # >50%
                score += 15
            elif metrics.win_rate > 0.4:  # >40%
                score += 10
            elif metrics.win_rate > 0.3:  # >30%
                score += 5
            
            # Convert to grade
            if score >= 85:
                return 'A'
            elif score >= 75:
                return 'B'
            elif score >= 65:
                return 'C'
            elif score >= 55:
                return 'D'
            else:
                return 'F'
                
        except Exception as e:
            logger.error(f"Failed to calculate performance grade: {e}")
            return 'F'
    
    def _assess_risk_levels(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """Assess risk levels across different metrics."""
        try:
            risk_assessment = {}
            
            # Volatility risk
            if metrics.volatility > 0.4:
                risk_assessment['volatility'] = 'HIGH'
            elif metrics.volatility > 0.2:
                risk_assessment['volatility'] = 'MEDIUM'
            else:
                risk_assessment['volatility'] = 'LOW'
            
            # Drawdown risk
            if metrics.max_drawdown > 0.2:
                risk_assessment['drawdown'] = 'HIGH'
            elif metrics.max_drawdown > 0.1:
                risk_assessment['drawdown'] = 'MEDIUM'
            else:
                risk_assessment['drawdown'] = 'LOW'
            
            # Concentration risk
            if len(self.portfolio_snapshots) > 0:
                latest_snapshot = self.portfolio_snapshots[-1]
                positions = latest_snapshot.get('positions', {})
                if positions:
                    max_position_pct = max(pos.get('pct_portfolio', 0) for pos in positions.values())
                    if max_position_pct > 0.4:
                        risk_assessment['concentration'] = 'HIGH'
                    elif max_position_pct > 0.25:
                        risk_assessment['concentration'] = 'MEDIUM'
                    else:
                        risk_assessment['concentration'] = 'LOW'
                else:
                    risk_assessment['concentration'] = 'LOW'
            else:
                risk_assessment['concentration'] = 'UNKNOWN'
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Failed to assess risk levels: {e}")
            return {'volatility': 'UNKNOWN', 'drawdown': 'UNKNOWN', 'concentration': 'UNKNOWN'}
    
    def _generate_improvement_suggestions(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate specific improvement suggestions based on performance."""
        suggestions = []
        
        try:
            # Return improvement
            if metrics.annualized_return < 0.05:
                suggestions.append("Consider increasing position sizes or improving signal quality to boost returns")
            
            # Risk management
            if metrics.max_drawdown > 0.15:
                suggestions.append("Implement tighter stop-loss levels to reduce maximum drawdown")
            
            if metrics.sharpe_ratio < 1.0:
                suggestions.append("Focus on risk-adjusted returns by improving signal-to-noise ratio")
            
            # Trade efficiency
            if metrics.win_rate < 0.45:
                suggestions.append("Review entry criteria to improve win rate")
            
            if metrics.profit_factor < 1.5:
                suggestions.append("Optimize profit-taking and loss-cutting strategies")
            
            # Portfolio management
            if len(self.portfolio_snapshots) > 0:
                latest_snapshot = self.portfolio_snapshots[-1]
                if latest_snapshot.get('cash_pct', 0) > 0.3:
                    suggestions.append("Consider deploying excess cash to increase market exposure")
                elif latest_snapshot.get('cash_pct', 0) < 0.05:
                    suggestions.append("Maintain higher cash reserves for opportunities and risk management")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate improvement suggestions: {e}")
            return ["Review overall trading strategy and risk management"]
    
    def _compare_to_benchmark(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Compare performance to benchmark."""
        try:
            benchmark_comparison = {
                'benchmark_return': self.benchmark_return,
                'excess_return': metrics.annualized_return - self.benchmark_return,
                'information_ratio': (metrics.annualized_return - self.benchmark_return) / metrics.volatility if metrics.volatility > 0 else 0.0,
                'outperformance': metrics.annualized_return > self.benchmark_return
            }
            
            return benchmark_comparison
            
        except Exception as e:
            logger.error(f"Failed to compare to benchmark: {e}")
            return {'error': str(e)}
    
    def _analyze_symbol_performance(self) -> Dict[str, Any]:
        """Analyze performance by symbol."""
        try:
            symbol_stats = {}
            
            for trade in self.trade_history:
                symbol = trade['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'trades': 0, 'total_pnl': 0.0, 'total_volume': 0.0,
                        'wins': 0, 'losses': 0
                    }
                
                stats = symbol_stats[symbol]
                stats['trades'] += 1
                stats['total_volume'] += trade.get('quantity', 0.0) * trade.get('price', 0.0)
                
                if 'pnl' in trade:
                    stats['total_pnl'] += trade['pnl']
                    if trade['pnl'] > 0:
                        stats['wins'] += 1
                    else:
                        stats['losses'] += 1
            
            # Calculate derived metrics
            for symbol, stats in symbol_stats.items():
                total_trades = stats['wins'] + stats['losses']
                stats['win_rate'] = stats['wins'] / total_trades if total_trades > 0 else 0.0
                stats['avg_pnl_per_trade'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0.0
            
            return symbol_stats
            
        except Exception as e:
            logger.error(f"Failed to analyze symbol performance: {e}")
            return {}
    
    def _analyze_time_performance(self) -> Dict[str, Any]:
        """Analyze performance by time periods."""
        try:
            if not self.performance_history:
                return {}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([asdict(m) for m in self.performance_history])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Daily performance
            daily_returns = df['total_return'].resample('D').last().pct_change().dropna()
            
            # Weekly performance
            weekly_returns = df['total_return'].resample('W').last().pct_change().dropna()
            
            time_analysis = {
                'daily_stats': {
                    'mean_return': daily_returns.mean(),
                    'volatility': daily_returns.std(),
                    'best_day': daily_returns.max(),
                    'worst_day': daily_returns.min()
                },
                'weekly_stats': {
                    'mean_return': weekly_returns.mean(),
                    'volatility': weekly_returns.std(),
                    'best_week': weekly_returns.max(),
                    'worst_week': weekly_returns.min()
                }
            }
            
            return time_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze time performance: {e}")
            return {}
    
    def _analyze_risk_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze detailed risk metrics."""
        try:
            # Value at Risk (simple historical method)
            returns_series = self._calculate_returns_series()
            
            if len(returns_series) > 20:
                var_95 = np.percentile(returns_series, 5)  # 5th percentile
                var_99 = np.percentile(returns_series, 1)  # 1st percentile
                
                # Expected Shortfall (Conditional VaR)
                es_95 = returns_series[returns_series <= var_95].mean()
                es_99 = returns_series[returns_series <= var_99].mean()
            else:
                var_95 = var_99 = es_95 = es_99 = 0.0
            
            risk_analysis = {
                'value_at_risk_95': var_95,
                'value_at_risk_99': var_99,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99,
                'volatility_percentile': self._calculate_volatility_percentile(metrics.volatility),
                'risk_adjusted_return': metrics.annualized_return / metrics.volatility if metrics.volatility > 0 else 0.0
            }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze risk metrics: {e}")
            return {}
    
    def _calculate_volatility_percentile(self, current_vol: float) -> float:
        """Calculate what percentile the current volatility represents."""
        try:
            if len(self.performance_history) < 10:
                return 50.0  # Default to median
            
            historical_vols = [m.volatility for m in self.performance_history]
            percentile = (sum(1 for v in historical_vols if v <= current_vol) / len(historical_vols)) * 100
            
            return percentile
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility percentile: {e}")
            return 50.0
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[Dict[str, str]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        try:
            # Performance recommendations
            if metrics.sharpe_ratio < self.min_sharpe_ratio:
                recommendations.append({
                    'category': 'Performance',
                    'priority': 'HIGH',
                    'recommendation': f"Improve Sharpe ratio from {metrics.sharpe_ratio:.2f} to >{self.min_sharpe_ratio}",
                    'action': 'Review signal quality and position sizing'
                })
            
            if metrics.max_drawdown > self.max_drawdown_threshold:
                recommendations.append({
                    'category': 'Risk',
                    'priority': 'HIGH',
                    'recommendation': f"Reduce maximum drawdown from {metrics.max_drawdown:.1%} to <{self.max_drawdown_threshold:.1%}",
                    'action': 'Implement tighter stop-losses and position limits'
                })
            
            if metrics.win_rate < self.min_win_rate:
                recommendations.append({
                    'category': 'Strategy',
                    'priority': 'MEDIUM',
                    'recommendation': f"Improve win rate from {metrics.win_rate:.1%} to >{self.min_win_rate:.1%}",
                    'action': 'Review entry criteria and market timing'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    def _get_report_period(self) -> str:
        """Get the reporting period."""
        if self.portfolio_snapshots:
            start_time = pd.to_datetime(self.portfolio_snapshots[0]['timestamp'])
            end_time = pd.to_datetime(self.portfolio_snapshots[-1]['timestamp'])
            return f"{start_time.date()} to {end_time.date()}"
        return "No data available"
    
    def _export_report(self, report: Dict[str, Any]) -> None:
        """Export performance report to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_report_{timestamp}.json"
            filepath = self.export_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")


def create_performance_analyzer(config: Dict[str, Any], initial_capital: float = 10000.0) -> PerformanceAnalyzer:
    """Factory function to create a performance analyzer."""
    return PerformanceAnalyzer(config, initial_capital)