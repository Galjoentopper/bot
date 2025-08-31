"""
Profit Optimization Module for Enhanced Trading Bot
==================================================

This module implements advanced profit realization strategies, risk management,
and portfolio optimization to maximize trading profitability.

Key Features:
- Dynamic profit-taking with volatility adjustment
- Trailing stop-loss mechanisms
- Portfolio rebalancing and diversification
- Cost-basis tracking and P&L calculation
- Market regime detection for strategy adaptation
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Enhanced position tracking with cost basis."""
    symbol: str
    quantity: float
    avg_cost: float
    total_cost: float
    entry_time: float
    last_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trailing_stop: Optional[float] = None
    profit_target: Optional[float] = None


@dataclass
class TradeSignal:
    """Enhanced trade signal with reasoning."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    quantity_pct: float  # Percentage of position/balance to trade
    reasoning: str
    risk_score: float
    expected_return: float


class ProfitOptimizer:
    """Advanced profit optimization and risk management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Configuration parameters
        self.min_profit_pct = config.get('min_profit_pct', 0.02)  # 2% minimum profit
        self.max_position_pct = config.get('max_position_pct', 0.25)  # 25% max position size
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.05)  # 5% trailing stop
        self.profit_target_pct = config.get('profit_target_pct', 0.15)  # 15% profit target
        self.max_holding_days = config.get('max_holding_days', 7)  # Maximum holding period
        self.volatility_adjustment = config.get('volatility_adjustment', True)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_portfolio_value = 0.0
        
        logger.info("ProfitOptimizer initialized with enhanced profit strategies")
    
    def calculate_dynamic_thresholds(self, symbol: str, market_data: pd.DataFrame, 
                                   base_threshold: float) -> Dict[str, float]:
        """Calculate dynamic trading thresholds based on market volatility and momentum."""
        try:
            if market_data.empty or len(market_data) < 20:
                return {'buy': base_threshold, 'sell': -base_threshold}
            
            # Calculate volatility metrics
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # 24 periods per day for 30m data
            
            # Calculate momentum
            price_change = (market_data['close'].iloc[-1] / market_data['close'].iloc[-10] - 1)
            
            # Calculate volume trend
            volume_trend = market_data['volume'].rolling(5).mean().iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1]
            
            # Adjust thresholds based on market conditions
            volatility_multiplier = max(0.5, min(2.0, volatility / 0.02))  # Adjust for volatility
            momentum_adjustment = np.clip(price_change * 0.5, -0.3, 0.3)  # Momentum bias
            volume_adjustment = np.clip((volume_trend - 1) * 0.2, -0.2, 0.2)  # Volume confirmation
            
            # Calculate adjusted thresholds
            buy_threshold = base_threshold * volatility_multiplier + momentum_adjustment + volume_adjustment
            sell_threshold = -base_threshold * volatility_multiplier - momentum_adjustment - volume_adjustment
            
            logger.debug(f"{symbol} dynamic thresholds: buy={buy_threshold:.6f}, sell={sell_threshold:.6f} "
                        f"(vol_mult={volatility_multiplier:.2f}, momentum={momentum_adjustment:.4f})")
            
            return {
                'buy': max(0.0001, buy_threshold),  # Minimum threshold
                'sell': min(-0.0001, sell_threshold),  # Maximum negative threshold
                'volatility': volatility,
                'momentum': price_change,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate dynamic thresholds for {symbol}: {e}")
            return {'buy': base_threshold, 'sell': -base_threshold}
    
    def calculate_optimal_position_size(self, symbol: str, prediction: float, confidence: float,
                                      current_balance: float, current_positions: Dict[str, float],
                                      market_data: pd.DataFrame) -> float:
        """Calculate optimal position size using Kelly Criterion and risk management."""
        try:
            # Base position size from Kelly Criterion approximation
            win_rate = min(0.9, max(0.1, confidence))  # Clamp between 10% and 90%
            avg_win = self.profit_target_pct
            avg_loss = self.trailing_stop_pct
            
            # Kelly fraction
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(0.5, kelly_fraction))  # Cap at 50%
            
            # Calculate portfolio heat (total risk exposure)
            total_position_value = sum(
                pos_size * current_positions.get(s, 0) 
                for s, pos_size in current_positions.items()
            )
            total_portfolio_value = current_balance + total_position_value
            current_exposure = total_position_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Adjust for current exposure
            max_new_position = min(self.max_position_pct, 1.0 - current_exposure)
            
            # Calculate volatility-adjusted size
            if len(market_data) > 20:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(24)  # 30-minute intervals
                vol_adjustment = max(0.5, min(1.5, 0.02 / volatility))  # Adjust for volatility
            else:
                vol_adjustment = 1.0
            
            # Final position size
            optimal_size = kelly_fraction * max_new_position * vol_adjustment
            optimal_size = max(0, min(self.max_position_pct, optimal_size))
            
            logger.debug(f"{symbol} optimal position size: {optimal_size:.3f} "
                        f"(kelly={kelly_fraction:.3f}, exposure={current_exposure:.3f}, vol_adj={vol_adjustment:.3f})")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal position size for {symbol}: {e}")
            return min(0.1, self.max_position_pct)  # Conservative fallback
    
    def update_trailing_stops(self, current_prices: Dict[str, float]) -> Dict[str, TradeSignal]:
        """Update trailing stops and generate sell signals when triggered."""
        signals = {}
        
        for symbol, position in self.positions.items():
            if position.quantity <= 0:
                continue
                
            current_price = current_prices.get(symbol, 0.0)
            if current_price <= 0:
                continue
            
            # Update unrealized P&L
            position.last_price = current_price
            position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
            
            # Initialize trailing stop if not set
            if position.trailing_stop is None:
                position.trailing_stop = current_price * (1 - self.trailing_stop_pct)
            
            # Update trailing stop (only move up for long positions)
            new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
            if new_trailing_stop > position.trailing_stop:
                position.trailing_stop = new_trailing_stop
                logger.debug(f"{symbol} trailing stop updated to {position.trailing_stop:.4f}")
            
            # Check for stop loss trigger
            if position.trailing_stop is not None and current_price <= position.trailing_stop:
                unrealized_pnl_pct = position.unrealized_pnl / position.total_cost
                signals[symbol] = TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    confidence=0.9,
                    quantity_pct=1.0,  # Sell entire position
                    reasoning=f"Trailing stop triggered at {position.trailing_stop:.4f} (P&L: {unrealized_pnl_pct:.2%})",
                    risk_score=0.1,
                    expected_return=unrealized_pnl_pct
                )
                logger.info(f"{symbol} trailing stop triggered: price={current_price:.4f}, stop={position.trailing_stop:.4f}")
            
            # Check for profit target
            elif position.profit_target and current_price >= position.profit_target:
                unrealized_pnl_pct = position.unrealized_pnl / position.total_cost
                signals[symbol] = TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    confidence=0.8,
                    quantity_pct=0.5,  # Sell half position
                    reasoning=f"Profit target reached at {position.profit_target:.4f} (P&L: {unrealized_pnl_pct:.2%})",
                    risk_score=0.2,
                    expected_return=unrealized_pnl_pct
                )
                logger.info(f"{symbol} profit target reached: price={current_price:.4f}, target={position.profit_target:.4f}")
        
        return signals
    
    def check_time_based_exits(self) -> Dict[str, TradeSignal]:
        """Check for time-based exit signals (stale positions)."""
        signals = {}
        current_time = time.time()
        max_holding_seconds = self.max_holding_days * 24 * 3600
        
        for symbol, position in self.positions.items():
            if position.quantity <= 0:
                continue
            
            holding_time = current_time - position.entry_time
            if holding_time > max_holding_seconds:
                unrealized_pnl_pct = position.unrealized_pnl / position.total_cost
                
                # Only exit if not losing too much
                if unrealized_pnl_pct > -0.1:  # Don't crystallize losses > 10%
                    signals[symbol] = TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        confidence=0.6,
                        quantity_pct=0.7,  # Sell most but not all
                        reasoning=f"Position held too long ({holding_time/86400:.1f} days, P&L: {unrealized_pnl_pct:.2%})",
                        risk_score=0.3,
                        expected_return=unrealized_pnl_pct
                    )
                    logger.info(f"{symbol} time-based exit: held {holding_time/86400:.1f} days")
        
        return signals
    
    def analyze_correlation_risk(self, symbols: List[str], market_data: Dict[str, pd.DataFrame],
                               current_positions: Dict[str, float]) -> Dict[str, float]:
        """Analyze correlation between holdings to manage concentration risk."""
        try:
            if len(symbols) < 2:
                return {}
            
            # Calculate returns for correlation analysis
            returns_data = {}
            for symbol in symbols:
                if symbol in market_data and len(market_data[symbol]) > 20:
                    returns = market_data[symbol]['close'].pct_change().dropna()
                    if len(returns) > 10:
                        returns_data[symbol] = returns.tail(50)  # Last 50 periods
            
            if len(returns_data) < 2:
                return {}
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data).fillna(0)
            correlation_matrix = returns_df.corr()
            
            # Calculate position-weighted correlation risk
            correlation_risks = {}
            for symbol in symbols:
                if symbol not in current_positions or current_positions[symbol] <= 0:
                    correlation_risks[symbol] = 0.0
                    continue
                
                # Calculate weighted correlation with other positions
                weighted_correlation = 0.0
                total_other_weight = 0.0
                
                for other_symbol in symbols:
                    if other_symbol != symbol and other_symbol in current_positions:
                        other_weight = current_positions[other_symbol]
                        if other_weight > 0 and symbol in correlation_matrix.columns and other_symbol in correlation_matrix.columns:
                            correlation = abs(correlation_matrix.loc[symbol, other_symbol])
                            weighted_correlation += correlation * other_weight
                            total_other_weight += other_weight
                
                if total_other_weight > 0:
                    avg_correlation = weighted_correlation / total_other_weight
                    correlation_risks[symbol] = avg_correlation
                else:
                    correlation_risks[symbol] = 0.0
            
            logger.debug(f"Correlation risks calculated: {correlation_risks}")
            return correlation_risks
            
        except Exception as e:
            logger.error(f"Failed to analyze correlation risk: {e}")
            return {}
    
    def generate_rebalancing_signals(self, current_positions: Dict[str, float],
                                   current_prices: Dict[str, float],
                                   current_balance: float) -> Dict[str, TradeSignal]:
        """Generate portfolio rebalancing signals."""
        signals = {}
        
        try:
            # Calculate current portfolio composition
            position_values = {}
            total_position_value = 0.0
            
            for symbol, quantity in current_positions.items():
                if quantity > 0 and symbol in current_prices:
                    value = quantity * current_prices[symbol]
                    position_values[symbol] = value
                    total_position_value += value
            
            total_portfolio_value = current_balance + total_position_value
            
            if total_portfolio_value == 0:
                return signals
            
            # Check for over-concentration
            for symbol, value in position_values.items():
                position_pct = value / total_portfolio_value
                
                # Reduce position if over-concentrated
                if position_pct > self.max_position_pct:
                    excess_pct = position_pct - self.max_position_pct
                    sell_pct = min(0.5, excess_pct / position_pct)  # Sell up to 50%
                    
                    signals[symbol] = TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        confidence=0.8,
                        quantity_pct=sell_pct,
                        reasoning=f"Rebalancing: over-concentrated at {position_pct:.1%} (target: {self.max_position_pct:.1%})",
                        risk_score=0.2,
                        expected_return=0.0
                    )
                    logger.info(f"{symbol} rebalancing sell: {position_pct:.1%} concentration, selling {sell_pct:.1%}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to generate rebalancing signals: {e}")
            return signals
    
    def update_performance_metrics(self, current_positions: Dict[str, float],
                                 current_prices: Dict[str, float],
                                 current_balance: float) -> Dict[str, Any]:
        """Update and return comprehensive performance metrics."""
        try:
            # Calculate current portfolio value
            current_portfolio_value = current_balance
            for symbol, quantity in current_positions.items():
                if quantity > 0 and symbol in current_prices:
                    current_portfolio_value += quantity * current_prices[symbol]
            
            # Update peak and drawdown
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Calculate total unrealized P&L
            self.total_unrealized_pnl = 0.0
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position.last_price = current_prices[symbol]
                    position.unrealized_pnl = (position.last_price - position.avg_cost) * position.quantity
                    self.total_unrealized_pnl += position.unrealized_pnl
            
            # Performance metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': current_portfolio_value,
                'balance': current_balance,
                'total_realized_pnl': self.total_realized_pnl,
                'total_unrealized_pnl': self.total_unrealized_pnl,
                'total_pnl': self.total_realized_pnl + self.total_unrealized_pnl,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'peak_value': self.peak_portfolio_value,
                'num_positions': len([p for p in self.positions.values() if p.quantity > 0]),
                'positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'avg_cost': pos.avg_cost,
                        'current_price': pos.last_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'unrealized_pnl_pct': pos.unrealized_pnl / pos.total_cost if pos.total_cost > 0 else 0,
                        'trailing_stop': pos.trailing_stop
                    }
                    for symbol, pos in self.positions.items() if pos.quantity > 0
                }
            }
            
            # Store for history
            self.portfolio_history.append(metrics)
            
            # Keep only last 1000 entries
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
            return {}
    
    def record_trade(self, symbol: str, action: str, quantity: float, price: float,
                    fee: float, reasoning: str) -> None:
        """Record a trade and update position tracking."""
        try:
            trade_time = time.time()
            
            if action.upper() == 'BUY':
                # Update or create position
                if symbol not in self.positions:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_cost=price,
                        total_cost=quantity * price + fee,
                        entry_time=trade_time
                    )
                else:
                    pos = self.positions[symbol]
                    new_total_cost = pos.total_cost + (quantity * price + fee)
                    new_quantity = pos.quantity + quantity
                    pos.avg_cost = new_total_cost / new_quantity
                    pos.quantity = new_quantity
                    pos.total_cost = new_total_cost
                
                # Set profit target
                self.positions[symbol].profit_target = price * (1 + self.profit_target_pct)
                
            elif action.upper() == 'SELL' and symbol in self.positions:
                pos = self.positions[symbol]
                if pos.quantity > 0:
                    # Calculate realized P&L
                    realized_pnl = (price - pos.avg_cost) * quantity - fee
                    self.total_realized_pnl += realized_pnl
                    
                    # Update position
                    pos.quantity -= quantity
                    pos.realized_pnl += realized_pnl
                    
                    # Remove position if fully sold
                    if pos.quantity <= 0.001:  # Small threshold for float precision
                        del self.positions[symbol]
            
            # Record trade
            trade_record = {
                'timestamp': trade_time,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'fee': fee,
                'reasoning': reasoning,
                'total_realized_pnl': self.total_realized_pnl
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Trade recorded: {action} {quantity:.6f} {symbol} @ {price:.4f} ({reasoning})")
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")