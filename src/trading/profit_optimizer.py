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

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
        self.min_profit_pct = config.get("min_profit_pct", 0.02)  # 2% minimum profit
        self.max_position_pct = config.get("max_position_pct", 0.25)  # 25% max position size
        self.trailing_stop_pct = config.get("trailing_stop_pct", 0.05)  # 5% trailing stop
        self.profit_target_pct = config.get("profit_target_pct", 0.15)  # 15% profit target
        self.max_holding_days = config.get("max_holding_days", 7)  # Maximum holding period
        self.volatility_adjustment = config.get("volatility_adjustment", True)
        self.correlation_threshold = config.get("correlation_threshold", 0.7)

        # Partial profit-taking parameters
        self.profit_scaling_levels = config.get("profit_scaling_levels", [0.02, 0.04, 0.06])
        self.profit_scaling_amounts = config.get("profit_scaling_amounts", [0.3, 0.4, 0.5])

        # Enhanced risk management parameters
        self.position_sizing_method = config.get("position_sizing_method", "fixed")
        self.max_correlation_exposure = config.get("max_correlation_exposure", 0.4)
        self.volatility_scaling = config.get("volatility_scaling", True)
        self.max_daily_trades = config.get("max_daily_trades", 10)
        self.min_time_between_trades = config.get("min_time_between_trades", 300)
        self.volatility_filter = config.get("volatility_filter", True)
        self.volatility_threshold = config.get("volatility_threshold", 0.03)
        self.rebalance_buffer_pct = config.get("rebalance_buffer_pct", 0.05)
        self.rebalance_target_buffer_pct = config.get("rebalance_target_buffer_pct", 0.01)
        self.min_rebalance_notional = config.get("min_rebalance_notional", 75.0)

        # Trade tracking for daily limits
        self.daily_trade_count = 0
        self.last_trade_time = 0
        self.current_date = datetime.now().date()

        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_portfolio_value = 0.0

        logger.info(
            f"ProfitOptimizer initialized with enhanced risk management: sizing_method={self.position_sizing_method}, volatility_scaling={self.volatility_scaling}"
        )

    def calculate_dynamic_thresholds(
        self, symbol: str, market_data: pd.DataFrame, base_threshold: float
    ) -> Dict[str, float]:
        """Calculate dynamic trading thresholds based on market volatility and momentum."""
        try:
            if market_data.empty or len(market_data) < 20:
                return {"buy": base_threshold, "sell": -base_threshold}

            # Calculate volatility metrics
            returns = market_data["close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # 24 periods per day for 30m data

            # Calculate momentum
            price_change = market_data["close"].iloc[-1] / market_data["close"].iloc[-10] - 1

            # Calculate volume trend
            recent_vol_mean = market_data["volume"].rolling(5).mean().iloc[-1]
            base_vol_mean = market_data["volume"].rolling(20).mean().iloc[-1]
            if base_vol_mean and base_vol_mean > 0:
                volume_trend = recent_vol_mean / base_vol_mean
            else:
                volume_trend = 1.0

            # Adjust thresholds based on market conditions
            volatility_multiplier = max(0.5, min(2.0, volatility / 0.02))  # Adjust for volatility
            momentum_adjustment = np.clip(price_change * 0.5, -0.3, 0.3)  # Momentum bias
            volume_adjustment = np.clip((volume_trend - 1) * 0.2, -0.2, 0.2)  # Volume confirmation

            # Calculate adjusted thresholds
            buy_threshold = (
                base_threshold * volatility_multiplier + momentum_adjustment + volume_adjustment
            )
            sell_threshold = (
                -base_threshold * volatility_multiplier - momentum_adjustment - volume_adjustment
            )

            logger.debug(
                f"{symbol} dynamic thresholds: buy={buy_threshold:.6f}, sell={sell_threshold:.6f} "
                f"(vol_mult={volatility_multiplier:.2f}, momentum={momentum_adjustment:.4f})"
            )

            return {
                "buy": max(0.0001, buy_threshold),  # Minimum threshold
                "sell": min(-0.0001, sell_threshold),  # Maximum negative threshold
                "volatility": volatility,
                "momentum": price_change,
                "volume_trend": volume_trend,
            }

        except Exception as e:
            logger.error(f"Failed to calculate dynamic thresholds for {symbol}: {e}")
            return {"buy": base_threshold, "sell": -base_threshold}

    def calculate_optimal_position_size(
        self,
        symbol: str,
        prediction: float,
        confidence: float,
        current_balance: float,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        market_data: pd.DataFrame,
    ) -> float:
        """Calculate optimal position size using enhanced Kelly Criterion and improved cash utilization."""
        try:
            # Enhanced Kelly Criterion with prediction strength
            win_rate = min(0.95, max(0.15, confidence))  # Wider range for better utilization
            avg_win = self.profit_target_pct
            avg_loss = self.trailing_stop_pct

            # Kelly fraction with prediction strength multiplier
            kelly_base = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            prediction_strength = min(2.0, abs(prediction) * 1000)  # Scale prediction impact
            kelly_fraction = kelly_base * prediction_strength
            kelly_fraction = max(0.02, min(0.4, kelly_fraction))  # Increased range: 2%-40%

            # Calculate portfolio metrics
            # Use current prices to value positions
            total_position_value = 0.0
            for s, qty in current_positions.items():
                price = current_prices.get(s, 0.0)
                if qty > 0 and price > 0:
                    total_position_value += qty * price
            total_portfolio_value = current_balance + total_position_value
            current_exposure = (
                total_position_value / total_portfolio_value if total_portfolio_value > 0 else 0
            )
            cash_ratio = (
                current_balance / total_portfolio_value if total_portfolio_value > 0 else 1.0
            )

            # Enhanced cash utilization - target 80-90% deployment
            target_cash_ratio = 0.15  # Keep only 15% cash
            if cash_ratio > target_cash_ratio:
                # Increase position sizes when we have excess cash
                cash_utilization_multiplier = min(2.0, cash_ratio / target_cash_ratio)
                kelly_fraction *= cash_utilization_multiplier

            # Dynamic max position based on portfolio size and diversification
            num_positions = len([p for p in current_positions.values() if p > 0])
            if num_positions < 3:  # Allow larger positions with fewer holdings
                dynamic_max_position = min(0.25, self.max_position_pct * 1.5)
            elif num_positions < 6:
                dynamic_max_position = self.max_position_pct * 1.2
            else:
                dynamic_max_position = self.max_position_pct

            # Volatility adjustment - more aggressive for stable assets
            if len(market_data) > 20:
                returns = market_data["close"].pct_change().dropna()
                volatility = returns.std() * np.sqrt(24)  # 30-minute intervals
                if volatility < 0.015:  # Low volatility - increase size
                    vol_adjustment = 1.3
                elif volatility < 0.025:  # Medium volatility
                    vol_adjustment = 1.1
                elif volatility > 0.05:  # High volatility - reduce size
                    vol_adjustment = 0.7
                else:
                    vol_adjustment = 1.0
            else:
                vol_adjustment = 1.0

            # Final position size calculation
            optimal_size = kelly_fraction * vol_adjustment
            optimal_size = max(0.01, min(dynamic_max_position, optimal_size))  # Min 1%, max dynamic

            # Ensure we don't exceed total portfolio limits (max 90% deployed)
            if current_exposure + optimal_size > 0.9:
                optimal_size = max(0.01, 0.9 - current_exposure)

            logger.debug(
                f"{symbol} enhanced position size: {optimal_size:.3f} "
                f"(kelly={kelly_fraction:.3f}, cash_ratio={cash_ratio:.3f}, "
                f"exposure={current_exposure:.3f}, vol_adj={vol_adjustment:.3f}, "
                f"pred_strength={prediction_strength:.3f})"
            )

            return optimal_size

        except Exception as e:
            logger.error(f"Failed to calculate optimal position size for {symbol}: {e}")
            return 0.05  # Less conservative fallback

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
                    action="SELL",
                    confidence=0.9,
                    quantity_pct=1.0,  # Sell entire position
                    reasoning=f"Trailing stop triggered at {position.trailing_stop:.4f} (P&L: {unrealized_pnl_pct:.2%})",
                    risk_score=0.1,
                    expected_return=unrealized_pnl_pct,
                )
                logger.info(
                    f"{symbol} trailing stop triggered: price={current_price:.4f}, stop={position.trailing_stop:.4f}"
                )

            # Check for profit target with partial profit-taking
            elif position.profit_target and current_price >= position.profit_target:
                unrealized_pnl_pct = position.unrealized_pnl / position.total_cost
                profit_pct = (current_price - position.avg_cost) / position.avg_cost

                # Check for partial profit-taking at different levels
                if hasattr(self, "profit_scaling_levels") and hasattr(
                    self, "profit_scaling_amounts"
                ):
                    for i, level in enumerate(self.profit_scaling_levels):
                        if profit_pct >= level and not hasattr(position, f"scaled_at_level_{i}"):
                            sell_amount = (
                                self.profit_scaling_amounts[i]
                                if i < len(self.profit_scaling_amounts)
                                else 0.3
                            )

                            signals[symbol] = TradeSignal(
                                symbol=symbol,
                                action="SELL",
                                confidence=0.8,
                                quantity_pct=sell_amount,
                                reasoning=f"Partial profit-taking at {profit_pct:.2%} (level {level:.1%})",
                                risk_score=0.2,
                                expected_return=profit_pct,
                            )

                            # Mark this level as triggered
                            setattr(position, f"scaled_at_level_{i}", True)
                            logger.info(
                                f"{symbol} partial profit-taking: {profit_pct:.2%} profit, selling {sell_amount:.1%}"
                            )
                            break
                else:
                    # Default profit target behavior
                    signals[symbol] = TradeSignal(
                        symbol=symbol,
                        action="SELL",
                        confidence=0.8,
                        quantity_pct=0.7,  # Sell most but keep some for potential upside
                        reasoning=f"Main profit target reached: {profit_pct:.2%}",
                        risk_score=0.2,
                        expected_return=unrealized_pnl_pct,
                    )
                    logger.info(
                        f"{symbol} profit target reached: price={current_price:.4f}, target={position.profit_target:.4f}"
                    )

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
                        action="SELL",
                        confidence=0.6,
                        quantity_pct=0.7,  # Sell most but not all
                        reasoning=f"Position held too long ({holding_time/86400:.1f} days, P&L: {unrealized_pnl_pct:.2%})",
                        risk_score=0.3,
                        expected_return=unrealized_pnl_pct,
                    )
                    logger.info(f"{symbol} time-based exit: held {holding_time/86400:.1f} days")

        return signals

    def analyze_correlation_risk(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, float],
    ) -> Dict[str, float]:
        """Analyze correlation between holdings to manage concentration risk."""
        try:
            if len(symbols) < 2:
                return {}

            # Calculate returns for correlation analysis
            returns_data = {}
            for symbol in symbols:
                if symbol in market_data and len(market_data[symbol]) > 20:
                    returns = market_data[symbol]["close"].pct_change().dropna()
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
                        if (
                            other_weight > 0
                            and symbol in correlation_matrix.columns
                            and other_symbol in correlation_matrix.columns
                        ):
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

    def calculate_kelly_position_size(
        self,
        symbol: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_balance: float,
        volatility: float = None,
    ) -> float:
        """Calculate optimal position size using Kelly criterion."""
        try:
            if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
                return self.max_position_pct  # Fallback to max position size

            # Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win/avg_loss), p = win_rate, q = 1-p
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Apply safety margin (use 25% of Kelly to reduce risk)
            kelly_fraction *= 0.25

            # Apply volatility scaling if enabled
            if self.volatility_scaling and volatility is not None:
                # Reduce position size for high volatility
                vol_adjustment = min(1.0, 0.02 / max(volatility, 0.01))
                kelly_fraction *= vol_adjustment

            # Ensure within bounds
            kelly_fraction = max(0.01, min(kelly_fraction, self.max_position_pct))

            logger.debug(
                f"{symbol} Kelly position size: {kelly_fraction:.3f} (win_rate={win_rate:.3f}, avg_win={avg_win:.3f}, avg_loss={avg_loss:.3f})"
            )
            return kelly_fraction

        except Exception as e:
            logger.error(f"Failed to calculate Kelly position size for {symbol}: {e}")
            return self.max_position_pct

    def calculate_volatility_adjusted_size(
        self, symbol: str, base_size: float, market_data: pd.DataFrame
    ) -> float:
        """Adjust position size based on current volatility."""
        try:
            if not self.volatility_scaling or len(market_data) < 20:
                return base_size

            # Calculate recent volatility (20-period)
            returns = market_data["close"].pct_change().dropna()
            if len(returns) < 10:
                return base_size

            current_vol = returns.tail(20).std() * (252**0.5)  # Annualized volatility
            target_vol = 0.15  # Target 15% annual volatility

            # Scale position size inversely with volatility
            vol_adjustment = min(2.0, target_vol / max(current_vol, 0.05))
            adjusted_size = base_size * vol_adjustment

            # Ensure within bounds
            adjusted_size = max(0.01, min(adjusted_size, self.max_position_pct))

            logger.debug(
                f"{symbol} volatility-adjusted size: {adjusted_size:.3f} (vol={current_vol:.3f}, adjustment={vol_adjustment:.3f})"
            )
            return adjusted_size

        except Exception as e:
            logger.error(f"Failed to calculate volatility-adjusted size for {symbol}: {e}")
            return base_size

    def check_trade_limits(self) -> bool:
        """Check if trade limits allow for new trades."""
        try:
            current_date = datetime.now().date()
            current_time = time.time()

            # Reset daily counter if new day
            if current_date != self.current_date:
                self.daily_trade_count = 0
                self.current_date = current_date

            # Check daily trade limit
            if self.daily_trade_count >= self.max_daily_trades:
                logger.debug(
                    f"Daily trade limit reached: {self.daily_trade_count}/{self.max_daily_trades}"
                )
                return False

            # Check minimum time between trades (skip if configured as 0 or less)
            if getattr(self, "min_time_between_trades", 0) and self.min_time_between_trades > 0:
                if current_time - self.last_trade_time < self.min_time_between_trades:
                    logger.debug(
                        f"Minimum time between trades not met: {current_time - self.last_trade_time:.0f}s < {self.min_time_between_trades}s"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to check trade limits: {e}")
            return True  # Allow trade if check fails

    def should_filter_by_volatility(self, symbol: str, market_data: pd.DataFrame) -> bool:
        """Check if trade should be filtered due to high volatility."""
        try:
            if not self.volatility_filter or len(market_data) < 10:
                return False

            # Calculate recent volatility
            returns = market_data["close"].pct_change().dropna()
            if len(returns) < 5:
                return False

            recent_vol = returns.tail(10).std()

            if recent_vol > self.volatility_threshold:
                logger.debug(
                    f"{symbol} filtered due to high volatility: {recent_vol:.4f} > {self.volatility_threshold:.4f}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to check volatility filter for {symbol}: {e}")
            return False

    def generate_rebalancing_signals(
        self,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        current_balance: float,
    ) -> Dict[str, TradeSignal]:
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

            rebalance_threshold = min(1.0, self.max_position_pct + self.rebalance_buffer_pct)
            target_pct = max(0.0, self.max_position_pct - self.rebalance_target_buffer_pct)

            # Check for over-concentration
            for symbol, value in position_values.items():
                position_pct = value / total_portfolio_value

                if position_pct <= rebalance_threshold:
                    continue

                target_value = target_pct * total_portfolio_value
                sell_value = value - target_value

                if sell_value <= self.min_rebalance_notional:
                    logger.debug(
                        f"Skipping rebalance for {symbol}: excess value €{sell_value:.2f} below minimum €{self.min_rebalance_notional:.2f}"
                    )
                    continue

                sell_pct = sell_value / value if value > 0 else 0.0
                if sell_pct <= 0:
                    continue

                sell_pct = min(0.5, sell_pct)

                signals[symbol] = TradeSignal(
                    symbol=symbol,
                    action="SELL",
                    confidence=0.8,
                    quantity_pct=sell_pct,
                    reasoning=(
                        f"Rebalancing: position {position_pct:.1%} > {rebalance_threshold:.1%} "
                        f"(target {target_pct:.1%}, min_notional €{self.min_rebalance_notional:.0f})"
                    ),
                    risk_score=0.2,
                    expected_return=0.0,
                )
                logger.info(
                    f"{symbol} rebalancing sell: {position_pct:.1%} concentration, trimming {sell_pct:.1%} "
                    f"(buffer {self.rebalance_buffer_pct:.1%})"
                )

            return signals

        except Exception as e:
            logger.error(f"Failed to generate rebalancing signals: {e}")
            return signals

    def update_performance_metrics(
        self,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        current_balance: float,
    ) -> Dict[str, Any]:
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
                self.current_drawdown = (
                    self.peak_portfolio_value - current_portfolio_value
                ) / self.peak_portfolio_value
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

            # Calculate total unrealized P&L
            self.total_unrealized_pnl = 0.0
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position.last_price = current_prices[symbol]
                    position.unrealized_pnl = (
                        position.last_price - position.avg_cost
                    ) * position.quantity
                    self.total_unrealized_pnl += position.unrealized_pnl

            # Performance metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": current_portfolio_value,
                "balance": current_balance,
                "total_realized_pnl": self.total_realized_pnl,
                "total_unrealized_pnl": self.total_unrealized_pnl,
                "total_pnl": self.total_realized_pnl + self.total_unrealized_pnl,
                "max_drawdown": self.max_drawdown,
                "current_drawdown": self.current_drawdown,
                "peak_value": self.peak_portfolio_value,
                "num_positions": len([p for p in self.positions.values() if p.quantity > 0]),
                "positions": {
                    symbol: {
                        "quantity": pos.quantity,
                        "avg_cost": pos.avg_cost,
                        "current_price": pos.last_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "unrealized_pnl_pct": (
                            pos.unrealized_pnl / pos.total_cost if pos.total_cost > 0 else 0
                        ),
                        "trailing_stop": pos.trailing_stop,
                    }
                    for symbol, pos in self.positions.items()
                    if pos.quantity > 0
                },
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

    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        fee: float,
        reasoning: str,
    ) -> None:
        """Record a trade and update position tracking."""
        try:
            trade_time = time.time()

            if action.upper() == "BUY":
                # Update or create position
                if symbol not in self.positions:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_cost=price,
                        total_cost=quantity * price + fee,
                        entry_time=trade_time,
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

                # Update trade tracking
                self.daily_trade_count += 1
                self.last_trade_time = trade_time

            elif action.upper() == "SELL" and symbol in self.positions:
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
                "timestamp": trade_time,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "fee": fee,
                "reasoning": reasoning,
                "total_realized_pnl": self.total_realized_pnl,
            }
            self.trade_history.append(trade_record)

            logger.info(
                f"Trade recorded: {action} {quantity:.6f} {symbol} @ {price:.4f} ({reasoning})"
            )

        except Exception as e:
            logger.error(f"Failed to record trade: {e}")

    def check_profit_targets(
        self, symbol: str, current_price: float, current_amount: float
    ) -> Optional[TradeSignal]:
        """Check if profit targets are reached for a specific position."""
        try:
            if symbol not in self.positions or self.positions[symbol].quantity <= 0:
                return None

            position = self.positions[symbol]
            if current_price <= 0 or current_amount <= 0:
                return None

            # Update unrealized P&L
            position.last_price = current_price
            position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity

            # Check profit target
            if position.profit_target and current_price >= position.profit_target:
                unrealized_pnl_pct = position.unrealized_pnl / position.total_cost
                profit_pct = (current_price - position.avg_cost) / position.avg_cost

                # Check for partial profit-taking at different levels
                if hasattr(self, "profit_scaling_levels") and hasattr(
                    self, "profit_scaling_amounts"
                ):
                    for i, level in enumerate(self.profit_scaling_levels):
                        if profit_pct >= level and not hasattr(position, f"scaled_at_level_{i}"):
                            sell_amount = (
                                self.profit_scaling_amounts[i]
                                if i < len(self.profit_scaling_amounts)
                                else 0.3
                            )

                            signal = TradeSignal(
                                symbol=symbol,
                                action="SELL",
                                confidence=0.8,
                                quantity_pct=sell_amount,
                                reasoning=f"Partial profit-taking at {profit_pct:.2%} (level {level:.1%})",
                                risk_score=0.2,
                                expected_return=profit_pct,
                            )

                            # Mark this level as triggered
                            setattr(position, f"scaled_at_level_{i}", True)
                            logger.info(
                                f"{symbol} partial profit-taking: {profit_pct:.2%} profit, selling {sell_amount:.1%}"
                            )
                            return signal
                else:
                    # Default profit target behavior
                    signal = TradeSignal(
                        symbol=symbol,
                        action="SELL",
                        confidence=0.8,
                        quantity_pct=0.7,  # Sell most but keep some for potential upside
                        reasoning=f"Main profit target reached: {profit_pct:.2%}",
                        risk_score=0.2,
                        expected_return=unrealized_pnl_pct,
                    )
                    logger.info(
                        f"{symbol} profit target reached: price={current_price:.4f}, target={position.profit_target:.4f}"
                    )
                    return signal

            return None

        except Exception as e:
            logger.error(f"Failed to check profit targets for {symbol}: {e}")
            return None
