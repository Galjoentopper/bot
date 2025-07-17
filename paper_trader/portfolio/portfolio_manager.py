"""Portfolio management for tracking positions, capital, and performance."""

import csv
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    take_profit: float
    stop_loss: float
    position_value: float
    confidence: float
    signal_strength: str
    trailing_stop: Optional[float] = None
    highest_price: Optional[float] = None
    
    def __post_init__(self):
        if self.highest_price is None:
            self.highest_price = self.entry_price

@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_time_hours: float
    confidence: float
    signal_strength: str

class PortfolioManager:
    """Manages trading portfolio, positions, and performance tracking."""

    def __init__(self, initial_capital: float = 10000.0, log_dir: str = 'paper_trader/logs',
                 max_positions_per_symbol: int = 1):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, List[Position]] = {}
        self.max_positions_per_symbol = max_positions_per_symbol
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
        # Statistics
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0

        # Track last closed time per symbol for cooldowns
        self.last_closed_time: Dict[str, datetime] = {}
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize CSV files
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialize CSV files for logging trades and portfolio state."""
        # Trades CSV
        trades_file = self.log_dir / 'trades.csv'
        if not trades_file.exists():
            with open(trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'entry_price', 'exit_price', 'quantity',
                    'entry_time', 'exit_time', 'pnl', 'pnl_pct', 'exit_reason',
                    'hold_time_hours', 'confidence', 'signal_strength'
                ])
        
        # Portfolio CSV
        portfolio_file = self.log_dir / 'portfolio.csv'
        if not portfolio_file.exists():
            with open(portfolio_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'total_capital', 'available_capital', 'invested_capital',
                    'total_pnl', 'total_pnl_pct', 'num_positions', 'max_drawdown',
                    'win_rate', 'total_trades'
                ])
    
    def open_position(self, symbol: str, entry_price: float, quantity: float,
                     take_profit: float, stop_loss: float, confidence: float = 0.0,
                     signal_strength: str = 'UNKNOWN') -> bool:
        """Open a new trading position."""
        try:
            current_positions = self.positions.get(symbol, [])
            if len(current_positions) >= self.max_positions_per_symbol:
                self.logger.warning(
                    f"Max positions per symbol reached for {symbol}"
                )
                return False
            
            position_value = entry_price * quantity
            
            # Check if we have enough capital
            if position_value > self.current_capital:
                self.logger.warning(
                    f"Insufficient capital for {symbol}: need {position_value:.2f}, "
                    f"have {self.current_capital:.2f}"
                )
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity,
                entry_time=datetime.now(),
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_value=position_value,
                confidence=confidence,
                signal_strength=signal_strength
            )
            
            # Update portfolio
            self.positions.setdefault(symbol, []).append(position)
            self.current_capital -= position_value
            
            self.logger.info(
                f"Opened position: {symbol} qty={quantity:.6f} @ {entry_price:.4f} "
                f"(value: {position_value:.2f})"
            )
            
            # Log portfolio state
            self._log_portfolio_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening position for {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, position: Position, exit_price: float,
                       reason: str = 'MANUAL') -> bool:
        """Close an existing position."""
        try:
            if symbol not in self.positions or position not in self.positions[symbol]:
                self.logger.warning(f"Position not found for {symbol}")
                return False
            
            # Calculate P&L
            pnl = (exit_price - position.entry_price) * position.quantity
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
            
            # Calculate hold time
            hold_time = datetime.now() - position.entry_time
            hold_time_hours = hold_time.total_seconds() / 3600
            
            # Update capital
            exit_value = exit_price * position.quantity
            self.current_capital += exit_value
            self.total_pnl += pnl
            
            # Update statistics
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update peak capital and drawdown
            total_capital = self.get_total_capital_value()
            if total_capital > self.peak_capital:
                self.peak_capital = total_capital
            else:
                current_drawdown = (self.peak_capital - total_capital) / self.peak_capital
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason,
                hold_time_hours=hold_time_hours,
                confidence=position.confidence,
                signal_strength=position.signal_strength
            )
            
            self.trades.append(trade)

            # Log trade
            self._log_trade(trade)

            # Record last closed time for cooldowns
            self.last_closed_time[symbol] = trade.exit_time
            
            # Remove position
            self.positions[symbol].remove(position)
            if not self.positions[symbol]:
                del self.positions[symbol]
            
            self.logger.info(
                f"Closed position: {symbol} @ {exit_price:.4f} "
                f"P&L: {pnl:.2f} ({pnl_pct:.2%}) - {reason}"
            )
            
            # Log portfolio state
            self._log_portfolio_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of all positions for a symbol."""
        if symbol not in self.positions:
            return 0.0
        return sum(pos.quantity * current_price for pos in self.positions[symbol])
    
    def get_position_pnl(self, symbol: str, current_price: float) -> tuple:
        """Get current P&L for a position."""
        if symbol not in self.positions:
            return 0.0, 0.0

        total_pnl = 0.0
        total_value = 0.0
        for pos in self.positions[symbol]:
            total_pnl += (current_price - pos.entry_price) * pos.quantity
            total_value += pos.entry_price * pos.quantity

        pnl_pct = total_pnl / total_value if total_value != 0 else 0.0

        return total_pnl, pnl_pct
    
    def get_total_capital_value(self, current_prices: Dict[str, float] = None) -> float:
        """Get total portfolio value including open positions."""
        total_value = self.current_capital
        
        if current_prices:
            for symbol, pos_list in self.positions.items():
                if symbol in current_prices:
                    total_value += self.get_position_value(symbol, current_prices[symbol])
                else:
                    # Use entry price if current price not available
                    total_value += sum(p.position_value for p in pos_list)
        else:
            # Use entry values for all positions
            for pos_list in self.positions.values():
                for position in pos_list:
                    total_value += position.position_value
        
        return total_value
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions."""
        return self.current_capital
    
    def get_invested_capital(self) -> float:
        """Get total capital invested in open positions."""
        total = 0.0
        for pos_list in self.positions.values():
            for position in pos_list:
                total += position.position_value
        return total
    
    def get_portfolio_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """Get comprehensive portfolio summary."""
        total_capital = self.get_total_capital_value(current_prices)
        total_return_pct = (total_capital - self.initial_capital) / self.initial_capital
        
        win_rate = 0.0
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
        
        avg_win = 0.0
        avg_loss = 0.0
        if self.trades:
            winning_trades = [t.pnl for t in self.trades if t.pnl > 0]
            losing_trades = [t.pnl for t in self.trades if t.pnl < 0]
            
            if winning_trades:
                avg_win = sum(winning_trades) / len(winning_trades)
            if losing_trades:
                avg_loss = sum(losing_trades) / len(losing_trades)
        
        # Current positions summary
        positions_summary = []
        total_unrealized_pnl = 0.0
        
        for symbol, pos_list in self.positions.items():
            for position in pos_list:
                current_price = current_prices.get(symbol, position.entry_price) if current_prices else position.entry_price
                pnl = (current_price - position.entry_price) * position.quantity
                pnl_pct = (current_price - position.entry_price) / position.entry_price
                total_unrealized_pnl += pnl

                positions_summary.append({
                    'symbol': symbol,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'quantity': position.quantity,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_time': str(datetime.now() - position.entry_time),
                    'confidence': position.confidence,
                    'signal_strength': position.signal_strength
                })
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_capital': total_capital,
            'available_capital': self.get_available_capital(),
            'invested_capital': self.get_invested_capital(),
            'total_return': total_capital - self.initial_capital,
            'total_return_pct': total_return_pct,
            'realized_pnl': self.total_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'num_positions': sum(len(p) for p in self.positions.values()),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'positions': positions_summary
        }
        
        return summary
    
    def _log_trade(self, trade: Trade):
        """Log completed trade to CSV file."""
        try:
            trades_file = self.log_dir / 'trades.csv'
            with open(trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.exit_time.isoformat(),
                    trade.symbol,
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    trade.pnl,
                    trade.pnl_pct,
                    trade.exit_reason,
                    trade.hold_time_hours,
                    trade.confidence,
                    trade.signal_strength
                ])
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
    
    def _log_portfolio_state(self):
        """Log current portfolio state to CSV file."""
        try:
            portfolio_file = self.log_dir / 'portfolio.csv'
            
            total_capital = self.get_total_capital_value()
            total_return_pct = (total_capital - self.initial_capital) / self.initial_capital
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            with open(portfolio_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    total_capital,
                    self.get_available_capital(),
                    self.get_invested_capital(),
                    self.total_pnl,
                    total_return_pct,
                    sum(len(p) for p in self.positions.values()),
                    self.max_drawdown,
                    win_rate,
                    self.total_trades
                ])
        except Exception as e:
            self.logger.error(f"Error logging portfolio state: {e}")
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades as dictionaries."""
        recent_trades = self.trades[-limit:] if len(self.trades) > limit else self.trades
        return [asdict(trade) for trade in recent_trades]
    
    def reset_portfolio(self):
        """Reset portfolio to initial state (for testing)."""
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.initial_capital
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        
        self.logger.info("Portfolio reset to initial state")