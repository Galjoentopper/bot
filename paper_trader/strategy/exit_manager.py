"""Exit management for open positions with multiple exit strategies."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

class ExitManager:
    """Manages exit conditions for open trading positions."""
    
    def __init__(self, trailing_stop_pct: float = 0.005, max_hold_hours: int = 2,
                 emergency_stop_pct: float = 0.05):
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_hours = max_hold_hours
        self.emergency_stop_pct = emergency_stop_pct
        
        self.logger = logging.getLogger(__name__)
    
    def check_exit_conditions(self, position, current_price: float) -> Optional[Dict]:
        """Check all exit conditions for a position."""
        try:
            # 1. Check take profit
            take_profit_exit = self._check_take_profit(position, current_price)
            if take_profit_exit:
                return take_profit_exit
            
            # 2. Check stop loss
            stop_loss_exit = self._check_stop_loss(position, current_price)
            if stop_loss_exit:
                return stop_loss_exit
            
            # 3. Check trailing stop
            trailing_stop_exit = self._check_trailing_stop(position, current_price)
            if trailing_stop_exit:
                return trailing_stop_exit
            
            # 4. Check time-based exit
            time_exit = self._check_time_exit(position)
            if time_exit:
                return time_exit
            
            # 5. Check emergency stop
            emergency_exit = self._check_emergency_stop(position, current_price)
            if emergency_exit:
                return emergency_exit
            
            # 6. Update trailing stop if price moved favorably
            self._update_trailing_stop(position, current_price)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions for {position.symbol}: {e}")
            return None
    
    def _check_take_profit(self, position, current_price: float) -> Optional[Dict]:
        """Check if take profit level is reached."""
        if current_price >= position.take_profit:
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            self.logger.info(
                f"Take profit triggered for {position.symbol}: "
                f"price {current_price:.4f} >= target {position.take_profit:.4f}"
            )
            
            return {
                'reason': 'TAKE_PROFIT',
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'timestamp': datetime.now()
            }
        return None
    
    def _check_stop_loss(self, position, current_price: float) -> Optional[Dict]:
        """Check if stop loss level is reached."""
        if current_price <= position.stop_loss:
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            self.logger.info(
                f"Stop loss triggered for {position.symbol}: "
                f"price {current_price:.4f} <= stop {position.stop_loss:.4f}"
            )
            
            return {
                'reason': 'STOP_LOSS',
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'timestamp': datetime.now()
            }
        return None
    
    def _check_trailing_stop(self, position, current_price: float) -> Optional[Dict]:
        """Check if trailing stop is triggered."""
        if hasattr(position, 'trailing_stop') and position.trailing_stop:
            if current_price <= position.trailing_stop:
                pnl = (current_price - position.entry_price) * position.quantity
                pnl_pct = (current_price - position.entry_price) / position.entry_price
                
                self.logger.info(
                    f"Trailing stop triggered for {position.symbol}: "
                    f"price {current_price:.4f} <= trailing stop {position.trailing_stop:.4f}"
                )
                
                return {
                    'reason': 'TRAILING_STOP',
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'timestamp': datetime.now()
                }
        return None
    
    def _check_time_exit(self, position) -> Optional[Dict]:
        """Check if maximum hold time is exceeded."""
        if self.max_hold_hours > 0:
            hold_time = datetime.now() - position.entry_time
            max_hold_time = timedelta(hours=self.max_hold_hours)
            
            if hold_time >= max_hold_time:
                self.logger.info(
                    f"Time exit triggered for {position.symbol}: "
                    f"held for {hold_time} >= {max_hold_time}"
                )
                
                return {
                    'reason': 'TIME_EXIT',
                    'exit_price': None,  # Will use current market price
                    'pnl': None,  # Will be calculated at exit
                    'pnl_pct': None,
                    'timestamp': datetime.now(),
                    'hold_time': hold_time
                }
        return None
    
    def _check_emergency_stop(self, position, current_price: float) -> Optional[Dict]:
        """Check for emergency stop (large adverse move)."""
        loss_pct = (position.entry_price - current_price) / position.entry_price
        
        if loss_pct >= self.emergency_stop_pct:
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            self.logger.warning(
                f"Emergency stop triggered for {position.symbol}: "
                f"loss {loss_pct:.2%} >= emergency threshold {self.emergency_stop_pct:.2%}"
            )
            
            return {
                'reason': 'EMERGENCY_STOP',
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'timestamp': datetime.now(),
                'loss_pct': loss_pct
            }
        return None
    
    def _update_trailing_stop(self, position, current_price: float):
        """Update trailing stop if price moved favorably."""
        try:
            # Calculate potential new trailing stop
            new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
            
            # Initialize trailing stop if not set
            if not hasattr(position, 'trailing_stop') or position.trailing_stop is None:
                # Only set trailing stop if we're in profit
                if current_price > position.entry_price:
                    position.trailing_stop = new_trailing_stop
                    position.highest_price = current_price
                    self.logger.debug(
                        f"Initialized trailing stop for {position.symbol}: {new_trailing_stop:.4f}"
                    )
                return
            
            # Update highest price seen
            if not hasattr(position, 'highest_price'):
                position.highest_price = max(position.entry_price, current_price)
            else:
                position.highest_price = max(position.highest_price, current_price)
            
            # Update trailing stop if price moved higher
            if current_price > position.highest_price * 0.999:  # Small tolerance for price fluctuations
                if new_trailing_stop > position.trailing_stop:
                    old_trailing_stop = position.trailing_stop
                    position.trailing_stop = new_trailing_stop
                    
                    self.logger.debug(
                        f"Updated trailing stop for {position.symbol}: "
                        f"{old_trailing_stop:.4f} -> {new_trailing_stop:.4f}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error updating trailing stop for {position.symbol}: {e}")
    
    def get_exit_summary(self, position, current_price: float) -> Dict:
        """Get summary of current exit levels for a position."""
        try:
            current_pnl = (current_price - position.entry_price) * position.quantity
            current_pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            hold_time = datetime.now() - position.entry_time
            remaining_time = None
            if self.max_hold_hours > 0:
                remaining_time = timedelta(hours=self.max_hold_hours) - hold_time
            
            summary = {
                'symbol': position.symbol,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'take_profit': position.take_profit,
                'stop_loss': position.stop_loss,
                'trailing_stop': getattr(position, 'trailing_stop', None),
                'current_pnl': current_pnl,
                'current_pnl_pct': current_pnl_pct,
                'hold_time': str(hold_time),
                'remaining_time': str(remaining_time) if remaining_time else None,
                'distance_to_take_profit': (position.take_profit - current_price) / current_price,
                'distance_to_stop_loss': (current_price - position.stop_loss) / current_price,
                'highest_price': getattr(position, 'highest_price', current_price)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating exit summary for {position.symbol}: {e}")
            return {}
    
    def update_exit_parameters(self, **kwargs):
        """Update exit management parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated exit parameter {key} to {value}")
            else:
                self.logger.warning(f"Unknown exit parameter: {key}")