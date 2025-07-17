"""Exit management for open positions with multiple exit strategies."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional



class ExitManager:
    """Manages exit conditions for open trading positions."""
    
    def __init__(self, trailing_stop_pct: float = 0.006, max_hold_hours: int = 2,
                 emergency_stop_pct: float = 0.05,
                 enable_prediction_exits: bool = True,
                 prediction_exit_min_confidence: float = 0.8,
                 prediction_exit_min_strength: str = 'STRONG',
                 dynamic_stop_loss_adjustment: bool = True,
                 min_profit_for_trailing: float = 0.005,
                 min_hold_time_minutes: int = 10):
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_hours = max_hold_hours
        self.emergency_stop_pct = emergency_stop_pct

        self.enable_prediction_exits = enable_prediction_exits
        self.prediction_exit_min_confidence = prediction_exit_min_confidence
        self.prediction_exit_min_strength = prediction_exit_min_strength
        self.dynamic_stop_loss_adjustment = dynamic_stop_loss_adjustment
        self.min_profit_for_trailing = min_profit_for_trailing
        self.min_hold_time_minutes = min_hold_time_minutes

        self.logger = logging.getLogger(__name__)
    
    def check_exit_conditions(self, position, current_price: float,
                             prediction_result: dict = None) -> Optional[Dict]:
        """Check all exit conditions including prediction reversal."""
        try:
            # 1. Emergency stop has highest priority
            emergency_exit = self._check_emergency_stop(position, current_price)
            if emergency_exit:
                return emergency_exit

            # 2. Check prediction reversal
            if prediction_result and self.enable_prediction_exits:
                prediction_exit = self._check_prediction_reversal(position, prediction_result)
                if prediction_exit:
                    return prediction_exit

            # Respect minimum hold time before normal exits
            if not self._check_min_hold_time(position):
                self._update_trailing_stop(position, current_price)
                return None

            # 3. Check take profit
            take_profit_exit = self._check_take_profit(position, current_price)
            if take_profit_exit:
                return take_profit_exit

            # 4. Check stop loss
            stop_loss_exit = self._check_stop_loss(position, current_price, prediction_result)
            if stop_loss_exit:
                return stop_loss_exit

            # 5. Check trailing stop
            trailing_stop_exit = self._check_trailing_stop(position, current_price)
            if trailing_stop_exit:
                return trailing_stop_exit

            # 6. Check time-based exit
            time_exit = self._check_time_exit(position)
            if time_exit:
                return time_exit

            # 7. Update trailing stop if price moved favorably
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
    
    def _check_stop_loss(self, position, current_price: float,
                         prediction_result: dict = None) -> Optional[Dict]:
        """Check if stop loss level is reached, adjusting dynamically."""
        if self.dynamic_stop_loss_adjustment:
            adjusted_stop = self._calculate_dynamic_stop_loss(position, prediction_result)
            if adjusted_stop != position.stop_loss:
                self.logger.debug(
                    f"Adjusted stop loss for {position.symbol}: {position.stop_loss:.4f} -> {adjusted_stop:.4f}"
                )
                position.stop_loss = adjusted_stop

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
            profit_pct = (current_price - position.entry_price) / position.entry_price
            if profit_pct < self.min_profit_for_trailing:
                return None
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

    def _check_prediction_reversal(self, position, prediction_result: dict) -> Optional[Dict]:
        """Check if model predicts significant adverse movement."""
        if prediction_result is None:
            return None

        confidence = prediction_result.get('confidence', 0)
        signal_strength = prediction_result.get('signal_strength', 'NEUTRAL')
        predicted_direction = prediction_result.get('direction', 'NEUTRAL')

        if (predicted_direction == 'DOWN' and
            confidence >= self.prediction_exit_min_confidence and
            signal_strength.upper() in ['STRONG', 'VERY_STRONG']):

            self.logger.info(
                f"Prediction reversal exit triggered for {position.symbol}: "
                f"Model predicts {predicted_direction} with {confidence:.1%} confidence"
            )

            return {
                'reason': 'PREDICTION_REVERSAL',
                'exit_price': None,
                'pnl': None,
                'pnl_pct': None,
                'timestamp': datetime.now(),
                'reversal_confidence': confidence,
                'reversal_signal': signal_strength
            }

        return None

    def _calculate_dynamic_stop_loss(self, position, prediction_result: dict) -> float:
        """Dynamically adjust stop loss based on prediction confidence."""
        base_stop_loss = position.stop_loss

        if prediction_result and prediction_result.get('direction') == 'DOWN':
            confidence = prediction_result.get('confidence', 0)

            if confidence > 0.8:
                adjustment_factor = 0.5  # Reduce stop loss distance by 50%
                entry_price = position.entry_price
                stop_distance = entry_price - base_stop_loss
                new_stop_loss = entry_price - (stop_distance * adjustment_factor)

                return max(new_stop_loss, base_stop_loss)

        return base_stop_loss

    def _check_min_hold_time(self, position) -> bool:
        """Return True if minimum hold time has passed."""
        hold_time = datetime.now() - position.entry_time
        return hold_time.total_seconds() >= (self.min_hold_time_minutes * 60)
    
    def _update_trailing_stop(self, position, current_price: float):
        """Update trailing stop if price moved favorably."""
        try:
            # Calculate potential new trailing stop
            new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
            
            # Initialize trailing stop if not set
            if not hasattr(position, 'trailing_stop') or position.trailing_stop is None:
                # Only set trailing stop if minimum profit achieved
                profit_pct = (current_price - position.entry_price) / position.entry_price
                if profit_pct >= self.min_profit_for_trailing:
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