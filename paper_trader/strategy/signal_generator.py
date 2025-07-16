"""Trading signal generation based on model predictions and risk management."""

import logging
from datetime import datetime
from typing import Dict, Optional

class SignalGenerator:
    """Generates trading signals based on model predictions and portfolio constraints."""
    
    def __init__(self, max_positions: int = 10, position_size_pct: float = 0.10,
                 take_profit_pct: float = 0.01, stop_loss_pct: float = 0.01,
                 min_confidence: float = 0.5, min_signal_strength: str = 'MODERATE',
                 min_expected_gain_pct: float = 0.005):
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence
        self.min_signal_strength = min_signal_strength
        self.min_expected_gain_pct = min_expected_gain_pct
        
        # Signal strength hierarchy
        self.signal_hierarchy = {
            'WEAK': 1,
            'NEUTRAL': 2,
            'MODERATE': 3,
            'STRONG': 4,
            'VERY_STRONG': 5
        }

        self.logger = logging.getLogger(__name__)

    def calculate_position_size(self, confidence: float, signal_strength: str) -> float:
        """Calculate dynamic position size based on confidence and signal strength."""
        confidence_multiplier = min(1.0, max(0.5, confidence))
        strength_multipliers = {
            'VERY_STRONG': 1.0,
            'STRONG': 0.8,
            'MODERATE': 0.6,
            'WEAK': 0.4,
            'NEUTRAL': 0.0
        }
        strength_multiplier = strength_multipliers.get(signal_strength, 0.5)
        return self.position_size_pct * confidence_multiplier * strength_multiplier
    
    def generate_signal(self, symbol: str, prediction: dict, current_price: float, 
                       portfolio) -> Optional[Dict]:
        """Generate trading signal based on prediction and portfolio state."""
        try:
            # Check if we already have a position in this symbol
            if symbol in portfolio.positions:
                self.logger.debug(f"Already have position in {symbol}, skipping")
                return None
            
            # Check portfolio constraints
            if len(portfolio.positions) >= self.max_positions:
                self.logger.debug(f"Maximum positions ({self.max_positions}) reached")
                return None
            
            # Validate prediction quality
            if not self._is_prediction_valid(prediction):
                return None
            
            # Determine signal direction
            price_change_pct = prediction['price_change_pct']
            confidence = prediction['confidence']
            signal_strength = prediction['signal_strength']
            
            # Check if signal meets minimum requirements
            if confidence < self.min_confidence:
                self.logger.debug(f"Confidence too low for {symbol}: {confidence:.3f}")
                return None
            
            min_strength_value = self.signal_hierarchy.get(self.min_signal_strength, 3)
            signal_strength_value = self.signal_hierarchy.get(signal_strength, 0)
            
            if signal_strength_value < min_strength_value:
                self.logger.debug(f"Signal strength too low for {symbol}: {signal_strength}")
                return None
            
            # Generate buy signal if expected gain exceeds threshold
            if price_change_pct > self.min_expected_gain_pct:
                return self._generate_buy_signal(symbol, current_price, prediction, portfolio)
            
            # No signal for neutral or negative predictions in this strategy
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _is_prediction_valid(self, prediction: dict) -> bool:
        """Validate prediction data quality."""
        required_fields = ['current_price', 'predicted_price', 'price_change_pct', 
                          'confidence', 'signal_strength']
        
        for field in required_fields:
            if field not in prediction:
                self.logger.warning(f"Missing field in prediction: {field}")
                return False
        
        # Check for reasonable values
        if prediction['confidence'] < 0 or prediction['confidence'] > 1:
            self.logger.warning(f"Invalid confidence value: {prediction['confidence']}")
            return False
        
        if abs(prediction['price_change_pct']) > 0.5:  # More than 50% change seems unrealistic
            self.logger.warning(f"Unrealistic price change: {prediction['price_change_pct']}")
            return False
        
        return True
    
    def _generate_buy_signal(self, symbol: str, current_price: float, 
                           prediction: dict, portfolio) -> Dict:
        """Generate a buy signal with position sizing and risk management."""
        try:
            # Calculate position size dynamically
            size_pct = self.calculate_position_size(
                prediction['confidence'], prediction['signal_strength']
            )
            available_capital = portfolio.get_available_capital()
            adjusted_position_value = available_capital * size_pct
            
            # Calculate quantity (assuming we can buy fractional shares)
            quantity = adjusted_position_value / current_price
            
            # Calculate take profit and stop loss levels
            take_profit_price = current_price * (1 + self.take_profit_pct)
            stop_loss_price = current_price * (1 - self.stop_loss_pct)
            
            signal = {
                'action': 'BUY',
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': current_price,
                'take_profit': take_profit_price,
                'stop_loss': stop_loss_price,
                'position_value': quantity * current_price,
                'confidence': prediction['confidence'],
                'signal_strength': prediction['signal_strength'],
                'expected_return_pct': prediction['price_change_pct'],
                'risk_reward_ratio': self.take_profit_pct / self.stop_loss_pct,
                'timestamp': datetime.now(),
                'prediction_data': prediction
            }
            
            self.logger.info(
                f"Generated BUY signal for {symbol}: "
                f"qty={quantity:.6f}, price={current_price:.4f}, "
                f"confidence={prediction['confidence']:.3f}, "
                f"strength={prediction['signal_strength']}"
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating buy signal for {symbol}: {e}")
            return None
    
    def update_signal_parameters(self, **kwargs):
        """Update signal generation parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated {key} to {value}")
            else:
                self.logger.warning(f"Unknown parameter: {key}")
    
    def get_signal_statistics(self) -> Dict:
        """Get current signal generation parameters."""
        return {
            'max_positions': self.max_positions,
            'position_size_pct': self.position_size_pct,
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'min_confidence': self.min_confidence,
            'min_signal_strength': self.min_signal_strength,
            'min_expected_gain_pct': self.min_expected_gain_pct,
            'signal_hierarchy': self.signal_hierarchy
        }