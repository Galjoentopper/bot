"""Trading signal generation based on model predictions and risk management."""

import logging
from datetime import datetime
from typing import Dict, Optional

# Import settings for configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import TradingSettings

class SignalGenerator:
    """Generates trading signals based on model predictions and portfolio constraints."""
    
    def __init__(self, max_positions: int = 10, base_position_size: float = 0.08,
                 max_position_size: float = 0.15, min_position_size: float = 0.02,
                 take_profit_pct: float = 0.015, stop_loss_pct: float = 0.008,
                 min_confidence: float = 0.7, min_signal_strength: str = 'MODERATE',
                 min_expected_gain_pct: float = 0.001,
                 max_positions_per_symbol: int = 1,
                 position_cooldown_minutes: int = 5,
                 data_collector=None,
                 max_daily_trades_per_symbol: int = 50,
                 settings: TradingSettings = None):
        if settings is None:
            settings = TradingSettings()
        self.settings = settings
        self.max_positions = max_positions
        self.max_positions_per_symbol = max_positions_per_symbol
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence
        self.min_signal_strength = min_signal_strength
        self.min_expected_gain_pct = min_expected_gain_pct
        self.position_cooldown_minutes = position_cooldown_minutes
        self.max_daily_trades_per_symbol = max_daily_trades_per_symbol
        self.data_collector = data_collector
        
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
        confidence_multiplier = max(self.settings.confidence_multiplier_min, min(self.settings.confidence_multiplier_max, confidence))
        strength_multipliers = {
            'VERY_STRONG': self.settings.very_strong_signal_multiplier,
            'STRONG': self.settings.strong_signal_multiplier,
            'MODERATE': self.settings.moderate_signal_multiplier,
            'WEAK': 0.5,
            'NEUTRAL': 0.3
        }
        strength_multiplier = strength_multipliers.get(signal_strength, 0.5)
        final_size = self.base_position_size * confidence_multiplier * strength_multiplier
        return max(self.min_position_size, min(self.max_position_size, final_size))
    
    def generate_signal(self, symbol: str, prediction: dict, current_price: float, 
                       portfolio) -> Optional[Dict]:
        """Generate trading signal based on prediction and portfolio state."""
        try:
            # Comprehensive logging for trading decision process
            self.logger.info(f"=== TRADING DECISION ANALYSIS FOR {symbol} ===")
            self.logger.info(f"Current price: {current_price}")
            self.logger.info(f"Prediction data: {prediction}")
            
            # Check if we already have too many positions for this symbol
            current_per_symbol = len(portfolio.positions.get(symbol, []))
            self.logger.info(f"Current positions for {symbol}: {current_per_symbol}/{self.max_positions_per_symbol}")
            if current_per_symbol >= self.max_positions_per_symbol:
                self.logger.warning(
                    f"❌ REJECTED: Maximum positions for {symbol} reached ({current_per_symbol}/{self.max_positions_per_symbol})"
                )
                return None

            # Cooldown after closing a position
            last_closed = getattr(portfolio, 'last_closed_time', {}).get(symbol)
            if last_closed:
                cooldown_remaining = self.position_cooldown_minutes * 60 - (datetime.now() - last_closed).total_seconds()
                self.logger.info(f"Cooldown check for {symbol}: {cooldown_remaining:.0f}s remaining")
                if cooldown_remaining > 0:
                    self.logger.warning(f"❌ REJECTED: Cooldown active for {symbol} ({cooldown_remaining:.0f}s remaining)")
                    return None

            # Limit daily trades per symbol
            if self.max_daily_trades_per_symbol and hasattr(portfolio, 'trades'):
                today_trades = [t for t in portfolio.trades if t.symbol == symbol and (datetime.now() - t.exit_time).total_seconds() < 86400]
                self.logger.info(f"Daily trades for {symbol}: {len(today_trades)}/{self.max_daily_trades_per_symbol}")
                if len(today_trades) >= self.max_daily_trades_per_symbol:
                    self.logger.warning(f"❌ REJECTED: Daily trade limit reached for {symbol} ({len(today_trades)}/{self.max_daily_trades_per_symbol})")
                    return None
            
            # Check portfolio constraints
            total_positions = sum(len(p) for p in portfolio.positions.values())
            self.logger.info(f"Total portfolio positions: {total_positions}/{self.max_positions}")
            if total_positions >= self.max_positions:
                self.logger.warning(f"❌ REJECTED: Maximum positions ({self.max_positions}) reached")
                return None
            
            # Validate prediction quality
            if not self._is_prediction_valid(prediction):
                self.logger.warning(f"❌ REJECTED: Invalid prediction data for {symbol}")
                return None
            
            # Determine signal direction
            price_change_pct = prediction['price_change_pct']
            confidence = prediction['confidence']
            signal_strength = prediction['signal_strength']
            
            self.logger.info(f"Signal analysis - Price change: {price_change_pct:.4f}, Confidence: {confidence:.3f}, Strength: {signal_strength}")
            
            # Check if signal meets minimum requirements
            self.logger.info(f"Confidence threshold check: {confidence:.3f} >= {self.min_confidence}")
            if confidence < self.min_confidence:
                self.logger.warning(f"❌ REJECTED: Confidence too low for {symbol}: {confidence:.3f} < {self.min_confidence}")
                return None
            
            min_strength_value = self.signal_hierarchy.get(self.min_signal_strength, 3)
            signal_strength_value = self.signal_hierarchy.get(signal_strength, 0)
            self.logger.info(f"Signal strength check: {signal_strength}({signal_strength_value}) >= {self.min_signal_strength}({min_strength_value})")
            
            if signal_strength_value < min_strength_value:
                self.logger.warning(f"❌ REJECTED: Signal strength too low for {symbol}: {signal_strength}({signal_strength_value}) < {self.min_signal_strength}({min_strength_value})")
                return None
            
            # Check market conditions before generating a signal
            market_conditions_ok = self._check_market_conditions(symbol)
            self.logger.info(f"Market conditions check: {market_conditions_ok}")
            if not market_conditions_ok:
                self.logger.warning(f"❌ REJECTED: Unfavorable market conditions for {symbol}")
                return None

            # Check strict entry conditions if enabled
            strict_conditions_ok = self._check_strict_entry_conditions(symbol, prediction)
            self.logger.info(f"Strict entry conditions check: {strict_conditions_ok}")
            if not strict_conditions_ok:
                self.logger.warning(f"❌ REJECTED: Failed strict entry conditions for {symbol}")
                return None

            # Check expected gain threshold
            self.logger.info(f"Expected gain check: {price_change_pct:.4f} > {self.min_expected_gain_pct:.4f}")
            if price_change_pct > self.min_expected_gain_pct:
                self.logger.info(f"✅ APPROVED: Generating BUY signal for {symbol}")
                signal = self._generate_buy_signal(symbol, current_price, prediction, portfolio)
                if signal:
                    self.logger.info(f"✅ SIGNAL GENERATED: {signal}")
                return signal
            else:
                self.logger.warning(f"❌ REJECTED: Expected gain too low for {symbol}: {price_change_pct:.4f} <= {self.min_expected_gain_pct:.4f}")
            
            # No signal for neutral or negative predictions in this strategy
            self.logger.info(f"=== NO SIGNAL GENERATED FOR {symbol} ===")
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

    def _check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are favorable for trading."""
        if not self.data_collector:
            self.logger.info(f"No data collector available, skipping market conditions check for {symbol}")
            return True
        try:
            recent_data = self.data_collector.get_buffer_data(symbol, 20)
            if recent_data is None or len(recent_data) < 20:
                self.logger.warning(f"Insufficient recent data for market conditions check: {len(recent_data) if recent_data is not None else 0}/20 candles")
                return False

            volatility = recent_data['close'].pct_change().std()
            self.logger.info(f"Market volatility for {symbol}: {volatility:.4f} (max: 0.03)")
            if volatility > 0.03:
                self.logger.warning(f"Market too volatile for {symbol}: {volatility:.4f} > 0.03")
                return False

            sma_short = recent_data['close'].rolling(5).mean().iloc[-1]
            sma_long = recent_data['close'].rolling(20).mean().iloc[-1]
            trend_strength = abs(sma_short - sma_long) / sma_long
            self.logger.info(f"Trend strength for {symbol}: {trend_strength:.4f} (min: {self.settings.trend_strength_threshold})")
            
            if trend_strength <= self.settings.trend_strength_threshold:
                self.logger.warning(f"Trend strength too weak for {symbol}: {trend_strength:.4f} <= {self.settings.trend_strength_threshold}")
                return False
            
            self.logger.info(f"Market conditions favorable for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Market condition check failed for {symbol}: {e}")
            return False

    def _check_strict_entry_conditions(self, symbol: str, prediction: dict) -> bool:
        """Check strict entry conditions if enabled in settings."""
        if not self.settings.enable_strict_entry_conditions:
            self.logger.info(f"Strict entry conditions disabled - PASS")
            return True
            
        try:
            self.logger.info(f"Checking strict entry conditions for {symbol}")
            
            # Check prediction uncertainty threshold
            uncertainty = prediction.get('uncertainty', 0.0)
            self.logger.info(f"Prediction uncertainty: {uncertainty:.3f} (max: {self.settings.max_prediction_uncertainty})")
            if uncertainty > self.settings.max_prediction_uncertainty:
                self.logger.warning(f"Prediction uncertainty too high for {symbol}: {uncertainty:.3f} > {self.settings.max_prediction_uncertainty}")
                return False
            
            # Check ensemble agreement - count how many models made predictions
            individual_predictions = prediction.get('individual_predictions', {})
            model_count = len(individual_predictions)
            self.logger.info(f"Ensemble model agreement: {model_count}/{self.settings.min_ensemble_agreement_count} models")
            if model_count < self.settings.min_ensemble_agreement_count:
                self.logger.warning(f"Not enough model agreement for {symbol}: {model_count}/{self.settings.min_ensemble_agreement_count}")
                return False
            
            # Check volume conditions if data available
            if self.data_collector:
                recent_data = self.data_collector.get_buffer_data(symbol, 20)
                if recent_data is not None and len(recent_data) >= 20:
                    recent_volume = recent_data['volume'].iloc[-1]
                    avg_volume = recent_data['volume'].rolling(10).mean().iloc[-1]
                    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                    self.logger.info(f"Volume ratio: {volume_ratio:.3f} (min: {self.settings.min_volume_ratio_threshold})")
                    
                    if volume_ratio < self.settings.min_volume_ratio_threshold:
                        self.logger.warning(f"Volume too low for {symbol}: {volume_ratio:.3f} < {self.settings.min_volume_ratio_threshold}")
                        return False
            
            # Apply confidence boost for very confident predictions
            confidence = prediction.get('confidence', 0.0)
            signal_strength = prediction.get('signal_strength', 'WEAK')
            
            # If confidence is very high, we can accept slightly weaker signals
            if confidence >= self.settings.strong_signal_confidence_boost:
                # Allow MODERATE signals if confidence is very high
                if signal_strength in ['MODERATE', 'STRONG', 'VERY_STRONG']:
                    self.logger.info(f"High confidence boost applied for {symbol}: {confidence:.3f} >= {self.settings.strong_signal_confidence_boost}")
                    return True
            
            self.logger.info(f"All strict entry conditions passed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking strict entry conditions for {symbol}: {e}")
            return False
    
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
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'base_position_size': self.base_position_size,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'min_confidence': self.min_confidence,
            'min_signal_strength': self.min_signal_strength,
            'min_expected_gain_pct': self.min_expected_gain_pct,
            'signal_hierarchy': self.signal_hierarchy,
            'position_cooldown_minutes': self.position_cooldown_minutes,
            'max_daily_trades_per_symbol': self.max_daily_trades_per_symbol
        }