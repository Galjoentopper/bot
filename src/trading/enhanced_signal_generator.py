"""
Enhanced Signal Generator with Profit Optimization
================================================

This module provides advanced signal generation that integrates:
- Model ensemble predictions
- Profit optimization strategies
- Risk management
- Market regime detection
- Portfolio diversification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .profit_optimizer import ProfitOptimizer, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Container for model prediction with metadata."""
    model_type: str
    prediction: float
    confidence: float
    features_used: int
    validation_score: Optional[float] = None


@dataclass
class MarketContext:
    """Market context for enhanced decision making."""
    symbol: str
    volatility: float
    momentum: float
    volume_trend: float
    price_trend: str  # 'bullish', 'bearish', 'sideways'
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


class EnhancedSignalGenerator:
    """Advanced signal generator with profit optimization integration."""
    
    def __init__(self, config: Dict[str, Any], profit_optimizer: ProfitOptimizer):
        self.config = config
        self.profit_optimizer = profit_optimizer
        
        # Configuration
        self.model_weights = config.get('model_weights', {'gru': 0.45, 'lightgbm': 0.45, 'ppo': 0.1})
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.ensemble_method = config.get('ensemble_method', 'weighted_average')
        self.regime_detection = config.get('regime_detection', True)
        
        # Market analysis parameters
        self.volatility_lookback = config.get('volatility_lookback', 20)
        self.momentum_lookback = config.get('momentum_lookback', 10)
        self.volume_lookback = config.get('volume_lookback', 20)
        
        logger.info("EnhancedSignalGenerator initialized with profit optimization")
    
    def generate_enhanced_signals(self, model_predictions: Dict[str, List[ModelPrediction]],
                                market_data: Dict[str, pd.DataFrame],
                                current_positions: Dict[str, float],
                                current_prices: Dict[str, float],
                                current_balance: float) -> Dict[str, TradeSignal]:
        """Generate enhanced trading signals with profit optimization."""
        
        all_signals = {}
        
        try:
            # 1. Generate risk management signals (highest priority)
            risk_signals = self._generate_risk_management_signals(
                current_positions, current_prices, current_balance
            )
            all_signals.update(risk_signals)
            
            # 2. Generate model-based signals
            model_signals = self._generate_model_based_signals(
                model_predictions, market_data, current_positions, current_prices, current_balance
            )
            
            # 3. Merge signals with priority to risk management
            final_signals = self._merge_signals(all_signals, model_signals)
            
            # 4. Apply portfolio optimization filters
            optimized_signals = self._apply_portfolio_optimization(
                final_signals, current_positions, current_prices, current_balance, market_data
            )
            
            logger.info(f"Generated {len(optimized_signals)} enhanced signals")
            return optimized_signals
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced signals: {e}")
            return {}
    
    def _generate_risk_management_signals(self, current_positions: Dict[str, float],
                                        current_prices: Dict[str, float],
                                        current_balance: float) -> Dict[str, TradeSignal]:
        """Generate risk management signals (stop-loss, profit-taking, rebalancing)."""
        signals = {}
        
        # Get trailing stop signals
        trailing_signals = self.profit_optimizer.update_trailing_stops(current_prices)
        signals.update(trailing_signals)
        
        # Get time-based exit signals
        time_signals = self.profit_optimizer.check_time_based_exits()
        signals.update(time_signals)
        
        # Get rebalancing signals
        rebalancing_signals = self.profit_optimizer.generate_rebalancing_signals(
            current_positions, current_prices, current_balance
        )
        signals.update(rebalancing_signals)
        
        logger.debug(f"Generated {len(signals)} risk management signals")
        return signals
    
    def _generate_model_based_signals(self, model_predictions: Dict[str, List[ModelPrediction]],
                                    market_data: Dict[str, pd.DataFrame],
                                    current_positions: Dict[str, float],
                                    current_prices: Dict[str, float],
                                    current_balance: float) -> Dict[str, TradeSignal]:
        """Generate signals based on model predictions."""
        signals = {}
        
        for symbol, predictions in model_predictions.items():
            try:
                if not predictions or symbol not in market_data:
                    continue
                
                # Analyze market context
                market_context = self._analyze_market_context(symbol, market_data[symbol])
                
                # Combine model predictions
                ensemble_prediction = self._combine_model_predictions(predictions)
                
                # Get dynamic thresholds
                thresholds = self.profit_optimizer.calculate_dynamic_thresholds(
                    symbol, market_data[symbol], self.config.get('thresholds', {}).get('default', 0.0005)
                )
                
                # Generate signal
                signal = self._generate_signal_from_prediction(
                    symbol, ensemble_prediction, thresholds, market_context,
                    current_positions, current_prices, current_balance, market_data[symbol]
                )
                
                if signal:
                    signals[symbol] = signal
                    
            except Exception as e:
                logger.error(f"Failed to generate model signal for {symbol}: {e}")
        
        return signals
    
    def _analyze_market_context(self, symbol: str, market_data: pd.DataFrame) -> MarketContext:
        """Analyze market context for enhanced signal generation."""
        try:
            if len(market_data) < 20:
                return MarketContext(symbol, 0.02, 0.0, 1.0, 'sideways')
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.tail(self.volatility_lookback).std() * np.sqrt(24)
            
            # Calculate momentum
            momentum = (market_data['close'].iloc[-1] / market_data['close'].iloc[-self.momentum_lookback] - 1)
            
            # Calculate volume trend
            recent_volume = market_data['volume'].tail(5).mean()
            baseline_volume = market_data['volume'].tail(self.volume_lookback).mean()
            volume_trend = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
            
            # Determine price trend
            sma_short = market_data['close'].tail(5).mean()
            sma_long = market_data['close'].tail(20).mean()
            
            if sma_short > sma_long * 1.005:
                price_trend = 'bullish'
            elif sma_short < sma_long * 0.995:
                price_trend = 'bearish'
            else:
                price_trend = 'sideways'
            
            # Calculate support and resistance levels
            recent_data = market_data.tail(50)
            resistance_level = recent_data['high'].max()
            support_level = recent_data['low'].min()
            
            return MarketContext(
                symbol=symbol,
                volatility=volatility,
                momentum=momentum,
                volume_trend=volume_trend,
                price_trend=price_trend,
                support_level=support_level,
                resistance_level=resistance_level
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze market context for {symbol}: {e}")
            return MarketContext(symbol, 0.02, 0.0, 1.0, 'sideways')
    
    def _combine_model_predictions(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Combine multiple model predictions into ensemble prediction."""
        try:
            if not predictions:
                return ModelPrediction('ensemble', 0.0, 0.0, 0)
            
            if len(predictions) == 1:
                return predictions[0]
            
            # Weighted average based on confidence and configured weights
            total_weight = 0.0
            weighted_sum = 0.0
            confidence_sum = 0.0
            total_features = 0
            
            for pred in predictions:
                # Get model weight
                model_weight = self.model_weights.get(pred.model_type, 1.0)
                
                # Combine model weight with prediction confidence
                combined_weight = model_weight * pred.confidence
                
                weighted_sum += pred.prediction * combined_weight
                confidence_sum += pred.confidence * model_weight
                total_weight += combined_weight
                total_features += pred.features_used
            
            if total_weight == 0:
                return ModelPrediction('ensemble', 0.0, 0.0, 0)
            
            ensemble_prediction = weighted_sum / total_weight
            ensemble_confidence = confidence_sum / sum(self.model_weights.get(p.model_type, 1.0) for p in predictions)
            
            logger.debug(f"Ensemble prediction: {ensemble_prediction:.6f} (confidence: {ensemble_confidence:.3f})")
            
            return ModelPrediction(
                model_type='ensemble',
                prediction=ensemble_prediction,
                confidence=ensemble_confidence,
                features_used=total_features
            )
            
        except Exception as e:
            logger.error(f"Failed to combine model predictions: {e}")
            return ModelPrediction('ensemble', 0.0, 0.0, 0)
    
    def _generate_signal_from_prediction(self, symbol: str, prediction: ModelPrediction,
                                       thresholds: Dict[str, float], market_context: MarketContext,
                                       current_positions: Dict[str, float],
                                       current_prices: Dict[str, float],
                                       current_balance: float,
                                       market_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Generate trading signal from ensemble prediction and market context."""
        try:
            current_position = current_positions.get(symbol, 0.0)
            current_price = current_prices.get(symbol, 0.0)
            
            if current_price <= 0:
                return None
            
            # Adjust prediction based on market context
            context_adjusted_prediction = self._apply_market_context_adjustment(
                prediction.prediction, market_context
            )
            
            # Get thresholds
            buy_threshold = thresholds.get('buy', 0.0005)
            sell_threshold = thresholds.get('sell', -0.0005)
            
            # Generate BUY signals
            if context_adjusted_prediction > buy_threshold and prediction.confidence > self.confidence_threshold:
                # Check if we should buy (not over-concentrated)
                optimal_position_size = self.profit_optimizer.calculate_optimal_position_size(
                    symbol, context_adjusted_prediction, prediction.confidence,
                    current_balance, current_positions, market_data
                )
                
                if optimal_position_size > 0.01:  # Minimum 1% position
                    return TradeSignal(
                        symbol=symbol,
                        action='BUY',
                        confidence=prediction.confidence,
                        quantity_pct=optimal_position_size,
                        reasoning=f"Ensemble prediction {context_adjusted_prediction:.6f} > {buy_threshold:.6f} "
                                f"(confidence: {prediction.confidence:.3f}, market: {market_context.price_trend})",
                        risk_score=1.0 - prediction.confidence,
                        expected_return=context_adjusted_prediction
                    )
            
            # Generate SELL signals for existing positions
            elif current_position > 0:
                sell_quantity_pct = 0.0
                sell_reason = ""
                
                # Strong negative prediction
                if context_adjusted_prediction < sell_threshold:
                    sell_quantity_pct = 0.75
                    sell_reason = f"Strong negative prediction {context_adjusted_prediction:.6f}"
                
                # Weak positive in bearish market
                elif (market_context.price_trend == 'bearish' and 
                      0 < context_adjusted_prediction < buy_threshold * 0.5):
                    sell_quantity_pct = 0.3
                    sell_reason = f"Weak signal in bearish market {context_adjusted_prediction:.6f}"
                
                # Low confidence prediction
                elif prediction.confidence < 0.4:
                    sell_quantity_pct = 0.25
                    sell_reason = f"Low confidence {prediction.confidence:.3f}"
                
                if sell_quantity_pct > 0:
                    return TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        confidence=max(0.5, 1.0 - prediction.confidence),
                        quantity_pct=sell_quantity_pct,
                        reasoning=sell_reason,
                        risk_score=prediction.confidence,
                        expected_return=context_adjusted_prediction
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate signal from prediction for {symbol}: {e}")
            return None
    
    def _apply_market_context_adjustment(self, prediction: float, context: MarketContext) -> float:
        """Apply market context adjustments to the raw prediction."""
        try:
            adjusted_prediction = prediction
            
            # Volatility adjustment
            if context.volatility > 0.05:  # High volatility
                adjusted_prediction *= 0.8  # Be more conservative
            elif context.volatility < 0.01:  # Low volatility
                adjusted_prediction *= 1.2  # Be more aggressive
            
            # Momentum adjustment
            momentum_factor = np.clip(1.0 + context.momentum * 0.5, 0.5, 1.5)
            adjusted_prediction *= momentum_factor
            
            # Volume confirmation
            if context.volume_trend > 1.5:  # High volume
                adjusted_prediction *= 1.1  # Increase confidence
            elif context.volume_trend < 0.7:  # Low volume
                adjusted_prediction *= 0.9  # Decrease confidence
            
            # Trend alignment
            if context.price_trend == 'bullish' and prediction > 0:
                adjusted_prediction *= 1.15
            elif context.price_trend == 'bearish' and prediction < 0:
                adjusted_prediction *= 1.15
            elif context.price_trend == 'bullish' and prediction < 0:
                adjusted_prediction *= 0.85
            elif context.price_trend == 'bearish' and prediction > 0:
                adjusted_prediction *= 0.85
            
            logger.debug(f"Market context adjustment: {prediction:.6f} -> {adjusted_prediction:.6f} "
                        f"(trend: {context.price_trend}, vol: {context.volatility:.3f})")
            
            return adjusted_prediction
            
        except Exception as e:
            logger.error(f"Failed to apply market context adjustment: {e}")
            return prediction
    
    def _merge_signals(self, priority_signals: Dict[str, TradeSignal],
                      secondary_signals: Dict[str, TradeSignal]) -> Dict[str, TradeSignal]:
        """Merge signals with priority to the first set."""
        merged = priority_signals.copy()
        
        for symbol, signal in secondary_signals.items():
            if symbol not in merged:
                merged[symbol] = signal
            else:
                # Keep priority signal but log conflict
                logger.debug(f"Signal conflict for {symbol}: keeping priority signal "
                           f"{merged[symbol].action} over {signal.action}")
        
        return merged
    
    def _apply_portfolio_optimization(self, signals: Dict[str, TradeSignal],
                                    current_positions: Dict[str, float],
                                    current_prices: Dict[str, float],
                                    current_balance: float,
                                    market_data: Dict[str, pd.DataFrame]) -> Dict[str, TradeSignal]:
        """Apply portfolio-level optimization to signals."""
        try:
            optimized_signals = {}
            
            # Calculate correlation risks
            symbols = list(signals.keys())
            correlation_risks = self.profit_optimizer.analyze_correlation_risk(
                symbols, market_data, current_positions
            )
            
            # Sort signals by expected return and confidence
            sorted_signals = sorted(
                signals.items(),
                key=lambda x: x[1].expected_return * x[1].confidence,
                reverse=True
            )
            
            total_portfolio_value = current_balance + sum(
                current_positions.get(s, 0.0) * current_prices.get(s, 0.0)
                for s in current_positions.keys()
            )
            
            current_cash_pct = current_balance / total_portfolio_value if total_portfolio_value > 0 else 1.0
            
            for symbol, signal in sorted_signals:
                # Always allow sell signals
                if signal.action == 'SELL':
                    optimized_signals[symbol] = signal
                    continue
                
                # For buy signals, apply portfolio constraints
                if signal.action == 'BUY':
                    # Check correlation risk
                    correlation_risk = correlation_risks.get(symbol, 0.0)
                    if correlation_risk > 0.8:  # High correlation
                        # Reduce position size
                        signal.quantity_pct *= 0.5
                        signal.reasoning += f" (reduced due to correlation risk: {correlation_risk:.2f})"
                    
                    # Check cash availability
                    if current_cash_pct < 0.1:  # Less than 10% cash
                        signal.quantity_pct *= 0.5
                        signal.reasoning += " (reduced due to low cash)"
                    
                    # Only add if still significant
                    if signal.quantity_pct > 0.01:
                        optimized_signals[symbol] = signal
                
                # Hold signals (no action needed)
            
            logger.debug(f"Portfolio optimization: {len(signals)} -> {len(optimized_signals)} signals")
            return optimized_signals
            
        except Exception as e:
            logger.error(f"Failed to apply portfolio optimization: {e}")
            return signals


def create_enhanced_signal_generator(config: Dict[str, Any]) -> EnhancedSignalGenerator:
    """Factory function to create an enhanced signal generator."""
    profit_optimizer = ProfitOptimizer(config.get('profit_optimization', {}))
    return EnhancedSignalGenerator(config, profit_optimizer)