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
from dataclasses import dataclass
from datetime import datetime
import logging

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
        """Initialize enhanced signal generator."""
        self.config = config
        self.profit_optimizer = profit_optimizer

        # Signal generation parameters
        signal_config = config.get("signal_generation", {})
        self.confidence_threshold = signal_config.get("confidence_threshold", 0.45)
        self.base_model_weights = signal_config.get(
            "model_weights", {"lightgbm": 0.55, "gru": 0.35, "ppo": 0.1}
        )
        self.model_weights = self.base_model_weights.copy()  # Dynamic weights
        self.ensemble_method = signal_config.get("ensemble_method", "weighted_average")
        self.regime_detection = signal_config.get("regime_detection", True)

        # Performance tracking for adaptive weighting
        self.model_performance_history = {}
        self.performance_window = 50  # Track last 50 predictions per model
        self.weight_adaptation_enabled = signal_config.get("adaptive_weights", True)
        self.weight_adaptation_rate = signal_config.get("weight_adaptation_rate", 0.1)

        # New aggressive thresholds
        self.base_buy_threshold = signal_config.get("base_buy_threshold", 0.0003)
        self.base_sell_threshold = signal_config.get("base_sell_threshold", -0.0003)
        self.volatility_threshold_multiplier = signal_config.get(
            "volatility_threshold_multiplier", 1.5
        )

        # Risk management
        risk_config = config.get("risk_management", {})
        self.max_position_pct = risk_config.get("max_position_pct", 0.15)
        self.stop_loss_pct = risk_config.get("stop_loss_pct", 0.05)
        self.take_profit_pct = risk_config.get("take_profit_pct", 0.10)

        # Market analysis parameters
        self.volatility_lookback = config.get("volatility_lookback", 20)
        self.momentum_lookback = config.get("momentum_lookback", 10)
        self.volume_lookback = config.get("volume_lookback", 20)

        logger.info(
            f"Enhanced signal generator initialized with confidence threshold: {self.confidence_threshold}"
        )
        logger.info(
            f"Adaptive model weighting: {'enabled' if self.weight_adaptation_enabled else 'disabled'}"
        )

    def generate_enhanced_signals(
        self,
        model_predictions: Dict[str, List[ModelPrediction]],
        market_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        current_balance: float,
    ) -> Dict[str, TradeSignal]:
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
                model_predictions,
                market_data,
                current_positions,
                current_prices,
                current_balance,
            )

            # 3. Merge signals with priority to risk management
            final_signals = self._merge_signals(all_signals, model_signals)

            # 4. Apply portfolio optimization filters
            optimized_signals = self._apply_portfolio_optimization(
                final_signals,
                current_positions,
                current_prices,
                current_balance,
                market_data,
            )

            logger.info(f"Generated {len(optimized_signals)} enhanced signals")
            return optimized_signals

        except Exception as e:
            logger.error(f"Failed to generate enhanced signals: {e}")
            return {}

    def _generate_risk_management_signals(
        self,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        current_balance: float,
    ) -> Dict[str, TradeSignal]:
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

    def _generate_model_based_signals(
        self,
        model_predictions: Dict[str, List[ModelPrediction]],
        market_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        current_balance: float,
    ) -> Dict[str, TradeSignal]:
        """Generate signals based on model predictions."""
        signals = {}

        for symbol, predictions in model_predictions.items():
            try:
                if not predictions or symbol not in market_data:
                    continue

                # Analyze market context
                market_context = self._analyze_market_context(
                    symbol, market_data[symbol]
                )

                # Combine model predictions
                ensemble_prediction = self._combine_model_predictions(predictions)

                # Get dynamic thresholds
                thresholds = self.profit_optimizer.calculate_dynamic_thresholds(
                    symbol,
                    market_data[symbol],
                    self.config.get("thresholds", {}).get("default", 0.0005),
                )

                # Generate signal
                signal = self._generate_signal_from_prediction(
                    symbol,
                    ensemble_prediction,
                    thresholds,
                    market_context,
                    current_positions,
                    current_prices,
                    current_balance,
                    market_data[symbol],
                )

                if signal:
                    signals[symbol] = signal

            except Exception as e:
                logger.error(f"Failed to generate model signal for {symbol}: {e}")

        return signals

    def _analyze_market_context(
        self, symbol: str, market_data: pd.DataFrame
    ) -> MarketContext:
        """Analyze market context for enhanced signal generation."""
        try:
            if len(market_data) < 20:
                return MarketContext(symbol, 0.02, 0.0, 1.0, "sideways")

            # Calculate volatility
            returns = market_data["close"].pct_change().dropna()
            volatility = returns.tail(self.volatility_lookback).std() * np.sqrt(24)

            # Calculate momentum
            momentum = (
                market_data["close"].iloc[-1]
                / market_data["close"].iloc[-self.momentum_lookback]
                - 1
            )

            # Calculate volume trend
            recent_volume = market_data["volume"].tail(5).mean()
            baseline_volume = market_data["volume"].tail(self.volume_lookback).mean()
            volume_trend = (
                recent_volume / baseline_volume if baseline_volume > 0 else 1.0
            )

            # Determine price trend
            sma_short = market_data["close"].tail(5).mean()
            sma_long = market_data["close"].tail(20).mean()

            if sma_short > sma_long * 1.005:
                price_trend = "bullish"
            elif sma_short < sma_long * 0.995:
                price_trend = "bearish"
            else:
                price_trend = "sideways"

            # Calculate support and resistance levels
            recent_data = market_data.tail(50)
            resistance_level = recent_data["high"].max()
            support_level = recent_data["low"].min()

            return MarketContext(
                symbol=symbol,
                volatility=volatility,
                momentum=momentum,
                volume_trend=volume_trend,
                price_trend=price_trend,
                support_level=support_level,
                resistance_level=resistance_level,
            )

        except Exception as e:
            logger.error(f"Failed to analyze market context for {symbol}: {e}")
            return MarketContext(symbol, 0.02, 0.0, 1.0, "sideways")

    def _combine_model_predictions(
        self, predictions: List[ModelPrediction]
    ) -> ModelPrediction:
        """Combine multiple model predictions into ensemble prediction."""
        try:
            if not predictions:
                return ModelPrediction("ensemble", 0.0, 0.0, 0)

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
                return ModelPrediction("ensemble", 0.0, 0.0, 0)

            ensemble_prediction = weighted_sum / total_weight
            ensemble_confidence = confidence_sum / sum(
                self.model_weights.get(p.model_type, 1.0) for p in predictions
            )

            logger.debug(
                f"Ensemble prediction: {ensemble_prediction:.6f} (confidence: {ensemble_confidence:.3f})"
            )

            return ModelPrediction(
                model_type="ensemble",
                prediction=ensemble_prediction,
                confidence=ensemble_confidence,
                features_used=total_features,
            )

        except Exception as e:
            logger.error(f"Failed to combine model predictions: {e}")
            return ModelPrediction("ensemble", 0.0, 0.0, 0)

    def update_model_performance(
        self,
        symbol: str,
        model_type: str,
        prediction: float,
        actual_return: float,
        trade_profitable: bool,
    ) -> None:
        """Update model performance tracking for adaptive weighting."""
        try:
            if not self.weight_adaptation_enabled:
                return

            key = f"{symbol}_{model_type}"
            if key not in self.model_performance_history:
                self.model_performance_history[key] = {
                    "predictions": [],
                    "actuals": [],
                    "profitable_trades": [],
                    "accuracy_score": 0.0,
                    "profit_rate": 0.0,
                    "last_updated": datetime.now(),
                }

            history = self.model_performance_history[key]

            # Add new performance data
            history["predictions"].append(prediction)
            history["actuals"].append(actual_return)
            history["profitable_trades"].append(trade_profitable)

            # Keep only recent history
            if len(history["predictions"]) > self.performance_window:
                history["predictions"] = history["predictions"][
                    -self.performance_window :
                ]
                history["actuals"] = history["actuals"][-self.performance_window :]
                history["profitable_trades"] = history["profitable_trades"][
                    -self.performance_window :
                ]

            # Calculate performance metrics
            if (
                len(history["predictions"]) >= 10
            ):  # Minimum samples for reliable metrics
                # Directional accuracy
                pred_directions = [1 if p > 0 else -1 for p in history["predictions"]]
                actual_directions = [1 if a > 0 else -1 for a in history["actuals"]]
                correct_directions = sum(
                    1 for p, a in zip(pred_directions, actual_directions) if p == a
                )
                history["accuracy_score"] = correct_directions / len(pred_directions)

                # Profit rate
                history["profit_rate"] = sum(history["profitable_trades"]) / len(
                    history["profitable_trades"]
                )

                history["last_updated"] = datetime.now()

                # Update model weights if enough data
                if len(history["predictions"]) >= 20:
                    self._update_adaptive_weights()

        except Exception as e:
            logger.error(
                f"Failed to update model performance for {symbol}_{model_type}: {e}"
            )

    def _update_adaptive_weights(self) -> None:
        """Update model weights based on recent performance."""
        try:
            model_scores = {}

            # Calculate average performance per model type
            for key, history in self.model_performance_history.items():
                if len(history["predictions"]) < 20:
                    continue

                model_type = key.split("_")[-1]  # Extract model type from key
                if model_type not in model_scores:
                    model_scores[model_type] = {
                        "accuracy_scores": [],
                        "profit_rates": [],
                        "sample_counts": [],
                    }

                model_scores[model_type]["accuracy_scores"].append(
                    history["accuracy_score"]
                )
                model_scores[model_type]["profit_rates"].append(history["profit_rate"])
                model_scores[model_type]["sample_counts"].append(
                    len(history["predictions"])
                )

            if not model_scores:
                return

            # Calculate composite scores for each model
            composite_scores = {}
            for model_type, scores in model_scores.items():
                if not scores["accuracy_scores"]:
                    continue

                # Weighted average based on sample size
                total_samples = sum(scores["sample_counts"])
                weights = [count / total_samples for count in scores["sample_counts"]]

                avg_accuracy = sum(
                    acc * w for acc, w in zip(scores["accuracy_scores"], weights)
                )
                avg_profit_rate = sum(
                    pr * w for pr, w in zip(scores["profit_rates"], weights)
                )

                # Composite score: 60% accuracy, 40% profitability
                composite_scores[model_type] = (
                    0.6 * avg_accuracy + 0.4 * avg_profit_rate
                )

            if len(composite_scores) < 2:
                return  # Need at least 2 models to adjust weights

            # Normalize scores and adjust weights
            total_score = sum(composite_scores.values())
            if total_score > 0:
                new_weights = {}
                for model_type in self.base_model_weights.keys():
                    if model_type in composite_scores:
                        # Blend current weight with performance-based weight
                        performance_weight = composite_scores[model_type] / total_score
                        current_weight = self.model_weights.get(model_type, 0.0)

                        # Gradual adaptation
                        new_weight = (
                            (1 - self.weight_adaptation_rate) * current_weight
                            + self.weight_adaptation_rate * performance_weight
                        )
                        new_weights[model_type] = new_weight
                    else:
                        # Keep base weight if no performance data
                        new_weights[model_type] = self.base_model_weights[model_type]

                # Normalize weights to sum to 1
                total_weight = sum(new_weights.values())
                if total_weight > 0:
                    self.model_weights = {
                        k: v / total_weight for k, v in new_weights.items()
                    }

                    logger.info(f"Updated adaptive model weights: {self.model_weights}")
                    logger.info(f"Performance scores: {composite_scores}")

        except Exception as e:
            logger.error(f"Failed to update adaptive weights: {e}")

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance for monitoring."""
        try:
            summary = {
                "current_weights": self.model_weights.copy(),
                "base_weights": self.base_model_weights.copy(),
                "adaptation_enabled": self.weight_adaptation_enabled,
                "model_performance": {},
            }

            for key, history in self.model_performance_history.items():
                if len(history["predictions"]) >= 10:
                    summary["model_performance"][key] = {
                        "sample_count": len(history["predictions"]),
                        "accuracy_score": history["accuracy_score"],
                        "profit_rate": history["profit_rate"],
                        "last_updated": history["last_updated"].isoformat(),
                    }

            return summary

        except Exception as e:
            logger.error(f"Failed to get model performance summary: {e}")
            return {"error": str(e)}

    def _generate_signal_from_prediction(
        self,
        symbol: str,
        prediction: ModelPrediction,
        thresholds: Dict[str, float],
        market_context: MarketContext,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        current_balance: float,
        market_data: pd.DataFrame,
    ) -> Optional[TradeSignal]:
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

            # Get dynamic thresholds - more aggressive approach
            base_buy = thresholds.get("buy", self.base_buy_threshold)
            base_sell = thresholds.get("sell", self.base_sell_threshold)

            # Adjust thresholds based on market volatility
            volatility_adj = 1.0
            if market_context.volatility > 0.03:  # High volatility
                volatility_adj = self.volatility_threshold_multiplier
            elif market_context.volatility < 0.01:  # Low volatility
                volatility_adj = 0.7

            buy_threshold = base_buy * volatility_adj
            sell_threshold = base_sell * volatility_adj

            # Generate BUY signals - more aggressive conditions
            confidence_check = prediction.confidence > self.confidence_threshold

            # Lower confidence threshold for strong predictions
            if context_adjusted_prediction > buy_threshold * 2:
                confidence_check = prediction.confidence > (
                    self.confidence_threshold * 0.8
                )

            if context_adjusted_prediction > buy_threshold and confidence_check:
                # Check if we should buy (not over-concentrated)
                optimal_position_size = (
                    self.profit_optimizer.calculate_optimal_position_size(
                        symbol,
                        context_adjusted_prediction,
                        prediction.confidence,
                        current_balance,
                        current_positions,
                        current_prices,
                        market_data,
                    )
                )

                if optimal_position_size > 0.005:  # Reduced minimum to 0.5% position
                    return TradeSignal(
                        symbol=symbol,
                        action="BUY",
                        confidence=prediction.confidence,
                        quantity_pct=optimal_position_size,
                        reasoning=f"Ensemble prediction {context_adjusted_prediction:.6f} > {buy_threshold:.6f} "
                        f"(confidence: {prediction.confidence:.3f}, market: {market_context.price_trend})",
                        risk_score=1.0 - prediction.confidence,
                        expected_return=context_adjusted_prediction,
                    )

            # Generate SELL signals for existing positions - More aggressive selling
            elif current_position > 0:
                sell_quantity_pct = 0.0
                sell_reason = ""

                # Strong negative prediction - sell most of position
                if context_adjusted_prediction < sell_threshold:
                    sell_quantity_pct = 0.8
                    sell_reason = (
                        f"Strong negative prediction {context_adjusted_prediction:.6f}"
                    )

                # Moderate negative prediction - partial sell
                elif context_adjusted_prediction < sell_threshold * 0.5:
                    sell_quantity_pct = 0.5
                    sell_reason = f"Moderate negative prediction {context_adjusted_prediction:.6f}"

                # Weak positive in bearish market - reduce exposure
                elif (
                    market_context.price_trend == "bearish"
                    and 0 < context_adjusted_prediction < buy_threshold * 0.3
                ):
                    sell_quantity_pct = 0.4
                    sell_reason = f"Weak signal in bearish market {context_adjusted_prediction:.6f}"

                # Low confidence prediction - reduce position
                elif prediction.confidence < 0.5:
                    sell_quantity_pct = 0.3
                    sell_reason = f"Low confidence {prediction.confidence:.3f}"

                # High volatility with neutral prediction - take profits
                elif (
                    market_context.volatility > 0.04
                    and abs(context_adjusted_prediction) < buy_threshold * 0.5
                ):
                    sell_quantity_pct = 0.25
                    sell_reason = f"High volatility with neutral signal (vol: {market_context.volatility:.3f})"

                # Stagnant prediction - partial exit
                elif abs(context_adjusted_prediction) < buy_threshold * 0.2:
                    sell_quantity_pct = 0.2
                    sell_reason = (
                        f"Stagnant prediction {context_adjusted_prediction:.6f}"
                    )

                if sell_quantity_pct > 0:
                    return TradeSignal(
                        symbol=symbol,
                        action="SELL",
                        confidence=max(0.6, 1.0 - prediction.confidence),
                        quantity_pct=sell_quantity_pct,
                        reasoning=sell_reason,
                        risk_score=prediction.confidence,
                        expected_return=context_adjusted_prediction,
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to generate signal from prediction for {symbol}: {e}")
            return None

    def _apply_market_context_adjustment(
        self, prediction: float, context: MarketContext
    ) -> float:
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
            if context.price_trend == "bullish" and prediction > 0:
                adjusted_prediction *= 1.15
            elif context.price_trend == "bearish" and prediction < 0:
                adjusted_prediction *= 1.15
            elif context.price_trend == "bullish" and prediction < 0:
                adjusted_prediction *= 0.85
            elif context.price_trend == "bearish" and prediction > 0:
                adjusted_prediction *= 0.85

            logger.debug(
                f"Market context adjustment: {prediction:.6f} -> {adjusted_prediction:.6f} "
                f"(trend: {context.price_trend}, vol: {context.volatility:.3f})"
            )

            return adjusted_prediction

        except Exception as e:
            logger.error(f"Failed to apply market context adjustment: {e}")
            return prediction

    def _merge_signals(
        self,
        priority_signals: Dict[str, TradeSignal],
        secondary_signals: Dict[str, TradeSignal],
    ) -> Dict[str, TradeSignal]:
        """Merge signals with priority to the first set."""
        merged = priority_signals.copy()

        for symbol, signal in secondary_signals.items():
            if symbol not in merged:
                merged[symbol] = signal
            else:
                # Keep priority signal but log conflict
                logger.debug(
                    f"Signal conflict for {symbol}: keeping priority signal "
                    f"{merged[symbol].action} over {signal.action}"
                )

        return merged

    def _apply_portfolio_optimization(
        self,
        signals: Dict[str, TradeSignal],
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        current_balance: float,
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, TradeSignal]:
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
                reverse=True,
            )

            total_portfolio_value = current_balance + sum(
                current_positions.get(s, 0.0) * current_prices.get(s, 0.0)
                for s in current_positions.keys()
            )

            current_cash_pct = (
                current_balance / total_portfolio_value
                if total_portfolio_value > 0
                else 1.0
            )

            for symbol, signal in sorted_signals:
                # Always allow sell signals
                if signal.action == "SELL":
                    optimized_signals[symbol] = signal
                    continue

                # For buy signals, apply portfolio constraints
                if signal.action == "BUY":
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

            logger.debug(
                f"Portfolio optimization: {len(signals)} -> {len(optimized_signals)} signals"
            )
            return optimized_signals

        except Exception as e:
            logger.error(f"Failed to apply portfolio optimization: {e}")
            return signals


def create_enhanced_signal_generator(config: Dict[str, Any]) -> EnhancedSignalGenerator:
    """Factory function to create an enhanced signal generator."""
    profit_optimizer = ProfitOptimizer(config.get("profit_optimization", {}))
    return EnhancedSignalGenerator(config, profit_optimizer)
