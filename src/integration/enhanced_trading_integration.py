"""
Enhanced Trading System Integration

Integrates model monitoring, risk management, and drawdown protection
with the existing trading system for comprehensive enterprise-grade trading.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import our new components
from ..ensemble.trading_ensemble import TradingEnsemble
from ..monitoring import (
    ABTestingFramework,
    DriftDetector,
    ModelPerformanceMonitor,
    PerformanceTracker,
)
from ..risk_management import (
    DrawdownProtector,
    DynamicPositionSizer,
    PortfolioRiskManager,
    RiskCalculator,
)
from ..risk_management.portfolio_manager import PortfolioConstraints
from ..risk_management.position_sizer import SizingMethod

# Import existing components (assume these exist)
try:
    from ..data_pipeline.feature_engine import FeatureEngine
    from ..notifier.enhanced_telegram import EnhancedTelegram
    from ..trading.enhanced_signal_generator import EnhancedSignalGenerator
except ImportError:
    # Mock classes if they don't exist yet
    class EnhancedSignalGenerator:
        pass

    class FeatureEngine:
        pass

    class EnhancedTelegram:
        pass


@dataclass
class TradingDecision:
    """Enhanced trading decision with risk and monitoring metadata"""

    # Basic decision info
    asset: str
    action: str  # 'buy', 'sell', 'hold'
    recommended_size: float
    confidence_score: float

    # Model predictions
    model_predictions: Dict[str, float]
    ensemble_prediction: float

    # Risk assessment
    position_risk: Dict[str, Any]
    portfolio_risk_impact: Dict[str, Any]
    var_impact: float

    # Monitoring metadata
    drift_status: str
    model_health: str
    ab_test_variant: Optional[str] = None

    # Protection constraints
    protection_constraints: List[str] = field(default_factory=list)
    risk_adjusted_size: float = 0.0
    final_size: float = 0.0

    # Execution metadata
    timestamp: datetime = field(default_factory=datetime.now)
    execution_approved: bool = False
    execution_reason: str = ""


class EnhancedTradingSystem:
    """Comprehensive enhanced trading system with monitoring and risk management"""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        data_dir: Path = Path("data"),
        models_dir: Path = Path("models"),
        initial_portfolio_value: float = 100000.0,
    ):
        """
        Initialize enhanced trading system

        Args:
            config_path: Path to trading configuration file
            data_dir: Directory for storing data and logs
            models_dir: Directory containing trained models
            initial_portfolio_value: Initial portfolio value
        """

        self.config_path = config_path
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.initial_portfolio_value = initial_portfolio_value

        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self._initialize_components()

        # Trading state
        self.current_positions = {}
        self.portfolio_value = initial_portfolio_value
        self.trading_session_active = False

        # Performance tracking
        self.trade_history = []
        self.daily_pnl = []

        # Portfolio context for ensemble-based models
        self.last_prices: Dict[str, float] = {}
        self.unrealized_pnls: Dict[str, float] = {}
        self.trading_ensemble = TradingEnsemble()

        self.logger.info("Enhanced Trading System initialized")

    def _initialize_components(self):
        """Initialize all trading system components"""

        # Model Performance Monitoring
        self.model_monitor = ModelPerformanceMonitor(
            data_dir=self.data_dir, monitoring_interval_minutes=60
        )

        # Risk Management
        self.risk_calculator = RiskCalculator(lookback_days=252)

        self.position_sizer = DynamicPositionSizer(
            base_position_size=0.05,  # 5%
            max_position_size=0.20,  # 20%
            target_volatility=0.15,  # 15%
        )

        portfolio_constraints = PortfolioConstraints(
            max_single_position=0.20, max_portfolio_var_95=0.03, max_drawdown_limit=0.15
        )

        self.portfolio_manager = PortfolioRiskManager(constraints=portfolio_constraints)

        # Drawdown Protection
        self.drawdown_protector = DrawdownProtector(
            portfolio_high_water_mark=self.initial_portfolio_value,
            notification_callback=self._send_risk_alert,
        )

        # A/B Testing Framework
        self.ab_testing = ABTestingFramework(db_path=self.data_dir / "ab_testing.db")

        # Legacy components (if available)
        try:
            self.signal_generator = EnhancedSignalGenerator()
            self.feature_engine = FeatureEngine()
            self.telegram_notifier = EnhancedTelegram()
        except Exception as e:
            self.logger.warning(f"Some legacy components not available: {e}")
            self.signal_generator = None
            self.feature_engine = None
            self.telegram_notifier = None

    def start_trading_session(
        self, symbols: List[str], enable_live_trading: bool = False
    ) -> Dict[str, Any]:
        """
        Start an enhanced trading session

        Args:
            symbols: List of trading symbols
            enable_live_trading: Whether to enable live trading (vs paper trading)

        Returns:
            Session startup status
        """

        self.logger.info(f"Starting enhanced trading session for symbols: {symbols}")

        try:
            # 1. Start model monitoring
            self.model_monitor.start_automated_monitoring()

            # 2. Initialize portfolio tracking
            self.drawdown_protector.update_portfolio_value(self.portfolio_value)

            # 3. Load historical data and set reference baselines
            for symbol in symbols:
                try:
                    # This would load historical data - mock for now
                    historical_returns = np.random.randn(252) * 0.02  # Mock daily returns
                    historical_features = pd.DataFrame(
                        np.random.randn(252, 20),  # Mock features
                        columns=[f"feature_{i}" for i in range(20)],
                    )
                    historical_predictions = np.random.randn(252) * 0.01

                    # Set reference data for monitoring
                    self.model_monitor.set_model_reference_data(
                        model_name=f"ensemble_{symbol}",
                        features_df=historical_features,
                        predictions=historical_predictions,
                        targets=historical_returns,
                    )

                    self.logger.info(f"Reference data set for {symbol}")

                except Exception as e:
                    self.logger.error(f"Error setting reference data for {symbol}: {e}")

            # 4. Mark session as active
            self.trading_session_active = True

            return {
                "status": "success",
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "symbols": symbols,
                "live_trading": enable_live_trading,
                "components_active": {
                    "model_monitoring": self.model_monitor.is_monitoring,
                    "risk_management": True,
                    "drawdown_protection": True,
                    "ab_testing": True,
                },
                "message": "Enhanced trading session started successfully",
            }

        except Exception as e:
            self.logger.error(f"Error starting trading session: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to start trading session",
            }

    async def process_trading_opportunity(
        self, symbol: str, market_data: Dict[str, Any], user_id: Optional[str] = None
    ) -> TradingDecision:
        """
        Process a trading opportunity with full monitoring and risk management

        Args:
            symbol: Trading symbol
            market_data: Current market data
            user_id: User ID for A/B testing (optional)

        Returns:
            Comprehensive trading decision
        """

        self.logger.debug(f"Processing trading opportunity for {symbol}")

        try:
            # 1. Feature Engineering (mock implementation)
            features_df = self._extract_features(market_data)

            latest_price = None
            for price_key in ("price", "close", "last_price"):
                if price_key in market_data:
                    try:
                        latest_price = float(market_data[price_key])
                        break
                    except (TypeError, ValueError):
                        continue

            if latest_price is not None:
                self.last_prices[symbol] = latest_price

            portfolio_state = self._build_portfolio_state(symbol, latest_price)
            try:
                self.trading_ensemble.update_portfolio_state(
                    balance=portfolio_state["balance"],
                    positions=portfolio_state["positions"],
                    last_prices=portfolio_state["last_prices"],
                    unrealized=portfolio_state["unrealized_pnl"],
                )
            except Exception as ensemble_state_error:
                self.logger.debug(
                    "Failed to update ensemble portfolio state for %s: %s",
                    symbol,
                    ensemble_state_error,
                )

            # 2. Model Predictions (mock implementation)
            model_predictions = await self._get_model_predictions(symbol, features_df)

            ensemble_prediction_override: Optional[float] = None
            if self.trading_ensemble and getattr(self.trading_ensemble, "models", {}):
                try:
                    ensemble_output = self.trading_ensemble.predict(
                        features_df,
                        prices=None,
                        update_weights=True,
                        symbol=symbol,
                        portfolio_state=portfolio_state,
                    )
                    ensemble_array = np.atleast_1d(np.asarray(ensemble_output))
                    if ensemble_array.size > 0:
                        ensemble_prediction_override = float(ensemble_array[-1])
                        model_predictions["ensemble_model"] = ensemble_prediction_override
                except Exception as ensemble_error:
                    self.logger.debug(
                        "Ensemble prediction unavailable for %s: %s", symbol, ensemble_error
                    )

            base_values = list(model_predictions.values())
            ensemble_prediction = (
                float(np.mean(base_values)) if base_values else 0.0
            )
            if ensemble_prediction_override is not None:
                ensemble_prediction = ensemble_prediction_override

            # 3. Model Health Check and Monitoring
            monitoring_result = self.model_monitor.monitor_model_prediction(
                model_name=f"ensemble_{symbol}",
                features_df=features_df,
                predictions=np.array([ensemble_prediction]),
                user_id=user_id,
            )

            # 4. Risk Assessment
            position_risk = await self._assess_position_risk(symbol, market_data)

            # 5. Position Sizing
            sizing_result = self.position_sizer.calculate_position_size(
                asset_returns=position_risk["historical_returns"],
                predicted_return=ensemble_prediction,
                current_portfolio_value=self.portfolio_value,
                current_positions=self.current_positions,
                sizing_method=SizingMethod.ADAPTIVE_VOLATILITY,
            )

            # 6. Portfolio Risk Impact
            portfolio_impact = await self._assess_portfolio_impact(
                symbol, sizing_result.recommended_size
            )

            # 7. Drawdown Protection Check
            trading_allowed = self.drawdown_protector.check_trading_allowed(
                trade_type="new_position",
                trade_size=sizing_result.recommended_size * self.portfolio_value,
            )

            # 8. Determine Trading Action
            trading_action = self._determine_trading_action(
                ensemble_prediction, monitoring_result, trading_allowed
            )

            # 9. Create Trading Decision
            decision = TradingDecision(
                asset=symbol,
                action=trading_action["action"],
                recommended_size=sizing_result.recommended_size,
                confidence_score=sizing_result.confidence_score,
                model_predictions=model_predictions,
                ensemble_prediction=ensemble_prediction,
                position_risk=position_risk,
                portfolio_risk_impact=portfolio_impact,
                var_impact=portfolio_impact.get("var_impact", 0.0),
                drift_status=monitoring_result.get("status", "unknown"),
                model_health=monitoring_result.get("status", "unknown"),
                ab_test_variant=self._get_ab_test_variant(user_id, symbol),
                protection_constraints=(
                    trading_allowed.get("reason", "").split(", ")
                    if not trading_allowed["allowed"]
                    else []
                ),
                risk_adjusted_size=sizing_result.risk_adjusted_size,
                final_size=(sizing_result.recommended_size if trading_allowed["allowed"] else 0.0),
                execution_approved=trading_allowed["allowed"] and trading_action["execute"],
                execution_reason=trading_action["reason"],
            )

            # 10. Log Decision
            self.logger.info(
                f"Trading decision for {symbol}: {decision.action} "
                f"(size: {decision.final_size:.3f}, confidence: {decision.confidence_score:.2f})"
            )

            return decision

        except Exception as e:
            self.logger.error(f"Error processing trading opportunity for {symbol}: {e}")

            # Return safe default decision
            return TradingDecision(
                asset=symbol,
                action="hold",
                recommended_size=0.0,
                confidence_score=0.0,
                model_predictions={},
                ensemble_prediction=0.0,
                position_risk={"error": str(e)},
                portfolio_risk_impact={"error": str(e)},
                var_impact=0.0,
                drift_status="error",
                model_health="error",
                protection_constraints=[f"Error: {str(e)}"],
                execution_approved=False,
                execution_reason=f"Error in decision process: {str(e)}",
            )

    def execute_trading_decision(self, decision: TradingDecision) -> Dict[str, Any]:
        """
        Execute a trading decision with full audit trail

        Args:
            decision: Trading decision to execute

        Returns:
            Execution result
        """

        if not decision.execution_approved:
            return {
                "status": "rejected",
                "reason": decision.execution_reason,
                "decision_id": id(decision),
            }

        try:
            # This would connect to actual broker/exchange
            # For now, simulate execution
            execution_result = self._simulate_trade_execution(decision)

            if execution_result["status"] == "filled":
                # Update portfolio state
                self._update_portfolio_state(decision, execution_result)

                # Update drawdown protection
                self.drawdown_protector.update_portfolio_value(self.portfolio_value)

                # Record trade
                self.trade_history.append(
                    {
                        "timestamp": decision.timestamp,
                        "symbol": decision.asset,
                        "action": decision.action,
                        "size": decision.final_size,
                        "price": execution_result.get("execution_price", 0),
                        "value": execution_result.get("execution_value", 0),
                        "decision_metadata": decision.__dict__,
                    }
                )

                # Send notifications
                if self.telegram_notifier:
                    self._send_trade_notification(decision, execution_result)

            return execution_result

        except Exception as e:
            self.logger.error(f"Error executing trade for {decision.asset}: {e}")
            return {"status": "error", "error": str(e), "decision_id": id(decision)}

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        # Model monitoring status
        monitoring_status = self.model_monitor.get_monitoring_dashboard_data()

        # Risk management status
        portfolio_analysis = None
        if self.current_positions:
            # Mock historical returns for portfolio analysis
            asset_returns = {
                asset: np.random.randn(252) * 0.02 for asset in self.current_positions.keys()
            }
            portfolio_analysis = self.portfolio_manager.analyze_portfolio_risk(
                positions=self.current_positions,
                asset_returns=asset_returns,
                portfolio_value=self.portfolio_value,
            )

        # Drawdown protection status
        protection_status = self.drawdown_protector.get_protection_status()

        # A/B testing status
        active_ab_tests = self.ab_testing.list_tests(status_filter="running")

        return {
            "timestamp": datetime.now(),
            "session_active": self.trading_session_active,
            "portfolio_value": self.portfolio_value,
            "positions_count": len(self.current_positions),
            "trades_today": len(
                [t for t in self.trade_history if t["timestamp"].date() == datetime.now().date()]
            ),
            "monitoring": {
                "status": monitoring_status["current_status"]["overall_status"],
                "confidence": monitoring_status["current_status"]["confidence_score"],
                "active_monitoring": monitoring_status["monitoring_active"],
            },
            "risk_management": {
                "portfolio_risk_status": (
                    portfolio_analysis["overall_risk_status"]
                    if portfolio_analysis
                    else "no_positions"
                ),
                "risk_limit_breaches": len(
                    [
                        l
                        for l in (portfolio_analysis or {}).get("risk_limits", {}).values()
                        if l.is_breached
                    ]
                ),
            },
            "drawdown_protection": {
                "protection_active": protection_status["protection_active"],
                "trading_suspended": protection_status["trading_suspended"],
                "current_drawdown": protection_status["current_drawdown_pct"],
                "emergency_stop": protection_status["emergency_stop_active"],
            },
            "ab_testing": {"active_tests": len(active_ab_tests)},
        }

    def _extract_features(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract features from market data (mock implementation)"""

        # Mock feature extraction - in reality this would use FeatureEngine
        features = np.random.randn(1, 20)  # 1 row, 20 features
        feature_names = [f"feature_{i}" for i in range(20)]

        return pd.DataFrame(features, columns=feature_names)

    async def _get_model_predictions(
        self, symbol: str, features_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Get predictions from all models (mock implementation)"""

        # Mock model predictions
        predictions = {
            "gru_model": np.random.randn() * 0.02,
            "lightgbm_model": np.random.randn() * 0.02,
            "ppo_model": np.random.randn() * 0.02,
        }

        return predictions

    async def _assess_position_risk(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risk for individual position"""

        # Mock historical returns
        historical_returns = np.random.randn(252) * 0.02

        # Calculate risk metrics
        risk_metrics = self.risk_calculator.calculate_comprehensive_risk(historical_returns)

        return {
            "historical_returns": historical_returns,
            "var_95": risk_metrics.var_1d_95,
            "volatility": risk_metrics.realized_volatility,
            "max_drawdown": risk_metrics.max_drawdown,
            "sharpe_ratio": risk_metrics.sharpe_ratio or 0.0,
        }

    async def _assess_portfolio_impact(self, symbol: str, position_size: float) -> Dict[str, Any]:
        """Assess impact of new position on portfolio"""

        if not self.current_positions:
            return {
                "var_impact": position_size * 0.02,  # Mock VaR impact
                "correlation_impact": 0.0,
                "concentration_impact": position_size,
            }

        # Mock portfolio impact calculation
        return {
            "var_impact": position_size * 0.03,
            "correlation_impact": np.random.random() * 0.5,
            "concentration_impact": position_size,
        }

    def _determine_trading_action(
        self,
        prediction: float,
        monitoring_result: Dict[str, Any],
        trading_allowed: Dict[str, bool],
    ) -> Dict[str, Any]:
        """Determine final trading action based on all inputs"""

        if not trading_allowed["allowed"]:
            return {
                "action": "hold",
                "execute": False,
                "reason": trading_allowed["reason"],
            }

        # Check model health
        if monitoring_result.get("status") in ["critical", "error"]:
            return {
                "action": "hold",
                "execute": False,
                "reason": "Model health issues detected",
            }

        # Determine action based on prediction
        if abs(prediction) < 0.005:  # Less than 0.5% predicted move
            return {
                "action": "hold",
                "execute": False,
                "reason": "Prediction signal too weak",
            }
        elif prediction > 0.005:
            return {
                "action": "buy",
                "execute": True,
                "reason": f"Positive prediction: {prediction:.3f}",
            }
        else:
            return {
                "action": "sell",
                "execute": True,
                "reason": f"Negative prediction: {prediction:.3f}",
            }

    def _get_ab_test_variant(self, user_id: Optional[str], symbol: str) -> Optional[str]:
        """Get A/B test variant for user/symbol"""

        if not user_id:
            return None

        # Check if there are active A/B tests for this model
        active_tests = self.ab_testing.list_tests(status_filter="running")

        for test in active_tests:
            if f"ensemble_{symbol}" in [
                test["details"]["control_model"],
                test["details"]["variant_model"],
            ]:
                variant = self.ab_testing.assign_variant(test["test_id"], user_id)
                return variant

        return None

    def _simulate_trade_execution(self, decision: TradingDecision) -> Dict[str, Any]:
        """Simulate trade execution (replace with real broker integration)"""

        # Mock execution
        execution_price = 100.0 + np.random.randn() * 0.1  # Mock price
        execution_value = decision.final_size * self.portfolio_value

        return {
            "status": "filled",
            "execution_price": execution_price,
            "execution_value": execution_value,
            "execution_time": datetime.now(),
            "fees": execution_value * 0.001,  # 0.1% fee
        }

    def _update_portfolio_state(self, decision: TradingDecision, execution_result: Dict[str, Any]):
        """Update portfolio state after trade execution"""

        if decision.action in ["buy", "sell"]:
            executed_value = execution_result["execution_value"]
            fees = execution_result.get("fees", 0)

            if decision.action == "buy":
                self.current_positions[decision.asset] = (
                    self.current_positions.get(decision.asset, 0) + executed_value
                )
                self.portfolio_value -= executed_value + fees
            else:  # sell
                self.current_positions[decision.asset] = (
                    self.current_positions.get(decision.asset, 0) - executed_value
                )
                self.portfolio_value += executed_value - fees

            # Remove zero positions
            if abs(self.current_positions.get(decision.asset, 0)) < 0.01:
                self.current_positions.pop(decision.asset, None)

            # Reset unrealized PnL snapshot for the asset until next valuation cycle
            self.unrealized_pnls[decision.asset] = self.unrealized_pnls.get(decision.asset, 0.0)

        # Update ensemble portfolio snapshot
        try:
            portfolio_state = self._build_portfolio_state(decision.asset)
            self.trading_ensemble.update_portfolio_state(
                balance=portfolio_state["balance"],
                positions=portfolio_state["positions"],
                last_prices=portfolio_state["last_prices"],
                unrealized=portfolio_state["unrealized_pnl"],
            )
        except Exception as ensemble_state_error:
            self.logger.debug(
                "Failed to refresh ensemble portfolio state post-trade for %s: %s",
                decision.asset,
                ensemble_state_error,
            )

    def _build_portfolio_state(
        self, symbol: str, latest_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Assemble portfolio snapshot for ensemble models."""

        positions_snapshot = {k: float(v) for k, v in self.current_positions.items()}

        last_price_snapshot = self.last_prices.copy()
        if latest_price is not None:
            last_price_snapshot[symbol] = latest_price

        unrealized_snapshot = self.unrealized_pnls.copy()
        if symbol not in unrealized_snapshot:
            unrealized_snapshot[symbol] = unrealized_snapshot.get(symbol, 0.0)

        return {
            "balance": float(self.portfolio_value),
            "positions": positions_snapshot,
            "last_prices": last_price_snapshot,
            "unrealized_pnl": unrealized_snapshot,
        }

    def _send_trade_notification(self, decision: TradingDecision, execution_result: Dict[str, Any]):
        """Send trade notification via Telegram"""

        message = (
            f"🔄 Trade Executed\\n"
            f"Symbol: {decision.asset}\\n"
            f"Action: {decision.action.upper()}\\n"
            f"Size: ${execution_result['execution_value']:,.0f}\\n"
            f"Price: ${execution_result['execution_price']:.2f}\\n"
            f"Confidence: {decision.confidence_score:.1%}\\n"
            f"Model Health: {decision.model_health}"
        )

        # This would send via actual Telegram notifier
        self.logger.info(f"Trade notification: {message}")

    def _send_risk_alert(self, message: str, severity: str):
        """Send risk alert notification"""

        self.logger.warning(f"Risk Alert ({severity}): {message}")

        # This would send via actual notification system

    def stop_trading_session(self) -> Dict[str, Any]:
        """Stop the trading session gracefully"""

        try:
            # Stop model monitoring
            self.model_monitor.stop_automated_monitoring()

            # Generate final reports
            final_report = self.model_monitor.generate_comprehensive_report()
            self.model_monitor.save_report(final_report)

            # Mark session as inactive
            self.trading_session_active = False

            self.logger.info("Trading session stopped successfully")

            return {
                "status": "success",
                "message": "Trading session stopped successfully",
                "final_portfolio_value": self.portfolio_value,
                "total_trades": len(self.trade_history),
                "session_pnl": self.portfolio_value - self.initial_portfolio_value,
            }

        except Exception as e:
            self.logger.error(f"Error stopping trading session: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Error stopping trading session",
            }
