"""
Drawdown Protection System

Implements automatic trading suspension, position reduction, and recovery
protocols based on portfolio drawdown levels and risk metrics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .risk_calculator import RiskCalculator, RiskMetrics


class DrawdownLevel(Enum):
    """Drawdown severity levels"""

    NORMAL = "normal"  # < 5% drawdown
    ELEVATED = "elevated"  # 5-10% drawdown
    HIGH = "high"  # 10-15% drawdown
    SEVERE = "severe"  # 15-25% drawdown
    CRITICAL = "critical"  # > 25% drawdown


class ProtectionAction(Enum):
    """Protection actions to take"""

    MONITOR = "monitor"
    REDUCE_POSITIONS = "reduce_positions"
    SUSPEND_NEW_TRADES = "suspend_new_trades"
    PARTIAL_LIQUIDATION = "partial_liquidation"
    FULL_LIQUIDATION = "full_liquidation"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class DrawdownEvent:
    """Record of a drawdown event"""

    start_date: datetime
    end_date: Optional[datetime]
    peak_value: float
    trough_value: float
    max_drawdown_pct: float
    duration_days: int
    recovery_date: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class ProtectionRule:
    """Drawdown protection rule"""

    drawdown_threshold: float  # Drawdown % that triggers this rule
    action: ProtectionAction  # Action to take
    position_reduction_pct: float = 0.0  # % to reduce positions (if applicable)
    max_daily_trades: int = 0  # Max trades per day (0 = no limit)
    min_recovery_days: int = 1  # Days before allowing recovery
    description: str = ""
    enabled: bool = True


class DrawdownProtector:
    """Comprehensive drawdown protection system"""

    def __init__(
        self,
        protection_rules: Optional[List[ProtectionRule]] = None,
        portfolio_high_water_mark: Optional[float] = None,
        notification_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize drawdown protection system

        Args:
            protection_rules: List of protection rules (default rules if None)
            portfolio_high_water_mark: Initial portfolio high water mark
            notification_callback: Function to call for alerts (message, severity)
        """

        self.protection_rules = protection_rules or self._create_default_protection_rules()
        self.portfolio_high_water_mark = portfolio_high_water_mark
        self.notification_callback = notification_callback

        self.logger = logging.getLogger(__name__)

        # State tracking
        self.current_drawdown_event = None
        self.drawdown_history = []
        self.active_protections = set()
        self.last_protection_check = None

        # Protection state
        self.trading_suspended = False
        self.position_reduction_active = False
        self.emergency_stop_active = False

        # Recovery tracking
        self.recovery_start_date = None
        self.recovery_conditions = {}

    def _create_default_protection_rules(self) -> List[ProtectionRule]:
        """Create default protection rules"""

        return [
            ProtectionRule(
                drawdown_threshold=0.05,  # 5%
                action=ProtectionAction.MONITOR,
                description="Monitor closely at 5% drawdown",
            ),
            ProtectionRule(
                drawdown_threshold=0.08,  # 8%
                action=ProtectionAction.REDUCE_POSITIONS,
                position_reduction_pct=0.25,  # Reduce positions by 25%
                description="Reduce positions by 25% at 8% drawdown",
            ),
            ProtectionRule(
                drawdown_threshold=0.12,  # 12%
                action=ProtectionAction.SUSPEND_NEW_TRADES,
                max_daily_trades=0,
                min_recovery_days=2,
                description="Suspend new trades at 12% drawdown",
            ),
            ProtectionRule(
                drawdown_threshold=0.18,  # 18%
                action=ProtectionAction.PARTIAL_LIQUIDATION,
                position_reduction_pct=0.50,  # Reduce by 50%
                min_recovery_days=5,
                description="Partial liquidation (50%) at 18% drawdown",
            ),
            ProtectionRule(
                drawdown_threshold=0.25,  # 25%
                action=ProtectionAction.EMERGENCY_STOP,
                min_recovery_days=10,
                description="Emergency stop at 25% drawdown",
            ),
        ]

    def update_portfolio_value(
        self, current_value: float, timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update portfolio value and check for drawdown protection triggers

        Args:
            current_value: Current portfolio value
            timestamp: Timestamp of value update (default: now)

        Returns:
            Dictionary with protection status and actions taken
        """

        if timestamp is None:
            timestamp = datetime.now()

        self.last_protection_check = timestamp

        # Update high water mark
        if self.portfolio_high_water_mark is None or current_value > self.portfolio_high_water_mark:
            if self.portfolio_high_water_mark is not None:
                self.logger.info(f"New portfolio high water mark: ${current_value:,.2f}")

                # Check if we're recovering from drawdown
                if self.current_drawdown_event and self.current_drawdown_event.is_active:
                    self._handle_drawdown_recovery(current_value, timestamp)

            self.portfolio_high_water_mark = current_value

        # Calculate current drawdown
        current_drawdown_pct = (
            self.portfolio_high_water_mark - current_value
        ) / self.portfolio_high_water_mark

        # Determine drawdown level
        drawdown_level = self._classify_drawdown_level(current_drawdown_pct)

        # Update or create drawdown event
        if current_drawdown_pct > 0.01:  # More than 1% drawdown
            if not self.current_drawdown_event or not self.current_drawdown_event.is_active:
                # Start new drawdown event
                self.current_drawdown_event = DrawdownEvent(
                    start_date=timestamp,
                    end_date=None,
                    peak_value=self.portfolio_high_water_mark,
                    trough_value=current_value,
                    max_drawdown_pct=current_drawdown_pct,
                    duration_days=0,
                )
                self.logger.warning(f"New drawdown event started: {current_drawdown_pct:.2%}")
            else:
                # Update existing drawdown event
                self.current_drawdown_event.trough_value = min(
                    self.current_drawdown_event.trough_value, current_value
                )
                self.current_drawdown_event.max_drawdown_pct = max(
                    self.current_drawdown_event.max_drawdown_pct, current_drawdown_pct
                )
                self.current_drawdown_event.duration_days = (
                    timestamp - self.current_drawdown_event.start_date
                ).days

        # Check protection rules and take actions
        actions_taken = self._check_and_apply_protection_rules(
            current_drawdown_pct, current_value, timestamp
        )

        # Prepare response
        response = {
            "timestamp": timestamp,
            "portfolio_value": current_value,
            "high_water_mark": self.portfolio_high_water_mark,
            "current_drawdown_pct": current_drawdown_pct,
            "drawdown_level": drawdown_level.value,
            "actions_taken": actions_taken,
            "active_protections": list(self.active_protections),
            "trading_suspended": self.trading_suspended,
            "position_reduction_active": self.position_reduction_active,
            "emergency_stop_active": self.emergency_stop_active,
            "drawdown_event": self.current_drawdown_event.__dict__
            if self.current_drawdown_event
            else None,
        }

        return response

    def _classify_drawdown_level(self, drawdown_pct: float) -> DrawdownLevel:
        """Classify drawdown severity level"""

        if drawdown_pct >= 0.25:
            return DrawdownLevel.CRITICAL
        elif drawdown_pct >= 0.15:
            return DrawdownLevel.SEVERE
        elif drawdown_pct >= 0.10:
            return DrawdownLevel.HIGH
        elif drawdown_pct >= 0.05:
            return DrawdownLevel.ELEVATED
        else:
            return DrawdownLevel.NORMAL

    def _check_and_apply_protection_rules(
        self, current_drawdown_pct: float, current_value: float, timestamp: datetime
    ) -> List[str]:
        """Check protection rules and apply necessary actions"""

        actions_taken = []

        # Sort rules by drawdown threshold (ascending)
        active_rules = [rule for rule in self.protection_rules if rule.enabled]
        active_rules.sort(key=lambda r: r.drawdown_threshold)

        for rule in active_rules:
            if current_drawdown_pct >= rule.drawdown_threshold:
                action_key = f"{rule.action.value}_{rule.drawdown_threshold}"

                # Check if this protection is already active
                if action_key in self.active_protections:
                    continue

                # Apply the protection action
                action_result = self._apply_protection_action(rule, current_value, timestamp)

                if action_result["applied"]:
                    self.active_protections.add(action_key)
                    actions_taken.append(action_result["description"])

                    # Record action in current drawdown event
                    if self.current_drawdown_event:
                        self.current_drawdown_event.actions_taken.append(
                            action_result["description"]
                        )

                    # Send notification
                    if self.notification_callback:
                        severity = (
                            "critical"
                            if rule.action
                            in [ProtectionAction.EMERGENCY_STOP, ProtectionAction.FULL_LIQUIDATION]
                            else "high"
                        )
                        self.notification_callback(action_result["description"], severity)

        return actions_taken

    def _apply_protection_action(
        self, rule: ProtectionRule, current_value: float, timestamp: datetime
    ) -> Dict[str, Any]:
        """Apply a specific protection action"""

        action_result = {"applied": False, "description": "", "details": {}}

        if rule.action == ProtectionAction.MONITOR:
            action_result.update(
                {
                    "applied": True,
                    "description": f"Enhanced monitoring activated at {rule.drawdown_threshold:.1%} drawdown",
                    "details": {"monitoring_level": "enhanced"},
                }
            )

        elif rule.action == ProtectionAction.REDUCE_POSITIONS:
            if not self.position_reduction_active:
                self.position_reduction_active = True
                action_result.update(
                    {
                        "applied": True,
                        "description": f"Position reduction ({rule.position_reduction_pct:.0%}) triggered at {rule.drawdown_threshold:.1%} drawdown",
                        "details": {
                            "reduction_percentage": rule.position_reduction_pct,
                            "trigger_drawdown": rule.drawdown_threshold,
                        },
                    }
                )

        elif rule.action == ProtectionAction.SUSPEND_NEW_TRADES:
            if not self.trading_suspended:
                self.trading_suspended = True
                action_result.update(
                    {
                        "applied": True,
                        "description": f"New trading suspended at {rule.drawdown_threshold:.1%} drawdown",
                        "details": {
                            "suspension_type": "new_trades_only",
                            "min_recovery_days": rule.min_recovery_days,
                        },
                    }
                )

        elif rule.action == ProtectionAction.PARTIAL_LIQUIDATION:
            action_result.update(
                {
                    "applied": True,
                    "description": f"Partial liquidation ({rule.position_reduction_pct:.0%}) initiated at {rule.drawdown_threshold:.1%} drawdown",
                    "details": {
                        "liquidation_percentage": rule.position_reduction_pct,
                        "liquidation_type": "partial",
                    },
                }
            )

        elif rule.action == ProtectionAction.EMERGENCY_STOP:
            if not self.emergency_stop_active:
                self.emergency_stop_active = True
                self.trading_suspended = True
                action_result.update(
                    {
                        "applied": True,
                        "description": f"EMERGENCY STOP activated at {rule.drawdown_threshold:.1%} drawdown",
                        "details": {
                            "stop_type": "emergency",
                            "all_trading_suspended": True,
                            "min_recovery_days": rule.min_recovery_days,
                        },
                    }
                )

        return action_result

    def _handle_drawdown_recovery(self, current_value: float, timestamp: datetime):
        """Handle drawdown recovery logic"""

        if not self.current_drawdown_event:
            return

        # Mark drawdown event as ended
        self.current_drawdown_event.is_active = False
        self.current_drawdown_event.end_date = timestamp
        self.current_drawdown_event.recovery_date = timestamp

        # Add to history
        self.drawdown_history.append(self.current_drawdown_event)

        self.logger.info(
            f"Drawdown recovery completed. Max drawdown was {self.current_drawdown_event.max_drawdown_pct:.2%}"
        )

        # Start recovery process
        self.recovery_start_date = timestamp
        self._initiate_recovery_process()

    def _initiate_recovery_process(self):
        """Initiate the recovery process after drawdown ends"""

        # Determine recovery conditions based on max drawdown experienced
        if not self.current_drawdown_event:
            return

        max_dd = self.current_drawdown_event.max_drawdown_pct

        if max_dd >= 0.25:  # Critical drawdown
            recovery_days = 14
        elif max_dd >= 0.15:  # Severe drawdown
            recovery_days = 7
        elif max_dd >= 0.10:  # High drawdown
            recovery_days = 5
        else:
            recovery_days = 2

        self.recovery_conditions = {
            "min_recovery_days": recovery_days,
            "recovery_start_date": self.recovery_start_date,
            "gradual_reentry": True,
            "enhanced_monitoring_days": recovery_days * 2,
        }

        self.logger.info(
            f"Recovery process initiated. Minimum recovery period: {recovery_days} days"
        )

    def check_trading_allowed(
        self,
        trade_type: str = "new_position",
        trade_size: float = 0.0,
        current_timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Check if trading is allowed given current protection status

        Args:
            trade_type: Type of trade ("new_position", "close_position", "adjust_position")
            trade_size: Size of proposed trade
            current_timestamp: Current timestamp (default: now)

        Returns:
            Dictionary with trading permission and reasoning
        """

        if current_timestamp is None:
            current_timestamp = datetime.now()

        # Check emergency stop
        if self.emergency_stop_active:
            return {
                "allowed": False,
                "reason": "Emergency stop is active",
                "action_required": "Manual review and approval required",
                "severity": "critical",
            }

        # Check trading suspension
        if self.trading_suspended:
            if trade_type == "close_position":
                return {
                    "allowed": True,
                    "reason": "Closing positions allowed during trading suspension",
                    "severity": "low",
                }
            else:
                return {
                    "allowed": False,
                    "reason": "New trading is suspended due to drawdown protection",
                    "action_required": "Wait for recovery conditions to be met",
                    "severity": "medium",
                }

        # Check recovery conditions
        if self.recovery_conditions and self.recovery_start_date:
            days_since_recovery = (current_timestamp - self.recovery_start_date).days
            min_recovery_days = self.recovery_conditions.get("min_recovery_days", 0)

            if days_since_recovery < min_recovery_days:
                if trade_type == "new_position":
                    return {
                        "allowed": False,
                        "reason": f"Still in recovery period ({days_since_recovery}/{min_recovery_days} days)",
                        "action_required": f"Wait {min_recovery_days - days_since_recovery} more days",
                        "severity": "medium",
                    }

        # Check position reduction requirements
        if self.position_reduction_active and trade_type == "new_position":
            return {
                "allowed": False,
                "reason": "Position reduction is active - no new positions allowed",
                "action_required": "Complete position reduction first",
                "severity": "medium",
            }

        # All checks passed
        return {"allowed": True, "reason": "No protection restrictions apply", "severity": "low"}

    def get_protection_status(self) -> Dict[str, Any]:
        """Get current protection status summary"""

        current_drawdown_pct = 0.0
        if self.portfolio_high_water_mark and self.current_drawdown_event:
            current_value = self.portfolio_high_water_mark - (
                self.current_drawdown_event.max_drawdown_pct * self.portfolio_high_water_mark
            )
            current_drawdown_pct = self.current_drawdown_event.max_drawdown_pct

        return {
            "timestamp": datetime.now(),
            "protection_active": len(self.active_protections) > 0,
            "active_protections": list(self.active_protections),
            "trading_suspended": self.trading_suspended,
            "position_reduction_active": self.position_reduction_active,
            "emergency_stop_active": self.emergency_stop_active,
            "current_drawdown_pct": current_drawdown_pct,
            "high_water_mark": self.portfolio_high_water_mark,
            "drawdown_events_count": len(self.drawdown_history),
            "in_recovery": self.recovery_conditions is not None,
            "recovery_conditions": self.recovery_conditions,
            "last_check": self.last_protection_check,
        }

    def reset_protections(
        self, reset_type: str = "partial", authorization_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reset protection systems (requires authorization for safety)

        Args:
            reset_type: "partial" (reset flags) or "full" (reset everything)
            authorization_code: Authorization code for safety (if required)

        Returns:
            Dictionary with reset results
        """

        # In production, you'd want proper authorization
        if authorization_code != "MANUAL_OVERRIDE_2024":
            return {
                "success": False,
                "reason": "Invalid or missing authorization code",
                "action": "none",
            }

        reset_actions = []

        if reset_type == "partial":
            # Reset active protection flags but keep history
            self.active_protections.clear()
            self.trading_suspended = False
            self.position_reduction_active = False
            # Keep emergency_stop_active for safety - require full reset

            reset_actions = [
                "Cleared active protections",
                "Re-enabled trading (except emergency stop)",
                "Disabled position reduction",
            ]

        elif reset_type == "full":
            # Reset everything (dangerous!)
            self.active_protections.clear()
            self.trading_suspended = False
            self.position_reduction_active = False
            self.emergency_stop_active = False
            self.recovery_conditions = None
            self.recovery_start_date = None

            reset_actions = [
                "FULL RESET: Cleared all protections",
                "Re-enabled all trading",
                "Cleared recovery conditions",
                "CAUTION: Emergency stop disabled",
            ]

        self.logger.warning(f"Protection reset performed: {reset_type}. Actions: {reset_actions}")

        return {
            "success": True,
            "reset_type": reset_type,
            "actions_taken": reset_actions,
            "timestamp": datetime.now(),
        }

    def get_historical_drawdowns(self, days: int = 90) -> List[Dict[str, Any]]:
        """Get historical drawdown events"""

        cutoff_date = datetime.now() - timedelta(days=days)

        recent_events = [
            {
                "start_date": event.start_date.isoformat(),
                "end_date": event.end_date.isoformat() if event.end_date else None,
                "max_drawdown_pct": event.max_drawdown_pct,
                "duration_days": event.duration_days,
                "actions_taken": event.actions_taken,
                "is_active": event.is_active,
                "recovery_date": event.recovery_date.isoformat() if event.recovery_date else None,
            }
            for event in self.drawdown_history
            if event.start_date >= cutoff_date
        ]

        # Add current event if active
        if self.current_drawdown_event and self.current_drawdown_event.is_active:
            recent_events.append(
                {
                    "start_date": self.current_drawdown_event.start_date.isoformat(),
                    "end_date": None,
                    "max_drawdown_pct": self.current_drawdown_event.max_drawdown_pct,
                    "duration_days": self.current_drawdown_event.duration_days,
                    "actions_taken": self.current_drawdown_event.actions_taken,
                    "is_active": True,
                    "recovery_date": None,
                }
            )

        return recent_events

    def calculate_drawdown_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive drawdown statistics"""

        if not self.drawdown_history:
            return {"status": "no_data", "message": "No historical drawdown events"}

        completed_events = [event for event in self.drawdown_history if not event.is_active]

        if not completed_events:
            return {"status": "insufficient_data", "message": "No completed drawdown events"}

        # Calculate statistics
        max_drawdowns = [event.max_drawdown_pct for event in completed_events]
        durations = [event.duration_days for event in completed_events]

        stats = {
            "total_events": len(completed_events),
            "avg_max_drawdown": np.mean(max_drawdowns),
            "worst_drawdown": np.max(max_drawdowns),
            "best_drawdown": np.min(max_drawdowns),
            "avg_duration_days": np.mean(durations),
            "longest_duration_days": np.max(durations),
            "shortest_duration_days": np.min(durations),
            "events_per_year": len(completed_events) / (max(durations) / 365) if durations else 0,
        }

        # Recovery statistics
        recovery_events = [event for event in completed_events if event.recovery_date]
        if recovery_events:
            recovery_times = [
                (event.recovery_date - event.end_date).days
                for event in recovery_events
                if event.end_date
            ]
            if recovery_times:
                stats["avg_recovery_days"] = np.mean(recovery_times)
                stats["max_recovery_days"] = np.max(recovery_times)

        return {
            "status": "success",
            "statistics": stats,
            "calculated_at": datetime.now().isoformat(),
        }
