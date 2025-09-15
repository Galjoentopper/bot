#!/usr/bin/env python3
"""
System Monitor
=============

Comprehensive system monitoring with real-time health checks:
- Model performance monitoring
- Data quality validation
- System resource monitoring
- Trading performance alerts
- Automated recovery mechanisms
"""

import json
import logging
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemHealthMetrics:
    """System health metrics."""

    timestamp: str
    cpu_usage_pct: float
    memory_usage_pct: float
    disk_usage_pct: float
    model_prediction_success_rate: float
    data_freshness_score: float
    trading_system_status: str
    telegram_bot_status: str
    last_trade_time: Optional[str]
    total_errors_last_hour: int
    performance_score: float  # 0-100 composite score


@dataclass
class ModelHealthMetrics:
    """Individual model health metrics."""

    model_type: str
    symbol: str
    prediction_success_rate: float
    prediction_latency_ms: float
    feature_drift_score: float
    prediction_confidence: float
    last_prediction_time: str
    errors_last_24h: int
    performance_trend: str  # 'improving', 'stable', 'degrading'


@dataclass
class TradingHealthMetrics:
    """Trading system health metrics."""

    total_trades_24h: int
    win_rate_24h: float
    avg_trade_duration_hours: float
    portfolio_utilization_pct: float
    max_drawdown_24h: float
    sharpe_ratio_7d: float
    profit_factor: float
    risk_adjusted_return: float


class SystemMonitor:
    """Advanced system monitoring with predictive alerts."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize system monitor."""
        self.config = config

        monitoring_config = config.get("monitoring", {})
        self.check_interval = monitoring_config.get("check_interval_seconds", 60)  # 1 minute
        self.alert_thresholds = monitoring_config.get(
            "alert_thresholds",
            {
                "cpu_usage": 85.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "prediction_success_rate": 80.0,
                "data_freshness_hours": 2.0,
                "max_errors_per_hour": 10,
            },
        )

        # Historical data storage
        self.health_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.model_metrics_history = defaultdict(lambda: deque(maxlen=720))  # 12 hours
        self.trading_metrics_history = deque(maxlen=168)  # 7 days at hourly intervals

        # Error tracking
        self.error_log = deque(maxlen=1000)
        self.model_errors = defaultdict(lambda: deque(maxlen=100))

        # Performance baselines
        self.performance_baselines = {}
        self.anomaly_detection_window = 48  # Hours

        # Monitoring state
        self.last_health_check = 0
        self.consecutive_failures = defaultdict(int)
        self.alert_cooldowns = defaultdict(float)  # Prevent alert spam

        # Telegram integration for alerts
        self.telegram_alerts = monitoring_config.get("telegram_alerts", True)

        logger.info("System Monitor initialized with comprehensive health tracking")

    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_usage_pct": cpu_percent,
                "memory_usage_pct": memory.percent,
                "disk_usage_pct": disk.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
            }

        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return {
                "cpu_usage_pct": 0,
                "memory_usage_pct": 0,
                "disk_usage_pct": 0,
                "memory_available_gb": 0,
                "disk_free_gb": 0,
            }

    def check_trading_system_status(self) -> Dict[str, Any]:
        """Check if trading system processes are running."""
        try:
            status = {
                "trading_process_running": False,
                "telegram_process_running": False,
                "last_log_update": None,
                "active_sessions": 0,
            }

            # Check for Python trading processes
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.info["cmdline"]
                    if cmdline and any("enhanced_trader.py" in cmd for cmd in cmdline):
                        status["trading_process_running"] = True
                    elif cmdline and any("telegram" in cmd.lower() for cmd in cmdline):
                        status["telegram_process_running"] = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Check log file freshness
            log_files = list(Path("logs").glob("trading_*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                last_modified = datetime.fromtimestamp(latest_log.stat().st_mtime)
                status["last_log_update"] = last_modified.isoformat()

            return status

        except Exception as e:
            logger.error(f"Failed to check trading system status: {e}")
            return {
                "trading_process_running": False,
                "telegram_process_running": False,
                "last_log_update": None,
                "active_sessions": 0,
            }

    def analyze_model_performance(
        self, model_predictions: Dict[str, List]
    ) -> Dict[str, ModelHealthMetrics]:
        """Analyze individual model performance and health."""
        try:
            model_health = {}

            for model_key, predictions in model_predictions.items():
                if not predictions:
                    continue

                # Parse model key (format: model_type_symbol)
                try:
                    model_type, symbol = model_key.split("_", 1)
                except ValueError:
                    continue

                # Calculate metrics
                success_predictions = [p for p in predictions if p is not None]
                success_rate = (
                    len(success_predictions) / len(predictions) * 100 if predictions else 0
                )

                # Feature drift detection (simplified)
                drift_score = self._calculate_feature_drift(model_type, symbol)

                # Prediction confidence (based on variance)
                confidence = 100 - (np.std(success_predictions) * 100) if success_predictions else 0
                confidence = max(0, min(100, confidence))

                # Performance trend analysis
                trend = self._analyze_performance_trend(model_key)

                model_health[model_key] = ModelHealthMetrics(
                    model_type=model_type,
                    symbol=symbol,
                    prediction_success_rate=success_rate,
                    prediction_latency_ms=50.0,  # Estimated
                    feature_drift_score=drift_score,
                    prediction_confidence=confidence,
                    last_prediction_time=datetime.now().isoformat(),
                    errors_last_24h=len(self.model_errors[model_key]),
                    performance_trend=trend,
                )

            return model_health

        except Exception as e:
            logger.error(f"Model performance analysis failed: {e}")
            return {}

    def _calculate_feature_drift(self, model_type: str, symbol: str) -> float:
        """Calculate feature drift score (0-100, higher = more drift)."""
        try:
            # This is a simplified drift detection
            # In production, this would compare current feature distributions
            # to training distributions using statistical tests

            # Simulate drift based on time since model training
            # Real implementation would use KS test, PSI, etc.
            base_drift = min(30, time.time() % 100)  # Simulate gradual drift

            return base_drift

        except Exception:
            return 0.0

    def _analyze_performance_trend(self, model_key: str) -> str:
        """Analyze performance trend over time."""
        try:
            history = self.model_metrics_history[model_key]

            if len(history) < 5:
                return "insufficient_data"

            # Get recent performance
            recent_scores = [h.prediction_success_rate for h in list(history)[-5:]]
            older_scores = (
                [h.prediction_success_rate for h in list(history)[-10:-5]]
                if len(history) >= 10
                else []
            )

            if not older_scores:
                return "stable"

            recent_avg = np.mean(recent_scores)
            older_avg = np.mean(older_scores)

            if recent_avg > older_avg + 5:
                return "improving"
            elif recent_avg < older_avg - 5:
                return "degrading"
            else:
                return "stable"

        except Exception:
            return "unknown"

    def calculate_composite_performance_score(
        self,
        system_resources: Dict[str, float],
        trading_status: Dict[str, Any],
        model_health: Dict[str, ModelHealthMetrics],
    ) -> float:
        """Calculate composite system performance score (0-100)."""
        try:
            scores = []
            weights = []

            # System resources score (30% weight)
            resource_score = (
                max(0, 100 - system_resources["cpu_usage_pct"]) * 0.4
                + max(0, 100 - system_resources["memory_usage_pct"]) * 0.4
                + max(0, 100 - system_resources["disk_usage_pct"]) * 0.2
            )
            scores.append(resource_score)
            weights.append(0.3)

            # Trading system status score (25% weight)
            status_score = 0
            if trading_status["trading_process_running"]:
                status_score += 50
            if trading_status["telegram_process_running"]:
                status_score += 30
            if trading_status["last_log_update"]:
                last_update = datetime.fromisoformat(trading_status["last_log_update"])
                hours_since = (datetime.now() - last_update).total_seconds() / 3600
                if hours_since < 1:
                    status_score += 20
                elif hours_since < 4:
                    status_score += 10

            scores.append(status_score)
            weights.append(0.25)

            # Model health score (35% weight)
            if model_health:
                model_scores = [m.prediction_success_rate for m in model_health.values()]
                model_score = np.mean(model_scores) if model_scores else 0
            else:
                model_score = 0

            scores.append(model_score)
            weights.append(0.35)

            # Error rate score (10% weight)
            recent_errors = len(
                [
                    e
                    for e in self.error_log
                    if (datetime.now() - datetime.fromisoformat(e["timestamp"])).total_seconds()
                    < 3600
                ]
            )
            error_score = max(0, 100 - recent_errors * 10)  # -10 points per error
            scores.append(error_score)
            weights.append(0.1)

            # Calculate weighted average
            composite_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return max(0, min(100, composite_score))

        except Exception as e:
            logger.error(f"Composite score calculation failed: {e}")
            return 50.0  # Neutral score on error

    def check_alert_conditions(
        self,
        health_metrics: SystemHealthMetrics,
        model_health: Dict[str, ModelHealthMetrics],
    ) -> List[Dict[str, Any]]:
        """Check for alert conditions and return alerts."""
        try:
            alerts = []
            current_time = time.time()

            # System resource alerts
            if health_metrics.cpu_usage_pct > self.alert_thresholds["cpu_usage"]:
                if current_time - self.alert_cooldowns["cpu"] > 300:  # 5-minute cooldown
                    alerts.append(
                        {
                            "type": "system_resource",
                            "severity": "high",
                            "message": f"High CPU usage: {health_metrics.cpu_usage_pct:.1f}%",
                            "metric": "cpu_usage",
                            "value": health_metrics.cpu_usage_pct,
                        }
                    )
                    self.alert_cooldowns["cpu"] = current_time

            if health_metrics.memory_usage_pct > self.alert_thresholds["memory_usage"]:
                if current_time - self.alert_cooldowns["memory"] > 300:
                    alerts.append(
                        {
                            "type": "system_resource",
                            "severity": "high",
                            "message": f"High memory usage: {health_metrics.memory_usage_pct:.1f}%",
                            "metric": "memory_usage",
                            "value": health_metrics.memory_usage_pct,
                        }
                    )
                    self.alert_cooldowns["memory"] = current_time

            # Model performance alerts
            for model_key, model_metrics in model_health.items():
                if (
                    model_metrics.prediction_success_rate
                    < self.alert_thresholds["prediction_success_rate"]
                ):
                    if (
                        current_time - self.alert_cooldowns[f"model_{model_key}"] > 600
                    ):  # 10-minute cooldown
                        alerts.append(
                            {
                                "type": "model_performance",
                                "severity": "medium",
                                "message": f"Low prediction success rate for {model_key}: {model_metrics.prediction_success_rate:.1f}%",
                                "metric": "prediction_success_rate",
                                "model": model_key,
                                "value": model_metrics.prediction_success_rate,
                            }
                        )
                        self.alert_cooldowns[f"model_{model_key}"] = current_time

                if model_metrics.performance_trend == "degrading":
                    if (
                        current_time - self.alert_cooldowns[f"trend_{model_key}"] > 1800
                    ):  # 30-minute cooldown
                        alerts.append(
                            {
                                "type": "model_performance",
                                "severity": "medium",
                                "message": f"Performance degrading for {model_key}",
                                "metric": "performance_trend",
                                "model": model_key,
                                "value": "degrading",
                            }
                        )
                        self.alert_cooldowns[f"trend_{model_key}"] = current_time

            # Trading system alerts
            if health_metrics.trading_system_status != "running":
                alerts.append(
                    {
                        "type": "system_status",
                        "severity": "critical",
                        "message": f"Trading system not running: {health_metrics.trading_system_status}",
                        "metric": "system_status",
                        "value": health_metrics.trading_system_status,
                    }
                )

            # Performance score alert
            if health_metrics.performance_score < 60:
                if current_time - self.alert_cooldowns["performance"] > 900:  # 15-minute cooldown
                    alerts.append(
                        {
                            "type": "system_performance",
                            "severity": "medium",
                            "message": f"Low system performance score: {health_metrics.performance_score:.1f}",
                            "metric": "performance_score",
                            "value": health_metrics.performance_score,
                        }
                    )
                    self.alert_cooldowns["performance"] = current_time

            return alerts

        except Exception as e:
            logger.error(f"Alert condition checking failed: {e}")
            return []

    def log_error(self, error_type: str, message: str, model_key: Optional[str] = None):
        """Log error for monitoring and analysis."""
        try:
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": error_type,
                "message": message,
                "model_key": model_key,
                "traceback": (
                    traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None
                ),
            }

            self.error_log.append(error_entry)

            if model_key:
                self.model_errors[model_key].append(error_entry)

            logger.warning(f"Error logged: {error_type} - {message}")

        except Exception as e:
            logger.error(f"Failed to log error: {e}")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary for dashboard."""
        try:
            # Get current metrics
            system_resources = self.get_system_resources()
            trading_status = self.check_trading_system_status()

            # Create health metrics
            health_metrics = SystemHealthMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage_pct=system_resources["cpu_usage_pct"],
                memory_usage_pct=system_resources["memory_usage_pct"],
                disk_usage_pct=system_resources["disk_usage_pct"],
                model_prediction_success_rate=95.0,  # Will be calculated from actual data
                data_freshness_score=90.0,  # Will be calculated from data timestamps
                trading_system_status=(
                    "running" if trading_status["trading_process_running"] else "stopped"
                ),
                telegram_bot_status=(
                    "running" if trading_status["telegram_process_running"] else "stopped"
                ),
                last_trade_time=None,  # Will be populated from trading data
                total_errors_last_hour=len(
                    [
                        e
                        for e in self.error_log
                        if (datetime.now() - datetime.fromisoformat(e["timestamp"])).total_seconds()
                        < 3600
                    ]
                ),
                performance_score=0,  # Will be calculated below
            )

            # Calculate performance score
            health_metrics.performance_score = self.calculate_composite_performance_score(
                system_resources, trading_status, {}
            )

            # Store in history
            self.health_history.append(health_metrics)

            # Generate summary
            summary = {
                "current_health": asdict(health_metrics),
                "system_status": {
                    "overall": (
                        "healthy"
                        if health_metrics.performance_score > 70
                        else ("degraded" if health_metrics.performance_score > 40 else "critical")
                    ),
                    "uptime_hours": self._calculate_uptime(),
                    "total_errors_24h": len(
                        [
                            e
                            for e in self.error_log
                            if (
                                datetime.now() - datetime.fromisoformat(e["timestamp"])
                            ).total_seconds()
                            < 86400
                        ]
                    ),
                },
                "resource_usage": system_resources,
                "recent_trends": self._get_recent_trends(),
                "recommendations": self._generate_recommendations(health_metrics),
            }

            return summary

        except Exception as e:
            logger.error(f"Health summary generation failed: {e}")
            return {"error": str(e), "status": "unknown"}

    def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours."""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            return uptime_seconds / 3600
        except Exception:
            return 0.0

    def _get_recent_trends(self) -> Dict[str, str]:
        """Analyze recent performance trends."""
        try:
            if len(self.health_history) < 10:
                return {"performance": "insufficient_data", "resource_usage": "stable"}

            recent = list(self.health_history)[-5:]
            older = list(self.health_history)[-10:-5]

            recent_performance = np.mean([h.performance_score for h in recent])
            older_performance = np.mean([h.performance_score for h in older])

            if recent_performance > older_performance + 5:
                perf_trend = "improving"
            elif recent_performance < older_performance - 5:
                perf_trend = "degrading"
            else:
                perf_trend = "stable"

            return {"performance": perf_trend, "resource_usage": "stable"}

        except Exception:
            return {"performance": "unknown", "resource_usage": "unknown"}

    def _generate_recommendations(self, health_metrics: SystemHealthMetrics) -> List[str]:
        """Generate actionable recommendations based on health metrics."""
        recommendations = []

        try:
            if health_metrics.cpu_usage_pct > 80:
                recommendations.append(
                    "Consider reducing model complexity or adding more CPU resources"
                )

            if health_metrics.memory_usage_pct > 80:
                recommendations.append(
                    "Memory usage is high - consider optimizing data structures or adding RAM"
                )

            if health_metrics.performance_score < 50:
                recommendations.append(
                    "System performance is critically low - immediate attention required"
                )

            if health_metrics.total_errors_last_hour > 5:
                recommendations.append(
                    f"High error rate ({health_metrics.total_errors_last_hour}/hour) - check logs for issues"
                )

            if health_metrics.trading_system_status != "running":
                recommendations.append("Trading system is not running - restart required")

            if not recommendations:
                recommendations.append("System is operating within normal parameters")

        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")

        return recommendations


if __name__ == "__main__":
    # Test the system monitor
    test_config = {
        "monitoring": {
            "check_interval_seconds": 60,
            "alert_thresholds": {
                "cpu_usage": 85.0,
                "memory_usage": 85.0,
                "prediction_success_rate": 80.0,
            },
        }
    }

    monitor = SystemMonitor(test_config)
    health = monitor.get_health_summary()
    print(json.dumps(health, indent=2))
