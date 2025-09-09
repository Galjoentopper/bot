"""
Model Performance Tracking System

Tracks model performance metrics over time, detects degradation,
and provides automated performance reporting with alerts.
"""

import json
import logging
import sqlite3
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

try:
    import empyrical

    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Individual performance metric record"""

    timestamp: datetime
    model_name: str
    metric_name: str
    metric_value: float
    data_period: str  # e.g., "1h", "1d", "1w"
    sample_size: int
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceAlert:
    """Alert for performance degradation"""

    timestamp: datetime
    model_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    recommendation: str


class PerformanceTracker:
    """Comprehensive model performance tracking"""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        retention_days: int = 90,
        baseline_window_days: int = 30,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize performance tracker

        Args:
            db_path: SQLite database path for storing metrics
            retention_days: Days to retain performance data
            baseline_window_days: Days to use for baseline calculation
            alert_thresholds: Custom alert thresholds by severity
        """
        self.db_path = db_path or Path("data/performance_tracking.db")
        self.retention_days = retention_days
        self.baseline_window_days = baseline_window_days

        # Default alert thresholds (percentage degradation)
        self.alert_thresholds = alert_thresholds or {
            "low": 0.05,  # 5% degradation
            "medium": 0.15,  # 15% degradation
            "high": 0.30,  # 30% degradation
            "critical": 0.50,  # 50% degradation
        }

        self.logger = logging.getLogger(__name__)

        # In-memory storage for recent metrics (backup to DB)
        self.recent_metrics = deque(maxlen=10000)
        self.model_baselines = {}

        # Initialize database
        self._init_database()
        self._load_baselines()

    def _init_database(self):
        """Initialize SQLite database for performance tracking"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    data_period TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    additional_info TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    baseline_value REAL NOT NULL,
                    degradation_pct REAL NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indices for better query performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_model_time
                ON performance_metrics(model_name, timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_model_time
                ON performance_alerts(model_name, timestamp)
            """
            )

        self.logger.info(f"Performance tracking database initialized at {self.db_path}")

    def record_regression_performance(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        data_period: str = "1h",
        additional_info: Optional[Dict] = None,
    ):
        """Record regression model performance metrics"""

        timestamp = datetime.now()
        sample_size = len(y_true)

        # Calculate standard regression metrics
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100,
        }

        # Calculate directional accuracy (for trading models)
        y_true_returns = np.diff(y_true)
        y_pred_returns = np.diff(y_pred)

        if len(y_true_returns) > 0:
            directional_accuracy = np.mean(np.sign(y_true_returns) == np.sign(y_pred_returns))
            metrics["directional_accuracy"] = directional_accuracy

        # Calculate additional trading-specific metrics if empyrical is available
        if EMPYRICAL_AVAILABLE and "returns" in str(additional_info):
            try:
                returns = y_pred  # Assuming predictions are returns
                if len(returns) > 10:  # Need sufficient data
                    metrics.update(
                        {
                            "sharpe_ratio": empyrical.sharpe_ratio(returns),
                            "sortino_ratio": empyrical.sortino_ratio(returns),
                            "calmar_ratio": empyrical.calmar_ratio(returns),
                            "max_drawdown": empyrical.max_drawdown(returns),
                            "volatility": empyrical.annual_volatility(returns),
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Could not calculate financial metrics: {e}")

        # Record each metric
        recorded_metrics = []
        for metric_name, metric_value in metrics.items():
            if not np.isfinite(metric_value):
                self.logger.warning(f"Skipping invalid metric {metric_name}: {metric_value}")
                continue

            perf_metric = PerformanceMetric(
                timestamp=timestamp,
                model_name=model_name,
                metric_name=metric_name,
                metric_value=float(metric_value),
                data_period=data_period,
                sample_size=sample_size,
                additional_info=additional_info,
            )

            self._store_metric(perf_metric)
            recorded_metrics.append(perf_metric)

        self.logger.info(f"Recorded {len(recorded_metrics)} performance metrics for {model_name}")

        # Check for performance alerts
        alerts = self._check_performance_alerts(model_name, recorded_metrics)
        if alerts:
            self._store_alerts(alerts)

        return recorded_metrics

    def record_classification_performance(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        data_period: str = "1h",
        additional_info: Optional[Dict] = None,
    ):
        """Record classification model performance metrics"""

        timestamp = datetime.now()
        sample_size = len(y_true)

        # Calculate classification metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Record each metric
        recorded_metrics = []
        for metric_name, metric_value in metrics.items():
            if not np.isfinite(metric_value):
                continue

            perf_metric = PerformanceMetric(
                timestamp=timestamp,
                model_name=model_name,
                metric_name=metric_name,
                metric_value=float(metric_value),
                data_period=data_period,
                sample_size=sample_size,
                additional_info=additional_info,
            )

            self._store_metric(perf_metric)
            recorded_metrics.append(perf_metric)

        # Check for alerts
        alerts = self._check_performance_alerts(model_name, recorded_metrics)
        if alerts:
            self._store_alerts(alerts)

        return recorded_metrics

    def record_custom_metrics(
        self,
        model_name: str,
        metrics_dict: Dict[str, float],
        data_period: str = "1h",
        sample_size: int = 1,
        additional_info: Optional[Dict] = None,
    ):
        """Record custom performance metrics"""

        timestamp = datetime.now()
        recorded_metrics = []

        for metric_name, metric_value in metrics_dict.items():
            if not np.isfinite(metric_value):
                continue

            perf_metric = PerformanceMetric(
                timestamp=timestamp,
                model_name=model_name,
                metric_name=metric_name,
                metric_value=float(metric_value),
                data_period=data_period,
                sample_size=sample_size,
                additional_info=additional_info,
            )

            self._store_metric(perf_metric)
            recorded_metrics.append(perf_metric)

        # Check for alerts
        alerts = self._check_performance_alerts(model_name, recorded_metrics)
        if alerts:
            self._store_alerts(alerts)

        return recorded_metrics

    def _store_metric(self, metric: PerformanceMetric):
        """Store performance metric to database and memory"""

        # Add to in-memory storage
        self.recent_metrics.append(metric)

        # Store to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO performance_metrics
                    (timestamp, model_name, metric_name, metric_value,
                     data_period, sample_size, additional_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metric.timestamp.isoformat(),
                        metric.model_name,
                        metric.metric_name,
                        metric.metric_value,
                        metric.data_period,
                        metric.sample_size,
                        json.dumps(metric.additional_info) if metric.additional_info else None,
                    ),
                )
        except Exception as e:
            self.logger.error(f"Error storing metric to database: {e}")

    def _check_performance_alerts(
        self, model_name: str, recent_metrics: List[PerformanceMetric]
    ) -> List[PerformanceAlert]:
        """Check for performance degradation and generate alerts"""

        alerts = []

        for metric in recent_metrics:
            # Get baseline for this model and metric
            baseline = self._get_baseline_value(model_name, metric.metric_name)

            if baseline is None:
                continue  # No baseline yet

            current_value = metric.metric_value

            # Calculate degradation (handling direction of metric)
            if self._is_higher_better_metric(metric.metric_name):
                # For metrics like accuracy, R², directional_accuracy - higher is better
                degradation_pct = (baseline - current_value) / baseline if baseline != 0 else 0
            else:
                # For metrics like MSE, MAE - lower is better
                degradation_pct = (current_value - baseline) / baseline if baseline != 0 else 0

            # Determine severity
            severity = self._get_alert_severity(degradation_pct)

            if severity:  # If degradation exceeds minimum threshold
                alert = PerformanceAlert(
                    timestamp=metric.timestamp,
                    model_name=model_name,
                    metric_name=metric.metric_name,
                    current_value=current_value,
                    baseline_value=baseline,
                    degradation_pct=degradation_pct,
                    severity=severity,
                    message=self._format_alert_message(metric, baseline, degradation_pct, severity),
                    recommendation=self._get_performance_recommendation(
                        metric, degradation_pct, severity
                    ),
                )
                alerts.append(alert)

        return alerts

    def _is_higher_better_metric(self, metric_name: str) -> bool:
        """Determine if higher values are better for this metric"""
        higher_better_metrics = {
            "r2",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "directional_accuracy",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
        }
        return metric_name.lower() in higher_better_metrics

    def _get_alert_severity(self, degradation_pct: float) -> Optional[str]:
        """Determine alert severity based on degradation percentage"""
        if degradation_pct >= self.alert_thresholds["critical"]:
            return "critical"
        elif degradation_pct >= self.alert_thresholds["high"]:
            return "high"
        elif degradation_pct >= self.alert_thresholds["medium"]:
            return "medium"
        elif degradation_pct >= self.alert_thresholds["low"]:
            return "low"
        else:
            return None  # No alert needed

    def _format_alert_message(
        self, metric: PerformanceMetric, baseline: float, degradation_pct: float, severity: str
    ) -> str:
        """Format alert message"""
        return (
            f"{severity.upper()} performance degradation detected in {metric.model_name}. "
            f"{metric.metric_name} degraded by {degradation_pct:.1%} "
            f"(from {baseline:.4f} to {metric.metric_value:.4f}) "
            f"over {metric.data_period} period with {metric.sample_size} samples."
        )

    def _get_performance_recommendation(
        self, metric: PerformanceMetric, degradation_pct: float, severity: str
    ) -> str:
        """Get recommendation based on performance degradation"""

        base_recommendations = {
            "low": "Monitor closely. Consider increasing validation frequency.",
            "medium": "Investigate data quality and model inputs. Consider parameter adjustment.",
            "high": "Immediate investigation required. Consider model retraining within 48 hours.",
            "critical": "Stop automated trading. Immediate model retraining or rollback required.",
        }

        model_specific = ""
        if "gru" in metric.model_name.lower() or "neural" in metric.model_name.lower():
            model_specific = " Check for overfitting or learning rate issues."
        elif "lightgbm" in metric.model_name.lower() or "gbm" in metric.model_name.lower():
            model_specific = " Review feature importance and data distribution changes."
        elif "ppo" in metric.model_name.lower() or "rl" in metric.model_name.lower():
            model_specific = " Check reward function and environment stability."

        return base_recommendations.get(severity, "Monitor and investigate.") + model_specific

    def _store_alerts(self, alerts: List[PerformanceAlert]):
        """Store performance alerts to database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                for alert in alerts:
                    conn.execute(
                        """
                        INSERT INTO performance_alerts
                        (timestamp, model_name, metric_name, current_value,
                         baseline_value, degradation_pct, severity, message, recommendation)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            alert.timestamp.isoformat(),
                            alert.model_name,
                            alert.metric_name,
                            alert.current_value,
                            alert.baseline_value,
                            alert.degradation_pct,
                            alert.severity,
                            alert.message,
                            alert.recommendation,
                        ),
                    )

            self.logger.warning(f"Stored {len(alerts)} performance alerts")

        except Exception as e:
            self.logger.error(f"Error storing alerts: {e}")

    def _load_baselines(self):
        """Load model performance baselines from historical data"""

        cutoff_date = datetime.now() - timedelta(days=self.baseline_window_days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT model_name, metric_name, AVG(metric_value) as avg_value
                    FROM performance_metrics
                    WHERE timestamp >= ?
                    GROUP BY model_name, metric_name
                    HAVING COUNT(*) >= 5
                """,
                    (cutoff_date.isoformat(),),
                )

                for row in cursor.fetchall():
                    model_name, metric_name, avg_value = row

                    if model_name not in self.model_baselines:
                        self.model_baselines[model_name] = {}
                    self.model_baselines[model_name][metric_name] = avg_value

            self.logger.info(f"Loaded baselines for {len(self.model_baselines)} models")

        except Exception as e:
            self.logger.error(f"Error loading baselines: {e}")

    def _get_baseline_value(self, model_name: str, metric_name: str) -> Optional[float]:
        """Get baseline value for model and metric"""
        return self.model_baselines.get(model_name, {}).get(metric_name)

    def update_baselines(self):
        """Update performance baselines from recent data"""
        self._load_baselines()

    def get_model_performance_summary(self, model_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a specific model"""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent metrics
                cursor = conn.execute(
                    """
                    SELECT metric_name, metric_value, timestamp, sample_size
                    FROM performance_metrics
                    WHERE model_name = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """,
                    (model_name, cutoff_time.isoformat()),
                )

                metrics_data = cursor.fetchall()

                # Get recent alerts
                cursor = conn.execute(
                    """
                    SELECT severity, COUNT(*) as count
                    FROM performance_alerts
                    WHERE model_name = ? AND timestamp >= ?
                    GROUP BY severity
                """,
                    (model_name, cutoff_time.isoformat()),
                )

                alerts_data = dict(cursor.fetchall())

        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

        if not metrics_data:
            return {"model_name": model_name, "status": "no_recent_data"}

        # Organize metrics
        metrics_by_name = defaultdict(list)
        for metric_name, value, timestamp, sample_size in metrics_data:
            metrics_by_name[metric_name].append(
                {"value": value, "timestamp": timestamp, "sample_size": sample_size}
            )

        # Calculate summary stats
        summary_metrics = {}
        for metric_name, values in metrics_by_name.items():
            recent_values = [v["value"] for v in values[-10:]]  # Last 10 measurements

            summary_metrics[metric_name] = {
                "current": recent_values[-1] if recent_values else None,
                "average": np.mean(recent_values),
                "std": np.std(recent_values),
                "trend": self._calculate_trend(recent_values),
                "sample_count": len(values),
                "baseline": self._get_baseline_value(model_name, metric_name),
            }

        return {
            "model_name": model_name,
            "time_window_hours": hours,
            "metrics": summary_metrics,
            "alerts": alerts_data,
            "status": self._determine_model_status(summary_metrics, alerts_data),
            "last_updated": max(row[2] for row in metrics_data) if metrics_data else None,
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from recent values"""
        if len(values) < 3:
            return "insufficient_data"

        # Simple trend calculation
        first_half = np.mean(values[: len(values) // 2])
        second_half = np.mean(values[len(values) // 2 :])

        if second_half > first_half * 1.05:
            return "improving"
        elif second_half < first_half * 0.95:
            return "degrading"
        else:
            return "stable"

    def _determine_model_status(
        self, summary_metrics: Dict[str, Any], alerts_data: Dict[str, int]
    ) -> str:
        """Determine overall model status"""

        if alerts_data.get("critical", 0) > 0:
            return "critical"
        elif alerts_data.get("high", 0) > 0:
            return "degraded"
        elif alerts_data.get("medium", 0) > 2:  # Multiple medium alerts
            return "concerning"
        elif any(m.get("trend") == "degrading" for m in summary_metrics.values()):
            return "declining"
        else:
            return "healthy"

    def get_all_models_status(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Get performance status for all models"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT model_name
                    FROM performance_metrics
                    WHERE timestamp >= ?
                """,
                    ((datetime.now() - timedelta(hours=hours)).isoformat(),),
                )

                model_names = [row[0] for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Error getting model names: {e}")
            return {}

        results = {}
        for model_name in model_names:
            results[model_name] = self.get_model_performance_summary(model_name, hours)

        return results

    def cleanup_old_data(self):
        """Clean up old performance data beyond retention period"""

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean metrics
                result = conn.execute(
                    """
                    DELETE FROM performance_metrics
                    WHERE timestamp < ?
                """,
                    (cutoff_date.isoformat(),),
                )

                metrics_deleted = result.rowcount

                # Clean alerts
                result = conn.execute(
                    """
                    DELETE FROM performance_alerts
                    WHERE timestamp < ?
                """,
                    (cutoff_date.isoformat(),),
                )

                alerts_deleted = result.rowcount

                # Vacuum database to reclaim space
                conn.execute("VACUUM")

            self.logger.info(
                f"Cleaned up {metrics_deleted} old metrics and {alerts_deleted} old alerts"
            )

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def export_performance_data(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Export performance data to pandas DataFrame"""

        query = "SELECT * FROM performance_metrics WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp"

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return pd.DataFrame()
