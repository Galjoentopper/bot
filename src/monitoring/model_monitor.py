"""
Comprehensive Model Performance Monitoring System

Main orchestrator for model monitoring, combining drift detection,
performance tracking, A/B testing, and automated reporting.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import schedule

from .ab_testing import ABTestConfig, ABTestingFramework, ABTestResult
from .drift_detector import DriftAlert, DriftDetector
from .performance_tracker import PerformanceAlert, PerformanceTracker


@dataclass
class MonitoringReport:
    """Comprehensive monitoring report"""

    timestamp: datetime
    model_name: str

    # Performance summary
    performance_status: str  # 'healthy', 'concerning', 'degraded', 'critical'
    performance_metrics: Dict[str, float]
    performance_alerts: List[Dict[str, Any]]

    # Drift summary
    drift_status: str  # 'stable', 'minor_drift', 'significant_drift', 'critical_drift'
    drift_alerts: List[Dict[str, Any]]

    # A/B testing summary
    ab_tests_active: List[Dict[str, Any]]

    # Overall recommendation
    overall_status: str  # 'healthy', 'monitor', 'investigate', 'action_required'
    recommendations: List[str]
    confidence_score: float  # 0-1 confidence in assessment


class ModelPerformanceMonitor:
    """Comprehensive model performance monitoring orchestrator"""

    def __init__(
        self,
        data_dir: Path = Path("data"),
        monitoring_interval_minutes: int = 60,
        report_retention_days: int = 30,
    ):
        """
        Initialize comprehensive model monitoring

        Args:
            data_dir: Directory for storing monitoring data
            monitoring_interval_minutes: How often to run monitoring checks
            report_retention_days: Days to retain monitoring reports
        """
        self.data_dir = data_dir
        self.monitoring_interval = monitoring_interval_minutes
        self.report_retention_days = report_retention_days

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.drift_detector = DriftDetector(alerts_file=data_dir / "drift_alerts.json")

        self.performance_tracker = PerformanceTracker(db_path=data_dir / "performance_tracking.db")

        self.ab_testing = ABTestingFramework(db_path=data_dir / "ab_testing.db")

        # Reports storage
        self.reports_dir = data_dir / "monitoring_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None

        # Model reference data storage
        self.model_reference_data = {}

        self.logger.info("Model Performance Monitor initialized")

    def set_model_reference_data(
        self,
        model_name: str,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ):
        """Set reference baseline data for a model"""

        self.logger.info(f"Setting reference data for model: {model_name}")

        # Store for drift detection
        self.drift_detector.set_reference_data(features_df, predictions, targets)

        # Store metadata for future use
        self.model_reference_data[model_name] = {
            "features_count": len(features_df.columns),
            "samples_count": len(features_df),
            "feature_names": features_df.columns.tolist(),
            "timestamp": datetime.now(),
            "has_targets": targets is not None,
        }

        self.logger.info(
            f"Reference data set for {model_name}: {len(features_df)} samples, {len(features_df.columns)} features"
        )

    def monitor_model_prediction(
        self,
        model_name: str,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
        targets: Optional[np.ndarray] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Monitor a single model prediction and return immediate analysis

        Args:
            model_name: Name of the model being monitored
            features_df: Input features used for prediction
            predictions: Model predictions
            targets: Actual target values (if available)
            user_id: User ID for A/B testing (if applicable)

        Returns:
            Dictionary with monitoring results and alerts
        """

        monitoring_results = {
            "timestamp": datetime.now(),
            "model_name": model_name,
            "sample_size": len(predictions),
            "alerts": [],
            "status": "healthy",
        }

        try:
            # 1. Record performance metrics
            if targets is not None:
                performance_metrics = self.performance_tracker.record_regression_performance(
                    model_name=model_name,
                    y_true=targets,
                    y_pred=predictions,
                    additional_info={"features_count": len(features_df.columns)},
                )
                monitoring_results["performance_recorded"] = True

            # 2. Check for drift
            drift_results = self.drift_detector.run_comprehensive_drift_check(
                current_features=features_df,
                current_predictions=predictions,
                current_targets=targets,
            )

            # Process drift alerts
            drift_alerts = []
            for drift_type, alerts in drift_results.items():
                for alert in alerts:
                    drift_alert = {
                        "type": "drift",
                        "drift_type": alert.drift_type,
                        "severity": alert.severity,
                        "feature": alert.feature,
                        "description": alert.description,
                        "recommendation": alert.recommendation,
                        "timestamp": alert.timestamp.isoformat(),
                    }
                    drift_alerts.append(drift_alert)

            monitoring_results["drift_alerts"] = drift_alerts

            # 3. Handle A/B testing (if user_id provided)
            ab_results = []
            if user_id:
                # Check if user is part of any active A/B tests
                for test_id, config in self.ab_testing.active_tests.items():
                    if (
                        config.control_model_name == model_name
                        or config.variant_model_name == model_name
                    ):
                        variant = self.ab_testing.assign_variant(test_id, user_id)

                        if variant:
                            # Record observation
                            primary_metric_value = np.mean(
                                predictions
                            )  # Simplified - should be actual performance metric
                            self.ab_testing.record_observation(
                                test_id=test_id,
                                user_id=user_id,
                                variant=variant,
                                primary_metric_value=primary_metric_value,
                            )

                            ab_results.append(
                                {
                                    "test_id": test_id,
                                    "variant": variant,
                                    "recorded": True,
                                }
                            )

            monitoring_results["ab_testing"] = ab_results

            # 4. Determine overall status
            has_critical_drift = any(alert["severity"] == "critical" for alert in drift_alerts)
            has_high_alerts = any(
                alert["severity"] in ["high", "critical"] for alert in drift_alerts
            )

            if has_critical_drift:
                monitoring_results["status"] = "critical"
            elif has_high_alerts:
                monitoring_results["status"] = "concerning"
            elif drift_alerts:
                monitoring_results["status"] = "monitor"

            monitoring_results["alerts"] = drift_alerts

        except Exception as e:
            self.logger.error(f"Error during model monitoring for {model_name}: {e}")
            monitoring_results["error"] = str(e)
            monitoring_results["status"] = "error"

        return monitoring_results

    def generate_comprehensive_report(self, model_name: Optional[str] = None) -> MonitoringReport:
        """Generate comprehensive monitoring report for model(s)"""

        timestamp = datetime.now()

        if model_name:
            # Single model report
            return self._generate_single_model_report(model_name, timestamp)
        else:
            # System-wide report
            return self._generate_system_wide_report(timestamp)

    def _generate_single_model_report(
        self, model_name: str, timestamp: datetime
    ) -> MonitoringReport:
        """Generate report for a single model"""

        # Get performance summary
        performance_summary = self.performance_tracker.get_model_performance_summary(
            model_name, hours=24
        )

        # Get drift summary
        drift_summary = self.drift_detector.get_drift_summary(hours=24)

        # Get A/B test status
        ab_tests = []
        for test_config in self.ab_testing.active_tests.values():
            if (
                test_config.control_model_name == model_name
                or test_config.variant_model_name == model_name
            ):
                test_status = self.ab_testing.get_test_status(test_config.test_id)
                if test_status:
                    ab_tests.append(test_status)

        # Determine status and recommendations
        performance_status = performance_summary.get("status", "unknown")

        drift_status = self._determine_drift_status(drift_summary)

        overall_status, recommendations, confidence = self._determine_overall_assessment(
            performance_status, drift_status, drift_summary, performance_summary
        )

        return MonitoringReport(
            timestamp=timestamp,
            model_name=model_name,
            performance_status=performance_status,
            performance_metrics=performance_summary.get("metrics", {}),
            performance_alerts=[],  # Would need to fetch from tracker
            drift_status=drift_status,
            drift_alerts=drift_summary.get("recent_alerts", []),
            ab_tests_active=ab_tests,
            overall_status=overall_status,
            recommendations=recommendations,
            confidence_score=confidence,
        )

    def _generate_system_wide_report(self, timestamp: datetime) -> MonitoringReport:
        """Generate system-wide monitoring report"""

        # Get all model statuses
        all_models_status = self.performance_tracker.get_all_models_status(hours=24)

        # Aggregate metrics and alerts
        all_performance_metrics = {}
        all_performance_alerts = []
        worst_performance_status = "healthy"

        for model_name, model_status in all_models_status.items():
            all_performance_metrics[model_name] = model_status.get("metrics", {})

            # Track worst status
            status = model_status.get("status", "unknown")
            if status in ["critical", "degraded", "concerning"] and status != "healthy":
                if worst_performance_status == "healthy":
                    worst_performance_status = status
                elif status == "critical":
                    worst_performance_status = "critical"

        # Get system-wide drift summary
        drift_summary = self.drift_detector.get_drift_summary(hours=24)
        drift_status = self._determine_drift_status(drift_summary)

        # Get all active A/B tests
        all_ab_tests = self.ab_testing.list_tests(status_filter="running")

        # Overall assessment
        overall_status, recommendations, confidence = self._determine_overall_assessment(
            worst_performance_status, drift_status, drift_summary, {}
        )

        return MonitoringReport(
            timestamp=timestamp,
            model_name="system_wide",
            performance_status=worst_performance_status,
            performance_metrics=all_performance_metrics,
            performance_alerts=all_performance_alerts,
            drift_status=drift_status,
            drift_alerts=drift_summary.get("recent_alerts", []),
            ab_tests_active=all_ab_tests,
            overall_status=overall_status,
            recommendations=recommendations,
            confidence_score=confidence,
        )

    def _determine_drift_status(self, drift_summary: Dict[str, Any]) -> str:
        """Determine drift status from summary"""

        total_alerts = drift_summary.get("total_alerts", 0)
        severity_dist = drift_summary.get("severity_distribution", {})

        if severity_dist.get("critical", 0) > 0:
            return "critical_drift"
        elif severity_dist.get("high", 0) > 0:
            return "significant_drift"
        elif total_alerts > 5:  # Multiple medium/low alerts
            return "minor_drift"
        else:
            return "stable"

    def _determine_overall_assessment(
        self,
        performance_status: str,
        drift_status: str,
        drift_summary: Dict[str, Any],
        performance_summary: Dict[str, Any],
    ) -> Tuple[str, List[str], float]:
        """Determine overall assessment and recommendations"""

        recommendations = []
        confidence = 0.8  # Default confidence

        # Critical conditions
        if performance_status == "critical" or drift_status == "critical_drift":
            overall_status = "action_required"
            recommendations.append("URGENT: Stop automated trading immediately")
            recommendations.append("Investigate critical performance/drift issues")
            recommendations.append("Consider model rollback or emergency retraining")
            confidence = 0.9

        # High priority conditions
        elif performance_status == "degraded" or drift_status == "significant_drift":
            overall_status = "investigate"
            recommendations.append("Investigate performance degradation within 24 hours")
            recommendations.append("Review recent market conditions and data quality")
            recommendations.append("Schedule model retraining within 48-72 hours")
            confidence = 0.85

        # Medium priority conditions
        elif performance_status == "concerning" or drift_status == "minor_drift":
            overall_status = "monitor"
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Review feature importance and data sources")
            recommendations.append("Consider gradual model updates")
            confidence = 0.75

        # Healthy state
        else:
            overall_status = "healthy"
            recommendations.append("System operating normally")
            recommendations.append("Continue regular monitoring schedule")

        # Add specific recommendations from drift summary
        drift_recommendations = drift_summary.get("recommendations", [])
        recommendations.extend(drift_recommendations[:2])  # Top 2 drift recommendations

        return overall_status, recommendations, confidence

    def save_report(self, report: MonitoringReport):
        """Save monitoring report to file"""

        report_filename = f"monitoring_report_{report.model_name}_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.reports_dir / report_filename

        # Convert report to dictionary for serialization
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "model_name": report.model_name,
            "performance_status": report.performance_status,
            "performance_metrics": report.performance_metrics,
            "performance_alerts": report.performance_alerts,
            "drift_status": report.drift_status,
            "drift_alerts": report.drift_alerts,
            "ab_tests_active": report.ab_tests_active,
            "overall_status": report.overall_status,
            "recommendations": report.recommendations,
            "confidence_score": report.confidence_score,
        }

        try:
            with open(report_path, "w") as f:
                json.dump(report_dict, f, indent=2)

            self.logger.info(f"Monitoring report saved: {report_path}")

            # Clean up old reports
            self._cleanup_old_reports()

        except Exception as e:
            self.logger.error(f"Error saving monitoring report: {e}")

    def _cleanup_old_reports(self):
        """Clean up old monitoring reports"""

        cutoff_date = datetime.now() - timedelta(days=self.report_retention_days)

        try:
            for report_file in self.reports_dir.glob("monitoring_report_*.json"):
                if report_file.stat().st_mtime < cutoff_date.timestamp():
                    report_file.unlink()

        except Exception as e:
            self.logger.error(f"Error cleaning up old reports: {e}")

    def start_automated_monitoring(self):
        """Start automated monitoring with scheduled reports"""

        if self.is_monitoring:
            self.logger.warning("Monitoring already running")
            return

        self.logger.info(
            f"Starting automated monitoring (interval: {self.monitoring_interval} minutes)"
        )

        # Schedule regular monitoring
        schedule.every(self.monitoring_interval).minutes.do(self._run_scheduled_monitoring)

        # Schedule daily comprehensive reports
        schedule.every().day.at("08:00").do(self._generate_daily_report)

        self.is_monitoring = True

        # Start monitoring loop in background
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self):
        """Background monitoring loop"""

        while self.is_monitoring:
            try:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    def _run_scheduled_monitoring(self):
        """Run scheduled monitoring checks"""

        try:
            # Update baselines
            self.performance_tracker.update_baselines()

            # Analyze any active A/B tests
            for test_id in list(self.ab_testing.active_tests.keys()):
                result = self.ab_testing.analyze_test(test_id)
                if result and result.conclusion.value != "inconclusive":
                    self.logger.info(f"A/B test {test_id} analysis: {result.conclusion.value}")

            self.logger.info("Scheduled monitoring checks completed")

        except Exception as e:
            self.logger.error(f"Error in scheduled monitoring: {e}")

    def _generate_daily_report(self):
        """Generate and save daily comprehensive report"""

        try:
            report = self.generate_comprehensive_report()
            self.save_report(report)

            # Log summary
            self.logger.info(
                f"Daily report generated - Status: {report.overall_status}, "
                f"Confidence: {report.confidence_score:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")

    def stop_automated_monitoring(self):
        """Stop automated monitoring"""

        if not self.is_monitoring:
            return

        self.logger.info("Stopping automated monitoring")

        self.is_monitoring = False
        schedule.clear()

        if self.monitoring_task:
            self.monitoring_task.cancel()

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""

        # Get recent reports
        recent_reports = []
        for report_file in sorted(self.reports_dir.glob("monitoring_report_*.json"), reverse=True)[
            :10
        ]:
            try:
                with open(report_file, "r") as f:
                    report_data = json.load(f)
                    recent_reports.append(report_data)
            except Exception as e:
                self.logger.warning(f"Error loading report {report_file}: {e}")

        # Get current status
        current_report = self.generate_comprehensive_report()

        # Get active A/B tests summary
        active_tests = self.ab_testing.list_tests(status_filter="running")

        dashboard_data = {
            "current_status": {
                "overall_status": current_report.overall_status,
                "performance_status": current_report.performance_status,
                "drift_status": current_report.drift_status,
                "confidence_score": current_report.confidence_score,
                "last_updated": current_report.timestamp.isoformat(),
            },
            "recent_reports": recent_reports,
            "active_ab_tests": len(active_tests),
            "recommendations": current_report.recommendations,
            "monitoring_active": self.is_monitoring,
        }

        return dashboard_data
