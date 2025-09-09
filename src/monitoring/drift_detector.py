"""
Model Drift Detection System

Detects data drift and concept drift in ML models using statistical tests
and monitoring model performance degradation over time.
"""

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from evidently.model_profile import Profile
    from evidently.model_profile.sections import (
        DataDriftProfileSection,
        RegressionPerformanceProfileSection,
    )
    from evidently.pipeline.column_mapping import ColumnMapping

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    warnings.warn("Evidently not available. Some drift detection features will be limited.")

from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class DriftAlert:
    """Alert for detected drift"""

    timestamp: datetime
    drift_type: str  # 'data_drift', 'concept_drift', 'performance_drift'
    severity: str  # 'low', 'medium', 'high', 'critical'
    feature: Optional[str]
    metric_name: str
    current_value: float
    reference_value: float
    threshold: float
    description: str
    recommendation: str


class DriftDetector:
    """Comprehensive drift detection for trading models"""

    def __init__(
        self,
        reference_window_days: int = 30,
        detection_window_days: int = 7,
        significance_level: float = 0.05,
        performance_threshold: float = 0.15,
        alerts_file: Optional[Path] = None,
    ):
        """
        Initialize drift detector

        Args:
            reference_window_days: Days of reference data for baseline
            detection_window_days: Days of current data for comparison
            significance_level: Statistical significance threshold
            performance_threshold: Performance degradation threshold (15%)
            alerts_file: File to store drift alerts
        """
        self.reference_window_days = reference_window_days
        self.detection_window_days = detection_window_days
        self.significance_level = significance_level
        self.performance_threshold = performance_threshold
        self.alerts_file = alerts_file or Path("logs/drift_alerts.json")

        self.logger = logging.getLogger(__name__)
        self.reference_stats = {}
        self.alerts_history = []

        # Ensure alerts directory exists
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)

    def set_reference_data(
        self,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ):
        """Set reference baseline data for comparison"""

        self.logger.info(f"Setting reference data with {len(features_df)} samples")

        # Store reference statistics
        self.reference_stats = {
            "features": {
                col: {
                    "mean": features_df[col].mean(),
                    "std": features_df[col].std(),
                    "min": features_df[col].min(),
                    "max": features_df[col].max(),
                    "quantiles": features_df[col].quantile([0.25, 0.5, 0.75]).to_dict(),
                }
                for col in features_df.columns
            },
            "predictions": {
                "mean": np.mean(predictions),
                "std": np.std(predictions),
                "distribution": predictions,  # Store for distribution tests
            },
            "timestamp": datetime.now(),
            "sample_size": len(features_df),
        }

        if targets is not None:
            self.reference_stats["performance"] = {
                "mse": mean_squared_error(targets, predictions),
                "mae": mean_absolute_error(targets, predictions),
                "targets_mean": np.mean(targets),
                "targets_std": np.std(targets),
            }

        self.logger.info("Reference data statistics computed successfully")

    def detect_data_drift(
        self, current_features: pd.DataFrame, feature_names: Optional[List[str]] = None
    ) -> List[DriftAlert]:
        """
        Detect data drift using statistical tests

        Returns list of drift alerts for features that have drifted
        """
        if not self.reference_stats:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        alerts = []
        feature_names = feature_names or current_features.columns.tolist()

        for feature in feature_names:
            if feature not in current_features.columns:
                continue

            if feature not in self.reference_stats["features"]:
                self.logger.warning(f"Feature {feature} not in reference data")
                continue

            current_values = current_features[feature].dropna()
            if len(current_values) == 0:
                continue

            ref_stats = self.reference_stats["features"][feature]

            # Kolmogorov-Smirnov test for distribution drift
            # Generate reference distribution samples for comparison
            ref_mean, ref_std = ref_stats["mean"], ref_stats["std"]
            ref_samples = np.random.normal(ref_mean, ref_std, len(current_values))

            ks_statistic, p_value = stats.ks_2samp(ref_samples, current_values)

            if p_value < self.significance_level:
                # Calculate drift magnitude
                mean_drift = abs(current_values.mean() - ref_mean) / ref_std if ref_std > 0 else 0

                severity = self._calculate_drift_severity(mean_drift, p_value)

                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type="data_drift",
                    severity=severity,
                    feature=feature,
                    metric_name="ks_test_p_value",
                    current_value=p_value,
                    reference_value=self.significance_level,
                    threshold=self.significance_level,
                    description=f"Data drift detected in feature {feature}. "
                    f"KS test p-value: {p_value:.4f}, mean drift: {mean_drift:.3f} std devs",
                    recommendation=self._get_drift_recommendation(feature, "data_drift", severity),
                )
                alerts.append(alert)

        return alerts

    def detect_concept_drift(
        self, current_predictions: np.ndarray, current_targets: np.ndarray
    ) -> List[DriftAlert]:
        """
        Detect concept drift by comparing model performance

        Returns list of performance drift alerts
        """
        if "performance" not in self.reference_stats:
            self.logger.warning(
                "No reference performance data available for concept drift detection"
            )
            return []

        alerts = []
        ref_perf = self.reference_stats["performance"]

        # Calculate current performance
        current_mse = mean_squared_error(current_targets, current_predictions)
        current_mae = mean_absolute_error(current_targets, current_predictions)

        # Check MSE drift
        mse_change = (current_mse - ref_perf["mse"]) / ref_perf["mse"]
        if mse_change > self.performance_threshold:
            severity = "critical" if mse_change > 0.5 else "high" if mse_change > 0.3 else "medium"

            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type="concept_drift",
                severity=severity,
                feature=None,
                metric_name="mse_degradation",
                current_value=current_mse,
                reference_value=ref_perf["mse"],
                threshold=ref_perf["mse"] * (1 + self.performance_threshold),
                description=f"Model performance degraded. MSE increased by {mse_change:.1%} "
                f"(from {ref_perf['mse']:.4f} to {current_mse:.4f})",
                recommendation=self._get_drift_recommendation(None, "concept_drift", severity),
            )
            alerts.append(alert)

        # Check MAE drift
        mae_change = (current_mae - ref_perf["mae"]) / ref_perf["mae"]
        if mae_change > self.performance_threshold:
            severity = "critical" if mae_change > 0.5 else "high" if mae_change > 0.3 else "medium"

            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type="concept_drift",
                severity=severity,
                feature=None,
                metric_name="mae_degradation",
                current_value=current_mae,
                reference_value=ref_perf["mae"],
                threshold=ref_perf["mae"] * (1 + self.performance_threshold),
                description=f"Model performance degraded. MAE increased by {mae_change:.1%} "
                f"(from {ref_perf['mae']:.4f} to {current_mae:.4f})",
                recommendation=self._get_drift_recommendation(None, "concept_drift", severity),
            )
            alerts.append(alert)

        return alerts

    def detect_prediction_drift(self, current_predictions: np.ndarray) -> List[DriftAlert]:
        """
        Detect drift in model predictions distribution

        Returns list of prediction drift alerts
        """
        if "predictions" not in self.reference_stats:
            return []

        alerts = []
        ref_pred_stats = self.reference_stats["predictions"]

        # Statistical tests on predictions
        ref_predictions = ref_pred_stats["distribution"]

        # Ensure we have enough samples
        min_samples = min(len(ref_predictions), len(current_predictions))
        if min_samples < 30:
            self.logger.warning("Insufficient samples for prediction drift detection")
            return alerts

        # KS test on predictions
        ks_stat, p_value = stats.ks_2samp(
            ref_predictions[:min_samples], current_predictions[:min_samples]
        )

        if p_value < self.significance_level:
            # Calculate magnitude of shift
            mean_shift = abs(np.mean(current_predictions) - ref_pred_stats["mean"])
            std_shift = abs(np.std(current_predictions) - ref_pred_stats["std"])

            severity = self._calculate_drift_severity(mean_shift / ref_pred_stats["std"], p_value)

            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type="prediction_drift",
                severity=severity,
                feature=None,
                metric_name="prediction_ks_test",
                current_value=p_value,
                reference_value=self.significance_level,
                threshold=self.significance_level,
                description=f"Prediction distribution drift detected. KS p-value: {p_value:.4f}, "
                f"mean shift: {mean_shift:.4f}, std shift: {std_shift:.4f}",
                recommendation=self._get_drift_recommendation(None, "prediction_drift", severity),
            )
            alerts.append(alert)

        return alerts

    def run_comprehensive_drift_check(
        self,
        current_features: pd.DataFrame,
        current_predictions: np.ndarray,
        current_targets: Optional[np.ndarray] = None,
    ) -> Dict[str, List[DriftAlert]]:
        """
        Run all drift detection checks and return comprehensive results
        """
        results = {"data_drift": [], "concept_drift": [], "prediction_drift": []}

        try:
            # Data drift detection
            results["data_drift"] = self.detect_data_drift(current_features)

            # Prediction drift detection
            results["prediction_drift"] = self.detect_prediction_drift(current_predictions)

            # Concept drift detection (only if targets available)
            if current_targets is not None:
                results["concept_drift"] = self.detect_concept_drift(
                    current_predictions, current_targets
                )

            # Store all alerts
            all_alerts = []
            for drift_type, alerts in results.items():
                all_alerts.extend(alerts)

            self._store_alerts(all_alerts)

            # Log summary
            total_alerts = sum(len(alerts) for alerts in results.values())
            if total_alerts > 0:
                self.logger.warning(f"Drift detection found {total_alerts} alerts")
                for drift_type, alerts in results.items():
                    if alerts:
                        severities = [a.severity for a in alerts]
                        self.logger.warning(
                            f"{drift_type}: {len(alerts)} alerts - {dict(pd.Series(severities).value_counts())}"
                        )
            else:
                self.logger.info("No drift detected")

        except Exception as e:
            self.logger.error(f"Error in drift detection: {str(e)}")

        return results

    def _calculate_drift_severity(self, magnitude: float, p_value: float) -> str:
        """Calculate drift severity based on magnitude and statistical significance"""
        if p_value < 0.001 or magnitude > 3:
            return "critical"
        elif p_value < 0.01 or magnitude > 2:
            return "high"
        elif p_value < 0.05 or magnitude > 1:
            return "medium"
        else:
            return "low"

    def _get_drift_recommendation(
        self, feature: Optional[str], drift_type: str, severity: str
    ) -> str:
        """Get recommendation based on drift type and severity"""

        recommendations = {
            "data_drift": {
                "low": f"Monitor feature {feature} closely. Consider feature importance analysis.",
                "medium": f"Investigate data source changes for feature {feature}. Update feature preprocessing.",
                "high": f"Feature {feature} shows significant drift. Consider model retraining or feature engineering.",
                "critical": f"Critical drift in feature {feature}. Immediate model retraining recommended.",
            },
            "concept_drift": {
                "low": "Monitor model performance. Consider increasing validation frequency.",
                "medium": "Performance degradation detected. Schedule model retraining within 1 week.",
                "high": "Significant performance drop. Retrain model within 2-3 days.",
                "critical": "Critical performance degradation. Stop automated trading and retrain immediately.",
            },
            "prediction_drift": {
                "low": "Prediction patterns changing. Monitor closely.",
                "medium": "Model output distribution shift detected. Review recent market conditions.",
                "high": "Significant prediction drift. Consider model ensemble or retraining.",
                "critical": "Critical prediction drift. Model may be unreliable - immediate attention required.",
            },
        }

        return recommendations.get(drift_type, {}).get(severity, "Monitor and investigate further.")

    def _store_alerts(self, alerts: List[DriftAlert]):
        """Store drift alerts to file"""
        if not alerts:
            return

        alert_dicts = []
        for alert in alerts:
            alert_dict = {
                "timestamp": alert.timestamp.isoformat(),
                "drift_type": alert.drift_type,
                "severity": alert.severity,
                "feature": alert.feature,
                "metric_name": alert.metric_name,
                "current_value": float(alert.current_value),
                "reference_value": float(alert.reference_value),
                "threshold": float(alert.threshold),
                "description": alert.description,
                "recommendation": alert.recommendation,
            }
            alert_dicts.append(alert_dict)

        # Load existing alerts
        existing_alerts = []
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, "r") as f:
                    existing_alerts = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading existing alerts: {e}")

        # Append new alerts
        existing_alerts.extend(alert_dicts)

        # Keep only last 1000 alerts to prevent file bloat
        if len(existing_alerts) > 1000:
            existing_alerts = existing_alerts[-1000:]

        # Save updated alerts
        try:
            with open(self.alerts_file, "w") as f:
                json.dump(existing_alerts, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving alerts: {e}")

    def get_recent_alerts(
        self, hours: int = 24, severity_filter: Optional[str] = None
    ) -> List[Dict]:
        """Get recent alerts within specified time window"""
        if not self.alerts_file.exists():
            return []

        try:
            with open(self.alerts_file, "r") as f:
                all_alerts = json.load(f)
        except Exception:
            return []

        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = []

        for alert in all_alerts:
            alert_time = datetime.fromisoformat(alert["timestamp"])
            if alert_time >= cutoff_time:
                if severity_filter is None or alert["severity"] == severity_filter:
                    recent_alerts.append(alert)

        return recent_alerts

    def get_drift_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of drift detection results"""
        recent_alerts = self.get_recent_alerts(hours)

        if not recent_alerts:
            return {
                "total_alerts": 0,
                "drift_types": {},
                "severity_distribution": {},
                "most_drifted_features": [],
                "recommendations": [],
            }

        # Analyze alerts
        drift_types = {}
        severities = {}
        feature_counts = {}

        for alert in recent_alerts:
            # Count by drift type
            drift_type = alert["drift_type"]
            drift_types[drift_type] = drift_types.get(drift_type, 0) + 1

            # Count by severity
            severity = alert["severity"]
            severities[severity] = severities.get(severity, 0) + 1

            # Count by feature
            feature = alert.get("feature")
            if feature:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Get top recommendations
        critical_alerts = [a for a in recent_alerts if a["severity"] == "critical"]
        high_alerts = [a for a in recent_alerts if a["severity"] == "high"]

        recommendations = []
        for alert in critical_alerts[:3]:  # Top 3 critical
            recommendations.append(alert["recommendation"])
        for alert in high_alerts[:2]:  # Top 2 high
            recommendations.append(alert["recommendation"])

        return {
            "total_alerts": len(recent_alerts),
            "drift_types": drift_types,
            "severity_distribution": severities,
            "most_drifted_features": sorted(
                feature_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "recommendations": list(set(recommendations)),  # Remove duplicates
        }
