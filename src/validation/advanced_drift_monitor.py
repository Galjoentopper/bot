"""Advanced feature drift monitoring with multiple detection algorithms."""

import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Statistical tests
try:
    from scipy import stats
    from scipy.spatial.distance import wasserstein_distance

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, some drift detection methods will be disabled")


@dataclass
class DriftAlert:
    """Drift detection alert."""

    timestamp: str
    symbol: str
    model_type: str
    drift_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_name: str
    metric_value: float
    threshold: float
    description: str
    feature_names: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None


class AdvancedFeatureDriftMonitor:
    """Advanced feature drift monitoring with multiple detection algorithms."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.window_size = self.config.get("window_size", 1000)
        self.baseline_size = self.config.get("baseline_size", 5000)
        self.alert_threshold = self.config.get("alert_threshold", 0.05)
        self.severe_threshold = self.config.get("severe_threshold", 0.01)
        self.critical_threshold = self.config.get("critical_threshold", 0.001)

        # Data storage
        self.baseline_data = defaultdict(lambda: defaultdict(deque))
        self.current_data = defaultdict(lambda: defaultdict(deque))
        self.drift_history = defaultdict(list)
        self.alerts = []

        # Drift detection methods
        self.detection_methods = {
            "ks_test": self._ks_test_drift,
            "wasserstein": self._wasserstein_drift,
            "psi": self._population_stability_index,
            "jensen_shannon": self._jensen_shannon_divergence,
            "chi_square": self._chi_square_test,
            "statistical_moments": self._statistical_moments_drift,
            "distribution_shift": self._distribution_shift_detection,
            "feature_importance": self._feature_importance_drift,
        }

    def add_baseline_data(self, data: pd.DataFrame, symbol: str, model_type: str) -> None:
        """Add data to baseline for drift comparison."""
        key = f"{symbol}_{model_type}"

        for column in data.columns:
            baseline_queue = self.baseline_data[key][column]

            # Add new data
            for value in data[column].values:
                if not np.isnan(value) and np.isfinite(value):
                    baseline_queue.append(float(value))

            # Maintain baseline size
            while len(baseline_queue) > self.baseline_size:
                baseline_queue.popleft()

        self.logger.debug(f"Added {len(data)} baseline samples for {key}")

    def add_current_data(self, data: pd.DataFrame, symbol: str, model_type: str) -> Dict[str, Any]:
        """Add current data and detect drift."""
        key = f"{symbol}_{model_type}"
        drift_results = {
            "symbol": symbol,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drift_scores": {},
            "alerts": [],
        }

        # Add data to current window
        for column in data.columns:
            current_queue = self.current_data[key][column]

            for value in data[column].values:
                if not np.isnan(value) and np.isfinite(value):
                    current_queue.append(float(value))

            # Maintain window size
            while len(current_queue) > self.window_size:
                current_queue.popleft()

        # Detect drift if we have enough data
        if self._has_sufficient_data(key):
            drift_results = self._detect_drift_comprehensive(symbol, model_type)

        return drift_results

    def _detect_drift_comprehensive(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Comprehensive drift detection using multiple methods."""
        key = f"{symbol}_{model_type}"
        results = {
            "symbol": symbol,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drift_scores": {},
            "method_results": {},
            "alerts": [],
        }

        baseline_data = self.baseline_data[key]
        current_data = self.current_data[key]

        # Run all detection methods
        for method_name, method_func in self.detection_methods.items():
            try:
                method_result = method_func(baseline_data, current_data, symbol, model_type)
                results["method_results"][method_name] = method_result

                # Check for drift
                if method_result.get("drift_detected", False):
                    results["drift_detected"] = True

                    # Create alert
                    alert = self._create_drift_alert(symbol, model_type, method_name, method_result)
                    results["alerts"].append(alert)
                    self.alerts.append(alert)

                # Store drift scores
                if "drift_score" in method_result:
                    results["drift_scores"][method_name] = method_result["drift_score"]

            except Exception as e:
                self.logger.error(f"Error in drift detection method {method_name}: {e}")
                results["method_results"][method_name] = {"error": str(e)}

        # Store results in history
        self.drift_history[key].append(results)

        # Limit history size
        if len(self.drift_history[key]) > 1000:
            self.drift_history[key] = self.drift_history[key][-1000:]

        return results

    def _ks_test_drift(
        self, baseline: Dict, current: Dict, symbol: str, model_type: str
    ) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for distribution drift."""
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}

        results = {
            "method": "ks_test",
            "drift_detected": False,
            "feature_results": {},
            "overall_drift_score": 0.0,
        }

        drift_scores = []

        for feature_name in baseline.keys():
            if (
                feature_name in current
                and len(baseline[feature_name]) > 0
                and len(current[feature_name]) > 0
            ):
                baseline_values = np.array(list(baseline[feature_name]))
                current_values = np.array(list(current[feature_name]))

                # Perform KS test
                ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)

                feature_result = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "drift_detected": p_value < self.alert_threshold,
                }

                results["feature_results"][feature_name] = feature_result
                drift_scores.append(ks_stat)

                if p_value < self.alert_threshold:
                    results["drift_detected"] = True

        if drift_scores:
            results["overall_drift_score"] = float(np.mean(drift_scores))
            results["drift_score"] = results["overall_drift_score"]

        return results

    def _wasserstein_drift(
        self, baseline: Dict, current: Dict, symbol: str, model_type: str
    ) -> Dict[str, Any]:
        """Wasserstein distance for distribution drift."""
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}

        results = {
            "method": "wasserstein",
            "drift_detected": False,
            "feature_results": {},
            "overall_drift_score": 0.0,
        }

        drift_scores = []

        for feature_name in baseline.keys():
            if (
                feature_name in current
                and len(baseline[feature_name]) > 0
                and len(current[feature_name]) > 0
            ):
                baseline_values = np.array(list(baseline[feature_name]))
                current_values = np.array(list(current[feature_name]))

                # Calculate Wasserstein distance
                distance = wasserstein_distance(baseline_values, current_values)

                # Normalize by baseline standard deviation
                baseline_std = np.std(baseline_values)
                normalized_distance = distance / (baseline_std + 1e-8)

                feature_result = {
                    "wasserstein_distance": float(distance),
                    "normalized_distance": float(normalized_distance),
                    "drift_detected": normalized_distance > 0.5,  # Threshold
                }

                results["feature_results"][feature_name] = feature_result
                drift_scores.append(normalized_distance)

                if normalized_distance > 0.5:
                    results["drift_detected"] = True

        if drift_scores:
            results["overall_drift_score"] = float(np.mean(drift_scores))
            results["drift_score"] = results["overall_drift_score"]

        return results

    def _population_stability_index(
        self, baseline: Dict, current: Dict, symbol: str, model_type: str
    ) -> Dict[str, Any]:
        """Population Stability Index (PSI) for drift detection."""
        results = {
            "method": "psi",
            "drift_detected": False,
            "feature_results": {},
            "overall_drift_score": 0.0,
        }

        drift_scores = []

        for feature_name in baseline.keys():
            if (
                feature_name in current
                and len(baseline[feature_name]) > 0
                and len(current[feature_name]) > 0
            ):
                baseline_values = np.array(list(baseline[feature_name]))
                current_values = np.array(list(current[feature_name]))

                # Calculate PSI
                psi_score = self._calculate_psi(baseline_values, current_values)

                feature_result = {
                    "psi_score": float(psi_score),
                    "drift_detected": psi_score > 0.2,  # Standard PSI threshold
                }

                results["feature_results"][feature_name] = feature_result
                drift_scores.append(psi_score)

                if psi_score > 0.2:
                    results["drift_detected"] = True

        if drift_scores:
            results["overall_drift_score"] = float(np.mean(drift_scores))
            results["drift_score"] = results["overall_drift_score"]

        return results

    def _jensen_shannon_divergence(
        self, baseline: Dict, current: Dict, symbol: str, model_type: str
    ) -> Dict[str, Any]:
        """Jensen-Shannon divergence for distribution comparison."""
        results = {
            "method": "jensen_shannon",
            "drift_detected": False,
            "feature_results": {},
            "overall_drift_score": 0.0,
        }

        drift_scores = []

        for feature_name in baseline.keys():
            if (
                feature_name in current
                and len(baseline[feature_name]) > 0
                and len(current[feature_name]) > 0
            ):
                baseline_values = np.array(list(baseline[feature_name]))
                current_values = np.array(list(current[feature_name]))

                # Calculate JS divergence
                js_div = self._calculate_js_divergence(baseline_values, current_values)

                feature_result = {
                    "js_divergence": float(js_div),
                    "drift_detected": js_div > 0.1,  # Threshold
                }

                results["feature_results"][feature_name] = feature_result
                drift_scores.append(js_div)

                if js_div > 0.1:
                    results["drift_detected"] = True

        if drift_scores:
            results["overall_drift_score"] = float(np.mean(drift_scores))
            results["drift_score"] = results["overall_drift_score"]

        return results

    def _chi_square_test(
        self, baseline: Dict, current: Dict, symbol: str, model_type: str
    ) -> Dict[str, Any]:
        """Chi-square test for categorical drift detection."""
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}

        results = {
            "method": "chi_square",
            "drift_detected": False,
            "feature_results": {},
            "overall_drift_score": 0.0,
        }

        drift_scores = []

        for feature_name in baseline.keys():
            if (
                feature_name in current
                and len(baseline[feature_name]) > 0
                and len(current[feature_name]) > 0
            ):
                baseline_values = np.array(list(baseline[feature_name]))
                current_values = np.array(list(current[feature_name]))

                # Discretize continuous values
                baseline_hist, bins = np.histogram(baseline_values, bins=10)
                current_hist, _ = np.histogram(current_values, bins=bins)

                # Avoid zero frequencies
                baseline_hist = baseline_hist + 1
                current_hist = current_hist + 1

                # Chi-square test
                chi2_stat, p_value = stats.chisquare(current_hist, baseline_hist)

                feature_result = {
                    "chi2_statistic": float(chi2_stat),
                    "p_value": float(p_value),
                    "drift_detected": p_value < self.alert_threshold,
                }

                results["feature_results"][feature_name] = feature_result
                drift_scores.append(chi2_stat)

                if p_value < self.alert_threshold:
                    results["drift_detected"] = True

        if drift_scores:
            results["overall_drift_score"] = float(np.mean(drift_scores))
            results["drift_score"] = results["overall_drift_score"]

        return results

    def _statistical_moments_drift(
        self, baseline: Dict, current: Dict, symbol: str, model_type: str
    ) -> Dict[str, Any]:
        """Statistical moments comparison for drift detection."""
        results = {
            "method": "statistical_moments",
            "drift_detected": False,
            "feature_results": {},
            "overall_drift_score": 0.0,
        }

        drift_scores = []

        for feature_name in baseline.keys():
            if (
                feature_name in current
                and len(baseline[feature_name]) > 0
                and len(current[feature_name]) > 0
            ):
                baseline_values = np.array(list(baseline[feature_name]))
                current_values = np.array(list(current[feature_name]))

                # Calculate moments
                baseline_moments = self._calculate_moments(baseline_values)
                current_moments = self._calculate_moments(current_values)

                # Compare moments
                moment_diffs = []
                for i, (b_moment, c_moment) in enumerate(zip(baseline_moments, current_moments)):
                    if b_moment != 0:
                        diff = abs(c_moment - b_moment) / abs(b_moment)
                    else:
                        diff = abs(c_moment)
                    moment_diffs.append(diff)

                avg_diff = np.mean(moment_diffs)

                feature_result = {
                    "baseline_moments": baseline_moments,
                    "current_moments": current_moments,
                    "moment_differences": moment_diffs,
                    "average_difference": float(avg_diff),
                    "drift_detected": avg_diff > 0.2,  # Threshold
                }

                results["feature_results"][feature_name] = feature_result
                drift_scores.append(avg_diff)

                if avg_diff > 0.2:
                    results["drift_detected"] = True

        if drift_scores:
            results["overall_drift_score"] = float(np.mean(drift_scores))
            results["drift_score"] = results["overall_drift_score"]

        return results

    def _distribution_shift_detection(
        self, baseline: Dict, current: Dict, symbol: str, model_type: str
    ) -> Dict[str, Any]:
        """Advanced distribution shift detection."""
        results = {
            "method": "distribution_shift",
            "drift_detected": False,
            "feature_results": {},
            "overall_drift_score": 0.0,
        }

        drift_scores = []

        for feature_name in baseline.keys():
            if (
                feature_name in current
                and len(baseline[feature_name]) > 0
                and len(current[feature_name]) > 0
            ):
                baseline_values = np.array(list(baseline[feature_name]))
                current_values = np.array(list(current[feature_name]))

                # Multiple shift detection metrics
                shift_score = self._calculate_distribution_shift(baseline_values, current_values)

                feature_result = {
                    "shift_score": float(shift_score),
                    "drift_detected": shift_score > 0.3,  # Threshold
                }

                results["feature_results"][feature_name] = feature_result
                drift_scores.append(shift_score)

                if shift_score > 0.3:
                    results["drift_detected"] = True

        if drift_scores:
            results["overall_drift_score"] = float(np.mean(drift_scores))
            results["drift_score"] = results["overall_drift_score"]

        return results

    def _feature_importance_drift(
        self, baseline: Dict, current: Dict, symbol: str, model_type: str
    ) -> Dict[str, Any]:
        """Feature importance-based drift detection."""
        results = {
            "method": "feature_importance",
            "drift_detected": False,
            "feature_results": {},
            "overall_drift_score": 0.0,
        }

        # This would require model-specific feature importance
        # For now, return a placeholder
        results["note"] = "Feature importance drift detection requires model integration"

        return results

    def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on baseline distribution
        _, bin_edges = np.histogram(baseline, bins=bins)

        # Calculate frequencies
        baseline_freq, _ = np.histogram(baseline, bins=bin_edges)
        current_freq, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions
        baseline_prop = baseline_freq / len(baseline)
        current_prop = current_freq / len(current)

        # Avoid division by zero
        baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
        current_prop = np.where(current_prop == 0, 0.0001, current_prop)

        # Calculate PSI
        psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))

        return psi

    def _calculate_js_divergence(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence."""
        # Create histograms
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins = np.linspace(min_val, max_val, 50)

        baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
        current_hist, _ = np.histogram(current, bins=bins, density=True)

        # Normalize
        baseline_hist = baseline_hist / np.sum(baseline_hist)
        current_hist = current_hist / np.sum(current_hist)

        # Avoid zeros
        baseline_hist = np.where(baseline_hist == 0, 1e-8, baseline_hist)
        current_hist = np.where(current_hist == 0, 1e-8, current_hist)

        # Calculate JS divergence
        m = 0.5 * (baseline_hist + current_hist)
        js_div = 0.5 * np.sum(baseline_hist * np.log(baseline_hist / m)) + 0.5 * np.sum(
            current_hist * np.log(current_hist / m)
        )

        return js_div

    def _calculate_moments(self, data: np.ndarray) -> List[float]:
        """Calculate first four statistical moments."""
        return [
            float(np.mean(data)),  # Mean
            float(np.var(data)),  # Variance
            float(stats.skew(data)) if SCIPY_AVAILABLE else 0.0,  # Skewness
            float(stats.kurtosis(data)) if SCIPY_AVAILABLE else 0.0,  # Kurtosis
        ]

    def _calculate_distribution_shift(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Calculate comprehensive distribution shift score."""
        scores = []

        # Quantile-based comparison
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        baseline_quantiles = np.quantile(baseline, quantiles)
        current_quantiles = np.quantile(current, quantiles)

        quantile_diff = np.mean(
            np.abs(current_quantiles - baseline_quantiles) / (np.abs(baseline_quantiles) + 1e-8)
        )
        scores.append(quantile_diff)

        # Range comparison
        baseline_range = np.ptp(baseline)
        current_range = np.ptp(current)
        range_diff = abs(current_range - baseline_range) / (baseline_range + 1e-8)
        scores.append(range_diff)

        # IQR comparison
        baseline_iqr = np.percentile(baseline, 75) - np.percentile(baseline, 25)
        current_iqr = np.percentile(current, 75) - np.percentile(current, 25)
        iqr_diff = abs(current_iqr - baseline_iqr) / (baseline_iqr + 1e-8)
        scores.append(iqr_diff)

        return np.mean(scores)

    def _create_drift_alert(
        self, symbol: str, model_type: str, method: str, result: Dict[str, Any]
    ) -> DriftAlert:
        """Create a drift alert from detection results."""
        drift_score = result.get("drift_score", result.get("overall_drift_score", 0.0))

        # Determine severity
        if drift_score > 0.8:
            severity = "critical"
        elif drift_score > 0.5:
            severity = "high"
        elif drift_score > 0.3:
            severity = "medium"
        else:
            severity = "low"

        # Generate recommendations
        recommendations = []
        if severity in ["high", "critical"]:
            recommendations.extend(
                [
                    "Consider retraining the model with recent data",
                    "Review feature engineering pipeline",
                    "Investigate data source changes",
                ]
            )
        elif severity == "medium":
            recommendations.extend(
                [
                    "Monitor model performance closely",
                    "Consider incremental model updates",
                ]
            )

        return DriftAlert(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            model_type=model_type,
            drift_type=method,
            severity=severity,
            metric_name=f"{method}_score",
            metric_value=drift_score,
            threshold=0.2,  # Default threshold
            description=f"Drift detected using {method} method",
            recommendations=recommendations,
        )

    def _has_sufficient_data(self, key: str) -> bool:
        """Check if we have sufficient data for drift detection."""
        baseline_data = self.baseline_data[key]
        current_data = self.current_data[key]

        if not baseline_data or not current_data:
            return False

        # Check if we have data for at least one feature
        for feature_name in baseline_data.keys():
            if (
                feature_name in current_data
                and len(baseline_data[feature_name]) >= 100
                and len(current_data[feature_name]) >= 50
            ):
                return True

        return False

    def get_drift_summary(self, symbol: str = None, model_type: str = None) -> Dict[str, Any]:
        """Get drift monitoring summary."""
        summary = {
            "total_alerts": len(self.alerts),
            "recent_alerts": [],
            "drift_trends": {},
            "model_status": {},
        }

        # Recent alerts (last 24 hours)
        recent_threshold = datetime.now() - timedelta(hours=24)
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert.timestamp)
            if alert_time > recent_threshold:
                if not symbol or alert.symbol == symbol:
                    if not model_type or alert.model_type == model_type:
                        summary["recent_alerts"].append(asdict(alert))

        # Model status summary
        for key, history in self.drift_history.items():
            if history:
                latest = history[-1]
                summary["model_status"][key] = {
                    "last_check": latest["timestamp"],
                    "drift_detected": latest["drift_detected"],
                    "drift_scores": latest.get("drift_scores", {}),
                }

        return summary

    def export_drift_data(self, filepath: str) -> None:
        """Export drift monitoring data to file."""
        export_data = {
            "alerts": [asdict(alert) for alert in self.alerts],
            "drift_history": dict(self.drift_history),
            "config": self.config,
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Drift data exported to {filepath}")


def create_advanced_drift_monitor(
    config: Dict[str, Any] = None,
) -> AdvancedFeatureDriftMonitor:
    """Create an advanced drift monitor instance."""
    return AdvancedFeatureDriftMonitor(config)
