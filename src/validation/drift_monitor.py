"""Real-time feature drift monitoring and alerting system."""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
import threading
import time
from dataclasses import dataclass

from ..utils.logger import Logger

@dataclass
class DriftAlert:
    """Drift detection alert."""
    timestamp: datetime
    model_type: str
    symbol: str
    drift_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metrics: Dict[str, Any]
    
class FeatureDriftMonitor:
    """Real-time feature drift monitoring system."""
    
    def __init__(self, config_dir: str = "./validation", window_size: int = 1000, alert_threshold: float = 0.1):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger("DriftMonitor")
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Data windows for each model-symbol combination
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats = {}
        
        # Alert system
        self.alerts = deque(maxlen=10000)  # Keep last 10k alerts
        self.alert_callbacks = []
        
        # Rate limiting for alerts to prevent log flooding
        self.alert_rate_limit = 10  # Max alerts per minute per model-symbol combination
        self.alert_timestamps = defaultdict(list)  # Track alert times
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # Drift detection methods
        self.drift_detectors = {
            'statistical': self._detect_statistical_drift,
            'distribution': self._detect_distribution_drift,
            'feature_importance': self._detect_feature_importance_drift,
            'correlation': self._detect_correlation_drift
        }
        
        # Thresholds for different drift types (relaxed to reduce false alarms)
        self.drift_thresholds = {
            'statistical': {
                'mean_shift': 5.0,      # Z-score threshold (increased from 2.0)
                'variance_ratio': 5.0,   # Variance ratio threshold (increased from 2.0)
                'low': 3.0,             # Increased from 1.5
                'medium': 5.0,          # Increased from 2.0
                'high': 8.0,            # Increased from 3.0
                'critical': 12.0        # Increased from 4.0
            },
            'distribution': {
                'ks_test': 0.001,       # Kolmogorov-Smirnov p-value (more strict)
                'js_divergence': 0.5,   # Jensen-Shannon divergence (increased from 0.1)
                'low': 0.2,             # Increased from 0.05
                'medium': 0.5,          # Increased from 0.1
                'high': 0.8,            # Increased from 0.2
                'critical': 1.0         # Increased from 0.3
            },
            'correlation': {
                'correlation_change': 0.7,  # Correlation coefficient change (increased from 0.3)
                'low': 0.3,             # Increased from 0.1
                'medium': 0.5,          # Increased from 0.2
                'high': 0.7,            # Increased from 0.3
                'critical': 0.9         # Increased from 0.5
            }
        }
        
        # Load existing baselines
        self._load_baselines()
        
    def add_alert_callback(self, callback):
        """Add callback function for drift alerts."""
        self.alert_callbacks.append(callback)
        
    def start_monitoring(self, check_interval: int = 60):
        """Start real-time drift monitoring."""
        if self.monitoring_active:
            self.logger.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.logger.info("Drift monitoring started")
        
    def stop_monitoring(self):
        """Stop real-time drift monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.logger.info("Drift monitoring stopped")
        
    def add_data_point(self, data: pd.DataFrame, model_type: str, symbol: str, timestamp: Optional[datetime] = None):
        """Add new data point for drift monitoring."""
        if timestamp is None:
            timestamp = datetime.now()
            
        key = f"{model_type}_{symbol}"
        
        with self.lock:
            # Calculate statistics for this data point
            stats = self._calculate_point_statistics(data, timestamp)
            
            # Add to window
            self.data_windows[key].append(stats)
            
            # Check for drift if we have enough data
            if len(self.data_windows[key]) >= 10:  # Minimum points for drift detection
                self._check_drift(key, model_type, symbol)
                
    def _calculate_point_statistics(self, data: pd.DataFrame, timestamp: datetime) -> Dict:
        """Calculate statistics for a single data point."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        stats = {
            'timestamp': timestamp,
            'row_count': len(data),
            'column_count': len(data.columns),
            'mean': numeric_data.mean().to_dict(),
            'std': numeric_data.std().to_dict(),
            'min': numeric_data.min().to_dict(),
            'max': numeric_data.max().to_dict(),
            'correlation_matrix': numeric_data.corr().to_dict(),
            'feature_names': list(data.columns)
        }
        
        return stats
        
    def _check_drift(self, key: str, model_type: str, symbol: str):
        """Check for drift in the data window."""
        window_data = list(self.data_windows[key])
        
        # Get baseline statistics
        baseline = self.baseline_stats.get(key)
        if not baseline:
            # Use first half of window as baseline
            mid_point = len(window_data) // 2
            baseline = self._aggregate_statistics(window_data[:mid_point])
            self.baseline_stats[key] = baseline
            
        # Use recent data for comparison
        recent_data = window_data[-10:]  # Last 10 points
        current_stats = self._aggregate_statistics(recent_data)
        
        # Run drift detection methods
        for detector_name, detector_func in self.drift_detectors.items():
            try:
                drift_result = detector_func(baseline, current_stats, key)
                if drift_result['drift_detected']:
                    self._create_alert(model_type, symbol, detector_name, drift_result)
            except Exception as e:
                self.logger.logger.error(f"Drift detection failed for {detector_name}: {e}")
                
    def _aggregate_statistics(self, stats_list: List[Dict]) -> Dict:
        """Aggregate statistics from multiple data points."""
        if not stats_list:
            return {}
            
        # Initialize aggregated stats
        agg_stats = {
            'timestamp_range': (stats_list[0]['timestamp'], stats_list[-1]['timestamp']),
            'sample_count': len(stats_list),
            'mean': {},
            'std': {},
            'min': {},
            'max': {},
            'correlation_matrix': {}
        }
        
        # Get all feature names
        all_features = set()
        for stats in stats_list:
            all_features.update(stats.get('feature_names', []))
            
        # Aggregate statistics for each feature
        for feature in all_features:
            means = [stats['mean'].get(feature, np.nan) for stats in stats_list if feature in stats.get('mean', {})]
            stds = [stats['std'].get(feature, np.nan) for stats in stats_list if feature in stats.get('std', {})]
            mins = [stats['min'].get(feature, np.nan) for stats in stats_list if feature in stats.get('min', {})]
            maxs = [stats['max'].get(feature, np.nan) for stats in stats_list if feature in stats.get('max', {})]
            
            # Remove NaN values
            means = [x for x in means if not np.isnan(x)]
            stds = [x for x in stds if not np.isnan(x)]
            mins = [x for x in mins if not np.isnan(x)]
            maxs = [x for x in maxs if not np.isnan(x)]
            
            if means:
                agg_stats['mean'][feature] = np.mean(means)
            if stds:
                agg_stats['std'][feature] = np.mean(stds)
            if mins:
                agg_stats['min'][feature] = np.min(mins)
            if maxs:
                agg_stats['max'][feature] = np.max(maxs)
                
        return agg_stats
        
    def _detect_statistical_drift(self, baseline: Dict, current: Dict, key: str) -> Dict:
        """Detect statistical drift using mean and variance changes."""
        result = {
            'drift_detected': False,
            'severity': 'low',
            'metrics': {},
            'details': []
        }
        
        thresholds = self.drift_thresholds['statistical']
        max_z_score = 0
        max_var_ratio = 0
        
        # Check mean shifts
        for feature in baseline.get('mean', {}):
            if feature in current.get('mean', {}):
                baseline_mean = baseline['mean'][feature]
                current_mean = current['mean'][feature]
                baseline_std = baseline.get('std', {}).get(feature, 1.0)
                
                if baseline_std > 0:
                    z_score = abs(current_mean - baseline_mean) / baseline_std
                    max_z_score = max(max_z_score, z_score)
                    
                    if z_score > thresholds['mean_shift']:
                        result['drift_detected'] = True
                        result['details'].append(f"Mean shift in {feature}: z-score = {z_score:.2f}")
                        
        # Check variance changes
        for feature in baseline.get('std', {}):
            if feature in current.get('std', {}):
                baseline_var = baseline['std'][feature] ** 2
                current_var = current['std'][feature] ** 2
                
                if baseline_var > 0:
                    var_ratio = current_var / baseline_var
                    max_var_ratio = max(max_var_ratio, var_ratio)
                    
                    if var_ratio > thresholds['variance_ratio'] or var_ratio < 1/thresholds['variance_ratio']:
                        result['drift_detected'] = True
                        result['details'].append(f"Variance change in {feature}: ratio = {var_ratio:.2f}")
                        
        # Determine severity
        max_drift = max(max_z_score, max_var_ratio)
        if max_drift >= thresholds['critical']:
            result['severity'] = 'critical'
        elif max_drift >= thresholds['high']:
            result['severity'] = 'high'
        elif max_drift >= thresholds['medium']:
            result['severity'] = 'medium'
        else:
            result['severity'] = 'low'
            
        result['metrics'] = {
            'max_z_score': max_z_score,
            'max_variance_ratio': max_var_ratio
        }
        
        return result
        
    def _detect_distribution_drift(self, baseline: Dict, current: Dict, key: str) -> Dict:
        """Detect distribution drift using statistical tests."""
        result = {
            'drift_detected': False,
            'severity': 'low',
            'metrics': {},
            'details': []
        }
        
        # This is a simplified version - in practice, you'd need the raw data
        # for proper distribution testing
        
        # For now, use mean and std to approximate distribution changes
        thresholds = self.drift_thresholds['distribution']
        max_divergence = 0
        
        for feature in baseline.get('mean', {}):
            if feature in current.get('mean', {}):
                # Simplified Jensen-Shannon divergence approximation
                baseline_mean = baseline['mean'][feature]
                current_mean = current['mean'][feature]
                baseline_std = baseline.get('std', {}).get(feature, 1.0)
                current_std = current.get('std', {}).get(feature, 1.0)
                
                # Simple divergence measure
                mean_diff = abs(baseline_mean - current_mean)
                std_diff = abs(baseline_std - current_std)
                divergence = (mean_diff + std_diff) / (baseline_std + 1e-8)
                
                max_divergence = max(max_divergence, divergence)
                
                if divergence > thresholds['js_divergence']:
                    result['drift_detected'] = True
                    result['details'].append(f"Distribution change in {feature}: divergence = {divergence:.3f}")
                    
        # Determine severity
        if max_divergence >= thresholds['critical']:
            result['severity'] = 'critical'
        elif max_divergence >= thresholds['high']:
            result['severity'] = 'high'
        elif max_divergence >= thresholds['medium']:
            result['severity'] = 'medium'
        else:
            result['severity'] = 'low'
            
        result['metrics'] = {'max_divergence': max_divergence}
        
        return result
        
    def _detect_feature_importance_drift(self, baseline: Dict, current: Dict, key: str) -> Dict:
        """Detect changes in feature importance patterns."""
        result = {
            'drift_detected': False,
            'severity': 'low',
            'metrics': {},
            'details': []
        }
        
        # This would require model-specific feature importance calculation
        # For now, use variance as a proxy for importance
        
        baseline_importance = {}
        current_importance = {}
        
        # Calculate relative importance based on variance
        baseline_vars = baseline.get('std', {})
        current_vars = current.get('std', {})
        
        if baseline_vars and current_vars:
            baseline_total = sum(v**2 for v in baseline_vars.values())
            current_total = sum(v**2 for v in current_vars.values())
            
            for feature in baseline_vars:
                if feature in current_vars and baseline_total > 0 and current_total > 0:
                    baseline_importance[feature] = (baseline_vars[feature]**2) / baseline_total
                    current_importance[feature] = (current_vars[feature]**2) / current_total
                    
            # Check for significant changes in relative importance
            max_importance_change = 0
            for feature in baseline_importance:
                if feature in current_importance:
                    change = abs(baseline_importance[feature] - current_importance[feature])
                    max_importance_change = max(max_importance_change, change)
                    
                    if change > 0.1:  # 10% change threshold
                        result['drift_detected'] = True
                        result['details'].append(f"Importance change in {feature}: {change:.3f}")
                        
            result['metrics'] = {'max_importance_change': max_importance_change}
            
        return result
        
    def _detect_correlation_drift(self, baseline: Dict, current: Dict, key: str) -> Dict:
        """Detect changes in feature correlation patterns."""
        result = {
            'drift_detected': False,
            'severity': 'low',
            'metrics': {},
            'details': []
        }
        
        # This is simplified - would need actual correlation matrices
        # For now, just flag as not implemented
        result['details'].append("Correlation drift detection not fully implemented")
        
        return result
        
    def _create_alert(self, model_type: str, symbol: str, drift_type: str, drift_result: Dict):
        """Create and process drift alert with rate limiting."""
        # Rate limiting check
        alert_key = f"{model_type}_{symbol}_{drift_type}"
        current_time = time.time()
        
        # Clean old timestamps (older than 1 minute)
        self.alert_timestamps[alert_key] = [
            ts for ts in self.alert_timestamps[alert_key] 
            if current_time - ts < 60
        ]
        
        # Check if we've exceeded rate limit
        if len(self.alert_timestamps[alert_key]) >= self.alert_rate_limit:
            # Skip this alert due to rate limiting
            return
            
        # Add current timestamp
        self.alert_timestamps[alert_key].append(current_time)
        
        alert = DriftAlert(
            timestamp=datetime.now(),
            model_type=model_type,
            symbol=symbol,
            drift_type=drift_type,
            severity=drift_result['severity'],
            message=f"{drift_type.title()} drift detected for {model_type} model on {symbol}",
            metrics=drift_result['metrics']
        )
        
        # Add to alerts queue
        self.alerts.append(alert)
        
        # Only log critical and high severity alerts to reduce noise
        if alert.severity in ['critical', 'high']:
            self.logger.logger.warning(
                f"DRIFT ALERT: {alert.message} (Severity: {alert.severity})"
            )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.logger.error(f"Alert callback failed: {e}")
                
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Perform periodic checks
                self._periodic_drift_check()
                time.sleep(check_interval)
            except Exception as e:
                self.logger.logger.error(f"Monitoring loop error: {e}")
                time.sleep(check_interval)
                
    def _periodic_drift_check(self):
        """Perform periodic drift checks on all monitored data."""
        with self.lock:
            for key in list(self.data_windows.keys()):
                if len(self.data_windows[key]) >= 10:
                    model_type, symbol = key.split('_', 1)
                    self._check_drift(key, model_type, symbol)
                    
    def _load_baselines(self):
        """Load existing baseline statistics."""
        baseline_file = self.config_dir / "drift_baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    self.baseline_stats = json.load(f)
                self.logger.logger.info(f"Loaded {len(self.baseline_stats)} baseline statistics")
            except Exception as e:
                self.logger.logger.error(f"Failed to load baselines: {e}")
                
    def save_baselines(self):
        """Save current baseline statistics."""
        baseline_file = self.config_dir / "drift_baselines.json"
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_baselines = {}
            for key, baseline in self.baseline_stats.items():
                serializable_baseline = baseline.copy()
                if 'timestamp_range' in serializable_baseline:
                    start, end = serializable_baseline['timestamp_range']
                    serializable_baseline['timestamp_range'] = [start.isoformat(), end.isoformat()]
                serializable_baselines[key] = serializable_baseline
                
            with open(baseline_file, 'w') as f:
                json.dump(serializable_baselines, f, indent=2)
            self.logger.logger.info(f"Saved {len(self.baseline_stats)} baseline statistics")
        except Exception as e:
            self.logger.logger.error(f"Failed to save baselines: {e}")
            
    def get_recent_alerts(self, hours: int = 24) -> List[DriftAlert]:
        """Get recent drift alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]
        
    def get_drift_summary(self) -> Dict:
        """Get summary of drift monitoring status."""
        recent_alerts = self.get_recent_alerts()
        
        summary = {
            'monitoring_active': self.monitoring_active,
            'monitored_models': len(self.data_windows),
            'baseline_count': len(self.baseline_stats),
            'recent_alerts': len(recent_alerts),
            'alert_breakdown': defaultdict(int)
        }
        
        for alert in recent_alerts:
            summary['alert_breakdown'][f"{alert.model_type}_{alert.severity}"] += 1
            
        return summary