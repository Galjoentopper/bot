"""Enhanced logging and diagnostics for schema decisions and drift events."""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque

@dataclass
class SchemaDecision:
    """Schema validation decision record."""
    timestamp: str
    symbol: str
    model_type: str
    decision_type: str  # 'accept', 'reject', 'transform'
    validation_method: str
    input_schema: Dict[str, Any]
    expected_schema: Dict[str, Any]
    validation_result: Dict[str, Any]
    action_taken: str
    confidence_score: float
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DriftEvent:
    """Drift detection event record."""
    timestamp: str
    symbol: str
    model_type: str
    event_type: str  # 'drift_detected', 'drift_resolved', 'baseline_updated'
    detection_method: str
    drift_magnitude: float
    affected_features: List[str]
    statistical_metrics: Dict[str, Any]
    alert_level: str  # 'info', 'warning', 'error', 'critical'
    response_action: str
    model_performance_impact: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ValidationMetrics:
    """Validation performance metrics."""
    timestamp: str
    symbol: str
    model_type: str
    total_validations: int
    successful_validations: int
    failed_validations: int
    average_processing_time_ms: float
    schema_mismatches: int
    drift_detections: int
    false_positives: int
    false_negatives: int
    accuracy_score: float

class EnhancedValidationLogger:
    """Enhanced logging system for validation and drift monitoring."""
    
    def __init__(self, log_dir: str = "logs/validation", config: Dict[str, Any] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Configure logging
        self.logger = self._setup_logger()
        
        # Event storage
        self.schema_decisions = deque(maxlen=10000)
        self.drift_events = deque(maxlen=10000)
        self.validation_metrics = defaultdict(list)
        
        # Performance tracking
        self.performance_stats = defaultdict(lambda: {
            'total_time': 0.0,
            'total_calls': 0,
            'error_count': 0,
            'success_count': 0
        })
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate': self.config.get('error_rate_threshold', 0.1),
            'processing_time': self.config.get('processing_time_threshold', 1000),  # ms
            'drift_frequency': self.config.get('drift_frequency_threshold', 0.05)
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging."""
        logger = logging.getLogger('validation_logger')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler for detailed logs
        log_file = self.log_dir / f"validation_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler for important events
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_schema_decision(self, 
                           symbol: str,
                           model_type: str,
                           decision_type: str,
                           validation_method: str,
                           input_schema: Dict[str, Any],
                           expected_schema: Dict[str, Any],
                           validation_result: Dict[str, Any],
                           action_taken: str,
                           confidence_score: float,
                           processing_time_ms: float,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a schema validation decision."""
        
        decision = SchemaDecision(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            model_type=model_type,
            decision_type=decision_type,
            validation_method=validation_method,
            input_schema=input_schema,
            expected_schema=expected_schema,
            validation_result=validation_result,
            action_taken=action_taken,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            metadata=metadata or {}
        )
        
        self.schema_decisions.append(decision)
        
        # Update performance stats
        key = f"{symbol}_{model_type}"
        stats = self.performance_stats[key]
        stats['total_time'] += processing_time_ms
        stats['total_calls'] += 1
        
        if decision_type == 'reject':
            stats['error_count'] += 1
        else:
            stats['success_count'] += 1
        
        # Log based on decision type
        log_data = {
            'event_type': 'schema_decision',
            'symbol': symbol,
            'model_type': model_type,
            'decision': decision_type,
            'method': validation_method,
            'confidence': confidence_score,
            'processing_time_ms': processing_time_ms,
            'action': action_taken
        }
        
        if decision_type == 'reject':
            self.logger.warning(f"Schema validation rejected: {json.dumps(log_data)}")
        elif confidence_score < 0.8:
            self.logger.info(f"Schema validation with low confidence: {json.dumps(log_data)}")
        else:
            self.logger.debug(f"Schema validation successful: {json.dumps(log_data)}")
        
        # Check for alerts
        self._check_schema_alerts(key, stats)
    
    def log_drift_event(self,
                       symbol: str,
                       model_type: str,
                       event_type: str,
                       detection_method: str,
                       drift_magnitude: float,
                       affected_features: List[str],
                       statistical_metrics: Dict[str, Any],
                       alert_level: str,
                       response_action: str,
                       model_performance_impact: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a drift detection event."""
        
        event = DriftEvent(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            model_type=model_type,
            event_type=event_type,
            detection_method=detection_method,
            drift_magnitude=drift_magnitude,
            affected_features=affected_features,
            statistical_metrics=statistical_metrics,
            alert_level=alert_level,
            response_action=response_action,
            model_performance_impact=model_performance_impact,
            metadata=metadata or {}
        )
        
        self.drift_events.append(event)
        
        # Log based on alert level
        log_data = {
            'event_type': 'drift_detection',
            'symbol': symbol,
            'model_type': model_type,
            'drift_type': event_type,
            'method': detection_method,
            'magnitude': drift_magnitude,
            'features': affected_features,
            'alert_level': alert_level,
            'action': response_action
        }
        
        if alert_level == 'critical':
            self.logger.error(f"Critical drift detected: {json.dumps(log_data)}")
        elif alert_level == 'error':
            self.logger.error(f"High drift detected: {json.dumps(log_data)}")
        elif alert_level == 'warning':
            self.logger.warning(f"Moderate drift detected: {json.dumps(log_data)}")
        else:
            self.logger.info(f"Drift event: {json.dumps(log_data)}")
    
    def log_validation_metrics(self,
                              symbol: str,
                              model_type: str,
                              metrics: Dict[str, Any]) -> None:
        """Log validation performance metrics."""
        
        validation_metrics = ValidationMetrics(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            model_type=model_type,
            total_validations=metrics.get('total_validations', 0),
            successful_validations=metrics.get('successful_validations', 0),
            failed_validations=metrics.get('failed_validations', 0),
            average_processing_time_ms=metrics.get('average_processing_time_ms', 0.0),
            schema_mismatches=metrics.get('schema_mismatches', 0),
            drift_detections=metrics.get('drift_detections', 0),
            false_positives=metrics.get('false_positives', 0),
            false_negatives=metrics.get('false_negatives', 0),
            accuracy_score=metrics.get('accuracy_score', 0.0)
        )
        
        key = f"{symbol}_{model_type}"
        self.validation_metrics[key].append(validation_metrics)
        
        # Keep only recent metrics
        if len(self.validation_metrics[key]) > 1000:
            self.validation_metrics[key] = self.validation_metrics[key][-1000:]
        
        self.logger.info(f"Validation metrics updated: {json.dumps(asdict(validation_metrics))}")
    
    def _check_schema_alerts(self, key: str, stats: Dict[str, Any]) -> None:
        """Check for schema validation alerts."""
        if stats['total_calls'] < 10:  # Need minimum samples
            return
        
        error_rate = stats['error_count'] / stats['total_calls']
        avg_processing_time = stats['total_time'] / stats['total_calls']
        
        # Error rate alert
        if error_rate > self.alert_thresholds['error_rate']:
            self.logger.error(f"High error rate detected for {key}: {error_rate:.2%}")
        
        # Processing time alert
        if avg_processing_time > self.alert_thresholds['processing_time']:
            self.logger.warning(f"Slow processing detected for {key}: {avg_processing_time:.1f}ms")
    
    def get_schema_decision_summary(self, 
                                   symbol: str = None, 
                                   model_type: str = None,
                                   hours: int = 24) -> Dict[str, Any]:
        """Get summary of schema decisions."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_decisions = []
        for decision in self.schema_decisions:
            decision_time = datetime.fromisoformat(decision.timestamp)
            if decision_time > cutoff_time:
                if (not symbol or decision.symbol == symbol) and \
                   (not model_type or decision.model_type == model_type):
                    filtered_decisions.append(decision)
        
        if not filtered_decisions:
            return {'message': 'No schema decisions found'}
        
        # Calculate summary statistics
        total_decisions = len(filtered_decisions)
        accepted = sum(1 for d in filtered_decisions if d.decision_type == 'accept')
        rejected = sum(1 for d in filtered_decisions if d.decision_type == 'reject')
        transformed = sum(1 for d in filtered_decisions if d.decision_type == 'transform')
        
        avg_confidence = sum(d.confidence_score for d in filtered_decisions) / total_decisions
        avg_processing_time = sum(d.processing_time_ms for d in filtered_decisions) / total_decisions
        
        # Method breakdown
        method_counts = defaultdict(int)
        for decision in filtered_decisions:
            method_counts[decision.validation_method] += 1
        
        return {
            'period_hours': hours,
            'total_decisions': total_decisions,
            'accepted': accepted,
            'rejected': rejected,
            'transformed': transformed,
            'acceptance_rate': accepted / total_decisions if total_decisions > 0 else 0,
            'average_confidence': avg_confidence,
            'average_processing_time_ms': avg_processing_time,
            'method_breakdown': dict(method_counts),
            'recent_decisions': [asdict(d) for d in filtered_decisions[-10:]]
        }
    
    def get_drift_event_summary(self,
                               symbol: str = None,
                               model_type: str = None,
                               hours: int = 24) -> Dict[str, Any]:
        """Get summary of drift events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_events = []
        for event in self.drift_events:
            event_time = datetime.fromisoformat(event.timestamp)
            if event_time > cutoff_time:
                if (not symbol or event.symbol == symbol) and \
                   (not model_type or event.model_type == model_type):
                    filtered_events.append(event)
        
        if not filtered_events:
            return {'message': 'No drift events found'}
        
        # Calculate summary statistics
        total_events = len(filtered_events)
        
        # Alert level breakdown
        alert_counts = defaultdict(int)
        for event in filtered_events:
            alert_counts[event.alert_level] += 1
        
        # Detection method breakdown
        method_counts = defaultdict(int)
        for event in filtered_events:
            method_counts[event.detection_method] += 1
        
        # Average drift magnitude
        avg_magnitude = sum(e.drift_magnitude for e in filtered_events) / total_events
        
        # Most affected features
        feature_counts = defaultdict(int)
        for event in filtered_events:
            for feature in event.affected_features:
                feature_counts[feature] += 1
        
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'period_hours': hours,
            'total_events': total_events,
            'alert_level_breakdown': dict(alert_counts),
            'method_breakdown': dict(method_counts),
            'average_drift_magnitude': avg_magnitude,
            'most_affected_features': top_features,
            'recent_events': [asdict(e) for e in filtered_events[-10:]]
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {},
            'overall_stats': {
                'total_schema_decisions': len(self.schema_decisions),
                'total_drift_events': len(self.drift_events),
                'active_models': len(self.performance_stats)
            }
        }
        
        # Per-model performance
        for key, stats in self.performance_stats.items():
            if stats['total_calls'] > 0:
                avg_time = stats['total_time'] / stats['total_calls']
                error_rate = stats['error_count'] / stats['total_calls']
                
                report['model_performance'][key] = {
                    'total_calls': stats['total_calls'],
                    'success_rate': stats['success_count'] / stats['total_calls'],
                    'error_rate': error_rate,
                    'average_processing_time_ms': avg_time,
                    'status': self._get_model_status(error_rate, avg_time)
                }
        
        return report
    
    def _get_model_status(self, error_rate: float, avg_time: float) -> str:
        """Determine model validation status."""
        if error_rate > 0.2 or avg_time > 2000:
            return 'critical'
        elif error_rate > 0.1 or avg_time > 1000:
            return 'warning'
        else:
            return 'healthy'
    
    def export_logs(self, filepath: str, format: str = 'json') -> None:
        """Export logs to file."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'schema_decisions': [asdict(d) for d in self.schema_decisions],
            'drift_events': [asdict(e) for e in self.drift_events],
            'validation_metrics': {k: [asdict(m) for m in v] for k, v in self.validation_metrics.items()},
            'performance_stats': dict(self.performance_stats),
            'config': self.config
        }
        
        filepath = Path(filepath)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Export as separate CSV files
            base_path = filepath.parent / filepath.stem
            
            # Schema decisions
            if self.schema_decisions:
                df_decisions = pd.DataFrame([asdict(d) for d in self.schema_decisions])
                df_decisions.to_csv(f"{base_path}_schema_decisions.csv", index=False)
            
            # Drift events
            if self.drift_events:
                df_events = pd.DataFrame([asdict(e) for e in self.drift_events])
                df_events.to_csv(f"{base_path}_drift_events.csv", index=False)
        
        self.logger.info(f"Logs exported to {filepath}")
    
    def cleanup_old_logs(self, days: int = 30) -> None:
        """Clean up old log files."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                self.logger.info(f"Deleted old log file: {log_file}")

def create_enhanced_logger(log_dir: str = "logs/validation", config: Dict[str, Any] = None) -> EnhancedValidationLogger:
    """Create an enhanced validation logger instance."""
    return EnhancedValidationLogger(log_dir, config)