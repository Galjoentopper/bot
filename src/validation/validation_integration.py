"""Integration module for schema validation and drift monitoring in the trading system."""

import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import time

from .schema_validator import SchemaValidator
from .drift_monitor import FeatureDriftMonitor, DriftAlert
from .advanced_drift_monitor import AdvancedFeatureDriftMonitor
from .enhanced_logger import EnhancedValidationLogger
from ..utils.logger import Logger

class ValidationManager:
    """Manages schema validation and drift monitoring for the trading system."""
    
    def __init__(self, config_dir: str = "./validation", models_dir: str = "./models", external_config: Dict = None):
        self.config_dir = Path(config_dir)
        self.models_dir = Path(models_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger("ValidationManager")
        
        # Load configuration first
        self.config = self._load_config()
        
        # Override with external config if provided (e.g., from training_config.yaml)
        if external_config:
            # Merge external drift monitoring configuration
            if 'drift_monitoring' in external_config:
                drift_config = external_config['drift_monitoring']
                # Map from training config naming to validation config naming
                self.config['drift_monitoring_enabled'] = drift_config.get('enabled', True)
                if not drift_config.get('enabled', True):
                    self.logger.logger.info("Drift monitoring disabled via external configuration")
        
        # Initialize components
        self.schema_validator = SchemaValidator(str(self.config_dir), str(self.models_dir))
        self.drift_monitor = FeatureDriftMonitor(str(self.config_dir))
        self.advanced_drift_monitor = AdvancedFeatureDriftMonitor(self.config.get('advanced_drift', {}))
        self.enhanced_logger = EnhancedValidationLogger(
            log_dir=str(self.config_dir / "logs"),
            config=self.config.get('logging', {})
        )
        
        # Validation history
        self.validation_history = []
        
        # Alert handlers
        self.alert_handlers = []
        
        # Setup drift monitor callbacks
        self.drift_monitor.add_alert_callback(self._handle_drift_alert)
        
    def _load_config(self) -> Dict:
        """Load validation configuration."""
        config_file = self.config_dir / "validation_config.json"
        
        default_config = {
            "validation_enabled": True,
            "drift_monitoring_enabled": True,
            "auto_start_monitoring": True,
            "validation_on_prediction": True,
            "validation_on_training": True,
            "alert_thresholds": {
                "critical_alert_cooldown": 300,  # 5 minutes
                "max_alerts_per_hour": 10
            },
            "advanced_drift": {
                "window_size": 1000,
                "baseline_size": 5000,
                "alert_threshold": 0.05,
                "severe_threshold": 0.01,
                "critical_threshold": 0.001
            },
            "logging": {
                "error_rate_threshold": 0.1,
                "processing_time_threshold": 1000,
                "drift_frequency_threshold": 0.05
            },
            "model_specific_settings": {
                "gru": {
                    "strict_validation": True,
                    "drift_sensitivity": "medium"
                },
                "lightgbm": {
                    "strict_validation": True,
                    "drift_sensitivity": "medium"
                },
                "ppo": {
                    "strict_validation": False,  # PPO is more flexible
                    "drift_sensitivity": "low"
                }
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.logger.error(f"Failed to load config: {e}")
                
        # Save current config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        return default_config
        
    def start_monitoring(self):
        """Start drift monitoring if enabled."""
        if self.config.get("drift_monitoring_enabled", True):
            self.drift_monitor.start_monitoring()
            self.logger.logger.info("Drift monitoring started for production server with trained models")
        else:
            self.logger.logger.info("Drift monitoring disabled in config")
            
    def stop_monitoring(self):
        """Stop drift monitoring."""
        self.drift_monitor.stop_monitoring()
        self.logger.logger.info("Drift monitoring stopped")
        
    def validate_model_input(self, data: pd.DataFrame, model_type: str, symbol: str, 
                           context: str = "prediction") -> Dict[str, Any]:
        """Validate model input data with context-aware settings and comprehensive monitoring."""
        start_time = time.time()
        
        # Check if validation is enabled for this context
        validation_key = f"validation_on_{context}"
        if not self.config.get(validation_key, True):
            return {
                'valid': True,
                'skipped': True,
                'reason': f'Validation disabled for {context}'
            }
            
        # Get model-specific settings
        model_settings = self.config.get("model_specific_settings", {}).get(model_type, {})
        strict_validation = model_settings.get("strict_validation", True)
        
        # Perform validation
        validation_result = self.schema_validator.validate_model_input(data, model_type, symbol)
        validation_result['context'] = context
        validation_result['strict_validation'] = strict_validation
        validation_result['advanced_drift'] = None
        
        # Determine decision type and confidence
        decision_type = 'accept' if validation_result.get('valid', False) else 'reject'
        confidence_score = validation_result.get('confidence', 0.0)
        
        # Add to monitoring if enabled
        if self.config.get("drift_monitoring_enabled", True):
            self.drift_monitor.add_data_point(data, model_type, symbol)
            
            # Advanced drift detection
            advanced_drift_result = self.advanced_drift_monitor.add_current_data(data, symbol, model_type)
            validation_result['advanced_drift'] = advanced_drift_result
            
            # Log drift events
            if advanced_drift_result.get('drift_detected', False):
                for alert in advanced_drift_result.get('alerts', []):
                    self.enhanced_logger.log_drift_event(
                        symbol=symbol,
                        model_type=model_type,
                        event_type='drift_detected',
                        detection_method=alert.drift_type,
                        drift_magnitude=alert.metric_value,
                        affected_features=alert.feature_names or [],
                        statistical_metrics=advanced_drift_result.get('drift_scores', {}),
                        alert_level=alert.severity,
                        response_action='logged',
                        metadata={'validation_id': validation_result['timestamp']}
                    )
            
        # Store validation history
        self.validation_history.append(validation_result)
        
        # Handle validation failures
        if not validation_result['valid']:
            self._handle_validation_failure(validation_result, strict_validation)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        validation_result['processing_time_ms'] = processing_time_ms
        
        # Log schema decision
        self.enhanced_logger.log_schema_decision(
            symbol=symbol,
            model_type=model_type,
            decision_type=decision_type,
            validation_method='comprehensive',
            input_schema=self._extract_schema(data),
            expected_schema=validation_result.get('expected_schema', {}),
            validation_result=validation_result,
            action_taken='accept' if validation_result['valid'] else 'reject',
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            metadata={'drift_detected': validation_result.get('advanced_drift', {}).get('drift_detected', False)}
        )
            
        return validation_result
        
    def _handle_validation_failure(self, validation_result: Dict, strict_validation: bool):
        """Handle validation failures based on strictness settings."""
        if strict_validation:
            self.logger.logger.error(
                f"STRICT VALIDATION FAILED: {validation_result['model_type']} - {validation_result['symbol']}"
            )
            for error in validation_result.get('errors', []):
                self.logger.logger.error(f"  - {error}")
        else:
            self.logger.logger.warning(
                f"Validation failed (non-strict): {validation_result['model_type']} - {validation_result['symbol']}"
            )
            for error in validation_result.get('errors', []):
                self.logger.logger.warning(f"  - {error}")
                
    def _handle_drift_alert(self, alert: DriftAlert):
        """Handle drift alerts from the monitoring system."""
        # Log the alert
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        self.logger.logger.log(
            log_level,
            f"DRIFT ALERT [{alert.severity.upper()}]: {alert.message}"
        )
        
        # Call registered alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.logger.error(f"Alert handler failed: {e}")
                
    def add_alert_handler(self, handler: Callable[[DriftAlert], None]):
        """Add custom alert handler."""
        self.alert_handlers.append(handler)
        
    def create_validation_decorator(self, model_type: str, symbol: str, context: str = "prediction"):
        """Create a decorator for automatic validation of model inputs."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Try to find DataFrame in arguments
                data_arg = None
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        data_arg = arg
                        break
                        
                if data_arg is None:
                    for value in kwargs.values():
                        if isinstance(value, pd.DataFrame):
                            data_arg = value
                            break
                            
                # Validate if data found
                if data_arg is not None:
                    validation_result = self.validate_model_input(data_arg, model_type, symbol, context)
                    
                    # Check if we should proceed based on validation
                    model_settings = self.config.get("model_specific_settings", {}).get(model_type, {})
                    strict_validation = model_settings.get("strict_validation", True)
                    
                    if not validation_result['valid'] and strict_validation:
                        raise ValueError(f"Validation failed for {model_type} model: {validation_result['errors']}")
                        
                # Call original function
                return func(*args, **kwargs)
                
            return wrapper
        return decorator
        
    def get_validation_summary(self, hours: int = 24) -> Dict:
        """Get validation summary for the specified time period."""
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        
        recent_validations = [
            v for v in self.validation_history 
            if pd.to_datetime(v['timestamp']) >= cutoff
        ]
        
        summary = {
            'period_hours': hours,
            'total_validations': len(recent_validations),
            'successful_validations': sum(1 for v in recent_validations if v['valid']),
            'failed_validations': sum(1 for v in recent_validations if not v['valid']),
            'by_model_type': {},
            'by_symbol': {},
            'by_context': {},
            'drift_summary': self.drift_monitor.get_drift_summary(),
            'recent_alerts': len(self.drift_monitor.get_recent_alerts(hours))
        }
        
        # Group by model type
        for validation in recent_validations:
            model_type = validation['model_type']
            if model_type not in summary['by_model_type']:
                summary['by_model_type'][model_type] = {'valid': 0, 'invalid': 0}
                
            if validation['valid']:
                summary['by_model_type'][model_type]['valid'] += 1
            else:
                summary['by_model_type'][model_type]['invalid'] += 1
                
        # Group by symbol
        for validation in recent_validations:
            symbol = validation['symbol']
            if symbol not in summary['by_symbol']:
                summary['by_symbol'][symbol] = {'valid': 0, 'invalid': 0}
                
            if validation['valid']:
                summary['by_symbol'][symbol]['valid'] += 1
            else:
                summary['by_symbol'][symbol]['invalid'] += 1
                
        # Group by context
        for validation in recent_validations:
            context = validation.get('context', 'unknown')
            if context not in summary['by_context']:
                summary['by_context'][context] = {'valid': 0, 'invalid': 0}
                
            if validation['valid']:
                summary['by_context'][context]['valid'] += 1
            else:
                summary['by_context'][context]['invalid'] += 1
                
        return summary
        
    def generate_validation_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate comprehensive validation report."""
        # Get validation results from schema validator
        schema_report = self.schema_validator.create_validation_report(self.validation_history)
        
        # Get drift monitoring summary
        drift_summary = self.drift_monitor.get_drift_summary()
        recent_alerts = self.drift_monitor.get_recent_alerts()
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_manager_config': self.config,
            'schema_validation': schema_report,
            'drift_monitoring': {
                'summary': drift_summary,
                'recent_alerts': [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'model_type': alert.model_type,
                        'symbol': alert.symbol,
                        'drift_type': alert.drift_type,
                        'severity': alert.severity,
                        'message': alert.message,
                        'metrics': alert.metrics
                    }
                    for alert in recent_alerts
                ]
            },
            'recommendations': self._generate_recommendations(schema_report, drift_summary, recent_alerts)
        }
        
        # Save report if requested
        if output_file:
            report_path = self.config_dir / output_file
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.logger.info(f"Validation report saved to {report_path}")
            
        return report
        
    def _generate_recommendations(self, schema_report: Dict, drift_summary: Dict, recent_alerts: List) -> List[str]:
        """Generate recommendations based on validation and drift analysis."""
        recommendations = []
        
        # Schema validation recommendations
        if schema_report['success_rate'] < 0.9:
            recommendations.append(
                f"Schema validation success rate is low ({schema_report['success_rate']:.1%}). "
                "Consider reviewing data preprocessing pipelines."
            )
            
        # Drift monitoring recommendations
        if drift_summary['recent_alerts'] > 5:
            recommendations.append(
                f"High number of drift alerts ({drift_summary['recent_alerts']}). "
                "Consider retraining models or updating feature engineering."
            )
            
        # Critical alerts
        critical_alerts = [alert for alert in recent_alerts if alert.severity == 'critical']
        if critical_alerts:
            recommendations.append(
                f"Critical drift alerts detected ({len(critical_alerts)}). "
                "Immediate attention required - models may be unreliable."
            )
            
        # Model-specific recommendations
        for model_type, stats in schema_report.get('summary', {}).items():
            if stats['invalid'] > stats['valid']:
                recommendations.append(
                    f"Model {model_type} has more validation failures than successes. "
                    "Check model compatibility and feature engineering."
                )
                
        if not recommendations:
            recommendations.append("All validation metrics are within acceptable ranges.")
            
        return recommendations
        
    def cleanup_old_data(self, days: int = 30):
        """Clean up old validation data and alerts."""
        cutoff = datetime.now() - pd.Timedelta(days=days)
        
        # Clean validation history
        original_count = len(self.validation_history)
        self.validation_history = [
            v for v in self.validation_history 
            if pd.to_datetime(v['timestamp']) >= cutoff
        ]
        cleaned_validations = original_count - len(self.validation_history)
        
        # Clean drift monitor alerts (handled internally by deque maxlen)
        
        self.logger.logger.info(f"Cleaned up {cleaned_validations} old validation records")
        
    def export_validation_data(self, output_dir: str):
        """Export validation data for external analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export validation history
        validation_df = pd.DataFrame(self.validation_history)
        if not validation_df.empty:
            validation_df.to_csv(output_path / "validation_history.csv", index=False)
            
        # Export drift alerts
        alerts = self.drift_monitor.get_recent_alerts(hours=24*30)  # Last 30 days
        if alerts:
            alerts_data = [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'model_type': alert.model_type,
                    'symbol': alert.symbol,
                    'drift_type': alert.drift_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    **alert.metrics
                }
                for alert in alerts
            ]
            alerts_df = pd.DataFrame(alerts_data)
            alerts_df.to_csv(output_path / "drift_alerts.csv", index=False)
            
        # Export configuration
        with open(output_path / "validation_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
            
        self.logger.logger.info(f"Validation data exported to {output_path}")

    def _extract_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract schema information from DataFrame."""
        if data is None or data.empty:
            return {}
        
        schema = {
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'shape': data.shape,
            'null_counts': data.isnull().sum().to_dict()
        }
        
        return schema

# Convenience function for easy integration
def create_validation_manager(config_dir: str = "./validation", models_dir: str = "./models", 
                            auto_start: bool = True, external_config: Dict = None) -> ValidationManager:
    """Create and optionally start a validation manager."""
    manager = ValidationManager(config_dir, models_dir, external_config)
    
    if auto_start and manager.config.get("auto_start_monitoring", True):
        manager.start_monitoring()
        
    return manager