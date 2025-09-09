"""
Validation Module
=================

Contains data validation, drift monitoring, and model validation components:
- DriftMonitor: Basic drift detection functionality
- AdvancedDriftMonitor: Enhanced drift monitoring with statistical tests
- SchemaValidator: Data schema validation
- MetadataManager: Model and data metadata management
- ValidationIntegration: Integrated validation pipeline
- WalkForwardValidator: Time series cross-validation
- EnhancedLogger: Validation-specific logging
"""

from .advanced_drift_monitor import AdvancedFeatureDriftMonitor
from .advanced_drift_monitor import DriftAlert as AdvancedDriftAlert
from .drift_monitor import DriftAlert, FeatureDriftMonitor
from .enhanced_logger import DriftEvent, SchemaDecision
from .metadata_manager import MetadataManager
from .metadata_manager import ModelMetadata as ValidationModelMetadata
from .schema_validator import SchemaValidator
from .validation_integration import ValidationManager
from .walk_forward_validator import PerformanceMetric, ValidationStrategy

__all__ = [
    "FeatureDriftMonitor",
    "DriftAlert",
    "AdvancedFeatureDriftMonitor",
    "AdvancedDriftAlert",
    "SchemaValidator",
    "MetadataManager",
    "ValidationModelMetadata",
    "ValidationManager",
    "ValidationStrategy",
    "PerformanceMetric",
    "SchemaDecision",
    "DriftEvent",
]
