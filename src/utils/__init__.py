"""
Utilities Module
================

Contains utility functions and helper classes:
- Logging: Enhanced logging infrastructure
- Config utilities: Configuration handling functions
- Calibration: Model calibration utilities
- Cross-validation: Model validation tools
- Metrics: Performance and evaluation metrics
- MLflow: MLflow integration utilities
- Model packaging and transfer utilities
- Training checkpoint management
"""

from .calibration import ProbabilityCalibrator
from .config import flatten_feature_config, prepare_feature_config, validate_feature_config
from .cross_validation import BlockingTimeSeriesSplit, PurgedTimeSeriesSplit
from .logger import TradingBotLogger, setup_logging
from .metrics import RegimeGate, TradingMetrics
from .mlflow_init import MLflowInitializer
from .model_packaging import ModelMetadata as UtilsModelMetadata
from .model_packaging import ModelPackager
from .model_transfer import ModelTransferManager
from .training_checkpoint import CheckpointMetadata, TrainingProgress

__all__ = [
    "setup_logging",
    "TradingBotLogger",
    "flatten_feature_config",
    "validate_feature_config",
    "prepare_feature_config",
    "ProbabilityCalibrator",
    "PurgedTimeSeriesSplit",
    "BlockingTimeSeriesSplit",
    "TradingMetrics",
    "RegimeGate",
    "MLflowInitializer",
    "ModelPackager",
    "UtilsModelMetadata",
    "ModelTransferManager",
    "TrainingProgress",
    "CheckpointMetadata",
]
