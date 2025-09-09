"""Configuration package.

Provides unified configuration management with hierarchical and environment-specific overrides:
- ConfigLoader: Main configuration loading interface
- HierarchicalConfig: Multi-level configuration support
- EnvironmentManager: Environment-specific config handling
- ConfigGenerator: Dynamic configuration generation
- ConfigSchema: Pydantic-based configuration validation
- EnvironmentValidator: Startup environment validation
"""

from .config_generator import ConfigGenerator
from .config_loader import ConfigLoader
from .config_schema import (
    EnvironmentConfig,
    ModelWeights,
    RiskManagementConfig,
    ThresholdConfig,
    TradingConfig,
    get_config_schema,
    validate_environment,
    validate_trading_config,
)
from .environment_manager import EnvironmentManager
from .environment_validator import (
    EnvironmentValidator,
    get_startup_validation_report,
    validate_startup_environment,
)
from .hierarchical_config import HierarchicalConfig

__all__ = [
    "ConfigLoader",
    "HierarchicalConfig",
    "EnvironmentManager",
    "ConfigGenerator",
    "TradingConfig",
    "EnvironmentConfig",
    "ModelWeights",
    "ThresholdConfig",
    "RiskManagementConfig",
    "validate_environment",
    "validate_trading_config",
    "get_config_schema",
    "EnvironmentValidator",
    "validate_startup_environment",
    "get_startup_validation_report",
]
