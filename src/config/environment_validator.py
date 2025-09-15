"""Environment validation and startup checks."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import ValidationError

# Add the parent directory to sys.path to enable imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.config.config_schema import (
        EnvironmentConfig,
        TradingConfig,
        validate_environment,
        validate_trading_config,
    )
except ImportError:
    # Fallback for direct execution
    from config_schema import (
        EnvironmentConfig,
        TradingConfig,
        validate_environment,
        validate_trading_config,
    )


logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """Validates environment configuration at startup."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.env_config: Optional[EnvironmentConfig] = None
        self.trading_config: Optional[TradingConfig] = None

    def validate_environment_variables(self) -> bool:
        """Validate required environment variables."""
        try:
            self.env_config = validate_environment()
            logger.info("Environment variables validated successfully")
            return True
        except ValidationError as e:
            for error in e.errors():
                field = error.get("loc", ["unknown"])[0]
                msg = error.get("msg", "validation error")
                self.errors.append(f"Environment variable {field}: {msg}")
            logger.error(f"Environment validation failed: {e}")
            return False

    def validate_config_file(self, config_path: str = "training_config.yaml") -> bool:
        """Validate trading configuration file."""
        config_file = Path(config_path)

        if not config_file.exists():
            self.errors.append(f"Configuration file not found: {config_path}")
            return False

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            self.trading_config = validate_trading_config(config_data)
            logger.info(f"Configuration file {config_path} validated successfully")
            return True

        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error in {config_path}: {e}")
            return False
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(loc) for loc in error.get("loc", ["unknown"]))
                msg = error.get("msg", "validation error")
                self.errors.append(f"Config field {field}: {msg}")
            logger.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Unexpected error reading {config_path}: {e}")
            return False

    def validate_directories(self) -> bool:
        """Validate required directories exist."""
        required_dirs = ["models", "logs", "data", "src"]

        success = True
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_name}")
                except Exception as e:
                    self.errors.append(f"Failed to create directory {dir_name}: {e}")
                    success = False
            elif not dir_path.is_dir():
                self.errors.append(f"Path exists but is not a directory: {dir_name}")
                success = False

        return success

    def validate_model_structure(self) -> bool:
        """Validate model directory structure."""
        models_dir = Path("models")
        if not models_dir.exists():
            self.warnings.append("Models directory does not exist - run import_models.sh first")
            return True

        expected_model_types = ["gru", "lightgbm", "ppo"]
        symbols = []

        if self.trading_config:
            symbols = self.trading_config.symbols

        success = True
        for model_type in expected_model_types:
            model_dir = models_dir / model_type
            if not model_dir.exists():
                self.warnings.append(f"Model directory missing: models/{model_type}")
                continue

            # Check for symbol-specific models if we have symbols
            if symbols:
                for symbol in symbols:
                    symbol_dir = model_dir / symbol
                    if not symbol_dir.exists():
                        self.warnings.append(
                            f"Model for {symbol} missing: models/{model_type}/{symbol}"
                        )

        return success

    def validate_permissions(self) -> bool:
        """Validate file system permissions."""
        test_paths = [
            ("logs", "write"),
            ("models", "read"),
            ("data", "write"),
            (".", "read"),
        ]

        success = True
        for path_str, operation in test_paths:
            path = Path(path_str)
            if not path.exists():
                continue

            try:
                if operation == "write":
                    # Test write permission
                    test_file = path / ".permission_test"
                    test_file.touch()
                    test_file.unlink()
                elif operation == "read":
                    # Test read permission
                    list(path.iterdir())

            except PermissionError:
                self.errors.append(f"Insufficient {operation} permission for: {path}")
                success = False
            except Exception as e:
                self.warnings.append(f"Permission check failed for {path}: {e}")

        return success

    def validate_python_environment(self) -> bool:
        """Validate Python environment and critical imports."""
        critical_modules = [
            "pandas",
            "numpy",
            "torch",
            "lightgbm",
            "yaml",
            "requests",
            "aiohttp",
        ]

        success = True
        for module_name in critical_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                self.errors.append(f"Critical module missing: {module_name} ({e})")
                success = False

        # Check Python version
        if sys.version_info < (3, 8):
            self.errors.append(f"Python 3.8+ required, got {sys.version_info}")
            success = False

        return success

    def validate_external_connectivity(self) -> bool:
        """Validate external service connectivity."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import requests

        test_urls = [
            ("Binance API", "https://api.binance.com/api/v3/ping"),
            (
                "Yahoo Finance",
                "https://query1.finance.yahoo.com/v8/finance/chart/BTCEUR",
            ),
        ]

        success = True

        def test_url(name: str, url: str) -> Tuple[str, bool, str]:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return name, True, "OK"
                else:
                    return name, False, f"HTTP {response.status_code}"
            except Exception as e:
                return name, False, str(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(test_url, name, url) for name, url in test_urls]

            for future in as_completed(futures):
                name, status, message = future.result()
                if status:
                    logger.info(f"Connectivity test passed: {name}")
                else:
                    self.warnings.append(f"Connectivity test failed: {name} - {message}")

        # Don't fail startup for connectivity issues
        return True

    def run_full_validation(self, config_path: str = "training_config.yaml") -> bool:
        """Run complete validation suite."""
        logger.info("Starting environment validation...")

        validations = [
            ("Python environment", self.validate_python_environment),
            ("Environment variables", self.validate_environment_variables),
            ("Configuration file", lambda: self.validate_config_file(config_path)),
            ("Directory structure", self.validate_directories),
            ("File permissions", self.validate_permissions),
            ("Model structure", self.validate_model_structure),
            ("External connectivity", self.validate_external_connectivity),
        ]

        all_passed = True

        for name, validation_func in validations:
            try:
                logger.info(f"Validating {name}...")
                passed = validation_func()
                if passed:
                    logger.info(f"✅ {name} validation passed")
                else:
                    logger.error(f"❌ {name} validation failed")
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {name} validation error: {e}")
                self.errors.append(f"{name} validation error: {e}")
                all_passed = False

        # Log summary
        if self.errors:
            logger.error("Validation Errors:")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning("Validation Warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        if all_passed and not self.errors:
            logger.info("🎉 All validations passed! System ready to start.")
            return True
        else:
            logger.error("💥 Validation failed! Please fix errors before starting.")
            return False

    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report."""
        return {
            "passed": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "environment_config": self.env_config.dict() if self.env_config else None,
            "trading_config_summary": (
                {
                    "symbols": (self.trading_config.symbols if self.trading_config else []),
                    "interval": (self.trading_config.interval if self.trading_config else None),
                    "paper_trading": (
                        self.trading_config.paper_trading if self.trading_config else True
                    ),
                }
                if self.trading_config
                else None
            ),
        }


def validate_startup_environment(config_path: str = "training_config.yaml") -> bool:
    """Validate environment at startup. Returns True if valid."""
    validator = EnvironmentValidator()
    return validator.run_full_validation(config_path)


def get_startup_validation_report(
    config_path: str = "training_config.yaml",
) -> Dict[str, Any]:
    """Get startup validation report."""
    validator = EnvironmentValidator()
    validator.run_full_validation(config_path)
    return validator.get_validation_report()


if __name__ == "__main__":
    # Command-line validation
    import argparse

    parser = argparse.ArgumentParser(description="Validate trading bot environment")
    parser.add_argument(
        "--config", default="training_config.yaml", help="Path to configuration file"
    )
    parser.add_argument("--json-report", action="store_true", help="Output JSON report")

    args = parser.parse_args()

    if args.json_report:
        report = get_startup_validation_report(args.config)
        print(json.dumps(report, indent=2))
    else:
        success = validate_startup_environment(args.config)
        sys.exit(0 if success else 1)
