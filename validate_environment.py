#!/usr/bin/env python3
"""Standalone environment validation script."""

import logging
import os
import sys
from pathlib import Path

import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_python_environment():
    """Validate Python environment and critical imports."""
    logger.info("🔍 Validating Python environment...")

    errors = []

    # Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python 3.8+ required, got {sys.version_info}")
    else:
        logger.info(
            f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    # Check critical modules
    critical_modules = ["pandas", "numpy", "torch", "lightgbm", "yaml", "requests", "aiohttp"]

    for module_name in critical_modules:
        try:
            __import__(module_name)
            logger.info(f"✅ {module_name} available")
        except ImportError as e:
            error_msg = f"Critical module missing: {module_name} ({e})"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")

    return len(errors) == 0, errors


def validate_directories():
    """Validate required directories exist."""
    logger.info("🔍 Validating directory structure...")

    errors = []
    warnings = []

    required_dirs = ["src", "logs"]
    optional_dirs = ["models", "data"]

    # Check required directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            error_msg = f"Required directory missing: {dir_name}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        else:
            logger.info(f"✅ {dir_name}/ exists")

    # Check optional directories
    for dir_name in optional_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            warning_msg = f"Optional directory missing: {dir_name} (will be created if needed)"
            warnings.append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
        else:
            logger.info(f"✅ {dir_name}/ exists")

    return len(errors) == 0, errors, warnings


def validate_config_file(config_path="training_config.yaml"):
    """Validate configuration file exists and is readable."""
    logger.info(f"🔍 Validating configuration file: {config_path}")

    errors = []

    config_file = Path(config_path)

    if not config_file.exists():
        error_msg = f"Configuration file not found: {config_path}"
        errors.append(error_msg)
        logger.error(f"❌ {error_msg}")
        return False, errors

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Basic structure validation
        expected_sections = ["data_acquisition", "training"]
        for section in expected_sections:
            if section not in config_data:
                error_msg = f"Missing required config section: {section}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
            else:
                logger.info(f"✅ Config section '{section}' found")

        # Check symbols
        if "data_acquisition" in config_data and "symbols" in config_data["data_acquisition"]:
            symbols = config_data["data_acquisition"]["symbols"]
            logger.info(f"✅ Found {len(symbols)} trading symbols: {', '.join(symbols)}")

        logger.info("✅ Configuration file is valid YAML")

    except yaml.YAMLError as e:
        error_msg = f"YAML parsing error in {config_path}: {e}"
        errors.append(error_msg)
        logger.error(f"❌ {error_msg}")
    except Exception as e:
        error_msg = f"Unexpected error reading {config_path}: {e}"
        errors.append(error_msg)
        logger.error(f"❌ {error_msg}")

    return len(errors) == 0, errors


def validate_environment_variables():
    """Validate environment variables."""
    logger.info("🔍 Validating environment variables...")

    warnings = []

    # Optional environment variables
    env_vars = {
        "TELEGRAM_BOT_TOKEN": "Telegram bot token for notifications",
        "TELEGRAM_CHAT_ID": "Telegram chat ID for notifications",
        "BITVAVO_API_KEY": "Bitvavo API key (optional)",
        "BITVAVO_API_SECRET": "Bitvavo API secret (optional)",
        "LOG_LEVEL": "Log level (defaults to INFO)",
    }

    for var_name, description in env_vars.items():
        value = os.getenv(var_name)
        if value:
            # Mask sensitive values
            if "TOKEN" in var_name or "SECRET" in var_name or "KEY" in var_name:
                masked_value = (
                    value[:8] + "*" * (len(value) - 8) if len(value) > 8 else "*" * len(value)
                )
                logger.info(f"✅ {var_name}: {masked_value}")
            else:
                logger.info(f"✅ {var_name}: {value}")
        else:
            warning_msg = f"{var_name} not set ({description})"
            warnings.append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")

    return True, [], warnings


def validate_permissions():
    """Validate file system permissions."""
    logger.info("🔍 Validating file system permissions...")

    errors = []
    test_paths = [("logs", "write"), (".", "read")]

    for path_str, operation in test_paths:
        path = Path(path_str)
        if not path.exists():
            if path_str == "logs":
                # Try to create logs directory
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"✅ Created logs directory")
                except Exception as e:
                    error_msg = f"Cannot create logs directory: {e}"
                    errors.append(error_msg)
                    logger.error(f"❌ {error_msg}")
            continue

        try:
            if operation == "write":
                # Test write permission
                test_file = path / ".permission_test"
                test_file.touch()
                test_file.unlink()
                logger.info(f"✅ Write permission OK for {path}")
            elif operation == "read":
                # Test read permission
                list(path.iterdir())
                logger.info(f"✅ Read permission OK for {path}")

        except PermissionError:
            error_msg = f"Insufficient {operation} permission for: {path}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        except Exception as e:
            warning_msg = f"Permission check failed for {path}: {e}"
            logger.warning(f"⚠️ {warning_msg}")

    return len(errors) == 0, errors


def main():
    """Run complete validation suite."""
    logger.info("🚀 Starting Trading Bot Environment Validation...")
    logger.info("=" * 60)

    all_errors = []
    all_warnings = []

    validations = [
        ("Python Environment", validate_python_environment),
        ("Directory Structure", validate_directories),
        ("Configuration File", validate_config_file),
        ("Environment Variables", validate_environment_variables),
        ("File Permissions", validate_permissions),
    ]

    all_passed = True

    for name, validation_func in validations:
        logger.info(f"\n📋 {name}")
        logger.info("-" * 40)

        try:
            if (
                validation_func == validate_directories
                or validation_func == validate_environment_variables
            ):
                passed, errors, warnings = validation_func()
                all_warnings.extend(warnings)
            else:
                passed, errors = validation_func()

            if passed:
                logger.info(f"✅ {name} validation passed")
            else:
                logger.error(f"❌ {name} validation failed")
                all_passed = False

            all_errors.extend(errors)

        except Exception as e:
            error_msg = f"{name} validation error: {e}"
            logger.error(f"❌ {error_msg}")
            all_errors.append(error_msg)
            all_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)

    if all_errors:
        logger.error(f"❌ Found {len(all_errors)} error(s):")
        for error in all_errors:
            logger.error(f"   • {error}")

    if all_warnings:
        logger.warning(f"⚠️ Found {len(all_warnings)} warning(s):")
        for warning in all_warnings:
            logger.warning(f"   • {warning}")

    if all_passed and not all_errors:
        logger.info("🎉 All validations passed! Trading bot environment is ready.")
        logger.info("\n✅ Next steps:")
        logger.info("   1. Start the trading system: python scripts/trader.py")
        logger.info("   2. Or run tests: python quick_test_system.py")
        return True
    else:
        logger.error("💥 Environment validation failed! Please fix errors before starting.")
        logger.error("\n🔧 Common fixes:")
        logger.error("   1. Install missing packages: pip install -r requirements.txt")
        logger.error("   2. Create missing directories: mkdir -p logs data models")
        logger.error("   3. Set environment variables in .env file")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
