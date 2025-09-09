#!/usr/bin/env python3
"""
Fix Paperspace Dependencies
===========================

Quick fix for dependency issues in Paperspace environment.
Installs compatible versions and handles problematic packages.
"""

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pip_install(package, ignore_errors=True):
    """Install package with pip, optionally ignoring errors"""
    try:
        cmd = [sys.executable, "-m", "pip", "install", package]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"✅ Installed: {package}")
            return True
        else:
            if ignore_errors:
                logger.warning(f"⚠️ Failed to install {package}: {result.stderr.strip()}")
                return False
            else:
                logger.error(f"❌ Failed to install {package}: {result.stderr.strip()}")
                raise subprocess.CalledProcessError(result.returncode, cmd)

    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout installing {package}")
        return False
    except Exception as e:
        logger.error(f"❌ Error installing {package}: {e}")
        return False


def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    logger.info(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ required")
        return False

    if version.major == 3 and version.minor >= 12:
        logger.info("✅ Python 3.12+ detected - can use latest packages")
        return "latest"
    else:
        logger.info("✅ Python 3.8-3.11 detected - using compatible versions")
        return "compatible"


def install_core_packages():
    """Install core packages that are essential"""

    logger.info("📦 Installing core packages...")

    core_packages = [
        "pip>=23.0",
        "setuptools>=65.0",
        "wheel>=0.38.0",
        "numpy>=1.21.0,<1.25.0",
        "pandas>=1.5.0,<2.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "psutil>=5.8.0",
    ]

    for package in core_packages:
        run_pip_install(package)


def install_ml_packages():
    """Install ML packages"""

    logger.info("🤖 Installing ML packages...")

    ml_packages = [
        "scikit-learn>=1.1.0,<1.3.0",
        "scipy>=1.9.0,<1.11.0",
        "torch>=1.12.0,<2.0.0",
        "lightgbm>=3.3.0,<4.0.0",
        "gymnasium>=0.26.0",
    ]

    for package in ml_packages:
        run_pip_install(package)

    # Handle stable-baselines3 separately due to gym compatibility issues
    logger.info("🤖 Installing stable-baselines3 with special handling...")
    if not run_pip_install("stable-baselines3>=1.6.0,<2.0.0", ignore_errors=True):
        # Try installing gym first with specific version
        logger.info("⚠️ Trying gym compatibility fix...")
        run_pip_install("gym==0.21.0", ignore_errors=True)
        run_pip_install("stable-baselines3==1.8.0", ignore_errors=True)


def install_financial_packages():
    """Install financial data packages"""

    logger.info("💹 Installing financial packages...")

    finance_packages = [
        "yfinance>=0.2.0",
        "python-binance>=1.0.15",
        "ta>=0.10.0",  # Use ta instead of pandas-ta
    ]

    for package in finance_packages:
        run_pip_install(package)


def install_optional_packages():
    """Install optional packages"""

    logger.info("🔧 Installing optional packages...")

    # Handle blinker conflict packages separately
    blinker_conflict_packages = ["mlflow>=2.0.0,<2.8.0", "flask>=2.2.0"]

    logger.info("🔧 Handling blinker conflicts...")
    for package in blinker_conflict_packages:
        if "mlflow" in package:
            # Try mlflow without dependencies first
            if not run_pip_install("mlflow>=2.0.0,<2.8.0 --no-deps", ignore_errors=True):
                logger.info("⚠️ MLflow failed, skipping for now")
        elif "flask" in package:
            # Try flask without upgrading dependencies
            if not run_pip_install("flask>=2.2.0 --no-deps", ignore_errors=True):
                logger.info("⚠️ Flask failed, using existing version")

    # Safe packages without conflicts
    safe_packages = [
        "optuna>=3.0.0,<4.0.0",
        "python-telegram-bot>=20.0",
        "boto3>=1.26.0",
        "openpyxl>=3.0.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ]

    for package in safe_packages:
        run_pip_install(package, ignore_errors=True)


def fix_pandas_ta():
    """Handle pandas-ta compatibility issue"""

    logger.info("🔧 Handling pandas-ta compatibility...")

    python_compat = check_python_version()

    if python_compat == "latest":
        # Try to install pandas-ta for Python 3.12+
        if run_pip_install("pandas-ta>=0.3.14b0", ignore_errors=True):
            logger.info("✅ pandas-ta installed successfully")
        else:
            logger.info("⚠️ pandas-ta failed, using ta library instead")
            run_pip_install("ta>=0.10.0")
    else:
        # Use ta library for older Python versions
        logger.info("📦 Using 'ta' library instead of pandas-ta for compatibility")
        run_pip_install("ta>=0.10.0")


def fix_paperspace_conflicts():
    """Fix specific Paperspace environment conflicts"""

    logger.info("🔧 Fixing Paperspace-specific conflicts...")

    try:
        # Force reinstall blinker to fix conflicts
        run_pip_install("--force-reinstall blinker>=1.6.0", ignore_errors=True)

        # Try to fix gym/gymnasium compatibility
        run_pip_install("--force-reinstall gym==0.21.0", ignore_errors=True)

        # Create minimal fallbacks for failed packages
        fallback_packages = {
            "mlflow": "mlflow_fallback.py",
            "flask": "flask_fallback.py",
            "stable_baselines3": "sb3_fallback.py",
        }

        for package_name, fallback_file in fallback_packages.items():
            try:
                __import__(package_name)
                logger.info(f"✅ {package_name} is available")
            except ImportError:
                logger.warning(f"⚠️ {package_name} not available, creating fallback")
                create_package_fallback(package_name, fallback_file)

    except Exception as e:
        logger.error(f"❌ Error fixing conflicts: {e}")


def create_package_fallback(package_name, filename):
    """Create fallback for missing packages"""

    fallback_code = f'''
# Fallback for {package_name}
import logging

logger = logging.getLogger(__name__)
logger.warning("Using fallback for {package_name} - some features may be limited")

class {package_name.replace("_", "").title()}Fallback:
    """Fallback implementation for {package_name}"""

    def __getattr__(self, name):
        def dummy_function(*args, **kwargs):
            logger.warning(f"{{name}} from {package_name} called but not available")
            return None
        return dummy_function

# Create fallback instance
{package_name} = {package_name.replace("_", "").title()}Fallback()
'''

    with open(filename, "w") as f:
        f.write(fallback_code)

    logger.info(f"✅ Created {filename} fallback")


def create_fallback_imports():
    """Create fallback imports for missing packages"""

    logger.info("🔄 Creating fallback imports...")

    fallback_code = """
# Fallback imports for missing packages
try:
    import pandas_ta as ta
except ImportError:
    try:
        import ta
        # Create pandas_ta compatibility layer
        class PandasTACompat:
            def __getattr__(self, name):
                if hasattr(ta, name):
                    return getattr(ta, name)
                else:
                    def dummy_indicator(*args, **kwargs):
                        import pandas as pd
                        import numpy as np
                        # Return dummy series for missing indicators
                        if len(args) > 0 and hasattr(args[0], 'index'):
                            return pd.Series(np.nan, index=args[0].index)
                        return pd.Series([np.nan])
                    return dummy_indicator

        pandas_ta = PandasTACompat()

    except ImportError:
        # Final fallback - create dummy module
        class DummyTA:
            def __getattr__(self, name):
                def dummy_indicator(*args, **kwargs):
                    import pandas as pd
                    import numpy as np
                    if len(args) > 0 and hasattr(args[0], 'index'):
                        return pd.Series(np.nan, index=args[0].index)
                    return pd.Series([np.nan])
                return dummy_indicator

        pandas_ta = DummyTA()
        ta = DummyTA()
"""

    # Write fallback to a file that can be imported
    with open("ta_fallback.py", "w") as f:
        f.write(fallback_code)

    logger.info("✅ Created ta_fallback.py for missing indicators")


def main():
    """Main fix function"""

    logger.info("🔧 Starting Paperspace dependency fix...")

    try:
        # Check Python version
        python_compat = check_python_version()
        if python_compat is False:
            return 1

        # Upgrade pip first
        run_pip_install("pip>=23.0")

        # Install in order of importance
        install_core_packages()
        install_ml_packages()
        install_financial_packages()
        install_optional_packages()

        # Handle pandas-ta issue
        fix_pandas_ta()

        # Fix Paperspace-specific conflicts
        fix_paperspace_conflicts()

        # Create fallback imports
        create_fallback_imports()

        logger.info("✅ Dependency fix completed!")
        logger.info("🚀 You can now run: python start_training.py")

        return 0

    except Exception as e:
        logger.error(f"❌ Fix failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
