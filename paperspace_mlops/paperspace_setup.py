#!/usr/bin/env python3
"""
Paperspace Setup Script
=======================

One-time setup script for Paperspace Gradient training environment.
Handles all environment preparation, dependency installation, and configuration.

Run this ONCE when starting a new Paperspace machine before training.

Usage:
    python paperspace_setup.py                  # Full setup
    python paperspace_setup.py --quick          # Skip optional components
    python paperspace_setup.py --verify-only    # Just verify environment
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PaperspaceSetup:
    """Handles complete Paperspace environment setup"""

    def __init__(self):
        self.is_paperspace = self._detect_paperspace()
        self.workspace_dir = Path("/notebooks" if self.is_paperspace else ".")
        self.bot_dir = self.workspace_dir / "bot"
        self.setup_complete = False

    def _detect_paperspace(self) -> bool:
        """Detect if running on Paperspace"""
        return (
            os.environ.get("PAPERSPACE_JOB_ID") is not None
            or os.environ.get("GRADIENT_JOB_ID") is not None
            or Path("/notebooks").exists()
        )

    def run_full_setup(self, quick: bool = False) -> bool:
        """Execute complete setup process"""

        logger.info("🚀 Starting Paperspace Setup")
        logger.info(f"📍 Environment: {'Paperspace' if self.is_paperspace else 'Local'}")
        logger.info(f"📁 Workspace: {self.workspace_dir}")
        logger.info("=" * 60)

        try:
            # Core setup steps
            self._setup_directories()
            self._setup_python_environment()
            self._install_dependencies()
            self._configure_environment()
            self._verify_data_availability()

            if not quick:
                self._setup_optional_tools()

            self._verify_setup()

            logger.info("\n" + "=" * 60)
            logger.info("✅ SETUP COMPLETE!")
            logger.info("🚀 Ready to run: python paperspace_training.py")
            logger.info("=" * 60)

            self.setup_complete = True
            return True

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

    def _setup_directories(self):
        """Create necessary directory structure"""
        logger.info("📁 Setting up directories...")

        directories = [
            self.bot_dir,
            self.bot_dir / "models",
            self.bot_dir / "logs",
            self.bot_dir / "exports",
            self.bot_dir / "data",  # For local databases
            self.workspace_dir / "outputs",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ Created: {dir_path}")

    def _setup_python_environment(self):
        """Configure Python environment and paths"""
        logger.info("🐍 Setting up Python environment...")

        # Add to Python path
        python_paths = [
            str(self.bot_dir),
            str(self.bot_dir / "src"),
            str(self.bot_dir / "paperspace_mlops"),
        ]

        for path in python_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
                logger.info(f"  ✅ Added to Python path: {path}")

        # Set environment variables
        os.environ["PYTHONPATH"] = ":".join(python_paths)
        os.environ["PAPERSPACE_WORKSPACE"] = str(self.workspace_dir)

        logger.info(f"  ✅ Python version: {sys.version}")

    def _install_dependencies(self):
        """Install required Python packages"""
        logger.info("📦 Installing dependencies...")

        # Core ML packages
        core_packages = [
            "torch>=2.0.0",
            "sklearn",
            "lightgbm",
            "stable-baselines3[extra]",
            "gymnasium",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "yfinance",
            "requests",
            "scipy",
            "matplotlib",
            "seaborn",
            "plotly",
            "optuna",
            "mlflow",
            "pyyaml",
            "python-dotenv",
            "psutil",
            "tqdm",
            "boto3",  # For S3 model uploads
        ]

        # Install packages
        for package in core_packages:
            try:
                logger.info(f"  📦 Installing {package}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package, "--quiet"], check=True
                )
                logger.info(f"  ✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"  ⚠️ Failed to install {package}: {e}")

    def _configure_environment(self):
        """Configure environment settings"""
        logger.info("⚙️ Configuring environment...")

        # Set training optimizations
        env_vars = {
            "TORCH_HOME": str(self.workspace_dir / "torch_cache"),
            "TRANSFORMERS_CACHE": str(self.workspace_dir / "transformers_cache"),
            "OPTUNA_STORAGE": f"sqlite:///{self.workspace_dir}/optuna.db",
            "MLFLOW_TRACKING_URI": f"file://{self.workspace_dir}/mlruns",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",  # Avoid warnings
        }

        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"  ✅ Set {key}: {value}")

        # Create cache directories
        cache_dirs = [
            Path(env_vars["TORCH_HOME"]),
            Path(env_vars["TRANSFORMERS_CACHE"]),
            Path(env_vars["MLFLOW_TRACKING_URI"].replace("file://", "")),
        ]

        for cache_dir in cache_dirs:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _verify_data_availability(self):
        """Verify that data folder exists and has databases"""
        logger.info("🗄️ Verifying data availability...")

        data_dir = self.bot_dir / "data"

        if not data_dir.exists():
            logger.warning(f"⚠️ Data directory not found: {data_dir}")
            logger.info("💡 Make sure to:")
            logger.info("   1. Upload your trading databases to the data/ folder")
            logger.info("   2. Or clone your repository with existing databases")
            return

        # Look for database files
        db_files = list(data_dir.glob("*.db"))

        if db_files:
            logger.info(f"  ✅ Found {len(db_files)} database file(s):")
            for db_file in db_files[:5]:  # Show first 5
                logger.info(f"    📄 {db_file.name}")
            if len(db_files) > 5:
                logger.info(f"    ... and {len(db_files) - 5} more")
        else:
            logger.warning("⚠️ No database files found in data/ folder")
            logger.info("💡 Training will proceed but may not have optimal data")

    def _setup_optional_tools(self):
        """Setup optional tools and utilities"""
        logger.info("🔧 Setting up optional tools...")

        # Setup Jupyter extensions if available
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "jupyter",
                    "jupyterlab",
                    "ipywidgets",
                    "--quiet",
                ],
                check=True,
            )
            logger.info("  ✅ Jupyter tools installed")
        except subprocess.CalledProcessError:
            logger.info("  ⚠️ Jupyter tools installation skipped")

        # Setup git if available
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"  ✅ Git available: {result.stdout.strip()}")

                # Configure git if not already configured
                try:
                    subprocess.run(
                        ["git", "config", "user.name", "Paperspace Training"], check=True
                    )
                    subprocess.run(
                        ["git", "config", "user.email", "training@paperspace.com"], check=True
                    )
                    logger.info("  ✅ Git configured")
                except subprocess.CalledProcessError:
                    pass
            else:
                logger.info("  ⚠️ Git not available")
        except FileNotFoundError:
            logger.info("  ⚠️ Git not found")

    def _verify_setup(self):
        """Verify that setup completed successfully"""
        logger.info("🔍 Verifying setup...")

        # Test core imports
        test_imports = [
            ("torch", "PyTorch"),
            ("sklearn", "Scikit-learn"),
            ("lightgbm", "LightGBM"),
            ("stable_baselines3", "Stable Baselines3"),
            ("pandas", "Pandas"),
            ("numpy", "NumPy"),
            ("yaml", "PyYAML"),
        ]

        failed_imports = []
        for module, name in test_imports:
            try:
                __import__(module)
                logger.info(f"  ✅ {name}")
            except ImportError:
                logger.error(f"  ❌ {name} not available")
                failed_imports.append(name)

        if failed_imports:
            raise RuntimeError(f"Missing required packages: {', '.join(failed_imports)}")

        # Verify GPU if available
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(f"  🚀 GPU available: {torch.cuda.get_device_name(0)}")
                logger.info(
                    f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
            else:
                logger.info("  💻 CPU mode (no GPU detected)")
        except Exception:
            logger.info("  💻 CPU mode")

        # Check disk space
        try:
            import shutil

            total, used, free = shutil.disk_usage(self.workspace_dir)
            logger.info(f"  💾 Disk space: {free/1e9:.1f} GB free of {total/1e9:.1f} GB total")
        except Exception:
            pass


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Paperspace Setup Script")
    parser.add_argument("--quick", action="store_true", help="Quick setup (skip optional tools)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing setup")

    args = parser.parse_args()

    setup = PaperspaceSetup()

    if args.verify_only:
        logger.info("🔍 Verification mode")
        setup._verify_setup()
        logger.info("✅ Verification complete")
        return

    success = setup.run_full_setup(quick=args.quick)

    if success:
        logger.info("\n🎉 Setup successful! Next steps:")
        logger.info("   1. Upload your data files to bot/data/ (if not already done)")
        logger.info("   2. Run: python paperspace_training.py")
        sys.exit(0)
    else:
        logger.error("❌ Setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
