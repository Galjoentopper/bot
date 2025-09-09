#!/usr/bin/env python3
"""
Paperspace MLOps Setup
======================

Streamlined setup for Paperspace training pipeline.
"""

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def install_packages():
    """Install essential packages"""
    
    logger.info("📦 Installing packages...")
    
    packages = [
        "numpy pandas pyyaml requests boto3 openpyxl ta yfinance",
        "python-telegram-bot scikit-learn lightgbm torch gymnasium",
        "structlog schedule mlflow optuna psutil tqdm"
    ]
    
    for package_group in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + package_group.split(), 
                         check=True, timeout=300)
            logger.info(f"✅ Installed: {package_group}")
        except Exception as e:
            logger.warning(f"⚠️ Failed: {package_group} - {e}")
    
    logger.info("✅ Package installation complete")


def main():
    """Main setup"""
    
    logger.info("🚀 Paperspace MLOps Setup")
    logger.info("=" * 40)
    
    try:
        install_packages()
        
        logger.info("✅ Setup complete!")
        logger.info("🎯 Next steps:")
        logger.info("1. python load_paperspace_secrets.py")
        logger.info("2. python paperspace_training_orchestrator.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())