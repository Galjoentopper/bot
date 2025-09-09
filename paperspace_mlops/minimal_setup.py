#!/usr/bin/env python3
"""
Minimal Paperspace Setup
========================

Gets the basic dependencies installed without requiring AWS credentials.
You can set credentials later.
"""

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def install_package_safe(package_spec):
    """Install a package safely without breaking existing installations"""
    try:
        cmd = [
            sys.executable, "-m", "pip", "install", 
            package_spec,
            "--no-deps",  # Don't install dependencies to avoid conflicts
            "--ignore-installed"  # Ignore if already installed
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            logger.info(f"✅ {package_spec}")
            return True
        else:
            logger.warning(f"⚠️ {package_spec} - {result.stderr.strip()[:100]}")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ {package_spec} - {str(e)[:100]}")
        return False


def install_essentials():
    """Install only the most essential packages"""
    
    logger.info("📦 Installing essential packages...")
    
    # Core packages that usually work
    core_packages = [
        "numpy",
        "pandas", 
        "pyyaml",
        "requests",
        "openpyxl",
        "ta",  # Technical analysis - simpler than pandas-ta
        "yfinance",
    ]
    
    success_count = 0
    for package in core_packages:
        if install_package_safe(package):
            success_count += 1
    
    logger.info(f"✅ Core packages: {success_count}/{len(core_packages)} installed")
    
    # AWS packages (needed for S3)
    aws_packages = [
        "boto3",
        "botocore"
    ]
    
    for package in aws_packages:
        if install_package_safe(package):
            success_count += 1
    
    # Telegram (if needed)
    install_package_safe("python-telegram-bot")
    
    total_attempted = len(core_packages) + len(aws_packages) + 1
    logger.info(f"📊 Overall: {success_count}/{total_attempted} packages installed")
    
    return success_count >= len(core_packages) * 0.7  # 70% success rate for core packages


def show_next_steps():
    """Show what to do next"""
    
    logger.info("🎯 Next Steps:")
    logger.info("=" * 50)
    logger.info("1. Set your AWS credentials in Paperspace:")
    logger.info("   - Go to your notebook environment settings")
    logger.info("   - Add these environment variables:")
    logger.info("     AWS_ACCESS_KEY_ID = your_access_key")
    logger.info("     AWS_SECRET_ACCESS_KEY = your_secret_key") 
    logger.info("     AWS_DEFAULT_REGION = us-east-1")
    logger.info("")
    logger.info("2. OR set them manually in a notebook cell:")
    logger.info('   import os')
    logger.info('   os.environ["AWS_ACCESS_KEY_ID"] = "your_key"')
    logger.info('   os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret"')
    logger.info("")
    logger.info("3. Then run:")
    logger.info("   python paperspace_mlops/load_paperspace_secrets.py")
    logger.info("   python paperspace_mlops/paperspace_training_orchestrator.py")
    logger.info("")


def main():
    """Main setup function"""
    
    logger.info("🚀 Minimal Paperspace Setup")
    logger.info("=" * 50)
    
    try:
        if install_essentials():
            logger.info("✅ Essential packages installed successfully!")
            show_next_steps()
            return 0
        else:
            logger.warning("⚠️ Some packages failed, but you can try to continue")
            show_next_steps()
            return 0
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())