#!/usr/bin/env python3
"""
Simple Paperspace Setup
======================

Minimal, robust setup for Paperspace environment that focuses on
getting the essential packages working without conflicts.
"""

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_safe_install(package):
    """Install package safely, ignoring common conflicts"""
    try:
        cmd = [sys.executable, "-m", "pip", "install", package, "--no-deps", "--ignore-installed"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            logger.info(f"✅ Installed: {package}")
            return True
        else:
            logger.warning(f"⚠️ Failed: {package}")
            return False

    except Exception as e:
        logger.warning(f"⚠️ Error with {package}: {e}")
        return False


def install_essentials():
    """Install only the most essential packages"""

    logger.info("📦 Installing essential packages...")

    essentials = [
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "boto3>=1.26.0",
        "openpyxl>=3.0.0",
        "ta>=0.10.0",  # Use ta instead of pandas-ta
        "python-telegram-bot>=20.0",
        "yfinance>=0.2.0",
    ]

    success_count = 0
    for package in essentials:
        if run_safe_install(package):
            success_count += 1

    logger.info(f"✅ Installed {success_count}/{len(essentials)} essential packages")
    return success_count >= len(essentials) * 0.8  # 80% success rate is acceptable


def setup_s3_credentials():
    """Setup S3 credentials - check if already available as environment variables"""

    logger.info("🔐 Checking S3 credentials...")

    import os

    # Check if AWS credentials are available as environment variables
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if aws_access_key and aws_secret_key:
        logger.info("✅ AWS credentials found in environment variables")
        logger.info(f"✅ Using region: {aws_region}")

        # Try to setup S3 bucket
        try:
            import sys

            sys.path.append("..")

            # Try importing the S3 setup
            try:
                from setup_s3_storage import S3StorageSetup

                logger.info("🏗️ Setting up S3 bucket...")
                s3_setup = S3StorageSetup()
                success = s3_setup.run_setup()

                if success:
                    logger.info("✅ S3 bucket setup completed successfully")
                    return True
                else:
                    logger.warning("⚠️ S3 bucket setup failed, but credentials are available")
                    return True  # Credentials are there, so pipeline can continue

            except ImportError:
                logger.warning("⚠️ S3 setup script not available, but credentials are set")
                return True  # Credentials are there, so pipeline can continue

        except Exception as e:
            logger.warning(f"⚠️ S3 setup error: {e}, but credentials are available")
            return True  # Credentials are there, so pipeline can continue

    else:
        logger.error("❌ AWS credentials not found in environment variables")
        logger.info("📋 Make sure these are set in Paperspace:")
        logger.info("1. AWS_ACCESS_KEY_ID (as secret)")
        logger.info("2. AWS_SECRET_ACCESS_KEY (as secret)")
        logger.info("3. AWS_DEFAULT_REGION=us-east-1")
        return False


def main():
    """Main simple setup"""

    logger.info("🚀 Starting Simple Paperspace Setup")
    logger.info("=" * 50)

    try:
        # Step 1: Install essential packages
        if not install_essentials():
            logger.warning("⚠️ Some packages failed, but continuing...")

        # Step 2: Setup S3 credentials
        if not setup_s3_credentials():
            logger.warning("⚠️ S3 setup failed - you'll need to set AWS credentials manually")

        logger.info("✅ Simple setup completed!")
        logger.info("🚀 You can now run: python paperspace_training_orchestrator.py")

        return 0

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
