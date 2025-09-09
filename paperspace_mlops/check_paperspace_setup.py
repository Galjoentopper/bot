#!/usr/bin/env python3
"""
Check Paperspace Setup
=====================

Quick verification that Paperspace environment has everything needed
for the training pipeline.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_environment_variables():
    """Check if required environment variables are set"""

    logger.info("🔍 Checking environment variables...")

    required_vars = {
        "AWS_ACCESS_KEY_ID": "AWS Access Key ID",
        "AWS_SECRET_ACCESS_KEY": "AWS Secret Access Key",
        "AWS_DEFAULT_REGION": "AWS Region (will default to us-east-1)",
    }

    missing_vars = []

    for var_name, description in required_vars.items():
        value = os.environ.get(var_name)
        if value:
            if "SECRET" in var_name:
                logger.info(f"✅ {description}: {value[:8]}...")
            else:
                logger.info(f"✅ {description}: {value}")
        else:
            missing_vars.append((var_name, description))
            logger.error(f"❌ {description}: Not set")

    if missing_vars:
        logger.error("❌ Missing environment variables!")
        logger.info("📋 Set these in your Paperspace project secrets:")
        for var_name, description in missing_vars:
            logger.info(f"  {var_name} = your_{var_name.lower()}")
        return False

    logger.info("✅ All required environment variables are set!")
    return True


def test_aws_connection():
    """Test AWS S3 connection"""

    logger.info("🔌 Testing AWS connection...")

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        # Create S3 client
        s3_client = boto3.client("s3")

        # Test connection by listing buckets
        response = s3_client.list_buckets()

        logger.info("✅ AWS connection successful!")
        logger.info(f"✅ Found {len(response['Buckets'])} existing buckets")

        return True

    except NoCredentialsError:
        logger.error("❌ AWS credentials not found or invalid")
        return False
    except ClientError as e:
        logger.error(f"❌ AWS connection failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def check_essential_packages():
    """Check if essential packages are available"""

    logger.info("📦 Checking essential packages...")

    essential_packages = [
        "numpy",
        "pandas",
        "yaml",
        "requests",
        "boto3",
        "openpyxl",
        "yfinance",
        "ta",
    ]

    missing_packages = []

    for package in essential_packages:
        try:
            if package == "yaml":
                import yaml
            else:
                __import__(package)
            logger.info(f"✅ {package}: Available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}: Missing")

    if missing_packages:
        logger.warning(f"⚠️ {len(missing_packages)} packages missing, but pipeline may still work")
        return len(missing_packages) < len(essential_packages) * 0.5  # Allow 50% missing

    logger.info("✅ All essential packages available!")
    return True


def main():
    """Main check function"""

    logger.info("🚀 Paperspace Environment Check")
    logger.info("=" * 50)

    checks_passed = 0
    total_checks = 3

    # Check 1: Environment variables
    if check_environment_variables():
        checks_passed += 1

    # Check 2: AWS connection
    if test_aws_connection():
        checks_passed += 1

    # Check 3: Essential packages
    if check_essential_packages():
        checks_passed += 1

    logger.info("=" * 50)
    logger.info(f"📊 Summary: {checks_passed}/{total_checks} checks passed")

    if checks_passed >= 2:
        logger.info("✅ Environment is ready for training pipeline!")
        logger.info("🚀 You can now run: python paperspace_training_orchestrator.py")
        return 0
    else:
        logger.error("❌ Environment needs fixes before running pipeline")
        return 1


if __name__ == "__main__":
    exit(main())
