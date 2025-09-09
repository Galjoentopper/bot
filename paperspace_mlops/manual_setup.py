#!/usr/bin/env python3
"""
Manual AWS Credentials Setup
============================

Simple script to manually set AWS credentials in Paperspace.
Run this in a notebook cell or modify with your actual credentials.
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_aws_credentials():
    """Set AWS credentials manually"""

    logger.info("🔐 Manual AWS Credentials Setup")
    logger.info("=" * 40)

    # REPLACE THESE WITH YOUR ACTUAL VALUES FROM PAPERSPACE SECRETS
    aws_credentials = {
        "AWS_ACCESS_KEY_ID": "AKIA...",  # Replace with your access key from Paperspace secrets
        "AWS_SECRET_ACCESS_KEY": "your_secret_key_here",  # Replace with your secret key
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_MODELS_BUCKET": "your-bucket-name",  # Replace if you have a specific bucket
    }

    # Set environment variables
    for key, value in aws_credentials.items():
        os.environ[key] = value
        if "SECRET" in key:
            logger.info(f"✅ Set {key}: {value[:8]}...")
        else:
            logger.info(f"✅ Set {key}: {value}")

    # Create .env file
    env_content = f"""# AWS Configuration
AWS_ACCESS_KEY_ID={aws_credentials['AWS_ACCESS_KEY_ID']}
AWS_SECRET_ACCESS_KEY={aws_credentials['AWS_SECRET_ACCESS_KEY']}
AWS_DEFAULT_REGION={aws_credentials['AWS_DEFAULT_REGION']}
AWS_MODELS_BUCKET={aws_credentials['AWS_MODELS_BUCKET']}
"""

    with open(".env", "w") as f:
        f.write(env_content)

    logger.info("✅ Created .env file")
    logger.info("✅ AWS credentials are now available!")
    logger.info("🚀 You can now run: python paperspace_training_orchestrator.py")


def verify_credentials():
    """Verify credentials are working"""

    try:
        import boto3

        s3_client = boto3.client("s3")
        s3_client.list_buckets()

        logger.info("✅ AWS connection verified!")
        return True

    except Exception as e:
        logger.error(f"❌ AWS connection failed: {e}")
        return False


if __name__ == "__main__":
    setup_aws_credentials()
    verify_credentials()
