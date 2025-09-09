#!/usr/bin/env python3
"""
S3 Setup Using Excel Credentials File
====================================

Reads AWS credentials from Excel file and sets up S3 storage
for the Paperspace MLOps pipeline.

Usage:
    python setup_s3_from_excel.py
"""

import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def install_requirements():
    """Install required packages for Excel reading"""
    required_packages = ["pandas", "openpyxl", "xlrd", "boto3"]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            logger.info(f"📦 Installing {package}...")
            import subprocess

            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


def read_excel_credentials(excel_path):
    """Read AWS credentials from Excel file"""

    try:
        import pandas as pd

        logger.info(f"📊 Reading credentials from: {excel_path}")

        # Try to read Excel file (support both .xls and .xlsx)
        try:
            # Try with openpyxl first (for .xlsx)
            df = pd.read_excel(excel_path, engine="openpyxl")
        except Exception:
            try:
                # Fall back to xlrd (for .xls)
                df = pd.read_excel(excel_path, engine="xlrd")
            except Exception:
                # Try without specifying engine
                df = pd.read_excel(excel_path)

        logger.info("✅ Excel file read successfully")
        logger.info(f"📋 Found columns: {list(df.columns)}")
        logger.info(f"📊 Data shape: {df.shape}")

        # Print first few rows to understand structure
        logger.info("📄 File contents:")
        for idx, row in df.iterrows():
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    logger.info(f"  {col}: {value}")
            if idx < 5:  # Show only first few rows
                logger.info("  ---")

        # Try to extract credentials - look for common patterns
        credentials = {}

        # Convert all data to strings for searching
        df_str = df.astype(str).fillna("")

        # Look for credentials in each cell
        for idx, row in df_str.iterrows():
            for col in df_str.columns:
                value = row[col].strip()

                # Check if this cell contains comma-separated access key and secret
                if "," in value and "AKIA" in value:
                    logger.info(f"🔍 Found comma-separated credentials: {value[:20]}...")
                    parts = [part.strip() for part in value.split(",")]

                    for part in parts:
                        # Look for AWS Access Key (starts with AKIA)
                        if part.startswith("AKIA") and len(part) == 20:
                            credentials["AWS_ACCESS_KEY_ID"] = part
                            logger.info(f"✅ Extracted Access Key: {part[:8]}...")

                        # Look for Secret Key (40 characters, contains alphanumeric + symbols)
                        elif len(part) >= 35 and any(c in part for c in ["+", "/", "="]):
                            credentials["AWS_SECRET_ACCESS_KEY"] = part
                            logger.info(f"✅ Extracted Secret Key: {part[:8]}...")

                # Look for individual keys
                elif value.startswith("AKIA") and len(value) == 20:
                    credentials["AWS_ACCESS_KEY_ID"] = value
                    logger.info(f"✅ Found Access Key: {value[:8]}...")

                # Look for Secret Key (long string with AWS secret key characteristics)
                elif len(value) >= 35 and any(c in value for c in ["+", "/", "="]):
                    credentials["AWS_SECRET_ACCESS_KEY"] = value
                    logger.info(f"✅ Found Secret Key: {value[:8]}...")

                # Look for region
                elif value.startswith("us-") and len(value) < 20:
                    credentials["AWS_DEFAULT_REGION"] = value
                    logger.info(f"✅ Found Region: {value}")

        # If still not found, try more flexible parsing
        if len(credentials) < 2:
            logger.info("🔍 Trying flexible parsing approach...")

            # Look through all values for anything that looks like AWS credentials
            all_text = " ".join([str(cell) for row in df_str.values for cell in row])

            # Split by common delimiters
            import re

            potential_parts = re.split(r"[,\s\t\n]+", all_text)

            for part in potential_parts:
                part = part.strip()

                if part.startswith("AKIA") and len(part) == 20:
                    credentials["AWS_ACCESS_KEY_ID"] = part
                    logger.info(f"✅ Found Access Key via parsing: {part[:8]}...")

                elif len(part) >= 35 and any(c in part for c in ["+", "/", "="]):
                    credentials["AWS_SECRET_ACCESS_KEY"] = part
                    logger.info(f"✅ Found Secret Key via parsing: {part[:8]}...")

        return credentials

    except Exception as e:
        logger.error(f"❌ Failed to read Excel file: {e}")
        return None


def setup_environment_variables(credentials):
    """Set environment variables from credentials"""

    logger.info("🔧 Setting up environment variables...")

    # Set default region if not provided
    if "AWS_DEFAULT_REGION" not in credentials:
        credentials["AWS_DEFAULT_REGION"] = "us-east-1"
        logger.info("✅ Using default region: us-east-1")

    # Validate required credentials
    required_keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing_keys = [key for key in required_keys if key not in credentials]

    if missing_keys:
        logger.error(f"❌ Missing required credentials: {missing_keys}")
        logger.error("Please check your Excel file contains:")
        logger.error("  - AWS Access Key (starts with AKIA)")
        logger.error("  - AWS Secret Access Key (40 characters)")
        return False

    # Set environment variables
    for key, value in credentials.items():
        os.environ[key] = value
        logger.info(f"✅ Set {key}: {value[:8] if 'SECRET' not in key else value[:8] + '...'}...")

    return True


def main():
    """Main function"""

    logger.info("🚀 Setting up S3 storage from Excel credentials")
    logger.info("=" * 60)

    try:
        # Install required packages
        install_requirements()

        # Find Excel file
        excel_path = Path("paperspace_mlops/sleutel.xls")
        if not excel_path.exists():
            # Try .xlsx extension
            excel_path = Path("paperspace_mlops/sleutel.xlsx")
            if not excel_path.exists():
                logger.error("❌ Excel file not found!")
                logger.error(
                    "Expected: paperspace_mlops/sleutel.xls or paperspace_mlops/sleutel.xlsx"
                )
                return 1

        # Read credentials from Excel
        credentials = read_excel_credentials(excel_path)
        if not credentials:
            logger.error("❌ Failed to extract credentials from Excel file")
            return 1

        # Setup environment variables
        if not setup_environment_variables(credentials):
            return 1

        # Import and run S3 setup
        logger.info("🎯 Running S3 setup...")
        from setup_s3_storage import S3StorageSetup

        s3_setup = S3StorageSetup()
        success = s3_setup.run_setup()

        if success:
            logger.info("🎉 Complete setup successful!")
            logger.info("Your Paperspace pipeline can now use S3 storage!")

            # Save credentials to environment file for future use
            env_file = Path(".env")
            with open(env_file, "w") as f:
                for key, value in credentials.items():
                    f.write(f"{key}={value}\n")
            logger.info(f"✅ Credentials saved to {env_file}")

            return 0
        else:
            logger.error("❌ S3 setup failed")
            return 1

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
