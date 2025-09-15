#!/usr/bin/env python3
"""
S3 Storage Setup for Paperspace MLOps Pipeline
==============================================

Creates and configures cost-optimized S3 bucket for model storage
with automatic lifecycle policies to minimize costs.

Usage:
    python setup_s3_storage.py

Environment Variables Required:
    AWS_ACCESS_KEY_ID: Your AWS access key
    AWS_SECRET_ACCESS_KEY: Your AWS secret key
    AWS_DEFAULT_REGION: AWS region (optional, defaults to us-east-1)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("❌ boto3 not installed. Installing...")
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "boto3"], check=True)
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class S3StorageSetup:
    """Setup cost-optimized S3 storage for model packages"""

    def __init__(self):
        self.region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.bucket_name = None
        self.s3_client = None

    def verify_credentials(self):
        """Verify AWS credentials are available"""

        logger.info("🔐 Verifying AWS credentials...")

        try:
            # Try to create S3 client
            self.s3_client = boto3.client("s3", region_name=self.region)

            # Test credentials by listing buckets
            self.s3_client.list_buckets()

            # Get account info
            sts_client = boto3.client("sts", region_name=self.region)
            account_info = sts_client.get_caller_identity()

            logger.info(
                f"✅ Credentials verified for account: {account_info.get('Account', 'Unknown')}"
            )
            logger.info(f"✅ Using region: {self.region}")

            return True

        except NoCredentialsError:
            logger.error("❌ No AWS credentials found!")
            logger.error("Please set environment variables:")
            logger.error("  export AWS_ACCESS_KEY_ID='your-access-key'")
            logger.error("  export AWS_SECRET_ACCESS_KEY='your-secret-key'")
            return False

        except ClientError as e:
            logger.error(f"❌ AWS credentials error: {e}")
            return False

        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return False

    def generate_bucket_name(self):
        """Generate unique bucket name"""

        timestamp = int(time.time())
        self.bucket_name = f"paperspace-models-{timestamp}"
        logger.info(f"📦 Generated bucket name: {self.bucket_name}")
        return self.bucket_name

    def create_bucket(self):
        """Create S3 bucket with cost optimization"""

        logger.info(f"🏗️ Creating S3 bucket: {self.bucket_name}")

        try:
            # Create bucket
            if self.region == "us-east-1":
                # us-east-1 doesn't need LocationConstraint
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )

            logger.info("✅ Bucket created successfully")

            # Enable versioning (good practice for model files)
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name, VersioningConfiguration={"Status": "Enabled"}
            )
            logger.info("✅ Versioning enabled")

            # Set public access block (security)
            self.s3_client.put_public_access_block(
                Bucket=self.bucket_name,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
            logger.info("✅ Public access blocked for security")

            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "BucketAlreadyExists":
                logger.warning("⚠️ Bucket name already exists, generating new name...")
                self.generate_bucket_name()
                return self.create_bucket()
            else:
                logger.error(f"❌ Failed to create bucket: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Unexpected error creating bucket: {e}")
            return False

    def setup_lifecycle_policy(self):
        """Setup cost-optimized lifecycle policy"""

        logger.info("📋 Setting up lifecycle policy for cost optimization...")

        lifecycle_config = {
            "Rules": [
                {
                    "ID": "ModelPackageOptimization",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "model_packages/"},
                    "Transitions": [
                        {
                            "Days": 30,  # Move to IA after 30 days (AWS minimum)
                            "StorageClass": "STANDARD_IA",
                        },
                        {
                            "Days": 60,
                            "StorageClass": "GLACIER",
                        },  # Archive after 60 days
                        {
                            "Days": 180,  # Deep archive after 180 days
                            "StorageClass": "DEEP_ARCHIVE",
                        },
                    ],
                },
                {
                    "ID": "CleanupIncompleteUploads",
                    "Status": "Enabled",
                    "Filter": {},
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                },
            ]
        }

        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name, LifecycleConfiguration=lifecycle_config
            )

            logger.info("✅ Lifecycle policy configured:")
            logger.info("  📁 Standard storage: 0-30 days")
            logger.info("  🏪 Standard-IA: 30-60 days (~$0.0125/GB/month)")
            logger.info("  🧊 Glacier: 60-180 days (~$0.004/GB/month)")
            logger.info("  ❄️ Deep Archive: 180+ days (~$0.00099/GB/month)")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to set lifecycle policy: {e}")
            return False

    def setup_cost_monitoring(self):
        """Setup cost monitoring tags"""

        logger.info("🏷️ Adding cost monitoring tags...")

        try:
            tags = [
                {"Key": "Project", "Value": "PaperspaceTradingBot"},
                {"Key": "Purpose", "Value": "ModelStorage"},
                {"Key": "Environment", "Value": "Production"},
                {"Key": "CostCenter", "Value": "MLOps"},
                {"Key": "CreatedBy", "Value": "AutomatedSetup"},
                {"Key": "CreatedDate", "Value": datetime.now().strftime("%Y-%m-%d")},
            ]

            self.s3_client.put_bucket_tagging(Bucket=self.bucket_name, Tagging={"TagSet": tags})

            logger.info("✅ Cost monitoring tags added")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add tags: {e}")
            return False

    def test_bucket_access(self):
        """Test bucket access with a small test file"""

        logger.info("🧪 Testing bucket access...")

        try:
            # Test upload
            test_content = b"Paperspace MLOps test file"
            test_key = "test/access_test.txt"

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=test_key,
                Body=test_content,
                StorageClass="STANDARD",  # Use Standard initially (IA requires 30-day minimum)
            )
            logger.info("✅ Upload test successful")

            # Test download
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=test_key)
            downloaded_content = response["Body"].read()

            if downloaded_content == test_content:
                logger.info("✅ Download test successful")
            else:
                logger.error("❌ Download test failed - content mismatch")
                return False

            # Test presigned URL generation
            download_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": test_key},
                ExpiresIn=3600,
            )
            logger.info("✅ Presigned URL generation successful")

            # Cleanup test file
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=test_key)
            logger.info("✅ Test cleanup completed")

            return True

        except Exception as e:
            logger.error(f"❌ Bucket access test failed: {e}")
            return False

    def generate_environment_config(self):
        """Generate environment configuration for the pipeline"""

        logger.info("📝 Generating environment configuration...")

        env_config = f"""
# S3 Configuration for Paperspace MLOps Pipeline
# Add these to your environment variables

export AWS_MODELS_BUCKET="{self.bucket_name}"
export AWS_DEFAULT_REGION="{self.region}"

# Your existing AWS credentials (if not already set)
# export AWS_ACCESS_KEY_ID="your-access-key-here"
# export AWS_SECRET_ACCESS_KEY="your-secret-key-here"

# For Paperspace Gradient, add these environment variables:
# 1. Go to your Paperspace project settings
# 2. Add these environment variables:
#    AWS_MODELS_BUCKET={self.bucket_name}
#    AWS_DEFAULT_REGION={self.region}
#    AWS_ACCESS_KEY_ID=your-access-key
#    AWS_SECRET_ACCESS_KEY=your-secret-key

# Cost estimates for typical usage (10 model packages/month, 50MB each):
# - Storage (Standard-IA): ~$0.006/month
# - Requests: ~$0.0001/month
# - Data transfer: ~$0.045/month
# - Total: ~$0.05/month
"""

        # Save to file
        config_file = Path("s3_config.env")
        with open(config_file, "w") as f:
            f.write(env_config)

        logger.info(f"✅ Configuration saved to: {config_file}")

        # Also create a .env template
        env_template = f"""AWS_MODELS_BUCKET={self.bucket_name}
AWS_DEFAULT_REGION={self.region}
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
"""

        env_file = Path(".env.s3")
        with open(env_file, "w") as f:
            f.write(env_template)

        logger.info(f"✅ Environment template saved to: {env_file}")

        return True

    def print_cost_breakdown(self):
        """Print detailed cost breakdown"""

        logger.info("💰 Cost Breakdown (Monthly estimates):")
        logger.info("=" * 50)
        logger.info("Storage Classes:")
        logger.info(f"  📁 Standard (0-1 days): $0.023/GB")
        logger.info(f"  🏪 Standard-IA (1-30 days): $0.0125/GB (~46% savings)")
        logger.info(f"  🧊 Glacier (30-90 days): $0.004/GB (~83% savings)")
        logger.info(f"  ❄️ Deep Archive (90+ days): $0.00099/GB (~96% savings)")
        logger.info("")
        logger.info("Typical Monthly Costs (10 packages, 50MB each):")
        logger.info(f"  💾 Storage: ~$0.006")
        logger.info(f"  📤 PUT requests: ~$0.0001")
        logger.info(f"  📥 GET requests: ~$0.000004")
        logger.info(f"  🌐 Data transfer out: ~$0.045")
        logger.info(f"  📊 Total: ~$0.051/month")
        logger.info("")
        logger.info("💡 Tips to reduce costs further:")
        logger.info("  - Use CloudFront for frequent downloads")
        logger.info("  - Compress model files before upload")
        logger.info("  - Clean up old/unused models regularly")

    def run_setup(self):
        """Run complete S3 setup process"""

        logger.info("🚀 Starting S3 Storage Setup for Paperspace MLOps")
        logger.info("=" * 60)

        # Step 1: Verify credentials
        if not self.verify_credentials():
            return False

        # Step 2: Generate bucket name
        self.generate_bucket_name()

        # Step 3: Create bucket
        if not self.create_bucket():
            return False

        # Step 4: Setup lifecycle policy
        if not self.setup_lifecycle_policy():
            return False

        # Step 5: Add cost monitoring tags
        if not self.setup_cost_monitoring():
            return False

        # Step 6: Test access
        if not self.test_bucket_access():
            return False

        # Step 7: Generate configuration
        if not self.generate_environment_config():
            return False

        # Step 8: Print cost breakdown
        self.print_cost_breakdown()

        logger.info("🎉 S3 Setup Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"✅ Bucket created: {self.bucket_name}")
        logger.info(f"✅ Region: {self.region}")
        logger.info(f"✅ Lifecycle policy: Configured for cost optimization")
        logger.info(f"✅ Configuration files: s3_config.env, .env.s3")
        logger.info("")
        logger.info("🔧 Next steps:")
        logger.info(f"1. Add environment variables from s3_config.env to your system")
        logger.info(f"2. For Paperspace: Add AWS variables to project settings")
        logger.info(f"3. Your Paperspace pipeline will automatically use S3!")

        return True


def main():
    """Main setup function"""

    setup = S3StorageSetup()

    try:
        success = setup.run_setup()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\n⚠️ Setup interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
