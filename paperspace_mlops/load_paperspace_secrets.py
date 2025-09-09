#!/usr/bin/env python3
"""
Load Paperspace Secrets
=======================

Loads secrets from Paperspace environment and sets them as environment variables.
"""

import logging
import os
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_paperspace_secrets():
    """Load secrets from Paperspace environment"""
    
    logger.info("🔐 Loading Paperspace secrets...")
    
    # Method 1: Check if secrets are in environment variables (newer Paperspace)
    secrets_found = 0
    
    secret_names = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY", 
        "AWS_DEFAULT_REGION"
    ]
    
    for secret_name in secret_names:
        value = os.environ.get(secret_name)
        if value:
            logger.info(f"✅ Found {secret_name} in environment")
            secrets_found += 1
        else:
            logger.info(f"⚠️ {secret_name} not found in environment")
    
    if secrets_found >= 2:  # At least access key and secret key
        logger.info("✅ AWS credentials found in environment variables")
        return True
    
    # Method 2: Check Paperspace secrets file location
    secrets_paths = [
        "/storage/secrets.json",
        "/notebooks/secrets.json", 
        "/tmp/secrets.json",
        "secrets.json"
    ]
    
    for secrets_path in secrets_paths:
        if os.path.exists(secrets_path):
            logger.info(f"📁 Found secrets file: {secrets_path}")
            try:
                with open(secrets_path, 'r') as f:
                    secrets = json.load(f)
                
                # Set AWS credentials from secrets
                for secret_name in secret_names:
                    if secret_name in secrets:
                        os.environ[secret_name] = secrets[secret_name]
                        logger.info(f"✅ Loaded {secret_name} from secrets file")
                        secrets_found += 1
                
                if secrets_found >= 2:
                    logger.info("✅ AWS credentials loaded from secrets file")
                    return True
                    
            except Exception as e:
                logger.warning(f"⚠️ Error reading secrets file {secrets_path}: {e}")
    
    # Method 3: Try to load from Paperspace environment variables with different names
    paperspace_secret_mappings = {
        "PAPERSPACE_AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
        "PAPERSPACE_AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        "PAPERSPACE_AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION"
    }
    
    for paperspace_name, aws_name in paperspace_secret_mappings.items():
        value = os.environ.get(paperspace_name)
        if value:
            os.environ[aws_name] = value
            logger.info(f"✅ Mapped {paperspace_name} -> {aws_name}")
            secrets_found += 1
    
    if secrets_found >= 2:
        logger.info("✅ AWS credentials loaded from Paperspace environment")
        return True
    
    # Method 4: Manual setup instructions
    logger.error("❌ No AWS credentials found!")
    logger.info("📋 To fix this in Paperspace:")
    logger.info("")
    logger.info("Option A - Environment Variables:")
    logger.info("1. Go to your Paperspace notebook")
    logger.info("2. Add environment variables in notebook settings:")
    logger.info("   AWS_ACCESS_KEY_ID=your_access_key")
    logger.info("   AWS_SECRET_ACCESS_KEY=your_secret_key")
    logger.info("   AWS_DEFAULT_REGION=us-east-1")
    logger.info("")
    logger.info("Option B - Set them manually in this notebook:")
    logger.info('   import os')
    logger.info('   os.environ["AWS_ACCESS_KEY_ID"] = "your_access_key"')
    logger.info('   os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret_key"')
    logger.info('   os.environ["AWS_DEFAULT_REGION"] = "us-east-1"')
    logger.info("")
    
    return False


def create_env_file_from_secrets():
    """Create .env file from loaded secrets"""
    
    logger.info("📝 Creating .env file from secrets...")
    
    env_content = f"""# AWS Configuration (loaded from Paperspace secrets)
AWS_ACCESS_KEY_ID={os.environ.get('AWS_ACCESS_KEY_ID', 'not_found')}
AWS_SECRET_ACCESS_KEY={os.environ.get('AWS_SECRET_ACCESS_KEY', 'not_found')}
AWS_DEFAULT_REGION={os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')}

# These will be set by the training pipeline
TELEGRAM_BOT_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    logger.info("✅ Created .env file with AWS credentials")


def main():
    """Main function"""
    
    logger.info("🚀 Loading Paperspace Secrets")
    logger.info("=" * 50)
    
    success = load_paperspace_secrets()
    
    if success:
        create_env_file_from_secrets()
        logger.info("✅ Secrets loaded successfully!")
        logger.info("🚀 You can now run the setup scripts")
        return 0
    else:
        logger.error("❌ Failed to load secrets")
        logger.info("Please follow the instructions above to set AWS credentials")
        return 1


if __name__ == "__main__":
    exit(main())