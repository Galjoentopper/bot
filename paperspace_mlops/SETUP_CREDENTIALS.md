# Quick AWS Credentials Setup for Paperspace

Since Paperspace secrets aren't automatically loading, use this simple approach:

## Option 1: Run in a Notebook Cell

```python
import os

# Set your AWS credentials (get these from your Paperspace secrets)
os.environ["AWS_ACCESS_KEY_ID"] = "AKIA..."  # Your access key from Paperspace
os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret_key_here"  # Your secret key
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_MODELS_BUCKET"] = "your-bucket-name"  # Use your actual bucket name

print("✅ AWS credentials set!")

# Verify they work
try:
    import boto3
    s3 = boto3.client('s3')
    s3.list_buckets()
    print("✅ AWS connection verified!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

## Option 2: Run the Manual Setup Script

```bash
!python bot/paperspace_mlops/manual_setup.py
```

Then edit the script with your actual credentials from Paperspace secrets.

## Option 3: Direct Environment Variables in Terminal

```bash
export AWS_ACCESS_KEY_ID="AKIA..."  # Your access key
export AWS_SECRET_ACCESS_KEY="your_secret_key_here"  # Your secret key
export AWS_DEFAULT_REGION="us-east-1"
export AWS_MODELS_BUCKET="your-bucket-name"
```

## After Setting Credentials

Run the training pipeline:

```bash
!python bot/paperspace_mlops/paperspace_training_orchestrator.py
```

The pipeline will now:
- ✅ Fetch data for all 5 symbols (BTCEUR, ETHEUR, ADAEUR, DOTEUR, LINKEUR)
- ✅ Train 15 models (3 types × 5 symbols)
- ✅ Upload to your S3 bucket
- ✅ Notify your production server
