# AWS S3 Setup Guide for Paperspace MLOps

Complete guide to setup cost-optimized S3 storage for your Paperspace training pipeline.

## 🚀 Quick Setup

### Step 1: Get AWS Credentials

1. **Sign up for AWS** (if you don't have an account):
   - Go to [aws.amazon.com](https://aws.amazon.com)
   - Create free tier account (includes 5GB S3 free for 12 months)

2. **Create IAM User** (recommended for security):
   ```bash
   # Go to AWS Console → IAM → Users → Create User
   # User name: paperspace-mlops
   # Attach policy: AmazonS3FullAccess
   ```

3. **Get Access Keys**:
   - In IAM → Users → [your-user] → Security credentials
   - Create access key → Application running outside AWS
   - **Save these securely!**

### Step 2: Set Environment Variables

```bash
# Add to your ~/.bashrc or ~/.zshrc
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="abc123..."
export AWS_DEFAULT_REGION="us-east-1"  # Cheapest region
```

### Step 3: Run Setup Script

```bash
# Install dependencies
pip install boto3

# Run the setup script
python setup_s3_storage.py
```

## 📋 What the Script Does

1. **✅ Verifies AWS credentials**
2. **🏗️ Creates optimized S3 bucket** with unique name
3. **📋 Sets up lifecycle policies** for automatic cost optimization:
   - Day 0-1: Standard storage
   - Day 1-30: Standard-IA (~46% cheaper)
   - Day 30-90: Glacier (~83% cheaper)
   - Day 90+: Deep Archive (~96% cheaper)
4. **🏷️ Adds cost monitoring tags**
5. **🧪 Tests bucket access**
6. **📝 Generates configuration files**

## 💰 Cost Breakdown

### Monthly Costs (10 model packages, 50MB each):
- **Storage**: ~$0.006 (Standard-IA pricing)
- **Requests**: ~$0.0001 (PUT/GET operations)
- **Transfer**: ~$0.045 (downloading models)
- **Total**: ~**$0.05/month** 🎯

### Storage Class Pricing:
| Class | Cost/GB/Month | Use Case |
|-------|---------------|----------|
| Standard | $0.023 | Active files (0-1 days) |
| Standard-IA | $0.0125 | Recent models (1-30 days) |
| Glacier | $0.004 | Archive (30-90 days) |
| Deep Archive | $0.00099 | Long-term storage (90+ days) |

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

### 1. Create Bucket
```bash
aws s3 mb s3://paperspace-models-$(date +%s) --region us-east-1
```

### 2. Set Lifecycle Policy
Create `lifecycle.json`:
```json
{
  "Rules": [
    {
      "ID": "ModelOptimization",
      "Status": "Enabled",
      "Filter": {"Prefix": "model_packages/"},
      "Transitions": [
        {"Days": 1, "StorageClass": "STANDARD_IA"},
        {"Days": 30, "StorageClass": "GLACIER"}
      ]
    }
  ]
}
```

Apply policy:
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket your-bucket-name \
  --lifecycle-configuration file://lifecycle.json
```

## 🎯 Integration with Paperspace

### Environment Variables for Paperspace

After running the setup script, add these to your Paperspace project:

1. **Go to**: Paperspace Console → Projects → [Your Project] → Settings
2. **Add Environment Variables**:
   ```
   AWS_MODELS_BUCKET=paperspace-models-1234567890
   AWS_DEFAULT_REGION=us-east-1
   AWS_ACCESS_KEY_ID=AKIA...
   AWS_SECRET_ACCESS_KEY=abc123...
   ```

### Your Pipeline Will Automatically:
- ✅ Upload trained models to S3
- ✅ Generate presigned download URLs
- ✅ Notify production server with download info
- ✅ Production server downloads and imports models

## 🛡️ Security Best Practices

### 1. IAM Policy (Minimal Permissions)
Instead of `AmazonS3FullAccess`, create custom policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::paperspace-models-*",
        "arn:aws:s3:::paperspace-models-*/*"
      ]
    }
  ]
}
```

### 2. Bucket Policies
The script automatically:
- ✅ Blocks all public access
- ✅ Enables versioning
- ✅ Sets up access logging

### 3. Credential Management
- ❌ Never commit credentials to code
- ✅ Use environment variables
- ✅ Rotate keys regularly
- ✅ Use IAM roles in production

## 🔍 Monitoring Costs

### AWS Cost Explorer
1. Go to AWS Console → Cost Management → Cost Explorer
2. Filter by Service: S3
3. Group by: Storage Class
4. View daily/monthly breakdown

### CloudWatch Metrics
- **BucketSizeBytes**: Monitor storage usage
- **ObjectCount**: Track number of objects
- **DataTransfer**: Monitor download costs

### Cost Alerts
```bash
# Set up billing alerts
aws budgets create-budget \
  --account-id YOUR-ACCOUNT-ID \
  --budget '{
    "BudgetName": "S3-Models-Budget",
    "BudgetLimit": {"Amount": "5", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
```

## 🚨 Troubleshooting

### Common Issues:

**1. "Access Denied" Error**
```bash
# Check credentials
aws sts get-caller-identity

# Verify bucket permissions
aws s3 ls s3://your-bucket-name
```

**2. "Bucket Already Exists"**
- Script automatically generates new name
- Or manually specify unique name

**3. "Region Mismatch"**
```bash
# Ensure consistent region
export AWS_DEFAULT_REGION="us-east-1"
```

**4. High Costs**
- Check lifecycle policies are applied
- Monitor data transfer (use CloudFront if needed)
- Compress model files before upload

## 📊 Cost Optimization Tips

### 1. Compression
```python
# In your pipeline, compress before upload
import gzip
with gzip.open('model.pkl.gz', 'wb') as f:
    pickle.dump(model, f)
```

### 2. Intelligent Tiering
```bash
# Enable for automatic optimization
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket your-bucket \
  --id EntireBucket \
  --intelligent-tiering-configuration '{
    "Id": "EntireBucket",
    "Status": "Enabled",
    "Filter": {"Prefix": ""},
    "Tiering": {
      "AccessTier": "ARCHIVE_ACCESS",
      "Days": 90
    }
  }'
```

### 3. Cleanup Old Models
```python
# Add to your pipeline
from datetime import datetime, timedelta

def cleanup_old_models(bucket, days=30):
    cutoff = datetime.now() - timedelta(days=days)
    # Delete objects older than cutoff
```

## 🎉 Ready to Go!

After setup, your pipeline automatically uses S3 for:
- ✅ **Paperspace**: Uploads trained models
- ✅ **Transfer**: Efficient, reliable delivery
- ✅ **Production**: Downloads and imports
- ✅ **Cost**: Optimized for minimal expense

**Total setup time**: ~5 minutes
**Monthly cost**: ~$0.05 for typical usage
**Reliability**: 99.999999999% (11 9's) durability

Your MLOps pipeline just got enterprise-grade storage at hobby-project prices! 🚀
