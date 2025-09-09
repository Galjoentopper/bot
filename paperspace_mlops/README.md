# Paperspace MLOps Pipeline

Complete automated MLOps pipeline for training trading bot models on Paperspace Gradient and automatically deploying them to production servers.

## 🚀 Quick Start

### On Paperspace Gradient

1. **Upload your trading bot repository** to Paperspace (or set environment variables for Git cloning)

2. **Run the setup script**:
   ```bash
   python paperspace_mlops/paperspace_setup.py
   ```

3. **Start training**:
   ```bash
   python start_training.py
   ```

Or run the orchestrator directly:
```bash
python paperspace_mlops/paperspace_training_orchestrator.py
```

### On Production Server

1. **Start the webhook server** (to receive models automatically):
   ```bash
   python start_production_webhook.py
   ```

2. **Configure environment variables** (see below)

## 📋 Prerequisites

### Paperspace Environment Variables

Set these in your Paperspace notebook environment:

```bash
# Optional: Git repository (if not uploading manually)
export TRADING_BOT_REPO_URL="https://github.com/your-username/trading-bot.git"

# Production server endpoints
export PRODUCTION_UPLOAD_ENDPOINT="https://your-production-server.com/api/upload"
export PRODUCTION_WEBHOOK_URL="https://your-production-server.com/webhook/models"
export PRODUCTION_API_KEY="your-api-key"

# Telegram notifications
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"

# Cloud storage (choose one)
export AWS_MODELS_BUCKET="your-s3-bucket"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Or GitHub releases
export GITHUB_MODELS_REPO="your-username/models-repo"
export GITHUB_TOKEN="your-github-token"

# Or email fallback
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export MODELS_RECIPIENT_EMAIL="production-server@yourdomain.com"
```

### Production Server Environment Variables

```bash
# API key for webhook authentication
export PRODUCTION_API_KEY="your-api-key"

# Webhook server port
export WEBHOOK_PORT="5000"

# Telegram notifications
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"

# Cloud credentials (if using cloud storage)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

## 🏗️ Architecture

### Paperspace Training Pipeline

```
1. Environment Setup
   ├── Install dependencies
   ├── Setup directories
   └── Configure MLflow

2. Data Acquisition
   ├── Fetch market data
   ├── Build datasets
   └── Feature engineering

3. Model Training
   ├── Train GRU models
   ├── Train LightGBM models
   ├── Train PPO models
   └── Parallel execution with time management

4. Model Packaging
   ├── Collect trained models
   ├── Package metadata
   └── Create zip archive

5. Model Transfer
   ├── Try HTTP upload
   ├── Fallback to cloud storage
   ├── Fallback to GitHub releases
   └── Final fallback to email

6. Production Notification
   └── Send webhook to production server
```

### Production Import Pipeline

```
1. Webhook Notification
   ├── Validate request
   ├── Extract download info
   └── Queue import job

2. Model Download
   ├── Download from cloud/GitHub
   ├── Validate package integrity
   └── Extract to temp directory

3. Model Import
   ├── Backup existing models
   ├── Run import_models.sh
   └── Validate imported models

4. Deployment
   ├── Test model loading
   ├── Send success notification
   └── Clean up temp files
```

## 📁 File Structure

```
paperspace_mlops/
├── README.md                           # This file
├── paperspace_setup.py                 # Paperspace environment setup
├── paperspace_training_orchestrator.py # Main training orchestrator
├── model_transfer_service.py           # Enhanced model transfer
└── production_import_handler.py        # Production import automation

# Root level scripts
├── start_training.py                   # Quick start training script
└── start_production_webhook.py         # Production webhook server
```

## 🛠️ Transfer Methods

The system supports multiple transfer methods with automatic fallback:

### 1. Direct HTTP Upload (Preferred)
- Direct upload to production server endpoint
- Fastest and most reliable method
- Requires `PRODUCTION_UPLOAD_ENDPOINT` and `PRODUCTION_API_KEY`

### 2. Cloud Storage
- **AWS S3**: Requires `AWS_MODELS_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- **Google Cloud**: Requires `GCP_MODELS_BUCKET` and service account credentials
- **Azure Blob**: Requires `AZURE_MODELS_CONTAINER`, `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`

### 3. GitHub Releases
- Upload as release assets to public/private repository
- Requires `GITHUB_MODELS_REPO` and `GITHUB_TOKEN`
- Good for version tracking and public models

### 4. Email (Fallback)
- Sends models as email attachments (max 20MB)
- Requires SMTP configuration
- Last resort method

## ⚙️ Configuration

### Time Management

The orchestrator automatically manages the 6-hour Paperspace limit:

- **Environment Setup**: 0.1 hours
- **Data Preparation**: 0.5 hours
- **Model Training**: 2-4 hours (adaptive)
- **Packaging**: 0.3 hours
- **Transfer**: 0.2 hours
- **Buffer**: 0.5 hours

Training parameters are automatically reduced if time is limited.

### Fast Training Mode

Automatically activated when time is limited:

- Reduces epochs for neural networks
- Limits boosting rounds for LightGBM
- Reduces timesteps for PPO
- Limits cross-validation splits

### Parallel Training

- Trains multiple models simultaneously
- Adaptive worker count based on available CPU cores
- Time budgets distributed across model/symbol combinations

## 📊 Monitoring

### Progress Tracking

- **Pipeline State**: `logs/pipeline_state.json`
- **Detailed Logs**: `logs/paperspace_training_*.log`
- **MLflow Tracking**: `mlruns/` directory

### Production Monitoring

- **Import Logs**: `logs/model_imports_*.log`
- **Webhook Logs**: `logs/webhook_server.log`
- **Telegram Notifications**: Real-time status updates

## 🔧 Troubleshooting

### Common Issues

**Environment Setup Fails**
```bash
# Check Python version (3.8+ required)
python --version

# Check available disk space
df -h

# Install missing packages manually
pip install torch lightgbm stable-baselines3
```

**Data Fetching Fails**
```bash
# Check internet connectivity
ping binance.com

# Verify API limits haven't been exceeded
# Wait 1 hour and retry
```

**Model Training Timeout**
```bash
# Reduce training parameters in training_config.yaml
# Or increase max_runtime_hours in orchestrator
```

**Transfer Fails**
```bash
# Check environment variables
env | grep -E "(PRODUCTION|AWS|GITHUB)"

# Test connectivity
curl -I $PRODUCTION_UPLOAD_ENDPOINT

# Check webhook endpoint
curl http://your-production-server:5000/health
```

### Recovery Procedures

**Failed Training**
- Emergency export automatically saves partial models
- Check `exports/emergency_models_*.zip`
- Manual transfer using backup methods

**Failed Import**
- Production server automatically restores backup
- Check `model_backups/` directory
- Manual rollback: `./import_models.sh backup_file.zip`

## 🔐 Security

### API Keys
- Never commit API keys to repository
- Use environment variables or secure storage
- Rotate keys regularly

### Network Security
- Use HTTPS for all endpoints
- Implement proper authentication
- Monitor access logs

### Model Validation
- All imported models are automatically validated
- Failed validation triggers automatic rollback
- Test predictions before deployment

## 📈 Performance Tips

### Paperspace Optimization
- Use GPU instances for neural network training
- Enable persistent storage for data caching
- Monitor memory usage during training

### Production Optimization
- Run webhook server on separate port
- Use reverse proxy for HTTPS termination
- Implement rate limiting for webhook endpoint

### Cost Optimization
- Use Paperspace free tier (6-hour limit)
- Cache datasets between runs
- Use spot instances for longer training

## 🆘 Support

### Getting Help

1. **Check Logs**: Always check logs first for error details
2. **Environment**: Verify all required environment variables are set
3. **Connectivity**: Test network connectivity to all endpoints
4. **Validation**: Run quick tests to isolate issues

### Debugging Commands

```bash
# Test production webhook
curl -X POST http://localhost:5000/health

# Validate environment
python -c "import torch, lightgbm, stable_baselines3; print('All packages available')"

# Test model import
python quick_test_system.py --models-only

# Check model structure
find models/ -name "*.pkl" -o -name "*.pt" | head -10
```

## 🚢 Deployment Workflow

### Complete End-to-End Workflow

1. **Prepare Paperspace**:
   ```bash
   # Upload repository or set git URL
   # Set environment variables
   # Run setup script
   python paperspace_mlops/paperspace_setup.py
   ```

2. **Start Production Webhook**:
   ```bash
   # On production server
   python start_production_webhook.py --port 5000
   ```

3. **Start Training**:
   ```bash
   # On Paperspace
   python start_training.py
   ```

4. **Monitor Progress**:
   - Watch Paperspace logs for training progress
   - Monitor production webhook for import notifications
   - Check Telegram for status updates

5. **Validate Deployment**:
   ```bash
   # On production server
   python quick_test_system.py
   ls -la models/
   ```

The entire process is designed to be hands-off once started, with automatic error handling and recovery at every step.
