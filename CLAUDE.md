# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is a **model training environment** for a cryptocurrency trading bot system. The machine serves as the dedicated training server that develops and exports machine learning models to S3 for deployment on production servers. The system uses Jupyter notebooks for training orchestration and includes comprehensive ML pipeline components.

**CRITICAL**: This is a training-only environment. Models are trained here, exported to AWS S3, then imported by production trading servers. Never attempt live trading or deployment operations on this training machine.

## Key Commands

### Environment Setup

**CRITICAL: Install dependencies in correct order to avoid training failures**

```bash
# Navigate to bot directory first
cd bot

# For training environment (Paperspace/Jupyter) - RECOMMENDED
pip install -r requirements-training.txt

# OR for full development environment (includes all features)
pip install -r requirements.txt

# Verify critical training dependencies
python -c "import ta, lightgbm, torch, stable_baselines3; print('✅ Training dependencies OK')"
```

**Common Dependency Issues:**
- `ModuleNotFoundError: No module named 'ta'` → Run `pip install -r requirements-training.txt`
- `ModuleNotFoundError: No module named 'lightgbm'` → Ensure LightGBM is installed
- Training notebook failures → Use `requirements-training.txt` for training-only environments

### Model Training Pipeline
```bash
# Main training workflow (run from bot/ directory)
cd bot
python paperspace_mlops/paperspace_training.py  # Primary training script

# User-friendly CLI wrapper - RECOMMENDED for interactive use
python scripts/run_full_training.py --full        # Full ensemble training
python scripts/run_full_training.py --quick       # Quick test run
python scripts/run_full_training.py --full --symbols BTCEUR ETHEUR  # Specific symbols
python scripts/run_full_training.py --full --models ppo gru          # Specific models

# Advanced ensemble training with full pipeline
python src/training/enhanced_ensemble_trainer.py --config config/training_config.yaml

# Train specific model types (note: correct config path)
python scripts/enhanced_trainer.py --config config/training_config.yaml --model-type gru
python scripts/enhanced_trainer.py --config config/training_config.yaml --model-type lightgbm
python scripts/enhanced_trainer.py --config config/training_config.yaml --model-type ppo
```

### S3 Model Export
```bash
# Setup S3 storage and export trained models
python setup_s3_storage.py

# Export models to S3 bucket for production deployment
python paperspace_mlops/export_to_s3.py

# Automated S3 export (included in training pipeline when AWS credentials available)
# S3 export happens automatically during training if environment variables are set
```

### Development and Testing
```bash
# Code quality and testing
black .          # Format Python code
flake8 .         # Lint code
pytest           # Run test suite (80% coverage required)
pytest -v        # Verbose test output with line numbers
pytest tests/    # Run specific test directory
pytest -m unit   # Run only unit tests
pytest -m integration  # Run only integration tests
pytest -m performance  # Run only performance tests
pytest --cov-report=html  # Generate HTML coverage report

# System validation
python final_test_system.py      # Comprehensive system validation
python quick_test_system.py      # Quick validation tests
python test_config_scenarios.py  # Configuration testing

# Script utilities
python scripts/validate_telegram_system.py  # Test notification system
python scripts/system_optimizer.py          # Optimize system performance
bash scripts/health_check.sh                # System health verification
```

### Jupyter Notebook Training

**MAIN TRAINING METHOD - Use this for ensemble training**

```bash
# Install training dependencies first (CRITICAL)
cd bot
pip install -r requirements-training.txt

# Start Jupyter for interactive training
jupyter notebook

# Open Train.ipynb in browser and run all cells
# This executes the complete ensemble training pipeline:
# - PPO (Reinforcement Learning) models
# - GRU (Recurrent Neural Network) models
# - LightGBM (Gradient Boosting) models
# - All 5 symbols: BTCEUR, ETHEUR, ADAEUR, DOTEUR, LINKEUR
```

**Train.ipynb Workflow:**
1. **Cell 1**: Environment validation - checks dependencies, GPU, AWS credentials
2. **Cell 2**: Full ensemble training execution - trains all 15 models (3 types × 5 symbols)
3. **Cell 3**: Results analysis - validates trained models and export status

**Common Training Issues:**
- `No module named 'ta'` → Install `pip install -r requirements-training.txt`
- `No module named 'lightgbm'` → Missing in current environment, install dependencies
- AWS export disabled → Set environment variables for S3 export (optional)
- GPU not detected → Training will use CPU (slower but functional)

## Architecture Overview

### Training Pipeline (`src/training/`)
- `enhanced_ensemble_trainer.py`: Advanced multi-model training orchestrator
- Implements walk-forward validation for time series data
- Supports hyperparameter optimization via Optuna
- Manages per-symbol model training with parallel processing

### Model Components (`src/models/`)
- `gru_trainer.py`: Recurrent neural network for sequential pattern recognition
- `lgbm_trainer.py`: Gradient boosting for structured feature learning
- `ppo_trainer.py`: Reinforcement learning agent for dynamic strategy optimization
- Each model includes validation, saving, and metadata persistence

### Data Processing (`src/data_pipeline/`)
- `features.py`: 200+ technical indicators and market features
- `target_engineering.py`: Multi-timeframe target variable creation
- `trading_features.py`: Advanced trading-specific feature engineering
- `dataset_builder.py`: Unified dataset construction for all models
- `data_preprocessor.py`: Normalization and preprocessing pipeline

### Validation Framework (`src/validation/`)
- `ensemble_validator.py`: Cross-validation for ensemble models
- `data_validator.py`: Data quality and consistency validation
- Walk-forward validation prevents data leakage
- Performance metrics collection and reporting

### Export and Deployment (`paperspace_mlops/`)
- `paperspace_training.py`: Main training orchestrator for Paperspace environment
- `export_to_s3.py`: Model packaging and S3 upload functionality
- `setup_s3_storage.py`: S3 configuration and bucket management
- Model packaging includes metadata for production compatibility

## Configuration Management

### Configuration Directory Structure: `config/`
- `training_config.yaml`: Master training configuration
- `superior_training_config.yaml`: Advanced ensemble configuration
- `telegram_config.yaml`: Notification system settings
- `logging_config.yaml`: Logging configuration
- `feature_config.yaml`: Feature engineering parameters

### Master Configuration: `config/training_config.yaml`

**Data Configuration**
- `symbols`: Trading pairs for model training ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']
- `interval`: Timeframe for training data ('30m' optimal)
- `lookback_days`: Historical data range (365 days default)
- `data_sources`: Primary/fallback data providers (binance/yfinance)
- `use_local_databases`: Use local SQLite databases (true for training environment)

**Training Parameters**
- `models`: List of model types to train ['ppo', 'gru', 'lightgbm'] (PPO first for priority)
- `validation_split`: Train/validation data split (0.2)
- `test_split`: Test data split (0.1)
- `cv_splits`: Cross-validation folds (5)
- `embargo_period`: Temporal embargo in hours (24)
- `optuna_trials`: Hyperparameter optimization iterations (100)
- `optuna_timeout`: Optimization timeout in seconds (7200)

**Resource Allocation**
- `max_workers`: Parallel training processes (8)
- `memory_limit`: Training memory constraints ('5GB')
- `gpu_enabled`: GPU utilization (true - RTX A4000 detected)
- `gpu_memory_fraction`: GPU memory allocation (0.8)
- `batch_size`: Training batch size (512)
- `epochs`: Maximum training epochs (100)
- `early_stopping_patience`: Early stopping patience (10)

### Environment Variables (`.env`)

**Required for S3 Export (optional - training works without AWS):**
```bash
# Create .env file in bot/ directory
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
AWS_MODELS_BUCKET=your_models_bucket
TELEGRAM_BOT_TOKEN=your_bot_token (for notifications)
```

**Note:** Models can be trained and saved locally without AWS credentials. S3 export is only needed for production deployment.

## Training Workflow

### Standard Training Process
1. **Data Acquisition**: Fetch historical market data for configured symbols
2. **Feature Engineering**: Generate 200+ technical indicators per symbol
3. **Model Training**: Train GRU, LightGBM, and PPO models with walk-forward validation
4. **Validation**: Cross-validate models with temporal splits
5. **Export**: Package models and metadata for S3 upload
6. **Deployment**: Export trained models to S3 for production server import

### Jupyter Notebook Workflow
1. **Setup**: Run `Train.ipynb` cells to initialize environment
2. **Training**: Execute training pipeline cells for model development
3. **Validation**: Review training metrics and validation results
4. **Export**: Upload completed models to S3 storage

### Advanced Features
- **Hyperparameter Optimization**: Automated tuning via Optuna
- **Walk-Forward Validation**: Prevents data leakage in time series
- **Multi-Symbol Training**: Parallel training across cryptocurrency pairs
- **Ensemble Integration**: Combines multiple model predictions

## Model Architecture Patterns

### Per-Symbol Training
Each symbol (BTCEUR, ETHEUR, etc.) has dedicated models stored in:
- `models/gru/{SYMBOL}/`
- `models/lightgbm/{SYMBOL}/`
- `models/ppo/{SYMBOL}/`

### Feature Consistency
- `ModelMetadata` class ensures feature alignment between training and inference
- Feature names and order persisted in `models/metadata/features_{SYMBOL}.json`
- Critical for production deployment compatibility

### Temporal Validation
- Walk-forward validation with embargo periods
- Prevents look-ahead bias in time series data
- Ensures models perform on truly unseen future data

## Development Guidelines

### Adding New Models
1. Inherit from appropriate base trainer class
2. Implement required methods: `train()`, `predict()`, `save_model()`, `load_model()`
3. Add model metadata persistence for feature consistency
4. Update `training_config.yaml` model list
5. Test with validation pipeline

### Feature Development
1. Add new features in `src/data_pipeline/features.py`
2. Ensure consistent naming and scaling
3. Update feature metadata tracking
4. Validate against existing models for consistency

### Configuration Changes
1. Modify `training_config.yaml` for parameter updates
2. Validate configuration with `test_config_scenarios.py`
3. Test training pipeline with new parameters
4. Document configuration changes

## S3 Integration

### Model Export Structure
```
models_bucket/
├── gru/
│   ├── BTCEUR/
│   │   ├── model.pt
│   │   └── metadata.json
│   └── ...
├── lightgbm/
│   └── ...
└── ppo/
    └── ...
```

### Export Process
- Models packaged with complete metadata
- Feature lists preserved for production alignment
- Training statistics included for monitoring
- Version control for model updates

## Test Framework Configuration

### Pytest Setup (`pytest.ini`)
- **Coverage Requirement**: 80% minimum code coverage
- **Test Categories**: Unit, integration, contract, performance, external, model, trading, config
- **Output Formats**: Terminal, HTML, XML, JUnit XML
- **Async Support**: Automatic asyncio mode for async tests

### Test Structure
- `tests/unit/`: Component-level tests
- `tests/integration/`: End-to-end pipeline tests
- `conftest.py`: Shared test fixtures and configuration

### Running Specific Test Categories
```bash
pytest -m unit         # Unit tests only
pytest -m integration  # Integration tests only
pytest -m performance  # Performance tests only
pytest -m slow         # Slow-running tests
pytest -m external     # Tests requiring external services
```

## Important Notes

- **Training Environment Only**: This server trains models; production servers handle trading
- **S3 Integration**: All model deployment occurs via S3 export/import workflow
- **GPU Optimization**: Training benefits significantly from GPU acceleration
- **Memory Management**: Large datasets require careful memory allocation
- **Security**: Never commit AWS credentials; use environment variables only
- **Validation Critical**: Always validate models before export to ensure production compatibility
- **CLI Wrapper**: Use `scripts/run_full_training.py` for user-friendly training execution
- **Configuration Path**: All configs are in `config/` directory, not root directory