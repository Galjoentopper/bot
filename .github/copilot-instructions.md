# Bot Kilo Trading System

ALWAYS follow these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

Bot Kilo is a comprehensive cryptocurrency trading system with advanced ML/RL model validation, feature drift monitoring, and metadata management capabilities. The system supports GRU (neural networks), LightGBM (gradient boosting), and PPO (reinforcement learning) models for trading Bitcoin, Ethereum, and other cryptocurrencies.

## Working Effectively

### Bootstrap and Setup
- Install dependencies: `pip install -r requirements.txt` -- takes 3 minutes, NEVER CANCEL. Set timeout to 5+ minutes.
- Initialize ML runtime: `python startup_init.py --verbose` -- takes 6 seconds, creates mlruns/, logs/, models/, checkpoints/ directories
- Check initialization: `python startup_init.py --check`

### Build and Training
- Train models (quick test): `python scripts/enhanced_trainer.py --models lightgbm --symbols BTCEUR --n-splits 2 --verbose --start-date 2024-01-01` -- takes 16 seconds
- Train all models: `python scripts/enhanced_trainer.py` -- takes 15-45 minutes, NEVER CANCEL. Set timeout to 60+ minutes.
- Resume training: `python scripts/enhanced_trainer.py --resume`
- Windows training: `train_models.bat` or `train_models.sh` for Linux

### Validation and Testing
- Validate models: `python scripts/validate_models.py --models-dir models --verbose` -- takes 4 seconds
- Generate features from metadata: `python scripts/generate_features_from_metadata.py --models-dir models --output-dir . --verbose` -- takes <1 second
- Test basic adapters: `python test_adapter_simple.py` -- takes 4.5 seconds
- Test metadata hygiene: `python test_metadata_hygiene.py` (may have logging issues)

### Running the Trading System
- Start enhanced trader: `python scripts/enhanced_trader.py` -- starts immediately, may be blocked by Binance API geo-restrictions
- Start with specific symbols: `python scripts/enhanced_trader.py --symbols BTCEUR --paper-trading`
- Deploy for production: `deploy_trading.bat` (Windows) - includes automatic model import and feature generation

### Validation Scenarios
ALWAYS manually validate any new code by running through these complete end-to-end scenarios:

#### Training to Deployment Workflow
1. `python startup_init.py --verbose` -- verify MLflow directories created
2. `python scripts/enhanced_trainer.py --models lightgbm --symbols BTCEUR --n-splits 2 --verbose --start-date 2024-01-01` -- verify model training completes
3. `python scripts/validate_models.py --models-dir models --verbose` -- verify model validation passes
4. `python scripts/generate_features_from_metadata.py --models-dir models --output-dir . --verbose` -- verify feature generation
5. `python scripts/enhanced_trader.py --symbols BTCEUR --paper-trading` -- verify trader starts and loads models (will show API geo-restrictions)

#### Model Loading and Feature Alignment
1. Check model features: LightGBM=113, GRU=119, PPO=13 features
2. Verify feature_mapping.json and feature_config.json are created
3. Test that enhanced_trader.py successfully loads models without schema drift errors

## Timing Expectations and Timeouts

- **NEVER CANCEL**: Dependency installation can take 3+ minutes. Always set timeout to 5+ minutes.
- **NEVER CANCEL**: Full model training can take 15-45 minutes. Always set timeout to 60+ minutes for production training.
- **NEVER CANCEL**: Model validation and testing typically complete in under 30 seconds.
- Quick training test (2 splits, 1 symbol): ~16 seconds
- Model validation: ~4 seconds
- Feature generation: <1 second
- System initialization: ~6 seconds

## Code Quality and Linting

- Format code: `black .` -- takes 8 seconds, 101 files need reformatting currently
- Check critical lint errors: `flake8 --select=E9,F63,F7,F82 .` -- takes 2 seconds
- Always run `black .` before committing changes
- Current codebase has formatting issues that should be addressed

## Configuration and Architecture

### Key Configuration Files
- `src/config/config_trading.yaml` - Trading configuration
- `src/config/config_training.yaml` - Training configuration  
- `feature_config.json` and `feature_mapping.json` - Feature alignment
- `.env` - Environment variables (API keys, Telegram tokens)

### Critical Components
- `scripts/enhanced_trader.py` - Main trading system (75KB, comprehensive)
- `scripts/enhanced_trainer.py` - Model training system (46KB)
- `src/validation/` - Validation and drift monitoring systems
- `src/data_pipeline/` - Feature engineering and data processing
- `src/models/` - Model adapters and trainers

### Directory Structure
```
/
├── scripts/                    # Main application scripts
├── src/                       # Source code modules
│   ├── data_pipeline/         # Feature engineering
│   ├── validation/            # Model validation & drift monitoring
│   ├── models/                # Model trainers and adapters
│   └── config/                # Configuration management
├── models/                    # Trained model artifacts
├── data/                      # Market data (SQLite databases)
├── mlruns/                    # MLflow experiment tracking
├── logs/                      # System logs
└── checkpoints/               # Training checkpoints
```

## Data and Dependencies

### Market Data
- Located in `data/` directory as SQLite databases
- Available symbols: BTCEUR, ETHEUR, ADAEUR, DOTEUR, LINKEUR  
- 30-minute intervals in .db files (e.g., `btceur_30m.db`)

### Python Dependencies
- Core: pandas>=2.0.0, numpy>=1.24.0, PyYAML>=6.0
- ML: torch>=2.0.0, lightgbm>=4.0.0, scikit-learn>=1.3.0
- RL: stable-baselines3>=2.7.0, gymnasium>=0.29.0
- APIs: python-binance>=1.0.0, ccxt>=4.0.0
- Monitoring: mlflow>=2.5.0, wandb>=0.15.0

## Common Tasks and Outputs

### Repository Root
```bash
ls -la
# Key files: README.md, requirements.txt, setup.py, deploy_trading.bat
# Key dirs: scripts/, src/, models/, data/, mlruns/, logs/
```

### Model Training Output
```bash
python scripts/enhanced_trainer.py --models lightgbm --symbols BTCEUR --n-splits 2 --verbose
# Creates: models/lightgbm/BTCEUR/lightgbm/YYYYMMDD_HHMMSS/model.pkl
# Creates: models/exports/model_transfer_bundle_YYYYMMDD_HHMMSS.zip
# Logs: Training metrics, model validation results
```

### Enhanced Trader Startup
```bash
python scripts/enhanced_trader.py
# Loads models, validates features, starts monitoring
# Shows: "Enhanced trader initialized with $10,000.00"
# Shows: "Loaded X models across Y symbols"
# Note: May show Binance API geo-restriction errors (expected)
```

## Known Issues and Limitations

### API Restrictions
- Binance API may be geo-restricted in some environments
- System will show "Service unavailable from a restricted location" errors
- This is expected behavior and does not indicate system failure

### Testing Infrastructure
- Some test files (like `test_metadata_hygiene.py`) have logging issues
- Use `python test_adapter_simple.py` for reliable testing
- No centralized pytest configuration - tests are individual scripts

### Development Environment
- Code formatting needs attention: 101 files require black formatting
- Some linting errors exist but don't affect functionality
- System works correctly despite formatting issues

## Troubleshooting Common Issues

### "Could not find experiment with ID 0" errors
1. Run `python startup_init.py` to initialize MLflow
2. Check that `mlruns/0/meta.yaml` exists
3. Verify logs/ and models/ directories are created

### Model loading failures
1. Check models exist: `ls -la models/`
2. Run model validation: `python scripts/validate_models.py --models-dir models --verbose`
3. Regenerate feature config: `python scripts/generate_features_from_metadata.py --models-dir models --output-dir . --verbose`

### Feature schema drift
1. The system includes robust feature alignment mechanisms
2. LightGBM models expect 113 features, GRU models expect 119 features, PPO models expect 13 features
3. Feature selector automatically handles alignment

## Production Deployment

### Deployment Checklist
1. Run `deploy_trading.bat` (Windows) or follow manual steps:
2. Ensure models directory exists and has trained models
3. Run feature generation: `python scripts/generate_features_from_metadata.py --models-dir models --output-dir . --verbose`
4. Validate models: `python scripts/validate_models.py --models-dir models --verbose`
5. Start trader: `python scripts/enhanced_trader.py`

### Monitoring and Logs
- All logs written to `logs/` directory
- Trading logs: `logs/trading.log`
- Comprehensive validation and drift monitoring included
- Telegram notifications configured (requires API token)