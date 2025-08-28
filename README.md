# Bot Kilo Trading System

A comprehensive trading system with advanced model validation, feature drift monitoring, and metadata management.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Enhanced Trading System
```bash
python scripts/enhanced_trader.py
```

### 3. Test Metadata Hygiene (Optional)
```bash
python test_metadata_hygiene.py
```

## Key Features

### Model Validation & Monitoring
- **Schema Validation**: Automatic validation of model inputs using Great Expectations
- **Feature Drift Detection**: Real-time monitoring of feature distributions and statistical properties
- **Advanced Drift Algorithms**: KS test, Wasserstein distance, PSI, Jensen-Shannon divergence
- **Enhanced Logging**: Comprehensive logging of validation decisions and drift events

### Metadata Management
- **Automated Regeneration**: Periodic regeneration of model metadata
- **Hygiene Validation**: Checks for missing, corrupted, or outdated metadata
- **Model Cleanup**: Automatic removal of outdated models based on age thresholds

### Model Support
- **GRU Models**: Recurrent neural networks for time series prediction
- **LightGBM Models**: Gradient boosting for structured data
- **PPO Models**: Reinforcement learning for trading decisions

## Configuration

The system uses configuration files in the `config/` directory:
- `config.yaml`: Main trading configuration
- `validation/`: Validation system configuration

### Key Configuration Options
```yaml
model_management:
  models_dir: "models"
  max_age_days: 30

validation:
  config_dir: "./validation"
  drift_detection:
    enabled: true
    threshold: 0.05
  logging:
    level: "INFO"
```

## File Structure

```
Bot_kilo/
├── scripts/
│   └── enhanced_trader.py          # Main trading script
├── src/
│   ├── validation/
│   │   ├── schema_validator.py     # Great Expectations validation
│   │   ├── drift_monitor.py        # Real-time drift monitoring
│   │   ├── advanced_drift_monitor.py # Advanced drift algorithms
│   │   ├── enhanced_logger.py      # Enhanced logging system
│   │   ├── metadata_manager.py     # Metadata lifecycle management
│   │   └── validation_integration.py # Integration layer
│   └── ...
├── test_metadata_hygiene.py        # Test script for metadata hygiene
└── README.md                       # This file
```

## Testing

### Metadata Hygiene Test
Run the metadata hygiene test to verify the system can:
- Regenerate model metadata
- Validate metadata integrity
- Clean up outdated models

```bash
python test_metadata_hygiene.py
```

### Manual Validation
To manually trigger validation processes:

```python
from scripts.enhanced_trader import EnhancedUnifiedPaperTrader

trader = EnhancedUnifiedPaperTrader()
trader.run_metadata_hygiene()  # Run metadata hygiene
report = trader.validation_manager.get_validation_report()  # Get validation report
```

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed
2. **Model Loading Errors**: Check that model files exist in the configured `models_dir`
3. **Validation Failures**: Review logs for specific validation errors and schema mismatches

### Logs
Logs are written to the `logs/` directory with detailed information about:
- Model loading and validation
- Feature drift detection
- Schema validation decisions
- Metadata hygiene processes

## Notes

- **Test Files**: Files with "test" in the name can be safely removed
- **Model Training**: This system is designed for inference only - train models on a separate machine
- **Windows Compatibility**: All scripts are designed to work on Windows systems

This system provides ML runtime initialization and model import functionality for crypto trading bot operations.

## Model Import

### Import Pre-trained Models
```bash
# Use the fixed import script (recommended)
import_models_fixed.bat

# Test the import functionality
python test_import_fixed.py
```

**Note:** The original `import_models.bat` has a colon character parsing error. Use `import_models_fixed.bat` instead.

### Model Transfer Packages
Place your model transfer packages (*.zip files) in the root directory. The import script will:
1. Validate the package structure
2. Import models to the `models/` directory
3. Move processed packages to `processed_packages/`

## Quick Start

### Windows
```bash
# Run training with automatic initialization
python scripts/enhanced_trainer.py

# Or use the shell script
bash train_models.sh
```

### Linux
```bash
# Make script executable
chmod +x train_models_linux.sh

# Run training with automatic initialization
./train_models_linux.sh
```

## Manual Initialization

If you need to initialize the ML runtime manually:

```bash
# Initialize all required directories and files
python startup_init.py

# Check if initialization is needed
python startup_init.py --check

# Verbose initialization
python startup_init.py --verbose
```

## Testing

Run the comprehensive test suite to validate the system:

```bash
# Run all tests
python test_initialization_system.py

# Run tests in quiet mode
python test_initialization_system.py --quiet

# Generate status report only
python test_initialization_system.py --report-only

# Debug specific issues
python test_debug_initialization.py
```

## What Gets Created

The initialization system creates:
- `mlruns/` - MLflow experiment tracking
- `mlruns/0/` - Default experiment (ID 0)
- `logs/` - Training and system logs
- `models/` - Saved model files
- `checkpoints/` - Model checkpoints
  - `checkpoints/gru/` - GRU model checkpoints
  - `checkpoints/lightgbm/` - LightGBM model checkpoints
  - `checkpoints/ppo/` - PPO model checkpoints

## Configuration

Training configuration is located at:
- `src/config/config_training.yaml`

The system automatically detects and uses the appropriate configuration file.

## Troubleshooting

### ML Runtime Issues
If you encounter "Could not find experiment with ID 0" errors:
1. Run `python startup_init.py` to initialize the environment
2. Check that `mlruns/0/meta.yaml` exists
3. Run the test suite to validate the system

### Model Import Issues
If you encounter "was unexpected at this time" error:
1. Use `import_models_fixed.bat` instead of `import_models.bat`
2. Ensure your model transfer package has a valid manifest
3. Run `python test_import_fixed.py` to verify the fix

## Cross-Platform Compatibility

This system works on both Windows and Linux, automatically handling path differences and file creation requirements.