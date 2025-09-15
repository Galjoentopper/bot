# Deploy Trader Script Documentation

## Overview

The `deploy_trader.bat` script provides flexible deployment of the trading bot with automatic symbol and model discovery from the configuration file. It reads `training_config.yaml`, extracts the symbols and models, verifies that the required models exist, and starts the trader with the appropriate parameters.

## Features

- **Flexible Symbol Selection**: Automatically reads symbols from `training_config.yaml`
- **Model Discovery**: Detects which models are available for each symbol
- **Model Verification**: Ensures required models exist before deployment
- **Multiple Deployment Modes**: Single cycle, continuous trading, or test mode
- **Enhanced Logging**: Comprehensive logging of deployment process
- **Error Handling**: Graceful handling of missing models or configuration issues

## Usage

### Basic Usage
```cmd
deploy_trader.bat
```

The script will:
1. Read symbols from `training_config.yaml`
2. Discover available models for each symbol
3. Verify model availability
4. Present deployment options
5. Start the trader with verified symbols and models

### Configuration File Structure

The script reads symbols and models from `training_config.yaml`:

```yaml
data_acquisition:
  symbols: ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']

training:
  models: ['gru', 'lightgbm', 'ppo']
```

Alternative configurations supported:
```yaml
# Alternative 1: Under 'data' section
data:
  symbols: ['BTCEUR', 'ETHEUR']

# Alternative 2: Root level
symbols: ['BTCEUR', 'ETHEUR']
```

### Enhanced Trader Script Arguments

The enhanced trader now supports flexible command-line arguments:

```cmd
python scripts/enhanced_trader.py --help

Options:
  --config CONFIG           Path to configuration file (default: auto-detect)
  --models-dir MODELS_DIR   Path to models directory (default: models)
  --symbols SYMBOLS [...]   Trading symbols to use (default: from config)
  --models MODELS [...]     Model types to use (default: from config)
  --test-mode              Run in test mode (validate configuration and models)
  --single-cycle           Run a single trading cycle instead of continuous loop
```

### Examples

#### Test specific symbols and models:
```cmd
python scripts/enhanced_trader.py --config training_config.yaml --symbols BTCEUR ETHEUR --models lightgbm --test-mode
```

#### Run single cycle with all config symbols:
```cmd
python scripts/enhanced_trader.py --config training_config.yaml --single-cycle
```

#### Continuous trading with specific setup:
```cmd
python scripts/enhanced_trader.py --config training_config.yaml --symbols BTCEUR --models gru lightgbm ppo
```

## Deployment Modes

The script offers three deployment modes:

### 1. Single Cycle Mode
- Runs one trading cycle for each verified symbol
- Useful for testing and validation
- Exits after completion

### 2. Continuous Trading Mode
- Runs indefinitely with the configured symbols and models
- Monitors markets continuously
- Requires manual stop (Ctrl+C)

### 3. Test Mode Only
- Validates configuration and model availability
- Does not perform actual trading
- Useful for deployment verification

## Model Verification Logic

The script verifies models using the following logic:

1. **Symbol Discovery**: Extracts symbols from configuration
2. **Model Availability Check**: For each symbol, checks if model directories exist:
   ```
   models/
   ├── gru/BTCEUR/
   ├── lightgbm/BTCEUR/
   └── ppo/BTCEUR/
   ```
3. **Minimum Requirements**: A symbol needs at least one available model to be included
4. **Model Aggregation**: Builds final list of available models across all verified symbols

## Error Handling

The script handles various error conditions:

- **Missing Configuration**: Prompts user to ensure `training_config.yaml` exists
- **Invalid YAML**: Reports parsing errors with suggestions
- **No Symbols Found**: Provides example configuration format
- **No Models Available**: Suggests running training first
- **Script Not Found**: Verifies trader script location

## Logging

Comprehensive logging is provided:

- **Deployment Log**: `logs/deployment.log` - Full deployment process
- **Error Log**: `logs/error.log` - Error-specific information
- **Console Output**: Real-time status updates

## Integration with Existing System

This script integrates seamlessly with the existing trading system:

- **Compatible with existing training scripts**
- **Works with current model directory structure**
- **Supports all existing configuration formats**
- **Maintains backward compatibility**

## Best Practices

1. **Always run test mode first** to validate setup
2. **Ensure models are trained** before deployment
3. **Monitor logs** for deployment issues
4. **Use single cycle mode** for initial validation
5. **Keep configuration file updated** with desired symbols

## Troubleshooting

### Common Issues

1. **"No trading symbols found in configuration"**
   - Ensure symbols are properly defined in YAML
   - Check indentation and list format

2. **"No symbols have complete model sets"**
   - Run training for missing symbols
   - Verify model directory structure

3. **"Trader configuration test failed"**
   - Check enhanced_trader.py exists
   - Verify Python dependencies installed

### Verification Commands

Test configuration extraction:
```cmd
python test_symbol_extraction.py training_config.yaml
```

Test enhanced trader:
```cmd
python scripts/enhanced_trader.py --test-mode
```

Validate model structure:
```cmd
dir models\gru
dir models\lightgbm
dir models\ppo
```
