# Enhanced Automated Trading System

A comprehensive automated trading system that executes trades every 30 minutes using machine learning models (GRU, LightGBM, PPO) with robust error handling, detailed logging, and CSV trade reporting.

## Prerequisites and Setup Requirements

### System Requirements
- Windows 10/11 with PowerShell 5.0+
- Python 3.8+ with pip
- Minimum 4GB RAM
- At least 2GB free disk space
- Stable internet connection

### Required Python Packages
The system will automatically install these packages:
- pandas, numpy, pyyaml
- python-binance, ccxt
- python-telegram-bot
- scikit-learn, lightgbm
- torch, stable-baselines3
- mlflow

## Configuration File Setup (training_config.yaml)

### Location
Place `training_config.yaml` in the root directory: `c:\Users\best test\Documents\GitHub\bot\training_config.yaml`

### Required Configuration Structure
```yaml
data_acquisition:
  symbols: ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']
  interval: '1h'
  lookback_days: 30
  data_sources: ['binance']

trading:
  initial_balance: 10000
  position_size: 0.1
  transaction_fee: 0.001
  slippage: 0.0005
  model_weights:
    gru: 0.4
    lightgbm: 0.4
    ppo: 0.2
  risk_management:
    max_position_size: 0.2
    stop_loss: 0.05
    take_profit: 0.1

logging:
  level: 'INFO'
  file: 'logs/trading.log'
```

### Symbol Configuration Requirements
- Each symbol must follow format: `[BASE][QUOTE]` (e.g., BTCEUR, ETHEUR)
- Symbols must be supported by your exchange (Binance by default)
- Each symbol requires corresponding model files

## Model Verification Process

### Model Directory Structure
Ensure the following directory structure exists:
```
models/
├── gru/
│   ├── BTCEUR/
│   ├── ETHEUR/
│   └── [other symbols]/
├── lightgbm/
│   ├── BTCEUR/
│   ├── ETHEUR/
│   └── [other symbols]/
└── ppo/
    ├── BTCEUR/
    ├── ETHEUR/
    └── [other symbols]/
```

### Model Requirements
- Each symbol needs at least 2 out of 3 model types (GRU, LightGBM, PPO)
- Model files must be properly trained and exported
- The system will automatically verify model availability before trading

## Running the Enhanced Deploy Trading Script

### Step 1: Open Command Prompt as Administrator
```cmd
cd "c:\Users\best test\Documents\GitHub\bot"
```

### Step 2: Execute the Enhanced Script
```cmd
deploy_trading.bat
```

### Step 3: Monitor System Startup
The script will:
1. Check system requirements
2. Validate environment and dependencies
3. Process configuration file
4. Verify model availability
5. Initialize trading system
6. Start 30-minute automated trading loop

### Step 4: Confirm System is Running
Look for this output:
```
========================================
  AUTOMATED TRADING SYSTEM ACTIVE
========================================

Configuration:
- Execution Interval: 1800 seconds (30 minutes)
- Trading Symbols: BTCEUR,ETHEUR,ADAEUR
- Model Types: gru lightgbm ppo
- Logs Directory: logs\
- CSV Reports: logs\trades_report.csv
```

## Monitoring and Log File Locations

### Log Files
- **Main Log**: `logs\trading.log` - General system operations
- **Deployment Log**: `logs\deployment.log` - Script execution details
- **Error Log**: `logs\error.log` - Error messages and stack traces

### Real-time Monitoring Commands
```cmd
# Monitor main trading log
tail -f logs\trading.log

# Monitor deployment progress
tail -f logs\deployment.log

# Check for errors
tail -f logs\error.log
```

### Log Rotation
- Logs are automatically rotated when they exceed 10MB
- Backup files are created with `.backup` extension
- Old logs are preserved for troubleshooting

## CSV Trade Report Format and Location

### Location
`logs\trades_report.csv`

### CSV Format
```csv
Timestamp,Symbol,Action,Quantity,Price,Status,Notes,Model,Confidence,PnL
2024-01-15 10:30:00,BTCEUR,BUY,0.1,45000.00,SUCCESS,Automated cycle completed,ENSEMBLE,0.85,+150.50
2024-01-15 11:00:00,ETHEUR,SELL,2.5,2800.00,SUCCESS,Take profit triggered,GRU,0.92,+75.25
2024-01-15 11:30:00,ADAEUR,CYCLE_FAILED,N/A,N/A,ERROR,API connection timeout,N/A,N/A,0.00
```

### CSV Fields Explanation
- **Timestamp**: Execution time (YYYY-MM-DD HH:MM:SS)
- **Symbol**: Trading pair (e.g., BTCEUR)
- **Action**: BUY, SELL, HOLD, CYCLE_COMPLETE, CYCLE_FAILED
- **Quantity**: Trade amount
- **Price**: Execution price
- **Status**: SUCCESS, ERROR, PENDING
- **Notes**: Additional information or error messages
- **Model**: Model used (GRU, LIGHTGBM, PPO, ENSEMBLE)
- **Confidence**: Model confidence score (0.0-1.0)
- **PnL**: Profit/Loss for the trade

### Daily Backups
CSV files are automatically backed up daily with format: `trades_report_YYYYMMDD.csv`

## Troubleshooting Common Issues

### Issue 1: "Configuration file not found"
**Solution**: Ensure `training_config.yaml` exists in the root directory
```cmd
dir training_config.yaml
```

### Issue 2: "No trading symbols found in configuration"
**Solution**: Check YAML syntax and ensure symbols are properly defined
```yaml
data_acquisition:
  symbols: ['BTCEUR', 'ETHEUR']  # Must be a list
```

### Issue 3: "No symbols have complete model sets"
**Solution**: Verify model directory structure and train missing models
```cmd
dir models\gru
dir models\lightgbm
dir models\ppo
```

### Issue 4: "Python dependencies missing"
**Solution**: Manually install required packages
```cmd
pip install pandas numpy pyyaml python-binance ccxt scikit-learn lightgbm torch
```

### Issue 5: "API connection failures"
**Solution**: 
- Check internet connection
- Verify API keys in configuration
- Check exchange status
- Review rate limits

### Issue 6: "Trading script test failed"
**Solution**: Run manual test
```cmd
python enhanced_trader.py --test-mode --config training_config.yaml
```

## Emergency Stop Procedures

### Method 1: Keyboard Interrupt
- Press `Ctrl+C` in the command prompt running the script
- System will complete current cycle and stop gracefully

### Method 2: Close Command Window
- Close the command prompt window
- This will immediately terminate all trading operations

### Method 3: Task Manager
1. Open Task Manager (`Ctrl+Shift+Esc`)
2. Find `cmd.exe` or `python.exe` processes
3. End the trading-related processes

### Method 4: Emergency Stop Script
Create `emergency_stop.bat`:
```batch
@echo off
taskkill /f /im python.exe
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq deploy_trading*"
echo Trading system emergency stopped!
pause
```

## Post-Stop Procedures

### After Emergency Stop
1. Check `logs\trading.log` for last operations
2. Review `logs\trades_report.csv` for pending trades
3. Manually verify exchange account status
4. Check for any open positions that need manual closure

### Safe Restart
1. Wait 5 minutes after stopping
2. Check system logs for any errors
3. Verify configuration files are intact
4. Restart using normal procedure

## System Maintenance

### Daily Tasks
- Review trading logs for errors
- Check CSV trade reports
- Monitor system performance
- Verify model performance

### Weekly Tasks
- Backup configuration files
- Archive old log files
- Update model files if needed
- Review trading performance metrics

### Monthly Tasks
- Retrain models with new data
- Update system dependencies
- Review and optimize configuration
- Analyze trading strategy performance

## Support and Contact

For technical issues:
1. Check log files first
2. Review this README
3. Test individual components
4. Document error messages and system state

## Version Information

- **System Version**: Enhanced Automated Trading v2.0
- **Last Updated**: 2024
- **Compatible Python**: 3.8+
- **Compatible Windows**: 10/11

---

**⚠️ IMPORTANT DISCLAIMER**: This is an automated trading system that involves financial risk. Always test thoroughly with small amounts before deploying with significant capital. Monitor the system regularly and be prepared to intervene manually if needed.