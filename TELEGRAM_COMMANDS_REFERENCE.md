# Telegram Bot Commands Reference

This document provides a comprehensive overview of all available Telegram commands for the Enterprise Crypto Trading Bot System.

## Overview

The system has **two complementary Telegram components**:

1. **Enhanced Telegram Notifier** (`src/notifier/enhanced_telegram.py`) - Advanced trading system controls
2. **Telegram Bot Listener** (`telegram_bot_listener.py`) - Basic bot functionality and logging

**BOTH components must be running simultaneously for full functionality.**

## Enhanced Trading System Commands
*From `src/notifier/enhanced_telegram.py`*

### System Control Commands

#### `/start`
- **Purpose**: Start the entire trading system
- **Action**: Executes `/opt/trading_bot/bot/scripts/tmux_manager.sh start`
- **Response**: "🚀 Trading system started successfully" or error message
- **Requirements**: Proper tmux_manager.sh script and systemd services

#### `/stop`
- **Purpose**: Stop the trading system
- **Action**: Executes `/opt/trading_bot/bot/scripts/tmux_manager.sh stop`
- **Response**: "⏹️ Trading system stopped" or error message
- **Safety**: Gracefully shuts down all trading processes

#### `/restart`
- **Purpose**: Restart the entire trading system
- **Action**: Executes stop then start sequence
- **Response**: "🔄 Trading system restarted successfully" or error message
- **Use Case**: System recovery, configuration updates

### System Monitoring Commands

#### `/status`
- **Purpose**: Get current system operational status
- **Information Provided**:
  - Trading bot status (✅ Running / ❌ Stopped)
  - Server hostname
  - System uptime
  - Last check timestamp
- **Technical**: Checks tmux session `trading_session`

#### `/health`
- **Purpose**: Comprehensive system health check
- **Action**: Executes `/opt/trading_bot/bot/scripts/health_check.sh`
- **Response**: Formatted health report or error details
- **Coverage**: All system components and dependencies

### Trading Performance Commands

#### `/performance`
- **Purpose**: Get detailed trading performance metrics
- **Data Source**: `/opt/trading_bot/bot/logs/performance_metrics.json`
- **Information Provided**:
  - Sharpe ratio and risk-adjusted returns
  - Total returns and profit/loss
  - Win rate and trade statistics
  - Portfolio performance metrics

#### `/balance`
- **Purpose**: Get current account balance and positions
- **Data Source**: `/opt/trading_bot/bot/logs/balance.json`
- **Information Provided**:
  - Cash balance in EUR
  - Portfolio value
  - Total equity
  - Active positions by symbol

#### `/trades`
- **Purpose**: Get recent trading activity
- **Data Source**: `/opt/trading_bot/bot/logs/trades_report.csv`
- **Action**: Shows last 5 trades with details
- **Format**: Symbol, action, quantity, price
- **Use Case**: Monitor recent trading decisions

### System Information Commands

#### `/logs`
- **Purpose**: Get recent system logs
- **Data Source**: `/var/log/trading_bot/trading_*.log`
- **Action**: Shows last 10 log entries
- **Format**: Raw log output in code block
- **Troubleshooting**: Essential for debugging issues

#### `/config`
- **Purpose**: Get current system configuration
- **Data Source**: `/opt/trading_bot/bot/training_config.yaml`
- **Information Shown**:
  - Trading symbols
  - Interval settings
  - Initial balance
  - Maximum position size
- **Security**: Sensitive data (API keys) excluded

## Basic Bot Commands
*From `telegram_bot_listener.py`*

### Testing and Connectivity Commands

#### `/test`
- **Purpose**: Test logging functionality and bot responsiveness
- **Response**: "✅ Logging test successful! Check the logs for details."
- **Logging**: Creates test entry in telegram_listener_*.log
- **Use Case**: Verify bot is receiving and processing commands

#### `/ping`
- **Purpose**: Simple connectivity test
- **Response**: "🏓 Pong! Bot is responding."
- **Logging**: Records ping interaction
- **Use Case**: Quick bot availability check

### Administrative Commands

#### `/logs` (Basic Version)
- **Purpose**: Request system logs (basic implementation)
- **Response**: "📋 Logs requested. Check system for details."
- **Note**: Different from enhanced version, primarily for testing
- **Logging**: Records log request in bot listener logs

## Command Path Configuration

### File System Structure
The commands reference these critical paths on the production server:

```
/opt/trading_bot/bot/
├── scripts/
│   ├── tmux_manager.sh         # System control
│   └── health_check.sh         # Health monitoring
├── logs/
│   ├── performance_metrics.json # Performance data
│   ├── balance.json           # Account balances
│   └── trades_report.csv      # Trading history
├── training_config.yaml       # System configuration
└── src/notifier/enhanced_telegram.py # Command handlers
```

### Log File Locations
- **Enhanced Commands**: `/var/log/trading_bot/trading_*.log`
- **Bot Listener**: `logs/telegram_listener_YYYYMMDD_HHMMSS.log`
- **Trading Activity**: `/opt/trading_bot/bot/logs/trades_report.csv`

## Deployment Requirements

### Server Setup
1. **Ubuntu Hetzner Server** with proper directory structure
2. **Tmux Manager Script** at `/opt/trading_bot/bot/scripts/tmux_manager.sh`
3. **Health Check Script** at `/opt/trading_bot/bot/scripts/health_check.sh`
4. **Log Directories** with proper permissions
5. **Configuration Files** with valid API tokens

### Environment Variables
```bash
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_telegram_chat_id"
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET_KEY="your_binance_secret_key"
```

### Service Dependencies
- **Systemd Services**: trading-bot.service for auto-start
- **Tmux Sessions**: For persistent trading processes
- **Python Environment**: With all required packages installed

## Security and Access Control

### Authentication
- Commands are restricted to configured `TELEGRAM_CHAT_ID`
- No public access or unauthorized command execution
- Error messages don't expose sensitive system information

### Data Protection
- API keys and secrets not shown in `/config` command
- Log outputs sanitized of sensitive information
- Balance and trading data only for authorized users

## Troubleshooting Guide

### Common Issues

#### Commands Not Responding
**Symptoms**: No response to any commands
**Cause**: `telegram_bot_listener.py` not running
**Solution**: Start the listener service
```bash
python3 telegram_bot_listener.py &
```

#### Advanced Commands Failing
**Symptoms**: Basic commands work, but `/status`, `/balance` fail
**Cause**: `EnhancedTelegramNotifier` not integrated or file paths incorrect
**Solution**: Verify paths and restart enhanced trader
```bash
python3 scripts/enhanced_trader.py &
```

#### File Not Found Errors
**Symptoms**: Commands return "file not found" errors
**Cause**: Incorrect paths or missing scripts
**Solution**: Verify all paths match actual file locations

### Diagnostic Commands
1. **Test Basic Connectivity**: `/ping`
2. **Test Logging**: `/test`
3. **Check System Status**: `/status`
4. **Verify Health**: `/health`

## Best Practices

### Regular Monitoring
- Use `/status` for daily operational checks
- Review `/performance` for trading analysis
- Monitor `/logs` for error detection
- Check `/balance` for portfolio tracking

### Emergency Procedures
1. **System Issues**: Use `/stop` to halt trading
2. **Performance Problems**: Check `/performance` and `/logs`
3. **Recovery**: Use `/restart` after resolving issues
4. **Manual Intervention**: SSH to server if Telegram fails

### Operational Workflow
1. **Morning Check**: `/status`, `/balance`, `/performance`
2. **During Trading**: Monitor `/trades` for activity
3. **Troubleshooting**: `/health`, `/logs` for diagnostics
4. **End of Day**: `/performance` for daily summary

## Integration Notes

### For Developers
- Commands are async functions in `EnhancedTelegramNotifier` class
- Error handling includes both technical errors and user-friendly messages
- Logging occurs at multiple levels (command receipt, execution, results)
- Path resolution uses absolute paths for production deployment

### For System Administrators
- All file paths assume `/opt/trading_bot/bot/` as root directory
- Log rotation should be configured for `/var/log/trading_bot/`
- Permissions must allow script execution and file access
- Backup procedures should include configuration and log files

## Command Summary

| Command | Source | Purpose | Response Type |
|---------|--------|---------|---------------|
| `/start` | Enhanced | Start trading system | System control |
| `/stop` | Enhanced | Stop trading system | System control |
| `/restart` | Enhanced | Restart system | System control |
| `/status` | Enhanced | System status | Monitoring |
| `/health` | Enhanced | Health check | Monitoring |
| `/performance` | Enhanced | Trading metrics | Analytics |
| `/balance` | Enhanced | Account balance | Financial |
| `/trades` | Enhanced | Recent trades | Financial |
| `/logs` | Enhanced | System logs | Diagnostics |
| `/config` | Enhanced | Configuration | Information |
| `/test` | Basic | Test logging | Diagnostics |
| `/ping` | Basic | Connectivity test | Diagnostics |

**Total Commands Available**: 12 commands across 2 components

---

*This reference is based on the current implementation as of the latest system analysis. Commands and functionality may be extended in future versions.*
