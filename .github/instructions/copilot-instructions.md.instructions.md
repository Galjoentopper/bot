---
applyTo: '**'
---
# Enterprise Crypto Trading Bot System

**ALWAYS follow these instructions first and update them with new information located in .kilocode\rules\memory-bank before proceeding with any development tasks.**

This is an enterprise-grade automated cryptocurrency trading system that maximizes Sharpe ratio through sophisticated machine learning strategies while maintaining operational stability on Ubuntu Hetzner server. The system uses multi-model ML ensemble (GRU neural networks, LightGBM, PPO reinforcement learning) for cryptocurrency trading with comprehensive Telegram-based monitoring and control.

## Project Overview and Architecture

### Core Mission
- **Maximize Sharpe Ratio**: Optimize risk-adjusted returns with target >1.5
- **Operational Stability**: Achieve 99%+ uptime with autonomous operation on Ubuntu Hetzner server
- **Risk Control**: Maintain strict drawdown limits (<5% maximum)
- **Enterprise Management**: Comprehensive Telegram-based monitoring, alerts, and remote control

### System Architecture
- **Multi-Service Design**: Independent but coordinated services for trading, monitoring, and communication
- **Production Environment**: Ubuntu 20.04+ server with systemd services, tmux sessions, automated health checks
- **Telegram Integration**: Dual-component system with outgoing notifications and incoming command processing
- **Real-time Processing**: 200+ technical indicators from Binance API (30-minute candles)

### Trading Scope and Risk Management
- **Symbols**: BTCEUR, ETHEUR, ADAEUR, DOTEUR, LINKEUR
- **Mode**: Paper trading with realistic fees and slippage simulation
- **Timeframe**: 30-minute candles optimized for data freshness
- **Risk Management**: Kelly criterion sizing, trailing stops, correlation analysis, maximum 25% portfolio per position

## Working Effectively

### First Priority: Update Instructions
Before any development work:
```bash
# Check for updates in memory bank
ls -la .kilocode/rules/memory-bank/
cat .kilocode/rules/memory-bank/*.md

# Update these instructions with any new information from memory bank files:
# - architecture.md: System design and component relationships
# - brief.md: Project overview and objectives
# - context.md: Development context and constraints
# - product.md: Product specifications and features
# - tasks.md: Current development tasks and priorities
# - tech.md: Technical stack and dependencies
```

### Bootstrap and Dependencies
Run these commands to set up the development environment:

```bash
# Install Python package and dependencies - NEVER CANCEL: Takes 3 minutes
cd /path/to/bot
pip3 install -e . --timeout 300

# Install enterprise dependencies for production environment
pip3 install python-telegram-bot ccxt pandas numpy scikit-learn lightgbm torch stable-baselines3
```

**TIMEOUT REQUIREMENT**: Always set timeout to 300+ seconds (5+ minutes) for package installation.

### Core Workflow Commands

#### 1. Data Collection and Management
```bash
# Fetch training data for all symbols - Takes 4 seconds
chmod +x fetch_training_data.sh
./fetch_training_data.sh

# Fetch data for specific symbol only
./fetch_training_data.sh --symbol BTCEUR

# Validate data quality and completeness
python3 -c "
from src.data.data_manager import DataManager
dm = DataManager()
dm.validate_data_quality()
"
```

#### 2. Model Training (Enterprise ML Pipeline)
```bash
# Enterprise training command (RECOMMENDED) - NEVER CANCEL: Takes 10+ minutes for full ensemble
python3 scripts/enhanced_trainer.py --config training_config.yaml --models gru lightgbm ppo --symbols BTCEUR ETHEUR ADAEUR DOTEUR LINKEUR --verbose --n-splits 5

# Single model training for development - Takes 15 seconds
python3 scripts/enhanced_trainer.py --config training_config.yaml --models lightgbm --symbols BTCEUR --verbose --n-splits 2

# Multi-model ensemble training - NEVER CANCEL: Takes 9.5 minutes
python3 scripts/enhanced_trainer.py --config training_config.yaml --models gru lightgbm ppo --symbols BTCEUR --verbose --n-splits 2

# Full production training - NEVER CANCEL: Takes 30+ minutes
python3 scripts/enhanced_trainer.py --config training_config.yaml --models gru lightgbm ppo --symbols BTCEUR ETHEUR ADAEUR DOTEUR LINKEUR --verbose --n-splits 5
```

**CRITICAL TIMING**: 
- Single model training: 15 seconds - set timeout to 60+ seconds
- Multi-model training: 9.5 minutes - set timeout to 15+ minutes  
- Full ensemble training: 30+ minutes - set timeout to 45+ minutes
- **NEVER CANCEL** these training operations

#### 3. Enterprise Trading System Execution
```bash
# Start complete trading system with tmux session management
./scripts/enhanced_tmux_manager.sh start

# Start individual components for debugging
python3 scripts/enhanced_trader.py &
python3 telegram_bot_listener.py &

# Check system status
./scripts/enhanced_tmux_manager.sh status

# Production deployment with systemd
sudo systemctl start trading-bot
sudo systemctl enable trading-bot
```

#### 4. Telegram Bot Management
```bash
# Test Telegram bot connection
python3 -c "
import asyncio
from debug_telegram import test_telegram_connection
asyncio.run(test_telegram_connection())
"

# Start Telegram command listener
python3 telegram_bot_listener.py

# Diagnose Telegram issues
python3 diagnose_telegram.py

# Test Telegram notifications
python3 -c "
from src.notifier.enhanced_telegram import EnhancedTelegramNotifier
notifier = EnhancedTelegramNotifier()
notifier.send_startup_notification()
"
```

## Validation and Testing

### Enterprise Validation Steps
After making changes, ALWAYS run these validation steps:

```bash
# 1. Validate package installation works
pip3 install -e . --timeout 300

# 2. Validate data collection works
./fetch_training_data.sh --symbol BTCEUR

# 3. Validate ML ensemble training works
python3 scripts/enhanced_trainer.py --config training_config.yaml --models lightgbm --symbols BTCEUR --verbose --n-splits 2

# 4. Validate backtesting system works
python3 -c "
from src.backtesting.backtest import Backtester
bt = Backtester(initial_capital=10000, transaction_fee=0.001, slippage=0.0005)
print('Enterprise backtesting validation: PASS')
"

# 5. Validate Telegram integration works
python3 diagnose_telegram.py

# 6. Validate risk management system
python3 -c "
from src.risk.risk_manager import RiskManager
rm = RiskManager()
rm.validate_risk_parameters()
print('Risk management validation: PASS')
"
```

### End-to-End Enterprise Validation Scenarios
Test these complete workflows after making changes:

1. **Complete ML Pipeline**: Data fetch → Ensemble training → Model validation → Performance metrics
2. **Trading System**: Load models → Risk validation → Portfolio management → Telegram notifications
3. **Operational Pipeline**: Systemd service → Tmux sessions → Health checks → Alert system
4. **Configuration Management**: Modify training_config.yaml → Retrain models → Deploy → Monitor

## System Architecture and Components

### Core Services Architecture
```
Trading Bot Ecosystem:
├── Enhanced Trader (scripts/enhanced_trader.py)     # Main trading engine
├── Telegram Bot Listener (telegram_bot_listener.py) # Command processing
├── Enhanced Telegram Notifier (src/notifier/)       # Outgoing notifications  
├── Risk Manager (src/risk/)                         # Risk control system
├── Data Manager (src/data/)                         # Data pipeline
├── ML Models (src/models/)                          # Ensemble models
└── Backtesting Engine (src/backtesting/)            # Performance validation
```

### Telegram Integration Architecture
The system has **dual Telegram components**:
1. **Outgoing Notifications** (`src/notifier/enhanced_telegram.py`) - System alerts and status updates
2. **Incoming Commands** (`telegram_bot_listener.py`) - User command processing

**Both components must run simultaneously for full functionality.**

### Production Environment Setup
```bash
# Ubuntu Hetzner server requirements
- Ubuntu 20.04+ with systemd
- Python 3.8+ with enterprise ML stack
- Tmux for session management
- Systemd services for auto-start
- Cron jobs for health checks
- Environment variables for API keys
```

## Known Issues and Enterprise Solutions

### Telegram Bot Command Issues
**Issue**: Commands not working despite receiving notifications
**Root Cause**: `telegram_bot_listener.py` service not running
**Solution**: 
```bash
# Start both services
python3 scripts/enhanced_trader.py &
python3 telegram_bot_listener.py &

# Or use enterprise tmux manager
./scripts/enhanced_tmux_manager.sh start
```

### train_models_linux.sh Limitation
- The shell wrapper script `train_models_linux.sh` fails due to missing `startup_init` module
- **ENTERPRISE SOLUTION**: Use `python3 scripts/enhanced_trainer.py` directly for all training operations
- This provides better error handling and enterprise logging

### API Access and Geolocation
- Binance API access may be restricted in some environments
- Models train successfully using cached data in `data/` directory
- Production deployment on Ubuntu Hetzner server resolves API restrictions

### Enterprise Monitoring and Alerting
- Comprehensive health checks via Telegram bot commands
- Performance monitoring with drift detection
- Automated failover and recovery procedures
- 24/7 operational status monitoring

## Configuration Management

### Enterprise Configuration Files
- `training_config.yaml`: Centralized ML and trading configuration
- `.env` or environment variables: API keys and sensitive data
- `systemd/trading-bot.service`: Production service configuration
- `requirements.txt`: Python dependencies with version pinning

### Critical Environment Variables
```bash
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET_KEY="your_binance_secret_key"
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_telegram_chat_id"
```

### Important Enterprise Directories
- `src/`: Core Python modules and enterprise trading logic
- `scripts/`: Training and trading entry points with enterprise features
- `models/`: Trained ML ensemble models (auto-created)
- `data/`: Market data databases with validation (auto-created)
- `logs/`: Comprehensive system logs with rotation (auto-created)
- `.kilocode/rules/memory-bank/`: Project documentation and specifications

## Cross-Platform Enterprise Deployment

This system is designed for:
- **Development Environment**: Cross-platform development and testing
- **Production Environment**: Ubuntu Hetzner server for live trading
- **Monitoring Environment**: Telegram-based remote monitoring and control

**Critical Constraint**: All live trading operations must execute exclusively on the remote Ubuntu Hetzner server.

## Enterprise Trading Features

### Machine Learning Ensemble
- **GRU Neural Networks**: Deep learning for time series prediction with LSTM architecture
- **LightGBM**: Gradient boosting for structured feature analysis
- **PPO Reinforcement Learning**: Proximal Policy Optimization for adaptive strategy learning

### Advanced Risk Management
- **Kelly Criterion**: Optimal position sizing based on win probability and expected return
- **Trailing Stops**: Dynamic stop-loss orders that follow favorable price movements
- **Correlation Analysis**: Portfolio diversification through correlation matrix analysis
- **Drawdown Control**: Maximum 5% portfolio drawdown with automatic position reduction

### Performance Analytics
- **Sharpe Ratio Optimization**: Target >1.5 with continuous monitoring
- **Performance Attribution**: Detailed analysis of return sources
- **Model Drift Detection**: Automatic retraining triggers when model performance degrades
- **Risk-Adjusted Metrics**: Sortino ratio, maximum drawdown, Value at Risk (VaR)

## Enterprise Operational Procedures

### Daily Operations
```bash
# Morning system check
./scripts/enhanced_tmux_manager.sh status
python3 diagnose_telegram.py

# Check overnight performance
/status command via Telegram
/performance command via Telegram

# Validate system health
/health command via Telegram
```

### Weekly Maintenance
```bash
# Model retraining (if drift detected)
python3 scripts/enhanced_trainer.py --config training_config.yaml --models gru lightgbm ppo --symbols all

# Performance review
/stats command via Telegram

# System backup and log rotation
./scripts/backup_system.sh
```

### Emergency Procedures
```bash
# Emergency stop
/emergency_stop command via Telegram

# System restart
sudo systemctl restart trading-bot

# Manual intervention
./scripts/enhanced_tmux_manager.sh attach
```

## Common Enterprise Tasks Reference

### Repository Structure
```
/path/to/bot/
├── .kilocode/rules/memory-bank/    # Project specifications (READ FIRST)
├── fetch_training_data.sh          # Data acquisition (4 seconds)
├── train_models_linux.sh           # Training wrapper (deprecated)
├── scripts/enhanced_trainer.py     # Enterprise ML training (30+ min)
├── scripts/enhanced_trader.py      # Main trading engine
├── telegram_bot_listener.py        # Telegram command processor
├── training_config.yaml            # Centralized configuration
├── requirements.txt                # Enterprise dependencies
├── setup.py                       # Package installation
├── src/                           # Core enterprise modules
├── data/                          # Market data (auto-created)
├── models/                        # ML ensemble models (auto-created)
├── logs/                          # Enterprise logging (auto-created)
└── systemd/                       # Production service files
```

### Telegram Bot Commands
Available commands for remote management:
- `/start` - Initialize bot and show commands
- `/status` - System operational status
- `/health` - Comprehensive health check
- `/performance` - Trading performance metrics
- `/positions` - Current portfolio positions
- `/balance` - Account balance and P&L
- `/stats` - Detailed performance statistics
- `/emergency_stop` - Emergency trading halt
- `/restart` - System restart procedures

## Critical Enterprise Reminders

1. **ALWAYS UPDATE INSTRUCTIONS**: Check `.kilocode/rules/memory-bank/` for new information before any development
2. **NEVER CANCEL** long-running ML training operations - enterprise models may take 30+ minutes
3. **ALWAYS** set appropriate timeouts (45+ minutes for full ensemble training)
4. **USE** `enhanced_trainer.py` directly for all enterprise ML training operations
5. **VALIDATE** changes with enterprise validation steps including risk management
6. **ENSURE** both Telegram services are running for complete functionality
7. **MONITOR** Sharpe ratio and maintain target >1.5 with <5% maximum drawdown
8. **DEPLOY** exclusively on Ubuntu Hetzner server for live trading operations
9. **MAINTAIN** 99%+ uptime through comprehensive monitoring and automated recovery
10. **SECURE** all API keys and sensitive configuration in environment variables

## Enterprise Success Metrics
- **Sharpe Ratio**: Target >1.5 (risk-adjusted returns)
- **Maximum Drawdown**: <5% strict limit
- **System Uptime**: 99%+ operational availability
- **Response Time**: <30 seconds for Telegram commands
- **Model Performance**: Monthly validation with drift detection
- **Risk Compliance**: 100% adherence to position sizing and correlation limits