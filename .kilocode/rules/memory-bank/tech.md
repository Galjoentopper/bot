# Technology Stack & Operations Guide

## Core Technologies

### **Python Runtime Environment**
- **Version**: Python 3.8+ (required for modern ML libraries)
- **Installation**: [`setup.py`](setup.py:7) - Bot package with editable install
- **Dependencies**: [`requirements.txt`](requirements.txt) - 100+ packages managed

### **Machine Learning Framework**
- **PyTorch**: [`torch>=2.0.0`](requirements.txt:12) - GRU neural network training/inference
- **LightGBM**: [`lightgbm>=4.0.0`](requirements.txt:15) - Gradient boosting ensemble models
- **Stable Baselines3**: [`stable-baselines3>=2.7.0`](requirements.txt:20) - PPO reinforcement learning
- **Scikit-learn**: [`scikit-learn>=1.3.0`](requirements.txt:16) - Data preprocessing, feature selection

### **Data Processing Stack**
- **Pandas**: [`pandas>=2.0.0`](requirements.txt:6) - Primary data manipulation
- **NumPy**: [`numpy>=1.24.0`](requirements.txt:7) - Numerical computations
- **Technical Analysis**: [`ta>=0.10.2`](requirements.txt:65), [`pandas-ta>=0.3.14b0`](requirements.txt:66)
- **Financial Data**: [`python-binance>=1.0.0`](requirements.txt:32), [`ccxt>=4.0.0`](requirements.txt:33)

## Production Infrastructure

### **Operating System: Ubuntu Hetzner Server**
- **OS**: Ubuntu 20.04+ LTS
- **Access**: SSH with key-based authentication only
- **Security**: UFW firewall, Fail2ban intrusion prevention
- **User**: `trader` user (non-root operations)

### **Service Management**
- **Primary Service**: [`systemd`](server/systemd/trading-bot.service)
  - **Control**: `sudo systemctl [start|stop|status|restart] trading-bot`
  - **Logs**: `sudo journalctl -u trading-bot -f`
  - **Auto-start**: Enabled on boot

### **Process Management**
- **Session Manager**: [`tmux`](server/scripts/tmux_manager.sh)
  - **Attach**: `/opt/trading_bot/scripts/tmux_manager.sh attach`
  - **Status**: `/opt/trading_bot/scripts/tmux_manager.sh status`
  - **Logs**: `/opt/trading_bot/scripts/tmux_manager.sh logs`

### **Scheduled Tasks**
- **Cron Jobs**: [`/etc/cron.d/trading_bot_monitor`](server/cron/trading_bot_monitor)
  - **Schedule**: Every 5 minutes
  - **Health Check**: [`scripts/health_check.sh`](server/scripts/health_check.sh)
  - **Auto-restart**: Automatic service recovery

## Deployment & Configuration

### **Environment Setup**
- **Config File**: [`.env`](.env.example) - Environment variables
  - **Location**: `/etc/trading_bot/.env` (production)
  - **Template**: [`.env.example`](.env.example) (local reference)
  - **Critical Values**: Telegram tokens, API keys, logging paths

### **Configuration Management**
- **Primary Config**: [`training_config.yaml`](training_config.yaml)
  - **Purpose**: Central configuration for all system components
  - **Location**: Project root (copied to server)
  - **Key Sections**: Data acquisition, trading parameters, model weights

- **Config Loader**: [`src/config/config_loader.py`](src/config/config_loader.py:24)
  - **Auto-detection**: Detects trader vs training scripts
  - **Fallback Chain**: Specific config → `config.yaml` → defaults

### **Model Management**
- **Model Directory**: `./models/` (configurable via config)
- **Storage Format**: 
  - GRU: `.pth` files (PyTorch)
  - LightGBM: `.pkl` files (Python pickle)
  - PPO: `.zip` files (Stable-baselines3)
- **Loading Strategy**: [`_load_model_with_fallbacks()`](scripts/enhanced_trader.py:531)

## Monitoring & Observability

### **Application Logging**
- **Framework**: Custom [`src/utils/logger.py`](src/utils/logger.py)
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Files**: [`logs/trading.log`](training_config.yaml:191), [`logs/performance_metrics.json`](training_config.yaml:191)
- **Rotation**: [`scripts/rotate_logs.sh`](scripts/rotate_logs.sh) - automatic cleanup

### **System Monitoring**
- **Health Checks**: [`scripts/health_check.sh`](scripts/health_check.sh)
  - **Checks**: Process status, memory usage, disk space, log errors
  - **Frequency**: Every 5 minutes via cron
  - **Alerts**: Telegram notifications on failures

### **Performance Tracking**
- **Metrics**: [`src/trading/trading_metrics.py`](src/trading/trading_metrics.py)
- **Analytics**: [`src/trading/performance_analytics.py`](src/trading/performance_analytics.py)
- **Reports**: [`scripts/generate_performance_report.sh`](scripts/generate_performance_report.sh)

### **Notification System**
- **Telegram Bot**: [`src/notifier/telegram.py`](src/notifier/telegram.py:5)
  - **Library**: [`python-telegram-bot>=20.0`](requirements.txt:37)
  - **Features**: Trade alerts, system status, error notifications
  - **Commands**: `/status`, `/start`, `/stop`, `/performance`, `/health`

## Development & Testing

### **Development Setup**
- **Package Install**: `pip install -e .` (editable install)
- **Dependencies**: `pip install -r requirements.txt`
- **Testing**: [`pytest>=7.4.0`](requirements.txt:85) with async support

### **Testing Scripts**
- **Main Tests**: [`test_trading_system.py`](test_trading_system.py)
- **Enhanced Tests**: [`comprehensive_test_system.py`](comprehensive_test_system.py)
- **Quick Tests**: [`quick_test_system.py`](quick_test_system.py)
- **Validation**: [`validate_fixes.py`](validate_fixes.py)

### **Development Tools**
- **Code Quality**: [`black>=23.0.0`](requirements.txt:88) (formatting), [`flake8>=6.0.0`](requirements.txt:89) (linting)
- **Notebooks**: [`jupyter>=1.0.0`](requirements.txt:92) for analysis
- **Debugging**: Rich logging with [`rich>=13.4.0`](requirements.txt:50)

## Operational Commands

### **System Control**
```bash
# Service management
sudo systemctl start trading-bot
sudo systemctl stop trading-bot
sudo systemctl status trading-bot
sudo systemctl restart trading-bot

# Tmux session management
/opt/trading_bot/scripts/tmux_manager.sh status
/opt/trading_bot/scripts/tmux_manager.sh attach
/opt/trading_bot/scripts/tmux_manager.sh logs
```

### **Health Monitoring**
```bash
# Manual health check
/opt/trading_bot/scripts/health_check.sh

# System logs
sudo journalctl -u trading-bot -f
tail -f /var/log/trading_bot/trading.log

# Performance metrics
cat /opt/trading_bot/logs/performance_metrics.json
```

### **Configuration Updates**
```bash
# Backup current config
/opt/trading_bot/scripts/backup_config.sh

# Deploy new configuration
/opt/trading_bot/scripts/deploy_trading.sh

# Restart with new config
sudo systemctl restart trading-bot
```

## Deployment Procedures

### **Initial Deployment**
1. **Upload Code**: `scp -r . trader@server:/opt/trading_bot/`
2. **Set Permissions**: `chmod +x scripts/*.sh`
3. **Install Service**: `sudo cp systemd/trading-bot.service /etc/systemd/system/`
4. **Enable Service**: `sudo systemctl enable trading-bot`
5. **Install Cron**: `sudo cp cron/trading_bot_monitor /etc/cron.d/`

### **Configuration Deployment**
1. **Backup Existing**: [`scripts/backup_config.sh`](scripts/backup_config.sh)
2. **Update Files**: Copy new [`training_config.yaml`](training_config.yaml), [`.env`](.env.example)
3. **Validate Config**: [`validate_fixes.py`](validate_fixes.py)
4. **Restart Service**: `sudo systemctl restart trading-bot`

### **Model Updates**
1. **Upload Models**: `scp models/* trader@server:/opt/trading_bot/models/`
2. **Verify Structure**: Check model directory organization
3. **Test Loading**: [`run_enhanced_trader_test.py`](scripts/run_enhanced_trader_test.py)
4. **Deploy**: Restart trading service

## Data Storage & Management

### **Data Directories**
- **Market Data**: [`./data/`](data/) - Historical OHLCV data
- **Models**: [`./models/`](model_packages/) - Trained ML models
- **Logs**: [`./logs/`](training_config.yaml:191) - Application and system logs
- **Cache**: [`./cache/`](cache/) - Temporary data and feature cache

### **Database Dependencies**
- **SQLite**: [`aiosqlite>=0.19.0`](requirements.txt:26) - Local data storage
- **Parquet**: [`pyarrow>=12.0.0`](requirements.txt:9) - Efficient data storage format

## Security & Access Control

### **Authentication**
- **SSH Keys**: Only key-based authentication enabled
- **API Keys**: Stored in environment variables, never in code
- **Permissions**: `trader` user for all operations (no root required)

### **Firewall & Protection**
- **UFW**: Ubuntu firewall configuration
- **Fail2ban**: Automatic IP blocking for intrusion attempts
- **Log Monitoring**: Automated alerting for security events

## Error Handling & Recovery

### **Circuit Breaker Pattern**
- **Implementation**: [`src/core/circuit_breaker.py`](src/core/circuit_breaker.py)
- **Purpose**: Prevent cascade failures during API outages
- **Triggers**: Multiple consecutive API failures, timeout thresholds

### **Retry Mechanisms**
- **API Calls**: [`_fetch_with_retry()`](scripts/trader.py:464) - Exponential backoff
- **Model Loading**: [`_load_model_with_fallbacks()`](scripts/enhanced_trader.py:531) - Multiple sources
- **Data Processing**: Graceful degradation with fallback strategies

### **Auto Recovery**
- **Systemd**: Automatic service restart on failure
- **Health Checks**: Proactive detection and restart via cron
- **Resource Management**: Memory limits and cleanup procedures

## Performance Optimization

### **Caching Strategy**
- **Market Data**: 60-second cache for API calls
- **Model Loading**: Lazy loading with in-memory caching
- **Feature Generation**: Sliding window approach for efficiency

### **Concurrency**
- **Async Processing**: [`asyncio`](scripts/enhanced_trader.py:912) for parallel data fetching
- **Thread Safety**: Model predictions in isolated threads
- **Resource Pooling**: Connection reuse for external APIs

## Backup & Disaster Recovery

### **Automated Backups**
- **Configuration**: [`scripts/backup_config.sh`](scripts/backup_config.sh) - Daily backups
- **Log Rotation**: [`scripts/rotate_logs.sh`](scripts/rotate_logs.sh) - Prevent disk overflow
- **Model Versioning**: Automatic model artifact preservation

### **Manual Backup Procedures**
- **Full System**: Archive `/opt/trading_bot/` directory
- **Critical Files**: [`training_config.yaml`](training_config.yaml), [`.env`](.env.example), model files
- **Database**: SQLite database files in [`./data/`](data/)

## External Integrations

### **Market Data Sources**
- **Primary**: Binance API via [`ccxt`](requirements.txt:33)
- **Fallback**: Yahoo Finance via [`yfinance>=0.2.18`](requirements.txt:34)
- **Rate Limiting**: Built-in API rate limit handling

### **Notification Services**
- **Telegram**: Real-time alerts and control commands
- **MLFlow**: [`mlflow>=2.5.0`](requirements.txt:40) - Experiment tracking
- **Wandb**: [`wandb>=0.15.0`](requirements.txt:41) - Performance monitoring

## Troubleshooting Tools

### **Diagnostic Scripts**
- **System Test**: [`test_trading_system.py`](test_trading_system.py)
- **Telegram Test**: [`test_telegram.py`](test_telegram.py)
- **Model Validation**: [`validate_fixes.py`](validate_fixes.py)
- **Health Check**: [`scripts/health_check.sh`](scripts/health_check.sh)

### **Log Analysis**
- **Trading Logs**: `/var/log/trading_bot/trading.log`
- **System Logs**: `sudo journalctl -u trading-bot`
- **Error Logs**: [`error.log`](error.log), [`temp_error.log`](temp_error.log)
- **Performance**: [`trades_report.csv`](trades_report.csv)

### **Debug Utilities**
- **Enhanced Logging**: [`src/utils/logger.py`](src/utils/logger.py) with context
- **Validation System**: [`src/validation/`](src/validation/) - Schema and drift monitoring
- **Memory Profiling**: [`psutil>=5.9.0`](requirements.txt:27) - Resource monitoring

## Constraints & Limitations

### **Resource Constraints**
- **Memory**: 16GB limit configured in [`training_config.yaml`](training_config.yaml:38)
- **CPU**: Multi-core processing with [`max_workers: 8`](training_config.yaml:28)
- **Disk**: Log rotation required to prevent disk overflow
- **Network**: Binance API rate limits (1200 requests/minute)

### **Operational Constraints**
- **Server Location**: Ubuntu Hetzner dedicated server only
- **No Local Execution**: All operations must run on remote server
- **SSH Access**: Required for all management operations
- **Time Zones**: UTC-based timestamps, CET/CEST local time

### **Trading Constraints**
- **Paper Trading Only**: No real money at risk
- **Symbols**: Limited to EUR pairs (BTCEUR, ETHEUR, ADAEUR, DOTEUR, LINKEUR)
- **Timeframes**: Optimized for 30-minute candles
- **Position Limits**: Maximum 25% portfolio per position

## Development Workflow

### **Local Development**
1. **Clone Repository**: `git clone <repo>`
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Editable Install**: `pip install -e .`
4. **Configuration**: Copy [`.env.example`](.env.example) to `.env`
5. **Test Locally**: Run [`quick_test_system.py`](quick_test_system.py)

### **Server Deployment**
1. **Upload Changes**: `scp` or `rsync` to `/opt/trading_bot/`
2. **Update Dependencies**: `pip install -r requirements.txt`
3. **Run Tests**: [`test_deploy_trader.py`](test_deploy_trader.py)
4. **Deploy**: [`scripts/deploy_trading.sh`](scripts/deploy_trading.sh)
5. **Monitor**: Check logs and Telegram alerts

## Maintenance Procedures

### **Routine Maintenance**
- **Daily**: Automated log rotation and health checks
- **Weekly**: Performance report generation
- **Monthly**: Model performance review and potential retraining
- **Quarterly**: Full system backup and disaster recovery test

### **Emergency Procedures**
- **Service Down**: `sudo systemctl restart trading-bot`
- **Memory Issues**: Check [`scripts/health_check.sh`](scripts/health_check.sh) for guidance
- **API Issues**: Monitor rate limits and connection status
- **Model Failures**: Use [`_load_model_with_fallbacks()`](scripts/enhanced_trader.py:531) strategy

## Tool Usage Patterns

### **Configuration Management**
- **ConfigLoader**: [`src/config/config_loader.py`](src/config/config_loader.py:24) - Auto-detection
- **Environment Manager**: [`src/config/environment_manager.py`](src/config/environment_manager.py)
- **Hierarchical Config**: [`src/config/hierarchical_config.py`](src/config/hierarchical_config.py)

### **Data Pipeline Tools**
- **Feature Engine**: [`src/data_pipeline/features.py`](src/data_pipeline/features.py:8) - 200+ indicators
- **Preprocessor**: [`src/data_pipeline/preprocess.py`](src/data_pipeline/preprocess.py:6) - Scaling/normalization
- **Validation**: [`src/validation/validation_integration.py`](src/validation/validation_integration.py:17)

### **Trading Tools**
- **Signal Generator**: [`src/trading/enhanced_signal_generator.py`](src/trading/enhanced_signal_generator.py:47)
- **Profit Optimizer**: [`src/trading/profit_optimizer.py`](src/trading/profit_optimizer.py:54)
- **Performance Analytics**: [`src/trading/performance_analytics.py`](src/trading/performance_analytics.py)

## Integration Points

### **External APIs**
- **Binance API**: Market data and trading (via [`ccxt`](requirements.txt:33))
- **Telegram Bot API**: Notifications and commands
- **MLFlow**: Model tracking and experiment management
- **Wandb**: Advanced performance monitoring (optional)

### **Internal Services**
- **Health Monitor**: [`src/core/health_monitor.py`](src/core/health_monitor.py)
- **Circuit Breaker**: [`src/core/circuit_breaker.py`](src/core/circuit_breaker.py)
- **Error Handler**: [`src/core/error_handler.py`](src/core/error_handler.py)
- **Shutdown Handler**: [`src/core/shutdown_handler.py`](src/core/shutdown_handler.py)

## Version Control & Package Management

### **Python Package Management**
- **Installation**: `pip install -e .` (editable development install)
- **Requirements**: [`requirements.txt`](requirements.txt) with version pinning
- **Setup**: [`setup.py`](setup.py:7) defines package structure and dependencies

### **Model Versioning**
- **Packaging**: [`src/utils/model_packaging.py`](src/utils/model_packaging.py)
- **Transfer**: [`src/utils/model_transfer.py`](src/utils/model_transfer.py) 
- **Metadata**: JSON files track model versions and performance

### **Configuration Versioning**
- **Backup Script**: [`scripts/backup_config.sh`](scripts/backup_config.sh)
- **Git Integration**: Version control for configuration changes
- **Deployment Logging**: Track configuration deployments

## Production Checklist

### **Pre-deployment Validation**
- [ ] All dependencies installed: `pip check`
- [ ] Configuration valid: [`validate_fixes.py`](validate_fixes.py)
- [ ] Models loadable: [`test_deploy_trader.py`](test_deploy_trader.py)
- [ ] Telegram working: [`test_telegram.py`](test_telegram.py)
- [ ] Health check passing: [`scripts/health_check.sh`](scripts/health_check.sh)

### **Post-deployment Monitoring**
- [ ] Service started: `systemctl status trading-bot`
- [ ] Logs clean: `journalctl -u trading-bot --since "5 minutes ago"`
- [ ] Telegram alerts: Startup notification received
- [ ] Trading active: Check first iteration logs
- [ ] Performance tracking: Metrics being recorded