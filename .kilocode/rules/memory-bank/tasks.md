# Operational Tasks & Maintenance Procedures

## System Health & Monitoring Tasks

### Check System Health Status
**Purpose**: Verify all system components are functioning correctly
**Frequency**: Daily or when issues suspected
**Files involved**: [`scripts/health_check.sh`](scripts/health_check.sh), [`scripts/enhanced_trader.py`](scripts/enhanced_trader.py)

**Steps**:
1. SSH to server: `ssh trader@server`
2. Run health check: `/opt/trading_bot/scripts/health_check.sh`
3. Check systemd status: `sudo systemctl status trading-bot`
4. Review tmux session: `/opt/trading_bot/scripts/tmux_manager.sh status`
5. Check recent logs: `sudo journalctl -u trading-bot --since "1 hour ago"`
6. Verify Telegram connectivity: `python3 test_telegram.py`

**Expected Results**: All components show "healthy" status, no error logs
**Troubleshooting**: If unhealthy, check specific component logs and restart services

### Monitor Trading Performance
**Purpose**: Track trading metrics and system performance
**Frequency**: Daily monitoring, weekly detailed analysis
**Files involved**: [`trades_report.csv`](trades_report.csv), [`logs/performance_metrics.json`](logs/performance_metrics.json)

**Steps**:
1. Check recent trades: `tail -20 /opt/trading_bot/logs/trades_report.csv`
2. Review performance metrics: `cat /opt/trading_bot/logs/performance_metrics.json | jq .`
3. Monitor portfolio value progression
4. Check Sharpe ratio and drawdown metrics
5. Analyze win/loss ratio and trade frequency
6. Verify model prediction accuracy per symbol

**Key Metrics to Monitor**:
- Sharpe ratio > 1.5 (target)
- Max drawdown < 5%
- Win rate > 50%
- Model prediction accuracy per symbol
- System uptime and response times

### Telegram Bot Maintenance
**Purpose**: Ensure Telegram notifications and commands are working
**Frequency**: Weekly verification
**Files involved**: [`test_telegram.py`](test_telegram.py), [`src/notifier/telegram.py`](src/notifier/telegram.py)

**Steps**:
1. Test basic connectivity: `python3 test_telegram.py`
2. Send test message via bot commands: `/status`
3. Verify all command responses: `/health`, `/performance`, `/balance`
4. Check notification delivery for trades and alerts
5. Verify environment variables: `cat /etc/trading_bot/.env`

**Common Issues**:
- Token expiration: Update `TELEGRAM_BOT_TOKEN` in `.env`
- Chat ID changes: Verify `TELEGRAM_CHAT_ID` 
- Network connectivity: Check firewall and API access

## Configuration Management Tasks

### Update Trading Configuration
**Purpose**: Deploy configuration changes to production
**Frequency**: As needed for parameter tuning
**Files involved**: [`training_config.yaml`](training_config.yaml), [`scripts/deploy_trading.sh`](scripts/deploy_trading.sh)

**Steps**:
1. Backup current config: `/opt/trading_bot/scripts/backup_config.sh`
2. Upload new config: `scp training_config.yaml trader@server:/opt/trading_bot/`
3. Validate configuration: `python3 validate_fixes.py`
4. Test config loading: `python3 -c "from src.config.config_loader import ConfigLoader; print(ConfigLoader().config)"`
5. Deploy changes: `/opt/trading_bot/scripts/deploy_trading.sh`
6. Restart service: `sudo systemctl restart trading-bot`
7. Verify startup: Monitor first 10 minutes of logs

**Rollback Procedure**:
- Restore from backup: `/opt/trading_bot/backups/`
- Restart service with previous config
- Verify system stability

### Update Model Weights and Thresholds
**Purpose**: Optimize trading performance by adjusting model ensemble weights and trading thresholds
**Frequency**: Monthly or when performance degrades
**Files involved**: [`training_config.yaml`](training_config.yaml:130-133), [`training_config.yaml`](training_config.yaml:117-123)

**Steps**:
1. Analyze recent model performance: Review prediction accuracy per model
2. Calculate new weights based on performance: Review Sharpe ratio contribution
3. Backup current configuration
4. Update model weights in [`training_config.yaml`](training_config.yaml:130-133)
5. Adjust per-symbol thresholds in [`training_config.yaml`](training_config.yaml:117-123)
6. Test configuration locally if possible
7. Deploy to production following standard deployment procedure
8. Monitor performance for 48 hours post-deployment
9. Revert if performance degrades

**Performance Indicators**:
- Individual model accuracy rates
- Contribution to overall portfolio returns
- Risk-adjusted performance metrics

## Model Management Tasks

### Deploy New Model Version
**Purpose**: Update ML models with newly trained versions
**Frequency**: When new models are trained (monthly/quarterly)
**Files involved**: Model files (`.pth`, `.pkl`, `.zip`), [`scripts/run_enhanced_trader_test.py`](scripts/run_enhanced_trader_test.py)

**Steps**:
1. Backup existing models: `tar -czf models_backup_$(date +%Y%m%d).tar.gz /opt/trading_bot/models/`
2. Upload new models: `scp -r models/ trader@server:/opt/trading_bot/`
3. Verify model structure: Check model directory organization
4. Test model loading: `python3 scripts/run_enhanced_trader_test.py`
5. Run validation tests: `python3 validate_fixes.py`
6. Deploy models: Restart trading service
7. Monitor first hour: Check for model loading errors
8. Verify predictions: Check model prediction accuracy

**Rollback Procedure**:
- Stop trading service
- Restore model backup
- Restart service and verify functionality

### Model Performance Analysis
**Purpose**: Analyze individual model performance and accuracy
**Frequency**: Weekly detailed analysis
**Files involved**: [`logs/performance_metrics.json`](logs/performance_metrics.json), model prediction logs

**Steps**:
1. Extract model-specific performance data
2. Calculate accuracy metrics per model type (GRU, LightGBM, PPO)
3. Analyze prediction vs actual return correlation
4. Review model contribution to overall portfolio returns
5. Identify underperforming models or symbols
6. Document findings for model retraining decisions
7. Update model weights if performance has significantly changed

**Key Analysis Areas**:
- Directional accuracy (buy/sell signal correctness)
- Return magnitude prediction accuracy
- Model consistency across different market conditions
- Risk-adjusted performance contribution

## Log Management Tasks

### Daily Log Rotation and Cleanup
**Purpose**: Prevent disk overflow and maintain log accessibility
**Frequency**: Daily (automated via cron)
**Files involved**: [`scripts/rotate_logs.sh`](scripts/rotate_logs.sh), [`server/cron/trading_bot_monitor`](server/cron/trading_bot_monitor)

**Manual Steps** (if automation fails):
1. Check disk usage: `df -h /var/log/trading_bot/`
2. Compress old logs: `gzip /var/log/trading_bot/trading.log.1`
3. Rotate current log: `mv /var/log/trading_bot/trading.log /var/log/trading_bot/trading.log.1`
4. Create new log file: `touch /var/log/trading_bot/trading.log`
5. Set permissions: `chmod 644 /var/log/trading_bot/trading.log`
6. Remove old compressed logs: `find /var/log/trading_bot/ -name "*.gz" -mtime +30 -delete`

### Log Analysis for Troubleshooting
**Purpose**: Identify and diagnose system issues from logs
**Frequency**: When issues occur or weekly review
**Files involved**: [`logs/trading.log`](logs/trading.log), [`error.log`](error.log), systemd logs

**Steps**:
1. Check for error patterns: `grep -i "error\|exception\|failed" /var/log/trading_bot/trading.log`
2. Analyze API failures: `grep -i "rate limit\|network error\|timeout" /var/log/trading_bot/trading.log`
3. Review model prediction failures: `grep -i "invalid.*prediction\|shape mismatch" /var/log/trading_bot/trading.log`
4. Check system resource issues: `grep -i "memory\|cpu\|disk" /var/log/trading_bot/trading.log`
5. Review systemd service logs: `sudo journalctl -u trading-bot --since "24 hours ago"`
6. Analyze trade execution patterns: `grep -i "BUY\|SELL" /var/log/trading_bot/trading.log | tail -50`

## Backup and Recovery Tasks

### Create Full System Backup
**Purpose**: Create comprehensive backup for disaster recovery
**Frequency**: Weekly full backup, daily incremental
**Files involved**: All system files, configuration, models, data

**Steps**:
1. Stop trading service: `sudo systemctl stop trading-bot`
2. Create backup directory: `mkdir -p /backup/trading_bot_$(date +%Y%m%d)`
3. Backup application code: `tar -czf /backup/trading_bot_$(date +%Y%m%d)/code.tar.gz /opt/trading_bot/`
4. Backup configuration: `cp /etc/trading_bot/.env /backup/trading_bot_$(date +%Y%m%d)/`
5. Backup models: `tar -czf /backup/trading_bot_$(date +%Y%m%d)/models.tar.gz /opt/trading_bot/models/`
6. Backup logs: `tar -czf /backup/trading_bot_$(date +%Y%m%d)/logs.tar.gz /var/log/trading_bot/`
7. Create backup manifest: Document backup contents and versions
8. Restart service: `sudo systemctl start trading-bot`
9. Verify backup integrity: Test restore procedure on backup

### Disaster Recovery Procedure
**Purpose**: Restore system from complete failure
**Frequency**: As needed during emergencies
**Files involved**: All backup files

**Steps**:
1. Provision new server or clean existing server
2. Install base dependencies: Python 3.8+, pip, git
3. Create trader user: `sudo adduser trader`
4. Restore application code: Extract from code.tar.gz
5. Restore configuration: Copy environment files
6. Restore models: Extract model files to correct locations
7. Install Python dependencies: `pip install -r requirements.txt`
8. Install systemd service: Copy and enable service files
9. Install cron jobs: Copy monitoring scripts
10. Set permissions: `chmod +x scripts/*.sh`
11. Start services: `sudo systemctl start trading-bot`
12. Verify functionality: Run full system test
13. Monitor for 24 hours: Ensure stable operation

## Performance Optimization Tasks

### Analyze and Optimize Trading Performance
**Purpose**: Improve Sharpe ratio and risk-adjusted returns
**Frequency**: Monthly comprehensive review
**Files involved**: [`trades_report.csv`](trades_report.csv), [`training_config.yaml`](training_config.yaml)

**Steps**:
1. Export trading history: `cp /opt/trading_bot/logs/trades_report.csv ./analysis/`
2. Calculate performance metrics: Sharpe ratio, max drawdown, win rate
3. Analyze per-symbol performance: Identify best/worst performing symbols
4. Review model contribution: Calculate returns per model type
5. Optimize thresholds: Adjust buy/sell thresholds based on analysis
6. Update model weights: Reweight ensemble based on performance
7. Test optimizations: Backtest new parameters if possible
8. Deploy optimized configuration
9. Monitor results: Track performance improvement over next week

### Optimize System Resource Usage
**Purpose**: Ensure efficient resource utilization
**Frequency**: Monthly system optimization
**Files involved**: System logs, memory/CPU monitoring data

**Steps**:
1. Monitor resource usage: `htop`, `free -h`, `df -h`
2. Analyze memory patterns: Check for memory leaks in logs
3. Review CPU utilization: Identify bottlenecks in data processing
4. Optimize caching: Review cache hit rates and expiration
5. Clean unused files: Remove old logs, cache, temporary files
6. Update resource limits: Adjust memory limits in configuration
7. Optimize concurrent processing: Tune async/threading parameters
8. Restart services: Apply resource optimizations
9. Monitor improvements: Track resource usage for next week

## Troubleshooting Procedures

### Diagnose and Fix Model Prediction Failures
**Purpose**: Resolve model loading or prediction issues
**Symptoms**: "Invalid prediction", "Shape mismatch", model loading errors
**Files involved**: [`scripts/enhanced_trader.py`](scripts/enhanced_trader.py:531), model files

**Diagnostic Steps**:
1. Check model file integrity: Verify model files exist and are not corrupted
2. Test model loading: `python3 scripts/run_enhanced_trader_test.py`
3. Validate feature generation: Check for NaN values in features
4. Review preprocessor compatibility: Ensure preprocessor matches model training
5. Check model metadata: Verify feature count and sequence length compatibility
6. Test with fallback models: Use alternative model loading strategies

**Resolution Steps**:
1. Use model fallback mechanism: [`_load_model_with_fallbacks()`](scripts/enhanced_trader.py:531)
2. Regenerate features with validation: Check data pipeline
3. Refit preprocessors: Ensure compatibility with current data
4. Update model weights: Reduce weight of failing models
5. Deploy model fixes: Restart with corrected configuration

### Resolve API Connection Issues
**Purpose**: Fix connectivity problems with external APIs (Binance, Telegram)
**Symptoms**: "Network error", "Rate limit exceeded", "Timeout"
**Files involved**: [`scripts/trader.py`](scripts/trader.py:464), API configuration

**Diagnostic Steps**:
1. Check network connectivity: `ping api.binance.com`
2. Verify API credentials: Check environment variables
3. Review rate limiting: Check API call frequency
4. Test API endpoints: Manual API calls to verify functionality
5. Check firewall rules: Ensure API endpoints are accessible

**Resolution Steps**:
1. Implement retry logic: Use [`_fetch_with_retry()`](scripts/trader.py:464)
2. Update API credentials: Refresh tokens if expired
3. Adjust rate limits: Reduce API call frequency
4. Use fallback data sources: Switch to Yahoo Finance if needed
5. Restart networking: `sudo systemctl restart networking`

### Fix Telegram Notification Issues
**Purpose**: Restore Telegram alerts and command functionality
**Symptoms**: Missing notifications, command failures, bot unresponsive
**Files involved**: [`test_telegram.py`](test_telegram.py), [`.env`](.env.example), [`src/notifier/telegram.py`](src/notifier/telegram.py)

**Diagnostic Steps**:
1. Test Telegram connectivity: `python3 test_telegram.py`
2. Verify bot token: Check token validity with Telegram
3. Confirm chat ID: Ensure chat ID is correct
4. Check network access: Verify Telegram API accessibility
5. Review error logs: Look for specific Telegram errors

**Resolution Steps**:
1. Update bot token: Get new token from BotFather if needed
2. Verify chat permissions: Ensure bot has send permissions
3. Restart notification service: Reload Telegram notifier
4. Test all commands: Verify `/status`, `/health`, `/performance` work
5. Monitor notification delivery: Confirm alerts are received

## Deployment and Maintenance Tasks

### Deploy Trading System Updates
**Purpose**: Update production system with code changes
**Frequency**: As needed for bug fixes or feature updates
**Files involved**: All source code, [`scripts/deploy_trading.sh`](scripts/deploy_trading.sh)

**Steps**:
1. Prepare local changes: Ensure all changes are tested locally
2. Create deployment package: `tar -czf trading_bot_update.tar.gz .`
3. Upload to server: `scp trading_bot_update.tar.gz trader@server:/tmp/`
4. Backup current system: Run full backup procedure
5. Stop trading service: `sudo systemctl stop trading-bot`
6. Extract updates: `cd /opt/trading_bot && tar -xzf /tmp/trading_bot_update.tar.gz`
7. Update dependencies: `pip install -r requirements.txt`
8. Set permissions: `chmod +x scripts/*.sh`
9. Validate configuration: `python3 validate_fixes.py`
10. Start service: `sudo systemctl start trading-bot`
11. Monitor startup: Check first 30 minutes of operation
12. Verify functionality: Run comprehensive system test

### Scheduled Maintenance Window
**Purpose**: Perform comprehensive system maintenance
**Frequency**: Monthly scheduled maintenance
**Duration**: 2-4 hours during market low-activity periods

**Preparation**:
1. Schedule maintenance window: Notify via Telegram
2. Stop trading: `sudo systemctl stop trading-bot`
3. Create full backup: Run complete backup procedure

**Maintenance Tasks**:
1. **System Updates**: `sudo apt update && sudo apt upgrade`
2. **Python Updates**: `pip install -r requirements.txt --upgrade`
3. **Log Cleanup**: Remove logs older than 90 days
4. **Model Validation**: Verify all models load correctly
5. **Configuration Review**: Check all configuration files
6. **Security Updates**: Update SSH keys, firewall rules
7. **Performance Optimization**: Analyze and optimize resource usage
8. **Backup Verification**: Test backup and restore procedures

**Post-Maintenance**:
1. Start trading service: `sudo systemctl start trading-bot`
2. Run comprehensive tests: Verify all functionality
3. Monitor for 2 hours: Ensure stable operation
4. Send completion notification: Confirm maintenance complete

### Emergency System Recovery
**Purpose**: Recover from critical system failures
**Frequency**: As needed during emergencies
**Files involved**: All system files, backup archives

**Emergency Response Steps**:
1. **Assess Severity**: Determine if this is a service issue or system failure
2. **Stop Trading**: Immediately stop all trading to prevent losses
3. **Identify Root Cause**: Check logs, system status, resource usage
4. **Attempt Quick Fix**: Restart services, clear cache, reset connections
5. **If Quick Fix Fails**: Initiate full recovery procedure
6. **Notify Stakeholders**: Send emergency alerts via Telegram
7. **Document Incident**: Record issue details and resolution

**Full Recovery Procedure**:
1. **Provision Backup System**: Prepare alternative server if needed
2. **Restore from Backup**: Use most recent full backup
3. **Validate Restoration**: Test all system components
4. **Resume Trading**: Restart with validated configuration
5. **Monitor Closely**: Watch for 24 hours post-recovery
6. **Post-Incident Review**: Analyze root cause and improve procedures

## Performance Analysis Tasks

### Generate Comprehensive Performance Report
**Purpose**: Create detailed analysis of trading and system performance
**Frequency**: Weekly detailed reports, monthly comprehensive analysis
**Files involved**: [`scripts/generate_performance_report.sh`](scripts/generate_performance_report.sh), performance data

**Steps**:
1. Run report generator: `/opt/trading_bot/scripts/generate_performance_report.sh`
2. Analyze trading metrics: Review Sharpe ratio, returns, drawdowns
3. Examine model performance: Check prediction accuracy per model
4. Review system metrics: CPU, memory, disk usage patterns
5. Identify trends: Look for performance improvements or degradations
6. Compare benchmarks: Check against market and target performance
7. Generate recommendations: Suggest optimizations based on analysis
8. Create summary report: Document key findings and actions
9. Share insights: Send report summary via Telegram

### Optimize Trading Parameters
**Purpose**: Fine-tune trading parameters for better performance
**Frequency**: Bi-weekly parameter optimization
**Files involved**: [`training_config.yaml`](training_config.yaml), performance history

**Analysis Areas**:
1. **Threshold Optimization**: Analyze current buy/sell thresholds vs performance
2. **Position Sizing**: Review Kelly criterion parameters and position limits
3. **Risk Management**: Evaluate stop-loss and profit-taking levels
4. **Model Weights**: Assess contribution of each model to returns
5. **Symbol Performance**: Identify best/worst performing trading pairs

**Optimization Process**:
1. Collect performance data: Last 30 days of trading results
2. Run parameter sensitivity analysis: Test different threshold values
3. Calculate optimal parameters: Use statistical analysis for best values
4. Implement changes: Update configuration with optimized parameters
5. Deploy and monitor: Track performance improvement
6. Document changes: Record parameter changes and results

## Alert and Notification Management

### Configure Alert Thresholds
**Purpose**: Optimize alert sensitivity and reduce noise
**Frequency**: Monthly threshold review
**Files involved**: [`src/validation/validation_integration.py`](src/validation/validation_integration.py), alert configuration

**Steps**:
1. Review alert history: Analyze frequency and accuracy of alerts
2. Identify false positives: Find alerts that didn't indicate real issues
3. Adjust sensitivity: Modify thresholds to reduce noise
4. Test alert delivery: Verify alerts reach Telegram reliably
5. Update configuration: Deploy new alert thresholds
6. Monitor alert quality: Track improvement in alert relevance

### Emergency Alert Response
**Purpose**: Respond to critical system alerts
**Frequency**: As needed when critical alerts received
**Response Time**: < 15 minutes for critical alerts

**Alert Response Procedure**:
1. **Acknowledge Alert**: Confirm receipt of alert
2. **Assess Severity**: Determine urgency and impact
3. **Initial Diagnosis**: Quick check of system status
4. **Containment**: Stop trading if necessary to prevent issues
5. **Investigation**: Deep dive into root cause
6. **Resolution**: Apply fix or workaround
7. **Verification**: Confirm issue is resolved
8. **Documentation**: Record incident and resolution
9. **Follow-up**: Review and improve alert/response procedures

## Data Management Tasks

### Validate Data Quality and Integrity
**Purpose**: Ensure market data and features are accurate and complete
**Frequency**: Daily data quality checks
**Files involved**: [`src/validation/validation_integration.py`](src/validation/validation_integration.py), data files

**Steps**:
1. Run data validation: Check for missing or corrupted data
2. Verify feature generation: Ensure all 200+ features are calculated correctly
3. Check data freshness: Confirm data is recent and up-to-date
4. Validate schema compliance: Ensure data matches expected format
5. Review drift monitoring: Check for data distribution changes
6. Test model inputs: Verify models receive correct data format
7. Document quality issues: Record any data problems found

### Clean and Optimize Data Storage
**Purpose**: Maintain efficient data storage and access
**Frequency**: Monthly data cleanup
**Files involved**: [`data/`](data/), [`cache/`](cache/), log files

**Steps**:
1. Archive old data: Move historical data to long-term storage
2. Clean cache files: Remove expired cache entries
3. Optimize database: Compact SQLite databases if used
4. Review data usage: Identify unnecessary data files
5. Implement data retention: Remove data older than required
6. Verify data access: Ensure trading system can access needed data
7. Monitor storage usage: Track disk usage patterns

---

## Quick Reference Commands

### Essential Status Commands
```bash
# System status
sudo systemctl status trading-bot
/opt/trading_bot/scripts/tmux_manager.sh status

# Recent logs
sudo journalctl -u trading-bot --since "1 hour ago"
tail -f /var/log/trading_bot/trading.log

# Performance check
cat /opt/trading_bot/logs/performance_metrics.json | jq .

# Health check
/opt/trading_bot/scripts/health_check.sh
```

### Emergency Commands
```bash
# Stop trading immediately
sudo systemctl stop trading-bot

# Restart trading system
sudo systemctl restart trading-bot

# Check for errors
grep -i "error\|exception" /var/log/trading_bot/trading.log | tail -20

# Monitor resource usage
htop
df -h
```

### Backup Commands
```bash
# Quick backup
/opt/trading_bot/scripts/backup_config.sh

# Full system backup
tar -czf trading_bot_backup_$(date +%Y%m%d).tar.gz /opt/trading_bot/