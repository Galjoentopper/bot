# Current Context: Stable Production System

## Project Directory Context
- **Main Directory**: `bot/` folder contains the complete trading system
- **Development Environment**: Windows local development with Ubuntu server deployment
- **Project Structure**: Organized with separate folders for src/, scripts/, server/, data/, etc.
- **Current Focus**: Memory bank documentation updates to reflect correct folder structure

## Current System State
- **Status**: Stable and operational on Ubuntu Hetzner server
- **Last Major Changes**: System is running in production with all three ML models (GRU, LightGBM, PPO) deployed
- **Trading Symbols**: BTCEUR, ETHEUR, ADAEUR, DOTEUR, LINKEUR (30-minute candles)
- **Operation Mode**: Continuous paper trading with profit optimization

## Primary Focus Areas
- **System Maintenance**: Ensuring 99%+ uptime and stable operation
- **Performance Monitoring**: Tracking Sharpe ratio and risk-adjusted returns
- **Operational Documentation**: Improving troubleshooting and maintenance procedures
- **Alert Management**: Telegram-based monitoring and notification systems

## Recent System Developments
- Enhanced trader script with robust model loading fallbacks
- Profit optimization system with Kelly criterion position sizing
- Comprehensive validation and drift monitoring framework
- Automated health checks and recovery procedures
- **Current**: Updated memory bank documentation to reflect bot/ as main project directory

## Immediate Priorities
1. **Documentation Enhancement**: Complete operational documentation for maintenance
2. **Monitoring Optimization**: Fine-tune alert thresholds and health check frequency
3. **Performance Analysis**: Review trading performance and risk metrics
4. **Backup Verification**: Ensure all backup and recovery procedures are functional

## Critical Maintenance Areas
- **Log Management**: Automated rotation via [`scripts/rotate_logs.sh`](scripts/rotate_logs.sh)
- **Health Monitoring**: Cron jobs running every 5 minutes for system checks
- **Model Performance**: Continuous tracking of prediction accuracy and trading results
- **Configuration Management**: Backup and versioning of critical configuration files

## Next Steps
- Establish routine maintenance schedules and procedures
- Document troubleshooting workflows for common operational issues
- Enhance monitoring dashboards and alerting systems
- Plan for model retraining and performance optimization cycles

## Known Stable Components
- **Data Pipeline**: 200+ feature generation working reliably
- **ML Models**: All three model types loading and predicting successfully
- **Trading Engine**: Enhanced signal generation with profit optimization active
- **Infrastructure**: Systemd services, tmux sessions, and cron jobs functioning
- **Notifications**: Telegram bot operational for alerts and status updates

## Maintenance Schedule
- **Daily**: Automated health checks, log rotation, performance tracking
- **Weekly**: Performance reports, model accuracy analysis
- **Monthly**: Full system review, backup validation
- **As Needed**: Configuration updates, model redeployment, troubleshooting