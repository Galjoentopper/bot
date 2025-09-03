# Product Overview: Enterprise Crypto Trading Bot

## Purpose & Vision

This crypto trading bot is an enterprise-grade automated trading system designed to operate continuously on a Ubuntu Hetzner server, maximizing profitability through sophisticated ML-driven strategies while maintaining operational stability and reliability.

## Core Problems Solved

### 1. **Manual Trading Limitations**
- Eliminates human emotion and bias from trading decisions
- Operates 24/7 without fatigue or attention lapses
- Processes multiple data streams simultaneously
- Maintains consistent strategy execution

### 2. **Market Complexity Management**
- Integrates multiple ML models (GRU neural network, LightGBM ensemble, PPO reinforcement learning)
- Processes 200+ technical indicators in real-time
- Adapts to changing market conditions automatically
- Manages risk across multiple cryptocurrency pairs

### 3. **Operational Reliability**
- Self-monitoring with health checks and alerts
- Automatic recovery from common failures
- Comprehensive logging and performance tracking
- Telegram notifications for critical events

## How the System Works

### **Production Operation Flow:**
1. **Data Acquisition**: Real-time market data from Binance API (30-minute candles)
2. **Feature Engineering**: 200+ technical indicators processed per symbol
3. **ML Prediction**: Three models generate ensemble predictions
4. **Signal Generation**: Enhanced signal generator with profit optimization
5. **Risk Management**: Dynamic position sizing, stop losses, correlation analysis
6. **Trade Execution**: Paper trading with realistic fees and slippage simulation
7. **Monitoring**: Continuous health monitoring with Telegram alerts

## User Experience Goals

### **Operational Stability**
- **Primary Goal**: System runs autonomously with minimal intervention
- **Reliability**: 99%+ uptime on production server
- **Predictability**: Consistent behavior and performance patterns
- **Maintainability**: Clear troubleshooting and maintenance procedures

### **Performance Excellence**
- **Primary Metric**: Maximize Sharpe ratio (risk-adjusted returns)
- **Secondary Goals**: Minimize drawdowns, maintain positive returns
- **Risk Management**: Strict position limits and correlation controls
- **Transparency**: Detailed performance reporting and analytics

### **Ease of Management**
- **Monitoring**: Real-time status via Telegram notifications
- **Control**: Simple commands for start/stop/status operations
- **Visibility**: Comprehensive logging and performance dashboards
- **Recovery**: Automated recovery with manual override capabilities

## Key Success Metrics

### **Financial Performance**
- Sharpe Ratio > 1.5 (target for risk-adjusted returns)
- Maximum Drawdown < 5% (strict risk control)
- Win Rate > 50% (consistent profitable trades)
- Annual Return > 8% (outperform benchmark)

### **Operational Performance**
- System Uptime > 99% (reliable operation)
- Alert Response < 15 minutes (quick issue resolution)
- Model Accuracy > 55% (directional prediction success)
- Trade Execution < 30 seconds (responsive to signals)

## Target Environment

### **Production Infrastructure**
- **Server**: Ubuntu 20.04+ on Hetzner dedicated server
- **Management**: Systemd services with tmux session handling
- **Monitoring**: Cron jobs, health checks, and automated reporting
- **Notifications**: Telegram bot for real-time alerts and status

### **Operational Context**
- **Deployment**: All operations occur on remote server (no local execution)
- **Access**: SSH-based management with secure key authentication
- **Maintenance**: Scheduled maintenance windows with minimal downtime
- **Backup**: Automated configuration and model backups

## Long-term Vision

### **Stability & Reliability**
- Self-healing system with minimal manual intervention
- Comprehensive monitoring and alerting ecosystem
- Robust error handling and recovery mechanisms
- Clear operational procedures for all scenarios

### **Performance Optimization**
- Continuous model performance monitoring
- Adaptive strategies based on market conditions
- Enhanced risk management and profit optimization
- Integration of additional data sources and indicators

### **Operational Excellence**
- Streamlined maintenance and troubleshooting procedures
- Comprehensive documentation for all system components
- Automated performance reporting and analysis
- Scalable infrastructure for future enhancements