# Enterprise Crypto Trading Bot

## Project Overview
An enterprise-grade automated cryptocurrency trading system that maximizes Sharpe ratio through sophisticated machine learning strategies while maintaining operational stability on Ubuntu Hetzner server.

## Main Objectives
- **Maximize Sharpe Ratio**: Optimize risk-adjusted returns with target >1.5
- **Operational Stability**: Achieve 99%+ uptime with autonomous operation
- **Risk Control**: Maintain strict drawdown limits (<5% maximum)
- **Ease of Management**: Comprehensive Telegram-based monitoring and alerts

## Key Features
- **Multi-Model ML Ensemble**: GRU neural networks + LightGBM ensemble + PPO reinforcement learning
- **Real-time Processing**: 200+ technical indicators from Binance API (30-minute candles)
- **Advanced Risk Management**: Kelly criterion sizing, trailing stops, correlation analysis
- **Enterprise Infrastructure**: Systemd services, tmux sessions, automated health checks
- **Performance Analytics**: Comprehensive monitoring with drift detection and alerts

## Core Technologies
- **Runtime**: Python 3.8+ with 100+ specialized dependencies
- **ML/AI**: PyTorch, LightGBM, Stable-baselines3, Scikit-learn
- **Data Stack**: Pandas, NumPy, CCXT, technical analysis libraries
- **Infrastructure**: Ubuntu 20.04+ server, Systemd, Tmux, Cron
- **Communications**: Telegram Bot API for notifications and control

## Trading Scope
- **Symbols**: BTCEUR, ETHEUR, ADAEUR, DOTEUR, LINKEUR
- **Mode**: Paper trading with realistic fees and slippage simulation
- **Timeframe**: 30-minute candles optimized for data freshness
- **Risk Limits**: Maximum 25% portfolio per position

## Significance
This system eliminates human trading limitations through 24/7 autonomous operation, sophisticated ML-driven decision making, and enterprise-grade reliability. It provides automated cryptocurrency trading with comprehensive risk management, performance monitoring, and operational procedures optimized for maximum profitability.

**Critical Constraint**: All operations must execute exclusively on the remote Ubuntu Hetzner server.