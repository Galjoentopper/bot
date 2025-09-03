# System Architecture: Enterprise Crypto Trading Bot

## High-Level System Overview

The crypto trading bot is built as a multi-layered, event-driven system designed for continuous operation on Ubuntu Hetzner server with comprehensive monitoring and fault tolerance.

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION SERVER                        │
│                   (Ubuntu Hetzner)                         │
├─────────────────────────────────────────────────────────────┤
│  Systemd Services → Tmux Sessions → Trading Applications   │
│  Cron Jobs → Health Monitoring → Telegram Alerts          │
│  Log Management → Performance Reports → Backup Systems    │
└─────────────────────────────────────────────────────────────┘
```

## Core System Components

### **1. Data Pipeline Layer**
**Location**: [`src/data_pipeline/`](src/data_pipeline/)
**Critical Path**: Market data → Feature engineering → Model input

#### Primary Components:
- **FeatureEngine** ([`src/data_pipeline/features.py`](src/data_pipeline/features.py:8))
  - Generates 200+ technical indicators
  - **Failure Point**: NaN/infinite values in indicators
  - **Troubleshooting**: Check [`_clean_features_for_inference()`](scripts/trader.py:584) function
  
- **DataPreprocessor** ([`src/data_pipeline/preprocess.py`](src/data_pipeline/preprocess.py:6))  
  - Normalizes and scales features for ML models
  - **Failure Point**: Schema mismatches between training/inference
  - **Troubleshooting**: Validate preprocessor compatibility via [`_ensure_preprocessor_fitted()`](scripts/enhanced_trader.py:803)

#### Data Flow:
```
Binance API (30m candles) → FeatureEngine → DataPreprocessor → ML Models
        ↓                        ↓              ↓             ↓
   OHLCV Data              200+ Features    Scaled Data    Predictions
```

#### **Critical Failure Points:**
1. **API Rate Limits**: [`_fetch_with_retry()`](scripts/trader.py:464) handles retries
2. **Feature Generation Errors**: [`_generate_features_with_validation()`](scripts/trader.py:661) 
3. **Data Validation Failures**: [`_validate_market_data()`](scripts/trader.py:703)

### **2. Machine Learning Layer**
**Location**: [`src/models/`](src/) (implied structure)
**Critical Path**: Features → Model predictions → Trading signals

#### Model Architecture:
- **GRU Neural Network**
  - **Input Shape**: (sequence_length, 113 features)
  - **Location**: Models stored as `.pth` files
  - **Failure Point**: Sequence length mismatches
  - **Troubleshooting**: [`_get_gru_prediction()`](scripts/trader.py:798)

- **LightGBM Ensemble** 
  - **Input Shape**: (1, 114 features)
  - **Location**: Models stored as `.pkl` files
  - **Failure Point**: Feature count mismatches
  - **Troubleshooting**: [`_get_lightgbm_prediction()`](scripts/trader.py:848)

- **PPO Reinforcement Learning**
  - **Input Shape**: (sequence_length, 13 features)
  - **Location**: Models stored as `.zip` files  
  - **Failure Point**: Observation space mismatches
  - **Troubleshooting**: [`_get_ppo_prediction()`](scripts/enhanced_trader.py:1469)

#### **Model Loading Strategy:**
```
Packaged Models → Imported Models → Best Walk-Forward → Latest → Unified Artifacts
      ↓                ↓                ↓              ↓           ↓
  High Priority    Medium Priority   Good Quality   Fallback   Last Resort
```

### **3. Trading Engine Layer**
**Location**: [`src/trading/`](src/trading/)
**Critical Path**: Signals → Risk management → Trade execution

#### Core Components:
- **EnhancedSignalGenerator** ([`src/trading/enhanced_signal_generator.py`](src/trading/enhanced_signal_generator.py:47))
  - **Purpose**: Combines model predictions with market context
  - **Failure Point**: Model ensemble weight calculation errors
  - **Key Method**: [`generate_enhanced_signals()`](src/trading/enhanced_signal_generator.py:92)

- **ProfitOptimizer** ([`src/trading/profit_optimizer.py`](src/trading/profit_optimizer.py:54))
  - **Purpose**: Dynamic position sizing, stop losses, profit targets
  - **Failure Point**: Kelly criterion calculations with invalid inputs
  - **Key Methods**: [`calculate_optimal_position_size()`](src/trading/profit_optimizer.py:140)

- **Risk Management System**
  - **Components**: Position limits, correlation analysis, drawdown controls
  - **Failure Point**: Portfolio value calculation errors
  - **Monitoring**: [`update_performance_metrics()`](src/trading/profit_optimizer.py:540)

### **4. Infrastructure Layer**
**Location**: [`server/`](server/) and root-level operational scripts
**Critical Path**: System services → Process management → Health monitoring

#### Service Management:
- **Systemd Service** ([`server/systemd/trading-bot.service`](server/systemd/trading-bot.service))
  - **Purpose**: System-level service management
  - **Failure Point**: Service crashes, memory limits
  - **Troubleshooting**: `systemctl status trading-bot`

- **Tmux Session Management** ([`server/scripts/tmux_manager.sh`](server/scripts/tmux_manager.sh))
  - **Purpose**: Process isolation and session persistence
  - **Failure Point**: Session disconnections, zombie processes
  - **Commands**: `tmux list-sessions`, `tmux attach`

#### Health Monitoring:
- **Cron-based Monitoring** ([`server/cron/trading_bot_monitor`](server/cron/trading_bot_monitor))
  - **Schedule**: Every 5 minutes
  - **Purpose**: Health checks, automated restart
  - **Failure Point**: Cron service failures, script permissions

### **5. Validation & Monitoring Layer**
**Location**: [`src/validation/`](src/validation/)
**Critical Path**: Data validation → Drift detection → Alert generation

#### Validation Components:
- **ValidationManager** ([`src/validation/validation_integration.py`](src/validation/validation_integration.py:17))
  - **Purpose**: Schema validation, drift monitoring
  - **Configuration**: [`validation_config.json`](src/validation/validation_integration.py:60)
  - **Failure Point**: False positive drift alerts

- **SchemaValidator** ([`src/validation/schema_validator.py`](src/validation/schema_validator.py) - implied)
  - **Purpose**: Feature schema consistency checks
  - **Failure Point**: Schema evolution without model updates

## Configuration Management

### **Primary Configuration**
**Location**: [`training_config.yaml`](training_config.yaml)
**Critical Settings**:
- **Symbols**: [`BTCEUR`, `ETHEUR`, `ADAEUR`, `DOTEUR`, `LINKEUR`](training_config.yaml:7)
- **Interval**: [`30m`](training_config.yaml:8) (30-minute candles)
- **Trading Thresholds**: [`per_symbol thresholds`](training_config.yaml:117-123)
- **Model Weights**: [`gru: 0.35, lightgbm: 0.55, ppo: 0.1`](training_config.yaml:130-133)

### **Environment Configuration**
**Location**: [`.env.example`](.env.example)
**Critical Values**:
- **Telegram Bot**: [`TELEGRAM_BOT_TOKEN`](.env.example:5), [`TELEGRAM_CHAT_ID`](.env.example:6)
- **Trading APIs**: [`BITVAVO_API_KEY`](.env.example:9), [`BITVAVO_API_SECRET`](.env.example:10)
- **Logging**: [`LOG_LEVEL`](.env.example:16), [`LOG_FILE`](.env.example:17)

## Design Patterns & Architectural Decisions

### **1. Multi-Model Ensemble Pattern**
- **Decision**: Use weighted ensemble of 3 different ML approaches
- **Rationale**: Reduces overfitting, improves robustness
- **Implementation**: [`_combine_predictions()`](scripts/trader.py:966)
- **Trade-off**: Increased complexity vs improved accuracy

### **2. Circuit Breaker Pattern**
- **Location**: [`src/core/circuit_breaker.py`](src/core/circuit_breaker.py)
- **Purpose**: Prevent cascade failures during API outages
- **Implementation**: Automatic service degradation
- **Monitoring**: Health check integration

### **3. Strategy Pattern for Model Loading**
- **Implementation**: [`_load_model_with_fallbacks()`](scripts/enhanced_trader.py:531)
- **Strategies**: Packaged → Imported → Best Walk-forward → Latest → Unified
- **Benefit**: Resilient model loading with graceful degradation

### **4. Observer Pattern for Notifications**
- **Implementation**: [`TelegramNotifier`](src/notifier/telegram.py:5)
- **Events**: Trade executions, system alerts, performance reports
- **Decoupling**: Trading logic independent of notification system

## Critical Dependencies

### **External APIs**
- **Binance API**: Market data source (CRITICAL - single point of failure)
- **Telegram API**: Alert notifications (HIGH - operational visibility)

### **Python Libraries**
- **ML/AI**: [`torch`](requirements.txt:12), [`lightgbm`](requirements.txt:15), [`stable-baselines3`](requirements.txt:20)
- **Data**: [`pandas`](requirements.txt:6), [`numpy`](requirements.txt:7), [`ccxt`](requirements.txt:33)
- **Infrastructure**: [`python-telegram-bot`](requirements.txt:37), [`aiohttp`](requirements.txt:25)

### **System Dependencies**
- **Ubuntu Server**: Base operating system
- **Python 3.8+**: Runtime environment  
- **Systemd**: Service management
- **Tmux**: Session management
- **Cron**: Scheduled tasks

## Failure Modes & Recovery Procedures

### **1. Model Prediction Failures**
- **Symptoms**: [`Invalid GRU prediction`](scripts/trader.py:838), shape mismatches
- **Recovery**: Fallback to subset of working models
- **Prevention**: Schema validation, model compatibility checks

### **2. API Connection Failures** 
- **Symptoms**: Network timeouts, rate limit exceeded
- **Recovery**: [`_fetch_with_retry()`](scripts/trader.py:464) with exponential backoff
- **Prevention**: Connection pooling, rate limiting

### **3. Data Pipeline Failures**
- **Symptoms**: NaN values, feature generation errors
- **Recovery**: [`_clean_features_for_inference()`](scripts/trader.py:584)
- **Prevention**: Input validation, robust error handling

### **4. System Resource Exhaustion**
- **Symptoms**: Memory leaks, CPU throttling
- **Recovery**: Automatic service restart via systemd
- **Prevention**: Resource monitoring, log rotation

## Performance Considerations

### **Latency-Critical Paths**
1. **Market Data → Trading Decision**: Target < 30 seconds
2. **Signal Generation → Trade Execution**: Target < 5 seconds  
3. **Health Check Response**: Target < 5 seconds

### **Memory Management**
- **Model Loading**: Lazy loading with caching
- **Data Pipeline**: Sliding window approach for historical data
- **Log Management**: [`scripts/rotate_logs.sh`](scripts/rotate_logs.sh) automatic rotation

### **Concurrency Strategy**
- **Async Data Fetching**: [`asyncio`](scripts/enhanced_trader.py:912) for parallel symbol processing
- **Thread Safety**: Model loading synchronization
- **Resource Pooling**: Connection reuse for API calls

## Monitoring & Observability

### **Health Check Endpoints**
- **System Health**: [`health_check()`](scripts/enhanced_trader.py:594)
- **Model Status**: Per-symbol model availability checks  
- **Performance Metrics**: Sharpe ratio, drawdown, win rate tracking

### **Logging Strategy**
- **Application Logs**: [`logs/trading.log`](training_config.yaml:191)
- **System Logs**: Systemd journal integration
- **Performance Logs**: [`logs/performance_metrics.json`](scripts/enhanced_trader.py:2517)

### **Alert Thresholds**
- **Critical**: System down, model failures
- **Warning**: Performance degradation, API issues
- **Info**: Trade executions, system status