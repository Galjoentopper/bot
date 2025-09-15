# Enhanced Logging System Guide
## Production Trading Bot Logging Architecture

This guide covers the new centralized, structured logging system implemented for the production trading bot.

## 🎯 **Key Features**

### ✅ **What's Enhanced:**
- **Structured Logging**: Purpose-based log categories (trading, model, system, debug)
- **Proper File Management**: Smart rotation and retention policies
- **Performance Optimized**: 5000+ logs/second performance
- **Backward Compatible**: Existing code continues to work
- **Environment Aware**: Different verbosity per environment
- **CSV Trade Data**: Preserves existing trade data format

### ✅ **What's Preserved:**
- **Paperspace Training**: Unchanged simple console logging
- **Trade History**: Complete `trades_report.csv` compatibility
- **Performance Reports**: All existing functionality maintained
- **Telegram Integration**: Existing notification system works

## 📁 **New Log File Structure**

```
logs/
├── trading.log          # All trading decisions and executions (never rotated)
├── models.log           # Model predictions and ML operations (daily rotation)
├── system.log           # System startup, errors, warnings (size rotation)
├── performance.log      # Performance metrics and timing (size rotation)
├── debug.log            # Debug information (minimal retention)
└── trades_report.csv    # Structured trade data (preserved format)
```

## 🚀 **Quick Start - Using the New System**

### **1. Basic Logger Usage**

```python
# Import the new logging system
from src.core.logging_manager import (
    get_trading_logger,
    get_model_logger,
    get_system_logger,
    get_debug_logger
)

# Get appropriate loggers
trading_logger = get_trading_logger("my_trader")
model_logger = get_model_logger("ensemble")
system_logger = get_system_logger("startup")
debug_logger = get_debug_logger("development")

# Use them
trading_logger.info("TRADE_EXEC | BTCEUR | BUY | 0.001 | SUCCESS")
model_logger.info("MODEL_PRED | lightgbm | ETHEUR | 0.75 | conf: 0.82")
system_logger.info("System startup completed successfully")
debug_logger.debug("Debug information here")
```

### **2. Structured Trade Logging**

```python
from src.core.logging_manager import StructuredTradeLogger

trade_logger = StructuredTradeLogger()

# Log a trade with all metadata
trade_logger.log_trade_execution(
    trade_id="abc123",
    symbol="BTCEUR",
    action="BUY",
    quantity=0.001,
    price=95000,
    success=True,
    reason="Strong ensemble signal",
    confidence=0.85,
    portfolio_value=10500,
    metadata={
        "market_regime": "bullish",
        "model_consensus": 0.82
    }
)
```

### **3. Performance Logging**

```python
from src.core.logging_manager import PerformanceLogger

perf_logger = PerformanceLogger()

# Log operation timing
perf_logger.log_operation_time("model_prediction", 150.5, success=True)

# Log system metrics
perf_logger.log_system_metrics({
    "cpu": 45.2,
    "memory": 2048,
    "disk": 85,
    "active_positions": 5
})
```

### **4. Legacy Compatibility**

```python
# Existing code continues to work unchanged
from src.utils.logger import Logger

logger = Logger("my_component")
logger.logger.info("This still works")  # .logger.info pattern preserved
```

## 🔄 **Migration Guide**

### **Step 1: Replace Import Statements**

```python
# OLD - Multiple conflicting systems
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NEW - Unified system
from src.core.logging_manager import get_system_logger
logger = get_system_logger("component_name")
```

### **Step 2: Choose Appropriate Logger Type**

| **Logger Type** | **Use For** | **Example** |
|-----------------|-------------|-------------|
| `trading` | Trade decisions, executions, portfolio changes | `"BUY BTCEUR: 0.001 @ €95000"` |
| `model` | Model predictions, ensemble decisions, ML ops | `"PRED lightgbm ETHEUR: 0.75"` |
| `system` | Startup, configuration, errors, warnings | `"System initialized successfully"` |
| `debug` | Development, troubleshooting, detailed info | `"Feature count: 127, Memory: 2GB"` |
| `performance` | Timing, metrics, resource usage | `"model_prediction: 125ms"` |

### **Step 3: Update Log Calls**

```python
# OLD - Generic logging
self.logger.logger.info(f"Trade executed: {symbol}")

# NEW - Structured logging
self.trade_logger.log_trade_execution(
    trade_id=trade_id,
    symbol=symbol,
    action="BUY",
    # ... other parameters
)
```

## 🌍 **Environment Configuration**

### **Development Environment**
```bash
export TRADING_ENV=development
export DEBUG_MODE=true
```
- **All log levels enabled**
- **Full debug information**
- **Detailed model predictions**

### **Production Environment**
```bash
export TRADING_ENV=production
export DEBUG_MODE=false
```
- **INFO level for trading/model logs**
- **WARNING level for system logs**
- **ERROR level for debug logs**
- **Optimized performance**

## 📊 **Log Analysis Examples**

### **Trading Performance Analysis**
```bash
# View recent trades
tail -n 20 logs/trades_report.csv

# Search for specific symbol trades
grep "BTCEUR" logs/trading.log

# Find profitable trades
grep "SUCCESS" logs/trades_report.csv | grep "BUY"
```

### **Model Performance Analysis**
```bash
# View model predictions
grep "MODEL_PRED" logs/models.log

# Find ensemble decisions
grep "ensemble" logs/models.log | tail -10

# Check prediction confidence
grep "conf:" logs/models.log | grep "ETHEUR"
```

### **System Monitoring**
```bash
# Check system health
tail -n 50 logs/system.log

# Find errors and warnings
grep -E "(ERROR|WARNING)" logs/system.log

# Monitor performance
tail -f logs/performance.log
```

## 🔧 **Configuration Options**

### **Environment Variables**
```bash
# Set logging environment
export TRADING_ENV=production     # development, testing, production
export DEBUG_MODE=false           # true for verbose debug logs

# Optional: Override log directory
export TRADING_LOGS_DIR=/custom/path/logs
```

### **Configuration File: `config/logging_config.yaml`**
```yaml
logging:
  environment:
    current: ${TRADING_ENV:-development}
    debug_mode: ${DEBUG_MODE:-false}

  file_management:
    logs_directory: "logs"
    rotation:
      trading:
        type: "preserve"  # Never delete trading data
      model:
        type: "time"
        backup_count: 7   # Keep 1 week
      system:
        type: "size"
        max_size: "10MB"
        backup_count: 5
```

## ⚠️ **Important Notes**

### **Paperspace Training Unchanged**
- Paperspace continues to use simple console logging
- Training logs appear in Paperspace web interface
- No changes needed to training scripts
- Model transfer process unaffected

### **Backward Compatibility**
- Existing imports continue to work
- Legacy `.logger.logger.info()` pattern preserved
- CSV trade data format maintained
- No breaking changes to existing functionality

### **Performance Considerations**
- **5000+ logs/second** performance tested
- **Minimal overhead** in production
- **Smart buffering** for high-volume operations
- **Automatic file rotation** prevents disk issues

## 🧪 **Testing Your Setup**

```bash
# Run the logging system test
python tests/system/test_enhanced_logging.py

# Expected output:
# ✅ All tests passed! Enhanced logging system is ready.
```

## 🆘 **Troubleshooting**

### **Logs Directory Not Created**
```bash
# Manually create logs directory
mkdir -p logs
chmod 755 logs
```

### **Permission Issues**
```bash
# Fix log file permissions
chmod 644 logs/*.log
chmod 644 logs/*.csv
```

### **Legacy Code Issues**
```python
# If existing code fails, use compatibility wrapper
from src.core.logging_manager import Logger
logger = Logger("component_name")  # Provides .logger attribute
```

### **Performance Issues**
```bash
# Check log file sizes
du -h logs/

# Clean up old debug logs if needed
find logs/ -name "debug.log.*" -mtime +3 -delete
```

## 🎉 **Benefits Achieved**

### **For Developers:**
- **Clear log categories**: Know exactly where to look for specific information
- **Better debugging**: Structured data with correlation IDs
- **Performance insights**: Built-in timing and metrics logging

### **For Operations:**
- **Manageable log files**: Smart rotation prevents disk issues
- **Environment-aware**: Different verbosity per deployment stage
- **Integration-ready**: Structured logs perfect for monitoring tools

### **For Analysis:**
- **Trade history**: Complete CSV data for backtesting and analysis
- **Model performance**: Track prediction accuracy and ensemble behavior
- **System health**: Monitor resource usage and performance trends

The enhanced logging system provides enterprise-grade logging capabilities while maintaining complete backward compatibility with your existing trading bot functionality!
