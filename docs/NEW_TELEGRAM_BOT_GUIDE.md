# New Telegram Bot System Guide
## Focused, Reliable Trading Bot Notifications

This guide covers the new, simplified Telegram bot system designed specifically for your trading bot requirements.

## 🎯 **What This Bot Does**

### **✅ Automatic Notifications:**
1. **🚀 Startup Message** - Shows symbols traded and models loaded
2. **📈 Daily Performance Report** - Sent automatically at 14:00 UTC
3. **🛑 Shutdown Message** - Confirms bot stopped with session summary
4. **🚨 Error Notifications** - Detailed error messages with context

### **✅ Working Commands:**
- `/help` - Show all available commands
- `/status` - Current system status and positions
- `/balance` - Portfolio overview and performance
- `/trades` - Recent trades (last 10)
- `/health` - System health check
- `/models` - Model status and performance
- `/uptime` - Bot runtime information
- `/symbols` - Trading symbols and current prices
- `/performance` - Extended performance statistics
- `/report` - Generate performance report now
- `/pause` - Pause trading (stop new trades)
- `/resume` - Resume trading

## 🚀 **Quick Start**

### **1. Set Environment Variables**
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

### **2. Test the System**
```bash
# Test configuration and functionality
python tests/system/test_telegram_bot.py
```

### **3. Start the Bot**
```bash
# Start Telegram bot
python bin/telegram_bot
```

### **4. Test Commands**
Send `/help` to your bot to see available commands, then try:
- `/status` - Check if everything is working
- `/balance` - See your portfolio
- `/health` - System health check

## 📊 **Message Examples**

### **Startup Message**
```
🚀 Trading Bot Started
━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Symbols: BTCEUR, ETHEUR, DOTEUR, ADAEUR, LINKEUR
🤖 Models Loaded: 15 total
  ├── GRU: 5 models
  ├── LightGBM: 5 models
  └── PPO: 5 models
💰 Initial Balance: €10,000
⏰ Started: 2025-09-12 14:30:15 UTC
━━━━━━━━━━━━━━━━━━━━━━━━━
```

### **Daily Performance Report (14:00 UTC)**
```
📈 Daily Performance Report
━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Date: 2025-09-12
💰 Portfolio: €10,247 (+2.47%)
📊 Trades: 12 executed (8 profitable)
🎯 Best: BTCEUR +€89 (0.89%)
📉 Worst: ETHEUR -€23 (-0.23%)

🏆 Top Performers:
  BTCEUR: +€89 (3 trades)
  DOTEUR: +€45 (2 trades)

⚠️ Positions:
  ETHEUR: €1,234 (12.3%)
  BTCEUR: €2,156 (21.5%)
━━━━━━━━━━━━━━━━━━━━━━━━━
```

### **Error Notification**
```
🚨 ERROR: FileNotFoundError
━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Time: 2025-09-12 15:23:17 UTC
🎯 Context: Model prediction
❌ Error: Model file not found
📍 Location: lgbm_trainer.py:145
💡 Action: System continues running
━━━━━━━━━━━━━━━━━━━━━━━━━
```

### **Shutdown Message**
```
🛑 Trading Bot Stopped
━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Stopped: 2025-09-12 18:45:32 UTC
💰 Final Balance: €10,247
📊 Session: 12 trades, +€247 profit
🕐 Runtime: 4h 15m
━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 🔧 **Integration with Trading System**

### **Automatic Integration**
The bot automatically integrates with your trading system when you pass the trader instance:

```python
# In your main trader script
from src.notifications.telegram_integration import integrate_telegram_with_trader

# Create Telegram integration
telegram_integration = integrate_telegram_with_trader(trader_instance)

# Initialize and start
await telegram_integration.initialize_telegram(bot_token, chat_id)
await telegram_integration.start_telegram_bot()

# The integration will automatically:
# - Send startup message when trader starts
# - Record all trades for daily stats
# - Send error notifications
# - Send shutdown message when stopped
```

### **Trade Notifications**
The bot intelligently sends individual trade notifications for:
- **High confidence trades** (>80% confidence)
- **Large trades** (>€500 value)
- **Significant P&L** (>€50 profit/loss)

Other trades are batched into the daily performance report.

## ⏰ **Daily Performance Report**

**When**: Every day at **14:00 UTC** (3:00 PM German time)

**What's Included**:
- Current portfolio value and daily return
- Number of trades executed and profitable trades
- Best and worst performing trades
- Top performing symbols
- Current position overview

**Manual Trigger**: Send `/report` command to generate immediately

## 🔍 **Command Details**

### **Status Commands**
| Command | Description | Example Response |
|---------|-------------|------------------|
| `/status` | Current system overview | Shows balance, P&L, active positions, models |
| `/balance` | Portfolio breakdown | Total, cash, positions, returns |
| `/trades` | Recent trades (last 10) | Trade details with P&L and timestamps |
| `/health` | System health check | Memory usage, CPU, model status |

### **Information Commands**
| Command | Description | Example Response |
|---------|-------------|------------------|
| `/models` | Model status and performance | Count, accuracy, last predictions |
| `/uptime` | Bot runtime | Start time, duration, trades today |
| `/symbols` | Trading symbols and prices | Current prices and 24h changes |
| `/performance` | Extended performance stats | Same as daily report |

### **Control Commands**
| Command | Description | Notes |
|---------|-------------|-------|
| `/help` | Show all commands | Always works |
| `/report` | Generate report now | Manual trigger for daily report |
| `/pause` | Pause trading | Stops new trades, existing positions remain |
| `/resume` | Resume trading | Re-enables trade execution |

## 🛡️ **Error Handling**

### **Automatic Error Notifications**
The bot automatically sends detailed error messages when:
- Model predictions fail
- Data fetching issues occur
- Trading execution problems happen
- System errors are encountered

### **Error Message Format**
- **Error type** and description
- **Exact time** of occurrence
- **Context** where error happened
- **Location** in code (file and line)
- **Action taken** (system continues/stops)

## 🧪 **Testing & Validation**

### **Test the Bot**
```bash
# Run comprehensive test suite
python tests/system/test_telegram_bot.py

# Expected output:
# ✅ Configuration validated
# ✅ Bot creation successful
# ✅ Startup message working
# ✅ Commands functional
# ✅ Error notifications working
# 🎉 All tests passed!
```

### **Manual Testing**
1. Start the bot: `python bin/telegram_bot`
2. Check you receive startup message
3. Try commands: `/help`, `/status`, `/balance`
4. Wait for daily report at 14:00 UTC (or use `/report`)
5. Stop the bot and check for shutdown message

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# Required
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
export TELEGRAM_CHAT_ID="123456789"

# Optional
export TRADING_ENV="production"  # Affects log levels
export DEBUG_MODE="false"        # Enable debug logging
```

### **Getting Bot Token and Chat ID**
1. **Bot Token**: Message @BotFather on Telegram, create new bot
2. **Chat ID**: Message @userinfobot or check message JSON

### **Integration with Trading Config**
The bot automatically reads from your `training_config.yaml`:
- Trading symbols from `symbols` section
- Model configuration for counts
- Any other trading parameters

## 🔧 **Architecture**

### **Clean, Simple Design**
```
src/notifications/
├── telegram_bot.py              # Main bot implementation
├── telegram_integration.py      # Trading system integration
└── __init__.py

bin/telegram_bot                 # Launcher script
tests/system/test_telegram_bot.py # Test suite
docs/NEW_TELEGRAM_BOT_GUIDE.md   # This guide
```

### **Key Features**
- **Single Implementation** - No more conflicting telegram scripts
- **Enhanced Logging Integration** - Uses your new logging system
- **Async/Await** - Proper async implementation for performance
- **Error Recovery** - Retry logic and graceful error handling
- **Memory Efficient** - Minimal resource usage
- **Production Ready** - Designed for 24/7 operation

## 🚀 **Migration from Old System**

### **What's Removed**
The old, complex system with multiple conflicting implementations has been replaced with this single, focused bot.

### **What's Preserved**
- All essential functionality
- Configuration compatibility
- Integration with trading system
- Essential commands and notifications

### **What's Improved**
- **Reliability** - Commands actually work
- **Simplicity** - Single implementation, no conflicts
- **Performance** - Proper async design
- **Logging** - Integrated with enhanced logging system
- **Maintenance** - Much easier to debug and modify

## 🎯 **Benefits**

### **For You**
- **Reliable notifications** - Always know what your bot is doing
- **Clear status info** - Commands that actually work and provide useful information
- **Error awareness** - Detailed error notifications help you fix issues quickly
- **Daily insights** - Automatic performance reports keep you informed

### **For System**
- **Clean architecture** - Single file, no conflicts
- **Better performance** - Proper async implementation
- **Easier maintenance** - Simple, focused codebase
- **Enhanced logging** - Integrated with your new logging system

Your new Telegram bot is designed to be **simple, reliable, and focused** on exactly what you need for trading bot notifications!
