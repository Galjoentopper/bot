# Paper Trader

A sophisticated paper trading system that uses machine learning models to simulate cryptocurrency trading on the Bitvavo exchange. The system implements real-time data collection, ensemble ML predictions, risk management, and Telegram notifications.

## Features

- **Real-time Data Collection**: 1-minute OHLCV data (default) via Bitvavo WebSocket and REST API
- **Machine Learning Integration**: Supports LSTM and XGBoost ensemble predictions
- **Advanced Risk Management**: 1% take profit/stop loss, trailing stops, time-based exits
- **Portfolio Management**: Track up to 10 concurrent positions with 10% position sizing
- **Telegram Notifications**: Real-time trade alerts and hourly portfolio updates
- **Comprehensive Logging**: CSV exports for trades and portfolio performance
- **Technical Analysis**: 20+ technical indicators for feature engineering

## Project Structure

```
bot/
├── main_paper_trader.py          # Main orchestrator script
├── .env                          # Environment variables (create this)
├── requirements_paper_trader.txt # Dependencies
├── paper_trader/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   └── bitvavo_collector.py  # Real-time data collection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── feature_engineer.py   # Technical indicators
│   │   └── model_loader.py       # ML model integration
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── signal_generator.py   # Trading signal generation
│   │   └── exit_manager.py       # Exit condition management
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── portfolio_manager.py  # Position and P&L tracking
│   ├── notifications/
│   │   ├── __init__.py
│   │   └── telegram_notifier.py  # Telegram integration
│   └── logs/
│       ├── __init__.py
│       ├── paper_trader.log      # System logs (auto-generated)
│       ├── trades.csv            # Trade history (auto-generated)
│       └── portfolio.csv         # Portfolio snapshots (auto-generated)
└── models/                       # Your trained ML models directory
    ├── BTCEUR_lstm_model.h5
    ├── BTCEUR_xgb_model.pkl
    ├── BTCEUR_scaler.pkl
    └── ...
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd bot
pip install -r requirements_paper_trader.txt
```

### 2. Create Environment File

Create a `.env` file in the `bot` directory with your configuration:

```env
# Bitvavo API (for data collection only - no actual trading)
BITVAVO_API_KEY=your_bitvavo_api_key
BITVAVO_API_SECRET=your_bitvavo_api_secret

# Telegram Bot (for notifications)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Trading Parameters
INITIAL_CAPITAL=10000.0
MAX_POSITIONS=10
MAX_POSITIONS_PER_SYMBOL=5
BASE_POSITION_SIZE=0.08
MAX_POSITION_SIZE=0.15
MIN_POSITION_SIZE=0.02
TAKE_PROFIT_PCT=0.015
STOP_LOSS_PCT=0.008
TRAILING_STOP_PCT=0.006
MIN_PROFIT_FOR_TRAILING=0.005
MAX_HOLD_HOURS=24
MIN_HOLD_TIME_MINUTES=10
POSITION_COOLDOWN_MINUTES=5
MAX_DAILY_TRADES_PER_SYMBOL=50
MIN_EXPECTED_GAIN_PCT=0.001

# Symbols to Trade (comma-separated)
SYMBOLS=BTC-EUR,ETH-EUR,ADA-EUR,DOT-EUR,LINK-EUR

# Model Configuration
MODEL_PATH=models
SEQUENCE_LENGTH=96

# Risk Management
MIN_CONFIDENCE=0.7
MIN_VOLUME_RATIO=1.0
MAX_VOLATILITY=0.05
```

### 3. Prepare Your Models

Ensure your trained models are in the `models` directory with the naming convention:
- `{SYMBOL}_lstm_model.h5` (TensorFlow/Keras LSTM model)
- `{SYMBOL}_xgb_model.pkl` (XGBoost model)
- `{SYMBOL}_scaler.pkl` (Feature scaler)
- `{SYMBOL}_feature_columns.pkl` (Feature column names)

Example for BTC-EUR:
- `BTCEUR_lstm_model.h5`
- `BTCEUR_xgb_model.pkl`
- `BTCEUR_scaler.pkl`
- `BTCEUR_feature_columns.pkl`

### 4. Setup Telegram Bot (Optional)

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your bot token
3. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)
4. Add these to your `.env` file

## Running the Paper Trader

```bash
cd bot
python main_paper_trader.py
```

The system will:
1. Initialize all components
2. Load your ML models
3. Start real-time data collection
4. Begin the 1-minute trading cycle (default)
5. Send Telegram notifications for trades and hourly updates

## Key Features Explained

### Trading Strategy
- **Entry**: Based on ensemble ML predictions (LSTM + XGBoost)
- **Position Sizing**: Dynamic 8% base size scaled by confidence and strength
- **Take Profit**: 1.5% above entry price
- **Stop Loss**: 0.8% below entry price
- **Trailing Stop**: 0.6% trailing stop after 0.5% profit
- **Time Exit**: Maximum 24-hour hold time with 10 minute minimum

### Risk Management
- Maximum 10 concurrent positions
- Minimum 70% confidence threshold for trades
- Cooling-off period between trades
- Volume and volatility filters
- Emergency stop conditions

### Data Collection
- Real-time 1-minute OHLCV data (default)
- WebSocket feed for live updates
- Historical data initialization
- Automatic data buffer management

### Feature Engineering
- 20+ technical indicators
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, MFI)
- Support/resistance levels

## Monitoring and Logs

### Log Files
- `paper_trader/logs/paper_trader.log`: System logs
- `paper_trader/logs/trades.csv`: Complete trade history
- `paper_trader/logs/portfolio.csv`: Portfolio snapshots

### Telegram Notifications
- Position opened/closed alerts
- Hourly portfolio updates
- Error alerts
- System status updates

## Performance Tracking

The system tracks:
- Total P&L and percentage returns
- Win rate and trade statistics
- Maximum drawdown
- Average win/loss amounts
- Hold times and exit reasons
- Individual position performance

## Safety Features

- **Paper Trading Only**: No actual money at risk
- **API Key Security**: Read-only data access
- **Error Handling**: Comprehensive exception management
- **Graceful Shutdown**: Proper cleanup on exit
- **Data Validation**: Input validation and sanity checks

## Customization

You can customize:
- Trading symbols in `.env`
- Risk parameters (stop loss, take profit)
- Position sizing and limits
- Technical indicators in `feature_engineer.py`
- Exit conditions in `exit_manager.py`
- Notification messages in `telegram_notifier.py`

## Troubleshooting

### Common Issues

1. **Models not loading**: Check file paths and naming convention
2. **API errors**: Verify Bitvavo API credentials
3. **Telegram not working**: Check bot token and chat ID
4. **Data issues**: Ensure stable internet connection
5. **Memory issues**: Reduce number of symbols or sequence length

### Debug Mode

Set logging level to DEBUG in `main_paper_trader.py`:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Disclaimer

This is a paper trading system for educational and testing purposes only. Past performance does not guarantee future results. Always test thoroughly before considering any real trading implementation.

## Support

Check the logs for detailed error messages and system status. The Telegram notifications will alert you to any critical issues during operation.