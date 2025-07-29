# Paper Trading System

A complete paper trading system that uses trained ML models to make trading decisions for cryptocurrencies (BTCEUR, ETHEUR, SOLEUR, XRPEUR, ADAEUR) using Bitvavo's API.

## Features

- **ML-Powered Predictions**: Uses trained XGBoost models to make trading decisions every minute
- **Real-time Data**: Connects to Bitvavo's websocket API with fallback to REST API
- **Risk Management**: Implements configurable take-profit (0.5%) and stop-loss (0.5%) levels
- **Paper Trading**: Simulates trading with realistic fees (0.3%) without real money
- **Telegram Notifications**: Sends trading alerts and hourly portfolio summaries
- **Portfolio Tracking**: Maintains complete trading history and P&L calculations

## Files

- `paper_trader.py` - Main trading engine with ML model integration
- `config.py` - Configuration settings (loads from .env)
- `run_paper_trader.py` - Launcher script
- `setup_telegram.py` - Telegram bot setup and testing
- `paper_trader_requirements.txt` - Required Python packages
- `demo_paper_trader.py` - Complete demonstration script
- `test_paper_trader.py` - Basic functionality test

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r paper_trader_requirements.txt
   ```

2. **Configure Environment**:
   Copy `.env.example` to `.env` and set your API credentials:
   ```bash
   BITVAVO_API_KEY=your_api_key_here
   BITVAVO_API_SECRET=your_api_secret_here
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token  # Optional
   TELEGRAM_CHAT_ID=your_telegram_chat_id      # Optional
   ```

3. **Test Telegram (Optional)**:
   ```bash
   python setup_telegram.py <bot_token> <chat_id>
   ```

4. **Run Demo**:
   ```bash
   python demo_paper_trader.py
   ```

5. **Start Paper Trading**:
   ```bash
   python run_paper_trader.py
   ```

## Configuration

Key settings in `config.py`:
- `SYMBOLS`: Trading pairs ['BTCEUR', 'ETHEUR', 'SOLEUR', 'XRPEUR', 'ADAEUR']
- `INITIAL_BALANCE`: Starting balance in EUR (default: 10000)
- `POSITION_SIZE_PCT`: Position size as % of balance (default: 0.1 = 10%)
- `TAKE_PROFIT_PCT`: Take profit percentage (default: 0.005 = 0.5%)
- `STOP_LOSS_PCT`: Stop loss percentage (default: 0.005 = 0.5%)
- `PREDICTION_INTERVAL`: Prediction frequency in seconds (default: 60)

## Model Requirements

The system expects trained models in the following structure:
- `models/xgboost/{symbol}_xgboost.pkl` - XGBoost model files
- `models/feature_columns/{symbol}_window_15_selected.pkl` - Feature column definitions

Compatible with models trained using the existing `train_hybrid_models.py` script.

## Features Created

The system creates 100+ technical indicators including:
- Price changes (1h, 4h, 24h timeframes)
- Volatility measures (multiple timeframes)
- Moving averages (EMA, SMA)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volume analysis
- Price action patterns
- Market microstructure features

## Risk Management

- **Position Sizing**: Configurable % of portfolio per trade
- **Take Profit**: Automatic exit at profit target
- **Stop Loss**: Automatic exit at loss limit
- **Fees**: Realistic 0.3% trading fees applied
- **Model-based Exits**: Closes positions when model signals sell

## Telegram Notifications

When configured, sends notifications for:
- Bot startup and connection status
- Trade executions (buy/sell with details)
- Hourly portfolio summaries
- Error conditions and reconnections

## Demo Results

The demo script shows a complete trading simulation:
```
=== FINAL RESULTS ===
Final Balance: €9994.17
Final Portfolio Value: €9994.17
Total P&L: €-5.83 (-0.06%)
Total Trades: 2

=== TRADE HISTORY ===
1. BUY: 0.009986 BTC @ €100137.87 (Value: €1000.00, Fees: €3.00)
2. SELL: 0.009986 BTC @ €100155.03 (P&L: €-2.83, Reason: model_signal)
```

## Technical Architecture

- **Async Websocket**: Real-time price updates from Bitvavo
- **API Fallback**: Automatic fallback to REST API if websocket fails
- **Feature Engineering**: Real-time calculation of 100+ technical features
- **ML Integration**: XGBoost classification models for buy/sell decisions
- **Risk Engine**: Sophisticated position management and exit logic
- **Portfolio Tracking**: Complete trade history and performance metrics

## Testing

Run the test suite to verify functionality:
```bash
python test_paper_trader.py  # Basic functionality test
python demo_paper_trader.py  # Complete trading simulation
```

## Support

The system is designed to work with the existing ML training pipeline and uses the same feature engineering approach for consistency.