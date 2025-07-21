# 🤖 Cryptocurrency Trading Bot

A sophisticated machine learning-powered cryptocurrency trading bot that uses ensemble predictions and advanced risk management for automated trading on the Bitvavo exchange.

## 🌟 Features

- **🧠 Machine Learning**: Hybrid LSTM + XGBoost ensemble models for price prediction
- **📊 Real-time Data**: Live data collection via Bitvavo WebSocket and REST API
- **🛡️ Risk Management**: Advanced position sizing, stop-loss, take-profit, and trailing stops
- **📱 Telegram Integration**: Real-time notifications and portfolio updates
- **📈 Technical Analysis**: 20+ technical indicators for comprehensive market analysis
- **🔄 Walk-Forward Validation**: Realistic backtesting with out-of-sample performance
- **📋 Comprehensive Logging**: Detailed trade logging and performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Bitvavo API credentials
- Telegram Bot token (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Galjoentopper/bot.git
cd bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

4. Run the paper trader:
```bash
python main_paper_trader.py
```

## 📂 Project Structure

```
bot/
├── 📊 data/                          # Market data storage
├── 🧠 models/                        # Trained ML models
├── 📋 logs/                          # Performance metrics and logs
├── 🏗️ paper_trader/                  # Core trading system
│   ├── config/                       # Configuration management
│   ├── data/                         # Data collection modules
│   ├── models/                       # ML model handlers
│   ├── strategy/                     # Trading strategy logic
│   ├── portfolio/                    # Portfolio management
│   └── notifications/                # Telegram notifications
├── 🧪 tests/                         # Test suites
├── 📚 docs/                          # Detailed documentation
├── main_paper_trader.py              # Main trading application
├── train_hybrid_models.py            # ML model training
├── run_backtest.py                   # Backtesting system
└── binance_data_collector.py         # Historical data collection
```

## 🔧 Configuration

Key configuration options in `.env`:

- **Trading**: Initial capital, position sizing, risk parameters
- **Models**: Confidence thresholds, signal strength requirements
- **API**: Bitvavo credentials and endpoints
- **Notifications**: Telegram bot configuration

## 📖 Documentation

Detailed documentation is available in the `docs/` folder:

- **[Training Guide](docs/README_TRAINING.md)**: ML model training and optimization
- **[Backtest Guide](docs/README_BACKTEST.md)**: Backtesting and performance analysis
- **[Data Collection](docs/README_DATA_COLLECTOR.md)**: Market data acquisition
- **[Optimization](docs/README_OPTIMIZATION.md)**: Parameter optimization strategies
- **[Paper Trader](docs/README_paper_trader.md)**: Live trading system details

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific test modules:
```bash
python test_bot_fixes.py
python test_trading_decisions.py
```

## 🛡️ Risk Management

The bot implements multiple layers of risk protection:

- **Position Sizing**: Configurable position sizes with maximum limits
- **Stop Loss**: Automatic stop-loss orders at configurable percentages
- **Take Profit**: Profit-taking at predetermined levels
- **Trailing Stops**: Dynamic stop-loss adjustment
- **Time-based Exits**: Maximum holding period enforcement
- **Daily Loss Limits**: Maximum daily loss thresholds

## 📊 Supported Cryptocurrencies

Default supported pairs (configurable):
- BTC-EUR
- ETH-EUR
- ADA-EUR
- SOL-EUR
- XRP-EUR

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## ⚠️ Disclaimer

This is educational software for paper trading only. Always conduct thorough testing before considering any real trading. Cryptocurrency trading involves significant risk.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.