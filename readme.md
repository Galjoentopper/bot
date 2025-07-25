# 🤖 Cryptocurrency Trading Bot

A sophisticated machine learning-powered cryptocurrency trading bot that uses ensemble predictions and advanced risk management for automated trading on the Bitvavo exchange.

## 🌟 Features

- **Machine Learning Prediction**: Ensemble model combining LSTM and XGBoost for price forecasting
- **Real-time Trading**: Paper trading with Bitvavo API integration
- **Advanced Risk Management**: Dynamic position sizing and stop-loss mechanisms
- **Telegram Notifications**: Real-time alerts for trades and performance updates
- **Performance Tracking**: Comprehensive logging and portfolio analysis

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
│   ├── BTCEUR_lstm_model.h5          # LSTM model for BTC-EUR
│   ├── BTCEUR_xgb_model.pkl          # XGBoost model for BTC-EUR
│   ├── BTCEUR_scaler.pkl             # Feature scaler for BTC-EUR
│   └── BTCEUR_feature_columns.pkl    # Feature column names
├── 📋 logs/                          # Performance metrics and logs
├── 🏗️ paper_trader/                  # Core trading system
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py               # Trading settings
│   ├── data/                         # Data collection modules
│   │   ├── __init__.py
│   │   └── bitvavo_collector.py      # Real-time data collection
│   ├── models/                       # ML model handlers
│   │   ├── __init__.py
│   │   ├── feature_engineer.py       # Technical indicators
│   │   └── model_loader.py           # ML model integration
│   ├── strategy/                     # Trading strategy logic
│   │   ├── __init__.py
│   │   ├── signal_generator.py       # Trading signal generation
│   │   └── exit_manager.py           # Exit condition management
│   ├── portfolio/                    # Portfolio management
│   │   ├── __init__.py
│   │   └── portfolio_manager.py      # Position and P&L tracking
│   ├── notifications/                # Notification systems
│   │   ├── __init__.py
│   │   ├── telegram_notifier.py      # Telegram integration
│   │   └── websocket_server.py       # Websocket notifications
│   └── logs/                         # System logs
│       ├── __init__.py
│       ├── paper_trader.log          # System logs (auto-generated)
│       ├── trades.csv                # Trade history (auto-generated)
│       └── portfolio.csv             # Portfolio snapshots (auto-generated)
├── 📚 docs/                          # Detailed documentation
│   ├── README_TRAINING.md            # ML model training guide
│   ├── README_BACKTEST.md            # Backtesting guide
│   ├── README_DATA_COLLECTOR.md      # Market data acquisition guide
│   ├── README_OPTIMIZATION.md        # Parameter optimization guide
│   └── README_paper_trader.md        # Live trading system details
├── main_paper_trader.py              # Main trading application
├── enhanced_main_paper_trader.py     # Enhanced trading application
├── train_hybrid_models.py            # ML model training
├── run_backtest.py                   # Backtesting system
└── binance_data_collection.py        # Historical data collection
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

## 🧪 Main Scripts

### main_paper_trader.py
The primary script that orchestrates the entire paper trading system. It:
- Initializes all trading components
- Loads ML models
- Starts real-time data collection from Bitvavo
- Executes the 1-minute trading cycle
- Manages portfolio and risk
- Sends notifications via Telegram

```bash
python main_paper_trader.py
```

### enhanced_main_paper_trader.py
An improved version of the paper trader with:
- Enhanced performance monitoring
- Circuit breakers for error handling
- Data quality monitoring
- Health monitoring systems
- Improved notification management

```bash
python enhanced_main_paper_trader.py
```

### train_hybrid_models.py
Script for training the hybrid machine learning models (LSTM + XGBoost) used for price prediction.

```bash
python train_hybrid_models.py
```

### run_backtest.py
Comprehensive backtesting system to evaluate trading strategies using historical data.

```bash
python run_backtest.py
```

### binance_data_collection.py
Script for collecting historical price data from Binance exchange for model training and backtesting.

```bash
python binance_data_collection.py
```

### optimized_variables.py
Scientific parameter optimization system that uses Bayesian optimization to find the best trading parameters.

```bash
python optimized_variables.py --symbols BTCEUR ETHEUR
```

## 🛡️ Risk Management

The system includes several risk management features:
- Dynamic position sizing based on volatility
- Automated stop-loss mechanisms
- Maximum position limits
- Circuit breakers for technical failures

## 📊 Supported Cryptocurrencies

The bot supports multiple cryptocurrency pairs on Bitvavo, including:
- BTC-EUR
- ETH-EUR
- And other major pairs (configurable in settings)

## 🧪 Testing

The project includes a comprehensive test suite to ensure reliability:

```bash
# Run core tests (recommended)
python run_tests.py core

# Run specific test categories
python run_tests.py config      # Configuration tests
python run_tests.py portfolio   # Portfolio management tests

# Run all tests
python run_tests.py all -v
```

For detailed testing information, see [docs/TESTING.md](docs/TESTING.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

When contributing:
1. Run the test suite: `python run_tests.py core`
2. Add tests for new functionality
3. Update documentation as needed

## ⚠️ Disclaimer

This software is for educational and research purposes only. Use at your own risk.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.