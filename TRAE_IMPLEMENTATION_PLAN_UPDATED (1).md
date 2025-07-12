# 🚀 Trading Bot 2.0 – Project Plan (Træ-Compatible)

## 📊 Data Strategy

**Primary Timeframe**: 15-minute candles only
✅ **Historical Range**: 2020-01-01 to present (~4+ years)
✅ **Storage Approach**: Keep 15m data 
✅ **Estimated Size**: ~140k records per coin (manageable for local storage)
5 coins in total: BTC/EUR ETH/EUR XRP/EUR ADA/EUR SOL/EUR

## 🔌 API Configuration

**Data Sources:**
✅ **Paper Trading**: Use **Bitvavo API** for real-time market data
- Real-time price feeds via WebSocket
- Order book data for accurate spread simulation
- Live market depth for realistic execution modeling

✅ **Historical Data Collection**: Use **Binance API** for bulk data loading
- Historical OHLCV data (15-minute timeframe only)
- Backfill from 2020-01-01 to present
- High-quality, reliable historical data
- No API credentials required (public data access)
used to train the model

**API Credentials Required:**
- `BITVAVO_API_KEY` and `BITVAVO_API_SECRET` (for paper trading only)

---

## 🤖 Træ Prompt Refinement (Confirmed Settings)

**1. LSTM Sequence Length**  
✅ Use **60 steps** = 15-minute candles × 60 = last **15 hours** of data  
> Reason: captures intraday and half-day price momentum

---

**2. XGBoost Features (Expanded Set)**  
✅ Include the following technical and sentiment-based features:

### 📈 Technical Indicators
- RSI (14)
- MACD (12,26,9)
- EMA (9, 21, 50)
- SMA (21, 50)
- Bollinger Bands (%B)
- ADX (trend strength)
- Stochastic Oscillator (%K, %D)
- ATR (volatility)
- OBV (volume flow)
- CCI (momentum)
- Volume change (%)

### 🧠 Sentiment Indicators
- Twitter polarity score (via FinBERT or Vader)
- Reddit post polarity (rolling average)
- News sentiment (CoinDesk headlines)
- Market-wide Fear & Greed Index (if available)

### 🔌 Model-Derived
- `lstm_delta` (predicted price movement from LSTM)

---

**3. Buy/Sell Thresholds**  
✅ Trade only if:
- **Buy**: `confidence > 0.7`  
- **Sell**: `confidence < 0.3`  
> Hold otherwise

---

**4. Risk Rules**  
✅ Max position size: **20% per coin**  
✅ Add stop-loss: **recommend -4%** per open position  
> Reason: balances trade agility and noise tolerance in volatile markets

---

**5. Model Selection Metric**  
✅ Use **Sharpe Ratio** for final model selection  
> Reason: balances return vs. volatility; perfect for portfolio comparisons

---

**6. Portfolio Allocation**  
✅ Use **Dynamic Position Sizing**  
> Allocate more capital to higher-confidence signals  
> Optionally scale by recent volatility or prediction certainty

---

**7. Telegram Alerts**  
✅ Each message should include:
- **Coin**: e.g., BTC-EUR  
- **Action**: BUY / SELL / HOLD  
- **Confidence**: model probability (e.g., 87%)  
- **Price**: current market price  
- **Position Size**: % allocated  
- **Portfolio Summary**:
  - Total equity (virtual)
  - Cash balance
  - Unrealized PnL
- **Daily Stats**:
  - Win rate today
  - Sharpe ratio (rolling)
  - Number of trades

> Weekly summary can include top performers, drawdowns, and PnL charts.

---

## 🏗️ Technical Implementation Guide

### 📁 Project Structure
```
trade_bot_2.0/
├── src/
│   ├── data/
│   │   ├── collectors/
│   │   │   ├── binance_collector.py    # Historical 15m data collection
│   │   │   └── bitvavo_collector.py    # Real-time data for paper trading
│   │   ├── processors/
│   │   │   ├── technical_indicators.py # RSI, MACD, EMA, etc.
│   │   │   └── sentiment_analyzer.py   # Twitter, Reddit, News sentiment
│   │   └── database.py                 # PostgreSQL connection & schemas
│   ├── models/
│   │   ├── lstm_model.py              # 60-step sequence LSTM
│   │   ├── xgboost_model.py           # Feature-rich XGBoost
│   │   └── ensemble.py                # Model combination logic
│   ├── trading/
│   │   ├── strategy.py                # Buy/sell logic with thresholds
│   │   ├── portfolio.py               # Dynamic position sizing
│   │   ├── risk_manager.py            # Stop-loss, position limits
│   │   └── paper_trader.py            # Bitvavo paper trading simulation
│   ├── monitoring/
│   │   ├── telegram_bot.py            # Alert system
│   │   └── metrics.py                 # Sharpe ratio, performance tracking
│   └── utils/
│       ├── config.py                  # API keys, settings
│       └── gui.py                     # Tkinter interface
├── setup.py                           # Interactive installation
├── requirements.txt                   # Dependencies
├── config.yaml                        # Configuration file
└── README.md                          # Setup instructions
```

### 🗄️ Database Schema (PostgreSQL)

**Core Tables:**
```sql
-- 15-minute OHLCV data
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20,8),
    high DECIMAL(20,8),
    low DECIMAL(20,8),
    close DECIMAL(20,8),
    volume DECIMAL(20,8),
    source VARCHAR(20) DEFAULT 'binance',
    UNIQUE(symbol, timestamp)
);

-- Model predictions with features
CREATE TABLE model_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    model_type VARCHAR(20) NOT NULL, -- 'lstm', 'xgboost', 'ensemble'
    prediction DECIMAL(10,6),
    confidence DECIMAL(5,4),
    features JSONB, -- Store all technical + sentiment features
    lstm_delta DECIMAL(10,6)
);

-- Trade execution log
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    trade_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'STOP_LOSS'
    quantity DECIMAL(20,8),
    price DECIMAL(20,8),
    confidence DECIMAL(5,4),
    strategy VARCHAR(50),
    is_paper_trade BOOLEAN DEFAULT true
);

-- Portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    total_equity DECIMAL(20,8),
    cash_balance DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    positions JSONB, -- {"BTC-EUR": {"quantity": 0.5, "avg_price": 45000}}
    daily_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6)
);

-- Sentiment data
CREATE TABLE sentiment_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    timestamp TIMESTAMP NOT NULL,
    source VARCHAR(20) NOT NULL, -- 'twitter', 'reddit', 'news', 'fear_greed'
    sentiment_score DECIMAL(5,4),
    text_content TEXT,
    url VARCHAR(500)
);
```

**Essential Indexes:**
```sql
CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_predictions_symbol_time ON model_predictions(symbol, timestamp DESC);
CREATE INDEX idx_trades_symbol_time ON trades(symbol, timestamp DESC);
CREATE INDEX idx_portfolio_time ON portfolio_snapshots(timestamp DESC);
CREATE INDEX idx_sentiment_symbol_time ON sentiment_data(symbol, timestamp DESC);
```

### 🔧 Key Implementation Classes

**1. Data Collection Pipeline**
```python
class BinanceCollector:
    def collect_historical_15m(self, symbol: str, start_date: str) -> pd.DataFrame
    def validate_data_integrity(self, df: pd.DataFrame) -> bool
    def store_to_database(self, df: pd.DataFrame) -> None

class BitvavoCollector:
    def connect_websocket(self) -> None
    def get_real_time_price(self, symbol: str) -> float
    def get_order_book(self, symbol: str) -> dict
```

**2. Feature Engineering**
```python
class TechnicalIndicators:
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series
    def calculate_macd(self, prices: pd.Series) -> tuple
    def calculate_bollinger_bands(self, prices: pd.Series) -> tuple
    def get_all_features(self, df: pd.DataFrame) -> pd.DataFrame

class SentimentAnalyzer:
    def get_twitter_sentiment(self, symbol: str) -> float
    def get_reddit_sentiment(self, symbol: str) -> float
    def get_fear_greed_index(self) -> float
```

**3. Model Training & Prediction**
```python
class LSTMModel:
    def __init__(self, sequence_length: int = 60)
    def prepare_sequences(self, df: pd.DataFrame) -> tuple
    def train(self, X: np.array, y: np.array) -> None
    def predict(self, sequence: np.array) -> float

class XGBoostModel:
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame
    def train(self, X: pd.DataFrame, y: pd.Series) -> None
    def predict_with_confidence(self, X: pd.DataFrame) -> tuple
```

**4. Trading Strategy**
```python
class TradingStrategy:
    def __init__(self, buy_threshold: float = 0.7, sell_threshold: float = 0.3)
    def should_buy(self, confidence: float, current_position: float) -> bool
    def should_sell(self, confidence: float, current_position: float) -> bool
    def calculate_position_size(self, confidence: float, portfolio_value: float) -> float

class RiskManager:
    def __init__(self, max_position_pct: float = 0.2, stop_loss_pct: float = 0.04)
    def check_position_limits(self, symbol: str, new_quantity: float) -> bool
    def check_stop_loss(self, symbol: str, current_price: float) -> bool
```

**5. Performance Monitoring**
```python
class PerformanceTracker:
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float
    def calculate_win_rate(self, trades: list) -> float
    def generate_daily_summary(self) -> dict

class TelegramBot:
    def send_trade_alert(self, trade_info: dict) -> None
    def send_daily_summary(self, summary: dict) -> None
    def send_weekly_report(self, report: dict) -> None
```

### ⚙️ Configuration Management

**config.yaml structure:**
```yaml
api_credentials:
  bitvavo:
    api_key: "${BITVAVO_API_KEY}"
    api_secret: "${BITVAVO_API_SECRET}"

trading_params:
  buy_threshold: 0.7
  sell_threshold: 0.3
  max_position_pct: 0.2
  stop_loss_pct: 0.04
  
model_params:
  lstm_sequence_length: 60
  technical_indicators:
    - rsi_14
    - macd_12_26_9
    - ema_9_21_50
    - bollinger_bands
    - adx
    - stochastic
    - atr
    - obv
    - cci
    - volume_change
  
data_collection:
  symbols: ["BTC-EUR", "ETH-EUR", "ADA-EUR"]
  start_date: "2020-01-01"
  timeframe: "15m"
  
telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"
```

### 🚀 Setup & Deployment

**setup.py features:**
- Interactive API credential input
- Database initialization
- Dependency installation
- Configuration file generation
- Initial data collection
- Model training pipeline setup

**GUI Features (Tkinter):**
- Setup wizard for first-time configuration
- Real-time portfolio monitoring
- Manual trading controls
- Model training progress
- Performance charts
- Alert management

This comprehensive plan provides clear implementation guidance for transforming the trading strategy into a fully functional program with proper architecture, database design, and modular components.