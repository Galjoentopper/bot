# 🚀 Walk-Forward Hybrid LSTM + XGBoost Model Training

This directory contains a comprehensive machine learning pipeline for training cryptocurrency trading models using a hybrid approach that combines LSTM (Long Short-Term Memory) neural networks with XGBoost gradient boosting, implemented with **walk-forward validation** for realistic out-of-sample performance estimation.

## 🏗️ Architecture Overview

### 🧠 Two-Model Hybrid System

```
📊 Raw OHLCV Data
       ↓
🔧 Feature Engineering (Technical Indicators)
       ↓
🧠 LSTM Model → lstm_delta (price change prediction)
       ↓
🌲 XGBoost Model (lstm_delta + features) → Final Trade Signal
```

### 🎯 Model Responsibilities

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **LSTM** | 60 candles of [price, volume] | `lstm_delta` (% price change) | Learn temporal patterns |
| **XGBoost** | Technical indicators + `lstm_delta` | Buy/Sell signal (binary) | Final trading decision |

## 📁 Project Structure

```
trade_bot_2.0/
├── 📊 data/                          # SQLite databases (from binance_data_collector.py)
│   ├── btceur_15m.db
│   ├── etheur_15m.db
│   ├── adaeur_15m.db
│   ├── soleur_15m.db
│   └── xrpeur_15m.db
├── 🤖 models/                        # Trained models (created by training)
│   ├── lstm/                         # LSTM models (.h5 files)
│   ├── xgboost/                      # XGBoost models (.pkl files)
│   └── scalers/                      # Data scalers (.pkl files)
├── 📊 results/                       # Training results and analysis
│   ├── training_summary.json
│   ├── {symbol}_evaluation.json
│   └── {symbol}_feature_importance.csv
├── 🔧 train_hybrid_models.py         # Main training script
├── 📋 requirements_ml.txt            # ML dependencies
└── 📖 README_TRAINING.md             # This file
```

## 🛠️ Installation

### 1. Install Dependencies

```bash
# Install ML requirements
pip install -r requirements_ml.txt

# Note: This project uses pandas-ta for technical analysis,
# which is easier to install than TA-Lib and provides the same functionality.
```

### 2. Ensure Data is Available

Make sure you've run the data collector first:

```bash
python binance_data_collector.py
```

## 🚀 Usage

### Quick Start

```bash
# Train models for all cryptocurrency pairs
python train_hybrid_models.py

# Use custom data and model directories with warm start
python train_hybrid_models.py --data-dir ./data --models-dir ./models --warm-start
```

### Command Line Options

- `--data-dir`: location of the SQLite files
- `--models-dir`: where trained models are saved
- `--warm-start`: load the previous window's model before training

### Walk-Forward Training Process

The script implements **walk-forward validation** with a sliding window approach:

#### 🔄 Walk-Forward Configuration
- **Training Window**: 6 months of data
- **Test Window**: 1 month of data  
- **Step Size**: 1 month (monthly retraining)
- **Minimum Samples**: 10,000 for training, 1,000 for LSTM sequences

#### 📊 Step 1: Data Preparation
- Loads 15-minute OHLCV data from SQLite databases
- Creates 30+ technical indicators:
  - **Price**: Returns, price changes (1h, 4h)
  - **Volume**: Volume ratios, changes
  - **Volatility**: ATR, rolling standard deviation
  - **Moving Averages**: EMA(9,21,50), SMA(200)
  - **Oscillators**: RSI, MACD, Bollinger Bands
  - **Market Structure**: VWAP, support/resistance levels
  - **Time Features**: Hour, day of week, weekend indicator
- Generates walk-forward time windows

#### 🔄 Step 2: Walk-Forward Loop
For each time window:

##### 🧠 LSTM Training
- **Input**: 60-step sequences of [price, volume] from training window
- **Architecture**: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(32) → Dense(1)`
- **Target**: Next 15-minute price change percentage
- **Split**: 80% train, 20% validation within window
- **Output**: `lstm_delta` predictions for both training and test periods

##### 🌲 XGBoost Training
- **Input**: Technical indicators + `lstm_delta` from training window
- **Target**: Binary classification (price up > 0.5%)
- **Features**: 25+ engineered features including LSTM output
- **Split**: 80% train, 20% validation within window
- **Optimization**: Reduced estimators for faster training

##### 📊 Window Evaluation
- **LSTM Metrics**: MAE, RMSE on test window
- **XGBoost Metrics**: Accuracy, AUC on test window
- **Aggregation**: Results collected across all windows

#### 📈 Step 3: Final Model Persistence
- Saves models from the **last window** (most recent)
- Generates feature importance from final XGBoost model
- Creates comprehensive walk-forward performance reports

## 📊 Expected Output

### Console Output
```
================================================================================
🚀 WALK-FORWARD HYBRID LSTM + XGBOOST TRAINING PIPELINE
================================================================================
📊 Symbols: BTCEUR, ETHEUR, ADAEUR, SOLEUR, XRPEUR
📅 Walk-Forward Config: 6M train → 1M test (step: 1M)
⏰ Started at: 2025-01-11 15:30:45

============================================================
🚀 Walk-Forward Training for BTCEUR
============================================================

📊 Data Preparation
📊 Loaded 95,847 candles for BTCEUR
📅 Date range: 2020-01-01 to 2025-01-11
✅ Created 32 technical features
🔄 Generated 54 walk-forward windows

🔄 Window 1/54: 2020-01-01 to 2020-08-01
🧠 LSTM Training: 17,280 sequences
🌲 XGBoost Training: 17,219 samples
📊 Window 1 Results:
   LSTM: MAE=0.008234, RMSE=0.015678
   XGBoost: Accuracy=0.6789, AUC=0.7456

🔄 Window 2/54: 2020-02-01 to 2020-09-01
...

🔄 Window 54/54: 2024-07-01 to 2025-01-01
📊 Window 54 Results:
   LSTM: MAE=0.007891, RMSE=0.014523
   XGBoost: Accuracy=0.7123, AUC=0.7834

✅ Final models saved for BTCEUR
🔍 Top 5 features: ['lstm_delta', 'rsi', 'macd_histogram', 'price_vs_ema9', 'volume_ratio']

📊 BTCEUR Summary (54 windows):
   LSTM: MAE=0.008012±0.000456, RMSE=0.015234±0.000789
   XGBoost: Accuracy=0.6934±0.0234, AUC=0.7612±0.0187
⏱️  Completed in 23.4 minutes

================================================================================
🎉 WALK-FORWARD TRAINING COMPLETED!
================================================================================
⏰ Total time: 187.3 minutes
✅ Successful: 5/5 symbols
📊 Total windows processed: 270

📈 Overall Performance:
   LSTM: MAE=0.008156±0.000523
   XGBoost: Accuracy=0.6891±0.0198, AUC=0.7534±0.0156

📁 Results saved to: results/walkforward_results_20250111_183045.json
```

### Generated Files

#### Models Directory
```
models/
├── lstm/
│   ├── btceur_lstm.h5              # Trained LSTM model
│   ├── btceur_history.pkl          # Training history
│   └── ... (other symbols)
├── xgboost/
│   ├── btceur_xgboost.pkl          # Trained XGBoost model
│   └── ... (other symbols)
└── scalers/
    ├── btceur_scaler.pkl           # Data scaler for LSTM
    └── ... (other symbols)
```

#### Results Directory
```
results/
├── walkforward_results_20250111_183045.json  # Comprehensive walk-forward results
├── btceur_feature_importance.csv             # Feature rankings from final model
├── etheur_feature_importance.csv             # Feature rankings from final model
└── ... (other symbols)
```

#### Walk-Forward Results Structure
```json
{
  "timestamp": "20250111_183045",
  "training_type": "walk_forward",
  "config": {
    "train_months": 6,
    "test_months": 1,
    "step_months": 1,
    "min_training_samples": 10000
  },
  "total_time_minutes": 187.3,
  "symbols_trained": 5,
  "total_windows": 270,
  "aggregated_metrics": {
    "avg_lstm_mae": 0.008156,
    "avg_lstm_rmse": 0.015234,
    "avg_xgb_accuracy": 0.6891,
    "avg_xgb_auc": 0.7534,
    "std_lstm_mae": 0.000523,
    "std_xgb_accuracy": 0.0198
  },
  "results_by_symbol": {
    "BTCEUR": [
      {
        "window": 1,
        "train_start": "2020-01-01",
        "train_end": "2020-07-01",
        "test_end": "2020-08-01",
        "lstm_mae": 0.008234,
        "lstm_rmse": 0.015678,
        "xgb_accuracy": 0.6789,
        "xgb_auc": 0.7456
      },
      // ... more windows
    ]
  }
}
```

## 🔧 Configuration

### Key Parameters (in `train_hybrid_models.py`)

```python
# Walk-Forward Configuration
train_months = 6                    # 6 months training window
test_months = 1                     # 1 month test window
step_months = 1                     # 1 month step size (monthly retraining)
min_training_samples = 10000        # Minimum samples for training

# LSTM Configuration
lstm_sequence_length = 60           # 15 hours of 15-min candles
prediction_horizon = 1              # Next 15-min candle

# XGBoost Configuration
price_change_threshold = 0.005      # 0.5% for binary classification
n_estimators = 100                  # Reduced for faster training
early_stopping_rounds = 10          # Reduced for faster training

# Within-Window Data Split
train_ratio = 0.8                   # 80% training within each window
val_ratio = 0.2                     # 20% validation within each window
```

## 🧠 Why Walk-Forward Validation?

### 🎯 Advantages Over Static Train/Test Split

| Method | Pros | Cons |
|--------|------|------|
| **Static Split** | Simple, fast | Overestimates performance, data leakage |
| **Walk-Forward** | Realistic, robust, no lookahead bias | Slower (multiple retrains) |

### 🔄 Walk-Forward Benefits

1. **Realistic Performance**: Simulates actual trading conditions where models are retrained periodically
2. **No Lookahead Bias**: Each model only uses data available at that point in time
3. **Adaptive Learning**: Models adapt to changing market conditions over time
4. **Robust Evaluation**: Multiple out-of-sample tests provide better performance estimates
5. **Market Regime Detection**: Performance varies across different market conditions

### 📊 Interpretation of Results

- **Mean Performance**: Average across all windows
- **Standard Deviation**: Consistency of performance
- **Window-by-Window**: Shows performance evolution over time
- **Final Models**: Most recent models for actual trading

## 📈 Performance Expectations

### Typical Results

| Metric | Expected Range | Good Performance |
|--------|----------------|------------------|
| **LSTM MAE** | 0.005 - 0.015 | < 0.010 |
| **LSTM RMSE** | 0.010 - 0.025 | < 0.018 |
| **XGBoost Accuracy** | 0.55 - 0.75 | > 0.65 |
| **XGBoost AUC** | 0.60 - 0.85 | > 0.70 |

### Feature Importance

Typically, the most important features are:
1. **lstm_delta** (LSTM prediction)
2. **RSI** (momentum)
3. **MACD histogram** (trend)
4. **Price vs EMA** (trend position)
5. **Volume ratio** (market activity)

## 🚨 Troubleshooting

### Common Issues

#### 1. pandas-ta Installation Error
```bash
# If pandas-ta installation fails, try:
pip install --upgrade pandas-ta

# Or install from source:
pip install git+https://github.com/twopirllc/pandas-ta
```

#### 2. Memory Issues
```python
# Reduce batch size in LSTM training
batch_size=16  # instead of 32

# Or reduce sequence length
lstm_sequence_length = 30  # instead of 60
```

#### 3. No Data Available
```bash
# Ensure data collector has run
python binance_data_collector.py

# Check data directory
ls data/
```

#### 4. TensorFlow GPU Issues
```python
# Force CPU usage if GPU causes problems
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## 🔄 Next Steps

After training completes:

1. **📊 Analyze Results**: Review `results/training_summary.json`
2. **🔍 Feature Analysis**: Check feature importance files
3. **🧪 Backtesting**: Implement trading strategy using trained models
4. **📈 Paper Trading**: Test with live data before real trading
5. **🔄 Retraining**: Set up periodic model updates

## 🎯 Integration with Trading Bot

### Loading Trained Models

```python
import pickle
import tensorflow as tf

# Load LSTM model
lstm_model = tf.keras.models.load_model('models/lstm/btceur_lstm.h5')

# Load XGBoost model
with open('models/xgboost/btceur_xgboost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Load scaler
with open('models/scalers/btceur_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

### Making Predictions

```python
# 1. Prepare LSTM input (last 60 candles)
lstm_input = prepare_lstm_sequence(recent_data)
lstm_input_scaled = scaler.transform(lstm_input)

# 2. Get LSTM prediction
lstm_delta = lstm_model.predict(lstm_input_scaled)

# 3. Prepare XGBoost features
features = create_technical_features(recent_data)
features['lstm_delta'] = lstm_delta[0]

# 4. Get final trading signal
trade_signal = xgb_model.predict_proba([features])[0][1]  # Probability of up move
```

## 📚 References

- **LSTM**: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **XGBoost**: [XGBoost Documentation](https://xgboost.readthedocs.io/)
- **Technical Analysis**: [pandas-ta Documentation](https://github.com/twopirllc/pandas-ta)
- **Time Series**: [Scikit-learn Time Series](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

**🎉 Happy Trading! Remember: Past performance doesn't guarantee future results. Always use proper risk management.**