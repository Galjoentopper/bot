# Feature Factory Architecture Documentation

This repository now includes a comprehensive **Feature Factory** implementation that provides consistent feature engineering across different models and time windows for cryptocurrency trading.

## Overview

The Feature Factory pattern centralizes feature engineering, scaling, and preparation for different model types (LSTM, XGBoost) and different time windows to capture various market dynamics at multiple time scales.

## Architecture Components

### 1. Feature Factory (`feature_factory.py`)

The core component that handles feature engineering and preparation:

**Key Features:**
- Calculates 20+ technical indicators using the `ta` library
- Supports multiple time windows (30, 60, 90 days)
- Handles different feature sets for LSTM vs XGBoost models
- Caches calculated features for performance
- Provides consistent scaling across models

**Technical Indicators Included:**
- Moving Averages (SMA, EMA) with multiple periods
- Momentum indicators (RSI, MACD, ROC, Stochastic)
- Volatility indicators (ATR, Bollinger Bands)
- Trend indicators (ADX)
- Custom features (price changes, volume ratios, etc.)

**Model-Specific Features:**
- **LSTM**: Sequential data with 22 core features for time series modeling
- **XGBoost**: Tabular data with 180+ aggregated features (mean, std, min, max, trend)

### 2. Model Manager (`model_manager.py`)

Manages multiple models with different time windows and combines their predictions:

**Key Features:**
- Loads LSTM and XGBoost models for different window sizes
- Generates predictions from all available models
- Combines predictions using weighted averaging
- Handles missing models gracefully
- Provides model information and status

**Prediction Combination:**
- Short-term models (30 days): Higher weight for immediate signals
- Medium-term models (60 days): Balanced weight for trend confirmation
- Long-term models (90 days): Lower weight for stable trends
- Equal weighting between LSTM and XGBoost model types

### 3. Data Fetcher (`data_fetcher.py`)

Provides clean data access with validation:

**Key Features:**
- Interfaces with existing SQLite databases
- Falls back to Binance API when local data unavailable
- Validates data quality and consistency
- Handles different time formats and data structures
- Provides current price information

### 4. Enhanced Paper Trader (`run_paper_trader_factory.py`)

Complete paper trading system using the Feature Factory architecture:

**Key Features:**
- Uses multiple time windows simultaneously
- Makes decisions based on combined model predictions
- Implements stop-loss and take-profit logic
- Comprehensive logging and performance tracking
- Handles errors gracefully

## Usage

### Basic Usage

```python
from feature_factory import FeatureFactory
from model_manager import ModelManager
from data_fetcher import DataFetcher

# 1. Load historical data
data_fetcher = DataFetcher('BTCUSDT')
historical_data = data_fetcher.get_historical_data(limit=1000)

# 2. Initialize feature factory
feature_factory = FeatureFactory(historical_data)

# 3. Get features for different models
lstm_features = feature_factory.get_features_for_model('lstm', window_size=30)
xgb_features = feature_factory.get_features_for_model('xgboost', window_size=60)

# 4. Initialize model manager
model_manager = ModelManager('./models', window_sizes=[30, 60, 90])

# 5. Get predictions
predictions = model_manager.predict(feature_factory)
print(f"Combined prediction: {predictions['combined']}")
```

### Running the Paper Trader

```bash
# Run the enhanced paper trader
python run_paper_trader_factory.py
```

The paper trader will:
1. Load historical data for the specified symbol
2. Initialize the feature factory with technical indicators
3. Load available models for different time windows
4. Run trading iterations with combined predictions
5. Log all decisions and performance metrics

### Configuration

Key parameters can be modified in `run_paper_trader_factory.py`:

```python
SYMBOL = "BTCUSDT"          # Trading symbol
INITIAL_BALANCE = 10000.0   # Starting balance
WINDOW_SIZES = [30, 60, 90] # Time windows in days
MODEL_DIR = "./models"      # Directory with trained models
ITERATIONS = 50             # Number of trading cycles
INTERVAL_SECONDS = 30       # Time between iterations
```

## Model File Structure

The system expects trained models in the following format:

```
models/
├── lstm_model_30.h5        # LSTM model for 30-day window
├── lstm_model_60.h5        # LSTM model for 60-day window  
├── lstm_model_90.h5        # LSTM model for 90-day window
├── xgb_model_30.pkl        # XGBoost model for 30-day window
├── xgb_model_60.pkl        # XGBoost model for 60-day window
└── xgb_model_90.pkl        # XGBoost model for 90-day window
```

## Feature Engineering Details

### LSTM Features (22 features)
- Core price data: open, high, low, close
- Volume data
- Momentum indicators: RSI, MACD components
- Volatility: ATR, Bollinger Bands
- Trend: Multiple EMAs and SMAs
- Custom: Price/volume changes and ratios

### XGBoost Features (180+ features)
- All LSTM features plus additional indicators
- Window aggregations: mean, std, min, max for each indicator
- Trend calculations over the window period
- Additional momentum and volatility indicators

### Time Windows
- **30 days**: Captures short-term price movements and volatility
- **60 days**: Balances short-term signals with medium-term trends  
- **90 days**: Provides long-term trend confirmation and stability

## Integration with Existing System

The Feature Factory architecture is designed to work alongside the existing paper trading system:

- Uses existing data collection infrastructure (`data/` directory)
- Compatible with existing model training pipelines
- Maintains the same trading decision framework
- Enhances prediction accuracy through multi-window analysis

## Error Handling

The system includes comprehensive error handling:

- Graceful degradation when models are missing
- Data validation with clear error messages
- API fallback when local data unavailable
- Logging of all errors and decisions
- Safe defaults for prediction failures

## Performance Considerations

- **Feature Caching**: Calculated indicators are cached to avoid recalculation
- **Selective Loading**: Only loads models that exist
- **Memory Management**: Limits historical data to necessary amounts
- **Batch Processing**: Efficiently processes multiple time windows

## Dependencies

```python
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
tensorflow>=2.15.0
xgboost>=1.7.0

# Technical analysis
ta>=0.10.0

# Other
requests>=2.28.0
```

## Testing

All components include basic validation:

```bash
# Test feature factory
python -c "from feature_factory import FeatureFactory; print('Feature Factory OK')"

# Test model manager  
python -c "from model_manager import ModelManager; print('Model Manager OK')"

# Test data fetcher
python -c "from data_fetcher import DataFetcher; print('Data Fetcher OK')"
```

## Future Enhancements

Potential improvements to the Feature Factory system:

1. **Additional Indicators**: More sophisticated technical indicators
2. **Dynamic Weighting**: Adaptive weights based on recent performance
3. **Feature Selection**: Automated feature importance analysis
4. **Real-time Updates**: Live feature calculation and model updates
5. **Multi-Asset Support**: Features engineered across multiple trading pairs
6. **Advanced Ensembling**: More sophisticated prediction combination methods

## Troubleshooting

Common issues and solutions:

1. **Missing Models**: System will work with any available models and show warnings for missing ones
2. **Data Issues**: Check data validation output and ensure proper OHLCV format
3. **API Errors**: System falls back to local data when API unavailable
4. **Memory Issues**: Reduce window sizes or historical data limits
5. **Feature Errors**: Check that all required columns are present in input data

For more detailed logging, set logging level to DEBUG in the configuration.