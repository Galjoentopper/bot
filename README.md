# Binance 15-Minute Data Collector

A Python program that collects historical 15-minute OHLCV data for cryptocurrency pairs from Binance without requiring API keys.

## Features

- ✅ **No API Key Required**: Uses Binance's public endpoints
- 🚀 **Hybrid Data Collection**: Downloads bulk monthly files first (much faster), then uses API for recent data
- 💾 **SQLite Storage**: Each coin stored in separate database files
- 📊 **Comprehensive Data**: OHLCV + volume, trades, and taker buy data
- 🔄 **Duplicate Prevention**: Automatically handles duplicate data
- 📈 **Progress Tracking**: Real-time progress updates during collection
- ⚡ **Optimized Speed**: Bulk files can download months of data in seconds vs hours with API

## Supported Pairs

- BTC/EUR
- ETH/EUR
- ADA/EUR
- SOL/EUR
- XRP/EUR

## Data Range

- **Timeframe**: 15-minute candles
- **Start Date**: January 1, 2020
- **End Date**: Current date
- **Estimated Size**: ~200-500MB total for all pairs

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Data Collector**:
   ```bash
   python binance_data_collector.py
   ```

## Output Structure

```
data/
├── btceur_15m.db    # Bitcoin data
├── etheur_15m.db    # Ethereum data
├── adaeur_15m.db    # Cardano data
├── soleur_15m.db    # Solana data
└── xrpeur_15m.db    # Ripple data
```

## Database Schema

Each SQLite file contains a `market_data` table with:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | INTEGER | Unix timestamp (ms) |
| datetime | TEXT | Human-readable datetime |
| open | REAL | Opening price (EUR) |
| high | REAL | Highest price (EUR) |
| low | REAL | Lowest price (EUR) |
| close | REAL | Closing price (EUR) |
| volume | REAL | Base asset volume |
| quote_volume | REAL | Quote asset volume (EUR) |
| trades | INTEGER | Number of trades |
| taker_buy_base | REAL | Taker buy base asset volume |
| taker_buy_quote | REAL | Taker buy quote asset volume |

## Usage Examples

### Query Data with Python

```python
import sqlite3
import pandas as pd

# Connect to Bitcoin data
conn = sqlite3.connect('data/btceur_15m.db')

# Get recent data
df = pd.read_sql_query("""
    SELECT datetime, open, high, low, close, volume 
    FROM market_data 
    ORDER BY timestamp DESC 
    LIMIT 100
""", conn)

print(df.head())
conn.close()
```

### Query Data with SQL

```sql
-- Get daily high/low for BTC in 2024
SELECT 
    DATE(datetime) as date,
    MIN(low) as daily_low,
    MAX(high) as daily_high,
    AVG(volume) as avg_volume
FROM market_data 
WHERE datetime >= '2024-01-01'
GROUP BY DATE(datetime)
ORDER BY date;
```

## Rate Limiting

- **Bulk File Delay**: 500ms between monthly file downloads
- **API Request Delay**: 100ms between API calls
- **Batch Size**: Up to 1000 candles per API request
- **Fallback**: 500 candles per request (regular mode)
- **Bulk Files**: Download entire months instantly (no rate limits)

## Error Handling

- Automatic retry with fallback API method
- Duplicate data prevention
- Progress tracking and logging
- Graceful handling of network issues

## Performance

- **Collection Time**: ~2-5 minutes for bulk files + ~5-10 minutes for recent API data
- **Storage**: ~40-100MB per pair for 4+ years of data
- **Memory Usage**: Low memory footprint with batch processing
- **Speed Improvement**: 5-10x faster than pure API approach using bulk monthly files

## Troubleshooting

### Common Issues

1. **Network Errors**: The program will retry automatically
2. **Rate Limiting**: Built-in delays prevent rate limit issues
3. **Incomplete Data**: Check logs for specific error messages

### Logs

The program provides detailed logging:
- Progress updates for each symbol
- Success/failure status
- Data summary statistics
- Error messages with timestamps

## Next Steps

After collecting data, you can:
1. Use it for backtesting trading strategies
2. Train machine learning models
3. Perform technical analysis
4. Build trading bots

## License

This project is for educational and research purposes. Please respect Binance's API terms of service.