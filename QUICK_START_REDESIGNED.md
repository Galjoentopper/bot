# Quick Start Guide: Redesigned WebSocket-Based Paper Trader

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_paper_trader.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

3. **Validate Installation**:
   ```bash
   python validate_redesign.py
   ```

## Running the Redesigned Trader

### Option 1: Use the New Redesigned Trader (Recommended)
```bash
python redesigned_main_paper_trader.py
```

### Option 2: Continue with Original Trader
```bash
python main_paper_trader.py
```

## Key Differences

| Feature | Original Trader | Redesigned Trader |
|---------|----------------|-------------------|
| **Price Source** | Mixed WebSocket/API | Dedicated WebSocket per symbol |
| **Prediction Trigger** | Time-based cycles | Price-update triggered |
| **Historical Data** | Mixed with real-time | Separate API-based collection |
| **Price Consistency** | Potential mismatches | 100% consistent real-time prices |
| **Connection Management** | Single WebSocket | Individual connections per symbol |
| **Error Recovery** | Basic reconnection | Exponential backoff + fallback |

## Expected Improvements

With the redesigned trader, you should see:

1. **✅ No more "unrealistic price changes"** - All trading decisions use consistent real-time prices
2. **✅ Faster response times** - Predictions triggered immediately on price updates
3. **✅ More accurate predictions** - Models use the latest real-time price data
4. **✅ Better reliability** - Dedicated connections with robust error handling
5. **✅ Cleaner logging** - Separate logs for trading decisions and technical details

## Monitoring

### Real-time Status
Check the logs for:
- `📊 HIGH VOLATILITY` - Significant price movements
- `🧠 PREDICTION` - Model predictions with confidence scores  
- `💰 EXECUTING BUY SIGNAL` - Trade executions
- `✅ POSITION OPENED/CLOSED` - Position management

### Connection Health
The trader automatically monitors:
- WebSocket connection status per symbol
- Price update frequencies
- Data quality and integrity
- Automatic reconnection on failures

### Performance Metrics
Look for improvements in:
- **Trade Execution Rate**: More signals should execute successfully
- **Price Synchronization**: No more large price discrepancies  
- **System Stability**: Fewer connection errors and timeouts
- **Prediction Accuracy**: Better model performance with fresh data

## Troubleshooting

### WebSocket Connection Issues
If you see connection problems:
1. Check internet connectivity
2. Verify Bitvavo API credentials
3. Check firewall settings for WebSocket connections
4. The system automatically falls back to API calls

### No Price Updates
If price updates aren't coming through:
1. Check if symbols are correctly configured in .env
2. Verify WebSocket URL is correct
3. Check Bitvavo service status
4. Review logs for connection errors

### Model Loading Issues  
If models fail to load:
1. Ensure models exist in the configured path
2. Check model file permissions
3. Verify TensorFlow installation
4. Review model compatibility with current TensorFlow version

## Configuration Options

Key environment variables for the redesigned trader:

```bash
# Trading symbols (comma-separated)
SYMBOLS=BTC-EUR,ETH-EUR,ADA-EUR,SOL-EUR,XRP-EUR

# Candle interval for historical data
CANDLE_INTERVAL=1m

# WebSocket configuration  
BITVAVO_WS_URL=wss://ws.bitvavo.com/v2
WEBSOCKET_SLEEP_SECONDS=60

# Trading parameters
MIN_CONFIDENCE_THRESHOLD=0.7
MIN_SIGNAL_STRENGTH=MODERATE
MAX_POSITIONS_PER_SYMBOL=1
```

## Next Steps

1. **Run Validation**: Ensure everything works with `python validate_redesign.py`
2. **Start Small**: Begin with paper trading to validate performance
3. **Monitor Closely**: Watch logs for the first few hours of operation
4. **Compare Performance**: Track improvements vs. the original trader
5. **Scale Up**: Increase position sizes once confident in the new system

## Support

If you encounter issues:
1. Check the `REDESIGNED_ARCHITECTURE.md` for detailed technical information
2. Review logs in `paper_trader/logs/` directory
3. Run validation tests to identify specific issues
4. The original trader remains available as a fallback option

The redesigned trader represents a complete rebuild of the pricing mechanism to eliminate the "unrealistic price changes" issue while providing a more robust, accurate, and reliable trading system.