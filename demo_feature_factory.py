#!/usr/bin/env python3
"""
Feature Factory Demonstration Script
===================================

This script demonstrates the Feature Factory architecture with mock data,
showing how the system works without requiring actual models or live data.

Usage: python demo_feature_factory.py
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import our Feature Factory components
from feature_factory import FeatureFactory
from model_manager import ModelManager
from data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_realistic_crypto_data(periods: int = 1000, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Generate realistic cryptocurrency price data for demonstration.
    
    Args:
        periods: Number of 15-minute candles to generate
        symbol: Symbol name for logging
        
    Returns:
        DataFrame with realistic OHLCV data
    """
    logger.info(f"Generating {periods} periods of realistic {symbol} data...")
    
    # Start with a realistic BTC price
    base_price = 45000.0
    
    # Generate timestamps (15-minute intervals)
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=periods//96), 
        periods=periods, 
        freq='15min'
    )
    
    # Generate realistic price movements with volatility clustering
    returns = np.random.normal(0, 0.01, periods)  # 1% volatility
    
    # Add some trend and volatility clustering
    trend = np.sin(np.arange(periods) / 100) * 0.005  # Slight sinusoidal trend
    volatility_clusters = np.abs(np.random.normal(0, 0.5, periods // 10))
    vol_expanded = np.repeat(volatility_clusters, 10)[:periods]
    
    # Apply volatility clustering
    returns = returns * (1 + vol_expanded)
    returns = returns + trend
    
    # Generate price series
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLC from close prices
    close = prices
    
    # Generate realistic high/low with some spread
    spread = np.random.uniform(0.002, 0.008, periods)  # 0.2% to 0.8% spread
    high = close * (1 + spread/2 + np.random.uniform(0, spread/2, periods))
    low = close * (1 - spread/2 - np.random.uniform(0, spread/2, periods))
    
    # Open is close of previous period with small gap
    open_prices = np.concatenate([[close[0]], close[:-1]]) * (1 + np.random.normal(0, 0.001, periods))
    
    # Generate volume (inversely correlated with price stability)
    volatility = np.abs(returns)
    base_volume = 1000
    volume = base_volume * (1 + volatility * 50) * np.random.uniform(0.5, 2.0, periods)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Ensure price constraints (high >= open,close,low and low <= open,close,high)
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close'], data['low']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close'], data['high']])
    
    logger.info(f"Generated data: {len(data)} records from {data['timestamp'].min()} to {data['timestamp'].max()}")
    logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    return data

def demonstrate_feature_factory():
    """Demonstrate the Feature Factory system with realistic data."""
    
    logger.info("=== Feature Factory Demonstration ===")
    
    # 1. Generate realistic data
    logger.info("\n1. Generating realistic market data...")
    mock_data = generate_realistic_crypto_data(periods=800, symbol="BTCUSDT")
    
    # 2. Initialize Feature Factory
    logger.info("\n2. Initializing Feature Factory...")
    feature_factory = FeatureFactory(mock_data)
    
    # 3. Calculate technical indicators
    logger.info("\n3. Calculating technical indicators...")
    feature_factory.calculate_all_technical_indicators()
    
    indicators = list(feature_factory.indicator_cache.keys())
    logger.info(f"Calculated {len(indicators)} technical indicators:")
    for i, indicator in enumerate(indicators[:10]):  # Show first 10
        logger.info(f"  - {indicator}")
    if len(indicators) > 10:
        logger.info(f"  ... and {len(indicators) - 10} more")
    
    # 4. Generate features for different models and windows
    logger.info("\n4. Generating features for different models...")
    
    window_sizes = [30, 60, 90]
    model_types = ['lstm', 'xgboost']
    
    for model_type in model_types:
        logger.info(f"\n--- {model_type.upper()} Features ---")
        for window in window_sizes:
            try:
                features = feature_factory.get_features_for_model(model_type, window)
                logger.info(f"  {window}-day window: {features['X'].shape} ({len(features['feature_names'])} features)")
                
                # Show some feature names
                if len(features['feature_names']) > 5:
                    sample_features = features['feature_names'][:5]
                    logger.info(f"    Sample features: {sample_features}")
                
            except Exception as e:
                logger.error(f"  Error generating {model_type} features for {window}-day window: {e}")
    
    # 5. Demonstrate prediction feature preparation
    logger.info("\n5. Demonstrating prediction features...")
    
    for model_type in model_types:
        for window in window_sizes:
            try:
                pred_features = feature_factory.get_prediction_features(model_type, window)
                if pred_features is not None:
                    logger.info(f"  {model_type.upper()} {window}-day prediction features: {pred_features.shape}")
                else:
                    logger.warning(f"  {model_type.upper()} {window}-day: No prediction features generated")
            except Exception as e:
                logger.error(f"  Error generating prediction features for {model_type} {window}-day: {e}")
    
    # 6. Demonstrate Model Manager (without actual models)
    logger.info("\n6. Demonstrating Model Manager...")
    
    model_manager = ModelManager('./models', window_sizes)
    model_info = model_manager.get_model_info()
    logger.info(f"Model availability: {model_info}")
    
    # Get predictions (will be defaults since no models exist)
    predictions = model_manager.predict(feature_factory)
    logger.info(f"Mock predictions: {predictions}")
    
    # 7. Show some technical analysis results
    logger.info("\n7. Technical Analysis Sample Results...")
    
    latest_data = mock_data.tail(5)
    latest_indicators = {}
    
    # Get latest values of key indicators
    for indicator in ['rsi', 'macd', 'bbands_upper', 'bbands_lower', 'atr']:
        if indicator in feature_factory.indicator_cache:
            latest_indicators[indicator] = feature_factory.indicator_cache[indicator].iloc[-1]
    
    logger.info("Latest Technical Indicators:")
    for indicator, value in latest_indicators.items():
        logger.info(f"  {indicator.upper()}: {value:.4f}")
    
    logger.info(f"\nLatest Prices:")
    logger.info(f"  Close: ${latest_data.iloc[-1]['close']:.2f}")
    logger.info(f"  High:  ${latest_data.iloc[-1]['high']:.2f}")
    logger.info(f"  Low:   ${latest_data.iloc[-1]['low']:.2f}")
    logger.info(f"  Volume: {latest_data.iloc[-1]['volume']:.0f}")
    
    # 8. Performance summary
    logger.info("\n8. Performance Summary...")
    
    feature_count = len(feature_factory.indicator_cache)
    data_points = len(mock_data)
    
    logger.info(f"  Processed {data_points} data points")
    logger.info(f"  Calculated {feature_count} technical indicators")
    logger.info(f"  Generated features for {len(model_types)} model types")
    logger.info(f"  Supported {len(window_sizes)} time windows")
    logger.info(f"  Cache contains {len(feature_factory.calculated_features)} feature sets")
    
    logger.info("\n=== Demonstration Complete ===")
    
    return feature_factory, model_manager, mock_data

def demonstrate_trading_signals(feature_factory: FeatureFactory, data: pd.DataFrame):
    """
    Demonstrate how trading signals would be generated.
    
    Args:
        feature_factory: Initialized FeatureFactory
        data: Market data DataFrame
    """
    logger.info("\n=== Trading Signal Demonstration ===")
    
    # Get latest technical indicators
    latest_rsi = feature_factory.indicator_cache['rsi'].iloc[-1]
    latest_macd = feature_factory.indicator_cache['macd'].iloc[-1]
    latest_macd_signal = feature_factory.indicator_cache['macd_signal'].iloc[-1]
    latest_price = data.iloc[-1]['close']
    latest_bb_upper = feature_factory.indicator_cache['bbands_upper'].iloc[-1]
    latest_bb_lower = feature_factory.indicator_cache['bbands_lower'].iloc[-1]
    
    logger.info("Current Market Conditions:")
    logger.info(f"  Price: ${latest_price:.2f}")
    logger.info(f"  RSI: {latest_rsi:.2f}")
    logger.info(f"  MACD: {latest_macd:.4f}")
    logger.info(f"  MACD Signal: {latest_macd_signal:.4f}")
    logger.info(f"  BB Upper: ${latest_bb_upper:.2f}")
    logger.info(f"  BB Lower: ${latest_bb_lower:.2f}")
    
    # Simple trading signals
    signals = []
    
    # RSI signals
    if latest_rsi > 70:
        signals.append("RSI Overbought (>70) - Consider SELL")
    elif latest_rsi < 30:
        signals.append("RSI Oversold (<30) - Consider BUY")
    
    # MACD signals
    if latest_macd > latest_macd_signal:
        macd_trend = "Bullish"
    else:
        macd_trend = "Bearish"
    signals.append(f"MACD Trend: {macd_trend}")
    
    # Bollinger Band signals
    if latest_price > latest_bb_upper:
        signals.append("Price above Bollinger Upper - Potential reversal")
    elif latest_price < latest_bb_lower:
        signals.append("Price below Bollinger Lower - Potential bounce")
    
    logger.info("\nTrading Signals:")
    for i, signal in enumerate(signals, 1):
        logger.info(f"  {i}. {signal}")
    
    # Overall sentiment
    bullish_signals = sum(1 for s in signals if any(word in s.lower() for word in ['buy', 'bullish', 'bounce']))
    bearish_signals = sum(1 for s in signals if any(word in s.lower() for word in ['sell', 'bearish', 'reversal']))
    
    if bullish_signals > bearish_signals:
        overall_sentiment = "BULLISH"
    elif bearish_signals > bullish_signals:
        overall_sentiment = "BEARISH"
    else:
        overall_sentiment = "NEUTRAL"
    
    logger.info(f"\nOverall Sentiment: {overall_sentiment}")
    logger.info(f"  Bullish signals: {bullish_signals}")
    logger.info(f"  Bearish signals: {bearish_signals}")

def main():
    """Main demonstration function."""
    logger.info("Starting Feature Factory Demonstration...")
    
    try:
        # Run the main demonstration
        feature_factory, model_manager, mock_data = demonstrate_feature_factory()
        
        # Show trading signals
        demonstrate_trading_signals(feature_factory, mock_data)
        
        logger.info("\n✅ Demonstration completed successfully!")
        logger.info("\nTo run the full paper trader with this architecture:")
        logger.info("  python run_paper_trader_factory.py")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()