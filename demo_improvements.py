#!/usr/bin/env python3
"""Demonstration script for all trading bot improvements."""

import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Import circuit breaker and resilience
from src.core.advanced_circuit_breaker import circuit_breaker
from src.core.resilience import bulkhead, retry, timeout

# Import our new structured logging
from src.core.structured_logger import (
    correlation_context,
    get_logger,
    get_trading_logger,
    log_performance,
)

# Demo logger
logger = get_logger(__name__)
trading_logger = get_trading_logger("demo")


# 1. Circuit Breaker Demo
@circuit_breaker("external_api", failure_threshold=3, timeout=5.0)
def unreliable_api_call(fail_rate=0.3):
    """Simulate an unreliable external API call."""
    if random.random() < fail_rate:
        raise ConnectionError("API temporarily unavailable")
    return {"data": "success", "timestamp": time.time()}


# 2. Retry with Backoff Demo
@retry(max_attempts=3, initial_delay=0.5, strategy="exponential_jitter")
def flaky_data_fetch(symbol: str):
    """Simulate a flaky data fetching operation."""
    if random.random() < 0.4:
        raise ConnectionError(f"Failed to fetch data for {symbol}")

    # Generate fake market data
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1H"),
            "open": np.random.uniform(45000, 55000, 100),
            "high": np.random.uniform(46000, 56000, 100),
            "low": np.random.uniform(44000, 54000, 100),
            "close": np.random.uniform(45000, 55000, 100),
            "volume": np.random.randint(100, 10000, 100),
        }
    )


# 3. Performance Logging Demo
@log_performance(log_args=True, log_result=False)
def calculate_technical_indicators(data: pd.DataFrame, symbol: str):
    """Calculate technical indicators with performance logging."""
    # Simulate heavy computation
    time.sleep(0.1)

    indicators = {
        "rsi": random.uniform(0, 100),
        "macd": random.uniform(-1, 1),
        "bb_upper": random.uniform(1.01, 1.05),
        "bb_lower": random.uniform(0.95, 0.99),
        "volume_sma": random.uniform(1000, 10000),
    }

    return indicators


# 4. Bulkhead Demo (Resource Isolation)
@bulkhead("model_prediction", max_concurrent=3, timeout_seconds=10.0)
def predict_price_movement(symbol: str, features: dict):
    """Simulate model prediction with resource isolation."""
    # Simulate model inference time
    time.sleep(random.uniform(0.1, 0.5))

    prediction = random.uniform(0, 1)
    confidence = random.uniform(0.6, 0.95)

    return {
        "symbol": symbol,
        "prediction": prediction,
        "confidence": confidence,
        "model_type": "gru",
    }


# 5. Async Operations Demo
async def async_data_pipeline(symbol: str):
    """Demonstrate async operations with logging and error handling."""

    with correlation_context(symbol=symbol, operation="data_pipeline"):
        logger.info(f"Starting data pipeline for {symbol}")

        try:
            # Simulate async data fetching
            await asyncio.sleep(0.1)
            data = flaky_data_fetch(symbol)

            # Calculate indicators
            indicators = calculate_technical_indicators(data, symbol)

            # Make prediction
            prediction = predict_price_movement(symbol, indicators)

            # Log trading decision
            if prediction["prediction"] > 0.7:
                trading_logger.log_trade(
                    symbol=symbol,
                    action="BUY",
                    amount=1000.0,
                    price=50000.0,
                    confidence=prediction["confidence"],
                )
            elif prediction["prediction"] < 0.3:
                trading_logger.log_trade(
                    symbol=symbol,
                    action="SELL",
                    amount=500.0,
                    price=50000.0,
                    confidence=prediction["confidence"],
                )

            logger.info(
                f"Pipeline completed for {symbol}",
                prediction=prediction["prediction"],
                confidence=prediction["confidence"],
            )

            return prediction

        except Exception as e:
            logger.error(f"Pipeline failed for {symbol}", exception=e)
            raise


# 6. Structured Logging Demo
def demonstrate_logging_features():
    """Demonstrate various logging features."""

    logger.info("=== Logging Features Demo ===")

    # Basic structured logging
    logger.info("System starting up", version="1.0.0", environment="production")

    # Trading-specific logging
    with correlation_context(symbol="BTCEUR", trade_id="trade_123"):
        trading_logger.log_model_prediction(
            symbol="BTCEUR",
            model_type="lightgbm",
            prediction=0.75,
            confidence=0.87,
            features_hash="abc123",
        )

        trading_logger.log_performance(
            operation="model_inference", duration_ms=125.5, success=True, model_accuracy=0.92
        )

    # System event logging
    logger.log_system_event(
        event_type="model_loaded",
        event_data={
            "model_type": "gru",
            "symbol": "ETHEUR",
            "version": "v1.2.3",
            "memory_usage_mb": 256,
        },
    )

    # Error logging with context
    try:
        raise ValueError("Demo error for logging")
    except Exception as e:
        logger.error(
            "Demonstration error occurred", error_code="DEMO_001", user_action="ignore", exception=e
        )


# 7. Circuit Breaker State Demo
def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker behavior."""

    logger.info("=== Circuit Breaker Demo ===")

    # Make multiple calls to show circuit breaker in action
    for i in range(10):
        try:
            result = unreliable_api_call(fail_rate=0.8)  # High failure rate
            logger.info(f"API call {i+1} succeeded", result=result)
        except Exception as e:
            logger.warning(f"API call {i+1} failed", error=str(e))

        time.sleep(0.1)


# 8. Performance Monitoring Demo
async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring across operations."""

    logger.info("=== Performance Monitoring Demo ===")

    symbols = ["BTCEUR", "ETHEUR", "ADAEUR"]

    # Run parallel operations with performance tracking
    tasks = []
    for symbol in symbols:
        task = asyncio.create_task(async_data_pipeline(symbol))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_predictions = [r for r in results if not isinstance(r, Exception)]
    failed_predictions = [r for r in results if isinstance(r, Exception)]

    logger.info(
        "Performance summary",
        total_symbols=len(symbols),
        successful_predictions=len(successful_predictions),
        failed_predictions=len(failed_predictions),
        success_rate=len(successful_predictions) / len(symbols),
    )


# Main demo function
async def main():
    """Run all improvement demonstrations."""

    logger.info("🚀 Trading Bot Improvements Demo Starting...")
    logger.info("=" * 60)

    # 1. Logging features
    demonstrate_logging_features()

    print("\n" + "─" * 60 + "\n")

    # 2. Circuit breaker behavior
    demonstrate_circuit_breaker()

    print("\n" + "─" * 60 + "\n")

    # 3. Performance monitoring
    await demonstrate_performance_monitoring()

    print("\n" + "─" * 60 + "\n")

    logger.info("✅ Demo completed successfully!")
    logger.info("🔍 Check the logs above to see:")
    logger.info("   • Structured JSON logging with correlation IDs")
    logger.info("   • Circuit breaker protecting against API failures")
    logger.info("   • Automatic retry logic with exponential backoff")
    logger.info("   • Performance monitoring and metrics")
    logger.info("   • Trading-specific logging for auditing")
    logger.info("   • Error handling with detailed context")


if __name__ == "__main__":
    # Set up some basic logging to console for the demo
    import logging

    # Run the demo
    print("🎯 Trading Bot Comprehensive Improvements Demo")
    print("=" * 60)
    print("This demo showcases all the production-ready improvements:")
    print("• Structured logging with correlation IDs")
    print("• Circuit breakers for external API calls")
    print("• Retry logic with exponential backoff")
    print("• Performance monitoring and metrics")
    print("• Resource isolation with bulkheads")
    print("• Async operations with error handling")
    print("• Trading-specific audit logging")
    print("=" * 60)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
