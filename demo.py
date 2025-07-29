#!/usr/bin/env python3
"""
Demo Script for Paper Trader and Simple Data Collector
======================================================

This script demonstrates both the paper trader and simple data collector
components working together. It shows how the trained models from 
train_hybrid_models.py can be used for live trading decisions.

Usage:
    python demo.py
"""

import os
import sys
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'-'*40}")
    print(f"📊 {title}")
    print(f"{'-'*40}")

def main():
    """Main demo function"""
    print_header("HYBRID LSTM + XGBOOST TRADING BOT DEMO")
    print("This demo shows the paper trader and data collector in action")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import components
    try:
        from paper_trader import PaperTrader
        from simple_binance_collector import SimpleBinanceCollector
        print("✅ Successfully imported trading components")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Demo 1: Data Collector
    print_section("Data Collector Demo")
    print("Testing the simplified Binance data collector...")
    
    collector = SimpleBinanceCollector(
        symbols=['BTCEUR', 'ETHEUR', 'ADAEUR'], 
        data_dir='data'
    )
    
    print(f"📁 Data directory: {collector.data_dir}")
    print(f"⏰ Interval: {collector.interval}")
    print(f"📊 Symbols: {', '.join(collector.symbols)}")
    
    # Get data summary
    summary = collector.get_data_summary()
    
    print("\nDatabase Summary:")
    for symbol, stats in summary.items():
        if 'status' in stats:
            print(f"  {symbol}: {stats['status']}")
        else:
            print(f"  {symbol}:")
            print(f"    📊 Records: {stats['total_records']:,}")
            print(f"    📅 Range: {stats['start_date']} to {stats['end_date']}")
            print(f"    💰 Price Range: €{stats['min_price']:,.2f} - €{stats['max_price']:,.2f}")
            print(f"    📈 Avg Volume: {stats['avg_volume']:,.2f}")
            print(f"    💾 Database Size: {stats['database_size_mb']:.1f} MB")
    
    # Demo 2: Paper Trader
    print_section("Paper Trader Demo")
    print("Testing the hybrid LSTM + XGBoost paper trader...")
    
    # Initialize paper trader
    trader = PaperTrader(
        symbols=['BTCEUR', 'ETHEUR', 'ADAEUR'],
        data_dir='data',
        models_dir='models',
        initial_cash=10000.0,
        position_size=0.15,  # 15% per position
        confidence_threshold=0.6,  # 60% confidence threshold
        min_confidence=0.45,  # Sell below 45%
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.10,  # 10% take profit
        max_positions=2  # Max 2 concurrent positions
    )
    
    print(f"💰 Initial cash: €{trader.initial_cash:,.2f}")
    print(f"📊 Position size: {trader.position_size*100:.0f}% of portfolio")
    print(f"🎯 Buy threshold: {trader.confidence_threshold:.0%}")
    print(f"📉 Sell threshold: {trader.min_confidence:.0%}")
    
    # Load models
    print("\n🤖 Loading trading models...")
    if not trader.load_models():
        print("❌ Failed to load models")
        return
    
    print("✅ Models loaded successfully!")
    
    # Show loaded models
    print("\nLoaded Components:")
    for key in trader.model_loader.models.keys():
        print(f"  📈 {key}")
    for key in trader.model_loader.scalers.keys():
        print(f"  📊 {key}_scaler")
    for key in trader.model_loader.feature_columns.keys():
        features_count = len(trader.model_loader.feature_columns[key])
        print(f"  🔧 {key}_features ({features_count} features)")
    
    # Demo trading cycle
    print_section("Live Trading Simulation")
    print("Running live trading cycle...")
    
    # Run trading cycle
    trader.run_trading_cycle()
    
    # Get detailed results
    stats = trader.get_portfolio_stats()
    
    print("\n📊 Trading Results:")
    print(f"💰 Portfolio Value: €{stats.total_value:,.2f}")
    print(f"💵 Cash Available: €{stats.cash:,.2f}")
    print(f"📈 Positions Value: €{stats.positions_value:,.2f}")
    print(f"🔄 Total Trades: {stats.total_trades}")
    print(f"✅ Winning Trades: {stats.winning_trades}")
    print(f"❌ Losing Trades: {stats.losing_trades}")
    
    if stats.total_trades > 0:
        print(f"🎯 Win Rate: {stats.win_rate:.1%}")
        print(f"💵 Total P&L: €{stats.total_pnl:,.2f}")
    
    return_pct = (stats.total_value / trader.initial_cash - 1) * 100
    print(f"📊 Total Return: {return_pct:+.2f}%")
    
    # Show current positions
    if trader.positions:
        print(f"\n🏦 Current Positions ({len(trader.positions)}):")
        for symbol, position in trader.positions.items():
            current_data = trader.get_latest_data(symbol, lookback_hours=1)
            if not current_data.empty:
                current_price = current_data['close'].iloc[-1]
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
                unrealized_pct = (current_price - position.entry_price) / position.entry_price
                print(f"  {symbol}:")
                print(f"    📊 Quantity: {position.quantity:.6f}")
                print(f"    💰 Entry: €{position.entry_price:.2f}")
                print(f"    📈 Current: €{current_price:.2f}")
                print(f"    💵 P&L: €{unrealized_pnl:+.2f} ({unrealized_pct:+.2%})")
                print(f"    🎯 Confidence: {position.entry_confidence:.1%}")
    else:
        print("\n🏦 No current positions")
    
    # Demo model predictions
    print_section("Model Predictions Demo")
    print("Showing individual model predictions for each symbol...")
    
    for symbol in trader.symbols:
        prediction = trader.make_prediction(symbol)
        if prediction:
            price, lstm_pred, xgb_prob = prediction
            print(f"\n{symbol}:")
            print(f"  💰 Current Price: €{price:,.2f}")
            print(f"  🧠 LSTM Prediction: {lstm_pred:.4f}")
            print(f"  🌲 XGBoost Probability: {xgb_prob:.2%}")
            
            # Trading signals
            should_buy = trader.should_buy(symbol, xgb_prob)
            should_sell = trader.should_sell(symbol, price, xgb_prob) if symbol in trader.positions else False
            
            if should_buy:
                print(f"  🟢 Signal: BUY (confidence: {xgb_prob:.1%})")
            elif should_sell:
                print(f"  🔴 Signal: SELL (confidence: {xgb_prob:.1%})")
            else:
                print(f"  ⚪ Signal: HOLD (confidence: {xgb_prob:.1%})")
        else:
            print(f"\n{symbol}: ❌ Prediction failed")
    
    # Summary
    print_section("Demo Summary")
    print("✅ Data Collector: Successfully reads and validates 15-minute data")
    print("✅ Paper Trader: Successfully loads models and makes predictions")
    print("✅ Trading Logic: Implements risk management and position tracking")
    print("✅ Integration: Both components work with train_hybrid_models.py output")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Run 'python paper_trader.py' for live paper trading")
    print(f"   2. Run 'python simple_binance_collector.py' to update data")
    print(f"   3. Use 'python train_hybrid_models.py' to retrain models")
    print(f"   4. Modify confidence thresholds and risk parameters as needed")

if __name__ == "__main__":
    main()