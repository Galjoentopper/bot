#!/usr/bin/env python3
"""
Debug data columns and ATR calculation
"""

import sys
sys.path.append('scripts')
from backtest_models import ModelBacktester, BacktestConfig
import sqlite3
import pandas as pd

def debug_data_columns():
    """Check what columns are available in the processed data"""
    print("🔍 Debugging data columns and ATR calculation...")
    
    config = BacktestConfig()
    backtester = ModelBacktester(config)
    
    # Load raw data
    print("Loading raw data...")
    symbol = 'BTCEUR'
    data = backtester.load_data(symbol)
    print(f"Raw data shape: {data.shape}")
    print(f"Raw data columns: {list(data.columns)}")
    print()
    
    # Calculate features
    print("Calculating features...")
    try:
        data_with_features = backtester.calculate_features(data)
        print(f"Data with features shape: {data_with_features.shape}")
        print(f"Data with features columns: {list(data_with_features.columns)}")
        print()
        
        # Check if ATR exists
        if 'atr' in data_with_features.columns:
            atr_sample = data_with_features['atr'].iloc[-10:]
            print(f"✅ ATR column exists!")
            print(f"Last 10 ATR values: {list(atr_sample)}")
            print(f"ATR mean: {data_with_features['atr'].mean():.6f}")
            print(f"ATR min: {data_with_features['atr'].min():.6f}")
            print(f"ATR max: {data_with_features['atr'].max():.6f}")
        else:
            print(f"❌ ATR column missing!")
        
        # Test position size calculation
        if 'atr' in data_with_features.columns:
            print(f"\n🔍 Testing position size calculation...")
            capital = 10000
            current_price = data_with_features['close'].iloc[-1]
            atr = data_with_features['atr'].iloc[-1]
            
            print(f"Capital: {capital}")
            print(f"Current price: {current_price:.2f}")
            print(f"ATR: {atr:.6f}")
            
            # BUY scenario
            stop_loss_buy = current_price - (atr * 2)
            position_size_buy = backtester.calculate_position_size(capital, current_price, stop_loss_buy)
            print(f"BUY: Stop loss = {stop_loss_buy:.2f}, Position size = {position_size_buy:.6f}")
            
            # SELL scenario 
            stop_loss_sell = current_price + (atr * 2)
            position_size_sell = backtester.calculate_position_size(capital, current_price, stop_loss_sell)
            print(f"SELL: Stop loss = {stop_loss_sell:.2f}, Position size = {position_size_sell:.6f}")
            
            # Debug the calculation step by step
            risk_amount = capital * config.risk_per_trade
            price_risk_buy = abs(current_price - stop_loss_buy)
            price_risk_sell = abs(current_price - stop_loss_sell)
            
            print(f"\nStep-by-step calculation:")
            print(f"Risk per trade: {config.risk_per_trade} ({config.risk_per_trade:.1%})")
            print(f"Risk amount: {risk_amount:.2f}")
            print(f"Price risk (BUY): {price_risk_buy:.6f}")
            print(f"Price risk (SELL): {price_risk_sell:.6f}")
            
            if price_risk_buy > 0:
                calc_position_buy = risk_amount / price_risk_buy
                max_position_buy = capital * 0.1
                final_position_buy = min(calc_position_buy, max_position_buy)
                print(f"Calculated position (BUY): {calc_position_buy:.6f}")
                print(f"Max position (10% of capital): {max_position_buy:.2f}")
                print(f"Final position (BUY): {final_position_buy:.6f}")
            
    except Exception as e:
        print(f"❌ Error calculating features: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_data_columns()