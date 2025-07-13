#!/usr/bin/env python3
"""
Test script to demonstrate the resume functionality for train_hybrid_models.py

This script shows how the training will resume from the last completed window
based on the metrics CSV file.
"""

import os
import pandas as pd
from train_hybrid_models import HybridModelTrainer

def test_resume_functionality():
    """
    Test the resume functionality by checking existing metrics
    """
    print("🧪 Testing Resume Functionality")
    print("=" * 50)
    
    # Initialize trainer
    trainer = HybridModelTrainer(symbols=['BTCEUR'])
    
    # Test the get_last_completed_window method
    symbol = 'BTCEUR'
    last_window = trainer.get_last_completed_window(symbol)
    
    print(f"\n📊 Resume Analysis for {symbol}:")
    print(f"   Last completed window: {last_window}")
    print(f"   Next window to train: {last_window + 1}")
    
    # Check what files exist
    metrics_file = f"logs/{symbol.lower()}_metrics.csv"
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        print(f"\n📄 Metrics file analysis:")
        print(f"   File: {metrics_file}")
        print(f"   Total rows: {len(df)}")
        print(f"   Window range: {df['window'].min()} to {df['window'].max()}")
        print(f"   Last 5 windows:")
        print(df[['window', 'lstm_mae', 'xgb_accuracy']].tail())
    
    # Check model files
    model_dirs = {
        'LSTM': f"models/lstm",
        'XGBoost': f"models/xgboost", 
        'Scalers': f"models/scalers"
    }
    
    print(f"\n🤖 Saved model files for {symbol}:")
    for model_type, model_dir in model_dirs.items():
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir) if symbol.lower() in f.lower()]
            print(f"   {model_type}: {len(files)} files")
            if files:
                # Show first few and last few files
                if len(files) <= 6:
                    print(f"      Files: {files}")
                else:
                    print(f"      First 3: {files[:3]}")
                    print(f"      Last 3: {files[-3:]}")
        else:
            print(f"   {model_type}: Directory not found")
    
    print(f"\n🔄 Resume Recommendation:")
    if last_window > 0:
        print(f"   ✅ Resume training is possible")
        print(f"   📊 {last_window} windows already completed")
        print(f"   🚀 Run: python train_hybrid_models.py --symbols BTCEUR")
        print(f"   📈 Training will automatically resume from window {last_window + 1}")
    else:
        print(f"   🆕 No previous training found, will start fresh")
        print(f"   🚀 Run: python train_hybrid_models.py --symbols BTCEUR")

if __name__ == "__main__":
    test_resume_functionality()