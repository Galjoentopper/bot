#!/usr/bin/env python3
"""
Hybrid Model Training Entry Point

This script provides a unified entry point for training the hybrid LSTM + XGBoost
machine learning models used by the trading bot.

The actual training implementation is located in train_models/train_hybrid_models.py

Usage:
    python train_hybrid_models.py

For more detailed training options and documentation, see:
    docs/README_TRAINING.md
"""

import sys
import os

def main():
    """Main entry point for model training"""
    print("🧠 Cryptocurrency Trading Bot - Model Training")
    print("=" * 50)
    print()
    
    # Path to the actual training script
    training_script = os.path.join(os.path.dirname(__file__), 'train_models', 'train_hybrid_models.py')
    
    if os.path.exists(training_script):
        print(f"🚀 Running: {training_script}")
        print("=" * 50)
        print()
        
        # Execute the training script
        import subprocess
        result = subprocess.run([sys.executable, training_script], 
                              cwd=os.path.dirname(__file__))
        return result.returncode
    else:
        print(f"❌ Error: {training_script} not found")
        print()
        print("Please ensure the train_models/ directory exists with train_hybrid_models.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())