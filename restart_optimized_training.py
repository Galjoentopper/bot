#!/usr/bin/env python3
"""
Optimized Training Restart Script
================================

This script restarts the BTCEUR training with optimized settings for maximum GPU utilization.
Key optimizations:
- Removed 15GB GPU memory limit
- Disabled mixed precision for speed
- Increased batch sizes (4096 training, 8192 inference)
- Reduced XGBoost complexity for faster training
- Added detailed progress monitoring
"""

import subprocess
import sys
import time
from datetime import datetime

def run_optimized_training():
    print("🚀 Starting Optimized BTCEUR Training")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🔧 Optimizations Applied:")
    print("   • Removed GPU memory limit (was 15GB)")
    print("   • Disabled mixed precision (using float32)")
    print("   • Increased batch sizes: 4096 training, 8192 inference")
    print("   • Reduced XGBoost complexity for speed")
    print("   • Added detailed progress monitoring")
    print("   • Reduced LSTM epochs from 200 to 100")
    print("="*60)
    
    try:
        # Run the training command
        cmd = [sys.executable, "train_hybrid_models.py", "--symbols", "BTCEUR"]
        
        print(f"\n🏃 Executing: {' '.join(cmd)}")
        print("\n📊 Monitor GPU usage with: nvidia-smi")
        print("💡 Expected improvements:")
        print("   • GPU utilization should increase from 8% to 50-80%")
        print("   • Window training time should reduce from 30min to 5-10min")
        print("\n" + "="*60)
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())
            
        # Wait for completion
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with return code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"\n❌ Error running training: {e}")

if __name__ == "__main__":
    run_optimized_training()