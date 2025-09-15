#!/usr/bin/env python3
"""
Test script to isolate PPO model loading issues.
"""
import os
import sys
import threading
import time
from pathlib import Path

import psutil


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_ppo_loading():
    """Test PPO model loading with detailed logging."""
    print(f"Starting PPO model test at {time.strftime('%H:%M:%S')}")
    print(f"Process ID: {os.getpid()}")
    print(f"Thread count: {threading.active_count()}")

    model_path = "models/ppo/BTCEUR/model.zip"

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return False

    print(f"Model file exists: {Path(model_path).stat().st_size / 1024 / 1024:.1f} MB")

    # Check memory before
    mem_before = get_memory_usage()
    print(f"Memory before loading: {mem_before:.1f} MB")

    try:
        print("Importing stable_baselines3...")
        from stable_baselines3 import PPO

        print("Import successful")

        print(f"Loading model from {model_path}...")
        start_time = time.time()

        # Load with timeout using threading
        model = None
        error = None

        def load_model():
            nonlocal model, error
            try:
                model = PPO.load(model_path, device="cpu")
                print("Model loaded successfully in thread")
            except Exception as e:
                error = e
                print(f"Error in thread: {e}")

        # Start loading in thread with timeout
        load_thread = threading.Thread(target=load_model)
        load_thread.start()

        # Wait with timeout
        load_thread.join(timeout=60)  # 60 second timeout

        if load_thread.is_alive():
            print("TIMEOUT: PPO.load() took longer than 60 seconds!")
            return False

        if error:
            print(f"ERROR during loading: {error}")
            return False

        if model is None:
            print("ERROR: Model is None after loading")
            return False

        load_time = time.time() - start_time
        mem_after = get_memory_usage()

        print(f"Model loaded successfully in {load_time:.2f}s")
        print(f"Memory after loading: {mem_after:.1f} MB (+{mem_after - mem_before:.1f} MB)")
        print(f"Thread count after load: {threading.active_count()}")

        # Test prediction
        print("Testing model prediction...")
        import numpy as np

        # Create dummy observation (adjust size based on model)
        obs = np.random.random((104,)).astype(np.float32)
        action, _ = model.predict(obs)
        print(f"Prediction successful: action={action}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("PPO Model Loading Test")
    print("=" * 50)

    success = test_ppo_loading()

    print("=" * 50)
    if success:
        print("TEST PASSED: PPO model loaded successfully")
    else:
        print("TEST FAILED: PPO model loading failed")

    sys.exit(0 if success else 1)
