#!/usr/bin/env python3
"""Test PPO model loading with updated memory configuration."""

import os
import sys
from pathlib import Path

import psutil

# Add src to path
sys.path.insert(0, "/opt/trading_bot/bot/src")

# Import PPO manager
from models.ppo_model_manager import get_ppo_manager


def print_memory_info():
    """Print current memory status."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print(f"System Memory Status:")
    print(f"  RAM Total: {memory.total / (1024**3):.2f} GB")
    print(f"  RAM Available: {memory.available / (1024**3):.2f} GB")
    print(f"  RAM Used: {memory.percent:.1f}%")
    print(f"  Swap Total: {swap.total / (1024**3):.2f} GB")
    print(f"  Swap Free: {swap.free / (1024**3):.2f} GB")
    print(f"  Total Available (RAM+Swap): {(memory.available + swap.free) / (1024**3):.2f} GB")
    print()


def test_ppo_loading():
    """Test loading PPO models."""
    print_memory_info()

    # Get PPO manager
    manager = get_ppo_manager()

    # Try to load a PPO model
    model_path = "/opt/trading_bot/bot/models/ppo/BTCEUR/model.zip"

    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        return False

    print(f"Attempting to load PPO model: {model_path}")
    print(f"File size: {Path(model_path).stat().st_size / (1024**3):.2f} GB")
    print()

    # Try to load the model
    model = manager.load_model(model_path)

    if model is not None:
        print("✅ PPO model loaded successfully!")
        print_memory_info()

        # Test prediction
        import numpy as np

        test_obs = np.random.randn(1, 13).astype(np.float32)
        action, _ = manager.predict(model_path, test_obs)
        print(f"Test prediction: {action}")

        # Cleanup
        manager.cleanup_all()
        print("\nModel cleaned up")
        print_memory_info()

        return True
    else:
        print("❌ Failed to load PPO model")
        return False


if __name__ == "__main__":
    success = test_ppo_loading()
    sys.exit(0 if success else 1)
