#!/usr/bin/env python3
"""
Test script for adaptive model weighting system
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml

from src.trading.enhanced_signal_generator import EnhancedSignalGenerator
from src.trading.profit_optimizer import ProfitOptimizer


def test_adaptive_weights():
    """Test the adaptive model weighting functionality."""
    print("=== Testing Adaptive Model Weighting System ===")

    # Load configuration
    with open("training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize components
    profit_optimizer = ProfitOptimizer(config.get("profit_optimization", {}))
    signal_generator = EnhancedSignalGenerator(config, profit_optimizer)

    print(f"\nInitial model weights:")
    print(f"Base weights: {signal_generator.base_model_weights}")
    print(f"Current weights: {signal_generator.model_weights}")
    print(f"Adaptive weights enabled: {signal_generator.weight_adaptation_enabled}")

    # Simulate model performance data
    print("\n=== Simulating Model Performance ===")

    # Simulate GRU performing well
    for i in range(10):
        signal_generator.update_model_performance(
            "gru", 0.8, 0.05, True
        )  # Good predictions, profitable

    # Simulate LightGBM performing poorly
    for i in range(10):
        signal_generator.update_model_performance(
            "lightgbm", 0.3, -0.02, False
        )  # Poor predictions, unprofitable

    # Simulate PPO mixed performance
    for i in range(5):
        signal_generator.update_model_performance("ppo", 0.6, 0.01, True)  # Decent predictions
    for i in range(5):
        signal_generator.update_model_performance("ppo", 0.4, -0.01, False)  # Some poor predictions

    print("\nAfter performance updates:")
    performance_summary = signal_generator.get_model_performance_summary()
    for model, stats in performance_summary.items():
        if isinstance(stats, dict) and "accuracy" in stats:
            print(
                f"{model}: accuracy={stats['accuracy']:.3f}, profit_rate={stats['profit_rate']:.3f}, "
                f"weight={stats['current_weight']:.3f} (base: {stats['base_weight']:.3f})"
            )

    print(f"\nAdaptive weights enabled: {performance_summary.get('adaptive_enabled', False)}")
    print(f"Updated model weights: {signal_generator.model_weights}")

    # Test weight adaptation over time
    print("\n=== Testing Weight Adaptation Over Time ===")

    # Continue with more GRU good performance
    for i in range(20):
        signal_generator.update_model_performance("gru", 0.85, 0.08, True)

    # Continue with LightGBM poor performance
    for i in range(20):
        signal_generator.update_model_performance("lightgbm", 0.25, -0.03, False)

    print("\nAfter extended performance tracking:")
    performance_summary = signal_generator.get_model_performance_summary()
    for model, stats in performance_summary.items():
        if isinstance(stats, dict) and "accuracy" in stats:
            print(
                f"{model}: accuracy={stats['accuracy']:.3f}, profit_rate={stats['profit_rate']:.3f}, "
                f"weight={stats['current_weight']:.3f} (change: {stats['current_weight'] - stats['base_weight']:+.3f})"
            )

    print(f"\nFinal model weights: {signal_generator.model_weights}")

    # Verify weight normalization
    total_weight = sum(signal_generator.model_weights.values())
    print(f"Total weight (should be ~1.0): {total_weight:.6f}")

    print("\n=== Test Completed Successfully ===")
