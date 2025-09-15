#!/usr/bin/env python3
"""
Test script to validate training pipeline for Train.ipynb
=========================================================

This script tests the complete training pipeline to ensure Train.ipynb will work correctly.
"""

import os
import sys
import traceback
from pathlib import Path

# Add paths
sys.path.append("/notebooks/bot")
sys.path.append("/notebooks/bot/src")


def test_imports():
    """Test all critical imports work."""
    print("Testing critical imports...")

    try:
        import lightgbm
        import numpy as np
        import pandas as pd
        import stable_baselines3
        import torch

        print("✅ Core ML libraries imported successfully")

        from src.data_pipeline.dataset_builder import DatasetBuilder
        from src.models.gru_trainer import GRUTrainer
        from src.models.lgbm_trainer import LightGBMTrainer
        from src.models.ppo_trainer import PPOTrainer

        print("✅ Model trainers imported successfully")

        from paperspace_mlops.paperspace_training import PaperspaceTraining

        print("✅ Main training script imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False


def test_data_availability():
    """Test data availability and loading."""
    print("\nTesting data availability...")

    try:
        from paperspace_mlops.paperspace_training import PaperspaceTraining

        trainer = PaperspaceTraining(max_hours=0.01)  # Very short timeout
        data_stats = trainer.verify_data_availability()

        total_samples = sum(data_stats.values())
        print(
            f"✅ Data verification successful: {total_samples:,} samples across {len(data_stats)} symbols"
        )

        if total_samples > 0:
            print("✅ Sufficient data available for training")
            return True
        else:
            print("❌ No data available for training")
            return False

    except Exception as e:
        print(f"❌ Data availability test failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_preparation():
    """Test dataset preparation pipeline."""
    print("\nTesting dataset preparation...")

    try:
        from paperspace_mlops.paperspace_training import PaperspaceTraining

        trainer = PaperspaceTraining(max_hours=0.01)
        datasets_result = trainer.prepare_datasets(["BTCEUR"])  # Test with one symbol

        if datasets_result.get("success") and datasets_result.get("datasets"):
            dataset_count = len(datasets_result["datasets"])
            sample_count = datasets_result.get("total_samples", 0)
            print(
                f"✅ Dataset preparation successful: {dataset_count} datasets, {sample_count:,} samples"
            )
            return True
        else:
            print(f"❌ Dataset preparation failed: {datasets_result}")
            return False

    except Exception as e:
        print(f"❌ Dataset preparation test failed: {e}")
        traceback.print_exc()
        return False


def test_model_trainers():
    """Test individual model trainers."""
    print("\nTesting model trainers...")

    try:
        import numpy as np

        from src.models.lgbm_trainer import LightGBMTrainer

        # Create minimal test data
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100) * 0.1
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20) * 0.1

        config = {
            "models": {"lightgbm": {"n_estimators": 5, "max_depth": 2}}  # Very small for quick test
        }

        trainer = LightGBMTrainer(config)
        result = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            experiment_name="pipeline_test",
            save_path=None,
        )

        if result.get("validation_metrics"):
            print(
                f"✅ Model training successful with R² = {result['validation_metrics'].get('r2', 'N/A'):.4f}"
            )
            return True
        else:
            print(f"❌ Model training failed: {result}")
            return False

    except Exception as e:
        print(f"❌ Model trainer test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration loading...")

    try:
        import yaml

        config_path = "/notebooks/bot/training_config.yaml"
        if not Path(config_path).exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        required_sections = ["data_acquisition", "training"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False

        # Check that models are defined in training section
        training_models = config.get("training", {}).get("models", [])
        if not training_models:
            print("❌ No models defined in training.models section")
            return False

        print(f"✅ Configuration loaded successfully with {len(config)} sections")
        symbols = config.get("data_acquisition", {}).get("symbols", [])
        models = training_models
        print(f"✅ Configuration: {len(symbols)} symbols, {len(models)} model types")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRAINING PIPELINE VALIDATION TEST")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("Data Availability Test", test_data_availability),
        ("Dataset Preparation Test", test_dataset_preparation),
        ("Model Trainer Test", test_model_trainers),
        ("Configuration Test", test_configuration),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Train.ipynb should work correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
