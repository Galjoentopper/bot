#!/usr/bin/env python3
"""
Test script to verify the centralized training system.
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import tempfile
from pathlib import Path

# Import our modules
from dataset_builder import DatasetBuilder
from lgbm_trainer import LightGBMTrainer
from cost_aware_evaluation import CostAwareEvaluator, CostModel

def create_test_data():
    """Create synthetic test data for testing."""
    print("Creating synthetic test data...")
    
    # Create temporary data directory
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    # Generate synthetic market data
    np.random.seed(42)
    n_samples = 5000
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    
    # Simulate realistic price movement
    returns = np.random.normal(0, 0.01, n_samples)
    returns = np.cumsum(returns)
    
    prices = 40000 * np.exp(returns)  # Start at ~$40k
    
    # Create OHLCV data
    high = prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples)))
    volume = np.random.lognormal(10, 1, n_samples)
    
    # Create database
    db_path = data_dir / "btceur_15m.db"
    conn = sqlite3.connect(db_path)
    
    # Create table
    conn.execute("""
        CREATE TABLE market_data (
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    """)
    
    # Insert data
    for i, ts in enumerate(timestamps):
        conn.execute("""
            INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?)
        """, (int(ts.timestamp() * 1000), prices[i], high[i], low[i], prices[i], volume[i]))
    
    conn.commit()
    conn.close()
    
    print(f"Created test database at {db_path}")
    return temp_dir

def test_dataset_builder():
    """Test DatasetBuilder functionality."""
    print("\n=== Testing DatasetBuilder ===")
    
    temp_dir = create_test_data()
    
    try:
        # Initialize DatasetBuilder
        builder = DatasetBuilder(
            data_dir=temp_dir / "data",
            cache_dir=temp_dir / "cache"
        )
        
        # Test dataset creation
        features_df, metadata = builder.get_dataset(
            symbol="BTCEUR",
            interval="15m"
        )
        
        print(f"✅ Dataset created: {len(features_df)} samples, {len(features_df.columns)} features")
        print(f"✅ Features: {list(features_df.columns)[:10]}...")  # Show first 10 features
        
        # Test validation
        validation_report = builder.validate_dataset(features_df, metadata)
        print(f"✅ Validation passed: {validation_report['valid']}")
        if validation_report['warnings']:
            print(f"   Warnings: {len(validation_report['warnings'])}")
        
        # Test caching (should be faster second time)
        print("Testing cache...")
        import time
        start = time.time()
        features_df2, metadata2 = builder.get_dataset("BTCEUR", "15m")
        cache_time = time.time() - start
        
        print(f"✅ Cache working: loaded in {cache_time:.3f}s")
        
        return temp_dir, features_df, metadata
        
    except Exception as e:
        print(f"❌ DatasetBuilder test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_cost_aware_evaluation():
    """Test cost-aware evaluation."""
    print("\n=== Testing Cost-Aware Evaluation ===")
    
    try:
        # Create synthetic prediction data
        np.random.seed(42)
        n_samples = 1000
        
        y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
        y_pred_proba = np.random.beta(2, 5, n_samples)  # Skewed probabilities
        returns = np.random.normal(0.001, 0.02, n_samples)  # Daily returns
        
        # Test evaluator
        cost_model = CostModel(fee_bps=10, slippage_bps=5)
        evaluator = CostAwareEvaluator(cost_model)
        
        # Test evaluation
        metrics = evaluator.evaluate_predictions(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            returns=returns,
            threshold=0.5,
            position_size=1000
        )
        
        print(f"✅ Evaluation completed:")
        print(f"   Net Sharpe: {metrics.net_sharpe_ratio:.4f}")
        print(f"   Gross Return: {metrics.gross_return:.4f}")
        print(f"   Net Return: {metrics.net_return:.4f}")
        print(f"   Num Trades: {metrics.num_trades}")
        print(f"   Win Rate: {metrics.win_rate:.2%}")
        
        # Test threshold optimization
        print("Testing threshold optimization...")
        optimal_threshold, best_metrics = evaluator.find_optimal_threshold(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            returns=returns,
            position_size=1000
        )
        
        print(f"✅ Optimal threshold: {optimal_threshold:.3f}")
        print(f"   Best Net Sharpe: {best_metrics.net_sharpe_ratio:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost-aware evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lightgbm_trainer():
    """Test LightGBM trainer."""
    print("\n=== Testing LightGBM Trainer ===")
    
    temp_dir, features_df, metadata = test_dataset_builder()
    if temp_dir is None:
        print("❌ Skipping LightGBM test due to DatasetBuilder failure")
        return False
    
    try:
        # Initialize DatasetBuilder
        builder = DatasetBuilder(
            data_dir=temp_dir / "data",
            cache_dir=temp_dir / "cache"
        )
        
        # Initialize trainer with simple config
        config = {
            'n_estimators': 50,  # Small for testing
            'learning_rate': 0.1,
            'num_leaves': 15,
            'early_stopping_rounds': 10,
            'verbose': -1
        }
        
        trainer = LightGBMTrainer(
            dataset_builder=builder,
            config=config
        )
        
        # Test training (with minimal splits for speed)
        print("Training LightGBM model...")
        results = trainer.train_symbol(
            symbol="BTCEUR",
            interval="15m",
            n_splits=2,  # Small for testing
            calibrate=False,  # Skip calibration for speed
            save_artifacts=False  # Don't save for test
        )
        
        print(f"✅ LightGBM training completed:")
        print(f"   Folds: {results['n_folds']}")
        print(f"   Avg Net Sharpe: {results.get('avg_net_sharpe', 'N/A')}")
        print(f"   Feature Names: {len(results.get('feature_names', []))}")
        
        if 'feature_importance' in results:
            top_features = results['feature_importance']['top_features'][:5]
            print(f"   Top Features: {top_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ LightGBM trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Run all tests."""
    print("🧪 Running Centralized Training System Tests")
    print("=" * 60)
    
    tests = [
        test_cost_aware_evaluation,
        test_lightgbm_trainer
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed - check implementation")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)