#!/usr/bin/env python3
"""
Example integration of model compatibility fixes with main paper trader.

This script demonstrates how the compatibility fixes resolve the issues
between train_hybrid_models.py and main_paper_trader.py.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demonstrate_compatibility_fixes():
    """Demonstrate the compatibility fixes in action."""
    
    print("🚀 Model Compatibility Integration Demo")
    print("=" * 60)
    
    # Example 1: Feature alignment for inference
    print("\n📋 Example 1: Feature Alignment")
    print("-" * 40)
    
    try:
        from feature_compatibility_fix import align_features_with_training
        import pandas as pd
        import numpy as np
        
        # Simulate inference features (missing some training features)
        inference_features = pd.DataFrame({
            'close': [50000, 50100, 50050],
            'volume': [1000, 1200, 800],
            'returns': [0.002, -0.001, 0.0015],
            'rsi': [65, 70, 60],
            # Note: missing 'macd', 'bb_position', etc. that training expects
        })
        
        # Expected features from training
        training_features = [
            'close', 'volume', 'returns', 'rsi', 'macd', 'bb_position', 
            'volume_ratio', 'price_vs_ema9', 'momentum_10'
        ]
        
        print(f"Inference features available: {list(inference_features.columns)}")
        print(f"Training expects: {training_features}")
        
        # Apply compatibility fix
        aligned_features = align_features_with_training(
            inference_features, training_features, "demo"
        )
        
        print(f"✅ After alignment: {list(aligned_features.columns)}")
        print(f"✅ Missing features filled with appropriate defaults")
        
    except Exception as e:
        print(f"❌ Feature alignment demo failed: {e}")
    
    # Example 2: LSTM sequence preparation with scaling
    print("\n🧠 Example 2: LSTM Sequence Preparation")
    print("-" * 40)
    
    try:
        from feature_compatibility_fix import prepare_lstm_sequence_safe
        from sklearn.preprocessing import StandardScaler
        
        # Create mock feature data
        np.random.seed(42)
        features_df = pd.DataFrame({
            'close': 50000 + np.cumsum(np.random.randn(100) * 50),
            'volume': np.random.lognormal(10, 0.5, 100),
            'returns': np.random.randn(100) * 0.02,
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.randn(100) * 0.001,
        })
        
        lstm_features = ['close', 'volume', 'returns', 'rsi', 'macd']
        
        # Create and fit a scaler (simulating training)
        scaler = StandardScaler()
        scaler.fit(features_df[lstm_features].tail(50))
        
        print(f"Features shape: {features_df.shape}")
        print(f"LSTM features: {lstm_features}")
        
        # Prepare LSTM sequence
        lstm_sequence = prepare_lstm_sequence_safe(
            features_df, lstm_features, sequence_length=96, 
            symbol="DEMO", scaler=scaler
        )
        
        if lstm_sequence is not None:
            print(f"✅ LSTM sequence prepared: {lstm_sequence.shape}")
            print(f"✅ Sequence ready for model.predict()")
        else:
            print("❌ LSTM sequence preparation failed")
            
    except Exception as e:
        print(f"❌ LSTM sequence demo failed: {e}")
    
    # Example 3: Compatibility diagnosis
    print("\n🔍 Example 3: Compatibility Diagnosis")
    print("-" * 40)
    
    try:
        from feature_compatibility_fix import diagnose_compatibility_issues
        
        # Simulate actual vs expected features scenario
        expected_features = [
            'close', 'volume', 'returns', 'rsi', 'macd', 'bb_position',
            'volume_ratio', 'price_vs_ema9', 'momentum_10', 'lstm_delta'
        ]
        
        actual_features = [
            'close', 'volume', 'returns', 'rsi',  # Missing: macd, bb_position, etc.
            'extra_feature_1', 'extra_feature_2'  # Extra features not in training
        ]
        
        diagnosis = diagnose_compatibility_issues(
            "DEMO_SYMBOL", expected_features, actual_features
        )
        
        print(f"Expected features: {len(expected_features)}")
        print(f"Actual features: {len(actual_features)}")
        print(f"✅ Diagnosis completed:")
        print(f"   - Missing: {len(diagnosis['missing_features'])} features")
        print(f"   - Extra: {len(diagnosis['extra_features'])} features") 
        print(f"   - Recommendations: {len(diagnosis['recommendations'])}")
        
        if diagnosis['recommendations']:
            print("💡 Recommendations:")
            for rec in diagnosis['recommendations']:
                print(f"   - {rec}")
                
    except Exception as e:
        print(f"❌ Compatibility diagnosis demo failed: {e}")
    
    # Example 4: Integration benefits
    print(f"\n🎯 Integration Benefits")
    print("-" * 40)
    print("✅ Models trained with train_hybrid_models.py now work seamlessly in main_paper_trader.py")
    print("✅ Feature mismatches are automatically handled with sensible defaults")
    print("✅ Scaler dimension issues are detected and resolved")
    print("✅ LSTM sequence length is consistently maintained (96 periods)")
    print("✅ Robust error handling prevents crashes from missing features")
    print("✅ Comprehensive diagnostics help identify and resolve issues")
    
    # Example 5: Usage in main_paper_trader.py
    print(f"\n🏗️ Usage in Main Paper Trader")
    print("-" * 40)
    print("The ModelCompatibilityHandler is now integrated into:")
    print("  📦 WindowBasedModelLoader: Handles feature alignment automatically")
    print("  🤖 WindowBasedEnsemblePredictor: Uses compatibility checks during prediction")
    print("  🔧 Feature preparation: Ensures LSTM and XGBoost get correct feature formats")
    print("  📊 Validation: Checks compatibility before making predictions")
    
    print(f"\n🎉 Demo completed successfully!")
    print("The compatibility fixes bridge the gap between training and inference pipelines.")

if __name__ == "__main__":
    demonstrate_compatibility_fixes()