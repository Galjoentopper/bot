#!/usr/bin/env python3
"""
Feature Compatibility Demonstration

This script demonstrates how the model inference pipeline handles feature 
compatibility issues in production scenarios, showing:

1. Loading of training feature requirements
2. Alignment of runtime features with model expectations
3. Handling of missing and extra features
4. Comprehensive warning logging
5. Graceful continuation of predictions despite compatibility issues
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

# Configure logging to show all compatibility messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

def simulate_production_scenario():
    """
    Simulate a production scenario where the model inference pipeline
    receives features that don't perfectly match training expectations.
    """
    print("🔧 Feature Compatibility System Demonstration")
    print("=" * 60)
    print("Simulating model inference with imperfect feature compatibility...")
    print()
    
    try:
        from paper_trader.models.model_compatibility import ModelCompatibilityHandler
        from feature_compatibility_fix import align_features_with_training
        
        # Initialize the compatibility handler
        handler = ModelCompatibilityHandler('models')
        print("✅ Compatibility handler initialized")
        
        # Simulate incoming market data with some missing features
        print("\n📊 Simulating incoming market data...")
        simulated_features = pd.DataFrame({
            # Core price data (available)
            'close': np.random.lognormal(np.log(50000), 0.02, 100),
            'volume': np.random.lognormal(10, 0.5, 100),
            'high': np.random.lognormal(np.log(50100), 0.02, 100),
            'low': np.random.lognormal(np.log(49900), 0.02, 100),
            
            # Some technical indicators (available)
            'rsi_14': np.random.uniform(20, 80, 100),
            'macd': np.random.normal(0, 100, 100),
            'bb_upper': np.random.lognormal(np.log(51000), 0.02, 100),
            'bb_lower': np.random.lognormal(np.log(49000), 0.02, 100),
            'volume_ratio': np.random.lognormal(0, 0.3, 100),
            
            # Extra features not used in training (will be dropped)
            'new_experimental_indicator': np.random.normal(0, 1, 100),
            'unused_feature': np.random.uniform(0, 1, 100),
            'debug_flag': np.ones(100),
            
            # Note: Many expected features are missing (returns, volatility, etc.)
        })
        
        print(f"📈 Market data received: {simulated_features.shape[1]} features, {len(simulated_features)} samples")
        print(f"   Available features: {list(simulated_features.columns[:5])}...")
        
        # Test the compatibility system
        symbol = "BTCEUR"
        window = 100
        
        print(f"\n🎯 Testing compatibility for {symbol} window {window}...")
        
        # Step 1: Load training requirements
        print("\n1️⃣ Loading training feature requirements...")
        metadata = handler.load_training_metadata(symbol, window)
        
        lstm_features_expected = len(metadata.get('lstm_features', []))
        xgb_features_expected = len(metadata.get('xgb_features', []))
        
        print(f"   📋 LSTM model expects: {lstm_features_expected} features")
        print(f"   📋 XGBoost model expects: {xgb_features_expected} features")
        
        # Step 2: Validate compatibility
        print("\n2️⃣ Validating feature compatibility...")
        validation = handler.validate_feature_compatibility(
            simulated_features, symbol, window, 'both'
        )
        
        lstm_compatible = validation.get('lstm_compatible', False)
        xgb_compatible = validation.get('xgboost_compatible', False)
        
        print(f"   🔍 LSTM compatibility: {'✅' if lstm_compatible else '⚠️'} {lstm_compatible}")
        print(f"   🔍 XGBoost compatibility: {'✅' if xgb_compatible else '⚠️'} {xgb_compatible}")
        
        # Show detailed diagnostics
        lstm_diag = validation.get('lstm_diagnosis', {})
        xgb_diag = validation.get('xgboost_diagnosis', {})
        
        lstm_missing = len(lstm_diag.get('missing_features', []))
        xgb_missing = len(xgb_diag.get('missing_features', []))
        
        print(f"   📊 LSTM missing features: {lstm_missing}")
        print(f"   📊 XGBoost missing features: {xgb_missing}")
        
        if lstm_missing > 0:
            print(f"      Missing LSTM features (first 5): {lstm_diag.get('missing_features', [])[:5]}")
        
        # Step 3: Demonstrate feature alignment and prediction preparation
        print("\n3️⃣ Preparing features for model inference...")
        print("   (Watch for compatibility warnings and feature filling)")
        
        # Prepare LSTM input
        print("\n   🧠 Preparing LSTM input...")
        lstm_input = handler.prepare_lstm_input(simulated_features, symbol, window)
        
        if lstm_input is not None:
            print(f"   ✅ LSTM input prepared: shape {lstm_input.shape}")
            print(f"      Value range: [{np.min(lstm_input):.3f}, {np.max(lstm_input):.3f}]")
            print(f"      Contains NaN: {np.isnan(lstm_input).any()}")
        else:
            print("   ❌ LSTM input preparation failed")
        
        # Prepare XGBoost features
        print("\n   🌳 Preparing XGBoost features...")
        xgb_features, expected_features = handler.align_xgboost_features(
            simulated_features, symbol, window, None, 50000.0
        )
        
        if xgb_features is not None:
            print(f"   ✅ XGBoost features prepared: shape {xgb_features.shape}")
            print(f"      Expected feature count: {len(expected_features)}")
            
            # Check for lstm_delta feature
            if 'lstm_delta' in xgb_features.columns:
                print(f"      ✅ lstm_delta feature added: {xgb_features['lstm_delta'].iloc[0]:.6f}")
            else:
                print("      ⚠️ lstm_delta feature not added (no LSTM prediction available)")
        else:
            print("   ❌ XGBoost feature preparation failed")
        
        # Step 4: Show that the system continues despite compatibility issues
        print("\n4️⃣ Demonstrating graceful degradation...")
        
        if lstm_input is not None or xgb_features is not None:
            print("   ✅ System successfully prepared features despite compatibility issues")
            print("   🎯 Models can now make predictions with filled/default values")
            print("   📝 All compatibility issues were logged with appropriate warnings")
        else:
            print("   ⚠️ Both model preparations failed - would trigger fallback mechanisms")
        
        # Step 5: Show recommendations
        print("\n5️⃣ System recommendations...")
        recommendations = []
        
        if lstm_missing > 5:
            recommendations.append(f"Consider improving feature engineering to include more LSTM features ({lstm_missing} missing)")
        
        if xgb_missing > 10:
            recommendations.append(f"XGBoost model has many missing features ({xgb_missing}) - verify feature pipeline")
        
        if not recommendations:
            recommendations.append("Feature compatibility is good - no major issues detected")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 60)
        print("✅ Demonstration completed successfully!")
        print("\nKey takeaways:")
        print("• Features are automatically aligned with training expectations")
        print("• Missing features are filled with appropriate defaults")
        print("• Extra features are safely removed")
        print("• Comprehensive warnings help identify issues")
        print("• System continues prediction despite compatibility problems")
        print("• All changes are logged for debugging and monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simulate_production_scenario()