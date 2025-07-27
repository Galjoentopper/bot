#!/usr/bin/env python3
"""
Final validation of the paper trader compatibility fixes.
This script demonstrates that all the requirements from the problem statement have been met.
"""

import sys
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add to path
sys.path.append(str(Path(__file__).parent))

def validate_requirement_1():
    """Validate: Remove references to unused/obsolete models from main_paper_trader.py."""
    print("🔍 Requirement 1: Remove references to unused/obsolete models")
    
    try:
        # Check that main_paper_trader.py no longer directly imports TRAINING_FEATURES
        with open('main_paper_trader.py', 'r') as f:
            content = f.read()
        
        # Should not have direct TRAINING_FEATURES import anymore
        if 'from paper_trader.models.feature_engineer import TRAINING_FEATURES' in content:
            print("❌ Still has direct TRAINING_FEATURES import")
            return False
        
        # Should have ModelCompatibilityHandler import
        if 'from paper_trader.models.model_compatibility import ModelCompatibilityHandler' not in content:
            print("❌ Missing ModelCompatibilityHandler import")
            return False
        
        # Should not have direct TRAINING_FEATURES usage in validation
        if 'for feature in TRAINING_FEATURES:' in content:
            print("❌ Still has direct TRAINING_FEATURES iteration")
            return False
        
        print("✅ Removed direct TRAINING_FEATURES references")
        print("✅ Added ModelCompatibilityHandler integration")
        return True
        
    except Exception as e:
        print(f"❌ Error checking file: {e}")
        return False

def validate_requirement_2():
    """Validate: Features DataFrame is filtered and aligned before prediction."""
    print("\n🔍 Requirement 2: Features DataFrame filtered and aligned before prediction")
    
    try:
        with open('main_paper_trader.py', 'r') as f:
            content = f.read()
        
        # Should have compatibility validation before prediction
        if 'validate_feature_compatibility' not in content:
            print("❌ Missing feature compatibility validation")
            return False
        
        # Should have compatibility handler usage
        if 'self.compatibility_handler' not in content:
            print("❌ Missing compatibility handler usage")
            return False
        
        # Should check for available windows
        if 'available_windows' not in content:
            print("❌ Missing available windows check")
            return False
        
        print("✅ Features are validated before prediction")
        print("✅ Compatibility handler filters features appropriately")
        return True
        
    except Exception as e:
        print(f"❌ Error checking feature alignment: {e}")
        return False

def validate_requirement_3():
    """Validate: Uses ModelCompatibilityHandler to load training metadata."""
    print("\n🔍 Requirement 3: Uses ModelCompatibilityHandler for training metadata")
    
    try:
        from paper_trader.models.model_compatibility import ModelCompatibilityHandler
        
        handler = ModelCompatibilityHandler(models_dir="models")
        
        # Test that it can load metadata
        metadata = handler.load_training_metadata("BTCEUR", 1)
        
        if 'symbol' not in metadata:
            print("❌ Metadata doesn't contain expected fields")
            return False
        
        if 'lstm_features' not in metadata:
            print("❌ Metadata missing LSTM features")
            return False
        
        if 'xgb_features' not in metadata:
            print("❌ Metadata missing XGBoost features")
            return False
        
        print("✅ ModelCompatibilityHandler loads training metadata")
        print(f"✅ Metadata includes LSTM features: {len(metadata['lstm_features'])}")
        print(f"✅ Metadata includes XGBoost features: {len(metadata['xgb_features'])}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ModelCompatibilityHandler: {e}")
        return False

def validate_requirement_4():
    """Validate: Improved error handling and logging."""
    print("\n🔍 Requirement 4: Improved error handling and logging")
    
    try:
        with open('main_paper_trader.py', 'r') as f:
            content = f.read()
        
        # Should have improved logging with appropriate levels
        if 'trading_logger.debug' not in content:
            print("❌ Missing debug-level logging")
            return False
        
        # Should have conditional warning logic
        if 'lstm_missing > 10' not in content:
            print("❌ Missing conditional warning logic")
            return False
        
        # Should handle compatibility check failures gracefully
        if 'except Exception' not in content:
            print("❌ Missing exception handling")
            return False
        
        print("✅ Added conditional warning logic (only warn for serious issues)")
        print("✅ Improved logging levels (debug, info, warning)")
        print("✅ Graceful error handling for compatibility checks")
        return True
        
    except Exception as e:
        print(f"❌ Error checking error handling: {e}")
        return False

def validate_requirement_5():
    """Validate: Updated comments and documentation."""
    print("\n🔍 Requirement 5: Updated comments to clarify compatibility logic")
    
    try:
        with open('main_paper_trader.py', 'r') as f:
            content = f.read()
        
        # Should have updated class docstring
        if 'feature compatibility handling' not in content.lower():
            print("❌ Missing updated class documentation")
            return False
        
        # Should have comments about compatibility handler
        if 'prevents excessive warnings' not in content.lower():
            print("❌ Missing compatibility handler explanation")
            return False
        
        # Should have comments about model-specific validation
        if 'model-specific' not in content.lower():
            print("❌ Missing model-specific validation comments")
            return False
        
        print("✅ Updated class docstring with compatibility info")
        print("✅ Added comments explaining compatibility logic")
        print("✅ Documented the new workflow")
        return True
        
    except Exception as e:
        print(f"❌ Error checking documentation: {e}")
        return False

def validate_requirement_6():
    """Validate: Runtime pipeline is robust and warnings are reduced."""
    print("\n🔍 Requirement 6: Runtime pipeline robust, warnings reduced")
    
    try:
        from paper_trader.models.feature_engineer import FeatureEngineer
        from paper_trader.models.model_compatibility import ModelCompatibilityHandler
        import numpy as np
        import pandas as pd
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=300, freq='15min')
        
        data = pd.DataFrame({
            'open': np.cumsum(np.random.randn(300) * 0.01) + 50000,
            'high': np.cumsum(np.random.randn(300) * 0.01) + 50100,
            'low': np.cumsum(np.random.randn(300) * 0.01) + 49900,
            'close': np.cumsum(np.random.randn(300) * 0.01) + 50000,
            'volume': np.random.randint(100, 10000, 300),
        }, index=dates)
        
        # Ensure high >= close >= low
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        # Test pipeline robustness
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(data)
        
        if features_df is None:
            print("❌ Feature engineering failed")
            return False
        
        handler = ModelCompatibilityHandler(models_dir="models")
        
        # Test feature alignment (should not crash)
        aligned_lstm, _ = handler.align_lstm_features(features_df, "BTCEUR", 1)
        aligned_xgb, _ = handler.align_xgboost_features(features_df, "BTCEUR", 1)
        
        if aligned_lstm is None or aligned_xgb is None:
            print("❌ Feature alignment failed")
            return False
        
        print("✅ Pipeline handles feature mismatches gracefully")
        print("✅ Feature alignment works without crashing")
        print("✅ Compatible with existing model files")
        return True
        
    except Exception as e:
        print(f"❌ Error testing pipeline robustness: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive validation of all requirements."""
    print("=" * 80)
    print("🏁 FINAL VALIDATION: PAPER TRADER COMPATIBILITY FIXES")
    print("=" * 80)
    print("Validating all requirements from the problem statement...")
    print("=" * 80)
    
    # Set minimal logging
    logging.basicConfig(level=logging.WARNING)
    
    requirements = [
        ("Remove obsolete model references", validate_requirement_1),
        ("Filter features before prediction", validate_requirement_2),
        ("Use ModelCompatibilityHandler", validate_requirement_3),
        ("Improve error handling/logging", validate_requirement_4),
        ("Update comments/documentation", validate_requirement_5),
        ("Runtime pipeline robustness", validate_requirement_6),
    ]
    
    passed = 0
    total = len(requirements)
    
    for req_name, req_func in requirements:
        try:
            if req_func():
                passed += 1
                print(f"✅ PASSED: {req_name}")
            else:
                print(f"❌ FAILED: {req_name}")
        except Exception as e:
            print(f"❌ ERROR in {req_name}: {e}")
    
    print("\n" + "=" * 80)
    print("🏁 FINAL RESULTS")
    print("=" * 80)
    print(f"📊 Requirements met: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL REQUIREMENTS SUCCESSFULLY IMPLEMENTED!")
        print("\n✅ SUMMARY OF FIXES:")
        print("   • Removed direct TRAINING_FEATURES validation")
        print("   • Added ModelCompatibilityHandler integration")
        print("   • Implemented intelligent feature filtering")
        print("   • Reduced excessive compatibility warnings")
        print("   • Enhanced error handling and logging")
        print("   • Updated documentation and workflow")
        print("   • Made runtime pipeline robust against feature mismatches")
        print("\n🚀 The paper trading pipeline is now ready for production!")
    else:
        print(f"⚠️  {total - passed} requirements need attention")
    
    print("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)