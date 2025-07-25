#!/usr/bin/env python3
"""Debug script to reproduce the XGBoost feature mismatch issue."""

import sys
sys.path.append('.')

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from paper_trader.models.feature_engineer import FeatureEngineer

def debug_xgboost_prediction():
    """Debug the exact issue in XGBoost prediction."""
    
    # Create sample data (same as in the actual pipeline)
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=300, freq='15min'),
        'open': np.random.uniform(90000, 100000, 300),
        'high': np.random.uniform(95000, 105000, 300),
        'low': np.random.uniform(85000, 95000, 300),
        'close': np.random.uniform(90000, 100000, 300),
        'volume': np.random.uniform(100000, 1000000, 300)
    }
    df = pd.DataFrame(data)
    
    print("=== DEBUGGING XGBOOST FEATURE MISMATCH ===")
    
    # Step 1: Engineer features
    fe = FeatureEngineer()
    features_df = fe.engineer_features(df)
    
    if features_df is None:
        print("❌ Feature engineering failed")
        return
    
    print(f"✅ Feature engineering successful: {len(features_df.columns)} features")
    
    # Step 2: Load model and feature columns
    try:
        model = xgb.XGBClassifier()
        model.load_model('models/xgboost/btceur_window_1.json')
        print(f"✅ XGBoost model loaded, expects {model.n_features_in_} features")
    except Exception as e:
        print(f"❌ Failed to load XGBoost model: {e}")
        return
    
    try:
        with open('models/feature_columns/btceur_window_1_selected.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        print(f"✅ Feature columns loaded: {len(feature_columns)} features")
        print(f"   Features: {feature_columns}")
    except Exception as e:
        print(f"❌ Failed to load feature columns: {e}")
        return
    
    # Step 3: Reproduce the exact code from _predict_xgboost_window
    print("\n=== REPRODUCING PREDICTION CODE ===")
    
    # Check which features are available
    available_features = [c for c in feature_columns if c in features_df.columns]
    missing_features = [c for c in feature_columns if c not in features_df.columns]
    
    print(f"Available features: {available_features}")
    print(f"Missing features: {missing_features}")
    
    # This is the exact code from the method
    feature_data = (features_df.reindex(columns=feature_columns)
                   .fillna(0)
                   .tail(1))
    
    print(f"Feature data shape: {feature_data.shape}")
    print(f"Feature data columns: {list(feature_data.columns)}")
    print(f"Feature data dtypes: {feature_data.dtypes.to_dict()}")
    
    # Check the actual values being passed
    print("\nFeature data values:")
    print(feature_data.to_string())
    
    # Try to make prediction
    try:
        print(f"\n=== MAKING PREDICTION ===")
        print(f"Passing data with shape {feature_data.shape} to model expecting {model.n_features_in_} features")
        prob_up = model.predict_proba(feature_data)
        print(f"✅ Prediction successful: {prob_up}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print(f"   Error type: {type(e)}")
        
        # Additional debugging
        print(f"\n=== ADDITIONAL DEBUG INFO ===")
        print(f"feature_data type: {type(feature_data)}")
        print(f"feature_data index: {feature_data.index}")
        print(f"feature_data values shape: {feature_data.values.shape}")
        print(f"feature_data values: {feature_data.values}")
        
        # Try to understand what's happening
        try:
            print(f"\nTrying to convert to numpy array...")
            arr = feature_data.to_numpy()
            print(f"Numpy array shape: {arr.shape}")
            print(f"Numpy array: {arr}")
            
            prob_up = model.predict_proba(arr)
            print(f"✅ Prediction with numpy array successful: {prob_up}")
        except Exception as e2:
            print(f"❌ Prediction with numpy array also failed: {e2}")

if __name__ == "__main__":
    debug_xgboost_prediction()