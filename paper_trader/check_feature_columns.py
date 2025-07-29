import pickle
import os

def check_feature_columns():
    feature_dir = "models/feature_columns"
    
    # Check a few examples to understand the structure
    test_files = [
        "etheur_window_1_selected.pkl",
        "etheur_window_5_selected.pkl", 
        "adaeur_window_1_selected.pkl",
        "soleur_window_1_selected.pkl",
        "xrpeur_window_1_selected.pkl"
    ]
    
    for file in test_files:
        file_path = os.path.join(feature_dir, file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    features = pickle.load(f)
                print(f"\n{file}:")
                print(f"  Type: {type(features)}")
                if isinstance(features, list):
                    print(f"  Count: {len(features)}")
                    print(f"  Features: {features[:10]}...")  # Show first 10
                elif isinstance(features, dict):
                    print(f"  Keys: {list(features.keys())}")
                else:
                    print(f"  Content: {features}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File not found: {file}")

if __name__ == "__main__":
    check_feature_columns()