import pickle
import os

# Check ADAEUR window 6 selected features
feature_path = r'C:\Users\best test\Documents\GitHub\bot\models\feature_columns\adaeur_window_6_selected.pkl'
print(f'Loading feature columns from: {feature_path}')

with open(feature_path, 'rb') as f:
    features = pickle.load(f)

print(f'Number of features: {len(features)}')
print('Features:')
for i, feature in enumerate(features):
    print(f'{i+1:2d}. {feature}')

# Also check a few other windows for comparison
print('\n' + '='*50)
print('Checking other windows for comparison:')

for window in [1, 2, 3]:
    try:
        path = f'C:\\Users\\best test\\Documents\\GitHub\\bot\\models\\feature_columns\\adaeur_window_{window}_selected.pkl'
        with open(path, 'rb') as f:
            feats = pickle.load(f)
        print(f'Window {window}: {len(feats)} features')
    except Exception as e:
        print(f'Window {window}: Error - {e}')