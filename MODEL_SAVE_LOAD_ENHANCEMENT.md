# Model Save/Load Compatibility Enhancement

## Overview

This enhancement patches the model saving and loading logic in `train_hybrid_models.py` to fully support both XGBoost and calibrated models (CalibratedClassifierCV).

## Problem Addressed

The original code used `model.save_model()` and `model.load_model()` for all models, which only works for pure XGBoost models. When using calibrated models (`CalibratedClassifierCV`), these are sklearn objects that require `joblib.dump()` and `joblib.load()`, causing compatibility issues.

## Solution Implemented

### Helper Functions Added

1. **`save_model_safe(model, filepath, logger=None)`**
   - Auto-detects model type (XGBoost vs sklearn/calibrated)
   - Uses appropriate save method (.save_model() or joblib.dump())
   - Automatically adjusts file extension (.json for XGBoost, .pkl for calibrated)
   - Comprehensive error handling and logging

2. **`load_model_safe(filepath, logger=None)`**  
   - Loads based on file extension (.json → XGBoost, .pkl → joblib)
   - Auto-detection fallback for ambiguous files
   - Returns loaded model or None on failure

3. **`get_feature_importance_safe(model, feature_names=None, logger=None)`**
   - Extracts feature importance from various model types
   - Handles XGBoost, calibrated models, and ensemble models
   - Graceful fallback when importance is not available

### File Organization

- **`models/xgboost/*.json`**: Pure XGBoost models (native format)
- **`models/xgboost/*.pkl`**: Calibrated/sklearn models (joblib format)
- Extensions are automatically managed by save operations
- Full backward compatibility with existing `.json` files

## Code Changes

### Training Script (`train_models/train_hybrid_models.py`)

- Replaced all `model.save_model()` calls with `save_model_safe()`
- Updated warm-start loading to use `load_model_safe()`
- Enhanced feature importance plotting with `get_feature_importance_safe()`
- Added comprehensive documentation and error handling

### Backtest Script (`scripts/backtest_models.py`)

- Updated model loading to support both `.json` and `.pkl` formats
- Enhanced fallback model discovery for both formats
- Integrated safe loading functions for compatibility

## Scientific Rationale

**Calibrated Models Benefits:**
- Provide better probability estimates than raw XGBoost models
- Essential for confidence-based trading strategies
- Enable advanced techniques like isotonic regression calibration
- Improve risk management through accurate probability assessment

**Compatibility Advantages:**
- Seamless integration with existing workflows
- No breaking changes to current model training
- Enables gradual migration to calibrated models
- Supports both pure XGBoost and advanced model types

## Usage Examples

### Training with Calibration
```python
# The trainer automatically uses calibrated models when enabled
trainer = HybridModelTrainer(...)
results = trainer.train_all_models()  # Automatically saves with correct format
```

### Manual Model Operations
```python
from train_hybrid_models import save_model_safe, load_model_safe, get_feature_importance_safe

# Save any model type
save_model_safe(xgb_model, "models/my_model")      # → my_model.json
save_model_safe(calibrated_model, "models/my_cal") # → my_cal.pkl

# Load any model type
model = load_model_safe("models/my_model.json")     # XGBoost model
model = load_model_safe("models/my_cal.pkl")       # Calibrated model

# Extract feature importance safely
features, importance = get_feature_importance_safe(model)
```

## Testing

The implementation has been tested with:
- Pure XGBoost models (save/load/feature importance)
- CalibratedClassifierCV models (save/load/feature importance extraction)
- Error handling and edge cases
- File extension auto-detection
- Backward compatibility with existing models

## Benefits

1. **Full Compatibility**: Works with both XGBoost and calibrated models
2. **Automatic Detection**: No manual specification of model type needed  
3. **Error Resilience**: Comprehensive error handling and logging
4. **Backward Compatible**: Existing models continue to work
5. **Future-Proof**: Easy to extend for additional model types
6. **Production Ready**: Thoroughly tested with real model scenarios

This enhancement enables the use of advanced model calibration techniques while maintaining full compatibility with the existing trading bot infrastructure.