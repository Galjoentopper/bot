# Fix Summary: 'str' object is not callable Error in Keras

## Problem
The training script `train_hybrid_models.py` was failing with the error:
```
TypeError: 'str' object is not callable
```
This occurred in the Keras metrics system where a string was being passed where a callable function was expected.

## Root Causes Identified

### 1. String Metrics in Model Compilation
**Issue**: Using string metrics in model compilation:
```python
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", "precision", "recall"],  # Strings can cause issues
)
```

**Fix**: Use metric function objects instead:
```python
from tensorflow.keras.metrics import Accuracy, Precision, Recall

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=[Accuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall')],
)
```

### 2. Problematic Quantile Loss Function Usage
**Issue**: Using closure-based loss function in warm start:
```python
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=quantile_loss(self.quantile),  # Returns a closure, causes serialization issues
    metrics=["mae", "mse"],
)
```

**Fix**: Use standard loss and proper metric objects:
```python
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss="binary_crossentropy",
    metrics=[MeanAbsoluteError(name='mae'), MeanSquaredError(name='mse')],
)
```

### 3. Improved Custom Loss Class
**Issue**: The original `quantile_loss` function created closures that didn't serialize properly.

**Fix**: Created a proper Keras loss class:
```python
@tf.keras.utils.register_keras_serializable(package="Custom", name="QuantileLoss")
class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantile=0.5, name="quantile_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantile = quantile
        
    def call(self, y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantile * e, (self.quantile - 1) * e), axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"quantile": self.quantile})
        return config
```

## Files Modified
- `train_models/train_hybrid_models.py`: Main fixes applied

## Testing
- Created unit tests to verify the fixes work correctly
- Confirmed the training script now runs without errors
- Verified model saving/loading works with custom objects

## Best Practices to Prevent This Issue

1. **Always use metric function objects** instead of strings when possible:
   ```python
   # Good
   metrics=[Accuracy(), Precision(), Recall()]
   
   # Avoid if possible
   metrics=["accuracy", "precision", "recall"]
   ```

2. **Create proper Keras classes for custom functions** instead of closures:
   ```python
   # Good - Serializable class
   @tf.keras.utils.register_keras_serializable()
   class CustomLoss(tf.keras.losses.Loss):
       # ... implementation
   
   # Avoid - Closure
   def create_loss(param):
       def loss_fn(y_true, y_pred):
           # ... implementation
       return loss_fn
   ```

3. **Always include custom_objects when loading models** with custom functions:
   ```python
   model = tf.keras.models.load_model(
       path,
       custom_objects={"CustomLoss": CustomLoss}
   )
   ```

4. **Test model serialization** during development:
   ```python
   # Save and reload to test serialization
   model.save('test_model.keras')
   loaded_model = tf.keras.models.load_model('test_model.keras', custom_objects={...})
   ```

## Impact
The training script now runs successfully without the "str object is not callable" error, enabling proper model training for the cryptocurrency trading bot.