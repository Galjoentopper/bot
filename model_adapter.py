#!/usr/bin/env python3
"""
ModelAdapter Interface and Implementations
==========================================

This module provides a unified interface for different model types:
1. Abstract ModelAdapter interface
2. Concrete implementations for LSTM, XGBoost, and other models  
3. Standardized training, prediction, and artifact management
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Provides unified interface for training, prediction, and artifact management
    across different model types (LSTM, XGBoost, LightGBM, etc.).
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model adapter.
        
        Args:
            name: Model name/identifier
            config: Model configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.metadata = {}
        
    @abstractmethod
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            train_idx: np.ndarray,
            val_idx: np.ndarray,
            **kwargs) -> 'ModelAdapter':
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            train_idx: Training sample indices
            val_idx: Validation sample indices
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate prediction probabilities (for classifiers).
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array
        """
        # Default implementation returns predictions as probabilities
        return self.predict(X)
    
    @abstractmethod
    def get_artifacts(self) -> Dict[str, Any]:
        """
        Get model artifacts for saving.
        
        Returns:
            Dictionary containing model state and metadata
        """
        pass
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save location
        """
        artifacts = self.get_artifacts()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save artifacts
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Saved {self.name} model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], name: str = None) -> 'ModelAdapter':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            name: Model name (if different from saved)
            
        Returns:
            Loaded model adapter
        """
        with open(filepath, 'rb') as f:
            artifacts = pickle.load(f)
        
        # Create instance and restore state
        instance = cls(name or artifacts.get('name', 'loaded_model'))
        instance._restore_from_artifacts(artifacts)
        
        logger.info(f"Loaded {instance.name} model from {filepath}")
        return instance
    
    @abstractmethod
    def _restore_from_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """
        Restore model state from artifacts.
        
        Args:
            artifacts: Model artifacts dictionary
        """
        pass


class LSTMModelAdapter(ModelAdapter):
    """Model adapter for LSTM/neural network models."""
    
    def __init__(self, name: str = "LSTM", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.scaler = None
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            train_idx: np.ndarray,
            val_idx: np.ndarray,
            **kwargs) -> 'LSTMModelAdapter':
        """Train LSTM model."""
        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}")
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        # Build model
        self.model = self._build_lstm_model(X_train.shape[1:])
        
        # Train
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.config.get('epochs', 50),
            batch_size=self.config.get('batch_size', 32),
            verbose=0,
            callbacks=self._get_callbacks()
        )
        
        self.is_fitted = True
        self.metadata['training_history'] = history.history
        logger.info(f"LSTM training completed: {len(history.epoch)} epochs")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate LSTM predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get LSTM model artifacts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Save model weights
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            self.model.save_weights(f.name)
            with open(f.name, 'rb') as model_file:
                model_weights = model_file.read()
        
        return {
            'name': self.name,
            'config': self.config,
            'model_weights': model_weights,
            'model_config': self.model.get_config(),
            'scaler': self.scaler,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
    
    def _restore_from_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore LSTM from artifacts."""
        import tensorflow as tf
        import tempfile
        
        self.config = artifacts['config']
        self.scaler = artifacts['scaler'] 
        self.metadata = artifacts['metadata']
        self.is_fitted = artifacts['is_fitted']
        
        # Restore model
        model_config = artifacts['model_config']
        self.model = tf.keras.Model.from_config(model_config)
        
        # Restore weights
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            f.write(artifacts['model_weights'])
            f.flush()
            self.model.load_weights(f.name)
    
    def _build_lstm_model(self, input_shape: Tuple[int, ...]) -> 'tf.keras.Model':
        """Build LSTM model architecture."""
        import tensorflow as tf
        from tensorflow.keras import layers
        
        model = tf.keras.Sequential([
            layers.LSTM(self.config.get('lstm_units', 64), 
                       return_sequences=True, 
                       input_shape=input_shape),
            layers.Dropout(self.config.get('dropout', 0.2)),
            layers.LSTM(self.config.get('lstm_units', 64) // 2),
            layers.Dropout(self.config.get('dropout', 0.2)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _get_callbacks(self) -> List:
        """Get training callbacks."""
        import tensorflow as tf
        
        return [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config.get('patience', 10),
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]


class XGBoostModelAdapter(ModelAdapter):
    """Model adapter for XGBoost models."""
    
    def __init__(self, name: str = "XGBoost", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            train_idx: np.ndarray,
            val_idx: np.ndarray,
            **kwargs) -> 'XGBoostModelAdapter':
        """Train XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(f"XGBoost not available: {e}")
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Default XGBoost parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        params = {**default_params, **self.config}
        
        # Create model
        self.model = xgb.XGBClassifier(**params)
        
        # Train with validation
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_fitted = True
        logger.info(f"XGBoost training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate XGBoost predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate XGBoost prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return self.model.predict_proba(X)[:, 1]
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get XGBoost model artifacts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return {
            'name': self.name,
            'config': self.config,
            'model': self.model,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
    
    def _restore_from_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore XGBoost from artifacts."""
        self.config = artifacts['config']
        self.model = artifacts['model']
        self.metadata = artifacts['metadata']
        self.is_fitted = artifacts['is_fitted']


class LightGBMModelAdapter(ModelAdapter):
    """Model adapter for LightGBM models."""
    
    def __init__(self, name: str = "LightGBM", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            train_idx: np.ndarray,
            val_idx: np.ndarray,
            **kwargs) -> 'LightGBMModelAdapter':
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(f"LightGBM not available: {e}")
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'objective': 'binary',
            'random_state': 42,
            'verbose': -1
        }
        params = {**default_params, **self.config}
        
        # Create model
        self.model = lgb.LGBMClassifier(**params)
        
        # Train
        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=['validation'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        self.is_fitted = True
        logger.info(f"LightGBM training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate LightGBM predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate LightGBM prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return self.model.predict_proba(X)[:, 1]
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get LightGBM model artifacts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return {
            'name': self.name,
            'config': self.config,
            'model': self.model,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
    
    def _restore_from_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore LightGBM from artifacts."""
        self.config = artifacts['config']
        self.model = artifacts['model']
        self.metadata = artifacts['metadata']
        self.is_fitted = artifacts['is_fitted']


class ModelFactory:
    """Factory for creating model adapters."""
    
    _adapters = {
        'lstm': LSTMModelAdapter,
        'xgboost': XGBoostModelAdapter,
        'lightgbm': LightGBMModelAdapter,
        'lgbm': LightGBMModelAdapter,  # Alias
    }
    
    @classmethod
    def create_adapter(cls, model_type: str, name: str = None, config: Dict[str, Any] = None) -> ModelAdapter:
        """
        Create model adapter by type.
        
        Args:
            model_type: Type of model ('lstm', 'xgboost', 'lightgbm')
            name: Model name
            config: Model configuration
            
        Returns:
            Model adapter instance
        """
        model_type_lower = model_type.lower()
        
        if model_type_lower not in cls._adapters:
            available = ', '.join(cls._adapters.keys())
            raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
        
        adapter_class = cls._adapters[model_type_lower]
        return adapter_class(name or model_type, config)
    
    @classmethod
    def register_adapter(cls, model_type: str, adapter_class: type):
        """
        Register custom model adapter.
        
        Args:
            model_type: Model type identifier
            adapter_class: ModelAdapter subclass
        """
        cls._adapters[model_type.lower()] = adapter_class
        logger.info(f"Registered custom adapter: {model_type}")
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._adapters.keys())