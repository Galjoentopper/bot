"""Model management with proper validation and metadata handling."""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from ..core.interfaces import IModelManager, ModelMetadata, ValidationResult
from ..core.base_service import BaseService
from ..core.container import injectable


@injectable
class ModelManager(BaseService, IModelManager):
    """Manages trading models with validation and metadata."""
    
    def __init__(self, models_dir: Optional[str] = None):
        """Initialize model manager.
        
        Args:
            models_dir: Directory containing model files
        """
        super().__init__()
        self._models_dir = Path(models_dir) if models_dir else Path('models')
        self._models_dir.mkdir(exist_ok=True)
        
        # Cache for loaded models and metadata
        self._model_cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        
        self._log_info(f"ModelManager initialized with directory: {self._models_dir}")
    
    def load_model(self, symbol: str, model_type: str) -> Any:
        """Load a model for the given symbol and type.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCEUR')
            model_type: Type of model (e.g., 'lstm', 'transformer', 'ensemble')
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model validation fails
        """
        try:
            model_key = f"{symbol}_{model_type}"
            
            # Check cache first
            if model_key in self._model_cache:
                self._log_info(f"Returning cached model for {model_key}")
                return self._model_cache[model_key]
            
            # Find model file
            model_path = self._find_model_file(symbol, model_type)
            if not model_path:
                raise FileNotFoundError(f"No model found for {symbol} with type {model_type}")
            
            # Load model based on file extension
            model = self._load_model_file(model_path)
            
            # Validate model
            validation_result = self._validate_model(model, symbol, model_type)
            if not validation_result.is_valid:
                raise ValueError(f"Model validation failed: {validation_result.errors}")
            
            # Cache the model
            self._model_cache[model_key] = model
            
            self._log_info(f"Successfully loaded model {model_key} from {model_path}")
            return model
            
        except Exception as e:
            self._log_error(f"Failed to load model {symbol}_{model_type}", exception=e)
            raise
    
    def get_model_metadata(self, symbol: str, model_type: str) -> ModelMetadata:
        """Get metadata for a model.
        
        Args:
            symbol: Trading symbol
            model_type: Type of model
            
        Returns:
            Model metadata
        """
        try:
            model_key = f"{symbol}_{model_type}"
            
            # Check cache first
            if model_key in self._metadata_cache:
                return self._metadata_cache[model_key]
            
            # Find metadata file
            metadata_path = self._find_metadata_file(symbol, model_type)
            if not metadata_path:
                # Create default metadata if file doesn't exist
                metadata = self._create_default_metadata(symbol, model_type)
            else:
                # Load metadata from file
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = self._dict_to_metadata(metadata_dict)
            
            # Cache metadata
            self._metadata_cache[model_key] = metadata
            
            return metadata
            
        except Exception as e:
            self._log_error(f"Failed to get metadata for {symbol}_{model_type}", exception=e)
            # Return default metadata on error
            return self._create_default_metadata(symbol, model_type)
    
    def validate_model_compatibility(self, model_metadata: ModelMetadata, feature_schema: Dict[str, Any]) -> ValidationResult:
        """Validate model-feature compatibility.
        
        Args:
            model_metadata: Model metadata
            feature_schema: Feature schema to validate against
            
        Returns:
            Validation result
        """
        try:
            errors = []
            warnings = []
            
            # Check feature compatibility
            model_features = set(model_metadata.features)
            schema_features = set(feature_schema.get('columns', []))
            
            missing_features = model_features - schema_features
            extra_features = schema_features - model_features
            
            if missing_features:
                errors.append(f"Missing required features: {list(missing_features)}")
            
            if extra_features:
                warnings.append(f"Extra features available: {list(extra_features)}")
            
            # Check feature count
            expected_count = len(model_metadata.features)
            actual_count = len(schema_features)
            
            if abs(actual_count - expected_count) > 10:  # Allow some tolerance
                warnings.append(f"Feature count mismatch: expected {expected_count}, got {actual_count}")
            
            # Check data types if available
            if 'dtypes' in feature_schema:
                schema_dtypes = feature_schema['dtypes']
                for feature in model_features.intersection(schema_features):
                    if feature in schema_dtypes:
                        # Basic type compatibility check
                        dtype = schema_dtypes[feature]
                        if 'object' in str(dtype) or 'string' in str(dtype):
                            warnings.append(f"Feature '{feature}' has non-numeric type: {dtype}")
            
            # Check model age
            if model_metadata.created_at:
                age_days = (datetime.now() - model_metadata.created_at).days
                if age_days > 30:  # Model older than 30 days
                    warnings.append(f"Model is {age_days} days old, consider retraining")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata={
                    'model_features': list(model_features),
                    'schema_features': list(schema_features),
                    'compatibility_score': len(model_features.intersection(schema_features)) / len(model_features) if model_features else 0
                }
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Compatibility validation failed: {str(e)}"],
                warnings=[],
                metadata={}
            )
    
    def save_model_metadata(self, symbol: str, model_type: str, metadata: ModelMetadata) -> bool:
        """Save model metadata to file.
        
        Args:
            symbol: Trading symbol
            model_type: Type of model
            metadata: Metadata to save
            
        Returns:
            True if successful
        """
        try:
            metadata_path = self._models_dir / f"{symbol}_{model_type}_metadata.json"
            
            # Convert metadata to dictionary
            metadata_dict = self._metadata_to_dict(metadata)
            
            # Save to file
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            # Update cache
            model_key = f"{symbol}_{model_type}"
            self._metadata_cache[model_key] = metadata
            
            self._log_info(f"Saved metadata for {model_key} to {metadata_path}")
            return True
            
        except Exception as e:
            self._log_error(f"Failed to save metadata for {symbol}_{model_type}", exception=e)
            return False
    
    def list_available_models(self) -> List[Dict[str, str]]:
        """List all available models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = []
            
            # Scan models directory for model files
            for file_path in self._models_dir.glob('*'):
                if file_path.is_file() and not file_path.name.endswith('_metadata.json'):
                    # Parse filename to extract symbol and model type
                    name_parts = file_path.stem.split('_')
                    if len(name_parts) >= 2:
                        symbol = name_parts[0]
                        model_type = '_'.join(name_parts[1:])
                        
                        models.append({
                            'symbol': symbol,
                            'model_type': model_type,
                            'file_path': str(file_path),
                            'file_size': file_path.stat().st_size,
                            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
            
            self._log_info(f"Found {len(models)} available models")
            return models
            
        except Exception as e:
            self._log_error("Failed to list available models", exception=e)
            return []
    
    def clear_cache(self) -> None:
        """Clear model and metadata cache."""
        self._model_cache.clear()
        self._metadata_cache.clear()
        self._log_info("Model cache cleared")
    
    def _find_model_file(self, symbol: str, model_type: str) -> Optional[Path]:
        """Find model file for given symbol and type."""
        # Common model file extensions
        extensions = ['.pkl', '.joblib', '.h5', '.pt', '.pth', '.onnx']
        
        for ext in extensions:
            model_path = self._models_dir / f"{symbol}_{model_type}{ext}"
            if model_path.exists():
                return model_path
        
        return None
    
    def _find_metadata_file(self, symbol: str, model_type: str) -> Optional[Path]:
        """Find metadata file for given symbol and type."""
        metadata_path = self._models_dir / f"{symbol}_{model_type}_metadata.json"
        return metadata_path if metadata_path.exists() else None
    
    def _load_model_file(self, model_path: Path) -> Any:
        """Load model from file based on extension."""
        extension = model_path.suffix.lower()
        
        if extension == '.pkl':
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif extension == '.joblib':
            import joblib
            return joblib.load(model_path)
        elif extension in ['.h5', '.keras']:
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            except ImportError:
                raise ImportError("TensorFlow not available for loading .h5 models")
        elif extension in ['.pt', '.pth']:
            try:
                import torch
                return torch.load(model_path, map_location='cpu')
            except ImportError:
                raise ImportError("PyTorch not available for loading .pt/.pth models")
        else:
            raise ValueError(f"Unsupported model file extension: {extension}")
    
    def _validate_model(self, model: Any, symbol: str, model_type: str) -> ValidationResult:
        """Validate loaded model."""
        errors = []
        warnings = []
        
        try:
            # Basic model validation
            if model is None:
                errors.append("Model is None")
                return ValidationResult(False, errors, warnings, {})
            
            # Check if model has required methods (basic duck typing)
            required_methods = ['predict']
            for method in required_methods:
                if not hasattr(model, method):
                    warnings.append(f"Model missing '{method}' method")
            
            # Model-specific validation
            if hasattr(model, '__class__'):
                model_class = model.__class__.__name__
                if 'sklearn' in str(type(model)):
                    # Scikit-learn model validation
                    if not hasattr(model, 'predict'):
                        errors.append("Scikit-learn model missing predict method")
                elif 'tensorflow' in str(type(model)) or 'keras' in str(type(model)):
                    # TensorFlow/Keras model validation
                    if not hasattr(model, 'predict'):
                        errors.append("TensorFlow model missing predict method")
                elif 'torch' in str(type(model)):
                    # PyTorch model validation
                    if not hasattr(model, 'forward') and not hasattr(model, '__call__'):
                        errors.append("PyTorch model missing forward method")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata={'model_type': str(type(model))}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Model validation failed: {str(e)}"],
                warnings=warnings,
                metadata={}
            )
    
    def _create_default_metadata(self, symbol: str, model_type: str) -> ModelMetadata:
        """Create default metadata for a model."""
        return ModelMetadata(
            model_type=model_type,
            symbol=symbol,
            version="1.0.0",
            features=[],
            created_at=datetime.now(),
            performance_metrics={},
            config={}
        )
    
    def _dict_to_metadata(self, metadata_dict: Dict[str, Any]) -> ModelMetadata:
        """Convert dictionary to ModelMetadata object."""
        # Handle datetime conversion
        created_at = metadata_dict.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()
        
        return ModelMetadata(
            model_type=metadata_dict.get('model_type', ''),
            symbol=metadata_dict.get('symbol', ''),
            version=metadata_dict.get('version', '1.0.0'),
            features=metadata_dict.get('features', []),
            created_at=created_at,
            performance_metrics=metadata_dict.get('performance_metrics', {}),
            config=metadata_dict.get('config', {})
        )
    
    def _metadata_to_dict(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Convert ModelMetadata object to dictionary."""
        return {
            'model_type': metadata.model_type,
            'symbol': metadata.symbol,
            'version': metadata.version,
            'features': metadata.features,
            'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
            'performance_metrics': metadata.performance_metrics,
            'config': metadata.config
        }