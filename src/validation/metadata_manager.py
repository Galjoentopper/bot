"""Automated metadata regeneration and hygiene processes."""

import os
import json
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class ModelMetadata:
    """Model metadata structure."""
    symbol: str
    model_type: str
    version: str
    created_at: str
    python_version: str
    dependencies: Dict[str, str]
    performance_metrics: Dict[str, float]
    file_path: str
    hash_md5: str
    source: str
    validated: bool = False
    feature_count: Optional[int] = None
    feature_names: Optional[List[str]] = None
    training_data_hash: Optional[str] = None
    last_validated: Optional[str] = None
    validation_errors: Optional[List[str]] = None

class MetadataManager:
    """Manages model metadata lifecycle and hygiene processes."""
    
    def __init__(self, models_dir: str, config: Dict[str, Any] = None):
        self.models_dir = Path(models_dir)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Metadata hygiene settings
        self.max_model_age_days = self.config.get('max_model_age_days', 30)
        self.validation_interval_hours = self.config.get('validation_interval_hours', 24)
        self.auto_cleanup = self.config.get('auto_cleanup', True)
        
    def regenerate_all_metadata(self) -> Dict[str, Any]:
        """Regenerate metadata for all models in the models directory."""
        results = {
            'processed': 0,
            'updated': 0,
            'errors': [],
            'warnings': []
        }
        
        self.logger.info("Starting metadata regeneration for all models")
        
        # Find all model files
        model_files = []
        for pattern in ['*.pkl', '*.joblib', '*.zip']:
            model_files.extend(self.models_dir.rglob(pattern))
        
        for model_file in model_files:
            try:
                results['processed'] += 1
                
                # Skip if not a model file
                if self._is_metadata_file(model_file):
                    continue
                    
                # Generate metadata
                metadata = self._generate_metadata_for_file(model_file)
                if metadata:
                    # Save metadata
                    metadata_path = self._get_metadata_path(model_file)
                    self._save_metadata(metadata, metadata_path)
                    results['updated'] += 1
                    self.logger.info(f"Updated metadata for {model_file.name}")
                else:
                    results['warnings'].append(f"Could not generate metadata for {model_file}")
                    
            except Exception as e:
                error_msg = f"Error processing {model_file}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        self.logger.info(f"Metadata regeneration complete: {results['updated']}/{results['processed']} updated")
        return results
    
    def validate_metadata_hygiene(self) -> Dict[str, Any]:
        """Validate metadata hygiene across all models."""
        results = {
            'total_models': 0,
            'valid_metadata': 0,
            'outdated_models': [],
            'missing_metadata': [],
            'invalid_metadata': [],
            'recommendations': []
        }
        
        # Find all model files
        model_files = []
        for pattern in ['*.pkl', '*.joblib', '*.zip']:
            model_files.extend(self.models_dir.rglob(pattern))
        
        for model_file in model_files:
            if self._is_metadata_file(model_file):
                continue
                
            results['total_models'] += 1
            metadata_path = self._get_metadata_path(model_file)
            
            if not metadata_path.exists():
                results['missing_metadata'].append(str(model_file))
                continue
            
            try:
                metadata = self._load_metadata(metadata_path)
                
                # Validate metadata structure
                validation_result = self._validate_metadata_structure(metadata)
                if not validation_result['is_valid']:
                    results['invalid_metadata'].append({
                        'file': str(model_file),
                        'errors': validation_result['errors']
                    })
                    continue
                
                # Check model age
                if self._is_model_outdated(metadata):
                    results['outdated_models'].append({
                        'file': str(model_file),
                        'age_days': self._get_model_age_days(metadata)
                    })
                
                results['valid_metadata'] += 1
                
            except Exception as e:
                results['invalid_metadata'].append({
                    'file': str(model_file),
                    'errors': [f"Failed to load metadata: {str(e)}"]
                })
        
        # Generate recommendations
        if results['missing_metadata']:
            results['recommendations'].append("Run metadata regeneration for models with missing metadata")
        
        if results['outdated_models']:
            results['recommendations'].append(f"Consider retraining {len(results['outdated_models'])} outdated models")
        
        if results['invalid_metadata']:
            results['recommendations'].append("Fix invalid metadata files or regenerate them")
        
        return results
    
    def cleanup_outdated_models(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up outdated models and their metadata."""
        results = {
            'candidates_for_removal': [],
            'removed': [],
            'errors': []
        }
        
        if not self.auto_cleanup:
            self.logger.info("Auto cleanup is disabled")
            return results
        
        # Find outdated models
        hygiene_results = self.validate_metadata_hygiene()
        
        for outdated_model in hygiene_results['outdated_models']:
            model_path = Path(outdated_model['file'])
            age_days = outdated_model['age_days']
            
            if age_days > self.max_model_age_days:
                results['candidates_for_removal'].append({
                    'file': str(model_path),
                    'age_days': age_days
                })
                
                if not dry_run:
                    try:
                        # Remove model file
                        model_path.unlink()
                        
                        # Remove metadata file
                        metadata_path = self._get_metadata_path(model_path)
                        if metadata_path.exists():
                            metadata_path.unlink()
                        
                        results['removed'].append(str(model_path))
                        self.logger.info(f"Removed outdated model: {model_path}")
                        
                    except Exception as e:
                        error_msg = f"Failed to remove {model_path}: {str(e)}"
                        results['errors'].append(error_msg)
                        self.logger.error(error_msg)
        
        return results
    
    def _generate_metadata_for_file(self, model_file: Path) -> Optional[ModelMetadata]:
        """Generate metadata for a specific model file."""
        try:
            # Extract basic info from path
            parts = model_file.parts
            symbol = None
            model_type = None
            
            # Try to extract symbol and model type from path
            for i, part in enumerate(parts):
                if part in ['gru', 'lightgbm', 'ppo']:
                    model_type = part
                    if i + 1 < len(parts):
                        symbol = parts[i + 1]
                    break
            
            if not symbol or not model_type:
                # Try alternative extraction
                if len(parts) >= 2:
                    symbol = parts[-2] if parts[-2] not in ['gru', 'lightgbm', 'ppo'] else 'UNKNOWN'
                    model_type = 'unknown'
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(model_file)
            
            # Get file stats
            stat = model_file.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            # Try to load model to get additional info
            feature_count = None
            feature_names = None
            
            try:
                if model_file.suffix == '.pkl':
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Extract feature information if available
                    if hasattr(model, 'n_features_in_'):
                        feature_count = model.n_features_in_
                    elif hasattr(model, 'feature_names_in_'):
                        feature_names = list(model.feature_names_in_)
                        feature_count = len(feature_names)
            except Exception as e:
                self.logger.debug(f"Could not load model {model_file} for feature extraction: {e}")
            
            metadata = ModelMetadata(
                symbol=symbol or 'UNKNOWN',
                model_type=model_type or 'unknown',
                version='1.0',
                created_at=created_at,
                python_version='3.8+',  # Default
                dependencies={},  # Would need to be populated separately
                performance_metrics={},  # Would need to be populated separately
                file_path=str(model_file),
                hash_md5=file_hash,
                source='regenerated',
                validated=False,
                feature_count=feature_count,
                feature_names=feature_names,
                last_validated=datetime.now().isoformat()
            )
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to generate metadata for {model_file}: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_metadata_path(self, model_file: Path) -> Path:
        """Get the metadata file path for a model file."""
        return model_file.parent / f"{model_file.stem}_metadata.json"
    
    def _is_metadata_file(self, file_path: Path) -> bool:
        """Check if a file is a metadata file."""
        return file_path.name.endswith('_metadata.json') or file_path.name == 'metadata.json'
    
    def _save_metadata(self, metadata: ModelMetadata, path: Path) -> None:
        """Save metadata to file."""
        with open(path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        """Load metadata from file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _validate_metadata_structure(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata structure."""
        required_fields = ['symbol', 'model_type', 'created_at', 'file_path', 'hash_md5']
        errors = []
        
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def _is_model_outdated(self, metadata: Dict[str, Any]) -> bool:
        """Check if a model is outdated based on its creation date."""
        try:
            created_at = datetime.fromisoformat(metadata['created_at'])
            age = datetime.now() - created_at
            return age.days > self.max_model_age_days
        except Exception:
            return False
    
    def _get_model_age_days(self, metadata: Dict[str, Any]) -> int:
        """Get model age in days."""
        try:
            created_at = datetime.fromisoformat(metadata['created_at'])
            age = datetime.now() - created_at
            return age.days
        except Exception:
            return 0

def create_metadata_manager(models_dir: str, config: Dict[str, Any] = None) -> MetadataManager:
    """Create a metadata manager instance."""
    return MetadataManager(models_dir, config)