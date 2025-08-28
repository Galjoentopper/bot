"""Feature adapter for legacy FeatureEngine."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from ..core.interfaces import IFeatureManager, ValidationResult
from ..core.base_service import BaseService
from ..core.container import injectable
from ..data_pipeline.features import FeatureEngine


@injectable
class FeatureAdapter(BaseService, IFeatureManager):
    """Adapter that wraps legacy FeatureEngine to implement IFeatureManager."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the feature adapter.
        
        Args:
            config: Optional configuration for feature engine
        """
        super().__init__()
        self._feature_engine = FeatureEngine(config)
        self._config = config or {}
        self._feature_schema = None
        
    async def initialize(self) -> None:
        """Initialize the feature adapter."""
        await super().initialize()
        self._log_info("FeatureAdapter initialized")
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features from input data.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with generated features
        """
        try:
            if data.empty:
                self._log_warning("Empty DataFrame provided for feature generation")
                return data
                
            self._log_info(f"Generating features for {len(data)} records")
            features_df = self._feature_engine.generate_all_features(data)
            
            # Update feature schema based on generated features
            self._update_feature_schema(features_df)
            
            self._log_info(f"Generated {len(features_df.columns) - len(data.columns)} new features")
            return features_df
            
        except Exception as e:
            self._log_error(f"Feature generation failed: {e}")
            return data
            
    def validate_features(self, data: pd.DataFrame) -> bool:
        """Validate features against expected schema.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if features are valid
        """
        try:
            if data.empty:
                self._log_warning("Empty DataFrame provided for validation")
                return False
                
            # Basic validation checks
            # Check for infinite values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            inf_count = np.isinf(data[numeric_cols]).sum().sum()
            if inf_count > 0:
                self._log_error(f"Found {inf_count} infinite values in features")
                return False
                
            # Check for excessive NaN values
            nan_count = data.isnull().sum().sum()
            total_values = len(data) * len(data.columns)
            nan_ratio = nan_count / total_values if total_values > 0 else 0
            
            if nan_ratio > 0.5:  # More than 50% NaN values
                self._log_error(f"Excessive NaN values: {nan_ratio:.2%} of total values")
                return False
            elif nan_ratio > 0.1:  # More than 10% NaN values
                self._log_warning(f"High NaN ratio: {nan_ratio:.2%} of total values")
                
            # Check for required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self._log_error(f"Missing required columns: {missing_cols}")
                return False
                
            self._log_info("Feature validation passed")
            return True
            
        except Exception as e:
            self._log_error(f"Feature validation failed: {e}")
            return False
            
    def detect_schema_drift(self, current_features: pd.DataFrame, reference_schema: Dict[str, Any]) -> ValidationResult:
        """Detect schema drift in features."""
        try:
            errors = []
            warnings = []
            
            # Check column differences
            current_columns = set(current_features.columns)
            reference_columns = set(reference_schema.get('columns', []))
            
            missing_columns = reference_columns - current_columns
            extra_columns = current_columns - reference_columns
            
            if missing_columns:
                errors.append(f"Missing columns: {list(missing_columns)}")
            
            if extra_columns:
                warnings.append(f"Extra columns detected: {list(extra_columns)}")
            
            # Check data types if available
            if 'dtypes' in reference_schema:
                reference_dtypes = reference_schema['dtypes']
                for col in current_columns.intersection(reference_columns):
                    if col in reference_dtypes:
                        current_dtype = str(current_features[col].dtype)
                        reference_dtype = reference_dtypes[col]
                        if current_dtype != reference_dtype:
                            warnings.append(f"Data type drift in column '{col}': {current_dtype} vs {reference_dtype}")
            
            # Check feature count ranges
            if 'feature_count' in reference_schema:
                expected_count = reference_schema['feature_count']
                actual_count = len(current_features.columns)
                if abs(actual_count - expected_count) > 5:  # Allow some tolerance
                    warnings.append(f"Feature count drift: {actual_count} vs expected {expected_count}")
            
            drift_detected = len(errors) > 0 or len(warnings) > 2  # Significant drift if many warnings
            
            return ValidationResult(
                is_valid=not drift_detected,
                errors=errors,
                warnings=warnings,
                metadata={
                    'current_columns': list(current_columns),
                    'reference_columns': list(reference_columns),
                    'drift_detected': drift_detected
                }
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema drift detection failed: {str(e)}"],
                warnings=[],
                metadata={}
            )
            
    def get_feature_schema(self) -> Dict[str, Any]:
        """Get current feature schema.
        
        Returns:
            Feature schema dictionary
        """
        if self._feature_schema is None:
            return self._generate_default_schema()
        return self._feature_schema.copy()
        
    def save_feature_schema(self, schema: Dict[str, Any], path: str) -> bool:
        """Save feature schema to file.
        
        Args:
            schema: Feature schema to save
            path: Path to save schema
            
        Returns:
            True if save was successful
        """
        try:
            import json
            schema_path = Path(path)
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
                
            self._feature_schema = schema
            self._log_info(f"Feature schema saved to {path}")
            return True
            
        except Exception as e:
            self._log_error(f"Failed to save feature schema: {e}")
            return False
            
    def load_feature_schema(self, path: str) -> bool:
        """Load feature schema from file.
        
        Args:
            path: Path to load schema from
            
        Returns:
            True if load was successful
        """
        try:
            import json
            schema_path = Path(path)
            
            if not schema_path.exists():
                self._log_warning(f"Schema file not found: {path}")
                return False
                
            with open(schema_path, 'r') as f:
                schema = json.load(f)
                
            self._feature_schema = schema
            self._log_info(f"Feature schema loaded from {path}")
            return True
            
        except Exception as e:
            self._log_error(f"Failed to load feature schema: {e}")
            return False
            
    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect feature drift between reference and current data.
        
        Args:
            reference_data: Reference DataFrame
            current_data: Current DataFrame to compare
            
        Returns:
            Drift detection results
        """
        try:
            from scipy import stats
            
            drift_results = {
                'has_drift': False,
                'drift_features': [],
                'drift_scores': {},
                'summary': {}
            }
            
            # Get common numeric columns
            ref_numeric = reference_data.select_dtypes(include=[np.number]).columns
            cur_numeric = current_data.select_dtypes(include=[np.number]).columns
            common_cols = list(set(ref_numeric) & set(cur_numeric))
            
            if not common_cols:
                self._log_warning("No common numeric columns for drift detection")
                return drift_results
                
            drift_threshold = 0.05  # p-value threshold
            drift_count = 0
            
            for col in common_cols:
                try:
                    # Remove NaN values
                    ref_values = reference_data[col].dropna()
                    cur_values = current_data[col].dropna()
                    
                    if len(ref_values) < 10 or len(cur_values) < 10:
                        continue
                        
                    # Kolmogorov-Smirnov test for distribution drift
                    ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
                    
                    drift_results['drift_scores'][col] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'has_drift': p_value < drift_threshold
                    }
                    
                    if p_value < drift_threshold:
                        drift_results['drift_features'].append(col)
                        drift_count += 1
                        
                except Exception as e:
                    self._log_warning(f"Drift detection failed for column {col}: {e}")
                    continue
                    
            drift_results['has_drift'] = drift_count > 0
            drift_results['summary'] = {
                'total_features_tested': len(common_cols),
                'features_with_drift': drift_count,
                'drift_ratio': drift_count / len(common_cols) if common_cols else 0
            }
            
            self._log_info(f"Drift detection completed: {drift_count}/{len(common_cols)} features show drift")
            return drift_results
            
        except Exception as e:
            self._log_error(f"Drift detection failed: {e}")
            return {'has_drift': False, 'error': str(e)}
            
    def _update_feature_schema(self, data: pd.DataFrame) -> None:
        """Update feature schema based on generated features."""
        try:
            schema = {
                'version': '1.0',
                'features': {},
                'metadata': {
                    'total_features': len(data.columns),
                    'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
                    'generated_at': pd.Timestamp.now().isoformat()
                }
            }
            
            for col in data.columns:
                col_info = {
                    'dtype': str(data[col].dtype),
                    'nullable': data[col].isnull().any(),
                    'unique_count': data[col].nunique()
                }
                
                if pd.api.types.is_numeric_dtype(data[col]):
                    col_info.update({
                        'min': float(data[col].min()) if not data[col].isnull().all() else None,
                        'max': float(data[col].max()) if not data[col].isnull().all() else None,
                        'mean': float(data[col].mean()) if not data[col].isnull().all() else None
                    })
                    
                schema['features'][col] = col_info
                
            self._feature_schema = schema
            
        except Exception as e:
            self._log_error(f"Failed to update feature schema: {e}")
            
    def _generate_default_schema(self) -> Dict[str, Any]:
        """Generate default feature schema."""
        return {
            'version': '1.0',
            'features': {},
            'metadata': {
                'total_features': 0,
                'numeric_features': 0,
                'generated_at': pd.Timestamp.now().isoformat()
            }
        }