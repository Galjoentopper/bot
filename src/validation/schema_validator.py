"""Schema validation and drift detection using Great Expectations."""

import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import great_expectations as gx
    from great_expectations.core.batch import RuntimeBatchRequest
    from great_expectations.checkpoint import SimpleCheckpoint
    HAS_GX = True
except ImportError:
    HAS_GX = False
    logging.warning("Great Expectations not available. Schema validation will use basic checks.")

from ..utils.logger import Logger

class SchemaValidator:
    """Schema validation and drift detection for trading models."""
    
    def __init__(self, config_dir: str = "./validation", models_dir: str = "./models"):
        self.config_dir = Path(config_dir)
        self.models_dir = Path(models_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger("SchemaValidator")
        self.has_gx = HAS_GX
        
        # Initialize Great Expectations context if available
        if self.has_gx:
            self._init_gx_context()
        
        # Default schema definitions for each model type (fallback)
        self.default_model_schemas = {
            'gru': {
                'expected_features': 119,
                'sequence_length': 20,
                'feature_types': ['float64', 'float32'],
                'required_columns': ['close', 'high', 'low', 'open', 'volume']
            },
            'lightgbm': {
                'expected_features': 113,
                'feature_types': ['float64', 'float32'],
                'required_columns': ['close', 'high', 'low', 'open', 'volume']
            },
            'ppo': {
                'expected_features': 13,  # 10 market + 3 portfolio
                'sequence_length': 32,  # Updated default
                'feature_types': ['float64', 'float32'],
                'required_columns': ['close', 'high', 'low', 'open', 'volume']
            }
        }
        
        # Load model-specific schemas (prioritize over defaults)
        self.model_schemas = self._load_model_specific_schemas()
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'feature_count_drift': 0.05,  # 5% change in feature count
            'statistical_drift': 2.0,     # 2 standard deviations
            'distribution_drift': 0.1,    # KL divergence threshold
            'missing_data_threshold': 0.1  # 10% missing data threshold
        }
        
    def _load_model_specific_schemas(self) -> Dict[str, Dict]:
        """Load model-specific schemas from metadata files, fallback to global mapping."""
        schemas = self.default_model_schemas.copy()
        
        # Try to load from model-specific metadata files first
        for model_type in ['gru', 'lightgbm', 'ppo']:
            for symbol in ['ADAEUR', 'BTCEUR', 'ETHEUR']:
                model_key = f"{model_type}_{symbol}"
                schema = self._load_schema_from_model_metadata(symbol, model_type)
                if schema:
                    schemas[model_type] = schema
                    self.logger.logger.debug(f"Loaded schema for {model_type} from model metadata")
                    break  # Use first found schema for this model type
        
        # Fallback to global feature_mapping.json
        global_schemas = self._load_schemas_from_feature_mapping()
        for model_type, schema in global_schemas.items():
            if model_type not in schemas or schemas[model_type] == self.default_model_schemas.get(model_type):
                schemas[model_type] = schema
                self.logger.logger.debug(f"Using global schema for {model_type}")
        
        return schemas
    
    def _load_schema_from_model_metadata(self, symbol: str, model_type: str) -> Optional[Dict]:
        """Load schema from model-specific metadata files."""
        # Search paths for model metadata
        search_paths = [
            self.models_dir / symbol / model_type,
            self.models_dir / 'packaged_models' / f"{model_type}_{symbol}",
            self.models_dir / 'imported_models' / f"{model_type}_{symbol}",
            self.models_dir / 'best_walkforward' / symbol / model_type,
            self.models_dir / 'latest_models' / symbol / model_type,
            self.models_dir / 'unified_artifacts' / symbol / model_type
        ]
        
        for path in search_paths:
            # Try features.json first
            features_file = path / 'features.json'
            if features_file.exists():
                try:
                    with open(features_file, 'r') as f:
                        features_data = json.load(f)
                    
                    if 'feature_names' in features_data:
                        expected_features = len(features_data['feature_names'])
                        schema = self.default_model_schemas.get(model_type, {}).copy()
                        schema['expected_features'] = expected_features
                        schema['feature_names'] = features_data['feature_names']
                        return schema
                except Exception as e:
                    self.logger.logger.debug(f"Failed to load {features_file}: {e}")
            
            # Try metadata.json
            metadata_file = path / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if 'feature_count' in metadata:
                        expected_features = metadata['feature_count']
                        schema = self.default_model_schemas.get(model_type, {}).copy()
                        schema['expected_features'] = expected_features
                        if 'sequence_length' in metadata:
                            schema['sequence_length'] = metadata['sequence_length']
                        return schema
                except Exception as e:
                    self.logger.logger.debug(f"Failed to load {metadata_file}: {e}")
        
        return None
    
    def _load_schemas_from_feature_mapping(self) -> Dict[str, Dict]:
        """Load schemas from global feature_mapping.json as fallback."""
        schemas = {}
        
        # Try multiple locations for feature_mapping.json
        mapping_paths = [
            Path('feature_mapping.json'),
            self.config_dir / 'feature_mapping.json',
            self.models_dir / 'feature_mapping.json',
            Path.cwd() / 'feature_mapping.json'
        ]
        
        for mapping_path in mapping_paths:
            if mapping_path.exists():
                try:
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                    
                    # Extract schemas from feature_mapping.json
                    if 'feature_counts' in mapping_data:
                        for model_key, count in mapping_data['feature_counts'].items():
                            if '_' in model_key:
                                model_type = model_key.split('_')[0]
                                if model_type not in schemas:
                                    schema = self.default_model_schemas.get(model_type, {}).copy()
                                    schema['expected_features'] = count
                                    schemas[model_type] = schema
                    
                    self.logger.logger.debug(f"Loaded schemas from {mapping_path}")
                    break
                except Exception as e:
                    self.logger.logger.debug(f"Failed to load {mapping_path}: {e}")
        
        return schemas
        
    def _init_gx_context(self):
        """Initialize Great Expectations context."""
        try:
            context_dir = self.config_dir / "gx"
            context_dir.mkdir(exist_ok=True)
            
            # Initialize or get existing context
            if (context_dir / "great_expectations.yml").exists():
                self.gx_context = gx.get_context(context_root_dir=str(context_dir))
            else:
                self.gx_context = gx.get_context(context_root_dir=str(context_dir))
                self._create_default_expectations()
                
        except Exception as e:
            self.logger.logger.warning(f"Failed to initialize Great Expectations: {e}")
            self.has_gx = False
            
    def _create_default_expectations(self):
        """Create default expectation suites for each model type."""
        if not self.has_gx:
            return
            
        for model_type, schema in self.model_schemas.items():
            suite_name = f"{model_type}_validation_suite"
            
            try:
                # Create expectation suite
                suite = self.gx_context.create_expectation_suite(
                    expectation_suite_name=suite_name,
                    overwrite_existing=True
                )
                
                # Add basic expectations
                suite.add_expectation(
                    gx.expectations.ExpectTableRowCountToBeBetween(
                        min_value=1,
                        max_value=None
                    )
                )
                
                # Feature count expectation
                expected_features = schema['expected_features']
                suite.add_expectation(
                    gx.expectations.ExpectTableColumnCountToEqual(
                        value=expected_features
                    )
                )
                
                # Data type expectations
                for col_type in schema['feature_types']:
                    suite.add_expectation(
                        gx.expectations.ExpectColumnValuesToBeOfType(
                            column="*",  # Apply to all columns
                            type_=col_type
                        )
                    )
                
                # No null values expectation
                suite.add_expectation(
                    gx.expectations.ExpectColumnValuesToNotBeNull(
                        column="*"
                    )
                )
                
                # Save suite
                self.gx_context.save_expectation_suite(suite)
                
            except Exception as e:
                self.logger.logger.error(f"Failed to create expectation suite for {model_type}: {e}")
                
    def validate_model_input(self, data: pd.DataFrame, model_type: str, symbol: str) -> Dict[str, Any]:
        """Validate model input data against schema."""
        validation_result = {
            'valid': True,
            'model_type': model_type,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Get schema for model type
        schema = self.model_schemas.get(model_type)
        if not schema:
            validation_result['errors'].append(f"Unknown model type: {model_type}")
            validation_result['valid'] = False
            return validation_result
            
        # Basic validation
        self._validate_basic_schema(data, schema, validation_result)
        
        # Great Expectations validation if available
        if self.has_gx:
            self._validate_with_gx(data, model_type, validation_result)
        
        # Statistical validation
        self._validate_statistical_properties(data, model_type, symbol, validation_result)
        
        return validation_result
        
    def _validate_basic_schema(self, data: pd.DataFrame, schema: Dict, result: Dict):
        """Perform basic schema validation."""
        # Check feature count
        expected_features = schema['expected_features']
        actual_features = len(data.columns)
        
        if actual_features != expected_features:
            error_msg = f"Feature count mismatch: expected {expected_features}, got {actual_features}"
            result['errors'].append(error_msg)
            result['valid'] = False
            
        result['metrics']['feature_count'] = actual_features
        result['metrics']['expected_feature_count'] = expected_features
        
        # Check data types
        valid_types = schema['feature_types']
        invalid_types = []
        
        for col in data.columns:
            if str(data[col].dtype) not in valid_types:
                invalid_types.append(f"{col}: {data[col].dtype}")
                
        if invalid_types:
            result['warnings'].append(f"Invalid data types: {invalid_types}")
            
        # Check for missing values
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            missing_ratio = total_missing / (len(data) * len(data.columns))
            result['metrics']['missing_data_ratio'] = missing_ratio
            
            if missing_ratio > self.drift_thresholds['missing_data_threshold']:
                result['errors'].append(f"Too much missing data: {missing_ratio:.2%}")
                result['valid'] = False
            else:
                result['warnings'].append(f"Missing data detected: {missing_ratio:.2%}")
                
        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_counts > 0:
            result['errors'].append(f"Infinite values detected: {inf_counts}")
            result['valid'] = False
            
    def _validate_with_gx(self, data: pd.DataFrame, model_type: str, result: Dict):
        """Validate using Great Expectations."""
        try:
            suite_name = f"{model_type}_validation_suite"
            
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name=f"{model_type}_data",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": "default_identifier"}
            )
            
            # Create validator
            validator = self.gx_context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            # Run validation
            validation_results = validator.validate()
            
            # Process results
            if not validation_results.success:
                result['valid'] = False
                for failed_expectation in validation_results.results:
                    if not failed_expectation.success:
                        result['errors'].append(
                            f"GX Validation failed: {failed_expectation.expectation_config.expectation_type}"
                        )
                        
            result['metrics']['gx_success_rate'] = validation_results.statistics.get('success_percent', 0)
            
        except Exception as e:
            result['warnings'].append(f"Great Expectations validation failed: {e}")
            
    def _validate_statistical_properties(self, data: pd.DataFrame, model_type: str, symbol: str, result: Dict):
        """Validate statistical properties and detect drift."""
        try:
            # Load historical statistics if available
            stats_file = self.config_dir / f"{model_type}_{symbol}_stats.json"
            
            current_stats = self._calculate_statistics(data)
            result['metrics']['current_stats'] = current_stats
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    historical_stats = json.load(f)
                    
                # Detect drift
                drift_detected = self._detect_statistical_drift(current_stats, historical_stats)
                
                if drift_detected:
                    result['warnings'].append("Statistical drift detected")
                    result['metrics']['drift_detected'] = True
                else:
                    result['metrics']['drift_detected'] = False
                    
            # Save current statistics
            with open(stats_file, 'w') as f:
                json.dump(current_stats, f, indent=2)
                
        except Exception as e:
            result['warnings'].append(f"Statistical validation failed: {e}")
            
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical properties of data."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'row_count': len(data),
            'column_count': len(data.columns),
            'mean': numeric_data.mean().to_dict(),
            'std': numeric_data.std().to_dict(),
            'min': numeric_data.min().to_dict(),
            'max': numeric_data.max().to_dict(),
            'skew': numeric_data.skew().to_dict(),
            'kurtosis': numeric_data.kurtosis().to_dict()
        }
        
        return stats
        
    def _detect_statistical_drift(self, current_stats: Dict, historical_stats: Dict) -> bool:
        """Detect statistical drift between current and historical data."""
        drift_detected = False
        threshold = self.drift_thresholds['statistical_drift']
        
        # Check mean drift
        for feature in current_stats['mean']:
            if feature in historical_stats['mean']:
                current_mean = current_stats['mean'][feature]
                historical_mean = historical_stats['mean'][feature]
                historical_std = historical_stats['std'].get(feature, 1.0)
                
                if historical_std > 0:
                    z_score = abs(current_mean - historical_mean) / historical_std
                    if z_score > threshold:
                        drift_detected = True
                        break
                        
        return drift_detected
        
    def create_validation_report(self, validation_results: List[Dict]) -> Dict:
        """Create comprehensive validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_validations': len(validation_results),
            'successful_validations': sum(1 for r in validation_results if r['valid']),
            'failed_validations': sum(1 for r in validation_results if not r['valid']),
            'drift_detections': sum(1 for r in validation_results if r.get('metrics', {}).get('drift_detected', False)),
            'summary': {},
            'details': validation_results
        }
        
        # Calculate success rate
        if report['total_validations'] > 0:
            report['success_rate'] = report['successful_validations'] / report['total_validations']
        else:
            report['success_rate'] = 0.0
            
        # Group by model type
        by_model = {}
        for result in validation_results:
            model_type = result['model_type']
            if model_type not in by_model:
                by_model[model_type] = {'valid': 0, 'invalid': 0, 'drift': 0}
                
            if result['valid']:
                by_model[model_type]['valid'] += 1
            else:
                by_model[model_type]['invalid'] += 1
                
            if result.get('metrics', {}).get('drift_detected', False):
                by_model[model_type]['drift'] += 1
                
        report['summary'] = by_model
        
        return report
        
    def save_validation_report(self, report: Dict, filename: Optional[str] = None):
        """Save validation report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
            
        report_path = self.config_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.logger.info(f"Validation report saved to {report_path}")
        
        return report_path