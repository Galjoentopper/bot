#!/usr/bin/env python3
"""
Feature Generation from Model Metadata

This script analyzes model metadata to generate appropriate features
that match the expected input format for each model.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureGenerator:
    """Generate features based on model metadata"""
    
    def __init__(self, models_dir: str, output_dir: str):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Feature configuration based on metadata analysis
        self.feature_config = {
            'technical_indicators': {
                'rsi': {'period': 14},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger_bands': {'period': 20, 'std': 2},
                'sma': {'periods': [5, 10, 20, 50]},
                'ema': {'periods': [5, 10, 20, 50]},
                'stochastic': {'k_period': 14, 'd_period': 3}
            },
            'price_features': {
                'returns': {'periods': [1, 5, 10, 20]},
                'volatility': {'window': 20},
                'price_ratios': ['high_low', 'close_open'],
                'price_position': ['close_to_high', 'close_to_low']
            },
            'volume_features': {
                'volume_sma': {'periods': [5, 10, 20]},
                'volume_ratio': {'period': 20},
                'price_volume': ['vwap', 'volume_price_trend']
            },
            'market_features': {
                'time_features': ['hour', 'day_of_week', 'month'],
                'market_regime': ['trend_strength', 'volatility_regime']
            }
        }
    
    def scan_models(self) -> Dict[str, Dict[str, Any]]:
        """Scan models directory and extract metadata"""
        models_info = {}
        
        logger.info(f"Scanning models directory: {self.models_dir}")
        
        if not self.models_dir.exists():
            logger.error(f"Models directory not found: {self.models_dir}")
            return models_info
        
        # Look for model directories and metadata files
        # Structure: models/model_type/symbol/imported_metadata.json
        for model_type_path in self.models_dir.iterdir():
            if model_type_path.is_dir():
                model_type = model_type_path.name
                logger.debug(f"Scanning model type: {model_type}")
                
                for symbol_path in model_type_path.iterdir():
                    if symbol_path.is_dir():
                        symbol = symbol_path.name
                        metadata_file = symbol_path / 'imported_metadata.json'
                        
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                
                                model_name = f"{model_type}_{symbol}"
                                models_info[model_name] = {
                                    'path': str(symbol_path),
                                    'metadata': metadata,
                                    'type': model_type,
                                    'symbol': symbol,
                                    'features': metadata.get('training_config', {}).get('features', {})
                                }
                                
                                logger.info(f"Found model: {model_name} (type: {model_type}, symbol: {symbol})")
                                
                            except Exception as e:
                                logger.error(f"Error reading metadata for {symbol_path}: {e}")
        
        return models_info
    
    def analyze_feature_requirements(self, models_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feature requirements across all models"""
        feature_summary = {
            'required_features': set(),
            'feature_types': set(),
            'max_features': 0,
            'selection_methods': set(),
            'symbols': set(),
            'model_types': set()
        }
        
        for model_name, info in models_info.items():
            metadata = info['metadata']
            features_config = info['features']
            
            # Extract feature information
            if 'feature_columns' in metadata and metadata['feature_columns']:
                feature_summary['required_features'].update(metadata['feature_columns'])
            
            # Extract from training config
            if features_config:
                for feature_type in features_config.keys():
                    if feature_type not in ['max_features', 'selection_method']:
                        feature_summary['feature_types'].add(feature_type)
                
                max_features = features_config.get('max_features', 0)
                if max_features > feature_summary['max_features']:
                    feature_summary['max_features'] = max_features
                
                selection_method = features_config.get('selection_method')
                if selection_method:
                    feature_summary['selection_methods'].add(selection_method)
            
            # Track symbols and model types
            feature_summary['symbols'].add(info['symbol'])
            feature_summary['model_types'].add(info['type'])
        
        # Convert sets to lists for JSON serialization
        for key in feature_summary:
            if isinstance(feature_summary[key], set):
                feature_summary[key] = list(feature_summary[key])
        
        return feature_summary
    
    def generate_feature_config(self, feature_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unified feature configuration"""
        config = {
            'version': '1.0',
            'generated_at': str(Path().cwd()),
            'models_analyzed': len(feature_summary.get('symbols', [])),
            'feature_generation': {
                'enabled_features': {},
                'max_features': max(feature_summary.get('max_features', 50), 50),
                'selection_method': 'mutual_info' if 'mutual_info' in feature_summary.get('selection_methods', []) else 'variance',
                'symbols': feature_summary.get('symbols', []),
                'model_types': feature_summary.get('model_types', [])
            }
        }
        
        # Enable features based on detected requirements
        detected_types = set(feature_summary.get('feature_types', []))
        
        for feature_category, feature_config in self.feature_config.items():
            if feature_category in detected_types or not detected_types:
                config['feature_generation']['enabled_features'][feature_category] = feature_config
        
        # Ensure we have at least basic features
        if not config['feature_generation']['enabled_features']:
            config['feature_generation']['enabled_features'] = {
                'technical_indicators': self.feature_config['technical_indicators'],
                'price_features': self.feature_config['price_features']
            }
        
        return config
    
    def save_feature_config(self, config: Dict[str, Any]) -> None:
        """Save feature configuration to output directory"""
        config_file = self.output_dir / 'feature_config.json'
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Feature configuration saved to: {config_file}")
            
            # Also save as YAML for easier reading
            yaml_file = self.output_dir / 'feature_config.yaml'
            with open(yaml_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Feature configuration also saved as: {yaml_file}")
            
        except Exception as e:
            logger.error(f"Error saving feature configuration: {e}")
    
    def generate_feature_mapping(self, models_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate feature mapping for each model based on actual metadata requirements"""
        mapping = {
            'models': {},
            'global_features': list(self.feature_config.keys()),
            'feature_counts': {}  # Track expected feature counts per model
        }
        
        for model_name, info in models_info.items():
            model_features = info['features']
            model_mapping = {
                'model_type': info['type'],
                'symbol': info['symbol'],
                'required_features': [],
                'feature_config': model_features,
                'expected_feature_count': None
            }
            
            # Extract actual feature requirements from metadata
            metadata = info.get('metadata', {})
            
            # Try to get feature count from various metadata sources
            expected_count = None
            
            # Method 1: Check if feature_columns has actual features
            if 'feature_columns' in metadata and metadata['feature_columns']:
                expected_count = len(metadata['feature_columns'])
                model_mapping['required_features'] = metadata['feature_columns']
                logger.info(f"Model {model_name}: Found {expected_count} explicit feature columns")
            
            # Method 2: Check training_config for max_features (DISABLED - using actual model requirements)
            # elif 'training_config' in metadata and 'features' in metadata['training_config']:
            #     features_config = metadata['training_config']['features']
            #     max_features = features_config.get('max_features', None)
            #     if max_features and max_features > 0:
            #         expected_count = max_features
            #         logger.info(f"Model {model_name}: Using max_features={expected_count} from training config")
            
            # Method 3: Check model_config for input dimensions
            elif 'model_config' in metadata:
                model_config = metadata['model_config']
                # Look for input_dim, n_features, or similar
                for key in ['input_dim', 'n_features', 'input_size', 'feature_dim']:
                    if key in model_config and model_config[key]:
                        expected_count = model_config[key]
                        logger.info(f"Model {model_name}: Using {key}={expected_count} from model config")
                        break
            
            # Method 4: Try to infer from actual model files or use dynamic feature counting
            if expected_count is None:
                logger.warning(f"Model {model_name}: No explicit feature count found, attempting dynamic detection")
                
                # Try to load actual model and inspect its expected input shape
                model_path = Path(info['path'])
                inferred_count = self._infer_feature_count_from_model(model_path, info['type'])
                
                if inferred_count:
                    expected_count = inferred_count
                    logger.info(f"Model {model_name}: Inferred {expected_count} features from model inspection")
                else:
                    # Fallback to reasonable defaults with logging
                    if info['type'] in ['lightgbm', 'xgboost', 'random_forest']:
                        expected_count = 100  # Conservative default, will be adjusted by feature selector
                        model_mapping['required_features'] = ['technical_indicators', 'price_features']
                        logger.info(f"Model {model_name}: Using conservative default {expected_count} features for tree-based model")
                    elif info['type'] in ['lstm', 'gru', 'rnn']:
                        expected_count = 100  # Conservative default, will be adjusted by feature selector
                        model_mapping['required_features'] = ['price_features', 'technical_indicators', 'volume_features']
                        logger.info(f"Model {model_name}: Using conservative default {expected_count} features for sequence model")
                    elif info['type'] in ['ppo', 'dqn', 'a3c']:
                        expected_count = 13  # PPO has specific observation space requirements
                        model_mapping['required_features'] = ['market_features', 'price_features', 'technical_indicators']
                        logger.info(f"Model {model_name}: Using specific observation space {expected_count} features for RL model")
                    else:
                        expected_count = 100  # Conservative fallback
                        model_mapping['required_features'] = ['technical_indicators', 'price_features']
                        logger.info(f"Model {model_name}: Using conservative default {expected_count} features for unknown model type")
            
            # If we still don't have required_features, generate based on enabled features
            if not model_mapping['required_features'] and 'training_config' in metadata:
                features_config = metadata['training_config'].get('features', {})
                enabled_features = []
                for feature_type in self.feature_config.keys():
                    if features_config.get(feature_type, False):
                        enabled_features.append(feature_type)
                if enabled_features:
                    model_mapping['required_features'] = enabled_features
                    logger.info(f"Model {model_name}: Generated required features from training config: {enabled_features}")
            
            model_mapping['expected_feature_count'] = expected_count
            mapping['models'][model_name] = model_mapping
            mapping['feature_counts'][f"{info['type']}_{info['symbol']}"] = expected_count
            
            logger.info(f"Model {model_name} ({info['type']}): Expected {expected_count} features")
        
        return mapping
    
    def _infer_feature_count_from_model(self, model_path: Path, model_type: str) -> Optional[int]:
        """Try to infer feature count from actual model files."""
        try:
            # Look for metadata files that might contain feature information
            metadata_files = [
                model_path / 'model_metadata.json',
                model_path / 'imported_metadata.json',
                model_path / 'preprocessor_metadata.json'
            ]
            
            for metadata_file in metadata_files:
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check for various feature count indicators
                    for key in ['n_features_in_', 'input_size', 'feature_count', 'n_features']:
                        if key in metadata and isinstance(metadata[key], int) and metadata[key] > 0:
                            logger.info(f"Found feature count {metadata[key]} from {key} in {metadata_file}")
                            return metadata[key]
            
            # Try to load preprocessor pickle files
            preprocessor_files = list(model_path.glob('preprocessor*.pkl'))
            for prep_file in preprocessor_files:
                try:
                    import pickle
                    with open(prep_file, 'rb') as f:
                        preprocessor = pickle.load(f)
                    
                    if hasattr(preprocessor, 'n_features_in_'):
                        feature_count = preprocessor.n_features_in_
                        logger.info(f"Inferred {feature_count} features from preprocessor {prep_file}")
                        return feature_count
                except Exception as e:
                    logger.debug(f"Failed to load preprocessor {prep_file}: {e}")
                    continue
            
            logger.debug(f"Could not infer feature count from model at {model_path}")
            return None
            
        except Exception as e:
            logger.debug(f"Error inferring feature count from {model_path}: {e}")
            return None
    
    def run(self) -> bool:
        """Run the feature generation process"""
        try:
            logger.info("Starting feature generation from model metadata...")
            
            # Scan models and extract metadata
            models_info = self.scan_models()
            
            if not models_info:
                logger.warning("No models found with metadata. Creating default configuration.")
                # Create default configuration
                default_config = {
                    'version': '1.0',
                    'generated_at': str(Path().cwd()),
                    'models_analyzed': 0,
                    'feature_generation': {
                        'enabled_features': {
                            'technical_indicators': self.feature_config['technical_indicators'],
                            'price_features': self.feature_config['price_features']
                        },
                        'max_features': 50,
                        'selection_method': 'variance',
                        'symbols': [],
                        'model_types': []
                    }
                }
                self.save_feature_config(default_config)
                return True
            
            # Analyze feature requirements
            feature_summary = self.analyze_feature_requirements(models_info)
            logger.info(f"Analyzed {len(models_info)} models")
            logger.info(f"Detected feature types: {feature_summary['feature_types']}")
            logger.info(f"Model types: {feature_summary['model_types']}")
            logger.info(f"Symbols: {feature_summary['symbols']}")
            
            # Generate unified feature configuration
            feature_config = self.generate_feature_config(feature_summary)
            
            # Generate model-specific feature mapping
            feature_mapping = self.generate_feature_mapping(models_info)
            
            # Save configurations
            self.save_feature_config(feature_config)
            
            # Save feature mapping
            mapping_file = self.output_dir / 'feature_mapping.json'
            with open(mapping_file, 'w') as f:
                json.dump(feature_mapping, f, indent=2)
            logger.info(f"Feature mapping saved to: {mapping_file}")
            
            # Create summary report
            summary_file = self.output_dir / 'feature_generation_summary.txt'
            with open(summary_file, 'w') as f:
                f.write("Feature Generation Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Models analyzed: {len(models_info)}\n")
                f.write(f"Feature types detected: {', '.join(feature_summary['feature_types'])}\n")
                f.write(f"Model types: {', '.join(feature_summary['model_types'])}\n")
                f.write(f"Symbols: {', '.join(feature_summary['symbols'])}\n")
                f.write(f"Max features: {feature_summary['max_features']}\n")
                f.write(f"Selection methods: {', '.join(feature_summary['selection_methods'])}\n\n")
                
                f.write("Generated Files:\n")
                f.write("- feature_config.json: Main feature configuration\n")
                f.write("- feature_config.yaml: Human-readable configuration\n")
                f.write("- feature_mapping.json: Model-specific feature mapping\n")
                f.write("- feature_generation_summary.txt: This summary\n")
            
            logger.info(f"Summary report saved to: {summary_file}")
            logger.info("Feature generation completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during feature generation: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Generate features from model metadata')
    parser.add_argument('--models-dir', required=True, help='Directory containing models')
    parser.add_argument('--output-dir', required=True, help='Output directory for feature configuration')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    generator = FeatureGenerator(args.models_dir, args.output_dir)
    success = generator.run()
    
    exit(0 if success else 1)

if __name__ == '__main__':
    main()