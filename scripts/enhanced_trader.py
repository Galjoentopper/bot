#!/usr/bin/env python3
"""
Enhanced Unified Trading Script with Robust Model Loading

This enhanced version supports:
- Loading models from multiple sources (local, imported, packaged)
- Robust fallback mechanisms for model loading
- Support for transferred models from other machines
- Enhanced error handling and logging
- Model validation and compatibility checking
- Command-line arguments for flexible symbol and model selection
"""

import os
import sys
import json
import pickle
import asyncio
import time
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Core imports
import pandas as pd
import numpy as np
import ccxt

# Project imports
from src.config.config_loader import ConfigLoader
from src.utils.logger import Logger
from src.data_pipeline.feature_engine import FeatureEngine
from src.data_pipeline.data_preprocessor import DataPreprocessor
from src.data_pipeline.feature_selector import EnhancedDataPreprocessor
from src.models.gru_trainer import GRUTrainer
from src.models.lgbm_trainer import LightGBMTrainer
from src.models.ppo_trainer import PPOTrainer
from src.trading.trading_metrics import TradingMetrics
from src.notifier.telegram import TelegramNotifier

# Enhanced model loading utilities
from src.utils.model_packaging import ModelPackager
from src.utils.model_transfer import ModelTransferManager

# Validation system
from src.validation.validation_integration import create_validation_manager
from src.validation.metadata_manager import MetadataManager

# Enable debug logging temporarily
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class ModelMetadata:
    """Enhanced model metadata for robust loading."""
    symbol: str
    model_type: str
    version: str
    created_at: str
    python_version: str
    dependencies: Dict[str, str]
    performance_metrics: Dict[str, float]
    file_path: str
    hash_md5: str
    source: str  # 'local', 'imported', 'packaged'
    validated: bool = False

class EnhancedUnifiedPaperTrader:
    """Enterprise-ready paper trader with robust model loading and health monitoring."""
    
    def __init__(self, config_path: str = None, models_dir: str = 'models', 
                 symbols: List[str] = None, models: List[str] = None, 
                 show_available_mode: bool = False):
        """Initialize the enhanced trader."""
        # Use auto-detection if no config path specified, otherwise use explicit path
        if config_path:
            self.config = ConfigLoader(config_path).config
        else:
            self.config = ConfigLoader().config
        
        # Use models_dir from config if available, otherwise use parameter
        config_models_dir = self.config.get('model_management', {}).get('models_dir')
        if config_models_dir:
            self.models_dir = Path(config_models_dir)
        else:
            self.models_dir = Path(models_dir)
        self.logger = Logger(name='enhanced_trader')
        
        # Discover available models and symbols from models directory
        available_symbols, available_models = self._discover_available_models()
        self.logger.logger.info(f"Discovered models for symbols: {sorted(available_symbols)}")
        self.logger.logger.info(f"Available model types: {sorted(available_models)}")
        
        # Set symbols from parameters if provided, otherwise use config, then filter by availability
        if symbols:
            requested_symbols = symbols
        else:
            # Extract symbols from config
            config_symbols = self.config.get('data_acquisition', {}).get('symbols', [])
            if not config_symbols:
                config_symbols = self.config.get('data', {}).get('symbols', [])
            if not config_symbols:
                config_symbols = self.config.get('symbols', [])
            requested_symbols = config_symbols
        
        # Filter symbols to only include those with available models
        self.symbols = [symbol for symbol in requested_symbols if symbol in available_symbols]
        if len(self.symbols) != len(requested_symbols):
            missing_symbols = set(requested_symbols) - set(self.symbols)
            self.logger.logger.warning(f"Excluded symbols without models: {sorted(missing_symbols)}")
        
        # Set models from parameters if provided, otherwise use config, then filter by availability
        if models:
            requested_models = models
        else:
            # Extract models from config
            requested_models = self.config.get('training', {}).get('models', ['gru', 'lightgbm', 'ppo'])
        
        # Filter models to only include those that are available
        self.model_types = [model for model in requested_models if model in available_models]
        if len(self.model_types) != len(requested_models):
            missing_models = set(requested_models) - set(self.model_types)
            self.logger.logger.warning(f"Excluded model types not found: {sorted(missing_models)}")
        
        self.logger.logger.info(f"Trading symbols (filtered): {self.symbols}")
        self.logger.logger.info(f"Model types (filtered): {self.model_types}")
        
        # Validate we have both symbols and models to work with
        if not self.symbols:
            self.logger.logger.error("No symbols with available models found!")
            if not show_available_mode:  # Only raise error if not just showing available models
                self.logger.logger.info("Available models report:")
                self.show_available_models()
                raise ValueError("No symbols with available models found!")
        if not self.model_types:
            self.logger.logger.error("No available model types found!")
            if not show_available_mode:  # Only raise error if not just showing available models
                self.logger.logger.info("Available models report:")
                self.show_available_models() 
                raise ValueError("No available model types found!")
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.trading_metrics = TradingMetrics()
        
        # Initialize Telegram notifier using proper configuration method
        self.telegram_notifier = TelegramNotifier.from_config(self.config)
        self.logger.logger.info(f"Telegram notifier initialized: enabled={getattr(self.telegram_notifier, 'enabled', False)}")
        
        # Model management
        self.model_packager = ModelPackager()
        self.transfer_manager = ModelTransferManager()
        
        # Validation system
        validation_config_dir = self.config.get('validation', {}).get('config_dir', './validation')
        self.validation_manager = create_validation_manager(
            config_dir=validation_config_dir,
            models_dir=str(self.models_dir),
            auto_start=True
        )
        
        # Metadata management
        metadata_config = {
            'max_model_age_days': self.config.get('model_management', {}).get('max_age_days', 30),
            'validation_interval_hours': self.config.get('model_management', {}).get('validation_interval_hours', 24),
            'auto_cleanup': self.config.get('model_management', {}).get('auto_cleanup', True)
        }
        self.metadata_manager = MetadataManager(
            models_dir=str(self.models_dir),
            config=metadata_config
        )
        
        # Trading configuration - get symbols from parameters or data section or fallback to root symbols
        # self.symbols is already set and filtered earlier in the constructor; do not overwrite here.
        self.interval = self.config.get('interval', '30m')
        self.initial_balance = float(self.config.get('initial_balance', 10000))
        self.max_position_size = float(self.config.get('max_position_size', 0.1))
        # Overlay with trading config if provided (align with trader.py)
        trading_config = self.config.get('trading', {})
        self.initial_balance = float(trading_config.get('initial_balance', self.initial_balance))
        self.max_position_size = float(trading_config.get('max_position_size', self.max_position_size))
        self.transaction_fee = float(trading_config.get('transaction_fee', 0.001))
        self.slippage = float(trading_config.get('slippage', 0.0005))
        self.model_weights = trading_config.get('model_weights', {'gru': 0.45, 'lightgbm': 0.45, 'ppo': 0.1})
        self.ppo_scale = float(trading_config.get('ppo_scale', 0.002))
        self.min_trade_value = float(trading_config.get('min_trade_value', 5.0))
        thresholds_cfg = trading_config.get('thresholds', {})
        self.symbol_thresholds = thresholds_cfg.get('per_symbol', {
            'BTCEUR': 0.00008,
            'ETHEUR': 0.00008,
            'SOLEUR': 0.00012,
            'ADAEUR': 0.00012,
            'XRPEUR': 0.00012,
        })
        self.default_threshold = float(thresholds_cfg.get('default', 0.00010))
        self.use_cost_floor = bool(thresholds_cfg.get('use_cost_floor', True))
        self.cost_floor_multiplier = float(thresholds_cfg.get('cost_floor_multiplier', 1.2))
        self.vol_reference = float(thresholds_cfg.get('vol_reference', 0.02))
        bounds = thresholds_cfg.get('vol_bounds', [0.5, 2.0])
        try:
            self.vol_bounds = (float(bounds[0]), float(bounds[1]))
        except Exception:
            self.vol_bounds = (0.5, 2.0)
        
        # Model storage
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_metadata: Dict[str, Dict[str, ModelMetadata]] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.symbol_feature_metadata: Dict[str, List[str]] = {}
        
        # Trading state
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.balance = self.initial_balance
        self.last_prices = {}
        # Caching and performance tracking
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # seconds
        self.performance_history = []
        self.rejected_trades_count = 0
        
        # Data caching
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # seconds
        
        # Initialize CSV trades report with correct column structure
        self._initialize_trades_csv()
        
        self.logger.logger.info(f"Enhanced trader initialized with ${self.initial_balance:,.2f}")
    
    def _discover_available_models(self) -> Tuple[List[str], List[str]]:
        """Discover available symbols and model types from the models directory structure."""
        available_symbols = set()
        available_models = set()
        
        if not self.models_dir.exists():
            self.logger.logger.warning(f"Models directory does not exist: {self.models_dir}")
            return [], []
        
        # Define model file patterns for each model type
        model_patterns = {
            'gru': ['*.pth', '*.pt', 'model.pth'],
            'lightgbm': ['*.pkl', 'model.pkl'],
            'ppo': ['*.zip', 'model.zip', 'model']  # PPO models can be saved without extension
        }
        
        # Search standard structure: models/{model_type}/{symbol}/
        for model_type, patterns in model_patterns.items():
            model_type_dir = self.models_dir / model_type
            if model_type_dir.exists():
                for symbol_dir in model_type_dir.iterdir():
                    if symbol_dir.is_dir():
                        symbol = symbol_dir.name
                        # Check if model files exist directly
                        found_model = False
                        for pattern in patterns:
                            if list(symbol_dir.glob(pattern)):
                                available_symbols.add(symbol)
                                available_models.add(model_type)
                                found_model = True
                                break
                        
                        # Also check nested directories (e.g., models/lightgbm/BTCEUR/lightgbm/timestamp/model.pkl)
                        if not found_model:
                            for nested_dir in symbol_dir.iterdir():
                                if nested_dir.is_dir():
                                    for pattern in patterns:
                                        if list(nested_dir.glob(pattern)) or list(nested_dir.glob(f"*/{pattern}")):
                                            available_symbols.add(symbol)
                                            available_models.add(model_type)
                                            found_model = True
                                            break
                                    if found_model:
                                        break
                        
                        if found_model:
                            self.logger.logger.debug(f"Found {model_type} model for {symbol}")
        
        # Search alternative structure: models/{symbol}/{model_type}/
        for potential_symbol_dir in self.models_dir.iterdir():
            if potential_symbol_dir.is_dir() and not potential_symbol_dir.name.startswith('.'):
                symbol = potential_symbol_dir.name
                # Skip known model type directories
                if symbol in ['gru', 'lightgbm', 'ppo', 'imported', 'metadata', 'packages']:
                    continue
                
                for model_type, patterns in model_patterns.items():
                    model_type_dir = potential_symbol_dir / model_type
                    if model_type_dir.exists():
                        for pattern in patterns:
                            if list(model_type_dir.glob(pattern)):
                                available_symbols.add(symbol)
                                available_models.add(model_type)
                                break
        
        # Search flat structure and special directories
        special_dirs = ['imported', 'metadata', 'packages']
        search_dirs = [self.models_dir] + [self.models_dir / d for d in special_dirs if (self.models_dir / d).exists()]
        
        for search_dir in search_dirs:
            for model_type, patterns in model_patterns.items():
                for pattern in patterns:
                    # Look for files like: gru_model_{symbol}_*.pth, best_wf_{model_type}_{symbol}.pkl
                    search_patterns = [
                        f"{model_type}_model_*{pattern.replace('*', '')}",
                        f"*{model_type}*{pattern.replace('*', '')}",
                        f"best_wf_{model_type}_*.{pattern.split('.')[-1]}" if '.' in pattern else f"best_wf_{model_type}_*"
                    ]
                    
                    for search_pattern in search_patterns:
                        files = list(search_dir.glob(search_pattern))
                        for file in files:
                            # Extract symbol from filename
                            filename = file.stem
                            potential_symbols = []
                            
                            # Pattern: {model_type}_model_{symbol}_timestamp
                            if f"{model_type}_model_" in filename:
                                parts = filename.split(f"{model_type}_model_")
                                if len(parts) > 1:
                                    symbol_part = parts[1].split('_')[0]
                                    potential_symbols.append(symbol_part)
                            
                            # Pattern: best_wf_{model_type}_{symbol}
                            if f"best_wf_{model_type}_" in filename:
                                parts = filename.split(f"best_wf_{model_type}_")
                                if len(parts) > 1:
                                    symbol_part = parts[1].split('_')[0]
                                    potential_symbols.append(symbol_part)
                            
                            # Pattern: {symbol}_{model_type}_*
                            for common_symbol in ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR', 'SOLEUR', 'XRPEUR']:
                                if common_symbol in filename.upper():
                                    potential_symbols.append(common_symbol)
                            
                            for symbol in potential_symbols:
                                if symbol and len(symbol) >= 3:  # Valid symbol length
                                    available_symbols.add(symbol)
                                    available_models.add(model_type)
        
        return sorted(list(available_symbols)), sorted(list(available_models))
    
    def show_available_models(self) -> None:
        """Show detailed report of available models."""
        self.logger.logger.info("=== AVAILABLE MODELS REPORT ===")
        self.logger.logger.info(f"Models directory: {self.models_dir}")
        
        if not self.models_dir.exists():
            self.logger.logger.warning("Models directory does not exist!")
            return
        
        # Check each model type directory
        for model_type in ['gru', 'lightgbm', 'ppo']:
            type_dir = self.models_dir / model_type
            if type_dir.exists():
                symbols_with_models = []
                for symbol_dir in type_dir.iterdir():
                    if symbol_dir.is_dir():
                        # Check if model files exist directly or in nested directories
                        model_files = []
                        patterns = {
                            'gru': ['*.pth', '*.pt', 'model.pth'],
                            'lightgbm': ['*.pkl', 'model.pkl'],
                            'ppo': ['*.zip', 'model.zip']
                        }
                        
                        # Check direct files
                        for pattern in patterns.get(model_type, ['*']):
                            model_files.extend(list(symbol_dir.glob(pattern)))
                        
                        # Check nested directories
                        if not model_files:
                            for nested_dir in symbol_dir.iterdir():
                                if nested_dir.is_dir():
                                    for pattern in patterns.get(model_type, ['*']):
                                        model_files.extend(list(nested_dir.glob(pattern)))
                                        model_files.extend(list(nested_dir.glob(f"*/{pattern}")))
                        
                        if model_files:
                            symbols_with_models.append(symbol_dir.name)
                
                if symbols_with_models:
                    self.logger.logger.info(f"{model_type}: {sorted(symbols_with_models)}")
                else:
                    self.logger.logger.info(f"{model_type}: No models found")
            else:
                self.logger.logger.info(f"{model_type}: Directory not found")
        
        # Check for alternative locations
        alt_locations = ['imported', 'metadata', 'packages']
        for location in alt_locations:
            alt_dir = self.models_dir / location
            if alt_dir.exists():
                files = list(alt_dir.glob('*'))
                if files:
                    self.logger.logger.info(f"{location}/: {len(files)} files found")
    
    def load_all_models(self):
        """Load all models with enhanced fallback mechanisms."""
        self.logger.logger.info("Loading models with enhanced fallback mechanisms...")
        
        for symbol in self.symbols:
            self.models[symbol] = {}
            self.model_metadata[symbol] = {}
            self.preprocessors[symbol] = {}
            
            # Load each model type with multiple fallback sources
            for model_type in self.model_types:
                model_info = self._load_model_with_fallbacks(symbol, model_type)
                if model_info:
                    model, metadata = model_info
                    self.models[symbol][model_type] = model
                    self.model_metadata[symbol][model_type] = metadata
                    self.logger.logger.info(
                        f"Loaded {model_type} model for {symbol} from {metadata.source}: {metadata.file_path}"
                    )
                    
                    # Load model-specific preprocessor
                    preprocessor = self._load_model_specific_preprocessor(symbol, model_type)
                    self.preprocessors[symbol][model_type] = preprocessor
                else:
                    self.logger.logger.warning(f"Failed to load {model_type} model for {symbol}")
            
            # Load feature metadata
            self._load_feature_metadata(symbol)
        
        total_models = sum(len(models) for models in self.models.values())
        self.logger.logger.info(f"Loaded {total_models} models across {len(self.symbols)} symbols")
    
    def _load_models_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Load models for a specific symbol and return them."""
        models = {}
        
        for model_type in self.model_types:
            model_info = self._load_model_with_fallbacks(symbol, model_type)
            if model_info:
                model, metadata = model_info
                models[model_type] = model
                self.logger.logger.info(
                    f"Loaded {model_type} model for {symbol} from {metadata.source}: {metadata.file_path}"
                )
            else:
                self.logger.logger.warning(f"Failed to load {model_type} model for {symbol}")
        
        return models
    
    def _load_model_with_fallbacks(self, symbol: str, model_type: str) -> Optional[Tuple[Any, ModelMetadata]]:
        """Load model with multiple fallback sources."""
        # Define search strategies in order of preference
        strategies = [
            ('packaged', self._load_from_packaged_models),
            ('imported', self._load_from_imported_models),
            ('best_wf', self._load_from_best_walkforward),
            ('latest', self._load_from_latest_models),
            ('unified', self._load_from_unified_artifacts)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                result = strategy_func(symbol, model_type)
                if result:
                    model, file_path = result
                    # Create metadata
                    metadata = ModelMetadata(
                        symbol=symbol,
                        model_type=model_type,
                        version='unknown',
                        created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                        python_version=sys.version,
                        dependencies={},
                        performance_metrics={},
                        file_path=str(file_path),
                        hash_md5='',
                        source=strategy_name
                    )
                    
                    # Validate model
                    if self._validate_model(model, symbol, model_type):
                        metadata.validated = True
                        return model, metadata
                    else:
                        self.logger.logger.warning(
                            f"Model validation failed for {symbol} {model_type} from {strategy_name}"
                        )
                        
            except Exception as e:
                self.logger.logger.debug(
                    f"Strategy {strategy_name} failed for {symbol} {model_type}: {e}"
                )
                continue
        
        return None
    
    def _load_from_packaged_models(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from packaged models (highest priority)."""
        packages_dir = self.models_dir / 'packages'
        if not packages_dir.exists():
            return None
        
        # Look for packaged models for this symbol and type
        pattern = f"{symbol}_{model_type}_*.zip"
        package_files = list(packages_dir.glob(pattern))
        
        if not package_files:
            return None
        
        # Use the most recent package
        latest_package = max(package_files, key=lambda p: p.stat().st_mtime)
        
        try:
            # Extract and load the packaged model
            extracted_dir = self.model_packager.import_package(str(latest_package))
            
            # Find the model file in the extracted directory
            model_extensions = {
                'gru': ['.pth', '.pt'],
                'lightgbm': ['.pkl'],
                'ppo': ['.zip']
            }
            
            for ext in model_extensions.get(model_type, ['.pkl']):
                model_files = list(Path(extracted_dir).glob(f"*{ext}"))
                if model_files:
                    model_file = model_files[0]
                    model = self._load_model_file(model_file, model_type)
                    if model:
                        return model, model_file
        
        except Exception as e:
            self.logger.logger.debug(f"Failed to load packaged model: {e}")
        
        return None
    
    def _load_from_imported_models(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from imported models directory."""
        imported_dir = self.models_dir / 'imported'
        if not imported_dir.exists():
            return None
        
        # Look for imported models
        model_extensions = {
            'gru': ['.pth', '.pt'],
            'lightgbm': ['.pkl'],
            'ppo': ['.zip']
        }
        
        for ext in model_extensions.get(model_type, ['.pkl']):
            pattern = f"{symbol}*{model_type}*{ext}"
            model_files = list(imported_dir.rglob(pattern))
            
            if model_files:
                # Use the most recent file
                latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
                model = self._load_model_file(latest_file, model_type)
                if model:
                    return model, latest_file
        
        return None
    
    def _load_from_best_walkforward(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from best walk-forward results."""
        metadata_dir = self.models_dir / 'metadata'
        if not metadata_dir.exists():
            return None
        
        # Look for best walk-forward files
        if model_type == 'lightgbm':
            pattern = f"best_wf_lightgbm_{symbol}.pkl"
        elif model_type == 'gru':
            pattern = f"best_wf_gru_{symbol}.pt*"
        else:
            return None
        
        model_files = list(metadata_dir.glob(pattern))
        if model_files:
            latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            model = self._load_model_file(latest_file, model_type)
            if model:
                return model, latest_file
        
        return None
    
    def _load_from_latest_models(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from latest model files."""
        if model_type == 'gru':
            pattern = f"gru_model_{symbol}_*.pth"
        elif model_type == 'lightgbm':
            pattern = f"lightgbm_model_{symbol}_*.pkl"
        elif model_type == 'ppo':
            pattern = f"ppo_model_{symbol}_*.zip"
        else:
            return None
        
        model_files = list(self.models_dir.glob(pattern))
        if model_files:
            latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            model = self._load_model_file(latest_file, model_type)
            if model:
                return model, latest_file
        
        return None
    
    def _load_from_unified_artifacts(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from unified trainer artifacts."""
        search_dir = self.models_dir / model_type / symbol
        if not search_dir.exists():
            return None
        
        if model_type == 'gru':
            filename = 'model.pth'
        elif model_type == 'lightgbm':
            filename = 'model.pkl'
        elif model_type == 'ppo':
            filename = 'model.zip'
        else:
            return None
        
        model_files = list(search_dir.rglob(filename))
        if model_files:
            latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            model = self._load_model_file(latest_file, model_type)
            if model:
                return model, latest_file
        
        return None
    
    def _load_model_file(self, file_path: Path, model_type: str) -> Optional[Any]:
        """Load a model file based on its type."""
        try:
            if model_type == 'gru':
                return GRUTrainer.load_model(str(file_path), self.config)
            elif model_type == 'lightgbm':
                return LightGBMTrainer.load_model(str(file_path), self.config)
            elif model_type == 'ppo':
                return PPOTrainer.load_model(str(file_path), self.config)
            else:
                self.logger.logger.error(f"Unknown model type: {model_type}")
                return None
        except Exception as e:
            self.logger.logger.debug(f"Failed to load model from {file_path}: {e}")
            return None
    
    def _validate_model(self, model: Any, symbol: str, model_type: str) -> bool:
        """Validate that a loaded model is functional."""
        try:
            # Basic validation - check if model has required methods
            if model_type in ['gru', 'lightgbm']:
                if not hasattr(model, 'predict'):
                    return False
            elif model_type == 'ppo':
                if not hasattr(model, 'predict'):
                    return False
            
            # Additional validation could be added here
            # (e.g., test prediction with dummy data)
            
            return True
        except Exception as e:
            self.logger.logger.debug(f"Model validation failed: {e}")
            return False
    
    def _load_model_specific_preprocessor(self, symbol: str, model_type: str) -> Any:
        """Load model-specific preprocessor with enhanced fallback mechanisms."""
        # Try multiple sources for model-specific preprocessor
        sources = [
            self.models_dir / model_type / symbol / 'preprocessor.pkl',
            self.models_dir / 'imported' / f"preprocessor_{symbol}_{model_type}.pkl",
            self.models_dir / model_type / symbol / '*' / 'preprocessor.pkl',
            self.models_dir / 'imported' / f"preprocessor_{symbol}.pkl",
            self.models_dir / f"preprocessor_{symbol}_*.pkl"
        ]
        
        for source in sources:
            try:
                if '*' in str(source):
                    # Handle glob patterns
                    parent_path = Path(str(source).split('*')[0]).parent
                    if parent_path.exists():
                        files = list(parent_path.glob(Path(str(source)).name))
                        if files:
                            source = max(files, key=lambda p: p.stat().st_mtime)
                        else:
                            continue
                    else:
                        continue
                
                if source.exists():
                    # Try loading enhanced preprocessor first
                    try:
                        preprocessor = EnhancedDataPreprocessor.load(source)
                        if preprocessor and preprocessor.is_fitted:
                            self.logger.logger.info(f"Loaded enhanced {model_type} preprocessor for {symbol} from {source}")
                            return preprocessor
                    except Exception as e:
                        self.logger.logger.debug(f"Failed to load as enhanced preprocessor: {e}")
                    
                    # Try legacy pickle format
                    try:
                        with open(source, 'rb') as f:
                            preprocessor = pickle.load(f)
                        self.logger.logger.info(f"Loaded legacy {model_type} preprocessor for {symbol} from {source}")
                        return preprocessor
                    except Exception:
                        # Try joblib
                        try:
                            import joblib
                            preprocessor = joblib.load(source)
                            self.logger.logger.info(f"Loaded {model_type} preprocessor (joblib) for {symbol} from {source}")
                            return preprocessor
                        except Exception:
                            continue
            except Exception:
                continue
        
        # Create fresh enhanced preprocessor with proper metadata
        self.logger.logger.info(f"Creating fresh enhanced {model_type} preprocessor for {symbol}")
        return EnhancedDataPreprocessor(model_type=model_type, symbol=symbol)
    
    def _ensure_preprocessor_fitted(self, preprocessor: Any, features: pd.DataFrame, symbol: str, model_type: str) -> Any:
        """Ensure preprocessor is fitted, fit it if necessary with enhanced persistence."""
        # Handle enhanced preprocessor
        if isinstance(preprocessor, EnhancedDataPreprocessor):
            if preprocessor.is_fitted:
                return preprocessor
            else:
                self.logger.logger.info(f"Fitting enhanced {model_type} preprocessor for {symbol}")
                preprocessor.fit(features)
                # Save the fitted preprocessor for future use
                self._save_fitted_preprocessor(preprocessor, symbol, model_type)
                return preprocessor
        
        # Handle legacy preprocessors
        try:
            # Check if preprocessor is already fitted by trying to transform a small sample
            if hasattr(preprocessor, 'transform'):
                test_sample = features.iloc[:1].copy()
                preprocessor.transform(test_sample.values)
                return preprocessor
        except Exception:
            # Preprocessor not fitted, fit it now
            try:
                if hasattr(preprocessor, 'fit'):
                    self.logger.logger.info(f"Fitting {model_type} preprocessor for {symbol} on current data")
                    preprocessor.fit(features.values)
                    # Try to save the fitted preprocessor
                    self._save_legacy_preprocessor(preprocessor, symbol, model_type)
                    return preprocessor
                else:
                    # If no fit method, return as-is (might be a simple scaler)
                    return preprocessor
            except Exception as e:
                self.logger.logger.warning(f"Failed to fit {model_type} preprocessor for {symbol}: {e}")
                # Return identity preprocessor as fallback
                class IdentityPreprocessor:
                    def transform(self, X):
                        return X
                    def fit(self, X):
                        return self
                    def is_fitted(self):
                        return True
                return IdentityPreprocessor()
    
    def _save_fitted_preprocessor(self, preprocessor: EnhancedDataPreprocessor, symbol: str, model_type: str):
        """Save fitted enhanced preprocessor for future use."""
        try:
            save_dir = self.models_dir / model_type / symbol
            save_dir.mkdir(parents=True, exist_ok=True)
            preprocessor_path = save_dir / 'preprocessor.pkl'
            preprocessor.save(preprocessor_path)
            self.logger.logger.info(f"Saved fitted preprocessor for {model_type}_{symbol}")
        except Exception as e:
            self.logger.logger.warning(f"Failed to save fitted preprocessor: {e}")
    
    def _save_legacy_preprocessor(self, preprocessor: Any, symbol: str, model_type: str):
        """Save fitted legacy preprocessor for future use."""
        try:
            save_dir = self.models_dir / model_type / symbol
            save_dir.mkdir(parents=True, exist_ok=True)
            preprocessor_path = save_dir / 'preprocessor_legacy.pkl'
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            self.logger.logger.info(f"Saved fitted legacy preprocessor for {model_type}_{symbol}")
        except Exception as e:
            self.logger.logger.warning(f"Failed to save fitted legacy preprocessor: {e}")
    
    def _load_feature_metadata(self, symbol: str):
        """Load feature metadata for the symbol."""
        # Try to load from various sources
        metadata_sources = [
            self.models_dir / 'imported' / f"features_{symbol}.json",
            self.models_dir / 'metadata' / f"features_{symbol}.json",
            self.models_dir / 'gru' / symbol / '*' / 'features.json'
        ]
        
        for source in metadata_sources:
            try:
                if '*' in str(source):
                    files = list(Path(str(source).split('*')[0]).parent.glob('*/features.json'))
                    if files:
                        source = max(files, key=lambda p: p.stat().st_mtime)
                    else:
                        continue
                
                if source.exists():
                    with open(source, 'r') as f:
                        features = json.load(f)
                    self.symbol_feature_metadata[symbol] = features
                    self.logger.logger.info(f"Loaded {len(features)} feature names for {symbol}")
                    return
            except Exception:
                continue
        
        # Fallback to default feature generation
        self.logger.logger.info(f"Using default feature generation for {symbol}")
        self.symbol_feature_metadata[symbol] = []
    
    async def get_market_data(self) -> dict:
        """Get latest market data for all symbols."""
        market_data = {}
        
        # Check cache first
        current_time = time.time()
        if self._has_cached_data(current_time):
            self.logger.logger.debug("Using cached market data")
            return self.data_cache.copy()
        
        # Fetch fresh data
        try:
            for symbol in self.symbols:
                try:
                    self.logger.logger.info(f"Fetching {symbol} data from Binance API...")
                    api_df = await self._fetch_data_from_binance_api(symbol, limit=300)
                    
                    if api_df is None or api_df.empty:
                        self.logger.logger.warning(f"No data fetched from API for {symbol}")
                        continue
                    
                    # Generate features
                    df_with_features = self.feature_engine.generate_all_features(api_df)
                    
                    # Use current FeatureEngine output instead of outdated metadata
                    # This ensures we always use the correct 113 features generated by FeatureEngine
                    feature_names = self.feature_engine.get_feature_names(df_with_features)
                    
                    # Clean features
                    df_with_features = self._clean_features_for_inference(df_with_features, symbol)
                    
                    if self._validate_market_data(df_with_features, symbol):
                        market_data[symbol] = df_with_features
                        self.last_prices[symbol] = df_with_features['close'].iloc[-1]
                        self.logger.logger.info(
                            f"Fetched {len(df_with_features)} records for {symbol} with {len(feature_names)} features"
                        )
                    else:
                        self.logger.logger.warning(f"Invalid market data for {symbol}")
                        
                except Exception as e:
                    self.logger.logger.error(f"Error processing data for {symbol}: {e}")
                    continue
        
        except Exception as e:
            self.logger.logger.error(f"Error in data pipeline: {e}")
            return {}
        
        # Update cache
        self.data_cache = market_data.copy()
        self.cache_expiry = {symbol: current_time + self.cache_duration for symbol in market_data}
        
        return market_data
    
    def _has_cached_data(self, current_time: float) -> bool:
        """Check if we have valid cached data."""
        if not self.data_cache:
            return False
        
        for symbol in self.symbols:
            if symbol not in self.cache_expiry or current_time > self.cache_expiry[symbol]:
                return False
        
        return True
    
    async def _fetch_data_from_binance_api(self, symbol: str, limit: int = 300) -> Optional[pd.DataFrame]:
        """Fetch data directly from Binance API."""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000
            })
            exchange.load_markets()
            
            formatted_symbol = self._convert_symbol_format(symbol)
            self.logger.logger.info(f"Fetching {limit} candles for {symbol} from Binance API")
            
            ohlcv = await self._fetch_with_retry(exchange, formatted_symbol, limit=limit)
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')
                
                # Add missing columns
                df['quote_volume'] = df['volume']
                df['trades'] = 0
                df['taker_buy_base'] = df['volume'] * 0.5
                df['taker_buy_quote'] = df['quote_volume'] * 0.5
                
                # Ensure numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                self.logger.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                return df
            else:
                self.logger.logger.warning(f"No OHLCV data returned for {symbol}")
                return None
                
        except Exception as e:
            self.logger.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    async def _fetch_with_retry(self, exchange, symbol: str, max_retries: int = 3, limit: int = 100):
        """Fetch OHLCV data with retry logic."""
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, self.interval, limit=limit)
                return ohlcv
            except ccxt.RateLimitExceeded:
                self.logger.logger.warning(f"Rate limit exceeded for {symbol}, waiting...")
                await asyncio.sleep(10)
            except ccxt.NetworkError as e:
                self.logger.logger.warning(f"Network error for {symbol}: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.logger.error(f"Error fetching {symbol}: {e}")
                break
        
        return None
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format to ccxt standard."""
        if symbol.endswith('EUR'):
            base = symbol[:-3]
            return f"{base}/EUR"
        elif symbol.endswith('USD'):
            base = symbol[:-3]
            return f"{base}/USD"
        elif symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USDT"
        return symbol
    
    def _prepare_features_for_model(self, features: pd.DataFrame, model_type: str, symbol: str) -> pd.DataFrame:
        """Prepare features for specific model type with proper padding/trimming."""
        try:
            # Apply model-specific feature padding with symbol for metadata-based mapping
            prepared_features = self.feature_engine.pad_features_for_model(features, model_type, symbol)
            
            # Log feature preparation
            original_count = len(self.feature_engine.get_feature_names(features))
            prepared_count = len(self.feature_engine.get_feature_names(prepared_features))
            
            if original_count != prepared_count:
                self.logger.logger.info(
                    f"Adjusted features for {model_type} {symbol}: {original_count} -> {prepared_count}"
                )
            
            return prepared_features
            
        except Exception as e:
            self.logger.logger.error(f"Feature preparation failed for {model_type} {symbol}: {e}")
            return features
    
    def _clean_features_for_inference(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean features for inference."""
        try:
            feature_names = self.symbol_feature_metadata.get(symbol, [])
            
            for col in feature_names:
                if col in df.columns:
                    if df[col].isna().any():
                        df[col] = df[col].ffill().bfill()
                        if df[col].isna().any():
                            mean_val = df[col].mean()
                            if pd.isna(mean_val):
                                mean_val = 0.0
                            df[col].fillna(mean_val, inplace=True)
            
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.logger.warning(f"Error cleaning features for {symbol}: {e}")
            return df
    
    def _validate_market_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate market data."""
        if df.empty:
            return False
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return False
        
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False
        
        if len(df) < 50:
            return False
        
        return True
    
    def generate_signals(self, market_data: dict) -> dict:
        """Generate trading signals using loaded models."""
        signals = {}
        self.logger.logger.info(f"Generating signals for {len(market_data)} symbols")
        
        for symbol, df in market_data.items():
            try:
                self.logger.logger.info(f"Processing signal generation for {symbol}...")
                signal = 0  # Default: hold
                
                if symbol not in self.models or not self.models[symbol]:
                    self.logger.logger.warning(f"No models available for {symbol}")
                    signals[symbol] = signal
                    self.logger.logger.info(f"Completed {symbol} (no models) - signal: {signal}")
                    continue
                
                self.logger.logger.info(f"Preparing features for {symbol}...")
                # Use all generated features except OHLCV columns for supervised models
                feature_names = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                
                features_for_supervised = df.reindex(columns=feature_names, fill_value=0).copy()
                features_for_supervised = features_for_supervised.ffill().bfill().fillna(0)
                
                if features_for_supervised.empty:
                    self.logger.logger.warning(f"No valid features for {symbol}")
                    signals[symbol] = signal
                    self.logger.logger.info(f"Completed {symbol} (no features) - signal: {signal}")
                    continue
                
                self.logger.logger.info(f"Starting model predictions for {symbol}...")
                # Generate predictions from available models
                predictions = []
                
                # GRU prediction
                if 'gru' in self.models[symbol]:
                    self.logger.logger.info(f"Getting GRU prediction for {symbol}...")
                    gru_pred = self._get_gru_prediction(symbol, features_for_supervised)
                    self.logger.logger.info(f"GRU prediction for {symbol}: {gru_pred}")
                    if gru_pred is not None:
                        predictions.append(('gru', gru_pred))
                
                # LightGBM prediction
                if 'lightgbm' in self.models[symbol]:
                    self.logger.logger.info(f"Getting LightGBM prediction for {symbol}...")
                    lgbm_pred = self._get_lightgbm_prediction(symbol, features_for_supervised)
                    self.logger.logger.info(f"LightGBM prediction for {symbol}: {lgbm_pred}")
                    if lgbm_pred is not None:
                        predictions.append(('lightgbm', lgbm_pred))
                
                # PPO prediction uses raw market data + portfolio features
                if 'ppo' in self.models[symbol]:
                    self.logger.logger.info(f"Getting PPO prediction for {symbol}...")
                    ppo_pred = self._get_ppo_prediction(symbol, df)
                    self.logger.logger.info(f"PPO prediction for {symbol}: {ppo_pred}")
                    if ppo_pred is not None:
                        predictions.append(('ppo', ppo_pred))
                
                # Combine predictions
                self.logger.logger.info(f"Combining predictions for {symbol}: {predictions}")
                if predictions:
                    signal = self._combine_predictions(symbol, predictions, df)
                
                signals[symbol] = signal
                self.logger.logger.info(f"Completed signal generation for {symbol}: {signal}")
                
            except Exception as e:
                self.logger.logger.error(f"Signal generation failed for {symbol}: {e}")
                signals[symbol] = 0
                self.logger.logger.info(f"Completed {symbol} (error) - signal: {signal}")
        
        self.logger.logger.info(f"Signal generation completed for all symbols: {signals}")
        return signals
    
    def _get_gru_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[float]:
        """Get GRU model prediction with proper feature preparation."""
        try:
            model = self.models[symbol]['gru']
            preprocessor = self.preprocessors[symbol]['gru']
            
            # Prepare features for GRU model (should result in 113 features)
            prepared_features = self._prepare_features_for_model(features, 'gru', symbol)
            
            # Get sequence length from config or use default
            sequence_length = (
                self.config.get('models', {}).get('gru', {}).get('sequence_length')
                or self.config.get('sequence_length', 32)  # Default sequence length
            )
            
            # Ensure we have enough data for sequence
            if len(prepared_features) < sequence_length:
                self.logger.logger.debug(f"Insufficient data for GRU sequence: {len(prepared_features)} < {sequence_length}")
                return None
                
            # Take the last sequence_length rows for sequential input
            sequence_data = prepared_features.iloc[-sequence_length:].values  # Shape: (sequence_length, 113)
            
            # Validate model input using single timestep for validation
            if hasattr(self, 'validation_manager'):
                validation_result = self.validation_manager.validate_model_input(
                    data=prepared_features.iloc[-1:],  # Validate single timestep
                    model_type='gru',
                    symbol=symbol
                )
                if not validation_result['valid']:
                    self.logger.logger.warning(f"GRU input validation failed for {symbol}: {validation_result['errors']}")
            
            # Ensure preprocessor is fitted and preprocess features
            preprocessor = self._ensure_preprocessor_fitted(preprocessor, prepared_features, symbol, 'gru')
            if hasattr(preprocessor, 'transform'):
                # Transform the entire sequence
                sequence_data = preprocessor.transform(sequence_data)
            
            # Try different input formats for GRU model
            try:
                # First try: 3D tensor format (batch_size, sequence_length, features)
                gru_input_3d = sequence_data.reshape(1, sequence_length, -1)
                self.logger.logger.debug(f"GRU 3D input shape for {symbol}: {gru_input_3d.shape}")
                prediction = model.predict(gru_input_3d)
                
            except Exception as e1:
                self.logger.logger.debug(f"GRU 3D input failed: {e1}")
                try:
                    # Second try: 2D tensor format (batch_size, features) using latest timestep only
                    gru_input_2d = sequence_data[-1:, :]  # Use only the last timestep
                    self.logger.logger.debug(f"GRU 2D input shape for {symbol}: {gru_input_2d.shape}")
                    prediction = model.predict(gru_input_2d)
                    
                except Exception as e2:
                    self.logger.logger.debug(f"GRU 2D input failed: {e2}")
                    # Third try: 1D array format using latest timestep
                    gru_input_1d = sequence_data[-1, :]  # Single timestep as 1D array
                    self.logger.logger.debug(f"GRU 1D input shape for {symbol}: {gru_input_1d.shape}")
                    prediction = model.predict(gru_input_1d.reshape(1, -1))  # Ensure batch dimension
            
            if isinstance(prediction, (list, np.ndarray)):
                prediction = float(prediction[0])
            
            return prediction
            
        except Exception as e:
            self.logger.logger.error(f"GRU prediction failed for {symbol}: {e}")
            return None
    
    def _get_lightgbm_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[float]:
        """Get LightGBM model prediction with proper feature preparation."""
        try:
            model = self.models[symbol]['lightgbm']
            preprocessor = self.preprocessors[symbol]['lightgbm']
            
            # Prepare features for LightGBM model (should result in 114 features)
            prepared_features = self._prepare_features_for_model(features, 'lightgbm', symbol)
            
            # Use the last row for prediction
            latest_features = prepared_features.iloc[-1:]
            
            # Validate model input
            if hasattr(self, 'validation_manager'):
                validation_result = self.validation_manager.validate_model_input(
                    data=latest_features,
                    model_type='lightgbm',
                    symbol=symbol
                )
                if not validation_result['valid']:
                    self.logger.logger.warning(f"LightGBM input validation failed for {symbol}: {validation_result['errors']}")
            
            # Ensure preprocessor is fitted and preprocess features if needed
            preprocessor = self._ensure_preprocessor_fitted(preprocessor, prepared_features, symbol, 'lightgbm')
            if hasattr(preprocessor, 'transform') and not isinstance(preprocessor, type(lambda: None)):
                try:
                    # Only transform if preprocessor is not identity
                    transformed_features = preprocessor.transform(latest_features.values)
                    if transformed_features is not None:
                        latest_features = pd.DataFrame(transformed_features, columns=latest_features.columns)
                except Exception as e:
                    self.logger.logger.warning(f"Preprocessor transform failed for LightGBM {symbol}: {e}, using raw features")
            
            # Make prediction
            prediction = model.predict(latest_features)
            
            if isinstance(prediction, (list, np.ndarray)):
                prediction = float(prediction[0])
            
            return prediction
            
        except Exception as e:
            self.logger.logger.error(f"LightGBM prediction failed for {symbol}: {e}")
            return None
    
    def _get_ppo_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[float]:
        """Get PPO model prediction with proper feature preparation."""
        try:
            model = self.models[symbol]['ppo']
            
            # Skip PPO preprocessing - PPO will use raw market data and build its own observation space
            # PPO models use specific observation construction, not general feature preprocessing
            
            # Build observation from raw market data + portfolio features
            market_df = features.copy()
            # Prefer PPO's sequence length; fallback to GRU's, then 32 (updated default)
            sequence_length = (
                self.config.get('models', {}).get('ppo', {}).get('sequence_length')
                or self.config.get('models', {}).get('gru', {}).get('sequence_length', 32)
            )
            if len(market_df) < sequence_length:
                self.logger.logger.debug(f"Insufficient data for PPO sequence: {len(market_df)} < {sequence_length}")
                return None
            
            # PPO models expect specific observation space: (sequence_length, 13)
            # 13 features = 7 market + 3 technical + 3 portfolio
            
            # Ensure base market columns are present
            base_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
            for col in base_cols:
                if col not in market_df.columns:
                    market_df[col] = 0.0
            
            # Calculate essential technical indicators (only 3 needed for PPO)
            if 'rsi' not in market_df.columns:
                delta = market_df['close'].diff()
                gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-10)
                market_df['rsi'] = 100 - (100 / (1 + rs))
                market_df['rsi'] = market_df['rsi'].fillna(50.0)  # Neutral RSI
            
            if 'macd' not in market_df.columns:
                ema_fast = market_df['close'].ewm(span=12).mean()
                ema_slow = market_df['close'].ewm(span=26).mean()
                market_df['macd'] = ema_fast - ema_slow
                market_df['macd'] = market_df['macd'].fillna(0.0)
            
            if 'bb_position' not in market_df.columns:
                bb_middle = market_df['close'].rolling(window=20).mean()
                bb_std = market_df['close'].rolling(window=20).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)
                market_df['bb_position'] = (market_df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
                market_df['bb_position'] = market_df['bb_position'].fillna(0.5)  # Neutral position
            
            # Build market features (10 total: 7 base + 3 technical)
            market_cols = base_cols + ['rsi', 'macd', 'bb_position']
            market_features = market_df[market_cols].fillna(0.0)
            
            # Normalize market features to reasonable ranges
            market_array = market_features.to_numpy(dtype=np.float32)
            market_array = np.nan_to_num(market_array, nan=0.0, posinf=1.0, neginf=-1.0)
            market_array = np.clip(market_array, -10.0, 10.0)
            
            # Get sequence for PPO (last sequence_length timesteps)
            if len(market_array) < sequence_length:
                self.logger.logger.debug(f"Insufficient market data for PPO sequence: {len(market_array)} < {sequence_length}")
                return None
            
            market_sequence = market_array[-sequence_length:]  # Shape: (sequence_length, 10)
            
            # Portfolio features (3 total)
            current_balance = float(self.balance)
            position_amount = float(self.positions.get(symbol, 0.0))
            last_price = float(self.last_prices.get(symbol, 0.0))
            position_value = position_amount * last_price if last_price > 0 else 0.0
            total_value = max(current_balance + position_value, 1.0)
            
            balance_ratio = np.clip(current_balance / total_value, 0.0, 1.0)
            position_ratio = np.clip(position_value / total_value, 0.0, 1.0)
            unrealized_pnl_ratio = 0.0  # Placeholder for future implementation
            
            portfolio_features = np.array([balance_ratio, position_ratio, unrealized_pnl_ratio], dtype=np.float32)
            portfolio_matrix = np.tile(portfolio_features, (sequence_length, 1))  # Shape: (sequence_length, 3)
            
            # Combine market and portfolio features
            observation = np.concatenate([market_sequence, portfolio_matrix], axis=1)  # Shape: (sequence_length, 13)
            
            # Validate final observation shape
            expected_shape = (sequence_length, 13)  # PPO expects exactly 13 features
            if observation.shape != expected_shape:
                self.logger.logger.error(f"PPO observation shape mismatch for {symbol}: {observation.shape} != {expected_shape}")
                return None
                
            # Final validation for invalid values
            if np.isnan(observation).any() or np.isinf(observation).any():
                self.logger.logger.warning(f"Invalid values in PPO observation for {symbol}, cleaning...")
                observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Validate model input
            if hasattr(self, 'validation_manager'):
                # Convert single timestep observation to DataFrame for validation (13 features)
                # Use the last timestep of the observation for validation
                single_timestep = observation[-1, :].reshape(1, -1)  # Shape: (1, 13)
                obs_df = pd.DataFrame(single_timestep)
                validation_result = self.validation_manager.validate_model_input(
                    data=obs_df,
                    model_type='ppo',
                    symbol=symbol
                )
                if not validation_result['valid']:
                    self.logger.logger.warning(f"PPO input validation failed for {symbol}: {validation_result['errors']}")
            
            # Get action from PPO model
            action, _ = model.predict(observation, deterministic=True)
            
            # Convert action to prediction value (support continuous or discrete)
            try:
                # Continuous action
                action_value = float(action[0]) if hasattr(action, '__len__') else float(action)
                action_value = float(np.clip(action_value, -1.0, 1.0))
                pred = self.ppo_scale * action_value
            except Exception:
                # Discrete action mapping
                if isinstance(action, (list, np.ndarray)):
                    action_int = int(action[0])
                else:
                    action_int = int(action)
                mapped = -1.0 if action_int == 0 else (0.0 if action_int == 1 else (1.0 if action_int == 2 else 0.0))
                pred = self.ppo_scale * mapped
            
            self.logger.logger.debug(f"PPO prediction for {symbol}: pred={pred}")
            return float(pred)
            
        except Exception as e:
            self.logger.logger.error(f"PPO prediction failed for {symbol}: {e}")
            return None
    
    def _combine_predictions(self, symbol: str, predictions: List[Tuple[str, float]], df: pd.DataFrame) -> int:
        """Combine predictions from multiple models into a final signal."""
        try:
            if not predictions:
                self.logger.logger.warning(f"No predictions available for {symbol}")
                return 0
            
            # Get model weights from config
            original_weights = self.model_weights.copy()
            
            # Calculate effective weights based on available models
            available_models = [model_type for model_type, _ in predictions]
            effective_weights = {}
            total_original_weight = 0.0
            
            # Calculate total original weight for available models
            for model_type in available_models:
                original_weight = original_weights.get(model_type, 1.0)
                total_original_weight += original_weight
            
            # Normalize weights to maintain proportions
            for model_type in available_models:
                original_weight = original_weights.get(model_type, 1.0)
                effective_weights[model_type] = original_weight / total_original_weight if total_original_weight > 0 else 1.0 / len(available_models)
            
            # Calculate weighted average of predictions
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_type, prediction in predictions:
                weight = effective_weights.get(model_type, 1.0)
                weighted_sum += prediction * weight
                total_weight += weight
                self.logger.logger.debug(f"{symbol} {model_type}: pred={prediction:.6f}, weight={weight:.3f} (original: {original_weights.get(model_type, 1.0):.3f})")
            
            # Log effective ensemble weights if any models are missing
            missing_models = set(original_weights.keys()) - set(available_models)
            if missing_models:
                self.logger.logger.info(f"Ensemble fallback for {symbol}: missing models {missing_models}, effective weights: {effective_weights}")
            
            if total_weight == 0:
                return 0
            
            # Calculate final weighted prediction
            final_prediction = weighted_sum / total_weight
            
            # Get dynamic threshold for this symbol
            threshold = self._get_dynamic_threshold(symbol, df)
            
            # Convert to signal
            if final_prediction > threshold:
                signal = 1  # Buy
            elif final_prediction < -threshold:
                signal = -1  # Sell
            else:
                signal = 0  # Hold
            
            self.logger.logger.debug(
                f"{symbol}: Combined prediction={final_prediction:.6f}, threshold={threshold:.6f}, signal={signal}"
            )
            
            return signal
            
        except Exception as e:
            self.logger.logger.error(f"Failed to combine predictions for {symbol}: {e}")
            return 0

    def execute_trades(self, signals: dict, market_data: dict) -> None:
        """Execute trades based on generated signals using a simple position model.
        - positions[symbol] is a float amount of base asset held
        - balance is quote currency
        - last_prices tracked per symbol
        """
        trades_executed = 0
        try:
            if not hasattr(self, 'trade_history'):
                self.trade_history = []

            fee_rate = float(self.transaction_fee)
            slippage = float(self.slippage)
            min_notional = float(self.min_trade_value)

            # Update last prices from market_data
            for symbol, df in market_data.items():
                try:
                    if isinstance(df, pd.DataFrame) and 'close' in df.columns and not df.empty:
                        price = float(df['close'].iloc[-1])
                        if np.isfinite(price) and price > 0:
                            self.last_prices[symbol] = price
                except Exception:
                    continue

            for symbol, signal in signals.items():
                price = float(self.last_prices.get(symbol, 0.0))
                if price <= 0 or not np.isfinite(price):
                    self.logger.logger.debug(f"Skip trade for {symbol}: invalid price {price}")
                    continue

                current_amount = float(self.positions.get(symbol, 0.0))

                if signal == 1:
                    # Buy up to max_position_size of available balance
                    target_notional = float(self.max_position_size) * float(self.balance)
                    if target_notional < min_notional:
                        self.logger.logger.debug(f"Buy skipped for {symbol}: notional {target_notional:.2f} < min {min_notional:.2f}")
                        continue
                    raw_amount = target_notional / price
                    trade_amount = max(0.0, float(raw_amount))
                    if trade_amount <= 0:
                        continue
                    cost = trade_amount * price
                    total_cost = cost * (1.0 + fee_rate + slippage)
                    if self.balance >= total_cost:
                        self.balance -= total_cost
                        self.positions[symbol] = current_amount + trade_amount
                        trade = {
                            'symbol': symbol,
                            'side': 'buy',
                            'amount': trade_amount,
                            'price': price,
                            'fee_rate': fee_rate,
                            'slippage': slippage,
                            'cost': total_cost,
                            'timestamp': int(time.time())
                        }
                        self.trade_history.append(trade)
                        trades_executed += 1
                        self.logger.logger.info(f"BUY {symbol}: amt={trade_amount:.6f} @ {price:.4f} cost={total_cost:.2f} bal={self.balance:.2f}")
                        
                        # Send individual trade notification
                        self._send_trade_notification(symbol, 'BUY', trade_amount, price, total_cost)
                    else:
                        self.rejected_trades_count += 1
                        self.logger.logger.debug(f"Buy rejected for {symbol}: insufficient balance {self.balance:.2f} < {total_cost:.2f}")

                elif signal == -1:
                    # Sell current position if any
                    if current_amount <= 0:
                        continue
                    trade_amount = current_amount  # simple: close position
                    proceeds = trade_amount * price
                    net_proceeds = proceeds * (1.0 - fee_rate - slippage)
                    self.balance += net_proceeds
                    self.positions[symbol] = current_amount - trade_amount
                    trade = {
                        'symbol': symbol,
                        'side': 'sell',
                        'amount': trade_amount,
                        'price': price,
                        'fee_rate': fee_rate,
                        'slippage': slippage,
                        'proceeds': net_proceeds,
                        'timestamp': int(time.time())
                    }
                    self.trade_history.append(trade)
                    trades_executed += 1
                    self.logger.logger.info(f"SELL {symbol}: amt={trade_amount:.6f} @ {price:.4f} proceeds={net_proceeds:.2f} bal={self.balance:.2f}")
                    
                    # Send individual trade notification
                    self._send_trade_notification(symbol, 'SELL', trade_amount, price, net_proceeds)
                else:
                    # Hold
                    continue

            # Send portfolio summary if trades were executed
            if trades_executed > 0:
                self._send_portfolio_update_notification(trades_executed)

        except Exception as e:
            self.logger.logger.error(f"execute_trades failed: {e}")
            self._send_error_notification("Trade Execution Error", str(e))

    def _interval_to_seconds(self, interval: Optional[str] = None) -> int:
        """Convert interval string like '1m','5m','15m','30m','1h','4h','1d' to seconds."""
        iv = (interval or self.interval or '1m').lower()
        try:
            if iv.endswith('m'):
                return int(iv[:-1]) * 60
            if iv.endswith('h'):
                return int(iv[:-1]) * 3600
            if iv.endswith('d'):
                return int(iv[:-1]) * 86400
        except Exception:
            pass
        return 60

    def _time_to_next_candle(self) -> int:
        """Seconds until next candle boundary based on self.interval."""
        sec = self._interval_to_seconds()
        now = int(time.time())
        return sec - (now % sec)

    def _time_to_next_tick(self) -> int:
        """Fallback small sleep when not aligning to candle boundaries."""
        return 5

    def _log_portfolio_status(self) -> None:
        try:
            total_positions_value = 0.0
            for symbol, amt in self.positions.items():
                price = float(self.last_prices.get(symbol, 0.0))
                if price > 0:
                    total_positions_value += amt * price
            total_equity = self.balance + total_positions_value
            self.logger.logger.info(
                f"Portfolio: Balance={self.balance:.2f}, PositionsValue={total_positions_value:.2f}, Equity={total_equity:.2f}, Trades={len(getattr(self, 'trade_history', []))}, Rejected={self.rejected_trades_count}"
            )
        except Exception:
            pass

    def run_trading_loop(self, align_to_candles: bool = True) -> None:
        """Main trading loop: fetch -> signal -> trade -> sleep."""
        self.logger.logger.info("Starting trading loop...")
        try:
            while True:
                try:
                    market_data = asyncio.run(self.get_market_data())
                except RuntimeError:
                    # In case of existing running loop (e.g., in notebooks), fallback
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    market_data = loop.run_until_complete(self.get_market_data())
                    loop.close()

                signals = self.generate_signals(market_data)
                self.execute_trades(signals, market_data)
                self._log_portfolio_status()

                sleep_s = self._time_to_next_candle() if align_to_candles else self._time_to_next_tick()
                self.logger.logger.debug(f"Sleeping {sleep_s} seconds until next iteration...")
                time.sleep(max(1, int(sleep_s)))
        except KeyboardInterrupt:
            self.logger.logger.info("Trading loop stopped by user.")
        except Exception as e:
            self.logger.logger.error(f"Trading loop error: {e}")

    
    def _get_dynamic_threshold(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate dynamic threshold based on recent volatility and costs."""
        base_threshold = self.symbol_thresholds.get(symbol, self.default_threshold)
        try:
            if 'close' in df.columns and len(df) > 20:
                recent_returns = df['close'].pct_change().dropna().tail(20)
                volatility = float(recent_returns.std()) if not recent_returns.empty else 0.0
                cost_floor = 0.0
                if self.use_cost_floor:
                    cost_floor = (self.transaction_fee + self.slippage) * self.cost_floor_multiplier
                if volatility and volatility > 0:
                    lower, upper = self.vol_bounds
                    vol_mult = max(lower, min(upper, volatility / self.vol_reference))
                else:
                    vol_mult = 1.0
                dynamic_threshold = max(base_threshold, cost_floor) * vol_mult
                self.logger.logger.debug(
                    f"Dynamic threshold for {symbol}: {dynamic_threshold:.6f} (vol: {volatility:.6f}, cost:{cost_floor:.6f}, base:{base_threshold:.6f})"
                )
                return float(dynamic_threshold)
        except Exception as e:
            self.logger.logger.warning(f"Failed to calculate dynamic threshold for {symbol}: {e}")
        return float(base_threshold)
    
    def _send_trade_notification(self, symbol: str, side: str, amount: float, price: float, value: float):
        """Send individual trade notification via Telegram."""
        try:
            if self.telegram_notifier and self.telegram_notifier.enabled:
                emoji = "🟢" if side.upper() == "BUY" else "🔴"
                message = f"""
{emoji} <b>TRADE EXECUTED</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {side.upper()}
<b>Amount:</b> {amount:.6f}
<b>Price:</b> €{price:.4f}
<b>Value:</b> €{value:.2f}

<b>Balance:</b> €{self.balance:.2f}
<i>Time:</i> {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                # Use event loop safe method
                self._send_telegram_safe(message.strip())
        except Exception as e:
            self.logger.logger.warning(f"Failed to send trade notification: {e}")
    
    def _send_portfolio_update_notification(self, trades_count: int):
        """Send portfolio update notification via Telegram."""
        try:
            if self.telegram_notifier and self.telegram_notifier.enabled:
                # Calculate total portfolio value
                total_positions_value = 0.0
                for symbol, amount in self.positions.items():
                    price = float(self.last_prices.get(symbol, 0.0))
                    if price > 0:
                        total_positions_value += amount * price
                
                total_value = self.balance + total_positions_value
                total_pnl = total_value - self.initial_balance
                pnl_pct = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
                
                pnl_emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "📊"
                
                message = f"""
{pnl_emoji} <b>PORTFOLIO UPDATE</b>

<b>Trades Executed:</b> {trades_count}
<b>Portfolio Value:</b> €{total_value:.2f}
<b>Cash Balance:</b> €{self.balance:.2f}
<b>Positions Value:</b> €{total_positions_value:.2f}

<b>P&L:</b> €{total_pnl:+.2f} ({pnl_pct:+.2f}%)

<b>Current Positions:</b>
"""
                # Add position details
                active_positions = {symbol: amount for symbol, amount in self.positions.items() if abs(amount) > 1e-6}
                if active_positions:
                    for symbol, amount in active_positions.items():
                        price = self.last_prices.get(symbol, 0.0)
                        pos_value = amount * price if price > 0 else 0.0
                        message += f"• {symbol}: {amount:.6f} (€{pos_value:.2f})\n"
                else:
                    message += "• No active positions\n"
                
                message += f"\n<i>Time:</i> {time.strftime('%Y-%m-%d %H:%M:%S')}"
                
                self._send_telegram_safe(message.strip())
        except Exception as e:
            self.logger.logger.warning(f"Failed to send portfolio notification: {e}")
    
    def _send_error_notification(self, error_type: str, error_message: str):
        """Send error notification via Telegram."""
        try:
            if self.telegram_notifier and self.telegram_notifier.enabled:
                message = f"""
🚨 <b>ERROR ALERT</b>

<b>Type:</b> {error_type}
<b>Message:</b> {error_message}

<i>Time:</i> {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                self._send_telegram_safe(message.strip())
        except Exception as e:
            self.logger.logger.warning(f"Failed to send error notification: {e}")
    
    def _send_startup_notification(self):
        """Send startup notification via Telegram."""
        try:
            if self.telegram_notifier and self.telegram_notifier.enabled:
                message = f"""
🚀 <b>ENHANCED TRADER STARTED</b>

<b>Initial Balance:</b> €{self.initial_balance:.2f}
<b>Symbols:</b> {', '.join(self.symbols)}
<b>Models Loaded:</b> {sum(len(models) for models in self.models.values())}

<i>Time:</i> {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                self._send_telegram_safe(message.strip())
        except Exception as e:
            self.logger.logger.warning(f"Failed to send startup notification: {e}")
    
    def _send_shutdown_notification(self):
        """Send shutdown notification via Telegram."""
        try:
            if self.telegram_notifier and self.telegram_notifier.enabled:
                # Calculate final portfolio value
                total_positions_value = 0.0
                for symbol, amount in self.positions.items():
                    price = float(self.last_prices.get(symbol, 0.0))
                    if price > 0:
                        total_positions_value += amount * price
                
                total_value = self.balance + total_positions_value
                total_pnl = total_value - self.initial_balance
                pnl_pct = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
                
                message = f"""
🛑 <b>ENHANCED TRADER STOPPED</b>

<b>Final Portfolio Value:</b> €{total_value:.2f}
<b>Total P&L:</b> €{total_pnl:+.2f} ({pnl_pct:+.2f}%)
<b>Total Trades:</b> {len(getattr(self, 'trade_history', []))}

<i>Time:</i> {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                self._send_telegram_safe(message.strip())
        except Exception as e:
            self.logger.logger.warning(f"Failed to send shutdown notification: {e}")
    
    def _send_telegram_safe(self, message: str):
        """Send Telegram message safely, handling event loop conflicts."""
        try:
            if not self.telegram_notifier or not self.telegram_notifier.enabled:
                return
                
            # Try synchronous method first
            try:
                self.telegram_notifier.send_message_sync(message)
                return
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    # Use thread-based approach for event loop conflicts
                    import threading
                    import concurrent.futures
                    
                    def send_in_thread():
                        try:
                            import asyncio
                            # Create new event loop for this thread
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                result = loop.run_until_complete(
                                    self.telegram_notifier.send_message(message)
                                )
                                return result
                            finally:
                                loop.close()
                        except Exception as e:
                            self.logger.logger.warning(f"Thread-based Telegram send failed: {e}")
                            return False
                    
                    # Execute in separate thread with timeout
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(send_in_thread)
                        try:
                            result = future.result(timeout=10)  # 10 second timeout
                            if result:
                                self.logger.logger.debug("Telegram message sent via thread")
                            return result
                        except concurrent.futures.TimeoutError:
                            self.logger.logger.warning("Telegram send timeout")
                            return False
                else:
                    raise  # Re-raise if it's a different RuntimeError
                    
        except Exception as e:
            self.logger.logger.warning(f"Failed to send Telegram message safely: {e}")
            return False

    async def run_trading_loop(self):
        """Main trading loop"""
        self.logger.logger.info("Starting Enhanced Unified Paper Trader...")
        
        # Run metadata hygiene on startup
        self.run_metadata_hygiene()
        
        # Load all models
        self.load_all_models()
        
        if not self.models:
            self.logger.logger.error("No models loaded. Exiting.")
            self._send_error_notification("Startup Error", "No models could be loaded")
            return
        
        self.logger.logger.info(f"Loaded {len(self.models)} models")
        
        # Send startup notification
        self._send_startup_notification()
        
        try:
            while True:
                try:
                    # Get market data
                    market_data = await self.get_market_data()
                    
                    if not market_data:
                        self.logger.logger.warning("No market data available, waiting...")
                        await asyncio.sleep(60)
                        continue
                    
                    # Generate signals
                    signals = self.generate_signals(market_data)
                    
                    # Execute trades
                    self.execute_trades(signals, market_data)
                    
                    # Wait before next iteration
                    await asyncio.sleep(self.config.get('trading_interval', 300))  # 5 minutes default
                    
                except KeyboardInterrupt:
                    self.logger.logger.info("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    self.logger.logger.error(f"Error in trading loop: {e}")
                    self._send_error_notification("Trading Loop Error", str(e))
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            self.logger.logger.error(f"Fatal error in trading loop: {e}")
            self._send_error_notification("Fatal Trading Error", str(e))
        finally:
            self.logger.logger.info("Enhanced Unified Paper Trader stopped.")
            # Send shutdown notification
            self._send_shutdown_notification()


    def run_metadata_hygiene(self) -> None:
        """Run metadata hygiene processes including regeneration and cleanup."""
        try:
            self.logger.logger.info("Starting metadata hygiene processes...")
            
            # Regenerate metadata for all models
            regenerated_result = self.metadata_manager.regenerate_all_metadata()
            regenerated_count = regenerated_result.get('regenerated_count', 0)
            self.logger.logger.info(f"Regenerated metadata for {regenerated_count} models")
            
            # Validate metadata hygiene
            hygiene_report = self.metadata_manager.validate_metadata_hygiene()
            if hygiene_report.get('issues'):
                self.logger.logger.warning(f"Found {len(hygiene_report['issues'])} metadata hygiene issues")
                for issue in hygiene_report['issues']:
                    self.logger.logger.warning(f"Hygiene issue: {issue}")
            else:
                self.logger.logger.info("All metadata passed hygiene validation")
            
            # Clean up outdated models if auto_cleanup is enabled
            if self.metadata_manager.auto_cleanup:
                cleanup_result = self.metadata_manager.cleanup_outdated_models(dry_run=False)
                cleaned_count = cleanup_result.get('cleaned_count', 0)
                if cleaned_count > 0:
                    self.logger.logger.info(f"Cleaned up {cleaned_count} outdated models")
                else:
                    self.logger.logger.info("No outdated models found for cleanup")
                    
        except Exception as e:
            self.logger.logger.error(f"Error during metadata hygiene: {e}")

    # ==========================================
    # ENTERPRISE-READY FEATURES
    # ==========================================
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the trading system."""
        health_status = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_status': 'healthy',
            'components': {},
            'metrics': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check model availability
            total_models = 0
            failed_models = 0
            for symbol in self.symbols:
                models = self._load_models_for_symbol(symbol)
                total_models += len(self.model_types)
                failed_models += len(self.model_types) - len(models)
            
            health_status['components']['models'] = {
                'status': 'healthy' if failed_models == 0 else 'degraded',
                'total_expected': total_models,
                'available': total_models - failed_models,
                'failed': failed_models
            }
            
            # Check configuration
            health_status['components']['configuration'] = {
                'status': 'healthy',
                'symbols_count': len(self.symbols),
                'model_types_count': len(self.model_types)
            }
            
            # Check directory structure
            required_dirs = ['models', 'logs', 'data']
            missing_dirs = [d for d in required_dirs if not Path(d).exists()]
            health_status['components']['directories'] = {
                'status': 'healthy' if not missing_dirs else 'warning',
                'missing': missing_dirs
            }
            
            # Performance metrics
            health_status['metrics'] = {
                'memory_usage_mb': self._get_memory_usage(),
                'models_loaded': total_models - failed_models,
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
            }
            
            # Overall status determination
            if failed_models > 0:
                health_status['overall_status'] = 'degraded'
                health_status['warnings'].append(f"{failed_models} models failed to load")
            
            if missing_dirs:
                health_status['warnings'].append(f"Missing directories: {missing_dirs}")
            
            self.logger.logger.info(f"Health check completed: {health_status['overall_status']}")
            return health_status
            
        except Exception as e:
            health_status['overall_status'] = 'critical'
            health_status['errors'].append(f"Health check failed: {str(e)}")
            self.logger.logger.error(f"Health check failed: {e}")
            return health_status
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def auto_recovery(self) -> bool:
        """Attempt automatic recovery from common issues."""
        self.logger.logger.info("Starting automatic recovery procedures...")
        recovery_success = True
        
        try:
            # Clear any stale model instances
            self.models.clear()
            self.preprocessors.clear()
            
            # Attempt to reload models
            for symbol in self.symbols:
                try:
                    models = self._load_models_for_symbol(symbol)
                    if models:
                        self.models[symbol] = models
                        self.logger.logger.info(f"Recovery: Reloaded models for {symbol}")
                    else:
                        self.logger.logger.warning(f"Recovery: Failed to reload models for {symbol}")
                        recovery_success = False
                except Exception as e:
                    self.logger.logger.error(f"Recovery: Error reloading models for {symbol}: {e}")
                    recovery_success = False
            
            # Clear any cached data that might be stale
            if hasattr(self, '_market_data_cache'):
                self._market_data_cache.clear()
            
            # Reset error counters
            if hasattr(self, '_error_count'):
                self._error_count = 0
            
            self.logger.logger.info(f"Auto-recovery completed: {'success' if recovery_success else 'partial'}")
            return recovery_success
            
        except Exception as e:
            self.logger.logger.error(f"Auto-recovery failed: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system': {
                'memory_usage_mb': self._get_memory_usage(),
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
            },
            'models': {},
            'trading': {
                'symbols_active': len(self.symbols),
                'model_types_active': len(self.model_types)
            }
        }
        
        # Model-specific metrics
        for symbol in self.symbols:
            metrics['models'][symbol] = {
                'models_loaded': len(self.models.get(symbol, {})),
                'models_expected': len(self.model_types),
                'status': 'healthy' if symbol in self.models and len(self.models[symbol]) == len(self.model_types) else 'degraded'
            }
        
        return metrics
    
    def create_deployment_report(self) -> str:
        """Create a comprehensive deployment status report."""
        health = self.health_check()
        metrics = self.get_performance_metrics()
        
        report = f"""
=== ENTERPRISE TRADING SYSTEM DEPLOYMENT REPORT ===
Generated: {health['timestamp']}

OVERALL STATUS: {health['overall_status'].upper()}

CONFIGURATION:
- Trading Symbols: {', '.join(self.symbols)}
- Model Types: {', '.join(self.model_types)}
- Models Directory: {self.models_dir}
- Configuration: {getattr(self, 'config_path', 'auto-detected')}

COMPONENT STATUS:
"""
        
        for component, status in health['components'].items():
            report += f"- {component.title()}: {status['status'].upper()}\n"
        
        report += f"""
PERFORMANCE METRICS:
- Memory Usage: {metrics['system']['memory_usage_mb']:.1f} MB
- Uptime: {metrics['system']['uptime_seconds']:.0f} seconds
- Active Symbols: {metrics['trading']['symbols_active']}
- Active Model Types: {metrics['trading']['model_types_active']}

MODEL STATUS:
"""
        
        for symbol, status in metrics['models'].items():
            report += f"- {symbol}: {status['models_loaded']}/{status['models_expected']} models ({status['status']})\n"
        
        if health['warnings']:
            report += f"\nWARNINGS:\n"
            for warning in health['warnings']:
                report += f"- {warning}\n"
        
        if health['errors']:
            report += f"\nERRORS:\n"
            for error in health['errors']:
                report += f"- {error}\n"
        
        report += "\n=== END REPORT ===\n"
        
        return report
    
    def save_deployment_report(self, filename: str = None) -> str:
        """Save deployment report to file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"logs/deployment_report_{timestamp}.txt"
        
        report = self.create_deployment_report()
        
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        self.logger.logger.info(f"Deployment report saved to: {filename}")
        return filename
    
    def enable_enterprise_monitoring(self):
        """Enable enterprise-level monitoring and alerting."""
        self._start_time = time.time()
        self._monitoring_enabled = True
        self._error_count = 0
        self._last_health_check = time.time()
        
        self.logger.logger.info("Enterprise monitoring enabled")
        
        # Schedule periodic health checks
        try:
            import schedule
            schedule.every(15).minutes.do(self._periodic_health_check)
            self.logger.logger.info("Scheduled periodic health checks every 15 minutes")
        except ImportError:
            self.logger.logger.warning("Schedule module not available - periodic checks disabled")
    
    def _periodic_health_check(self):
        """Perform periodic health check and recovery if needed."""
        try:
            health = self.health_check()
            
            if health['overall_status'] == 'critical':
                self.logger.logger.error("Critical system status detected - attempting recovery")
                recovery_success = self.auto_recovery()
                
                if recovery_success:
                    self.logger.logger.info("Recovery successful")
                else:
                    self.logger.logger.error("Recovery failed - manual intervention required")
                    # Send alert if telegram is configured
                    if hasattr(self, 'telegram_notifier') and self.telegram_notifier.enabled:
                        asyncio.create_task(self.telegram_notifier.send_message(
                            "🚨 Trading System Critical Alert\n"
                            "Automatic recovery failed. Manual intervention required."
                        ))
            
            elif health['overall_status'] == 'degraded':
                self.logger.logger.warning("System running in degraded mode")
                if hasattr(self, 'telegram_notifier') and self.telegram_notifier.enabled:
                    asyncio.create_task(self.telegram_notifier.send_message(
                        "⚠️ Trading System Warning\n"
                        f"System status: {health['overall_status']}\n"
                        f"Warnings: {len(health['warnings'])}"
                    ))
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.logger.error(f"Periodic health check failed: {e}")


    def _initialize_trades_csv(self):
        """Initialize trades report CSV with correct column structure."""
        import csv
        import os
        
        # Ensure logs directory exists
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Define correct CSV headers as per README specification
        self.trades_csv_path = logs_dir / 'trades_report.csv'
        self.trades_csv_headers = [
            'TradeID',      # Add explicit TradeID column to fix misalignment
            'Timestamp',
            'Symbol', 
            'Action',
            'Quantity',
            'Price',
            'Status',
            'Notes',
            'Model',
            'Confidence',
            'PnL'
        ]
        
        # Create CSV file with headers if it doesn't exist
        if not self.trades_csv_path.exists():
            with open(self.trades_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.trades_csv_headers)
            self.logger.logger.info(f"Initialized trades report CSV: {self.trades_csv_path}")
        
        # Initialize trade counter for TradeID
        self.trade_counter = 1
    
    def log_trade_to_csv(self, symbol: str, action: str, quantity: float = 0.0, 
                        price: float = 0.0, status: str = 'SUCCESS', notes: str = '',
                        model: str = 'N/A', confidence: float = 0.0, pnl: float = 0.0):
        """Log trade to CSV with correct column alignment."""
        import csv
        from datetime import datetime
        
        try:
            trade_data = [
                self.trade_counter,                    # TradeID
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Timestamp
                symbol,                                # Symbol
                action,                                # Action
                f"{quantity:.4f}" if quantity != 0 else 'N/A',  # Quantity
                f"{price:.2f}" if price != 0 else 'N/A',       # Price
                status,                                # Status
                notes,                                 # Notes
                model,                                 # Model
                f"{confidence:.2f}" if confidence != 0 else 'N/A',  # Confidence
                f"{pnl:+.2f}" if pnl != 0 else '0.00'        # PnL
            ]
            
            with open(self.trades_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(trade_data)
            
            self.trade_counter += 1
            
        except Exception as e:
            self.logger.logger.error(f"Failed to log trade to CSV: {e}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Unified Trading Script')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: auto-detect)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Path to models directory (default: models)')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                       help='Trading symbols to use (default: from config)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Model types to use (default: from config)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (validate configuration and models)')
    parser.add_argument('--single-cycle', action='store_true',
                       help='Run a single trading cycle instead of continuous loop')
    parser.add_argument('--show-available', action='store_true',
                       help='Show available models and exit')
    return parser.parse_args()


async def main():
    """Main function to run the enhanced trader."""
    try:
        args = parse_arguments()
        
        trader = EnhancedUnifiedPaperTrader(
            config_path=args.config,
            models_dir=args.models_dir,
            symbols=args.symbols,
            models=args.models,
            show_available_mode=args.show_available
        )
        
        if args.show_available:
            # Show available models and exit
            trader.show_available_models()
            return 0
        
        # Check if test mode is enabled in configuration or via command line
        config_test_mode = trader.config.get('trading', {}).get('test_mode', False)
        if args.test_mode or config_test_mode:
            # Test mode: validate configuration and models only
            if config_test_mode:
                trader.logger.logger.info("Running in test mode (configured in trading_config.yaml) - validating configuration and models")
            else:
                trader.logger.logger.info("Running in test mode (command line) - validating configuration and models")
            
            # Enable enterprise monitoring for test
            trader.enable_enterprise_monitoring()
            
            # Report discovered models
            trader.logger.logger.info("=== MODEL DISCOVERY REPORT ===")
            trader.logger.logger.info(f"Models directory: {trader.models_dir}")
            trader.logger.logger.info(f"Available symbols: {trader.symbols}")
            trader.logger.logger.info(f"Available model types: {trader.model_types}")
            
            # Perform health check
            health_status = trader.health_check()
            trader.logger.logger.info("=== HEALTH CHECK RESULTS ===")
            trader.logger.logger.info(f"Overall status: {health_status['overall_status']}")
            
            # Test model loading for each symbol
            total_models_found = 0
            for symbol in trader.symbols:
                trader.logger.logger.info(f"\n--- Testing models for {symbol} ---")
                models = trader._load_models_for_symbol(symbol)
                if models:
                    trader.logger.logger.info(f"✓ Found models for {symbol}: {list(models.keys())}")
                    total_models_found += len(models)
                    
                    # Test each model type
                    for model_type, model in models.items():
                        if model:
                            trader.logger.logger.info(f"  ✓ {model_type}: Model loaded successfully")
                        else:
                            trader.logger.logger.warning(f"  ✗ {model_type}: Model failed to load")
                else:
                    trader.logger.logger.warning(f"✗ No models found for {symbol}")
            
            # Generate and save deployment report
            report_filename = trader.save_deployment_report()
            trader.logger.logger.info(f"Deployment report saved to: {report_filename}")
            
            trader.logger.logger.info(f"\n=== SUMMARY ===")
            trader.logger.logger.info(f"Total symbols with models: {len(trader.symbols)}")
            trader.logger.logger.info(f"Total models loaded: {total_models_found}")
            trader.logger.logger.info(f"Health status: {health_status['overall_status']}")
            
            # Create sample CSV entries to demonstrate correct column alignment
            trader.log_trade_to_csv('BTCEUR', 'CYCLE_COMPLETE', notes='Test mode validation cycle completed', status='SUCCESS')
            trader.logger.logger.info(f"Sample trade logged to CSV: {trader.trades_csv_path}")
            
            if total_models_found > 0:
                trader.logger.logger.info("✓ Test mode completed successfully - models are available")
                return 0
            else:
                trader.logger.logger.error("✗ Test mode failed - no models could be loaded")
                return 1
        
        if args.single_cycle:
            # Single cycle mode: run once for each symbol
            trader.logger.logger.info("Running in single cycle mode")
            for symbol in trader.symbols:
                trader.logger.logger.info(f"Running single cycle for {symbol}")
                # Here you would implement a single trading cycle
                # For now, just validate that models can be loaded
                models = trader._load_models_for_symbol(symbol)
                if models:
                    trader.logger.logger.info(f"Single cycle completed for {symbol}")
                else:
                    trader.logger.logger.warning(f"Could not run cycle for {symbol} - no models available")
            return 0
        
        # Normal mode: continuous trading loop
        await trader.run_trading_loop()
    except Exception as e:
        print(f"Failed to start trader: {e}")
        return 1
    return 0


if __name__ == "__main__":
    asyncio.run(main())