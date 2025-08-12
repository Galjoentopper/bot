#!/usr/bin/env python3
"""
GRU Trainer using Centralized DatasetBuilder
=============================================

This module provides a GRU (Gated Recurrent Unit) trainer that uses
the centralized DatasetBuilder for consistent data processing.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from sklearn.preprocessing import StandardScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from dataset_builder import DatasetBuilder
from model_adapter import ModelAdapter
from time_series_cv import get_time_series_folds
from cost_aware_evaluation import CostAwareEvaluator
from calibration_utils import calibrate_model_probabilities

logger = logging.getLogger(__name__)


class GRUModelAdapter(ModelAdapter):
    """GRU model adapter for sequence modeling."""
    
    def __init__(self, name: str = "GRU", config: Optional[Dict[str, Any]] = None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for GRU models")
        
        super().__init__(name, config)
        self.scaler = None
        self.sequence_length = config.get('sequence_length', 60) if config else 60
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            train_idx: np.ndarray,
            val_idx: np.ndarray,
            **kwargs) -> 'GRUModelAdapter':
        """Train GRU model on sequences."""
        
        # Split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)
        
        # Build model
        self.model = self._build_gru_model(X_train.shape[1:])
        
        # Train
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 32),
            verbose=0,
            callbacks=self._get_callbacks()
        )
        
        self.is_fitted = True
        self.metadata['training_history'] = {k: [float(x) for x in v] for k, v in history.history.items()}
        
        logger.info(f"GRU training completed: {len(history.epoch)} epochs")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate GRU predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Scale features
        X_scaled = self.scaler.transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(X.shape)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate GRU prediction probabilities."""
        return self.predict(X)  # GRU outputs are already probabilities
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get GRU model artifacts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Save model in SavedModel format (more reliable than weights)
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model')
            self.model.save(model_path)
            
            # Read saved model files
            model_files = {}
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, model_path)
                    with open(file_path, 'rb') as f:
                        model_files[rel_path] = f.read()
        
        return {
            'name': self.name,
            'config': self.config,
            'model_files': model_files,
            'scaler': self.scaler,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted,
            'sequence_length': self.sequence_length
        }
    
    def _restore_from_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore GRU from artifacts."""
        import tempfile
        import shutil
        
        self.config = artifacts['config']
        self.scaler = artifacts['scaler']
        self.metadata = artifacts['metadata']
        self.is_fitted = artifacts['is_fitted']
        self.sequence_length = artifacts['sequence_length']
        
        # Restore model from saved files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model')
            os.makedirs(model_path, exist_ok=True)
            
            # Write model files
            for rel_path, file_data in artifacts['model_files'].items():
                full_path = os.path.join(model_path, rel_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'wb') as f:
                    f.write(file_data)
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
    
    def _build_gru_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build GRU model architecture."""
        config = self.config or {}
        
        model = models.Sequential([
            layers.GRU(
                config.get('gru_units', 64), 
                return_sequences=True, 
                input_shape=input_shape,
                dropout=config.get('dropout', 0.1),
                recurrent_dropout=config.get('recurrent_dropout', 0.1)
            ),
            layers.GRU(
                config.get('gru_units', 64) // 2,
                dropout=config.get('dropout', 0.1),
                recurrent_dropout=config.get('recurrent_dropout', 0.1)
            ),
            layers.Dense(config.get('dense_units', 32), activation='relu'),
            layers.Dropout(config.get('dropout', 0.1)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        optimizer = config.get('optimizer', 'adam')
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001))
        else:
            opt = optimizer
            
        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _get_callbacks(self) -> list:
        """Get training callbacks."""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('patience', 15),
                restore_best_weights=True,
                min_delta=1e-4
            )
        ]
        
        if self.config.get('reduce_lr_on_plateau', True):
            callbacks_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=0
                )
            )
        
        return callbacks_list


class GRUTrainer:
    """
    GRU trainer using centralized DatasetBuilder.
    
    Provides end-to-end training pipeline with:
    - Centralized dataset assembly
    - Time-series cross-validation
    - Cost-aware evaluation
    - Probability calibration
    """
    
    def __init__(self, 
                 dataset_builder: DatasetBuilder,
                 config: Optional[Dict[str, Any]] = None,
                 cost_model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize GRU trainer.
        
        Args:
            dataset_builder: Centralized dataset builder
            config: GRU model configuration
            cost_model_config: Cost model configuration
        """
        self.dataset_builder = dataset_builder
        self.config = config or self._get_default_config()
        self.cost_evaluator = CostAwareEvaluator()
        
        # Configure cost model if provided
        if cost_model_config:
            from cost_aware_evaluation import CostModel
            cost_model = CostModel(**cost_model_config)
            self.cost_evaluator = CostAwareEvaluator(cost_model)
    
    def train_symbol(self, 
                    symbol: str,
                    interval: str = "15m",
                    n_splits: int = 5,
                    calibrate: bool = True,
                    save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Train GRU model for a single symbol.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            n_splits: Number of CV splits
            calibrate: Whether to calibrate probabilities
            save_artifacts: Whether to save model artifacts
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training GRU model for {symbol}")
        
        # Get dataset
        features_df, metadata = self.dataset_builder.get_dataset(
            symbol=symbol, 
            interval=interval
        )
        
        # Validate dataset
        validation_report = self.dataset_builder.validate_dataset(features_df, metadata)
        if not validation_report['valid']:
            raise ValueError(f"Dataset validation failed: {validation_report['errors']}")
        
        # Prepare sequences for GRU
        X_sequences, y, timestamps = self._prepare_sequences(features_df)
        
        if len(X_sequences) == 0:
            raise ValueError("No valid sequences created")
        
        # Time-series cross-validation
        cv_folds = get_time_series_folds(
            timestamps=pd.DatetimeIndex(timestamps),
            n_splits=n_splits,
            embargo_pct=0.02
        )
        
        # Train on each fold
        fold_results = []
        models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            logger.info(f"Training fold {fold_idx + 1}/{len(cv_folds)}")
            
            # Create model adapter
            model_adapter = GRUModelAdapter(
                name=f"GRU_{symbol}_fold_{fold_idx}",
                config=self.config
            )
            
            # Train
            model_adapter.fit(X_sequences, y, train_idx, val_idx)
            
            # Evaluate
            fold_metrics = self._evaluate_fold(
                model_adapter, X_sequences, y, train_idx, val_idx, features_df, timestamps
            )
            fold_metrics['fold'] = fold_idx
            fold_results.append(fold_metrics)
            
            models.append(model_adapter)
        
        # Aggregate results
        results = self._aggregate_fold_results(fold_results)
        results['symbol'] = symbol
        results['metadata'] = metadata.__dict__
        results['validation_report'] = validation_report
        
        # Train final model on full dataset if requested
        if save_artifacts:
            final_model = self._train_final_model(X_sequences, y, symbol)
            
            # Calibrate probabilities if requested
            if calibrate:
                final_model, calibrator = self._calibrate_model(final_model, X_sequences, y)
                results['calibrated'] = True
            else:
                calibrator = None
                results['calibrated'] = False
            
            # Save artifacts
            self._save_artifacts(final_model, calibrator, symbol, results)
        
        logger.info(f"GRU training completed for {symbol}: {results['avg_net_sharpe']:.4f} net Sharpe")
        return results
    
    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Prepare sequences for GRU training."""
        sequence_length = self.config.get('sequence_length', 60)
        
        # Select features (exclude target and non-numeric columns)
        feature_columns = [col for col in df.columns 
                          if col != 'target' and df[col].dtype in ['float64', 'int64']]
        
        if len(feature_columns) == 0:
            raise ValueError("No numeric features found")
        
        # Extract features and targets
        X = df[feature_columns].values
        y = df['target'].values if 'target' in df.columns else None
        
        if y is None:
            raise ValueError("No target column found")
        
        # Create sequences
        sequences = []
        targets = []
        seq_timestamps = []
        
        for i in range(sequence_length, len(X)):
            sequence = X[i-sequence_length:i]
            target = y[i]
            timestamp = df.index[i]
            
            # Skip if any NaN in sequence or target
            if not (np.isnan(sequence).any() or np.isnan(target)):
                sequences.append(sequence)
                targets.append(target)
                seq_timestamps.append(timestamp)
        
        return np.array(sequences), np.array(targets), pd.DatetimeIndex(seq_timestamps)
    
    def _evaluate_fold(self, 
                      model: GRUModelAdapter,
                      X: np.ndarray, 
                      y: np.ndarray,
                      train_idx: np.ndarray,
                      val_idx: np.ndarray,
                      features_df: pd.DataFrame,
                      timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """Evaluate model performance on validation fold."""
        
        # Get predictions
        val_proba = model.predict_proba(X[val_idx])
        
        # Get returns for cost-aware evaluation
        returns_col = 'returns_1' if 'returns_1' in features_df.columns else 'close'
        if returns_col == 'close':
            # Calculate returns if not available
            returns = features_df['close'].pct_change().values
        else:
            returns = features_df[returns_col].values
        
        # Align returns with validation indices (considering sequence offset)
        sequence_length = self.config.get('sequence_length', 60)
        val_returns = returns[val_idx + sequence_length]
        
        # Find optimal threshold and evaluate
        optimal_threshold, metrics = self.cost_evaluator.find_optimal_threshold(
            y_true=y[val_idx],
            y_pred_proba=val_proba,
            returns=val_returns,
            position_size=self.config.get('position_size', 1000.0)
        )
        
        return {
            'optimal_threshold': optimal_threshold,
            'net_sharpe': metrics.net_sharpe_ratio,
            'gross_return': metrics.gross_return,
            'net_return': metrics.net_return,
            'num_trades': metrics.num_trades,
            'win_rate': metrics.win_rate,
            'max_drawdown': metrics.max_drawdown,
            'total_costs': metrics.total_costs
        }
    
    def _aggregate_fold_results(self, fold_results: list) -> Dict[str, Any]:
        """Aggregate results across folds."""
        if not fold_results:
            return {}
        
        # Calculate averages
        metrics = ['net_sharpe', 'gross_return', 'net_return', 'win_rate', 'max_drawdown']
        aggregated = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        # Sum total metrics
        sum_metrics = ['num_trades', 'total_costs']
        for metric in sum_metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'total_{metric.replace("total_", "")}'] = np.sum(values)
        
        aggregated['n_folds'] = len(fold_results)
        aggregated['fold_results'] = fold_results
        
        return aggregated
    
    def _train_final_model(self, X: np.ndarray, y: np.ndarray, symbol: str) -> GRUModelAdapter:
        """Train final model on full dataset."""
        # Use 80/20 split for final training
        split_idx = int(0.8 * len(X))
        train_idx = np.arange(split_idx)
        val_idx = np.arange(split_idx, len(X))
        
        final_model = GRUModelAdapter(
            name=f"GRU_{symbol}_final",
            config=self.config
        )
        
        final_model.fit(X, y, train_idx, val_idx)
        return final_model
    
    def _calibrate_model(self, model: GRUModelAdapter, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Any]:
        """Calibrate model probabilities."""
        # Use last 20% for calibration
        split_idx = int(0.8 * len(X))
        X_cal = X[split_idx:]
        y_cal = y[split_idx:]
        
        calibrated_model, calibrator = calibrate_model_probabilities(
            model=model,
            X_cal=X_cal,
            y_cal=y_cal,
            method="isotonic"
        )
        
        return calibrated_model, calibrator
    
    def _save_artifacts(self, 
                       model: GRUModelAdapter,
                       calibrator: Optional[Any],
                       symbol: str,
                       results: Dict[str, Any]) -> None:
        """Save model artifacts."""
        # Create artifact directory
        artifacts_dir = Path("models") / "gru" / symbol.lower()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifacts_dir / "model.pkl"
        model.save(model_path)
        
        # Save calibrator if available
        if calibrator:
            calibrator_path = artifacts_dir / "calibrator.pkl"
            calibrator.save(calibrator_path)
        
        # Save training results
        results_path = artifacts_dir / "training_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create latest symlink
        latest_dir = artifacts_dir.parent / "latest"
        if latest_dir.is_symlink():
            latest_dir.unlink()
        elif latest_dir.exists():
            import shutil
            shutil.rmtree(latest_dir)
        
        latest_dir.symlink_to(artifacts_dir.name, target_is_directory=True)
        
        logger.info(f"Artifacts saved to {artifacts_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default GRU configuration."""
        return {
            'sequence_length': 60,
            'gru_units': 64,
            'dense_units': 32,
            'dropout': 0.1,
            'recurrent_dropout': 0.1,
            'epochs': 100,
            'batch_size': 32,
            'patience': 15,
            'learning_rate': 0.001,
            'position_size': 1000.0,
            'reduce_lr_on_plateau': True,
            'optimizer': 'adam'
        }