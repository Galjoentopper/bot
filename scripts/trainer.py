#!/usr/bin/env python3
"""
Centralized Training Script using DatasetBuilder
===============================================

This script provides a unified interface for training multiple model types
using the centralized DatasetBuilder and modern ML practices:

- Centralized dataset assembly with caching
- Time-series cross-validation with leakage prevention  
- Cost-aware evaluation with realistic trading costs
- Parallel training across symbols
- Unified artifact management
- MLflow integration for experiment tracking
"""

import os
import sys
import argparse
import json
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dataset_builder import DatasetBuilder
from gru_trainer import GRUTrainer
from lgbm_trainer import LightGBMTrainer
from ppo_trainer import PPOTrainer
from cost_aware_evaluation import CostModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CentralizedTrainer:
    """
    Centralized trainer that orchestrates multiple model types using
    the DatasetBuilder for consistent data processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize centralized trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        
        # Initialize DatasetBuilder
        self.dataset_builder = DatasetBuilder(
            data_dir=config.get('data_dir'),
            cache_dir=config.get('cache_dir'),
            feature_config=config.get('feature_config')
        )
        
        # Initialize trainers
        self.trainers = {}
        if 'gru' in config['models']:
            self.trainers['gru'] = GRUTrainer(
                dataset_builder=self.dataset_builder,
                config=config.get('gru_config'),
                cost_model_config=config.get('cost_model')
            )
        
        if 'lightgbm' in config['models'] or 'lgbm' in config['models']:
            self.trainers['lightgbm'] = LightGBMTrainer(
                dataset_builder=self.dataset_builder,
                config=config.get('lightgbm_config'),
                cost_model_config=config.get('cost_model')
            )
        
        if 'ppo' in config['models']:
            self.trainers['ppo'] = PPOTrainer(
                dataset_builder=self.dataset_builder,
                config=config.get('ppo_config')
            )
        
        # Set up MLflow if configured
        self.use_mlflow = config.get('mlflow', {}).get('enabled', False)
        if self.use_mlflow:
            self._setup_mlflow()
        
        logger.info(f"Initialized trainer with models: {list(self.trainers.keys())}")
    
    def train_all(self) -> Dict[str, Any]:
        """
        Train all models for all symbols.
        
        Returns:
            Dictionary containing all training results
        """
        start_time = time.time()
        
        # Get symbols to train
        symbols = self.config['symbols']
        models = list(self.trainers.keys())
        
        logger.info(f"Starting training for {len(symbols)} symbols and {len(models)} models")
        
        # Create training tasks
        tasks = []
        for symbol in symbols:
            for model_name in models:
                tasks.append((symbol, model_name))
        
        # Execute training (parallel or sequential)
        if self.config.get('parallel', False):
            results = self._train_parallel(tasks)
        else:
            results = self._train_sequential(tasks)
        
        # Aggregate results
        total_time = time.time() - start_time
        summary = self._create_summary(results, total_time)
        
        # Save results
        self._save_results(summary)
        
        # Log to MLflow
        if self.use_mlflow:
            self._log_to_mlflow(summary)
        
        logger.info(f"Training completed in {total_time:.1f}s")
        return summary
    
    def _train_sequential(self, tasks: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Train models sequentially."""
        results = {}
        
        for i, (symbol, model_name) in enumerate(tasks, 1):
            logger.info(f"Training {model_name} for {symbol} ({i}/{len(tasks)})")
            
            try:
                trainer = self.trainers[model_name]
                result = trainer.train_symbol(
                    symbol=symbol,
                    interval=self.config.get('interval', '15m'),
                    n_splits=self.config.get('n_splits', 5),
                    calibrate=self.config.get('calibrate', True),
                    save_artifacts=self.config.get('save_artifacts', True)
                )
                
                results[f"{model_name}_{symbol}"] = result
                logger.info(f"✅ {model_name}_{symbol} completed")
                
            except Exception as e:
                logger.error(f"❌ {model_name}_{symbol} failed: {e}")
                results[f"{model_name}_{symbol}"] = {'error': str(e)}
        
        return results
    
    def _train_parallel(self, tasks: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Train models in parallel using multiprocessing."""
        max_workers = min(self.config.get('max_workers', mp.cpu_count()), len(tasks))
        logger.info(f"Using {max_workers} parallel workers")
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(self._train_single_task, symbol, model_name): (symbol, model_name)
                for symbol, model_name in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                symbol, model_name = future_to_task[future]
                task_key = f"{model_name}_{symbol}"
                
                try:
                    result = future.result()
                    results[task_key] = result
                    logger.info(f"✅ {task_key} completed")
                    
                except Exception as e:
                    logger.error(f"❌ {task_key} failed: {e}")
                    results[task_key] = {'error': str(e)}
        
        return results
    
    def _train_single_task(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """Train a single model for a single symbol (for multiprocessing)."""
        # Recreate trainer in subprocess
        trainer_class = type(self.trainers[model_name])
        trainer = trainer_class(
            dataset_builder=self.dataset_builder,
            config=getattr(self.trainers[model_name], 'config', {}),
            cost_model_config=self.config.get('cost_model')
        )
        
        return trainer.train_symbol(
            symbol=symbol,
            interval=self.config.get('interval', '15m'),
            n_splits=self.config.get('n_splits', 5),
            calibrate=self.config.get('calibrate', True),
            save_artifacts=self.config.get('save_artifacts', True)
        )
    
    def _create_summary(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Create training summary."""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        failed_results = {k: v for k, v in results.items() if 'error' in v}
        
        summary = {
            'config': self.config,
            'total_time': total_time,
            'total_tasks': len(results),
            'successful_tasks': len(successful_results),
            'failed_tasks': len(failed_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Calculate aggregate metrics
        if successful_results:
            net_sharpes = []
            net_returns = []
            
            for result in successful_results.values():
                if 'avg_net_sharpe' in result:
                    net_sharpes.append(result['avg_net_sharpe'])
                if 'avg_net_return' in result:
                    net_returns.append(result['avg_net_return'])
            
            if net_sharpes:
                summary['avg_net_sharpe'] = sum(net_sharpes) / len(net_sharpes)
                summary['best_net_sharpe'] = max(net_sharpes)
            
            if net_returns:
                summary['avg_net_return'] = sum(net_returns) / len(net_returns)
                summary['best_net_return'] = max(net_returns)
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]) -> None:
        """Save training results."""
        # Create results directory
        results_dir = Path('results') / 'training'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f'training_results_{timestamp}.json'
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save latest results
        latest_path = results_dir / 'latest_results.json'
        with open(latest_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow for experiment tracking."""
        try:
            import mlflow
            
            mlflow_config = self.config['mlflow']
            
            # Set tracking URI if specified
            if 'tracking_uri' in mlflow_config:
                mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
            
            # Set experiment
            experiment_name = mlflow_config.get('experiment_name', 'centralized_training')
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLflow initialized with experiment: {experiment_name}")
            
        except ImportError:
            logger.warning("MLflow not available, skipping experiment tracking")
            self.use_mlflow = False
    
    def _log_to_mlflow(self, summary: Dict[str, Any]) -> None:
        """Log results to MLflow."""
        try:
            import mlflow
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params({
                    'symbols': ','.join(self.config['symbols']),
                    'models': ','.join(self.config['models']),
                    'n_splits': self.config.get('n_splits', 5),
                    'interval': self.config.get('interval', '15m'),
                    'parallel': self.config.get('parallel', False),
                    'calibrate': self.config.get('calibrate', True)
                })
                
                # Log metrics
                mlflow.log_metrics({
                    'total_time': summary['total_time'],
                    'success_rate': summary['success_rate'],
                    'total_tasks': summary['total_tasks'],
                    'successful_tasks': summary['successful_tasks'],
                    'failed_tasks': summary['failed_tasks']
                })
                
                # Log aggregate performance metrics
                if 'avg_net_sharpe' in summary:
                    mlflow.log_metric('avg_net_sharpe', summary['avg_net_sharpe'])
                if 'best_net_sharpe' in summary:
                    mlflow.log_metric('best_net_sharpe', summary['best_net_sharpe'])
                if 'avg_net_return' in summary:
                    mlflow.log_metric('avg_net_return', summary['avg_net_return'])
                
                # Log dataset metadata
                dataset_metadata = self.dataset_builder.feature_config
                mlflow.log_dict(dataset_metadata, 'dataset_config.json')
                
                logger.info("Results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        'symbols': ['BTCEUR', 'ETHEUR'],
        'models': ['lightgbm', 'gru'],
        'interval': '15m',
        'n_splits': 5,
        'calibrate': True,
        'save_artifacts': True,
        'parallel': False,
        'max_workers': mp.cpu_count(),
        
        # Cost model
        'cost_model': {
            'fee_bps': 10.0,
            'slippage_bps': 5.0,
            'min_position_size': 10.0,
            'max_position_size': 100000.0
        },
        
        # Model-specific configs
        'gru_config': {
            'sequence_length': 60,
            'gru_units': 64,
            'epochs': 50,
            'batch_size': 32,
            'patience': 10
        },
        
        'lightgbm_config': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'early_stopping_rounds': 50
        },
        
        'ppo_config': {
            'episodes': 50,
            'hidden_units': 128,
            'max_steps_per_episode': 500
        },
        
        # MLflow config
        'mlflow': {
            'enabled': False,
            'experiment_name': 'centralized_training'
        }
    }


def main():
    """Main training execution."""
    parser = argparse.ArgumentParser(
        description='Centralized training script with DatasetBuilder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LightGBM for BTC and ETH
  python scripts/trainer.py --symbols BTCEUR ETHEUR --models lightgbm
  
  # Train all models with parallel execution
  python scripts/trainer.py --models lightgbm gru ppo --parallel --max-workers 4
  
  # Train with custom parameters
  python scripts/trainer.py --n-splits 3 --fee-bps 15 --slippage-bps 8
        """
    )
    
    # Data options
    parser.add_argument('--symbols', nargs='+', 
                       choices=['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR'],
                       default=['BTCEUR', 'ETHEUR'],
                       help='Symbols to train (default: BTCEUR ETHEUR)')
    
    parser.add_argument('--models', nargs='+',
                       choices=['lightgbm', 'lgbm', 'gru', 'ppo'],
                       default=['lightgbm'],
                       help='Models to train (default: lightgbm)')
    
    parser.add_argument('--interval', default='15m',
                       help='Data interval (default: 15m)')
    
    # Cross-validation options
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of CV splits (default: 5)')
    
    parser.add_argument('--embargo', type=float, default=0.02,
                       help='Embargo percentage for time series CV (default: 0.02)')
    
    # Cost model options
    parser.add_argument('--fee-bps', type=float, default=10.0,
                       help='Trading fees in basis points (default: 10.0)')
    
    parser.add_argument('--slippage-bps', type=float, default=5.0,
                       help='Slippage in basis points (default: 5.0)')
    
    # Training options
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel training across symbols/models')
    
    parser.add_argument('--max-workers', type=int, default=mp.cpu_count(),
                       help=f'Maximum parallel workers (default: {mp.cpu_count()})')
    
    parser.add_argument('--no-calibrate', action='store_true',
                       help='Disable probability calibration')
    
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving artifacts')
    
    # Caching options
    parser.add_argument('--cache-dir', 
                       help='Feature cache directory (default: data/features)')
    
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild feature cache')
    
    # MLflow options
    parser.add_argument('--mlflow', action='store_true',
                       help='Enable MLflow experiment tracking')
    
    parser.add_argument('--experiment-name', default='centralized_training',
                       help='MLflow experiment name')
    
    # Other options
    parser.add_argument('--config', 
                       help='JSON configuration file path')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config['symbols'] = args.symbols
    config['models'] = [m.replace('lgbm', 'lightgbm') for m in args.models]  # Normalize
    config['interval'] = args.interval
    config['n_splits'] = args.n_splits
    config['parallel'] = args.parallel
    config['max_workers'] = args.max_workers
    config['calibrate'] = not args.no_calibrate
    config['save_artifacts'] = not args.no_save
    
    # Update cost model
    config['cost_model']['fee_bps'] = args.fee_bps
    config['cost_model']['slippage_bps'] = args.slippage_bps
    
    # Update MLflow config
    config['mlflow']['enabled'] = args.mlflow
    config['mlflow']['experiment_name'] = args.experiment_name
    
    # Set cache directory if specified
    if args.cache_dir:
        config['cache_dir'] = args.cache_dir
    
    # Set random seed for reproducibility
    import numpy as np
    import random
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
    except ImportError:
        pass
    
    # Print configuration
    logger.info("Training Configuration:")
    logger.info(f"  Symbols: {config['symbols']}")
    logger.info(f"  Models: {config['models']}")
    logger.info(f"  CV Splits: {config['n_splits']}")
    logger.info(f"  Parallel: {config['parallel']}")
    logger.info(f"  Cost Model: {config['cost_model']['fee_bps']}bps fees, {config['cost_model']['slippage_bps']}bps slippage")
    
    # Initialize and run trainer
    try:
        trainer = CentralizedTrainer(config)
        results = trainer.train_all()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Time: {results['total_time']:.1f}s")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Tasks: {results['successful_tasks']}/{results['total_tasks']}")
        
        if 'avg_net_sharpe' in results:
            print(f"Average Net Sharpe: {results['avg_net_sharpe']:.4f}")
        if 'best_net_sharpe' in results:
            print(f"Best Net Sharpe: {results['best_net_sharpe']:.4f}")
        
        # Print individual results
        print("\nIndividual Results:")
        for task_name, result in results['results'].items():
            if 'error' in result:
                print(f"  ❌ {task_name}: {result['error']}")
            else:
                net_sharpe = result.get('avg_net_sharpe', 'N/A')
                print(f"  ✅ {task_name}: Net Sharpe {net_sharpe}")
        
        print(f"\nResults saved to results/training/")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()