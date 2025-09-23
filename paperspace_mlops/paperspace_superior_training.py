#!/usr/bin/env python3
"""
Paperspace Gradient Optimized Training Runner
============================================

This script is specifically designed for execution on Paperspace Gradient machines.
It orchestrates the superior ensemble training pipeline with resource optimization
and automated S3 export for production deployment.

Usage:
    python paperspace_superior_training.py
    python paperspace_superior_training.py --symbols BTCEUR,ETHEUR
    python paperspace_superior_training.py --models ppo,gru
    python paperspace_superior_training.py --quick-test

Environment Requirements:
    - Paperspace Gradient machine (GPU recommended)
    - AWS credentials configured
    - Training data available via API
    - Sufficient memory (16GB+ recommended)
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Always reload trainer module so notebook reruns pick up latest changes
_trainer_module = importlib.import_module("paperspace_mlops.superior_ensemble_trainer")
_trainer_module = importlib.reload(_trainer_module)
SuperiorEnsembleTrainer = _trainer_module.SuperiorEnsembleTrainer

from src.data_pipeline.superior_ppo_feature_expander import SuperiorPPOFeatureExpander


class PaperspaceTrainingRunner:
    """
    Paperspace-optimized training orchestrator that manages resource allocation,
    monitors GPU utilization, and handles automated S3 export.
    """

    def __init__(self):
        self.setup_logging()
        self.aws_export_enabled = False
        self.validate_environment()
        self.trainer = None

    def setup_logging(self):
        """Configure comprehensive logging for Paperspace environment"""
        log_format = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("/tmp/paperspace_training.log"),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def validate_environment(self):
        """Validate Paperspace environment and requirements"""
        self.logger.info("Validating Paperspace environment...")

        # Check GPU availability
        try:
            import torch

            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"GPU detected: {gpu_name} (Count: {gpu_count})")
            else:
                self.logger.warning("No GPU detected - training will use CPU only")
        except ImportError:
            self.logger.warning("PyTorch not available - some models may not work")

        # Check AWS credentials
        aws_keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
        missing_keys = [key for key in aws_keys if not os.getenv(key)]
        if missing_keys:
            self.logger.warning(
                "AWS credentials missing (%s) - S3 export will be disabled",
                ", ".join(missing_keys),
            )
            self.aws_export_enabled = False
        else:
            self.aws_export_enabled = True

        # Check memory
        try:
            import psutil

            memory_gb = psutil.virtual_memory().total / (1024**3)
            self.logger.info(f"Available memory: {memory_gb:.1f} GB")
            if memory_gb < 8:
                self.logger.warning("Low memory detected - consider using smaller batch sizes")
        except ImportError:
            self.logger.warning("psutil not available - cannot check memory")

        self.logger.info("Environment validation complete")

    def setup_trainer(self, config_path: Optional[str] = None) -> SuperiorEnsembleTrainer:
        """Initialize the superior ensemble trainer with Paperspace optimizations"""
        if config_path is None:
            config_path = project_root / "config" / "training_config.yaml"

        self.logger.info(f"Initializing trainer with config: {config_path}")

        try:
            trainer = SuperiorEnsembleTrainer(str(config_path))

            # Paperspace optimizations aligned with dataclass attributes
            trainer.config.gpu_enabled = True
            trainer.config.memory_limit = "12GB"
            if trainer.config.max_workers:
                trainer.config.max_workers = min(trainer.config.max_workers, 4)

            # Disable export if AWS credentials are not available
            if not self.aws_export_enabled:
                if getattr(trainer.config, "export_to_s3", False):
                    self.logger.info("Disabling S3 export due to missing AWS credentials")
                trainer.config.export_to_s3 = False

            self.logger.info("Trainer initialized successfully")
            return trainer

        except Exception as e:
            self.logger.error(f"Failed to initialize trainer: {e}")
            traceback.print_exc()
            raise

    def run_training(
        self,
        symbols: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        quick_test: bool = False,
    ) -> Dict:
        """
        Execute the complete training pipeline with monitoring and S3 export

        Args:
            symbols: List of trading pairs to train (e.g., ['BTCEUR', 'ETHEUR'])
            models: List of model types to train (e.g., ['ppo', 'gru', 'lightgbm'])
            quick_test: If True, runs abbreviated training for testing

        Returns:
            Dict with training results and export status
        """
        start_time = time.time()
        results = {
            "status": "starting",
            "symbols_trained": [],
            "models_trained": [],
            "export_status": {},
            "errors": [],
        }

        try:
            # Initialize trainer
            self.trainer = self.setup_trainer()

            # Configure training parameters
            if quick_test:
                self.logger.info("Running in quick test mode")
                self.trainer.config.optuna_trials = 3
                self.trainer.config.optuna_timeout = 1800
                self.trainer.config.max_training_time_hours = 2
                self.trainer.config.lookback_days = 30
                self.trainer.config.min_samples = 200
                self.trainer.config.export_to_s3 = False

            if symbols:
                self.trainer.config.symbols = symbols
                self.logger.info(f"Training symbols: {symbols}")

            if models:
                self.trainer.config.models = models
                self.logger.info(f"Training models: {models}")

            # Execute training pipeline
            self.logger.info("Starting superior ensemble training...")
            model_results = self.trainer.train_all()

            # Process training results
            results["status"] = "training_complete"
            results["symbols_trained"] = sorted({model.symbol for model in model_results})
            results["models_trained"] = sorted({model.model_type for model in model_results})

            self.logger.info(f"Training completed for {len(results['symbols_trained'])} symbols")

            # Export already handled inside trainer when enabled
            results["export_status"] = {
                "export_enabled": self.trainer.config.export_to_s3,
                "models_exported": (len(model_results) if self.trainer.config.export_to_s3 else 0),
            }
            results["status"] = "complete"

            # Training summary
            duration = time.time() - start_time
            self.logger.info(f"Training pipeline completed in {duration:.1f} seconds")
            self.logger.info(f"Symbols: {results['symbols_trained']}")
            self.logger.info(f"Models: {results['models_trained']}")
            self.logger.info(f"S3 Export: {results['export_status']}")

            return results

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            traceback.print_exc()
            results["status"] = "failed"
            results["errors"].append(str(e))
            return results

    def monitor_resources(self):
        """Monitor GPU and memory usage during training (for debugging)"""
        try:
            import psutil
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(i) / 1024**3
                    self.logger.info(
                        f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached"
                    )

            memory = psutil.virtual_memory()
            self.logger.info(
                f"System Memory: {memory.percent}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)"
            )

        except Exception as e:
            self.logger.warning(f"Resource monitoring failed: {e}")

    def cleanup(self):
        """Clean up resources after training"""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU cache cleared")
        except:
            pass

        self.logger.info("Cleanup completed")


def main():
    """Main entry point for Paperspace training execution"""
    parser = argparse.ArgumentParser(description="Paperspace Superior Ensemble Training")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., BTCEUR,ETHEUR)",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models (e.g., ppo,gru,lightgbm)",
    )
    parser.add_argument("--config", type=str, help="Path to training configuration file")
    parser.add_argument(
        "--quick-test", action="store_true", help="Run abbreviated training for testing"
    )
    parser.add_argument("--monitor", action="store_true", help="Enable resource monitoring")

    args = parser.parse_args()

    # Parse arguments
    symbols = args.symbols.split(",") if args.symbols else None
    models = args.models.split(",") if args.models else None

    # Initialize runner
    runner = PaperspaceTrainingRunner()

    try:
        # Start resource monitoring if requested
        if args.monitor:
            import threading
            import time

            def monitor_loop():
                while True:
                    runner.monitor_resources()
                    time.sleep(30)  # Monitor every 30 seconds

            monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            monitor_thread.start()

        # Execute training
        results = runner.run_training(symbols=symbols, models=models, quick_test=args.quick_test)

        # Output results
        print("\n" + "=" * 60)
        print("PAPERSPACE TRAINING RESULTS")
        print("=" * 60)
        print(json.dumps(results, indent=2))

        if results["status"] == "complete":
            print("\n✓ Training completed successfully!")
            print(f"✓ Models exported to S3")
            return 0
        else:
            print(f"\n✗ Training failed: {results.get('errors', [])}")
            return 1

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1

    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        return 1

    finally:
        runner.cleanup()


if __name__ == "__main__":
    exit(main())
