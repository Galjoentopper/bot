"""
Paperspace Training Orchestrator
===============================

Complete MLOps pipeline for automated training on Paperspace Gradient
with intelligent time management and automatic export to production server.

Features:
- Smart data fetching with caching
- Parallel model training with time optimization
- Automated model packaging and export
- Production server notification and deployment
- Failure recovery and monitoring
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests
import yaml

# Paperspace environment detection
IS_PAPERSPACE = (
    os.environ.get("PAPERSPACE_JOB_ID") is not None or os.environ.get("GRADIENT_JOB_ID") is not None
)


class PaperspaceOrchestrator:
    """Main orchestrator for automated training pipeline"""

    def __init__(self, config_path: str = "training_config.yaml"):
        # Look for config file in multiple locations
        possible_paths = [
            config_path,
            f"../{config_path}",
            f"../../{config_path}",
            "/notebooks/bot/training_config.yaml",
            "/content/bot/training_config.yaml",
            "/workspace/bot/training_config.yaml",
        ]

        self.config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.config_path = path
                print(f"✅ Found config file: {path}")
                break

        if not self.config_path:
            # Create a basic config file if none found
            print("⚠️ No config file found, creating basic configuration...")
            self.config_path = self._create_basic_config()

        self.original_config_path = config_path
        self.start_time = datetime.now()
        self.max_runtime_hours = 5.5  # Leave 30 min buffer
        self.logger = self._setup_logging()

        # Load configuration
        self.config = self._load_config()

        # Pipeline state
        self.pipeline_state = {
            "start_time": self.start_time,
            "current_stage": "initialization",
            "completed_stages": [],
            "models_trained": {},
            "errors": [],
            "warnings": [],
        }

        # Paths
        if IS_PAPERSPACE:
            self.workspace_dir = Path("/notebooks")
        else:
            self.workspace_dir = Path(".")

        self.data_dir = self.workspace_dir / "data"
        self.models_dir = self.workspace_dir / "models"
        self.export_dir = self.workspace_dir / "exports"
        self.logs_dir = self.workspace_dir / "logs"

        # Create directories (only if we have write permissions)
        for dir_path in [self.data_dir, self.models_dir, self.export_dir, self.logs_dir]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                self.logger.warning(f"⚠️ Cannot create directory {dir_path} - using temp directory")
                # Fallback to temp directory for testing
                import tempfile

                temp_base = Path(tempfile.mkdtemp())
                self.data_dir = temp_base / "data"
                self.models_dir = temp_base / "models"
                self.export_dir = temp_base / "exports"
                self.logs_dir = temp_base / "logs"
                for temp_dir in [self.data_dir, self.models_dir, self.export_dir, self.logs_dir]:
                    temp_dir.mkdir(parents=True, exist_ok=True)
                break

        self.logger.info(f"🚀 Paperspace Training Orchestrator initialized")
        self.logger.info(f"📁 Workspace: {self.workspace_dir}")
        self.logger.info(f"⏰ Max runtime: {self.max_runtime_hours} hours")

    def _create_basic_config(self) -> str:
        """Create a basic configuration file for Paperspace"""

        basic_config = {
            "symbols": ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"],
            "interval": "30m",
            "lookback_days": 180,
            "model_weights": {"lightgbm": 0.55, "gru": 0.35, "ppo": 0.1},
            "thresholds": {
                "per_symbol": {
                    "BTCEUR": {"buy": 0.6, "sell": 0.4},
                    "ETHEUR": {"buy": 0.6, "sell": 0.4},
                    "ADAEUR": {"buy": 0.6, "sell": 0.4},
                    "DOTEUR": {"buy": 0.6, "sell": 0.4},
                    "LINKEUR": {"buy": 0.6, "sell": 0.4},
                }
            },
            "models": {
                "gru": {
                    "epochs": 50,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "sequence_length": 60,
                },
                "lightgbm": {
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "verbose": 0,
                    "n_estimators": 100,
                },
                "ppo": {
                    "total_timesteps": 50000,
                    "learning_rate": 0.0003,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                },
            },
        }

        config_path = "paperspace_training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(basic_config, f, default_flow_style=False)

        print(f"✅ Created basic config file: {config_path}")
        return config_path

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"paperspace_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        return logging.getLogger(__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration"""

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            self.logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config

        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {e}")
            raise

    def get_time_remaining(self) -> float:
        """Get remaining runtime in hours"""

        elapsed = datetime.now() - self.start_time
        remaining = self.max_runtime_hours - elapsed.total_seconds() / 3600
        return max(0, remaining)

    def should_continue(self, min_hours_needed: float = 0.5) -> bool:
        """Check if we have enough time to continue"""

        remaining = self.get_time_remaining()
        should_continue = remaining >= min_hours_needed

        if not should_continue:
            self.logger.warning(
                f"⏰ Time constraint: {remaining:.2f}h remaining, need {min_hours_needed}h"
            )

        return should_continue

    def update_pipeline_state(self, stage: str, status: str = "completed", **kwargs):
        """Update pipeline state tracking"""

        self.pipeline_state["current_stage"] = stage

        if status == "completed":
            self.pipeline_state["completed_stages"].append(stage)

        # Add any additional info
        for key, value in kwargs.items():
            self.pipeline_state[key] = value

        # Save state to file
        state_file = self.logs_dir / "pipeline_state.json"
        with open(state_file, "w") as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)

        elapsed = datetime.now() - self.start_time
        remaining = self.get_time_remaining()

        self.logger.info(
            f"🔄 Stage: {stage} ({status}) | Elapsed: {elapsed.total_seconds()/3600:.1f}h | Remaining: {remaining:.1f}h"
        )

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute the complete MLOps pipeline"""

        try:
            self.logger.info("🎯 Starting Full MLOps Pipeline")
            self.logger.info("=" * 60)

            # Stage 1: Environment Setup
            if self.should_continue(0.1):
                self.setup_environment()
            else:
                raise TimeoutError("Insufficient time for environment setup")

            # Stage 2: Data Acquisition
            if self.should_continue(0.5):
                data_status = self.fetch_and_prepare_data()
                if not data_status["success"]:
                    raise RuntimeError("Data preparation failed")
            else:
                raise TimeoutError("Insufficient time for data preparation")

            # Stage 3: Smart Training
            if self.should_continue(2.0):  # Need at least 2 hours for training
                training_results = self.execute_smart_training()
                if not training_results["success"]:
                    self.logger.warning("⚠️ Some models failed training")
            else:
                raise TimeoutError("Insufficient time for model training")

            # Stage 4: Model Packaging
            if self.should_continue(0.3):
                packaging_results = self.package_trained_models()
                if not packaging_results["success"]:
                    raise RuntimeError("Model packaging failed")
            else:
                raise TimeoutError("Insufficient time for model packaging")

            # Stage 5: Export and Transfer
            if self.should_continue(0.2):
                export_results = self.export_and_transfer_models()
                if not export_results["success"]:
                    self.logger.warning("⚠️ Model transfer encountered issues")
            else:
                raise TimeoutError("Insufficient time for model export")

            # Stage 6: Final reporting
            self.generate_final_report()

            self.logger.info("🎉 Pipeline completed successfully!")

            return {
                "success": True,
                "pipeline_state": self.pipeline_state,
                "runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            }

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")

            # Try to save whatever we have
            try:
                self.emergency_export()
            except Exception as export_error:
                self.logger.error(f"❌ Emergency export failed: {export_error}")

            return {
                "success": False,
                "error": str(e),
                "pipeline_state": self.pipeline_state,
                "runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            }

    def setup_environment(self):
        """Setup training environment and install dependencies"""

        self.update_pipeline_state("environment_setup", "started")

        try:
            # Clone/download the training repository if needed
            if IS_PAPERSPACE:
                self._setup_paperspace_environment()

            # Install/verify dependencies
            self._verify_dependencies()

            # Setup MLflow tracking
            self._setup_mlflow()

            self.update_pipeline_state("environment_setup", "completed")
            self.logger.info("✅ Environment setup completed")

        except Exception as e:
            self.logger.error(f"❌ Environment setup failed: {e}")
            raise

    def _setup_paperspace_environment(self):
        """Setup Paperspace-specific environment"""

        self.logger.info("🔧 Setting up Paperspace environment")

        # Check if we need to clone the repository
        if not (self.workspace_dir / "src").exists():
            # Option 1: Clone from Git (if public repo)
            repo_url = os.environ.get("TRADING_BOT_REPO_URL")
            if repo_url:
                cmd = f"git clone {repo_url} {self.workspace_dir}"
                subprocess.run(cmd, shell=True, check=True)

            # Option 2: Download from URL (if private)
            else:
                self.logger.info("📦 Repository should be pre-uploaded to Paperspace")

        # Install requirements
        requirements_file = self.workspace_dir / "requirements.txt"
        if requirements_file.exists():
            cmd = f"pip install -r {requirements_file}"
            subprocess.run(cmd, shell=True, check=True)

        # Install package in editable mode
        if (self.workspace_dir / "setup.py").exists():
            cmd = f"pip install -e {self.workspace_dir}"
            subprocess.run(cmd, shell=True, check=True)

    def _verify_dependencies(self):
        """Verify all required dependencies are available"""

        required_packages = [
            "torch",
            "lightgbm",
            "stable-baselines3",
            "pandas",
            "numpy",
            "scikit-learn",
            "yfinance",
            "python-binance",
            "mlflow",
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.logger.warning(f"⚠️ Missing packages: {missing_packages}")
            # Try to install them
            cmd = f"pip install {' '.join(missing_packages)}"
            subprocess.run(cmd, shell=True)

    def _setup_mlflow(self):
        """Setup MLflow tracking"""

        mlflow_dir = self.workspace_dir / "mlruns"
        mlflow_dir.mkdir(exist_ok=True)

        os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir}"

    def fetch_and_prepare_data(self) -> Dict[str, Any]:
        """Intelligent data fetching with caching"""

        self.update_pipeline_state("data_preparation", "started")

        try:
            symbols = self.config["data_acquisition"]["symbols"]
            interval = self.config["data_acquisition"]["interval"]
            lookback_days = self.config["data_acquisition"]["lookback_days"]

            self.logger.info(f"📊 Fetching data for {len(symbols)} symbols")
            self.logger.info(f"📈 Interval: {interval}, Lookback: {lookback_days} days")

            # Use existing DatasetBuilder with smart caching
            from src.data_pipeline.dataset_builder import DatasetBuilder

            dataset_builder = DatasetBuilder(
                config=self.config,
                cache_dir=self.data_dir / "cache",
                force_refresh=False,  # Use cache if available
            )

            # Build datasets for all symbols
            datasets = {}
            failed_symbols = []

            for symbol in symbols:
                try:
                    self.logger.info(f"📊 Processing {symbol}...")

                    dataset = dataset_builder.build_dataset(symbol=symbol, save_to_disk=True)

                    if dataset is not None and len(dataset) > 1000:  # Minimum data requirement
                        datasets[symbol] = dataset
                        self.logger.info(f"✅ {symbol}: {len(dataset)} samples")
                    else:
                        failed_symbols.append(symbol)
                        self.logger.warning(f"⚠️ {symbol}: Insufficient data")

                except Exception as e:
                    failed_symbols.append(symbol)
                    self.logger.error(f"❌ {symbol}: {str(e)}")

            if not datasets:
                raise RuntimeError("No valid datasets could be created")

            # Update config to only include successful symbols
            self.config["data_acquisition"]["symbols"] = list(datasets.keys())

            self.update_pipeline_state(
                "data_preparation",
                "completed",
                datasets_created=len(datasets),
                failed_symbols=failed_symbols,
            )

            return {"success": True, "datasets": len(datasets), "failed_symbols": failed_symbols}

        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            self.update_pipeline_state("data_preparation", "failed", error=str(e))
            return {"success": False, "error": str(e)}

    def execute_smart_training(self) -> Dict[str, Any]:
        """Execute optimized training with time management"""

        self.update_pipeline_state("model_training", "started")

        try:
            symbols = self.config["data_acquisition"]["symbols"]
            model_types = self.config["training"]["models"]

            total_models = len(symbols) * len(model_types)
            time_remaining = self.get_time_remaining()
            time_per_model = (
                time_remaining - 0.8
            ) / total_models  # Reserve 0.8h for packaging/export

            self.logger.info(f"🎯 Training {total_models} models")
            self.logger.info(f"⏱️ Time budget: {time_per_model:.2f}h per model")

            if time_per_model < 0.1:  # Less than 6 minutes per model
                self.logger.warning("⚠️ Very limited time per model - using fast training mode")
                self._enable_fast_training_mode()

            # Smart parallel training
            training_results = self._execute_parallel_training(symbols, model_types, time_per_model)

            successful_models = sum(
                1 for result in training_results.values() if result.get("success", False)
            )

            self.update_pipeline_state(
                "model_training",
                "completed",
                total_models=total_models,
                successful_models=successful_models,
                training_results=training_results,
            )

            return {
                "success": successful_models > 0,
                "total_models": total_models,
                "successful_models": successful_models,
                "results": training_results,
            }

        except Exception as e:
            self.logger.error(f"❌ Training execution failed: {e}")
            self.update_pipeline_state("model_training", "failed", error=str(e))
            return {"success": False, "error": str(e)}

    def _enable_fast_training_mode(self):
        """Enable fast training mode for time-constrained situations"""

        self.logger.info("🚀 Enabling fast training mode")

        # Reduce training parameters for speed
        if "gru" in self.config["training"]["models"]:
            self.config.setdefault("gru", {})
            self.config["gru"]["epochs"] = min(self.config["gru"].get("epochs", 50), 20)
            self.config["gru"]["batch_size"] = max(self.config["gru"].get("batch_size", 32), 128)

        if "lightgbm" in self.config["training"]["models"]:
            self.config.setdefault("lightgbm", {})
            self.config["lightgbm"]["num_boost_round"] = min(
                self.config["lightgbm"].get("num_boost_round", 1000), 300
            )

        if "ppo" in self.config["training"]["models"]:
            self.config.setdefault("ppo", {})
            self.config["ppo"]["total_timesteps"] = min(
                self.config["ppo"].get("total_timesteps", 100000), 50000
            )

        # Reduce cross-validation
        self.config["training"]["cv_splits"] = min(self.config["training"].get("cv_splits", 5), 3)

    def _execute_parallel_training(
        self, symbols: List[str], model_types: List[str], time_per_model: float
    ) -> Dict[str, Any]:
        """Execute training with smart parallelization"""

        results = {}

        # Determine optimal parallelization
        max_workers = min(psutil.cpu_count(), 4)  # Limit to 4 for memory reasons

        # Create training tasks
        training_tasks = []
        for symbol in symbols:
            for model_type in model_types:
                task = {
                    "symbol": symbol,
                    "model_type": model_type,
                    "time_budget": time_per_model,
                    "task_id": f"{model_type}_{symbol}",
                }
                training_tasks.append(task)

        self.logger.info(f"🔄 Starting parallel training with {max_workers} workers")

        # Execute training tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._train_single_model, task): task for task in training_tasks
            }

            for future in future_to_task:
                task = future_to_task[future]
                task_id = task["task_id"]

                try:
                    result = future.result(timeout=time_per_model * 3600 + 300)  # Add 5 min buffer
                    results[task_id] = result

                    status = "✅" if result.get("success", False) else "❌"
                    self.logger.info(f"{status} {task_id}: {result.get('message', 'Unknown')}")

                except Exception as e:
                    results[task_id] = {"success": False, "error": str(e)}
                    self.logger.error(f"❌ {task_id}: {str(e)}")

                # Check time constraints
                if not self.should_continue(0.5):
                    self.logger.warning("⏰ Time limit approaching - stopping new training tasks")
                    break

        return results

    def _train_single_model(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single model with timeout protection"""

        symbol = task["symbol"]
        model_type = task["model_type"]
        time_budget = task["time_budget"]

        try:
            # Import enhanced trainer
            sys.path.append(str(self.workspace_dir))
            from scripts.enhanced_trainer import EnhancedTrainer

            # Create trainer instance
            trainer = EnhancedTrainer(config=self.config)

            # Execute training with timeout
            start_time = time.time()

            result = trainer.train_model(
                model_type=model_type,
                symbol=symbol,
                max_time_seconds=time_budget * 3600 * 0.9,  # Use 90% of budget
            )

            execution_time = time.time() - start_time

            return {
                "success": True,
                "symbol": symbol,
                "model_type": model_type,
                "execution_time": execution_time,
                "message": f"Training completed in {execution_time:.1f}s",
                "result": result,
            }

        except Exception as e:
            return {
                "success": False,
                "symbol": symbol,
                "model_type": model_type,
                "error": str(e),
                "message": f"Training failed: {str(e)}",
            }

    def package_trained_models(self) -> Dict[str, Any]:
        """Package all trained models for export"""

        self.update_pipeline_state("model_packaging", "started")

        try:
            from src.utils.model_packaging import ModelPackager

            packager = ModelPackager(models_dir=self.models_dir, output_dir=self.export_dir)

            # Find all trained models
            model_files = list(self.models_dir.rglob("*.pkl")) + list(self.models_dir.rglob("*.pt"))

            if not model_files:
                raise RuntimeError("No trained model files found")

            self.logger.info(f"📦 Packaging {len(model_files)} model files")

            # Create comprehensive package
            package_info = packager.create_comprehensive_package(
                package_name=f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                include_metadata=True,
                include_config=True,
            )

            self.update_pipeline_state(
                "model_packaging",
                "completed",
                packaged_models=len(model_files),
                package_info=package_info,
            )

            return {"success": True, "package_info": package_info, "model_files": len(model_files)}

        except Exception as e:
            self.logger.error(f"❌ Model packaging failed: {e}")
            self.update_pipeline_state("model_packaging", "failed", error=str(e))
            return {"success": False, "error": str(e)}

    def export_and_transfer_models(self) -> Dict[str, Any]:
        """Export models and transfer to production server"""

        self.update_pipeline_state("model_export", "started")

        try:
            # Find packaged models
            package_files = list(self.export_dir.glob("*.zip"))

            if not package_files:
                raise RuntimeError("No packaged model files found")

            transfer_results = []

            for package_file in package_files:
                self.logger.info(f"📤 Transferring {package_file.name}")

                # Try multiple transfer methods
                transfer_result = self._transfer_package(package_file)
                transfer_results.append(transfer_result)

            successful_transfers = sum(
                1 for result in transfer_results if result.get("success", False)
            )

            self.update_pipeline_state(
                "model_export",
                "completed",
                total_packages=len(package_files),
                successful_transfers=successful_transfers,
            )

            return {
                "success": successful_transfers > 0,
                "total_packages": len(package_files),
                "successful_transfers": successful_transfers,
                "transfer_results": transfer_results,
            }

        except Exception as e:
            self.logger.error(f"❌ Model export failed: {e}")
            self.update_pipeline_state("model_export", "failed", error=str(e))
            return {"success": False, "error": str(e)}

    def _transfer_package(self, package_file: Path) -> Dict[str, Any]:
        """Transfer a single package to production server using enhanced transfer service"""

        # Use the enhanced model transfer service
        from paperspace_mlops.model_transfer_service import transfer_models_package

        metadata = {
            "pipeline_start_time": self.start_time.isoformat(),
            "pipeline_state": self.pipeline_state,
            "paperspace_job_id": os.environ.get("PAPERSPACE_JOB_ID", "unknown"),
            "symbols_trained": self.config["data_acquisition"]["symbols"],
            "models_trained": self.config["training"]["models"],
        }

        return transfer_models_package(package_file, self.config, metadata)

    def generate_final_report(self):
        """Generate comprehensive pipeline report"""

        self.update_pipeline_state("reporting", "started")

        report = {
            "pipeline_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
                "success": len(self.pipeline_state["errors"]) == 0,
            },
            "stages_completed": self.pipeline_state["completed_stages"],
            "models_trained": self.pipeline_state.get("training_results", {}),
            "system_info": {
                "platform": "paperspace_gradient",
                "job_id": os.environ.get("PAPERSPACE_JOB_ID", "unknown"),
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
            },
            "errors": self.pipeline_state["errors"],
            "warnings": self.pipeline_state["warnings"],
        }

        # Save report
        report_file = (
            self.export_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"📊 Final report saved: {report_file}")
        self.update_pipeline_state("reporting", "completed")

    def emergency_export(self):
        """Emergency export of any available models"""

        self.logger.warning("🚨 Executing emergency export")

        try:
            # Find any model files
            model_files = list(self.models_dir.rglob("*.pkl")) + list(self.models_dir.rglob("*.pt"))

            if model_files:
                # Create emergency package
                emergency_zip = (
                    self.export_dir
                    / f"emergency_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                )

                with zipfile.ZipFile(emergency_zip, "w") as zf:
                    for model_file in model_files:
                        zf.write(model_file, model_file.name)

                self.logger.info(f"🚨 Emergency package created: {emergency_zip}")

                # Try to transfer
                self._transfer_package(emergency_zip)

        except Exception as e:
            self.logger.error(f"❌ Emergency export failed: {e}")


def main():
    """Main entry point for Paperspace training"""

    import argparse

    parser = argparse.ArgumentParser(description="Paperspace MLOps Training Orchestrator")
    parser.add_argument(
        "--config", default="training_config.yaml", help="Training configuration file"
    )
    parser.add_argument("--max-hours", type=float, default=5.5, help="Maximum runtime hours")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual training)")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PaperspaceOrchestrator(config_path=args.config)
    orchestrator.max_runtime_hours = args.max_hours

    if args.dry_run:
        print("🧪 DRY RUN MODE - No actual training will be performed")
        return

    # Execute pipeline
    try:
        result = orchestrator.run_full_pipeline()

        if result["success"]:
            print("🎉 Pipeline completed successfully!")
            return 0
        else:
            print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            return 1

    except KeyboardInterrupt:
        print("⚠️ Pipeline interrupted by user")
        orchestrator.emergency_export()
        return 1

    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        orchestrator.emergency_export()
        return 1


if __name__ == "__main__":
    exit(main())
