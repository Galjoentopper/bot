#!/usr/bin/env python3
"""
Paperspace Training Script
==========================

Main training script for Paperspace Gradient. Uses existing data from the data/ folder
(no data fetching) and trains models with optimal time management and export.

This script assumes:
- paperspace_setup.py has been run successfully
- Data files exist in bot/data/ folder (uploaded from production server)
- Environment is properly configured

Usage:
    python paperspace_training.py                    # Train all models
    python paperspace_training.py --models gru lgbm  # Train specific models
    python paperspace_training.py --symbols BTCEUR   # Train specific symbols
    python paperspace_training.py --dry-run          # Test without training
    python paperspace_training.py --fast             # Fast mode for time limits
"""

import json
import logging
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.notifier.telegram import TelegramNotifier

import numpy as np
import psutil
import yaml

# Robustly add project root and src/ to sys.path regardless of CWD or notebooks mount
try:
    _this_file = Path(__file__).resolve()
    _project_root = _this_file.parent.parent  # repo root (parent of paperspace_mlops)
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    _src_root = _project_root / "src"
    if _src_root.exists() and str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
except Exception:
    pass

# S3 upload support
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# Legacy fallback for older notebook mounting conventions
sys.path.append("/notebooks/bot") if Path("/notebooks").exists() else None
sys.path.append("/notebooks/bot/src") if Path("/notebooks").exists() else None

# Telegram notifications
try:
    from src.notifier.telegram import TelegramNotifier

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PaperspaceTraining:
    """Main training orchestrator for Paperspace"""

    def __init__(self, config_path: str = "training_config.yaml", max_hours: float = 5.5):
        self.start_time = datetime.now()
        self.max_runtime_hours = max_hours
        self.is_paperspace = Path("/notebooks").exists()

        # Directory setup
        if self.is_paperspace:
            self.workspace_dir = Path("/notebooks")
            self.bot_dir = self.workspace_dir / "bot"
        else:
            self.workspace_dir = Path(".")
            self.bot_dir = self.workspace_dir

        self.data_dir = self.bot_dir / "data"
        self.models_dir = self.bot_dir / "models"
        self.export_dir = self.workspace_dir / "exports"
        self.logs_dir = self.workspace_dir / "logs"

        # Create directories
        for dir_path in [self.models_dir, self.export_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config(config_path)
        self._set_global_seeds()

        # Initialize Telegram notifications
        self.telegram_notifier = self._init_telegram_notifier()

        # Training state
        self.pipeline_state = {
            "start_time": self.start_time,
            "current_stage": "initialization",
            "models_trained": {},
            "export_path": None,
            "errors": [],
        }

        logger.info(f"🚀 Paperspace Training initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"⏰ Max runtime: {self.max_runtime_hours} hours")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        # Look for config file
        possible_paths = [
            self.bot_dir / config_path,
            self.bot_dir / "paperspace_mlops" / config_path,
            self.workspace_dir / config_path,
            config_path,
        ]

        for path in possible_paths:
            if Path(path).exists():
                logger.info(f"✅ Loading config: {path}")
                with open(path, "r") as f:
                    return yaml.safe_load(f)

        raise FileNotFoundError(f"Config file {config_path} not found in any location")

    def _set_global_seeds(self):
        """Set global random seeds for reproducibility"""
        try:
            seed = self.config.get("training", {}).get("random_seed", 42)

            # Set Python seed
            import random

            random.seed(seed)

            # Set NumPy seed
            import numpy as np

            np.random.seed(seed)

            # Set PyTorch seeds
            try:
                import torch

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except ImportError:
                pass

            logger.info(f"🎯 Set global random seed: {seed}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set global seeds: {e}")

    def _init_telegram_notifier(self) -> Optional["TelegramNotifier"]:
        """Initialize Telegram notifier for training updates"""
        if not TELEGRAM_AVAILABLE:
            logger.info("📱 Telegram notifications disabled - library not available")
            return None

        try:
            # Get Telegram configuration from environment or config
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID")

            if not bot_token or not chat_id:
                logger.info("📱 Telegram notifications disabled - missing credentials")
                return None

            notifier = TelegramNotifier(
                bot_token=bot_token,
                chat_id=chat_id,
                enabled=True,
                rate_limit_per_sec=1.0,
                max_retries=3,
            )

            logger.info("📱 Telegram notifications enabled")
            return notifier

        except Exception as e:
            logger.warning(f"📱 Failed to initialize Telegram notifier: {e}")
            return None

    def verify_data_availability(self) -> Dict[str, int]:
        """Verify data files are available and get sample counts"""
        logger.info("🗄️ Verifying data availability...")

        if not self.data_dir.exists():
            raise RuntimeError(f"Data directory not found: {self.data_dir}")

        # Find database files
        db_files = list(self.data_dir.glob("*.db"))

        if not db_files:
            raise RuntimeError(f"No database files found in: {self.data_dir}")

        # Get sample counts for each database
        data_stats = {}
        symbols = self.config.get("data_acquisition", {}).get("symbols", [])
        interval = self.config.get("data_acquisition", {}).get("interval", "30m")

        for symbol in symbols:
            db_file = self.data_dir / f"{symbol.lower()}_{interval}.db"
            if db_file.exists():
                try:
                    import sqlite3

                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM market_data")
                    count = cursor.fetchone()[0]
                    conn.close()
                    data_stats[symbol] = count
                    logger.info(f"  ✅ {symbol}: {count:,} samples")
                except Exception as e:
                    logger.warning(f"  ⚠️ {symbol}: Could not read database - {e}")
                    data_stats[symbol] = 0
            else:
                logger.warning(f"  ❌ {symbol}: Database not found")
                data_stats[symbol] = 0

        total_samples = sum(data_stats.values())
        logger.info(f"📊 Total samples available: {total_samples:,}")

        if total_samples == 0:
            raise RuntimeError("No data available for training")

        return data_stats

    def should_continue(self, additional_hours: float) -> bool:
        """Check if we have enough time to continue"""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        remaining = self.max_runtime_hours - elapsed

        if remaining < additional_hours:
            logger.warning(
                f"⏰ Insufficient time: {remaining:.1f}h remaining, need {additional_hours:.1f}h"
            )
            return False
        return True

    def prepare_datasets(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Prepare datasets from local databases"""
        self.pipeline_state["current_stage"] = "data_preparation"
        logger.info("📊 Preparing datasets from local databases...")

        try:
            from src.data_pipeline.dataset_builder import DatasetBuilder
        except ImportError:
            raise ImportError("Could not import DatasetBuilder. Make sure src/ is in Python path")

        if symbols is None:
            symbols = self.config.get("data_acquisition", {}).get("symbols", [])

        interval = self.config.get("data_acquisition", {}).get("interval", "30m")

        # Initialize dataset builder
        builder = DatasetBuilder(
            data_dir=str(self.data_dir),
            cache_dir=str(self.models_dir / "metadata"),
            config=self.config,
        )

        datasets = {}
        failed_symbols = []

        for symbol in symbols:
            try:
                logger.info(f"📥 Building dataset for {symbol}...")

                result = builder.build_dataset(symbol=symbol, interval=interval, use_cache=True)

                if result and len(result) > 4:
                    X, y, timestamps, feature_names, metadata = result
                    datasets[symbol] = {
                        "data": (X, y, timestamps),
                        "features": feature_names,
                        "metadata": metadata,
                        "sample_count": len(X),
                    }
                    logger.info(f"  ✅ {symbol}: {len(X)} samples, {len(feature_names)} features")
                else:
                    logger.error(f"  ❌ {symbol}: Failed to build dataset")
                    failed_symbols.append(symbol)

            except Exception as e:
                logger.error(f"  ❌ {symbol}: {e}")
                failed_symbols.append(symbol)

        if not datasets:
            raise RuntimeError("No datasets could be prepared")

        # Update config to only include successful symbols
        self.config["data_acquisition"]["symbols"] = list(datasets.keys())

        logger.info(f"✅ Prepared {len(datasets)} datasets")
        logger.info(f"❌ Failed: {len(failed_symbols)} symbols")

        return {
            "success": True,
            "datasets": datasets,
            "failed_symbols": failed_symbols,
            "total_samples": sum(d["sample_count"] for d in datasets.values()),
        }

    def train_models(
        self,
        datasets: Dict[str, Any],
        model_types: Optional[List[str]] = None,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """Train models using prepared datasets"""
        self.pipeline_state["current_stage"] = "model_training"
        logger.info("🎯 Starting model training...")

        if model_types is None:
            model_types = self.config.get("training", {}).get("models", ["gru", "lightgbm", "ppo"])

        symbols = list(datasets["datasets"].keys())
        total_tasks = len(symbols) * len(model_types)

        # Calculate time allocation
        elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        remaining_time = self.max_runtime_hours - elapsed - 1.0  # Keep 1h buffer for export
        time_per_task = remaining_time / total_tasks if total_tasks > 0 else 0.5

        if fast_mode:
            time_per_task = min(time_per_task, 0.25)  # Max 15 minutes per model in fast mode

        logger.info(f"⏱️  Time allocation: {time_per_task:.1f}h per model")
        logger.info(
            f"🎯 Training {total_tasks} models ({len(symbols)} symbols × {len(model_types)} types)"
        )

        # Prepare training tasks
        training_tasks = []
        for symbol in symbols:
            for model_type in model_types:
                training_tasks.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "dataset": datasets["datasets"][symbol],
                        "time_limit": time_per_task,
                        "fast_mode": fast_mode,
                    }
                )

        # Execute training (sequential to avoid memory issues)
        trained_models = {}
        failed_models = []

        for i, task in enumerate(training_tasks, 1):
            if not self.should_continue(0.5):
                logger.warning(f"⏰ Time limit reached, stopping at {i-1}/{total_tasks}")
                break

            logger.info(f"🔄 Training {i}/{total_tasks}: {task['symbol']} {task['model_type']}")

            try:
                result = self._train_single_model(task)
                if result["success"]:
                    model_key = f"{task['symbol']}_{task['model_type']}"
                    trained_models[model_key] = result
                    logger.info(f"  ✅ Success: {result.get('score', 'N/A')}")
                else:
                    failed_models.append(f"{task['symbol']}_{task['model_type']}")
                    logger.error(f"  ❌ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"  ❌ Exception: {e}")
                failed_models.append(f"{task['symbol']}_{task['model_type']}")

        self.pipeline_state["models_trained"] = trained_models

        logger.info(
            f"✅ Training complete: {len(trained_models)} successful, {len(failed_models)} failed"
        )

        return {
            "success": len(trained_models) > 0,
            "trained_models": trained_models,
            "failed_models": failed_models,
            "total_trained": len(trained_models),
            "total_failed": len(failed_models),
        }

    def _train_single_model(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single model with timeout protection"""
        import signal
        import traceback

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Training timeout for {task['symbol']} {task['model_type']}")

        try:
            # Set timeout
            if hasattr(signal, "SIGALRM"):  # Unix only
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(task["time_limit"] * 3600))

            # Import trainer
            model_type = task["model_type"]
            if model_type == "gru":
                from src.models.gru_trainer import GRUTrainer

                trainer = GRUTrainer(self.config)
            elif model_type == "lightgbm":
                from src.models.lgbm_trainer import LightGBMTrainer

                trainer = LightGBMTrainer(self.config)
            elif model_type == "ppo":
                from src.models.ppo_trainer import PPOTrainer

                trainer = PPOTrainer(self.config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Prepare data
            X, y, timestamps = task["dataset"]["data"]

            # Clean data: ensure X is numeric and handle any datetime issues
            if hasattr(X, "select_dtypes"):  # if X is DataFrame
                # Keep only numeric columns
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X = X[numeric_cols].values
            elif isinstance(X, np.ndarray):
                # Ensure array is numeric
                X = np.asarray(X, dtype=np.float32)

            # Ensure y is numeric
            y = np.asarray(y, dtype=np.float32)

            # Train model
            model_path = self.models_dir / model_type / task["symbol"]
            model_path.mkdir(parents=True, exist_ok=True)

            # Call train method with appropriate signature for each model type
            if model_type == "gru":
                # GRU trainer handles sequence creation internally
                # Just pass 2D data and let the trainer handle reshaping
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                logger.info(
                    f"GRU input data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}"
                )

                result = trainer.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    save_path=str(model_path),
                    experiment_name=f"gru_{task['symbol'].lower()}",
                    verbose=False,
                )
            elif model_type == "lightgbm":
                # LightGBM expects train/validation split
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                result = trainer.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    experiment_name=f"lgbm_{task['symbol'].lower()}",
                    save_path=str(model_path),
                )
            elif model_type == "ppo":
                # PPO expects DataFrame with proper columns including 'close'
                import pandas as pd

                # Create DataFrame from features with timestamps as index
                features = task["dataset"]["features"]
                logger.info(f"PPO input X shape: {X.shape}, expected features: {len(features)}")

                # Ensure feature count matches
                if X.shape[1] != len(features):
                    logger.warning(
                        f"Feature count mismatch: X has {X.shape[1]} features, but expected {len(features)}"
                    )
                    # Use the minimum to avoid index errors
                    min_features = min(X.shape[1], len(features))
                    X = X[:, :min_features]
                    features = features[:min_features]
                    logger.info(f"Truncated to {min_features} features for consistency")

                df_data = pd.DataFrame(X, columns=features)
                df_data.index = pd.to_datetime(timestamps)

                # Add target column
                df_data["target"] = y[: len(df_data)]

                # Ensure we have a 'close' column - look for it in feature names first
                if "close" not in df_data.columns:
                    # Look for close-related columns
                    close_candidates = [col for col in df_data.columns if "close" in col.lower()]
                    if close_candidates:
                        df_data["close"] = df_data[close_candidates[0]]
                        logger.info(f"Using {close_candidates[0]} as close price for PPO")
                    else:
                        # Create synthetic close from cumulative returns of target
                        df_data["close"] = (
                            1 + df_data["target"].shift(1).fillna(0)
                        ).cumprod() * 50000  # Start at ~50k
                        logger.info("Created synthetic close price from target returns for PPO")

                # Split train/eval
                split_idx = int(len(df_data) * 0.8)
                train_data = df_data[:split_idx].copy()
                eval_data = df_data[split_idx:].copy()

                logger.info(f"PPO data shapes - train: {train_data.shape}, eval: {eval_data.shape}")
                logger.info(f"PPO columns: {list(train_data.columns)}")

                result = trainer.train(
                    train_data=train_data,
                    eval_data=eval_data,
                    total_timesteps=50000 if task["fast_mode"] else 100000,
                    experiment_name=f"ppo_{task['symbol'].lower()}",
                    save_path=str(model_path),
                )

                # Export per-symbol PPO feature index for deployment pinning
                try:
                    # Build a minimal OHLCV frame from available data
                    import pandas as pd

                    if "close" in train_data.columns:
                        close = train_data["close"].astype(float)
                    else:
                        # Fallback: synthetic close from target
                        tgt = train_data.get("target", pd.Series(0.0, index=train_data.index))
                        close = (1 + tgt.shift(1).fillna(0)).cumprod() * 1000.0
                    open_ = close.shift(1).fillna(close)
                    high = pd.concat([open_, close], axis=1).max(axis=1)
                    low = pd.concat([open_, close], axis=1).min(axis=1)
                    volume = (
                        train_data["volume"]
                        if "volume" in train_data.columns
                        else pd.Series(1.0, index=train_data.index)
                    )
                    ohlcv = pd.DataFrame(
                        {
                            "open": open_.values,
                            "high": high.values,
                            "low": low.values,
                            "close": close.values,
                            "volume": volume.values,
                        },
                        index=train_data.index,
                    )
                    trainer.export_feature_index(task["symbol"], ohlcv, output_dir=str(model_path))
                except Exception as e:
                    logger.warning(f"⚠️ PPO feature index export failed for {task['symbol']}: {e}")
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            return {
                "success": True,
                "model_type": model_type,
                "symbol": task["symbol"],
                "model_path": str(model_path),
                "score": result.get("score", 0.0),
                "metrics": result.get("metrics", {}),
                "training_time": result.get("training_time", 0.0),
            }

        except TimeoutError as e:
            return {"success": False, "error": f"Timeout: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Training error: {str(e)}"}
        finally:
            # Clear timeout
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

    def export_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Export trained models as zip archive"""
        self.pipeline_state["current_stage"] = "model_export"
        logger.info("📦 Exporting trained models...")

        if not training_results.get("trained_models"):
            logger.warning("⚠️ No trained models to export")
            return {"success": False, "error": "No models to export"}

        # Create export archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"trading_models_{timestamp}.zip"
        export_path = self.export_dir / export_filename

        try:
            with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                models_added = 0
                manifest_files: List[Dict[str, Any]] = []

                # Add model files
                for model_key, model_info in training_results["trained_models"].items():
                    model_path = Path(model_info["model_path"])

                    if model_path.exists():
                        # Add all files in model directory
                        for file_path in model_path.rglob("*"):
                            if file_path.is_file():
                                arc_path = f"models/{file_path.relative_to(self.models_dir)}"
                                zipf.write(file_path, arc_path)
                                # Compute checksum and metadata for manifest
                                try:
                                    import hashlib
                                    import time as _t

                                    h = hashlib.sha256()
                                    with open(file_path, "rb") as f:
                                        for chunk in iter(lambda: f.read(8192), b""):
                                            h.update(chunk)
                                    manifest_files.append(
                                        {
                                            "path": arc_path,
                                            "sha256": h.hexdigest(),
                                            "size": file_path.stat().st_size,
                                            "mtime": int(file_path.stat().st_mtime),
                                        }
                                    )
                                except Exception:
                                    pass
                        models_added += 1

                # Add metadata
                metadata = {
                    "export_info": {
                        "timestamp": timestamp,
                        "training_start": self.start_time.isoformat(),
                        "export_time": datetime.now().isoformat(),
                        "models_count": models_added,
                        "paperspace_job": os.environ.get("PAPERSPACE_JOB_ID", "local"),
                    },
                    "training_config": self.config,
                    "models": training_results["trained_models"],
                    "training_stats": {
                        "total_trained": training_results["total_trained"],
                        "total_failed": training_results["total_failed"],
                    },
                }

                zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

                # Write manifest.json with checksums and versions
                manifest = {
                    "created_at": datetime.now().isoformat(),
                    "models_count": models_added,
                    "files": manifest_files,
                    "s3_prefix": "model_packages/",
                    "version": timestamp,
                }
                zipf.writestr("manifest.json", json.dumps(manifest, indent=2))

                # Add feature metadata if available
                metadata_dir = self.models_dir / "metadata"
                if metadata_dir.exists():
                    for file_path in metadata_dir.glob("*.json"):
                        zipf.write(file_path, f"metadata/{file_path.name}")

            file_size = export_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✅ Export created: {export_path} ({file_size:.1f} MB)")

            self.pipeline_state["export_path"] = str(export_path)

            return {
                "success": True,
                "export_path": str(export_path),
                "file_size_mb": file_size,
                "models_exported": models_added,
            }

        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return {"success": False, "error": str(e)}

    def upload_to_s3(self, export_result: Dict[str, Any]) -> Dict[str, Any]:
        """Upload model package to S3 storage"""
        if not export_result.get("success") or not export_result.get("export_path"):
            logger.warning("⚠️ No export to upload")
            return {"success": False, "error": "No export available"}

        if not S3_AVAILABLE:
            logger.warning("⚠️ boto3 not available, skipping S3 upload")
            return {"success": False, "error": "boto3 not installed"}

        # Check for S3 configuration
        bucket_name = os.environ.get("AWS_MODELS_BUCKET")
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        if not bucket_name:
            logger.warning("⚠️ AWS_MODELS_BUCKET not configured, skipping S3 upload")
            return {"success": False, "error": "S3 bucket not configured"}

        logger.info(f"☁️ Uploading to S3: {bucket_name}")

        try:
            # Initialize S3 client
            s3_client = boto3.client("s3", region_name=region)

            # Test credentials
            s3_client.list_buckets()

            export_path = Path(export_result["export_path"])
            s3_key = f"model_packages/{export_path.name}"

            # Upload with progress
            file_size = export_path.stat().st_size
            logger.info(f"📤 Uploading {export_path.name} ({file_size / (1024*1024):.1f} MB)...")

            # Upload file
            s3_client.upload_file(
                str(export_path),
                bucket_name,
                s3_key,
                ExtraArgs={
                    "StorageClass": "STANDARD",  # Use Standard initially, lifecycle policy will optimize
                    "Metadata": {
                        "paperspace-job": os.environ.get("PAPERSPACE_JOB_ID", "local"),
                        "created-at": datetime.now().isoformat(),
                        "models-count": str(export_result.get("models_exported", 0)),
                    },
                },
            )

            # Generate presigned download URL (valid for 7 days)
            download_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": s3_key},
                ExpiresIn=7 * 24 * 3600,  # 7 days
            )

            logger.info(f"✅ S3 upload successful: s3://{bucket_name}/{s3_key}")
            logger.info(f"🔗 Download URL generated (expires in 7 days)")

            return {
                "success": True,
                "s3_bucket": bucket_name,
                "s3_key": s3_key,
                "s3_url": f"s3://{bucket_name}/{s3_key}",
                "download_url": download_url,
                "file_size_mb": file_size / (1024 * 1024),
            }

        except NoCredentialsError:
            logger.error("❌ AWS credentials not found")
            logger.error("Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            return {"success": False, "error": "AWS credentials not found"}

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                logger.error(f"❌ S3 bucket '{bucket_name}' does not exist")
                logger.error("Run setup_s3_storage.py to create the bucket")
            else:
                logger.error(f"❌ S3 upload failed: {e}")
            return {"success": False, "error": str(e)}

        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")
            return {"success": False, "error": str(e)}

    def _send_success_notification(self, result: Dict[str, Any]) -> None:
        """Send Telegram notification for successful training completion"""
        if not self.telegram_notifier:
            return

        try:
            # Extract key information from result
            training_result = result.get("training", {})
            export_result = result.get("export", {})
            s3_result = result.get("s3_upload", {})
            runtime_hours = result.get("runtime_hours", 0)

            models_trained = []
            for model_type, model_results in training_result.get("model_results", {}).items():
                successful_symbols = [s for s, r in model_results.items() if r.get("success")]
                if successful_symbols:
                    models_trained.append(
                        f"{model_type.upper()}: {len(successful_symbols)} symbols"
                    )

            message = f"""
🎉 <b>MODEL TRAINING COMPLETED</b>

<b>Runtime:</b> {runtime_hours:.1f} hours
<b>Models Trained:</b>
{chr(10).join(f'• {m}' for m in models_trained) if models_trained else '• No models completed'}

<b>Export Status:</b> {'✅ Success' if export_result.get('success') else '❌ Failed'}
<b>S3 Upload:</b> {'✅ Success' if s3_result.get('success') else '❌ Failed'}

🤖 <b>Ready for deployment!</b>
Type <code>/import</code> to import models in the cryptobot.

<i>Training Server • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""

            self.telegram_notifier.send_message_sync(message)
            logger.info("📱 Success notification sent to Telegram")

        except Exception as e:
            logger.warning(f"📱 Failed to send success notification: {e}")

    def _send_failure_notification(self, error: str, pipeline_state: Dict[str, Any]) -> None:
        """Send Telegram notification for training failure"""
        if not self.telegram_notifier:
            return

        try:
            current_stage = pipeline_state.get("current_stage", "unknown")
            runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            errors = pipeline_state.get("errors", [])

            message = f"""
🚨 <b>MODEL TRAINING FAILED</b>

<b>Stage:</b> {current_stage}
<b>Runtime:</b> {runtime_hours:.1f} hours
<b>Error:</b> {error}

<b>Pipeline Errors:</b>
{chr(10).join(f'• {e}' for e in errors[-3:]) if errors else '• No specific errors logged'}

Please check the training logs for more details.

<i>Training Server • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""

            self.telegram_notifier.send_message_sync(message)
            logger.info("📱 Failure notification sent to Telegram")

        except Exception as e:
            logger.warning(f"📱 Failed to send failure notification: {e}")

    def run_full_pipeline(
        self,
        model_types: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """Execute the complete training pipeline"""
        logger.info("🚀 Starting full training pipeline")
        logger.info("=" * 60)

        try:
            # Stage 1: Verify data
            if not self.should_continue(0.1):
                raise TimeoutError("Insufficient time for pipeline")

            data_stats = self.verify_data_availability()

            # Stage 2: Prepare datasets
            if not self.should_continue(0.5):
                raise TimeoutError("Insufficient time for data preparation")

            datasets_result = self.prepare_datasets(symbols)

            # Stage 3: Train models
            if not self.should_continue(1.0):
                raise TimeoutError("Insufficient time for model training")

            training_result = self.train_models(datasets_result, model_types, fast_mode)

            # Stage 4: Export models
            if not self.should_continue(0.3):
                logger.warning("⚠️ Skipping export due to time constraints")
                export_result = {"success": False, "error": "Timeout"}
                s3_result = {"success": False, "error": "Export skipped"}
            else:
                export_result = self.export_models(training_result)

                # Stage 5: Upload to S3 (if configured and not disabled)
                if (
                    export_result["success"]
                    and self.should_continue(0.2)
                    and not getattr(self, "skip_s3", False)
                ):
                    s3_result = self.upload_to_s3(export_result)
                else:
                    s3_result = {"success": False, "error": "Skipped or insufficient time"}

            # Final summary
            elapsed = (datetime.now() - self.start_time).total_seconds() / 3600

            logger.info("\n" + "=" * 60)
            logger.info("🎉 PIPELINE COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total time: {elapsed:.1f} hours")
            logger.info(f"📊 Datasets: {len(datasets_result.get('datasets', {}))}")
            logger.info(f"🎯 Models trained: {training_result.get('total_trained', 0)}")
            logger.info(f"❌ Models failed: {training_result.get('total_failed', 0)}")

            if export_result["success"]:
                logger.info(f"📦 Export: {export_result['export_path']}")
                logger.info(f"💾 Size: {export_result['file_size_mb']:.1f} MB")

            if s3_result["success"]:
                logger.info(f"☁️ S3 Upload: {s3_result['s3_url']}")
                logger.info(f"🔗 Download URL: Available for 7 days")

            return {
                "success": True,
                "data_stats": data_stats,
                "datasets": datasets_result,
                "training": training_result,
                "export": export_result,
                "s3_upload": s3_result,
                "runtime_hours": elapsed,
                "pipeline_state": self.pipeline_state,
            }

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {"success": False, "error": str(e), "pipeline_state": self.pipeline_state}


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Paperspace Training Script")
    parser.add_argument(
        "--config", default="training_config.yaml", help="Training configuration file"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gru", "lightgbm", "ppo"],
        help="Models to train (default: all)",
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to train (default: all from config)")
    parser.add_argument("--max-hours", type=float, default=5.5, help="Maximum runtime hours")
    parser.add_argument("--fast", action="store_true", help="Fast training mode")
    parser.add_argument("--dry-run", action="store_true", help="Verify setup without training")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 upload even if configured")

    args = parser.parse_args()

    try:
        trainer = PaperspaceTraining(config_path=args.config, max_hours=args.max_hours)
        trainer.skip_s3 = args.no_s3

        if args.dry_run:
            logger.info("🔍 DRY RUN MODE - Verifying setup...")
            data_stats = trainer.verify_data_availability()
            datasets_result = trainer.prepare_datasets(args.symbols)
            logger.info("✅ Setup verification complete")
            logger.info(f"📊 Available data: {sum(data_stats.values()):,} samples")
            logger.info(f"📊 Prepared datasets: {len(datasets_result.get('datasets', {}))}")
            return

        # Run full pipeline
        result = trainer.run_full_pipeline(
            model_types=args.models, symbols=args.symbols, fast_mode=args.fast
        )

        if result["success"]:
            logger.info("🎉 Training completed successfully!")
            if result.get("export", {}).get("success"):
                logger.info(f"📦 Models exported to: {result['export']['export_path']}")

            # Send success notification
            trainer._send_success_notification(result)
            sys.exit(0)
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"❌ Training failed: {error_msg}")

            # Send failure notification
            trainer._send_failure_notification(error_msg, result.get("pipeline_state", {}))
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()

        # Send failure notification for unexpected errors
        try:
            trainer._send_failure_notification(str(e), trainer.pipeline_state)
        except:
            pass  # Don't fail on notification failure

        sys.exit(1)


if __name__ == "__main__":
    main()
