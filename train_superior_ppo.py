#!/usr/bin/env python3
"""
Superior PPO Training Script
===========================

This script implements the superior multi-timeframe PPO training approach
that restores the profitable architecture from the old BTCEUR model while
solving the resource management issues.

Usage:
    python train_superior_ppo.py --symbol BTCEUR --timesteps 200000
    python train_superior_ppo.py --symbol BTCEUR --timesteps 200000 --resume
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Fix import paths
from src.models.resource_aware_ppo_trainer import ResourceAwarePPOTrainer
from src.data_pipeline.superior_ppo_feature_expander import SuperiorPPOFeatureExpander

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/superior_training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "superior_training_config.yaml") -> dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"✅ Configuration loaded from {config_path}")
    return config


def load_training_data(symbol: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare training data."""
    logger.info(f"🔄 Loading training data for {symbol}")

    # This is a placeholder - replace with your actual data loading logic
    data_path = f"data/{symbol}_30m.parquet"  # Adjust path as needed

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Load data
    df = pd.read_parquet(data_path)
    logger.info(f"📊 Loaded {len(df)} rows of data for {symbol}")

    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Split into train/eval
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx].copy()
    eval_data = df.iloc[split_idx:].copy()

    logger.info(f"📊 Train: {len(train_data)} rows, Eval: {len(eval_data)} rows")
    return train_data, eval_data


def train_superior_ppo(
    symbol: str,
    total_timesteps: int = 200000,
    config_path: str = "superior_training_config.yaml",
    resume_from: str = None
):
    """Train superior PPO model with multi-timeframe features."""

    logger.info("🚀 Starting Superior PPO Training")
    logger.info(f"   Symbol: {symbol}")
    logger.info(f"   Target timesteps: {total_timesteps:,}")
    logger.info(f"   Feature engineering: Superior multi-timeframe")

    # Load configuration
    config = load_config(config_path)

    # Load training data
    train_data, eval_data = load_training_data(symbol, config)

    # Initialize resource-aware PPO trainer
    trainer = ResourceAwarePPOTrainer(config)

    # Load existing model if resuming
    if resume_from and os.path.exists(resume_from):
        logger.info(f"🔄 Resuming training from {resume_from}")
        trainer.load_pretrained_weights(resume_from)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/superior/{symbol}"
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"{output_dir}/superior_ppo_{timestamp}"

    try:
        # Train the model
        results = trainer.train(
            train_data=train_data,
            eval_data=eval_data,
            total_timesteps=total_timesteps,
            experiment_name=f"superior_ppo_{symbol}",
            save_path=model_path
        )

        # Log results
        logger.info("✅ Superior PPO Training completed successfully!")
        logger.info(f"   Trained timesteps: {results.get('trained_timesteps', 0):,}")
        logger.info(f"   Features used: {results.get('feature_count', 0)} superior multi-timeframe")
        logger.info(f"   Model saved: {model_path}")

        # Compare with old model approach
        logger.info("\n🎯 ARCHITECTURAL COMPARISON:")
        logger.info("   OLD MODEL (Superior): Multi-timeframe targets")
        logger.info("     - return_1h, cost_adj_return_1h, profitable_1h")
        logger.info("     - direction_1h, confidence_1h, regime_direction_1h")
        logger.info("     - Forward-looking: 'What will happen?'")
        logger.info("   ")
        logger.info("   NEW MODEL (Restored): Same superior approach")
        logger.info("     - Same multi-timeframe target engineering")
        logger.info("     - Resource-aware training prevents OOM")
        logger.info("     - Progressive training stages")

        return results

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        # Cleanup
        trainer.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Superior PPO Model')
    parser.add_argument('--symbol', type=str, default='BTCEUR',
                       help='Trading symbol to train (default: BTCEUR)')
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Total training timesteps (default: 200000)')
    parser.add_argument('--config', type=str, default='superior_training_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to model to resume training from')
    parser.add_argument('--demo', action='store_true',
                       help='Run with reduced timesteps for demo')

    args = parser.parse_args()

    # Demo mode
    if args.demo:
        args.timesteps = 50000
        logger.info("🎮 Demo mode: Reduced to 50k timesteps")

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    try:
        # Train the model
        results = train_superior_ppo(
            symbol=args.symbol,
            total_timesteps=args.timesteps,
            config_path=args.config,
            resume_from=args.resume
        )

        logger.info("🎉 Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())