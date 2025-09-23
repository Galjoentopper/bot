#!/usr/bin/env python3
"""
Normalization Statistics Regeneration Script
============================================

This script regenerates clean normalization statistics for all models
to fix the "corrupted statistics" warnings in the trading system.

Usage:
    python scripts/regenerate_normalization_stats.py [--dry-run] [--models model1,model2]
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from src.config.config_loader import ConfigLoader
from src.config.secure_env_manager import get_env_manager
from src.data_pipeline.data_fetcher import CryptoDataFetcher
from src.data_pipeline.features import FeatureEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NormalizationStatsRegenerator:
    """Regenerates clean normalization statistics for all models."""

    def __init__(self, dry_run: bool = False):
        """Initialize the regenerator."""
        self.dry_run = dry_run
        self.project_root = project_root
        self.backup_dir = (
            self.project_root
            / "data"
            / "stats_backup"
            / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Load configuration
        try:
            self.config = ConfigLoader().load_config()
            self.env_manager = get_env_manager()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

        # Initialize data fetcher and feature engine
        self.data_fetcher = CryptoDataFetcher(self.config)
        self.feature_engine = FeatureEngine(self.config)

        # Trading symbols to process
        self.symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]

        logger.info(f"NormalizationStatsRegenerator initialized {'(DRY RUN)' if dry_run else ''}")
        logger.info(f"Processing symbols: {', '.join(self.symbols)}")

    def backup_existing_stats(self):
        """Backup existing normalization statistics."""
        logger.info("Backing up existing normalization statistics...")

        stats_files_found = 0

        # Create backup directory
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Look for normalization statistics files
        for stats_pattern in [
            "**/*vecnormalize*",
            "**/normalization_stats.json",
            "**/scaler_*.pkl",
            "**/preprocessor.pkl",
        ]:
            for stats_file in self.project_root.glob(stats_pattern):
                if stats_file.is_file():
                    stats_files_found += 1
                    relative_path = stats_file.relative_to(self.project_root)
                    backup_path = self.backup_dir / relative_path

                    logger.info(f"Backing up: {relative_path}")

                    if not self.dry_run:
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(stats_file, backup_path)

        logger.info(
            f"Backup completed: {stats_files_found} files {'would be' if self.dry_run else ''} backed up"
        )
        if not self.dry_run:
            logger.info(f"Backup location: {self.backup_dir}")

    def fetch_clean_data(self, symbol: str, limit: int = 2000) -> pd.DataFrame:
        """Fetch clean market data for a symbol."""
        logger.info(f"Fetching clean data for {symbol} (limit: {limit})")

        try:
            # Fetch data using the existing data fetcher
            data = self.data_fetcher.fetch_ohlcv(symbol, interval="30m", limit=limit)

            if data.empty:
                raise ValueError(f"No data fetched for {symbol}")

            # Basic data validation and cleaning
            data = self._clean_market_data(data)

            logger.info(f"Successfully fetched {len(data)} clean records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise

    def _clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean market data with conservative bounds."""
        logger.debug("Cleaning market data with conservative bounds")

        # Remove any rows with NaN or infinite values
        data = data.dropna()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        # Apply conservative bounds for crypto prices
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col in data.columns:
                # Conservative bounds: $0.01 to $200,000
                data[col] = data[col].clip(lower=0.01, upper=200000)

        # Clean volume data
        if "volume" in data.columns:
            # Volume should be positive
            data["volume"] = data["volume"].clip(lower=0, upper=1e12)  # Max 1T volume

        # Remove any duplicate timestamps
        if "timestamp" in data.columns:
            data = data.drop_duplicates(subset=["timestamp"])

        # Sort by timestamp
        if "timestamp" in data.columns:
            data = data.sort_values("timestamp")

        logger.debug(f"Cleaned data shape: {data.shape}")
        return data

    def generate_clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate clean features using the feature engine."""
        logger.debug("Generating clean features")

        try:
            # Use feature engine to generate features
            features_df = self.feature_engine.generate_all_features(data)

            # Additional cleaning for generated features
            features_df = self._clean_generated_features(features_df)

            logger.debug(f"Generated features shape: {features_df.shape}")
            return features_df

        except Exception as e:
            logger.error(f"Failed to generate features: {e}")
            raise

    def _clean_generated_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean generated features."""
        # Replace infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill NaN values
        features_df = features_df.fillna(method="ffill").fillna(method="bfill")

        # Fill any remaining NaN with 0
        features_df = features_df.fillna(0)

        # Clip extreme values for non-OHLCV columns
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = [col for col in features_df.columns if col not in ohlcv_cols]

        for col in feature_cols:
            if col in features_df.columns:
                # Use 99th percentile bounds for features
                percentile_99 = features_df[col].quantile(0.99)
                percentile_1 = features_df[col].quantile(0.01)
                if not pd.isna(percentile_99) and not pd.isna(percentile_1):
                    features_df[col] = features_df[col].clip(
                        lower=percentile_1, upper=percentile_99
                    )

        return features_df

    def calculate_clean_statistics(self, features_df: pd.DataFrame) -> dict:
        """Calculate clean normalization statistics."""
        logger.debug("Calculating clean normalization statistics")

        # Get numeric columns only
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns

        statistics = {}

        for col in numeric_cols:
            col_data = features_df[col]

            # Remove any remaining outliers using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            if not pd.isna(IQR) and IQR > 1e-10:
                # Clean data using IQR bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                clean_data = col_data.clip(lower=lower_bound, upper=upper_bound)
            else:
                clean_data = col_data

            # Calculate statistics
            statistics[col] = {
                "mean": float(clean_data.mean()),
                "std": float(clean_data.std()),
                "min": float(clean_data.min()),
                "max": float(clean_data.max()),
                "q25": float(clean_data.quantile(0.25)),
                "q75": float(clean_data.quantile(0.75)),
                "count": int(len(clean_data)),
                "created_at": datetime.now().isoformat(),
            }

        logger.debug(f"Calculated statistics for {len(statistics)} features")
        return statistics

    def save_clean_statistics(self, symbol: str, statistics: dict):
        """Save clean statistics to appropriate locations."""
        logger.info(f"Saving clean statistics for {symbol}")

        # Create statistics directory
        stats_dir = self.project_root / "data" / "normalization_stats"
        if not self.dry_run:
            stats_dir.mkdir(parents=True, exist_ok=True)

        # Save global statistics file
        global_stats_file = stats_dir / f"{symbol}_normalization_stats.json"

        if not self.dry_run:
            with open(global_stats_file, "w") as f:
                json.dump(statistics, f, indent=2)

        logger.info(f"Saved statistics to: {global_stats_file}")

        # Also save to model-specific locations if they exist
        model_dirs = list((self.project_root / "models").glob(f"*/{symbol}"))
        for model_dir in model_dirs:
            model_stats_file = model_dir / "normalization_stats.json"
            if not self.dry_run:
                with open(model_stats_file, "w") as f:
                    json.dump(statistics, f, indent=2)
            logger.info(f"Saved model-specific statistics to: {model_stats_file}")

    def regenerate_for_symbol(self, symbol: str):
        """Regenerate normalization statistics for a specific symbol."""
        logger.info(f"=== Processing {symbol} ===")

        try:
            # Fetch clean data
            clean_data = self.fetch_clean_data(symbol)

            # Generate clean features
            features_df = self.generate_clean_features(clean_data)

            # Calculate clean statistics
            statistics = self.calculate_clean_statistics(features_df)

            # Save statistics
            self.save_clean_statistics(symbol, statistics)

            logger.info(f"✅ Successfully regenerated statistics for {symbol}")

        except Exception as e:
            logger.error(f"❌ Failed to regenerate statistics for {symbol}: {e}")
            raise

    def run(self, symbols_filter: list = None):
        """Run the normalization statistics regeneration."""
        symbols_to_process = symbols_filter if symbols_filter else self.symbols

        logger.info("🚀 Starting normalization statistics regeneration")
        logger.info(f"Processing symbols: {', '.join(symbols_to_process)}")

        if self.dry_run:
            logger.info("⚠️ DRY RUN MODE - No files will be modified")

        try:
            # Backup existing statistics
            self.backup_existing_stats()

            # Process each symbol
            successful_symbols = []
            failed_symbols = []

            for symbol in symbols_to_process:
                try:
                    self.regenerate_for_symbol(symbol)
                    successful_symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Failed processing {symbol}: {e}")
                    failed_symbols.append(symbol)
                    continue

            # Summary
            logger.info("📊 REGENERATION SUMMARY")
            logger.info(f"✅ Successful: {len(successful_symbols)} symbols")
            if successful_symbols:
                logger.info(f"   {', '.join(successful_symbols)}")

            if failed_symbols:
                logger.error(f"❌ Failed: {len(failed_symbols)} symbols")
                logger.error(f"   {', '.join(failed_symbols)}")

            if successful_symbols:
                logger.info("🎉 Normalization statistics regeneration completed successfully!")
                if not self.dry_run:
                    logger.info("💡 Restart the trading system to use the new clean statistics")
            else:
                logger.error("😞 No symbols were processed successfully")
                return False

            return True

        except Exception as e:
            logger.error(f"Critical error during regeneration: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Regenerate clean normalization statistics")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of symbols to process (default: all)",
    )

    args = parser.parse_args()

    # Parse symbols filter
    symbols_filter = None
    if args.models:
        symbols_filter = [s.strip().upper() for s in args.models.split(",")]

    try:
        regenerator = NormalizationStatsRegenerator(dry_run=args.dry_run)
        success = regenerator.run(symbols_filter)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
