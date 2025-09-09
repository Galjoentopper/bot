#!/usr/bin/env python3
"""
Clean Data Cache
===============

Remove old datasets and cache files to force fresh data fetching.
"""

import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_data_directories():
    """Clean all data-related directories"""
    
    logger.info("🧹 Cleaning data directories...")
    
    # Directories to clean
    data_dirs = [
        "data",
        "models/metadata", 
        "models/cache",
        "/notebooks/data",
        "/notebooks/models/metadata",
        "/notebooks/models/cache",
        "./data/cache"
    ]
    
    files_removed = 0
    dirs_removed = 0
    
    for data_dir in data_dirs:
        try:
            data_path = Path(data_dir)
            if data_path.exists():
                logger.info(f"📁 Cleaning {data_path}...")
                
                # Remove all files in directory
                for item in data_path.rglob("*"):
                    if item.is_file():
                        try:
                            item.unlink()
                            files_removed += 1
                            if files_removed % 10 == 0:
                                logger.info(f"  Removed {files_removed} files...")
                        except Exception as e:
                            logger.warning(f"  Failed to remove {item}: {e}")
                
                # Remove empty subdirectories
                for item in data_path.rglob("*"):
                    if item.is_dir() and not any(item.iterdir()):
                        try:
                            item.rmdir()
                            dirs_removed += 1
                        except Exception as e:
                            logger.warning(f"  Failed to remove {item}: {e}")
                            
                logger.info(f"✅ Cleaned {data_path}")
            else:
                logger.info(f"⚠️ Directory doesn't exist: {data_path}")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning {data_dir}: {e}")
    
    logger.info(f"🎯 Cleanup complete: {files_removed} files, {dirs_removed} directories removed")


def clean_cache_files():
    """Remove specific cache file patterns"""
    
    logger.info("🗑️ Cleaning cache files...")
    
    cache_patterns = [
        "*.parquet",
        "*.pkl", 
        "*.json",
        "*_cache.csv",
        "*_metadata.json",
        "features_*.json"
    ]
    
    search_dirs = [".", "data", "models", "/notebooks", "/notebooks/bot"]
    
    for search_dir in search_dirs:
        try:
            search_path = Path(search_dir)
            if search_path.exists():
                for pattern in cache_patterns:
                    for cache_file in search_path.rglob(pattern):
                        try:
                            if cache_file.is_file():
                                cache_file.unlink()
                                logger.info(f"  Removed: {cache_file}")
                        except Exception as e:
                            logger.warning(f"  Failed to remove {cache_file}: {e}")
        except Exception as e:
            logger.warning(f"Error searching {search_dir}: {e}")


def create_fresh_directories():
    """Create fresh data directories"""
    
    logger.info("📁 Creating fresh directories...")
    
    fresh_dirs = [
        "data",
        "data/cache", 
        "models",
        "models/metadata",
        "logs"
    ]
    
    for fresh_dir in fresh_dirs:
        try:
            fresh_path = Path(fresh_dir)
            fresh_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created: {fresh_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create {fresh_dir}: {e}")


def main():
    """Clean everything and prepare for fresh data"""
    
    logger.info("🚀 Data Cleanup and Reset")
    logger.info("=" * 50)
    
    try:
        clean_data_directories()
        logger.info("-" * 30)
        clean_cache_files()
        logger.info("-" * 30) 
        create_fresh_directories()
        
        logger.info("✅ Data cleanup complete!")
        logger.info("🎯 Ready for fresh data fetching")
        logger.info("📋 Next step: python paperspace_training_orchestrator.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())