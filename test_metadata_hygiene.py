#!/usr/bin/env python3
"""
Test script for metadata hygiene functionality.

This script demonstrates the automated metadata regeneration and hygiene processes
implemented in Phase 4 of the strat.txt plan.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from scripts.enhanced_trader import EnhancedUnifiedPaperTrader
from src.utils.logger import Logger

def test_metadata_hygiene():
    """Test the metadata hygiene functionality."""
    logger = Logger(name='metadata_test')
    
    try:
        logger.info("Initializing Enhanced Trader for metadata hygiene test...")
        
        # Initialize trader with default config
        trader = EnhancedUnifiedPaperTrader()
        
        logger.info("Running metadata hygiene processes...")
        
        # Run metadata hygiene
        trader.run_metadata_hygiene()
        
        logger.info("Metadata hygiene test completed successfully!")
        
        # Get validation report
        if hasattr(trader, 'validation_manager'):
            report = trader.validation_manager.get_validation_report()
            logger.info(f"Validation report summary: {len(report.get('drift_events', []))} drift events, "
                       f"{len(report.get('schema_decisions', []))} schema decisions")
        
    except Exception as e:
        logger.error(f"Metadata hygiene test failed: {e}")
        raise

if __name__ == "__main__":
    test_metadata_hygiene()