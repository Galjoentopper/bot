#!/usr/bin/env python3
"""
Cross-Platform ML Runtime Initialization
=======================================
Automatically creates missing ML runtime files before training startup.
Prevents "Could not find experiment with ID 0" errors on Linux and Windows.
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.mlflow_init import MLflowInitializer
except ImportError:
    print("Warning: MLflowInitializer not found. Creating basic directory structure.")
    MLflowInitializer = None

# Configure logging to be silent by default
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create basic directory structure for ML operations."""
    directories = [
        'mlruns',
        'logs',
        'models',
        'data',
        'checkpoints'
    ]
    
    created_dirs = []
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
            except Exception as e:
                logger.warning(f"Failed to create directory {dir_path}: {e}")
    
    return created_dirs

def initialize_mlflow():
    """Initialize MLflow with required experiments."""
    if MLflowInitializer is None:
        # Fallback: create basic mlruns structure manually
        mlruns_path = Path('./mlruns')
        mlruns_path.mkdir(exist_ok=True)
        
        # Create experiment 0 directory and meta.yaml
        exp_0_dir = mlruns_path / '0'
        exp_0_dir.mkdir(exist_ok=True)
        
        meta_file = exp_0_dir / 'meta.yaml'
        if not meta_file.exists():
            meta_content = """artifact_location: file:///{}/0
creation_time: 1640995200000
experiment_id: '0'
lifecycle_stage: active
name: Default
""".format(str(mlruns_path.resolve()).replace('\\', '/'))
            
            try:
                with open(meta_file, 'w') as f:
                    f.write(meta_content)
                return True
            except Exception as e:
                logger.warning(f"Failed to create meta.yaml: {e}")
                return False
        return True
    
    # Use MLflowInitializer if available
    try:
        initializer = MLflowInitializer('./mlruns')
        
        # Ensure default experiment (ID 0) exists
        if not initializer.ensure_default_experiment():
            logger.warning("Failed to create default experiment")
            return False
        
        # Create main trading experiment (ID 1)
        experiments = {"crypto_trading_bot": "1"}
        if not initializer.initialize_mlflow_tracking(experiments):
            logger.warning("Failed to initialize MLflow tracking")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"MLflow initialization failed: {e}")
        return False

def verify_checkpoint_structure():
    """Verify and create checkpoint directory structure."""
    checkpoint_dirs = [
        'checkpoints',
        'checkpoints/gru',
        'checkpoints/lightgbm', 
        'checkpoints/ppo'
    ]
    
    created_dirs = []
    for dir_name in checkpoint_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
            except Exception as e:
                logger.warning(f"Failed to create checkpoint directory {dir_path}: {e}")
    
    return created_dirs

def check_initialization_needed():
    """
    Check if ML runtime initialization is needed
    Returns True if any required directories or files are missing
    """
    required_dirs = ['mlruns', 'logs', 'models', 'data', 'checkpoints']
    
    # Check if any required directory is missing
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            return True
    
    # Check if MLflow experiment 0 exists
    experiment_0_path = os.path.join('mlruns', '0', 'meta.yaml')
    if not os.path.exists(experiment_0_path):
        return True
    
    return False

def initialize_runtime(verbose=False):
    """Main initialization function."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Starting ML runtime initialization...")
    
    success = True
    
    # Create basic directory structure
    created_dirs = create_directory_structure()
    if verbose and created_dirs:
        logger.info(f"Created directories: {', '.join(created_dirs)}")
    
    # Initialize MLflow
    if not initialize_mlflow():
        logger.error("MLflow initialization failed")
        success = False
    elif verbose:
        logger.info("MLflow initialized successfully")
    
    # Verify checkpoint structure
    checkpoint_dirs = verify_checkpoint_structure()
    if verbose and checkpoint_dirs:
        logger.info(f"Created checkpoint directories: {', '.join(checkpoint_dirs)}")
    
    return success

def main():
    """Command line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize ML runtime environment')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--check', '-c', action='store_true',
                       help='Check if initialization is needed')
    
    args = parser.parse_args()
    
    if args.check:
        # Check if mlruns/0/meta.yaml exists
        meta_file = Path('./mlruns/0/meta.yaml')
        if meta_file.exists():
            if args.verbose:
                print("ML runtime already initialized")
            sys.exit(0)
        else:
            if args.verbose:
                print("ML runtime initialization needed")
            sys.exit(1)
    
    # Run initialization
    success = initialize_runtime(verbose=args.verbose)
    
    if success:
        if args.verbose:
            print("ML runtime initialization completed successfully")
        sys.exit(0)
    else:
        if args.verbose:
            print("ML runtime initialization failed")
        sys.exit(1)

if __name__ == '__main__':
    main()