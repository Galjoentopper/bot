#!/usr/bin/env python3
"""
Paperspace Gradient Notebook Setup Script
=========================================

This script sets up the complete trading bot training environment
on a fresh Paperspace Gradient notebook. Run this first when starting
a new training session.

Usage:
    python paperspace_setup.py

Environment Variables Required:
    TRADING_BOT_REPO_URL: Git repository URL (optional)
    PRODUCTION_SERVER_URL: Production server endpoint for model transfer
    PRODUCTION_API_KEY: API key for production server
    TELEGRAM_BOT_TOKEN: Telegram bot token for notifications
    TELEGRAM_CHAT_ID: Telegram chat ID for notifications
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, check=True, shell=True):
    """Run a shell command with logging"""
    logger.info(f"🔧 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"✅ Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {e}")
        if e.stderr:
            logger.error(f"❌ Error: {e.stderr.strip()}")
        if check:
            raise
        return e


def setup_environment():
    """Setup the Paperspace environment"""
    logger.info("🚀 Setting up Paperspace environment...")

    # Check if we're in Paperspace
    is_paperspace = (
        os.environ.get("PAPERSPACE_JOB_ID") is not None
        or os.environ.get("GRADIENT_JOB_ID") is not None
        or Path("/notebooks").exists()
    )

    if is_paperspace:
        logger.info("✅ Paperspace environment detected")
        workspace_dir = Path("/notebooks")
    else:
        logger.info("⚠️ Not in Paperspace - using local directory")
        workspace_dir = Path(".")

    os.chdir(workspace_dir)

    # Update system packages
    logger.info("📦 Updating system packages...")
    run_command("apt-get update", check=False)
    run_command("apt-get install -y git wget curl unzip", check=False)

    return workspace_dir


def setup_repository(workspace_dir):
    """Setup the trading bot repository"""
    logger.info("📁 Setting up repository...")

    repo_url = os.environ.get("TRADING_BOT_REPO_URL")

    if repo_url:
        # Clone from repository
        logger.info(f"📥 Cloning repository: {repo_url}")
        run_command(f"git clone {repo_url} trading_bot")
        os.chdir(workspace_dir / "trading_bot")
    else:
        # Repository should be pre-uploaded to Paperspace
        logger.info("📦 Repository should be pre-uploaded to Paperspace")

        # Look for existing trading bot directory
        possible_dirs = ["trading_bot", "bot", "."]
        trading_dir = None

        for dir_name in possible_dirs:
            check_dir = workspace_dir / dir_name
            if (check_dir / "src").exists() and (check_dir / "training_config.yaml").exists():
                trading_dir = check_dir
                break

        if trading_dir:
            logger.info(f"✅ Found trading bot directory: {trading_dir}")
            os.chdir(trading_dir)
        else:
            logger.error("❌ Trading bot repository not found!")
            logger.error("Please either:")
            logger.error("1. Set TRADING_BOT_REPO_URL environment variable")
            logger.error("2. Upload the repository to /notebooks/trading_bot")
            sys.exit(1)


def install_dependencies():
    """Install Python dependencies"""
    logger.info("🐍 Installing Python dependencies...")

    # Upgrade pip
    run_command("pip install --upgrade pip")

    # Install requirements if available
    if Path("requirements.txt").exists():
        logger.info("📋 Installing from requirements.txt")
        run_command("pip install -r requirements.txt")
    else:
        # Install essential packages manually
        logger.info("📦 Installing essential packages")
        essential_packages = [
            "torch>=1.9.0",
            "lightgbm>=3.3.0",
            "stable-baselines3>=1.5.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "yfinance>=0.1.70",
            "python-binance>=1.0.15",
            "mlflow>=1.25.0",
            "optuna>=2.10.0",
            "telegram-bot>=13.0",
            "pyyaml>=6.0",
            "psutil>=5.8.0",
            "requests>=2.26.0",
            "boto3>=1.20.0",  # For AWS S3 if needed
        ]

        for package in essential_packages:
            run_command(f"pip install {package}", check=False)

    # Install package in editable mode if setup.py exists
    if Path("setup.py").exists():
        logger.info("🔧 Installing package in editable mode")
        run_command("pip install -e .")


def setup_directories():
    """Create necessary directories"""
    logger.info("📁 Creating directories...")

    directories = ["data", "models", "logs", "exports", "reports", "temp"]

    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {dir_name}")


def configure_environment():
    """Configure environment variables and settings"""
    logger.info("⚙️ Configuring environment...")

    # Set MLflow tracking
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir.absolute()}"

    # Configure for Paperspace
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available
    os.environ["PYTHONPATH"] = str(Path.cwd())

    # Set memory limits for training
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    logger.info("✅ Environment configured")


def verify_setup():
    """Verify the setup is working"""
    logger.info("🔍 Verifying setup...")

    # Check Python packages
    required_imports = [
        "torch",
        "lightgbm",
        "stable_baselines3",
        "pandas",
        "numpy",
        "sklearn",
        "yfinance",
        "binance",
        "mlflow",
    ]

    missing_packages = []
    for package in required_imports:
        try:
            __import__(package)
            logger.info(f"✅ {package} imported successfully")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"⚠️ {package} import failed")

    # Check key files
    required_files = [
        "training_config.yaml",
        "src/data_pipeline/dataset_builder.py",
        "scripts/enhanced_trainer.py",
        "paperspace_mlops/paperspace_training_orchestrator.py",
    ]

    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"✅ {file_path} found")
        else:
            missing_files.append(file_path)
            logger.warning(f"⚠️ {file_path} missing")

    # Check GPU availability
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"✅ GPU available: {torch.cuda.get_device_name()}")
            logger.info(
                f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            logger.warning("⚠️ No GPU available - training will use CPU")
    except:
        logger.warning("⚠️ Could not check GPU status")

    if missing_packages or missing_files:
        logger.error("❌ Setup verification failed")
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
        return False

    logger.info("✅ Setup verification passed!")
    return True


def create_quick_start_script():
    """Create a quick start script for training"""
    logger.info("📝 Creating quick start script...")

    script_content = """#!/usr/bin/env python3
\"\"\"
Quick Start Script for Paperspace Training
\"\"\"

import os
import sys
from pathlib import Path

# Ensure we're in the right directory
if not Path('paperspace_mlops/paperspace_training_orchestrator.py').exists():
    print("❌ Please run this from the trading bot root directory")
    sys.exit(1)

# Add to Python path
sys.path.append(str(Path.cwd()))

# Import and run orchestrator
from paperspace_mlops.paperspace_training_orchestrator import PaperspaceOrchestrator

def main():
    print("🚀 Starting Paperspace Training Pipeline")
    print("=" * 50)

    # Initialize orchestrator
    orchestrator = PaperspaceOrchestrator(config_path='training_config.yaml')

    # Run full pipeline
    result = orchestrator.run_full_pipeline()

    if result['success']:
        print("🎉 Training pipeline completed successfully!")
        print(f"⏱️ Total runtime: {result['runtime_hours']:.2f} hours")
    else:
        print(f"❌ Training pipeline failed: {result.get('error', 'Unknown error')}")

    return 0 if result['success'] else 1

if __name__ == "__main__":
    exit(main())
"""

    with open("start_training.py", "w") as f:
        f.write(script_content)

    # Make executable
    run_command("chmod +x start_training.py")
    logger.info("✅ Created start_training.py")


def display_instructions():
    """Display setup completion instructions"""
    logger.info("🎯 Setup Complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("To start training, run:")
    logger.info("  python start_training.py")
    logger.info("")
    logger.info("Or run the orchestrator directly:")
    logger.info("  python paperspace_mlops/paperspace_training_orchestrator.py")
    logger.info("")
    logger.info("Environment variables you may want to set:")
    logger.info("  PRODUCTION_SERVER_URL: For model transfer")
    logger.info("  PRODUCTION_API_KEY: API key for production server")
    logger.info("  TELEGRAM_BOT_TOKEN: For notifications")
    logger.info("  TELEGRAM_CHAT_ID: Your Telegram chat ID")
    logger.info("")
    logger.info("Monitor progress in:")
    logger.info("  logs/paperspace_training_*.log")
    logger.info("  logs/pipeline_state.json")
    logger.info("")


def main():
    """Main setup function"""
    try:
        logger.info("🚀 Paperspace Trading Bot Setup Starting...")
        logger.info("=" * 60)

        # Setup steps
        workspace_dir = setup_environment()
        setup_repository(workspace_dir)
        install_dependencies()
        setup_directories()
        configure_environment()

        # Verification
        if not verify_setup():
            logger.error("❌ Setup verification failed")
            sys.exit(1)

        # Final setup
        create_quick_start_script()
        display_instructions()

        logger.info("🎉 Setup completed successfully!")

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
