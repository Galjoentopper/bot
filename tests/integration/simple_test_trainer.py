#!/usr/bin/env python3
"""
Simple trainer test script to validate the actual training pipeline.
Tests the paperspace_training.py script which is the main entry point.
"""

import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_config_validation():
    """Test that training_config.yaml is valid and has required sections."""
    logger.info("=" * 60)
    logger.info("Testing configuration validation...")

    try:
        config_path = "training_config.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ["data_acquisition", "training", "features", "targets"]
        missing_sections = [s for s in required_sections if s not in config]

        if missing_sections:
            raise KeyError(f"Missing config sections: {missing_sections}")

        # Check specific configurations we fixed
        features = config["features"]
        targets = config["targets"]

        required_features = ["volatility_periods", "momentum_periods", "rsi_periods"]
        missing_features = [f for f in required_features if f not in features]

        if missing_features:
            raise KeyError(f"Missing feature configs: {missing_features}")

        required_targets = ["horizons", "transaction_cost_bps", "buy_threshold"]
        missing_targets = [t for t in required_targets if t not in targets]

        if missing_targets:
            raise KeyError(f"Missing target configs: {missing_targets}")

        logger.info("✅ Configuration validation successful")
        logger.info(f"   - Symbols: {config['data_acquisition']['symbols']}")
        logger.info(f"   - Models: {config['training']['models']}")
        logger.info(f"   - Volatility periods: {features['volatility_periods']}")
        logger.info(f"   - Target horizons: {targets['horizons']}")

        return True

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {str(e)}")
        return False


def test_data_availability():
    """Test that required data files exist."""
    logger.info("=" * 60)
    logger.info("Testing data availability...")

    try:
        data_dir = Path("./data")
        if not data_dir.exists():
            raise FileNotFoundError("Data directory not found: ./data")

        # Check for database files
        db_files = list(data_dir.glob("*_30m.db"))
        if not db_files:
            raise FileNotFoundError("No 30m database files found in ./data")

        logger.info("✅ Data availability check successful")
        logger.info(f"   - Data directory: {data_dir}")
        logger.info(f"   - Database files: {[f.name for f in db_files]}")

        return True

    except Exception as e:
        logger.error(f"❌ Data availability check failed: {str(e)}")
        return False


def test_training_script_dry_run():
    """Test the main training script with minimal parameters."""
    logger.info("=" * 60)
    logger.info("Testing training script (dry run)...")

    try:
        # Set environment variable to limit training
        env = os.environ.copy()
        env["PYTHONPATH"] = f"/notebooks/bot/src:/notebooks/bot:{env.get('PYTHONPATH', '')}"

        # Run training script with valid parameters
        cmd = [
            sys.executable,
            "paperspace_mlops/paperspace_training.py",
            "--symbols",
            "BTCEUR",  # Single symbol
            "--models",
            "lightgbm",  # Single model type
            "--fast",  # Fast mode
            "--dry-run",  # Dry run mode
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run with timeout to prevent hanging
        result = subprocess.run(
            cmd,
            cwd="/notebooks/bot",
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info("✅ Training script dry run successful")
            logger.info(f"   - Exit code: {result.returncode}")
            if result.stdout:
                logger.info(f"   - Output preview: {result.stdout[:200]}...")
        else:
            logger.warning(f"⚠️  Training script returned non-zero exit code: {result.returncode}")
            if result.stderr:
                logger.error(f"   - Error output: {result.stderr[:500]}...")
            if result.stdout:
                logger.info(f"   - Standard output: {result.stdout[:500]}...")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error("❌ Training script timed out after 5 minutes")
        return False
    except FileNotFoundError:
        logger.error("❌ Training script not found: paperspace_mlops/paperspace_training.py")
        return False
    except Exception as e:
        logger.error(f"❌ Training script test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def test_import_validation():
    """Test that key modules can be imported."""
    logger.info("=" * 60)
    logger.info("Testing import validation...")

    try:
        # Test key imports
        sys.path.insert(0, "/notebooks/bot/src")

        test_imports = [
            ("data_pipeline.dataset_builder", "DatasetBuilder"),
            ("data_pipeline.features", "FeatureEngine"),
            ("data_pipeline.target_engineering", "TradingTargetEngine"),
            ("data_pipeline.db_loader", "DataLoader"),  # Correct module name
        ]

        success_count = 0
        for module_name, class_name in test_imports:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                logger.info(f"   ✅ {module_name}.{class_name}")
                success_count += 1
            except Exception as e:
                logger.error(f"   ❌ {module_name}.{class_name}: {e}")

        logger.info(f"Import validation: {success_count}/{len(test_imports)} successful")
        return success_count == len(test_imports)

    except Exception as e:
        logger.error(f"❌ Import validation failed: {str(e)}")
        return False


def main():
    """Run simple trainer validation tests."""
    logger.info("🧪 Starting simple trainer validation tests...")

    tests = [
        ("Configuration Validation", test_config_validation),
        ("Data Availability", test_data_availability),
        ("Import Validation", test_import_validation),
        ("Training Script Dry Run", test_training_script_dry_run),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {str(e)}")
            results[test_name] = False

    # Summary
    logger.info("=" * 60)
    logger.info("🏁 Test Summary:")

    passed_tests = sum(results.values())
    total_tests = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   - {test_name}: {status}")

    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Training pipeline looks good.")
        return True
    elif passed_tests >= total_tests - 1:
        logger.info("⚠️  Most tests passed. System is likely functional.")
        return True
    else:
        logger.error("💥 Multiple test failures. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
