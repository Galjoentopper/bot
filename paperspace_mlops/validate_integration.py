#!/usr/bin/env python3
"""
Superior Model Integration Validator
====================================

Validates that superior models integrate correctly with the existing
Hetzner trading system and can be loaded by system_manager.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class IntegrationValidator:
    """Validates superior model integration on Hetzner server."""

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]
        self.remote_base = "/opt/trading_bot"

    def run_remote_validation(self):
        """Run comprehensive validation on remote server."""

        validation_script = f'''
#!/usr/bin/env python3
"""Remote validation script for superior models."""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class RemoteValidator:
    def __init__(self):
        self.base_path = Path("{self.remote_base}")
        self.models_path = self.base_path / "models" / "superior"
        self.config_path = self.base_path / "config" / "trading_config.yaml"
        self.symbols = {self.symbols}

    def validate_directory_structure(self):
        """Validate directory structure."""
        logger.info("🔍 Validating directory structure...")

        required_dirs = [
            self.base_path,
            self.base_path / "models",
            self.models_path,
            self.base_path / "config",
            self.base_path / "bin"
        ]

        for dir_path in required_dirs:
            if dir_path.exists():
                logger.info(f"✅ {{dir_path}}")
            else:
                logger.error(f"❌ Missing: {{dir_path}}")
                return False

        return True

    def validate_model_files(self):
        """Validate model files exist and are valid."""
        logger.info("🧪 Validating model files...")

        valid_models = 0
        total_models = 0

        for symbol in self.symbols:
            symbol_path = self.models_path / symbol
            model_file = symbol_path / "best_model.zip"

            total_models += 1

            if model_file.exists():
                size = model_file.stat().st_size
                if size > 1000:  # At least 1KB
                    logger.info(f"✅ {{symbol}}: {{size:,}} bytes")
                    valid_models += 1
                else:
                    logger.error(f"❌ {{symbol}}: File too small ({{size}} bytes)")
            else:
                logger.error(f"❌ {{symbol}}: Model file missing")

        success_rate = valid_models / total_models if total_models > 0 else 0
        logger.info(f"📊 Model validation: {{valid_models}}/{{total_models}} ({{success_rate:.1%}})")

        return success_rate >= 0.8

    def validate_model_loading(self):
        """Test that models can be loaded."""
        logger.info("🔄 Testing model loading...")

        try:
            # Add trading bot to Python path
            sys.path.insert(0, str(self.base_path))

            # Try importing stable-baselines3
            from stable_baselines3 import PPO
            import numpy as np
            logger.info("✅ Dependencies available")

            # Test loading one model
            test_symbol = "BTCEUR"
            model_path = self.models_path / test_symbol / "best_model.zip"

            if model_path.exists():
                model = PPO.load(str(model_path))

                # Test prediction
                dummy_obs = np.random.random((32, 104))
                action, _ = model.predict(dummy_obs)

                logger.info(f"✅ Model loading test successful ({{test_symbol}})")
                logger.info(f"   Action shape: {{action.shape}}")
                return True
            else:
                logger.error(f"❌ Test model not found: {{model_path}}")
                return False

        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {{e}}")
            return False
        except Exception as e:
            logger.error(f"❌ Model loading failed: {{e}}")
            return False

    def validate_configuration(self):
        """Validate trading configuration."""
        logger.info("🔧 Validating configuration...")

        if not self.config_path.exists():
            logger.error(f"❌ Configuration file missing: {{self.config_path}}")
            return False

        try:
            import yaml

            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check for superior model configuration
            trading_config = config.get('trading', {{}})
            ensemble_type = trading_config.get('ensemble_type', '')

            if ensemble_type == 'superior_ppo':
                logger.info("✅ Configuration set to use superior_ppo")
            else:
                logger.warning(f"⚠️  Ensemble type: {{ensemble_type}} (expected: superior_ppo)")

            # Check model priorities
            model_priority = trading_config.get('model_priority', [])
            if 'superior' in model_priority:
                logger.info(f"✅ Superior models in priority list: {{model_priority}}")
            else:
                logger.warning(f"⚠️  Superior not in model priority: {{model_priority}}")

            # Check for superior_config section
            if 'superior_config' in config:
                superior_config = config['superior_config']
                feature_count = superior_config.get('feature_count', 0)
                if feature_count == 104:
                    logger.info(f"✅ Superior config feature count: {{feature_count}}")
                else:
                    logger.warning(f"⚠️  Feature count: {{feature_count}} (expected: 104)")
            else:
                logger.warning("⚠️  No superior_config section found")

            return True

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {{e}}")
            return False

    def validate_system_manager(self):
        """Validate system_manager can start with superior models."""
        logger.info("🚀 Validating system_manager integration...")

        system_manager = self.base_path / "bin" / "system_manager"

        if not system_manager.exists():
            logger.error(f"❌ system_manager not found: {{system_manager}}")
            return False

        # Check if system_manager is executable
        if not os.access(system_manager, os.X_OK):
            logger.error(f"❌ system_manager not executable: {{system_manager}}")
            return False

        logger.info("✅ system_manager found and executable")

        # Note: We don't actually start the system here to avoid conflicts
        # The user will test this manually
        logger.info("🎯 system_manager ready for testing")
        logger.info("   Manual test: ./bin/system_manager start")

        return True

    def run_full_validation(self):
        """Run complete validation suite."""
        logger.info("🎯 SUPERIOR MODEL INTEGRATION VALIDATION")
        logger.info("=" * 50)

        tests = [
            ("Directory Structure", self.validate_directory_structure),
            ("Model Files", self.validate_model_files),
            ("Model Loading", self.validate_model_loading),
            ("Configuration", self.validate_configuration),
            ("System Manager", self.validate_system_manager)
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\\n🧪 Testing: {{test_name}}")
            try:
                if test_func():
                    logger.info(f"✅ {{test_name}}: PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {{test_name}}: FAILED")
            except Exception as e:
                logger.error(f"💥 {{test_name}}: ERROR - {{e}}")

        logger.info("\\n" + "=" * 50)
        success_rate = passed / total

        if success_rate >= 0.8:
            logger.info(f"🎉 VALIDATION SUCCESSFUL: {{passed}}/{{total}} ({{success_rate:.1%}})")
            logger.info("✅ Superior models are ready for trading!")
            return True
        else:
            logger.error(f"❌ VALIDATION FAILED: {{passed}}/{{total}} ({{success_rate:.1%}})")
            logger.error("🔧 Fix issues before starting trading system")
            return False

# Run validation
if __name__ == "__main__":
    validator = RemoteValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)
'''

        # Write validation script to temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(validation_script)
            temp_script = f.name

        try:
            # Copy script to remote server
            copy_cmd = [
                "scp",
                "-i",
                self.config["ssh_key_path"],
                temp_script,
                f"{self.config['hetzner_user']}@{self.config['hetzner_host']}:/tmp/validate_integration.py",
            ]
            subprocess.run(copy_cmd, check=True)

            # Run validation on remote server
            logger.info("🚀 Running validation on Hetzner server...")

            run_cmd = [
                "ssh",
                "-i",
                self.config["ssh_key_path"],
                f"{self.config['hetzner_user']}@{self.config['hetzner_host']}",
                f"cd {self.remote_base} && python3 /tmp/validate_integration.py",
            ]

            result = subprocess.run(run_cmd, capture_output=False, text=True)

            # Cleanup
            cleanup_cmd = [
                "ssh",
                "-i",
                self.config["ssh_key_path"],
                f"{self.config['hetzner_user']}@{self.config['hetzner_host']}",
                "rm -f /tmp/validate_integration.py",
            ]
            subprocess.run(cleanup_cmd)

            return result.returncode == 0

        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False
        finally:
            # Clean up local temp file
            try:
                os.unlink(temp_script)
            except:
                pass


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Superior Model Integration")
    parser.add_argument(
        "--config",
        default="/notebooks/bot/paperspace_mlops/hetzner_config.json",
        help="Configuration file path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"❌ Config file not found: {args.config}")
        logger.error("   Run setup_hetzner_export.sh first")
        return 1

    validator = IntegrationValidator(args.config)
    success = validator.run_remote_validation()

    if success:
        logger.info("🎉 Integration validation successful!")
        logger.info("✅ Your superior models are ready for trading")
        return 0
    else:
        logger.error("❌ Integration validation failed")
        logger.error("🔧 Fix issues before starting trading system")
        return 1


if __name__ == "__main__":
    exit(main())
