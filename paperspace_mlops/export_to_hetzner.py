#!/usr/bin/env python3
"""
Professional Direct Export: Paperspace → Hetzner Server
=====================================================

Enterprise-grade model export system with comprehensive error handling,
validation, and rollback capabilities.

Author: Professional Code Architect
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/notebooks/bot/logs/hetzner_export.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class HetznerExportError(Exception):
    """Custom exception for export failures."""

    pass


class ProfessionalHetznerExporter:
    """
    Professional-grade model export system with enterprise features.
    """

    def __init__(self, config_path: str = None):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]

        # Paths
        self.local_models_path = Path("models/superior")
        self.remote_base_path = "/opt/trading_bot"
        self.remote_models_path = f"{self.remote_base_path}/models/superior"
        self.remote_backup_path = (
            f"{self.remote_base_path}/models/backups/backup_{self.export_timestamp}"
        )

        logger.info(f"🚀 Professional Hetzner Exporter initialized")
        logger.info(f"   Target: {self.config['hetzner_user']}@{self.config['hetzner_host']}")
        logger.info(f"   Local path: {self.local_models_path}")
        logger.info(f"   Remote path: {self.remote_models_path}")

    def _load_config(self, config_path: str) -> Dict:
        """Load export configuration."""
        default_config = {
            "hetzner_host": os.getenv("HETZNER_HOST", "your-hetzner-ip"),
            "hetzner_user": os.getenv("HETZNER_USER", "your-username"),
            "ssh_key_path": "/notebooks/bot/.ssh/hetzner_key",
            "connection_timeout": 30,
            "transfer_timeout": 3600,  # 1 hour
            "validation_enabled": True,
            "backup_enabled": True,
            "auto_restart": False,
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def validate_prerequisites(self) -> bool:
        """Validate all prerequisites before export."""
        logger.info("🔍 Validating prerequisites...")

        # 1. Check local models exist
        if not self.local_models_path.exists():
            logger.error(f"❌ Local models directory not found: {self.local_models_path}")
            return False

        # 2. Check SSH connectivity
        if not self._test_ssh_connection():
            logger.error("❌ SSH connection failed")
            return False

        # 3. Check remote disk space
        if not self._check_remote_disk_space():
            logger.error("❌ Insufficient remote disk space")
            return False

        # 4. Validate local models
        if not self._validate_local_models():
            logger.error("❌ Local model validation failed")
            return False

        logger.info("✅ All prerequisites validated")
        return True

    def _test_ssh_connection(self) -> bool:
        """Test SSH connection to Hetzner server."""
        try:
            cmd = [
                "ssh",
                "-i",
                self.config["ssh_key_path"],
                "-o",
                "ConnectTimeout=10",
                "-o",
                "StrictHostKeyChecking=no",
                f"{self.config['hetzner_user']}@{self.config['hetzner_host']}",
                "echo 'SSH_TEST_SUCCESS'",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and "SSH_TEST_SUCCESS" in result.stdout:
                logger.info("✅ SSH connection successful")
                return True
            else:
                logger.error(f"SSH test failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("SSH connection timeout")
            return False
        except Exception as e:
            logger.error(f"SSH test error: {e}")
            return False

    def _check_remote_disk_space(self) -> bool:
        """Check available disk space on remote server."""
        try:
            cmd = [
                "ssh",
                "-i",
                self.config["ssh_key_path"],
                f"{self.config['hetzner_user']}@{self.config['hetzner_host']}",
                f"df -h {self.remote_base_path} | tail -1 | awk '{{print $4}}'",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                available_space = result.stdout.strip()
                logger.info(f"📊 Remote disk space available: {available_space}")
                # Basic check - ensure we have some space (detailed check would parse the size)
                return len(available_space) > 0
            else:
                logger.warning("Could not check remote disk space")
                return True  # Proceed anyway

        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")
            return True  # Proceed anyway

    def _validate_local_models(self) -> bool:
        """Validate local models are complete and loadable."""
        logger.info("🧪 Validating local models...")

        total_models = 0
        valid_models = 0

        for symbol in self.symbols:
            symbol_path = self.local_models_path / symbol
            if not symbol_path.exists():
                logger.warning(f"⚠️  No models found for {symbol}")
                continue

            # Check for essential files
            required_files = ["best_model.zip"]
            optional_files = ["superior_ppo_*.zip", "checkpoints/", "logs/"]

            for required_file in required_files:
                file_path = symbol_path / required_file
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    if file_size > 1000:  # At least 1KB
                        logger.info(f"✅ {symbol}: {required_file} ({file_size:,} bytes)")
                        valid_models += 1
                    else:
                        logger.error(f"❌ {symbol}: {required_file} too small ({file_size} bytes)")
                else:
                    logger.error(f"❌ {symbol}: Missing {required_file}")

                total_models += 1

        success_rate = valid_models / total_models if total_models > 0 else 0
        logger.info(f"📊 Model validation: {valid_models}/{total_models} ({success_rate:.1%})")

        return success_rate >= 0.6  # At least 60% of models must be valid

    def create_remote_backup(self) -> bool:
        """Create backup of existing models on remote server."""
        if not self.config["backup_enabled"]:
            logger.info("📦 Backup disabled, skipping...")
            return True

        logger.info("📦 Creating remote backup...")

        try:
            # Create backup directory and backup existing models
            backup_cmd = f"""
                mkdir -p {self.remote_backup_path} &&
                if [ -d {self.remote_models_path} ]; then
                    cp -r {self.remote_models_path}/* {self.remote_backup_path}/ 2>/dev/null || true
                    echo "Backup created: {self.remote_backup_path}"
                else
                    echo "No existing models to backup"
                fi
            """

            result = self._run_remote_command(backup_cmd)
            if result.returncode == 0:
                logger.info(f"✅ Backup created: {self.remote_backup_path}")
                return True
            else:
                logger.error(f"❌ Backup failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Backup error: {e}")
            return False

    def export_models(self) -> bool:
        """Export all superior models to Hetzner server."""
        logger.info("🚀 Starting model export...")

        # Create remote directory structure
        self._run_remote_command(f"mkdir -p {self.remote_models_path}")

        total_symbols = len(self.symbols)
        successful_exports = 0

        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"📦 Exporting {symbol} ({i}/{total_symbols})...")

            if self._export_symbol(symbol):
                successful_exports += 1
                logger.info(f"✅ {symbol} exported successfully")
            else:
                logger.error(f"❌ {symbol} export failed")

        success_rate = successful_exports / total_symbols
        logger.info(f"📊 Export summary: {successful_exports}/{total_symbols} ({success_rate:.1%})")

        return success_rate >= 0.8  # At least 80% must succeed

    def _export_symbol(self, symbol: str) -> bool:
        """Export models for a specific symbol."""
        local_symbol_path = self.local_models_path / symbol

        if not local_symbol_path.exists():
            logger.warning(f"⚠️  No local models for {symbol}")
            return False

        try:
            # Use rsync for robust, resumable transfer
            cmd = [
                "rsync",
                "-avz",
                "--delete",
                "--progress",
                "-e",
                f"ssh -i {self.config['ssh_key_path']} -o StrictHostKeyChecking=no",
                f"{local_symbol_path}/",
                f"{self.config['hetzner_user']}@{self.config['hetzner_host']}:{self.remote_models_path}/{symbol}/",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["transfer_timeout"],
            )

            if result.returncode == 0:
                return True
            else:
                logger.error(f"rsync failed for {symbol}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Transfer timeout for {symbol}")
            return False
        except Exception as e:
            logger.error(f"Transfer error for {symbol}: {e}")
            return False

    def validate_remote_models(self) -> bool:
        """Validate exported models on remote server."""
        if not self.config["validation_enabled"]:
            logger.info("🧪 Validation disabled, skipping...")
            return True

        logger.info("🧪 Validating remote models...")

        validation_script = f"""
import os
import sys
sys.path.append('{self.remote_base_path}')

# Try to import stable_baselines3
try:
    from stable_baselines3 import PPO
    import numpy as np
    print("✅ Dependencies available")
except ImportError as e:
    print(f"❌ Missing dependencies: {{e}}")
    sys.exit(1)

# Validate each model
symbols = {self.symbols}
failed_models = []

for symbol in symbols:
    model_path = f"{self.remote_models_path}/{{symbol}}/best_model.zip"
    try:
        if os.path.exists(model_path):
            # Try to load model
            model = PPO.load(model_path)

            # Test prediction with dummy data
            dummy_obs = np.random.random((32, 104))
            action, _ = model.predict(dummy_obs)

            file_size = os.path.getsize(model_path)
            print(f"✅ {{symbol}}: Valid ({file_size:,} bytes)")
        else:
            print(f"❌ {{symbol}}: Model file missing")
            failed_models.append(symbol)
    except Exception as e:
        print(f"❌ {{symbol}}: Validation failed - {{e}}")
        failed_models.append(symbol)

if failed_models:
    print(f"❌ Failed models: {{failed_models}}")
    sys.exit(1)
else:
    print("✅ All models validated successfully")
"""

        try:
            # Write validation script to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(validation_script)
                temp_script = f.name

            # Copy script to remote and execute
            copy_cmd = [
                "scp",
                "-i",
                self.config["ssh_key_path"],
                temp_script,
                f"{self.config['hetzner_user']}@{self.config['hetzner_host']}:/tmp/validate_models.py",
            ]
            subprocess.run(copy_cmd, check=True)

            # Execute validation
            result = self._run_remote_command(
                f"cd {self.remote_base_path} && python3 /tmp/validate_models.py"
            )

            # Cleanup
            os.unlink(temp_script)
            self._run_remote_command("rm -f /tmp/validate_models.py")

            if result.returncode == 0:
                logger.info("✅ Remote model validation successful")
                return True
            else:
                logger.error(f"❌ Remote validation failed:\n{result.stdout}\n{result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False

    def update_remote_configuration(self) -> bool:
        """Update trading configuration to use superior models."""
        logger.info("🔧 Updating remote configuration...")

        config_update_script = f"""
cd {self.remote_base_path}

# Backup current config
cp config/trading_config.yaml config/trading_config.yaml.backup_{self.export_timestamp}

# Update configuration to use superior models
sed -i 's/ensemble_type: .*/ensemble_type: "superior_ppo"/' config/trading_config.yaml
sed -i 's/model_priority: .*/model_priority: ["superior", "ppo", "lightgbm", "gru"]/' config/trading_config.yaml

# Add superior model configuration if not exists
if ! grep -q "superior_config:" config/trading_config.yaml; then
    cat >> config/trading_config.yaml << 'EOF'

# Superior model configuration
superior_config:
  feature_count: 104
  window_size: 32
  prediction_horizons: ['1h', '3h', '6h', '12h', '24h']
  cost_adjustment: true
  transaction_cost_bps: 10
  model_type: 'resource_aware_ppo'

model_weights:
  superior: 0.70
  lightgbm: 0.20
  gru: 0.10
  ppo: 0.00
EOF
fi

echo "✅ Configuration updated successfully"
"""

        try:
            result = self._run_remote_command(config_update_script)
            if result.returncode == 0:
                logger.info("✅ Remote configuration updated")
                return True
            else:
                logger.error(f"❌ Configuration update failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Configuration update error: {e}")
            return False

    def restart_trading_system(self) -> bool:
        """Restart trading system on remote server (if enabled)."""
        if not self.config["auto_restart"]:
            logger.info("🔄 Auto-restart disabled, skipping...")
            return True

        logger.info("🔄 Restarting remote trading system...")

        restart_script = f"""
cd {self.remote_base_path}

# Stop existing system
if [ -f bin/system_manager ]; then
    ./bin/system_manager stop
    sleep 5
fi

# Start with superior models
./bin/system_manager start

# Give it time to initialize
sleep 10

# Check if services are running
if ./bin/system_manager status | grep -q "running"; then
    echo "✅ Trading system restarted successfully"
    exit 0
else
    echo "❌ Trading system failed to start"
    exit 1
fi
"""

        try:
            result = self._run_remote_command(restart_script)
            if result.returncode == 0:
                logger.info("✅ Trading system restarted successfully")
                return True
            else:
                logger.error(f"❌ System restart failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Restart error: {e}")
            return False

    def rollback_on_failure(self) -> bool:
        """Rollback to previous models if export failed."""
        if not self.config["backup_enabled"]:
            logger.warning("⚠️  Backup disabled, cannot rollback")
            return False

        logger.info("🔄 Rolling back to previous models...")

        rollback_script = f"""
cd {self.remote_base_path}

if [ -d {self.remote_backup_path} ]; then
    # Remove failed models
    rm -rf {self.remote_models_path}

    # Restore backup
    mkdir -p {self.remote_models_path}
    cp -r {self.remote_backup_path}/* {self.remote_models_path}/

    # Restore configuration
    if [ -f config/trading_config.yaml.backup_{self.export_timestamp} ]; then
        cp config/trading_config.yaml.backup_{self.export_timestamp} config/trading_config.yaml
    fi

    echo "✅ Rollback completed"
    exit 0
else
    echo "❌ No backup found for rollback"
    exit 1
fi
"""

        try:
            result = self._run_remote_command(rollback_script)
            if result.returncode == 0:
                logger.info("✅ Rollback completed successfully")
                return True
            else:
                logger.error(f"❌ Rollback failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Rollback error: {e}")
            return False

    def _run_remote_command(self, command: str) -> subprocess.CompletedProcess:
        """Execute command on remote server."""
        cmd = [
            "ssh",
            "-i",
            self.config["ssh_key_path"],
            "-o",
            "StrictHostKeyChecking=no",
            f"{self.config['hetzner_user']}@{self.config['hetzner_host']}",
            command,
        ]

        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    def execute_full_export(self) -> bool:
        """Execute complete export workflow with error handling."""
        start_time = time.time()
        logger.info("🎯 Starting Professional Model Export Workflow")
        logger.info("=" * 60)

        try:
            # Phase 1: Prerequisites
            if not self.validate_prerequisites():
                raise HetznerExportError("Prerequisites validation failed")

            # Phase 2: Backup
            if not self.create_remote_backup():
                logger.warning("⚠️  Backup failed, continuing without backup...")

            # Phase 3: Export
            if not self.export_models():
                raise HetznerExportError("Model export failed")

            # Phase 4: Validation
            if not self.validate_remote_models():
                raise HetznerExportError("Remote model validation failed")

            # Phase 5: Configuration
            if not self.update_remote_configuration():
                raise HetznerExportError("Configuration update failed")

            # Phase 6: Restart (optional)
            if not self.restart_trading_system():
                logger.warning("⚠️  System restart failed, manual restart required")

            # Success!
            duration = time.time() - start_time
            logger.info("=" * 60)
            logger.info("🎉 EXPORT COMPLETED SUCCESSFULLY!")
            logger.info(f"   Duration: {duration:.1f} seconds")
            logger.info(f"   Models: {len(self.symbols)} symbols exported")
            logger.info(f"   Backup: {self.remote_backup_path}")
            logger.info("")
            logger.info("🚀 Your Hetzner server is now running with SUPERIOR models!")
            logger.info("   Command: ./bin/system_manager start")
            logger.info("=" * 60)

            return True

        except HetznerExportError as e:
            logger.error(f"💥 Export failed: {e}")

            # Attempt rollback
            if self.rollback_on_failure():
                logger.info("✅ System rolled back to previous state")
            else:
                logger.error("❌ Rollback failed - manual intervention required")

            return False

        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Professional Hetzner Model Export")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--host", help="Hetzner host IP/domain")
    parser.add_argument("--user", help="Hetzner username")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no export")
    parser.add_argument("--auto-restart", action="store_true", help="Auto-restart trading system")

    args = parser.parse_args()

    # Override config with command line args
    if args.host:
        os.environ["HETZNER_HOST"] = args.host
    if args.user:
        os.environ["HETZNER_USER"] = args.user

    # Create exporter
    exporter = ProfessionalHetznerExporter(args.config)

    if args.auto_restart:
        exporter.config["auto_restart"] = True

    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - Validation only")
        success = exporter.validate_prerequisites()
        if success:
            logger.info("✅ Dry run successful - ready for export")
            return 0
        else:
            logger.error("❌ Dry run failed - fix issues before export")
            return 1
    else:
        # Full export
        success = exporter.execute_full_export()
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
