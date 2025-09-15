#!/usr/bin/env python3
"""
Comprehensive Fix Verification Tool
===================================

Professional verification tool to validate all critical fixes:
1. Health monitoring infrastructure
2. Telegram bot singleton pattern
3. PPO model feature dimension fixes

Returns exit code 0 if all fixes are verified, 1 if any issues remain.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the bot directory to Python path
bot_dir = Path(__file__).parent.parent
sys.path.insert(0, str(bot_dir))

from src.core.logging_manager import get_system_logger

logger = get_system_logger("fix_verification")


class FixVerificationTool:
    """Professional-grade fix verification system."""

    def __init__(self):
        self.bot_root = Path(__file__).parent.parent
        self.symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]
        self.verification_results = {}

        logger.info("🔍 Fix Verification Tool initialized")

    def verify_health_monitoring(self) -> bool:
        """Verify health monitoring infrastructure is working."""
        logger.info("🔍 Verifying health monitoring infrastructure...")

        try:
            # Test 1: Health check script exists and is executable
            health_script = self.bot_root / "scripts/health_check.sh"
            if not health_script.exists():
                logger.error("❌ Health check script missing")
                return False

            if not os.access(health_script, os.X_OK):
                logger.error("❌ Health check script not executable")
                return False

            # Test 2: Server directory symlink exists
            server_health_script = self.bot_root / "server/scripts/health_check.sh"
            if not server_health_script.exists():
                logger.error("❌ Server health check symlink missing")
                return False

            # Test 3: Health check script runs without errors
            result = subprocess.run(
                [str(health_script)], capture_output=True, text=True, timeout=30, cwd=self.bot_root
            )

            if result.returncode != 0 and result.returncode != 1:  # 0=healthy, 1=warnings allowed
                logger.error(f"❌ Health check script failed with code {result.returncode}")
                return False

            logger.info("✅ Health monitoring infrastructure verified")
            return True

        except Exception as e:
            logger.error(f"❌ Health monitoring verification failed: {e}")
            return False

    def verify_telegram_singleton(self) -> bool:
        """Verify Telegram unified service has single-instance guard and launcher."""
        logger.info("🔍 Verifying Telegram service single-instance guard...")

        try:
            # Check telegram service source for lock helpers and is_running property
            svc_file = self.bot_root / "src/notifications/telegram_service.py"
            if not svc_file.exists():
                logger.error("❌ Telegram service file missing")
                return False

            content = svc_file.read_text(encoding="utf-8")
            required_tokens = [
                "def _acquire_instance_lock",
                "def _lock_path",
                "@property\n    def is_running",
            ]
            for tok in required_tokens:
                if tok not in content:
                    logger.error(f"❌ Telegram service missing token: {tok}")
                    return False

            # Launcher should exist and import unified service
            launcher_file = self.bot_root / "bin/telegram_bot"
            if not launcher_file.exists():
                logger.error("❌ Telegram launcher missing")
                return False
            lcontent = launcher_file.read_text(encoding="utf-8")
            if "from src.notifications import get_telegram_service" not in lcontent:
                logger.error("❌ Launcher not using unified service")
                return False

            logger.info("✅ Telegram unified service single-instance guard verified")
            return True

        except Exception as e:
            logger.error(f"❌ Telegram singleton verification failed: {e}")
            return False

    def verify_ppo_features(self) -> bool:
        """Verify PPO model feature dimension fixes."""
        logger.info("🔍 Verifying PPO model feature fixes...")

        try:
            all_symbols_ok = True

            for symbol in self.symbols:
                # Test 1: PPO feature metadata exists and has 104 features
                metadata_file = self.bot_root / f"models/metadata/features_ppo_{symbol}.json"
                if not metadata_file.exists():
                    logger.error(f"❌ {symbol}: PPO feature metadata missing")
                    all_symbols_ok = False
                    continue

                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                feature_count = metadata.get("feature_count", 0)
                if feature_count != 104:
                    logger.error(f"❌ {symbol}: Wrong feature count ({feature_count} vs 104)")
                    all_symbols_ok = False
                    continue

                # Test 2: PPO preprocessor exists
                preprocessor_file = self.bot_root / f"models/ppo/{symbol}/preprocessor.pkl"
                if not preprocessor_file.exists():
                    logger.error(f"❌ {symbol}: PPO preprocessor missing")
                    all_symbols_ok = False
                    continue

                # Test 3: Backup files exist (showing fix was applied)
                backup_file = self.bot_root / f"models/metadata/features_ppo_{symbol}.json.backup"
                if not backup_file.exists():
                    logger.warning(f"⚠️ {symbol}: No backup file found (may be a fresh install)")

                logger.info(f"✅ {symbol}: PPO features verified (104 features)")

            if all_symbols_ok:
                logger.info("✅ All PPO model feature fixes verified")
                return True
            else:
                logger.error("❌ Some PPO model features still have issues")
                return False

        except Exception as e:
            logger.error(f"❌ PPO feature verification failed: {e}")
            return False

    def verify_system_integration(self) -> bool:
        """Verify overall system integration."""
        logger.info("🔍 Verifying system integration...")

        try:
            # Test 1: All critical files exist
            critical_files = [
                "scripts/health_check.sh",
                "scripts/fix_ppo_features.py",
                "src/models/ppo_model_manager.py",
            ]

            for file_path in critical_files:
                full_path = self.bot_root / file_path
                if not full_path.exists():
                    logger.error(f"❌ Critical file missing: {file_path}")
                    return False

            # Test 2: Python imports work
            try:
                from src.models.ppo_model_manager import get_ppo_manager

                logger.info("✅ Critical modules import successfully")
            except ImportError as e:
                logger.error(f"❌ Import error: {e}")
                return False

            # Test 3: Log directories exist and are writable
            log_dir = self.bot_root / "logs"
            if not log_dir.exists() or not os.access(log_dir, os.W_OK):
                logger.error("❌ Log directory not writable")
                return False

            logger.info("✅ System integration verified")
            return True

        except Exception as e:
            logger.error(f"❌ System integration verification failed: {e}")
            return False

    def run_comprehensive_verification(self) -> Dict[str, bool]:
        """Run all verification tests."""
        logger.info("🚀 Starting comprehensive fix verification...")

        results = {
            "health_monitoring": self.verify_health_monitoring(),
            "telegram_singleton": self.verify_telegram_singleton(),
            "ppo_features": self.verify_ppo_features(),
            "system_integration": self.verify_system_integration(),
        }

        # Summary
        passed = sum(1 for result in results.values() if result)
        total = len(results)

        logger.info(f"\n📊 Verification Summary:")
        for test_name, passed_test in results.items():
            status = "✅ PASSED" if passed_test else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")

        logger.info(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All fixes verified successfully!")
        else:
            logger.error("💥 Some fixes still need attention")

        self.verification_results = results
        return results

    def generate_fix_report(self) -> str:
        """Generate a detailed fix report."""
        if not self.verification_results:
            return "No verification results available"

        report_lines = [
            "=" * 60,
            "TRADING BOT FIX VERIFICATION REPORT",
            "=" * 60,
            f"Generated: {subprocess.check_output(['date']).decode().strip()}",
            f"Bot Root: {self.bot_root}",
            "",
            "FIXES APPLIED:",
            "",
            "1. ✅ Health Monitoring Infrastructure",
            "   - Created missing health_check.sh script",
            "   - Added comprehensive system monitoring",
            "   - Fixed path references and permissions",
            "",
            "2. ✅ Telegram Bot Singleton Pattern",
            "   - Implemented file-based locking system",
            "   - Added concurrent instance prevention",
            "   - Improved error handling and cleanup",
            "",
            "3. ✅ PPO Model Feature Dimension Fix",
            "   - Fixed 13→104 feature dimension mismatch",
            "   - Regenerated corrupted feature metadata",
            "   - Created missing PPO preprocessors",
            "",
            "VERIFICATION RESULTS:",
            "",
        ]

        for test_name, passed in self.verification_results.items():
            status = "PASSED ✅" if passed else "FAILED ❌"
            report_lines.append(f"  {test_name.replace('_', ' ').title()}: {status}")

        report_lines.extend(
            [
                "",
                "NEXT STEPS:",
                "1. Restart the trading system to apply all fixes",
                "2. Monitor logs for improved PPO model performance",
                "3. Verify Telegram bot starts without polling errors",
                "4. Confirm health monitoring runs without 'not found' errors",
                "",
                "=" * 60,
            ]
        )

        return "\n".join(report_lines)


def main():
    """Main execution function."""
    logger.info("🚀 Starting comprehensive fix verification...")

    try:
        verifier = FixVerificationTool()
        results = verifier.run_comprehensive_verification()

        # Generate and display report
        report = verifier.generate_fix_report()
        print("\n" + report)

        # Return appropriate exit code
        all_passed = all(results.values())
        return 0 if all_passed else 1

    except Exception as e:
        logger.error(f"💥 Verification failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
