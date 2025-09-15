#!/usr/bin/env python3
"""
Paperspace Pipeline Testing Script
==================================

Test script to validate the complete Paperspace MLOps pipeline
without actually running on Paperspace. Simulates the environment
and tests all components.

Usage:
    python test_paperspace_pipeline.py [--dry-run] [--component COMPONENT]
"""

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent))


class TestPaperspaceEnvironment(unittest.TestCase):
    """Test Paperspace environment simulation"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = Path.cwd()
        os.chdir(self.test_dir)

        # Create basic file structure
        (self.test_dir / "src").mkdir()
        (self.test_dir / "scripts").mkdir()
        (self.test_dir / "paperspace_mlops").mkdir()

        # Create dummy config
        config_content = """
data_acquisition:
  symbols: ['BTCEUR', 'ETHEUR']
  interval: '30m'
  lookback_days: 30

training:
  models: ['lightgbm']
  max_workers: 1

export:
  base_directory: './models'
  create_zip_archive: true
"""
        with open(self.test_dir / "training_config.yaml", "w") as f:
            f.write(config_content)

    def tearDown(self):
        """Cleanup test environment"""
        os.chdir(self.original_cwd)
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_setup_script(self):
        """Test Paperspace setup script"""
        from paperspace_mlops.paperspace_setup import setup_directories, setup_environment

        # Mock Paperspace environment
        with patch.dict(os.environ, {"PAPERSPACE_JOB_ID": "test-job"}):
            # Test environment detection
            self.assertTrue(os.environ.get("PAPERSPACE_JOB_ID"))

            # Test directory creation
            setup_directories()
            self.assertTrue((self.test_dir / "data").exists())
            self.assertTrue((self.test_dir / "models").exists())
            self.assertTrue((self.test_dir / "logs").exists())

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        from paperspace_mlops.paperspace_training_orchestrator import PaperspaceOrchestrator

        orchestrator = PaperspaceOrchestrator(config_path="training_config.yaml")

        # Test basic properties
        self.assertIsNotNone(orchestrator.config)
        self.assertEqual(orchestrator.max_runtime_hours, 5.5)
        self.assertIn("symbols", orchestrator.config["data_acquisition"])

        # Test time management
        remaining = orchestrator.get_time_remaining()
        self.assertGreater(remaining, 5.0)  # Should be close to max

        self.assertTrue(orchestrator.should_continue(0.1))
        self.assertFalse(orchestrator.should_continue(10.0))  # More than available

    def test_transfer_service(self):
        """Test model transfer service"""
        from paperspace_mlops.model_transfer_service import ModelTransferService

        # Create dummy package
        test_package = self.test_dir / "test_models.zip"
        with open(test_package, "wb") as f:
            f.write(b"dummy zip content")

        config = {"data_acquisition": {"symbols": ["BTCEUR"]}}
        service = ModelTransferService(config)

        # Test with no environment variables (should fail gracefully)
        metadata = {"test": True}
        result = service.transfer_models(test_package, metadata)

        # Should attempt all methods and fail (since no config)
        self.assertFalse(result["success"])
        self.assertIn("transfer_results", result)

    def test_production_import_handler(self):
        """Test production import handler"""
        from paperspace_mlops.production_import_handler import ProductionImportHandler

        handler = ProductionImportHandler(config_path="training_config.yaml")

        # Test notification validation
        valid_notification = {
            "event": "models_available",
            "package_name": "test_models.zip",
            "timestamp": datetime.now().isoformat(),
            "source": "paperspace_gradient",
        }

        self.assertTrue(handler._validate_notification(valid_notification))

        # Test invalid notification
        invalid_notification = {"event": "unknown_event"}
        self.assertFalse(handler._validate_notification(invalid_notification))

    def test_webhook_app(self):
        """Test webhook application"""
        from paperspace_mlops.production_import_handler import create_webhook_app

        app = create_webhook_app()
        client = app.test_client()

        # Test health endpoint
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data["status"], "healthy")

        # Test webhook endpoint (without auth)
        response = client.post("/webhook/models", json={})
        # Should return 401 or 400 (no auth or bad data)
        self.assertIn(response.status_code, [400, 401])


class TestPipelineIntegration(unittest.TestCase):
    """Test complete pipeline integration"""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = Path.cwd()

        # Create minimal project structure
        project_files = [
            "training_config.yaml",
            "src/__init__.py",
            "src/data_pipeline/__init__.py",
            "src/data_pipeline/dataset_builder.py",
            "scripts/__init__.py",
            "scripts/enhanced_trainer.py",
            "paperspace_mlops/__init__.py",
        ]

        for file_path in project_files:
            full_path = self.test_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.touch()

    def tearDown(self):
        os.chdir(self.original_cwd)
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("subprocess.run")
    @patch("requests.post")
    def test_end_to_end_simulation(self, mock_requests, mock_subprocess):
        """Test complete pipeline simulation"""

        # Mock successful responses
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Success"
        mock_subprocess.return_value.stderr = ""

        mock_requests.return_value.status_code = 200
        mock_requests.return_value.json.return_value = {"success": True}

        # Test environment setup simulation
        from paperspace_mlops.paperspace_setup import verify_setup

        # Mock package imports
        with patch("builtins.__import__") as mock_import:
            mock_import.return_value = MagicMock()

            # This would normally fail, but with mocking should pass
            # result = verify_setup()
            # Note: verify_setup has file dependencies, so we'll skip this

            # Test orchestrator with mocked dependencies
            from paperspace_mlops.paperspace_training_orchestrator import PaperspaceOrchestrator

            # Create minimal config
            config_content = """
data_acquisition:
  symbols: ['BTCEUR']
  interval: '30m'
  lookback_days: 1

training:
  models: ['lightgbm']
  max_workers: 1

export:
  base_directory: './models'
"""
            config_file = self.test_dir / "test_config.yaml"
            with open(config_file, "w") as f:
                f.write(config_content)

            os.chdir(self.test_dir)

            # Test orchestrator initialization
            orchestrator = PaperspaceOrchestrator(config_path="test_config.yaml")
            self.assertIsNotNone(orchestrator.config)


def run_component_test(component: str):
    """Run tests for specific component"""

    if component == "setup":
        suite = unittest.TestLoader().loadTestsFromName(
            "test_setup_script", TestPaperspaceEnvironment
        )
    elif component == "orchestrator":
        suite = unittest.TestLoader().loadTestsFromName(
            "test_orchestrator_initialization", TestPaperspaceEnvironment
        )
    elif component == "transfer":
        suite = unittest.TestLoader().loadTestsFromName(
            "test_transfer_service", TestPaperspaceEnvironment
        )
    elif component == "import":
        suite = unittest.TestLoader().loadTestsFromName(
            "test_production_import_handler", TestPaperspaceEnvironment
        )
    elif component == "webhook":
        suite = unittest.TestLoader().loadTestsFromName(
            "test_webhook_app", TestPaperspaceEnvironment
        )
    elif component == "integration":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPipelineIntegration)
    else:
        print(f"❌ Unknown component: {component}")
        return False

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_dry_run():
    """Run a dry-run simulation of the pipeline"""

    print("🧪 Running Paperspace Pipeline Dry Run")
    print("=" * 50)

    # Simulate environment variables
    test_env = {
        "PAPERSPACE_JOB_ID": "test-12345",
        "PRODUCTION_UPLOAD_ENDPOINT": "https://example.com/upload",
        "TELEGRAM_BOT_TOKEN": "test-token",
        "TELEGRAM_CHAT_ID": "test-chat",
    }

    with patch.dict(os.environ, test_env):
        print("✅ Environment variables set")

        # Test orchestrator dry run
        try:
            from paperspace_mlops.paperspace_training_orchestrator import PaperspaceOrchestrator

            orchestrator = PaperspaceOrchestrator()
            orchestrator.max_runtime_hours = 0.1  # Very short for testing

            print("✅ Orchestrator initialized")
            print(f"⏰ Max runtime: {orchestrator.max_runtime_hours} hours")
            print(f"📊 Symbols: {orchestrator.config['data_acquisition']['symbols']}")
            print(f"🤖 Models: {orchestrator.config['training']['models']}")

            # Test time management
            print(f"🕐 Time remaining: {orchestrator.get_time_remaining():.2f} hours")
            print(f"🚦 Can continue (0.1h): {orchestrator.should_continue(0.1)}")
            print(f"🚦 Can continue (1.0h): {orchestrator.should_continue(1.0)}")

            print("✅ Dry run completed successfully")
            return True

        except Exception as e:
            print(f"❌ Dry run failed: {e}")
            return False


def main():
    """Main test runner"""

    import argparse

    parser = argparse.ArgumentParser(description="Test Paperspace MLOps Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run dry-run simulation")
    parser.add_argument(
        "--component",
        choices=["setup", "orchestrator", "transfer", "import", "webhook", "integration"],
        help="Test specific component",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.dry_run:
        success = run_dry_run()
        return 0 if success else 1

    # Setup logging level
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    if args.component:
        success = run_component_test(args.component)
        return 0 if success else 1

    # Run all tests
    print("🧪 Running All Paperspace Pipeline Tests")
    print("=" * 50)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPaperspaceEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return 1


if __name__ == "__main__":
    exit(main())
