"""
Production Server Import Handler
================================

Automated model import handler for the production server.
Handles incoming model packages from Paperspace training and
automatically imports them into the production system.

Features:
- Webhook endpoint for notifications
- Automatic model download and import
- Validation and testing of imported models
- Backup of previous models
- Integration with existing import_models.sh
"""

import json
import logging
import os
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)


class ProductionImportHandler:
    """Handles automated model imports on production server"""

    def __init__(self, config_path: str = "training_config.yaml"):
        self.config_path = config_path
        self.models_dir = Path("./models")
        self.backup_dir = Path("./model_backups")
        self.temp_dir = Path("./temp_imports")
        self.logs_dir = Path("./logs")

        # Create directories
        for dir_path in [self.backup_dir, self.temp_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for import operations"""
        log_file = self.logs_dir / f"model_imports_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def handle_webhook_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook notification from Paperspace"""

        logger.info("📥 Received model notification from Paperspace")
        logger.info(f"Package: {notification_data.get('package_name', 'unknown')}")

        try:
            # Validate notification
            if not self._validate_notification(notification_data):
                return {"success": False, "error": "Invalid notification data"}

            # Check if transfer was successful
            if not notification_data.get("transfer_success", False):
                logger.warning("⚠️ Paperspace reported transfer failure")
                return {"success": False, "error": "Paperspace transfer failed"}

            # Find download URL from transfer results
            download_info = self._extract_download_info(
                notification_data.get("transfer_results", [])
            )
            if not download_info:
                logger.warning("⚠️ No valid download URL found")
                return {"success": False, "error": "No download URL available"}

            # Download and import models
            import_result = self._download_and_import_models(download_info, notification_data)

            return import_result

        except Exception as e:
            logger.error(f"❌ Import failed: {e}")
            return {"success": False, "error": str(e)}

    def _validate_notification(self, data: Dict[str, Any]) -> bool:
        """Validate incoming notification data"""

        required_fields = ["event", "package_name", "timestamp", "source"]
        for field in required_fields:
            if field not in data:
                logger.error(f"❌ Missing required field: {field}")
                return False

        if data["event"] != "models_available":
            logger.error(f"❌ Unknown event type: {data['event']}")
            return False

        if data["source"] != "paperspace_gradient":
            logger.warning(f"⚠️ Unexpected source: {data['source']}")

        return True

    def _extract_download_info(self, transfer_results: list) -> Optional[Dict[str, Any]]:
        """Extract download information from transfer results"""

        # Prioritize transfer methods
        method_priority = [
            "http_direct",
            "aws_s3",
            "gcp_storage",
            "azure_blob",
            "github_release",
            "email",
        ]

        successful_transfers = [r for r in transfer_results if r.get("success", False)]

        if not successful_transfers:
            return None

        # Find best transfer method
        for method in method_priority:
            for result in successful_transfers:
                if result.get("method") == method:
                    return result

        # Return first successful transfer if no preferred method found
        return successful_transfers[0]

    def _download_and_import_models(
        self, download_info: Dict[str, Any], notification_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Download model package and import it"""

        package_name = notification_data["package_name"]
        temp_package = self.temp_dir / package_name

        try:
            # Download package
            logger.info(f"📥 Downloading model package: {package_name}")
            self._download_package(download_info, temp_package)

            # Validate package
            if not self._validate_package(temp_package):
                return {"success": False, "error": "Package validation failed"}

            # Backup existing models
            backup_path = self._backup_existing_models()

            # Import models
            import_result = self._import_models(temp_package)

            if import_result["success"]:
                # Validate imported models
                validation_result = self._validate_imported_models()

                if validation_result["success"]:
                    logger.info("✅ Model import completed successfully")

                    # Cleanup temp files
                    temp_package.unlink()

                    # Send success notification
                    self._send_import_notification(True, notification_data, import_result)

                    return {
                        "success": True,
                        "imported_models": import_result.get("imported_models", []),
                        "backup_path": str(backup_path),
                        "validation": validation_result,
                    }
                else:
                    # Restore backup if validation failed
                    logger.error("❌ Model validation failed - restoring backup")
                    self._restore_backup(backup_path)

                    return {
                        "success": False,
                        "error": "Model validation failed",
                        "restored_backup": True,
                    }
            else:
                return import_result

        except Exception as e:
            logger.error(f"❌ Download/import failed: {e}")
            return {"success": False, "error": str(e)}

    def _download_package(self, download_info: Dict[str, Any], target_path: Path) -> None:
        """Download package from available source"""

        method = download_info.get("method")

        if method == "http_direct":
            # Download from direct URL
            response = requests.get(download_info["download_url"], stream=True, timeout=300)
            response.raise_for_status()

            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        elif method == "aws_s3":
            # Download from S3
            import boto3

            s3_client = boto3.client("s3")
            bucket = download_info["bucket"]
            key = download_info["key"]
            s3_client.download_file(bucket, key, str(target_path))

        elif method == "github_release":
            # Download from GitHub
            response = requests.get(download_info["download_url"], stream=True, timeout=300)
            response.raise_for_status()

            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        else:
            raise ValueError(f"Unsupported download method: {method}")

        logger.info(f"✅ Downloaded {target_path.stat().st_size / (1024*1024):.1f} MB")

    def _validate_package(self, package_path: Path) -> bool:
        """Validate downloaded package"""

        try:
            # Check if it's a valid zip file
            with zipfile.ZipFile(package_path, "r") as zf:
                # Check for required files/structure
                file_list = zf.namelist()

                # Look for model files
                model_files = [f for f in file_list if f.endswith((".pkl", ".pt", ".zip"))]
                if not model_files:
                    logger.error("❌ No model files found in package")
                    return False

                # Check for metadata
                metadata_files = [f for f in file_list if "metadata" in f.lower()]
                if not metadata_files:
                    logger.warning("⚠️ No metadata files found")

                logger.info(f"✅ Package contains {len(model_files)} model files")
                return True

        except Exception as e:
            logger.error(f"❌ Package validation failed: {e}")
            return False

    def _backup_existing_models(self) -> Path:
        """Backup existing models before import"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"models_backup_{timestamp}"

        if self.models_dir.exists():
            logger.info(f"💾 Backing up existing models to {backup_path}")
            shutil.copytree(self.models_dir, backup_path)
        else:
            logger.info("ℹ️ No existing models to backup")
            backup_path.mkdir(exist_ok=True)

        return backup_path

    def _import_models(self, package_path: Path) -> Dict[str, Any]:
        """Import models using existing import script"""

        try:
            # Copy package to root directory (where import_models.sh expects it)
            root_package = Path(".") / package_path.name
            shutil.copy2(package_path, root_package)

            # Run import script
            logger.info("🔄 Running import_models.sh")
            result = subprocess.run(
                ["./import_models.sh"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Clean up copied package
            root_package.unlink(missing_ok=True)

            if result.returncode == 0:
                logger.info("✅ import_models.sh completed successfully")

                # Parse output to find imported models
                imported_models = self._parse_import_output(result.stdout)

                return {
                    "success": True,
                    "imported_models": imported_models,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                logger.error(f"❌ import_models.sh failed: {result.stderr}")
                return {
                    "success": False,
                    "error": f"Import script failed: {result.stderr}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Import script timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_import_output(self, output: str) -> list:
        """Parse import script output to extract imported models"""

        imported_models = []

        lines = output.split("\n")
        for line in lines:
            # Look for patterns indicating successful model import
            if "extracted to models/" in line.lower():
                # Extract model path
                parts = line.split("models/")
                if len(parts) > 1:
                    model_path = "models/" + parts[1].split()[0]
                    imported_models.append(model_path)

        return imported_models

    def _validate_imported_models(self) -> Dict[str, Any]:
        """Validate imported models can be loaded"""

        try:
            # Run quick validation test
            logger.info("🔍 Validating imported models")
            result = subprocess.run(
                ["python", "quick_test_system.py", "--models-only"],
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ Model validation passed")
                return {"success": True, "validation_output": result.stdout}
            else:
                logger.error(f"❌ Model validation failed: {result.stderr}")
                return {
                    "success": False,
                    "error": f"Validation failed: {result.stderr}",
                    "validation_output": result.stdout,
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Model validation timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _restore_backup(self, backup_path: Path) -> None:
        """Restore models from backup"""

        try:
            if self.models_dir.exists():
                shutil.rmtree(self.models_dir)

            if backup_path.exists() and any(backup_path.iterdir()):
                shutil.copytree(backup_path, self.models_dir)
                logger.info(f"✅ Restored models from backup: {backup_path}")
            else:
                logger.info("ℹ️ No backup to restore")

        except Exception as e:
            logger.error(f"❌ Failed to restore backup: {e}")

    def _send_import_notification(
        self, success: bool, original_notification: Dict[str, Any], import_result: Dict[str, Any]
    ) -> None:
        """Send notification about import result"""

        # Send Telegram notification if configured
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        if telegram_token and telegram_chat_id:
            try:
                status = "✅ SUCCESS" if success else "❌ FAILED"
                package_name = original_notification.get("package_name", "unknown")

                message = f"""
🤖 *Model Import Notification*

Status: {status}
Package: `{package_name}`
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

                if success:
                    imported_models = import_result.get("imported_models", [])
                    message += f"Imported Models: {len(imported_models)}\n"
                    if imported_models:
                        message += f"Models: {', '.join(imported_models[:3])}"
                        if len(imported_models) > 3:
                            message += f" and {len(imported_models) - 3} more"
                else:
                    error = import_result.get("error", "Unknown error")
                    message += f"Error: {error}"

                # Send message
                requests.post(
                    f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                    json={"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"},
                    timeout=10,
                )

            except Exception as e:
                logger.warning(f"⚠️ Failed to send Telegram notification: {e}")


# Flask app for webhook endpoint
def create_webhook_app() -> Flask:
    """Create Flask app for webhook endpoint"""

    app = Flask(__name__)
    import_handler = ProductionImportHandler()

    @app.route("/webhook/models", methods=["POST"])
    def handle_models_webhook():
        """Handle incoming model notifications"""

        try:
            # Validate API key
            api_key = request.headers.get("X-API-Key")
            expected_key = os.environ.get("PRODUCTION_API_KEY")

            if expected_key and api_key != expected_key:
                return jsonify({"error": "Invalid API key"}), 401

            # Process notification
            notification_data = request.get_json()
            result = import_handler.handle_webhook_notification(notification_data)

            if result["success"]:
                return jsonify(result), 200
            else:
                return jsonify(result), 400

        except Exception as e:
            logger.error(f"❌ Webhook error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

    return app


if __name__ == "__main__":
    # Run webhook server
    app = create_webhook_app()
    port = int(os.environ.get("WEBHOOK_PORT", 5000))

    logger.info(f"🚀 Starting webhook server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
