"""
Enhanced Model Transfer Service
===============================

Robust model transfer system with multiple fallback methods for getting
trained models from Paperspace to the production server.

Supports:
- Direct HTTP upload to production server
- Cloud storage (AWS S3, Google Cloud, Azure)
- GitHub releases
- Email transfer (fallback)
- Webhook notifications to production server
"""

import json
import logging
import os
import smtplib
import time
import zipfile
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import requests

logger = logging.getLogger(__name__)


class ModelTransferService:
    """Enhanced model transfer service with multiple methods"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transfer_methods = [
            self._transfer_http_direct,
            self._transfer_cloud_storage,
            self._transfer_github_release,
            self._transfer_email,
        ]

    def transfer_models(self, package_file: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer models using the first available method"""

        logger.info(f"🚀 Starting model transfer: {package_file.name}")

        transfer_results = []
        success = False

        for i, method in enumerate(self.transfer_methods):
            try:
                logger.info(f"🔄 Trying transfer method {i+1}/{len(self.transfer_methods)}")
                result = method(package_file, metadata)
                transfer_results.append(result)

                if result.get("success", False):
                    success = True
                    logger.info(f"✅ Transfer successful via {result.get('method', 'unknown')}")
                    break
                else:
                    logger.warning(
                        f"⚠️ Method {i+1} failed: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                error_msg = f"Method {i+1} exception: {str(e)}"
                logger.error(f"❌ {error_msg}")
                transfer_results.append(
                    {"success": False, "method": f"method_{i+1}", "error": error_msg}
                )

        # Send notification regardless of transfer success
        try:
            self._notify_production_server(package_file, metadata, success, transfer_results)
        except Exception as e:
            logger.warning(f"⚠️ Notification failed: {e}")

        return {
            "success": success,
            "transfer_results": transfer_results,
            "package_size_mb": package_file.stat().st_size / (1024 * 1024),
            "timestamp": datetime.now().isoformat(),
        }

    def _transfer_http_direct(self, package_file: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer via direct HTTP upload to production server"""

        upload_endpoint = os.environ.get("PRODUCTION_UPLOAD_ENDPOINT")
        if not upload_endpoint:
            return {
                "success": False,
                "method": "http_direct",
                "error": "No upload endpoint configured",
            }

        logger.info(f"📤 Uploading to: {upload_endpoint}")

        with open(package_file, "rb") as f:
            files = {"model_package": (package_file.name, f, "application/zip")}

            headers = {
                "X-API-Key": os.environ.get("PRODUCTION_API_KEY", ""),
                "X-Package-Metadata": json.dumps(
                    {
                        **metadata,
                        "transfer_method": "http_direct",
                        "upload_timestamp": datetime.now().isoformat(),
                        "source": "paperspace_gradient",
                    }
                ),
            }

            # Upload with progress tracking
            response = requests.post(
                upload_endpoint, files=files, headers=headers, timeout=600  # 10 minute timeout
            )

            response.raise_for_status()

        return {
            "success": True,
            "method": "http_direct",
            "endpoint": upload_endpoint,
            "response": response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else response.text,
            "status_code": response.status_code,
        }

    def _transfer_cloud_storage(
        self, package_file: Path, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transfer via cloud storage (AWS S3, Google Cloud, Azure)"""

        # Try AWS S3 first
        aws_bucket = os.environ.get("AWS_MODELS_BUCKET")
        if aws_bucket:
            return self._upload_to_s3(package_file, aws_bucket, metadata)

        # Try Google Cloud Storage
        gcp_bucket = os.environ.get("GCP_MODELS_BUCKET")
        if gcp_bucket:
            return self._upload_to_gcp(package_file, gcp_bucket, metadata)

        # Try Azure Blob Storage
        azure_container = os.environ.get("AZURE_MODELS_CONTAINER")
        if azure_container:
            return self._upload_to_azure(package_file, azure_container, metadata)

        return {"success": False, "method": "cloud_storage", "error": "No cloud storage configured"}

    def _upload_to_s3(
        self, package_file: Path, bucket: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload to AWS S3"""

        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )

            # Generate key with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            key = f"model_packages/{timestamp}_{package_file.name}"

            # Upload with metadata
            extra_args = {
                "Metadata": {
                    "source": "paperspace_gradient",
                    "upload_timestamp": datetime.now().isoformat(),
                    "package_metadata": json.dumps(metadata),
                }
            }

            s3_client.upload_file(str(package_file), bucket, key, ExtraArgs=extra_args)

            # Generate presigned URL for download
            download_url = s3_client.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=86400  # 24 hours
            )

            return {
                "success": True,
                "method": "aws_s3",
                "bucket": bucket,
                "key": key,
                "download_url": download_url,
                "location": f"s3://{bucket}/{key}",
            }

        except Exception as e:
            return {"success": False, "method": "aws_s3", "error": str(e)}

    def _upload_to_gcp(
        self, package_file: Path, bucket: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload to Google Cloud Storage"""

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket_obj = client.bucket(bucket)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"model_packages/{timestamp}_{package_file.name}"

            blob = bucket_obj.blob(blob_name)
            blob.metadata = {
                "source": "paperspace_gradient",
                "upload_timestamp": datetime.now().isoformat(),
                "package_metadata": json.dumps(metadata),
            }

            blob.upload_from_filename(str(package_file))

            return {
                "success": True,
                "method": "gcp_storage",
                "bucket": bucket,
                "blob_name": blob_name,
                "location": f"gs://{bucket}/{blob_name}",
            }

        except Exception as e:
            return {"success": False, "method": "gcp_storage", "error": str(e)}

    def _upload_to_azure(
        self, package_file: Path, container: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload to Azure Blob Storage"""

        try:
            from azure.storage.blob import BlobServiceClient

            account_name = os.environ.get("AZURE_STORAGE_ACCOUNT")
            account_key = os.environ.get("AZURE_STORAGE_KEY")

            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"model_packages/{timestamp}_{package_file.name}"

            blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)

            with open(package_file, "rb") as data:
                blob_client.upload_blob(
                    data,
                    metadata={
                        "source": "paperspace_gradient",
                        "upload_timestamp": datetime.now().isoformat(),
                        "package_metadata": json.dumps(metadata),
                    },
                )

            return {
                "success": True,
                "method": "azure_blob",
                "container": container,
                "blob_name": blob_name,
                "location": f"azure://{account_name}/{container}/{blob_name}",
            }

        except Exception as e:
            return {"success": False, "method": "azure_blob", "error": str(e)}

    def _transfer_github_release(
        self, package_file: Path, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transfer via GitHub release"""

        github_repo = os.environ.get("GITHUB_MODELS_REPO")  # Format: owner/repo
        github_token = os.environ.get("GITHUB_TOKEN")

        if not github_repo or not github_token:
            return {
                "success": False,
                "method": "github_release",
                "error": "GitHub configuration missing",
            }

        try:
            # Create release
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag_name = f"models-{timestamp}"

            release_data = {
                "tag_name": tag_name,
                "name": f"Training Models {timestamp}",
                "body": f"Automated model package from Paperspace\n\nMetadata:\n```json\n{json.dumps(metadata, indent=2)}\n```",
                "draft": False,
                "prerelease": True,
            }

            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            }

            # Create release
            release_response = requests.post(
                f"https://api.github.com/repos/{github_repo}/releases",
                json=release_data,
                headers=headers,
            )
            release_response.raise_for_status()
            release_info = release_response.json()

            # Upload asset
            upload_url = release_info["upload_url"].replace("{?name,label}", "")

            with open(package_file, "rb") as f:
                asset_response = requests.post(
                    f"{upload_url}?name={package_file.name}",
                    data=f,
                    headers={**headers, "Content-Type": "application/zip"},
                )
                asset_response.raise_for_status()

            return {
                "success": True,
                "method": "github_release",
                "repo": github_repo,
                "release_url": release_info["html_url"],
                "download_url": asset_response.json()["browser_download_url"],
                "tag_name": tag_name,
            }

        except Exception as e:
            return {"success": False, "method": "github_release", "error": str(e)}

    def _transfer_email(self, package_file: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer via email (fallback method)"""

        smtp_server = os.environ.get("SMTP_SERVER")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_username = os.environ.get("SMTP_USERNAME")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        recipient_email = os.environ.get("MODELS_RECIPIENT_EMAIL")

        if not all([smtp_server, smtp_username, smtp_password, recipient_email]):
            return {"success": False, "method": "email", "error": "Email configuration missing"}

        # Check file size (most email providers limit to 25MB)
        file_size_mb = package_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 20:  # 20MB limit for safety
            return {
                "success": False,
                "method": "email",
                "error": f"File too large for email: {file_size_mb:.1f}MB",
            }

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = smtp_username
            msg["To"] = recipient_email
            msg["Subject"] = f"Trading Bot Models - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            # Body
            body = f"""
Automated model package from Paperspace Gradient training.

Package: {package_file.name}
Size: {file_size_mb:.1f} MB
Timestamp: {datetime.now().isoformat()}

Metadata:
{json.dumps(metadata, indent=2)}

Please import these models to the production server using:
./import_models.sh

Best regards,
Paperspace Training Bot
"""
            msg.attach(MIMEText(body, "plain"))

            # Attach file
            with open(package_file, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {package_file.name}")
            msg.attach(part)

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)

            text = msg.as_string()
            server.sendmail(smtp_username, recipient_email, text)
            server.quit()

            return {
                "success": True,
                "method": "email",
                "recipient": recipient_email,
                "file_size_mb": file_size_mb,
            }

        except Exception as e:
            return {"success": False, "method": "email", "error": str(e)}

    def _notify_production_server(
        self,
        package_file: Path,
        metadata: Dict[str, Any],
        transfer_success: bool,
        transfer_results: List[Dict],
    ) -> None:
        """Send notification to production server about model availability"""

        webhook_url = os.environ.get("PRODUCTION_WEBHOOK_URL")
        if not webhook_url:
            logger.info("No webhook URL configured - skipping notification")
            return

        notification_data = {
            "event": "models_available",
            "timestamp": datetime.now().isoformat(),
            "package_name": package_file.name,
            "package_size_mb": package_file.stat().st_size / (1024 * 1024),
            "transfer_success": transfer_success,
            "transfer_results": transfer_results,
            "metadata": metadata,
            "source": "paperspace_gradient",
            "job_id": os.environ.get("PAPERSPACE_JOB_ID", "unknown"),
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": os.environ.get("PRODUCTION_API_KEY", ""),
            "User-Agent": "PaperspaceTrainingBot/1.0",
        }

        try:
            response = requests.post(
                webhook_url, json=notification_data, headers=headers, timeout=30
            )

            if response.status_code == 200:
                logger.info("✅ Production server notified successfully")
            else:
                logger.warning(f"⚠️ Notification returned status {response.status_code}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to notify production server: {e}")


# Convenience function for use in orchestrator
def transfer_models_package(
    package_file: Path, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to transfer a model package"""

    if metadata is None:
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "source": "paperspace_gradient",
            "symbols": config.get("data_acquisition", {}).get("symbols", []),
            "models": config.get("training", {}).get("models", []),
        }

    transfer_service = ModelTransferService(config)
    return transfer_service.transfer_models(package_file, metadata)
