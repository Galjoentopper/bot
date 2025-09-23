#!/usr/bin/env python3
"""
S3 Model Export Script
=====================

Export trained models to AWS S3 for production deployment.
Creates a timestamped zip archive of all models and uploads to configured S3 bucket.

Usage:
    python export_to_s3.py                    # Export all models
    python export_to_s3.py --models-dir path  # Specify models directory
    python export_to_s3.py --bucket name      # Override bucket name
"""

import argparse
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import boto3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class S3ModelExporter:
    """Handle model export to S3"""

    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or os.environ.get("AWS_MODELS_BUCKET")
        if not self.bucket_name:
            raise ValueError("No S3 bucket specified. Set AWS_MODELS_BUCKET environment variable.")

        self.s3_client = boto3.client("s3")
        logger.info(f"Initialized S3 exporter for bucket: {self.bucket_name}")

    def export_models(self, models_dir: Path, include_validation: bool = True) -> Dict:
        """Export models to S3 bucket"""

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        # Create timestamped archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"models_export_{timestamp}.zip"
        archive_path = models_dir.parent / archive_name

        logger.info(f"Creating model archive: {archive_name}")

        files_added = 0
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add all model files
            for file_path in models_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(models_dir.parent)
                    zipf.write(file_path, arcname)
                    files_added += 1

            # Add validation stats if requested
            if include_validation:
                validation_dir = models_dir.parent / "validation"
                if validation_dir.exists():
                    for val_file in validation_dir.glob("*.json"):
                        arcname = val_file.relative_to(models_dir.parent)
                        zipf.write(val_file, arcname)
                        files_added += 1

            # Add metadata
            metadata = {
                "export_timestamp": timestamp,
                "export_date": datetime.now().isoformat(),
                "files_included": files_added,
                "models_structure": self._get_models_structure(models_dir),
            }

            metadata_json = json.dumps(metadata, indent=2)
            zipf.writestr("export_metadata.json", metadata_json)
            files_added += 1

        logger.info(
            f"Archive created with {files_added} files ({archive_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )

        # Upload to S3
        s3_key = f"exports/{archive_name}"
        logger.info(f"Uploading to s3://{self.bucket_name}/{s3_key}")

        try:
            with open(archive_path, "rb") as f:
                self.s3_client.upload_fileobj(f, self.bucket_name, s3_key)

            # Generate presigned URL for easy download
            download_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=86400,  # 24 hours
            )

            result = {
                "success": True,
                "archive_name": archive_name,
                "s3_key": s3_key,
                "s3_url": f"s3://{self.bucket_name}/{s3_key}",
                "download_url": download_url,
                "files_count": files_added,
                "archive_size_mb": archive_path.stat().st_size / 1024 / 1024,
                "metadata": metadata,
            }

            # Clean up local archive
            archive_path.unlink()
            logger.info("Local archive cleaned up")

            logger.info(f"✅ Export successful: {result['s3_url']}")
            return result

        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")
            # Clean up local archive on failure
            if archive_path.exists():
                archive_path.unlink()
            raise

    def _get_models_structure(self, models_dir: Path) -> Dict:
        """Get structure of models directory"""
        structure = {}

        for model_type_dir in models_dir.iterdir():
            if model_type_dir.is_dir():
                model_type = model_type_dir.name
                structure[model_type] = {}

                for symbol_dir in model_type_dir.iterdir():
                    if symbol_dir.is_dir():
                        symbol = symbol_dir.name
                        files = [f.name for f in symbol_dir.iterdir() if f.is_file()]
                        structure[model_type][symbol] = {
                            "files": files,
                            "file_count": len(files),
                        }

        return structure

    def list_exports(self) -> List[Dict]:
        """List previous exports in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix="exports/")

            exports = []
            for obj in response.get("Contents", []):
                exports.append(
                    {
                        "key": obj["Key"],
                        "size_mb": obj["Size"] / 1024 / 1024,
                        "last_modified": obj["LastModified"].isoformat(),
                        "download_url": self.s3_client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": self.bucket_name, "Key": obj["Key"]},
                            ExpiresIn=3600,  # 1 hour
                        ),
                    }
                )

            return sorted(exports, key=lambda x: x["last_modified"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list exports: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(description="Export trained models to S3")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/notebooks/bot/models"),
        help="Path to models directory",
    )
    parser.add_argument("--bucket", type=str, help="S3 bucket name (overrides env var)")
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Do not include validation stats in export",
    )
    parser.add_argument(
        "--list-exports", action="store_true", help="List previous exports and exit"
    )

    args = parser.parse_args()

    try:
        exporter = S3ModelExporter(bucket_name=args.bucket)

        if args.list_exports:
            exports = exporter.list_exports()
            if exports:
                logger.info(f"Found {len(exports)} previous exports:")
                for exp in exports[:10]:  # Show last 10
                    logger.info(
                        f"  {exp['key']} ({exp['size_mb']:.1f} MB) - {exp['last_modified']}"
                    )
            else:
                logger.info("No previous exports found")
            return

        result = exporter.export_models(
            models_dir=args.models_dir, include_validation=not args.no_validation
        )

        print("\n" + "=" * 60)
        print("🚀 EXPORT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Archive: {result['archive_name']}")
        print(f"S3 Location: {result['s3_url']}")
        print(f"Files: {result['files_count']}")
        print(f"Size: {result['archive_size_mb']:.1f} MB")
        print(f"\n🔗 Download URL (valid 24h):")
        print(result["download_url"])
        print("=" * 60)

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
