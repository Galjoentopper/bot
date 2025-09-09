#!/usr/bin/env python3
"""
Model Packaging System for Distributed Trading Bot

This module provides comprehensive model packaging, versioning, and transfer utilities
for training models on one machine and deploying them on another.

Features:
- Model packaging with metadata and dependencies
- Version management and compatibility checking
- Easy transfer between machines
- Validation and integrity checks
- Rollback capabilities
"""

import hashlib
import json
import logging
import os
import pickle
import platform
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import torch and lightgbm for model info extraction
try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without torch
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import lightgbm as lgb  # type: ignore

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without lightgbm
    lgb = None  # type: ignore
    LIGHTGBM_AVAILABLE = False


@dataclass
class ModelMetadata:
    """Comprehensive metadata for packaged models"""

    model_name: str
    model_type: str  # 'gru', 'lightgbm', 'ppo'
    symbol: str
    version: str
    created_at: str
    training_machine: str
    python_version: str
    dependencies: Dict[str, str]
    model_hash: str
    preprocessor_hash: Optional[str]
    feature_columns: List[str]
    target_type: str
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    file_paths: Dict[
        str, Optional[str]
    ]  # relative paths within package (some optional like preprocessor)
    compatibility_info: Dict[str, Any]
    notes: str = ""


class ModelPackager:
    """Handles model packaging, versioning, and transfer operations"""

    def __init__(self, base_dir: str = "models", package_dir: str = "model_packages"):
        self.base_dir = Path(base_dir)
        self.package_dir = Path(package_dir)
        self.package_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.package_dir / "exports").mkdir(exist_ok=True)
        (self.package_dir / "imports").mkdir(exist_ok=True)
        (self.package_dir / "backups").mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def import_package(self, package_path: str) -> str:
        """Import a packaged model zip into a temporary extraction directory.

        Returns path to extracted directory containing model artifacts.
        Safe no-op if already extracted previously (creates hash-based folder).
        """
        package = Path(package_path)
        if not package.exists():
            raise FileNotFoundError(f"Package not found: {package}")
        extract_root = self.package_dir / "imports"
        extract_root.mkdir(exist_ok=True)
        # Use hash of file for stable extraction folder
        h = hashlib.sha256()
        with open(package, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        dest = extract_root / h.hexdigest()[:16]
        if dest.exists():
            return str(dest)
        dest.mkdir()
        try:
            with zipfile.ZipFile(package, "r") as zf:
                zf.extractall(dest)
        except Exception as e:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(f"Failed to extract package {package}: {e}")
        return str(dest)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_system_info(self) -> Dict[str, str]:
        """Get system and dependency information"""
        deps = {
            "python": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
        }

        # Add ML library versions if available
        if TORCH_AVAILABLE and torch is not None:
            deps["torch"] = getattr(torch, "__version__", "unknown")
        if LIGHTGBM_AVAILABLE and lgb is not None:
            deps["lightgbm"] = getattr(lgb, "__version__", "unknown")

        try:
            import numpy as np

            deps["numpy"] = np.__version__
        except ImportError:
            pass

        try:
            import pandas as pd

            deps["pandas"] = pd.__version__
        except ImportError:
            pass

        try:
            import sklearn

            deps["scikit-learn"] = sklearn.__version__
        except ImportError:
            pass

        return deps

    def _extract_model_info(self, model_path: Path, model_type: str) -> Dict[str, Any]:
        """Extract model-specific information"""
        info: Dict[str, Any] = {}

        if (
            model_type == "gru"
            and TORCH_AVAILABLE
            and torch is not None
            and model_path.suffix == ".pt"
        ):
            try:
                model_data = torch.load(model_path, map_location="cpu")  # type: ignore[attr-defined]
                if isinstance(model_data, dict):
                    info["model_state_keys"] = list(model_data.keys())
                    if "model_state_dict" in model_data:
                        state_dict = model_data["model_state_dict"]
                        if isinstance(state_dict, dict):
                            info["layer_info"] = {
                                k: str(getattr(v, "shape", "")) for k, v in state_dict.items()
                            }
            except Exception as e:  # pragma: no cover - best effort
                self.logger.warning(f"Could not extract GRU model info: {e}")

        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE and model_path.suffix == ".pkl":
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                if hasattr(model, "num_trees"):
                    info["num_trees"] = model.num_trees()
                if hasattr(model, "num_feature"):
                    info["num_features"] = model.num_feature()
            except Exception as e:  # pragma: no cover - best effort
                self.logger.warning(f"Could not extract LightGBM model info: {e}")

        return info

    def package_model(
        self,
        model_path: Union[str, Path],
        model_type: str,
        symbol: str,
        preprocessor_path: Optional[Union[str, Path]] = None,
        feature_columns: Optional[List[str]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        notes: str = "",
    ) -> str:
        """Package a model with all its dependencies and metadata."""

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path_obj}")

        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"

        package_name = f"{model_type}_{symbol}_{version}"
        package_path = self.package_dir / "exports" / f"{package_name}.zip"

        model_hash = self._calculate_file_hash(model_path_obj)
        preprocessor_hash = None
        if preprocessor_path:
            preprocessor_path_obj = Path(preprocessor_path)
            if preprocessor_path_obj.exists():
                preprocessor_hash = self._calculate_file_hash(preprocessor_path_obj)
        model_info = self._extract_model_info(model_path_obj, model_type)

        metadata = ModelMetadata(
            model_name=package_name,
            model_type=model_type,
            symbol=symbol,
            version=version,
            created_at=datetime.now().isoformat(),
            training_machine=platform.node(),
            python_version=sys.version,
            dependencies=self._get_system_info(),
            model_hash=model_hash,
            preprocessor_hash=preprocessor_hash,
            feature_columns=feature_columns or [],
            target_type=(
                training_config.get("target_type", "unknown") if training_config else "unknown"
            ),
            performance_metrics=performance_metrics or {},
            training_config=training_config or {},
            file_paths={
                "model": f"model{model_path_obj.suffix}",
                "preprocessor": "preprocessor.pkl" if preprocessor_hash else None,
                "metadata": "metadata.json",
            },
            compatibility_info=model_info,
            notes=notes,
        )

        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(model_path_obj, f"model{model_path_obj.suffix}")
            if preprocessor_path and Path(preprocessor_path).exists():
                zipf.write(str(preprocessor_path), "preprocessor.pkl")
            metadata_json = json.dumps(asdict(metadata), indent=2, default=str)
            zipf.writestr("metadata.json", metadata_json)
            zipf.writestr("README.md", self._generate_readme(metadata))

        self.logger.info(f"Model packaged successfully: {package_path}")
        return str(package_path)

    def _generate_readme(self, metadata: ModelMetadata) -> str:
        """Generate README content for the model package"""
        readme = f"""# {metadata.model_name}

## Model Information
- **Type**: {metadata.model_type.upper()}
- **Symbol**: {metadata.symbol}
- **Version**: {metadata.version}
- **Created**: {metadata.created_at}
- **Training Machine**: {metadata.training_machine}

## Performance Metrics
{json.dumps(metadata.performance_metrics, indent=2)}

## Dependencies
{json.dumps(metadata.dependencies, indent=2)}

## Feature Columns ({len(metadata.feature_columns)})
{json.dumps(metadata.feature_columns, indent=2)}

## Training Configuration
{json.dumps(metadata.training_config, indent=2)}

## Notes
{metadata.notes}

## Usage
This package can be imported using the ModelPackager.import_model() method.
"""
        return readme

    def list_packages(
        self, model_type: Optional[str] = None, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available model packages"""
        packages = []
        export_dir = self.package_dir / "exports"

        for package_file in export_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(package_file, "r") as zipf:
                    metadata_content = zipf.read("metadata.json").decode("utf-8")
                    metadata = json.loads(metadata_content)

                    # Filter by criteria
                    if model_type and metadata.get("model_type") != model_type:
                        continue
                    if symbol and metadata.get("symbol") != symbol:
                        continue

                    packages.append(
                        {
                            "file": str(package_file),
                            "name": metadata.get("model_name"),
                            "type": metadata.get("model_type"),
                            "symbol": metadata.get("symbol"),
                            "version": metadata.get("version"),
                            "created_at": metadata.get("created_at"),
                            "size_mb": round(package_file.stat().st_size / (1024 * 1024), 2),
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Could not read package {package_file}: {e}")

        return sorted(packages, key=lambda x: x["created_at"], reverse=True)

    def import_model(
        self,
        package_path: Union[str, Path],
        target_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """Import a model package and extract its contents."""
        package_path_obj = Path(package_path)
        if not package_path_obj.exists():
            raise FileNotFoundError(f"Package not found: {package_path_obj}")

        with zipfile.ZipFile(package_path_obj, "r") as zipf:
            metadata_content = zipf.read("metadata.json").decode("utf-8")
            metadata = json.loads(metadata_content)

        target_dir_path = (
            self.base_dir / metadata["model_type"] / metadata["symbol"]
            if target_dir is None
            else Path(target_dir)
        )
        target_dir_path.mkdir(parents=True, exist_ok=True)

        extracted_files: Dict[str, str] = {}
        with zipfile.ZipFile(package_path_obj, "r") as zipf:
            for file_info in zipf.filelist:
                if file_info.filename in ["metadata.json", "README.md"]:
                    continue
                extracted_path = target_dir_path / file_info.filename
                with open(extracted_path, "wb") as f:
                    f.write(zipf.read(file_info.filename))
                extracted_files[file_info.filename] = str(extracted_path)

        metadata_path = target_dir_path / "imported_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        extracted_files["metadata"] = str(metadata_path)

        self.logger.info(f"Model imported successfully to: {target_dir_path}")
        return extracted_files

    def validate_package(self, package_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate a model package for integrity and compatibility."""
        issues: List[str] = []
        package_path_obj = Path(package_path)
        if not package_path_obj.exists():
            return False, ["Package file does not exist"]
        try:
            with zipfile.ZipFile(package_path_obj, "r") as zipf:
                files = zipf.namelist()
                if "metadata.json" not in files:
                    issues.append("Missing metadata.json")
                    return False, issues
                metadata = json.loads(zipf.read("metadata.json").decode("utf-8"))
                model_file = metadata.get("file_paths", {}).get("model")
                if not model_file or model_file not in files:
                    issues.append("Model file missing from package")
                if model_file and model_file in files:
                    import tempfile

                    with tempfile.NamedTemporaryFile() as tmp:
                        tmp.write(zipf.read(model_file))
                        tmp.flush()
                        actual_hash = self._calculate_file_hash(Path(tmp.name))
                        expected_hash = metadata.get("model_hash")
                        if expected_hash and actual_hash != expected_hash:
                            issues.append("Model file hash mismatch - file may be corrupted")
                current_deps = self._get_system_info()
                package_deps = metadata.get("dependencies", {})
                if "python" in package_deps:
                    package_py_version = package_deps["python"].split()[0]
                    current_py_version = current_deps["python"].split()[0]
                    if package_py_version != current_py_version:
                        issues.append(
                            f"Python version mismatch: package={package_py_version}, current={current_py_version}"
                        )
                for dep in ["torch", "lightgbm", "numpy", "pandas"]:
                    if dep in package_deps and dep not in current_deps:
                        issues.append(f"Missing dependency: {dep}")
        except Exception as e:  # pragma: no cover - best effort
            issues.append(f"Error validating package: {e}")
        return len(issues) == 0, issues

    def create_transfer_bundle(
        self,
        model_types: List[str],
        symbols: List[str],
        output_path: Union[str, Path],
    ) -> str:
        """Create a bundle of multiple models for easy transfer."""
        output_path_path = Path(output_path)
        packages = self.list_packages()
        selected_packages = [
            pkg for pkg in packages if pkg["type"] in model_types and pkg["symbol"] in symbols
        ]
        if not selected_packages:
            raise ValueError("No packages found matching criteria")
        with zipfile.ZipFile(output_path_path, "w", zipfile.ZIP_DEFLATED) as bundle_zip:
            bundle_info = {
                "created_at": datetime.now().isoformat(),
                "packages": [],
                "config_files": [],
                "validation_scripts": [],
                "requirements_included": False,
                "transfer_instructions": "Extract this bundle and use ModelPackager.import_model() for each package",
            }
            for pkg in selected_packages:
                pkg_path = Path(pkg["file"])
                bundle_zip.write(pkg_path, pkg_path.name)
                bundle_info["packages"].append(
                    {
                        "name": pkg["name"],
                        "type": pkg["type"],
                        "symbol": pkg["symbol"],
                        "file": pkg_path.name,
                    }
                )
            project_root = (
                Path(self.base_dir).parent.parent
                if "models" in str(self.base_dir)
                else Path(self.base_dir).parent
            )
            config_dir = project_root / "src" / "config"
            for config_file in ["config_trading.yaml", "config_training.yaml"]:
                config_path = config_dir / config_file
                if config_path.exists():
                    bundle_zip.write(config_path, f"config/{config_file}")
                    bundle_info["config_files"].append(f"config/{config_file}")
            for src_path, bundle_path in [
                ("validate_models.bat", "validate_models.bat"),
                ("scripts/validate_models.py", "scripts/validate_models.py"),
            ]:
                full_path = project_root / src_path
                if full_path.exists():
                    bundle_zip.write(full_path, bundle_path)
                    bundle_info["validation_scripts"].append(bundle_path)
            requirements_path = project_root / "requirements.txt"
            if requirements_path.exists():
                bundle_zip.write(requirements_path, "requirements.txt")
                bundle_info["requirements_included"] = True
            bundle_zip.writestr("bundle_info.json", json.dumps(bundle_info, indent=2))
            bundle_zip.writestr("import_models.py", self._generate_transfer_script())
        self.logger.info(
            "Transfer bundle created: %s with %d models",
            output_path_path,
            len(selected_packages),
        )
        return str(output_path_path)

    def _generate_transfer_script(self) -> str:
        """Generate a script to help with model importing"""
        script = '''#!/usr/bin/env python3
"""
Model Import Script
Automatically import all models from a transfer bundle
"""

import json
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.model_packaging import ModelPackager

def main():
    packager = ModelPackager()

    # Read bundle info
    with open("bundle_info.json", "r") as f:
        bundle_info = json.load(f)

    print(f"Importing {len(bundle_info['packages'])} models...")

    for pkg_info in bundle_info["packages"]:
        print(f"Importing {pkg_info['name']}...")
        try:
            result = packager.import_model(pkg_info["file"])
            print(f"  ✓ Imported to: {list(result.values())[0]}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("Import complete!")

if __name__ == "__main__":
    main()
'''
        return script
