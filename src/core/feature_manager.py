"""Feature schema management and validation."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .base_service import BaseService
from .container import injectable
from .interfaces import IFeatureManager, ILogger, ValidationResult


@dataclass
class FeatureSchema:
    """Feature schema definition."""

    name: str
    dtype: str
    nullable: bool
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SchemaVersion:
    """Schema version information."""

    version: str
    created_at: datetime
    features: List[FeatureSchema]
    metadata: Dict[str, Any]


@injectable
class FeatureManager(BaseService, IFeatureManager):
    """Enhanced feature manager with schema validation and drift detection."""

    def __init__(self, logger: ILogger):
        super().__init__(logger)
        self._schemas: Dict[str, Dict[str, SchemaVersion]] = {}  # symbol -> model_type -> schema
        self._feature_mappings: Dict[str, Dict[str, Any]] = {}
        self._schema_dir = Path("src/schemas")
        self._feature_config_path = Path("feature_config.json")

    def initialize(self) -> bool:
        """Initialize feature manager."""
        if not super().initialize():
            return False

        try:
            self._schema_dir.mkdir(exist_ok=True)
            self._load_feature_mappings()
            self._load_all_schemas()
            return True
        except Exception as e:
            self._log_error("Failed to initialize feature manager", exception=e)
            return False

    def load_feature_schema(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Load feature schema for a specific symbol and model type."""
        self._ensure_initialized()

        schema_key = f"{symbol}_{model_type}"

        # Check if schema is cached
        if symbol in self._schemas and model_type in self._schemas[symbol]:
            schema_version = self._schemas[symbol][model_type]
            return self._schema_version_to_dict(schema_version)

        # Try to load from file
        schema_file = self._schema_dir / f"{schema_key}_schema.json"
        if schema_file.exists():
            schema = self._load_schema_file(schema_file)
            self._cache_schema(symbol, model_type, schema)
            return schema

        # Generate schema from feature mappings
        if schema_key in self._feature_mappings:
            schema = self._generate_schema_from_mapping(symbol, model_type)
            return schema

        # Return default schema
        return self._get_default_schema(symbol, model_type)

    def validate_features(self, features: pd.DataFrame, schema: Dict[str, Any]) -> ValidationResult:
        """Validate features against schema."""
        errors = []
        warnings = []
        metadata = {
            "feature_count": len(features.columns),
            "row_count": len(features),
            "validation_timestamp": datetime.now().isoformat(),
        }

        try:
            schema_features = schema.get("features", {})

            # Check for missing required features
            required_features = set(schema_features.keys())
            actual_features = set(features.columns)

            missing_features = required_features - actual_features
            extra_features = actual_features - required_features

            for feature in missing_features:
                errors.append(f"Missing required feature: {feature}")

            for feature in extra_features:
                warnings.append(f"Extra feature not in schema: {feature}")

            # Validate individual features
            for feature_name, feature_schema in schema_features.items():
                if feature_name in features.columns:
                    feature_errors = self._validate_feature_column(
                        features[feature_name], feature_name, feature_schema
                    )
                    errors.extend(feature_errors)

            # Check for data quality issues
            quality_issues = self._check_data_quality(features)
            warnings.extend(quality_issues)

            metadata["missing_features"] = list(missing_features)
            metadata["extra_features"] = list(extra_features)
            metadata["null_counts"] = features.isnull().sum().to_dict()

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def detect_schema_drift(
        self, current_features: pd.DataFrame, reference_schema: Dict[str, Any]
    ) -> ValidationResult:
        """Detect schema drift in features."""
        errors = []
        warnings = []
        metadata = {
            "drift_detection_timestamp": datetime.now().isoformat(),
            "reference_schema_version": reference_schema.get("version", "unknown"),
        }

        try:
            # Generate current schema
            current_schema = self._generate_schema_from_dataframe(current_features)

            # Compare schemas
            drift_results = self._compare_schemas(reference_schema, current_schema)

            # Categorize drift types
            for drift in drift_results:
                if drift["severity"] == "high":
                    errors.append(drift["message"])
                else:
                    warnings.append(drift["message"])

            metadata["drift_count"] = len(drift_results)
            metadata["high_severity_drifts"] = len(
                [d for d in drift_results if d["severity"] == "high"]
            )
            metadata["current_feature_count"] = len(current_features.columns)
            metadata["reference_feature_count"] = len(reference_schema.get("features", {}))

        except Exception as e:
            errors.append(f"Drift detection error: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def save_schema(
        self, symbol: str, model_type: str, features: pd.DataFrame, version: str = None
    ) -> bool:
        """Save feature schema from DataFrame."""
        try:
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")

            schema = self._generate_schema_from_dataframe(features, version)
            schema_key = f"{symbol}_{model_type}"
            schema_file = self._schema_dir / f"{schema_key}_schema.json"

            with open(schema_file, "w") as f:
                json.dump(schema, f, indent=2, default=str)

            self._cache_schema(symbol, model_type, schema)
            self._log_info(f"Saved schema for {schema_key} version {version}")
            return True

        except Exception as e:
            self._log_error(f"Failed to save schema for {symbol}_{model_type}", exception=e)
            return False

    def _load_feature_mappings(self):
        """Load feature mappings from configuration."""
        if self._feature_config_path.exists():
            try:
                with open(self._feature_config_path, "r") as f:
                    self._feature_mappings = json.load(f)
                self._log_info("Loaded feature mappings")
            except Exception as e:
                self._log_warning("Failed to load feature mappings", exception=e)
                self._feature_mappings = {}

    def _load_all_schemas(self):
        """Load all available schemas."""
        if not self._schema_dir.exists():
            return

        for schema_file in self._schema_dir.glob("*_schema.json"):
            try:
                schema = self._load_schema_file(schema_file)
                # Parse symbol and model_type from filename
                filename = schema_file.stem.replace("_schema", "")
                parts = filename.split("_")
                if len(parts) >= 2:
                    symbol = parts[0]
                    model_type = "_".join(parts[1:])
                    self._cache_schema(symbol, model_type, schema)
            except Exception as e:
                self._log_warning(f"Failed to load schema file {schema_file}", exception=e)

    def _load_schema_file(self, schema_file: Path) -> Dict[str, Any]:
        """Load schema from JSON file."""
        with open(schema_file, "r") as f:
            return json.load(f)

    def _cache_schema(self, symbol: str, model_type: str, schema: Dict[str, Any]):
        """Cache schema in memory."""
        if symbol not in self._schemas:
            self._schemas[symbol] = {}

        # Convert dict to SchemaVersion object
        features = []
        for name, feature_def in schema.get("features", {}).items():
            features.append(
                FeatureSchema(
                    name=name,
                    dtype=feature_def.get("dtype", "float64"),
                    nullable=feature_def.get("nullable", True),
                    min_value=feature_def.get("min_value"),
                    max_value=feature_def.get("max_value"),
                    description=feature_def.get("description"),
                    tags=feature_def.get("tags", []),
                )
            )

        schema_version = SchemaVersion(
            version=schema.get("version", "1.0.0"),
            created_at=datetime.fromisoformat(schema.get("created_at", datetime.now().isoformat())),
            features=features,
            metadata=schema.get("metadata", {}),
        )

        self._schemas[symbol][model_type] = schema_version

    def _generate_schema_from_mapping(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Generate schema from feature mapping."""
        schema_key = f"{symbol}_{model_type}"
        mapping = self._feature_mappings.get(schema_key, {})

        features = {}
        for feature_name in mapping.get("features", []):
            features[feature_name] = {
                "dtype": "float64",
                "nullable": True,
                "description": f"Feature for {symbol} {model_type} model",
            }

        return {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "symbol": symbol,
            "model_type": model_type,
            "features": features,
            "metadata": {"generated_from_mapping": True},
        }

    def _generate_schema_from_dataframe(
        self, df: pd.DataFrame, version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """Generate schema from DataFrame."""
        features = {}

        for col in df.columns:
            dtype_str = str(df[col].dtype)
            features[col] = {
                "dtype": dtype_str,
                "nullable": df[col].isnull().any(),
                "min_value": (
                    float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None
                ),
                "max_value": (
                    float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None
                ),
                "null_count": int(df[col].isnull().sum()),
                "description": f"Auto-generated schema for {col}",
            }

        return {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "features": features,
            "metadata": {
                "auto_generated": True,
                "row_count": len(df),
                "feature_count": len(df.columns),
            },
        }

    def _get_default_schema(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Get default schema when none is available."""
        return {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "symbol": symbol,
            "model_type": model_type,
            "features": {},
            "metadata": {"default_schema": True},
        }

    def _validate_feature_column(
        self, series: pd.Series, feature_name: str, feature_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate a single feature column."""
        errors = []

        # Check data type
        expected_dtype = feature_schema.get("dtype", "float64")
        if str(series.dtype) != expected_dtype:
            errors.append(
                f"Feature {feature_name}: expected dtype {expected_dtype}, got {series.dtype}"
            )

        # Check nullability
        if not feature_schema.get("nullable", True) and series.isnull().any():
            errors.append(
                f"Feature {feature_name}: contains null values but schema specifies non-nullable"
            )

        # Check value ranges
        if pd.api.types.is_numeric_dtype(series):
            min_val = feature_schema.get("min_value")
            max_val = feature_schema.get("max_value")

            if min_val is not None and series.min() < min_val:
                errors.append(
                    f"Feature {feature_name}: minimum value {series.min()} below schema minimum {min_val}"
                )

            if max_val is not None and series.max() > max_val:
                errors.append(
                    f"Feature {feature_name}: maximum value {series.max()} above schema maximum {max_val}"
                )

        return errors

    def _check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check for data quality issues."""
        warnings = []

        # Check for high null percentages
        null_percentages = df.isnull().mean()
        high_null_features = null_percentages[null_percentages > 0.5].index.tolist()

        for feature in high_null_features:
            warnings.append(f"Feature {feature} has {null_percentages[feature]:.1%} null values")

        # Check for constant features
        for col in df.columns:
            if df[col].nunique() <= 1:
                warnings.append(f"Feature {col} appears to be constant")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                warnings.append(f"Feature {col} contains infinite values")

        return warnings

    def _compare_schemas(
        self, reference_schema: Dict[str, Any], current_schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare two schemas and identify drifts."""
        drifts = []

        ref_features = set(reference_schema.get("features", {}).keys())
        cur_features = set(current_schema.get("features", {}).keys())

        # Missing features
        missing = ref_features - cur_features
        for feature in missing:
            drifts.append(
                {
                    "type": "missing_feature",
                    "feature": feature,
                    "message": f"Feature {feature} missing from current data",
                    "severity": "high",
                }
            )

        # New features
        new = cur_features - ref_features
        for feature in new:
            drifts.append(
                {
                    "type": "new_feature",
                    "feature": feature,
                    "message": f"New feature {feature} not in reference schema",
                    "severity": "medium",
                }
            )

        # Type changes
        common_features = ref_features & cur_features
        for feature in common_features:
            ref_dtype = reference_schema["features"][feature].get("dtype")
            cur_dtype = current_schema["features"][feature].get("dtype")

            if ref_dtype != cur_dtype:
                drifts.append(
                    {
                        "type": "dtype_change",
                        "feature": feature,
                        "message": f"Feature {feature} dtype changed from {ref_dtype} to {cur_dtype}",
                        "severity": "high",
                    }
                )

        return drifts

    def _schema_version_to_dict(self, schema_version: SchemaVersion) -> Dict[str, Any]:
        """Convert SchemaVersion to dictionary."""
        features = {}
        for feature in schema_version.features:
            features[feature.name] = {
                "dtype": feature.dtype,
                "nullable": feature.nullable,
                "min_value": feature.min_value,
                "max_value": feature.max_value,
                "description": feature.description,
                "tags": feature.tags,
            }

        return {
            "version": schema_version.version,
            "created_at": schema_version.created_at.isoformat(),
            "features": features,
            "metadata": schema_version.metadata,
        }
