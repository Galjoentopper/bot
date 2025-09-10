"""
Data Validation Module
======================

Comprehensive validation system to prevent training failures due to insufficient
or poor quality data. Implements pre-training checks and data quality metrics.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation failures."""

    pass


class TradingDataValidator:
    """
    Comprehensive data validator for trading model training.

    Validates data quality, quantity, and suitability for financial modeling.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data validator.

        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or self._get_default_config()
        logger.info("TradingDataValidator initialized")

    def _get_default_config(self) -> Dict:
        """Get default validation configuration."""
        return {
            "min_samples_absolute": 5000,  # Minimum samples for any training
            "min_samples_gru": 10000,  # Minimum for GRU (sequence data)
            "min_samples_ppo": 5000,  # Minimum for PPO (episodes)
            "min_samples_lgbm": 2000,  # Minimum for LightGBM
            "max_missing_ratio": 0.05,  # Max 5% missing data
            "min_unique_values": 10,  # Min unique values per feature
            "min_variance_threshold": 1e-8,  # Min variance for features
            "max_correlation_threshold": 0.95,  # Max correlation between features
            "min_temporal_coverage_hours": 168,  # Min 1 week of data
            "required_columns": ["open", "high", "low", "close", "volume"],
            "price_sanity_checks": True,
            "volume_sanity_checks": True,
            "temporal_continuity_check": True,
        }

    def validate_for_training(
        self, data: pd.DataFrame, symbol: str, model_type: str = "all"
    ) -> Dict[str, Any]:
        """
        Comprehensive validation for training data.

        Args:
            data: Training dataset
            symbol: Trading symbol
            model_type: Type of model ('gru', 'lgbm', 'ppo', 'all')

        Returns:
            Validation results dictionary

        Raises:
            DataValidationError: If critical validation checks fail
        """
        logger.info(f"🔍 Validating data for {symbol} ({model_type})")

        results = {
            "symbol": symbol,
            "model_type": model_type,
            "total_samples": len(data),
            "validation_timestamp": datetime.now().isoformat(),
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "critical_failures": [],
            "is_valid": True,
        }

        try:
            # 1. Basic structure validation
            self._validate_basic_structure(data, results)

            # 2. Sample count validation
            self._validate_sample_count(data, model_type, results)

            # 3. Data quality validation
            self._validate_data_quality(data, results)

            # 4. Financial data sanity checks
            self._validate_financial_sanity(data, results)

            # 5. Temporal validation
            self._validate_temporal_properties(data, results)

            # 6. Feature validation
            self._validate_features(data, results)

            # Determine overall validation result
            results["is_valid"] = len(results["critical_failures"]) == 0

            if results["is_valid"]:
                logger.info(f"✅ Data validation passed for {symbol}")
            else:
                logger.error(
                    f"❌ Data validation failed for {symbol}: {results['critical_failures']}"
                )

        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            results["critical_failures"].append(f"Validation error: {str(e)}")
            results["is_valid"] = False

        return results

    def _validate_basic_structure(self, data: pd.DataFrame, results: Dict):
        """Validate basic DataFrame structure."""
        check_name = "basic_structure"

        try:
            # Check if DataFrame is not empty
            if data.empty:
                results["critical_failures"].append("Dataset is empty")
                return

            # Check required columns
            missing_cols = [
                col for col in self.config["required_columns"] if col not in data.columns
            ]
            if missing_cols:
                results["critical_failures"].append(f"Missing required columns: {missing_cols}")
                return

            # Check index type
            if not isinstance(data.index, pd.DatetimeIndex):
                results["warnings"].append(
                    "Index is not DatetimeIndex - temporal analysis may be limited"
                )

            results["checks_passed"].append(check_name)
            logger.debug(f"✅ {check_name} validation passed")

        except Exception as e:
            results["checks_failed"].append(f"{check_name}: {str(e)}")

    def _validate_sample_count(self, data: pd.DataFrame, model_type: str, results: Dict):
        """Validate minimum sample requirements."""
        check_name = "sample_count"

        try:
            sample_count = len(data)

            # Get minimum requirements based on model type
            min_required = self._get_min_samples_for_model(model_type)

            if sample_count < min_required:
                results["critical_failures"].append(
                    f"Insufficient samples: {sample_count} < {min_required} (required for {model_type})"
                )
                return

            # Check for reasonable upper bound (detect data issues)
            max_reasonable = 50000  # ~1 year of 30min data
            if sample_count > max_reasonable:
                results["warnings"].append(
                    f"Very large dataset: {sample_count} samples - verify data quality"
                )

            results["total_samples"] = sample_count
            results["min_required"] = min_required
            results["checks_passed"].append(check_name)
            logger.debug(f"✅ {check_name}: {sample_count} >= {min_required} samples")

        except Exception as e:
            results["checks_failed"].append(f"{check_name}: {str(e)}")

    def _get_min_samples_for_model(self, model_type: str) -> int:
        """Get minimum sample requirement for model type."""
        model_mins = {
            "gru": self.config["min_samples_gru"],
            "lgbm": self.config["min_samples_lgbm"],
            "lightgbm": self.config["min_samples_lgbm"],
            "ppo": self.config["min_samples_ppo"],
            "all": self.config["min_samples_absolute"],
        }
        return model_mins.get(model_type.lower(), self.config["min_samples_absolute"])

    def _validate_data_quality(self, data: pd.DataFrame, results: Dict):
        """Validate data quality metrics."""
        check_name = "data_quality"

        try:
            # Check missing data ratio
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 1.0

            results["missing_ratio"] = missing_ratio

            if missing_ratio > self.config["max_missing_ratio"]:
                results["critical_failures"].append(
                    f"Too much missing data: {missing_ratio:.1%} > {self.config['max_missing_ratio']:.1%}"
                )
                return

            # Check for completely missing columns
            completely_missing = data.columns[data.isnull().all()].tolist()
            if completely_missing:
                results["warnings"].append(f"Completely missing columns: {completely_missing}")

            # Check for constant columns
            constant_cols = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].nunique() <= 1:
                    constant_cols.append(col)

            if constant_cols:
                results["warnings"].append(f"Constant columns detected: {constant_cols}")

            results["checks_passed"].append(check_name)
            logger.debug(f"✅ {check_name}: {missing_ratio:.1%} missing data")

        except Exception as e:
            results["checks_failed"].append(f"{check_name}: {str(e)}")

    def _validate_financial_sanity(self, data: pd.DataFrame, results: Dict):
        """Validate financial data sanity."""
        check_name = "financial_sanity"

        try:
            # Price sanity checks
            if self.config["price_sanity_checks"]:
                price_cols = ["open", "high", "low", "close"]
                available_price_cols = [col for col in price_cols if col in data.columns]

                for col in available_price_cols:
                    # Check for negative prices
                    if (data[col] <= 0).any():
                        results["critical_failures"].append(f"Non-positive prices found in {col}")
                        return

                    # Check for extreme price movements (>50% in one period)
                    price_changes = data[col].pct_change().abs()
                    extreme_moves = (price_changes > 0.5).sum()
                    if extreme_moves > len(data) * 0.01:  # More than 1% extreme moves
                        results["warnings"].append(
                            f"Many extreme price movements in {col}: {extreme_moves} instances"
                        )

                # OHLC consistency checks
                if all(col in data.columns for col in ["open", "high", "low", "close"]):
                    # High should be >= max(open, close)
                    invalid_high = (data["high"] < np.maximum(data["open"], data["close"])).sum()
                    # Low should be <= min(open, close)
                    invalid_low = (data["low"] > np.minimum(data["open"], data["close"])).sum()

                    if invalid_high > 0:
                        results["warnings"].append(f"Invalid high prices: {invalid_high} instances")
                    if invalid_low > 0:
                        results["warnings"].append(f"Invalid low prices: {invalid_low} instances")

            # Volume sanity checks
            if self.config["volume_sanity_checks"] and "volume" in data.columns:
                # Check for negative volume
                if (data["volume"] < 0).any():
                    results["critical_failures"].append("Negative volume found")
                    return

                # Check for zero volume periods
                zero_volume_ratio = (data["volume"] == 0).mean()
                if zero_volume_ratio > 0.1:  # More than 10% zero volume
                    results["warnings"].append(f"High zero volume ratio: {zero_volume_ratio:.1%}")

            results["checks_passed"].append(check_name)
            logger.debug(f"✅ {check_name} validation passed")

        except Exception as e:
            results["checks_failed"].append(f"{check_name}: {str(e)}")

    def _validate_temporal_properties(self, data: pd.DataFrame, results: Dict):
        """Validate temporal properties of the data."""
        check_name = "temporal_properties"

        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                results["warnings"].append(
                    "Cannot validate temporal properties - index not datetime"
                )
                return

            # Check temporal coverage
            time_span = data.index.max() - data.index.min()
            min_coverage = timedelta(hours=self.config["min_temporal_coverage_hours"])

            if time_span < min_coverage:
                results["critical_failures"].append(
                    f"Insufficient temporal coverage: {time_span} < {min_coverage}"
                )
                return

            # Check for large gaps in data
            if self.config["temporal_continuity_check"]:
                time_diffs = data.index.to_series().diff()
                median_interval = time_diffs.median()
                large_gaps = (time_diffs > median_interval * 3).sum()  # Gaps > 3x normal

                if large_gaps > len(data) * 0.05:  # More than 5% large gaps
                    results["warnings"].append(f"Many large time gaps: {large_gaps} instances")

            # Check for duplicate timestamps
            duplicate_times = data.index.duplicated().sum()
            if duplicate_times > 0:
                results["warnings"].append(f"Duplicate timestamps: {duplicate_times}")

            results["temporal_span"] = str(time_span)
            results["checks_passed"].append(check_name)
            logger.debug(f"✅ {check_name}: {time_span} coverage")

        except Exception as e:
            results["checks_failed"].append(f"{check_name}: {str(e)}")

    def _validate_features(self, data: pd.DataFrame, results: Dict):
        """Validate feature quality for machine learning."""
        check_name = "feature_validation"

        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            # Check feature variance
            low_variance_features = []
            for col in numeric_cols:
                if data[col].var() < self.config["min_variance_threshold"]:
                    low_variance_features.append(col)

            if low_variance_features:
                results["warnings"].append(
                    f"Low variance features: {len(low_variance_features)} features"
                )

            # Check for high correlation (potential multicollinearity)
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr().abs()
                # Get upper triangle of correlation matrix
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                high_corr_pairs = []
                for col in upper_triangle.columns:
                    high_corr = upper_triangle[col][
                        upper_triangle[col] > self.config["max_correlation_threshold"]
                    ]
                    for idx in high_corr.index:
                        high_corr_pairs.append((col, idx, high_corr[idx]))

                if high_corr_pairs:
                    results["warnings"].append(
                        f"High correlation features: {len(high_corr_pairs)} pairs"
                    )

            results["feature_count"] = len(numeric_cols)
            results["checks_passed"].append(check_name)
            logger.debug(f"✅ {check_name}: {len(numeric_cols)} features validated")

        except Exception as e:
            results["checks_failed"].append(f"{check_name}: {str(e)}")

    def get_validation_summary(self, validation_results: List[Dict]) -> Dict:
        """
        Generate summary of validation results across multiple datasets.

        Args:
            validation_results: List of validation result dictionaries

        Returns:
            Summary dictionary
        """
        summary = {
            "total_datasets": len(validation_results),
            "valid_datasets": sum(1 for r in validation_results if r["is_valid"]),
            "invalid_datasets": sum(1 for r in validation_results if not r["is_valid"]),
            "symbols_validated": [r["symbol"] for r in validation_results],
            "symbols_valid": [r["symbol"] for r in validation_results if r["is_valid"]],
            "symbols_invalid": [r["symbol"] for r in validation_results if not r["is_valid"]],
            "common_issues": [],
            "recommendations": [],
        }

        # Analyze common issues
        all_failures = []
        for result in validation_results:
            all_failures.extend(result["critical_failures"])

        # Count failure types
        failure_counts = {}
        for failure in all_failures:
            failure_type = failure.split(":")[0] if ":" in failure else failure
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

        summary["common_issues"] = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)

        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(summary, validation_results)

        return summary

    def _generate_recommendations(self, summary: Dict, results: List[Dict]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if summary["invalid_datasets"] > 0:
            recommendations.append(
                f"🔍 {summary['invalid_datasets']} datasets failed validation - review data quality"
            )

        # Check for common sample size issues
        insufficient_samples = [
            r for r in results if "Insufficient samples" in str(r.get("critical_failures", []))
        ]
        if insufficient_samples:
            recommendations.append(
                "📈 Increase data collection period or reduce training frequency for symbols with insufficient data"
            )

        # Check for missing data issues
        missing_data_issues = [r for r in results if r.get("missing_ratio", 0) > 0.01]
        if missing_data_issues:
            recommendations.append(
                "🔧 Implement data imputation or filtering for datasets with missing values"
            )

        return recommendations


def validate_training_data(
    datasets: Dict[str, pd.DataFrame], model_type: str = "all", config: Optional[Dict] = None
) -> Tuple[Dict[str, Dict], Dict]:
    """
    Convenience function to validate multiple datasets for training.

    Args:
        datasets: Dictionary of {symbol: dataframe}
        model_type: Type of model to validate for
        config: Validation configuration

    Returns:
        Tuple of (individual_results, summary)
    """
    validator = TradingDataValidator(config)

    results = {}
    for symbol, data in datasets.items():
        try:
            results[symbol] = validator.validate_for_training(data, symbol, model_type)
        except Exception as e:
            logger.error(f"Validation failed for {symbol}: {e}")
            results[symbol] = {
                "symbol": symbol,
                "is_valid": False,
                "critical_failures": [f"Validation error: {str(e)}"],
                "checks_passed": [],
                "checks_failed": ["validation_error"],
            }

    summary = validator.get_validation_summary(list(results.values()))

    return results, summary


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=1000, freq="30T")
    sample_data = pd.DataFrame(
        {
            "open": np.random.uniform(100, 110, 1000),
            "high": np.random.uniform(105, 115, 1000),
            "low": np.random.uniform(95, 105, 1000),
            "close": np.random.uniform(100, 110, 1000),
            "volume": np.random.uniform(1000, 10000, 1000),
        },
        index=dates,
    )

    # Validate
    validator = TradingDataValidator()
    result = validator.validate_for_training(sample_data, "BTCEUR", "gru")

    print("Validation Result:")
    print(f"Valid: {result['is_valid']}")
    print(f"Checks passed: {result['checks_passed']}")
    print(f"Critical failures: {result['critical_failures']}")
