"""
Enterprise Data Quality Validation Framework
==========================================

Comprehensive data quality validation system for financial time series data.
Provides robust validation, cleaning, and monitoring capabilities.
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Data validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of data validation check"""
    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    affected_rows: int = 0
    suggested_action: str = ""


class MarketDataValidator:
    """
    Enterprise-grade financial data validator with comprehensive checks
    for OHLCV data quality, mathematical stability, and market reasonableness.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.validation_results: List[ValidationResult] = []
        logger.info("MarketDataValidator initialized for enterprise data quality")

    def _get_default_config(self) -> Dict:
        """Default validation configuration with market-specific parameters"""
        return {
            # Market-specific price ranges (EUR pairs)
            "price_ranges": {
                "BTCEUR": {"min": 15000, "max": 150000, "typical_volatility": 0.05},
                "ETHEUR": {"min": 800, "max": 12000, "typical_volatility": 0.06},
                "ADAEUR": {"min": 0.15, "max": 6.0, "typical_volatility": 0.08},
                "DOTEUR": {"min": 3.0, "max": 80.0, "typical_volatility": 0.08},
                "LINKEUR": {"min": 2.0, "max": 100.0, "typical_volatility": 0.08},
            },

            # Volume validation thresholds
            "volume_validation": {
                "min_volume": 0.01,  # Minimum valid volume
                "spike_threshold": 50.0,  # Volume spike detection (x median)
                "zero_volume_threshold": 0.05,  # Max % of zero volume periods
            },

            # Price validation parameters
            "price_validation": {
                "max_gap_pct": 0.2,  # Maximum price gap (20%)
                "max_daily_change": 0.5,  # Maximum daily change (50%)
                "min_price_precision": 1e-8,  # Minimum price precision
                "ohlc_consistency_tolerance": 0.001,  # OHLC consistency (0.1%)
            },

            # Time series validation
            "time_validation": {
                "max_missing_periods": 0.02,  # Max 2% missing data
                "min_data_points": 1000,  # Minimum data points required
                "expected_frequency": "30min",  # Expected data frequency
                "duplicate_tolerance": 0.001,  # Max % duplicate timestamps
            },

            # Statistical validation
            "statistical_validation": {
                "max_outlier_pct": 0.05,  # Max 5% outliers
                "outlier_std_threshold": 5.0,  # Outlier detection threshold
                "skewness_threshold": 3.0,  # Maximum skewness
                "kurtosis_threshold": 10.0,  # Maximum kurtosis
            }
        }

    def validate_market_data(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, bool]:
        """
        Comprehensive market data validation and cleaning.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            symbol: Trading symbol (e.g., 'BTCEUR')

        Returns:
            Tuple of (cleaned_df, validation_passed)
        """
        logger.info(f"🔍 Starting comprehensive validation for {symbol}")
        self.validation_results.clear()

        # Create working copy
        cleaned_df = df.copy()

        # Core validation checks
        cleaned_df = self._validate_data_structure(cleaned_df, symbol)
        cleaned_df = self._validate_price_ranges(cleaned_df, symbol)
        cleaned_df = self._validate_ohlc_consistency(cleaned_df, symbol)
        cleaned_df = self._validate_volume_quality(cleaned_df, symbol)
        cleaned_df = self._validate_time_series_integrity(cleaned_df, symbol)
        cleaned_df = self._validate_statistical_properties(cleaned_df, symbol)
        cleaned_df = self._detect_and_handle_outliers(cleaned_df, symbol)

        # Generate validation summary
        validation_passed = self._generate_validation_summary()

        logger.info(f"✅ Validation complete for {symbol}: {'PASSED' if validation_passed else 'FAILED'}")
        return cleaned_df, validation_passed

    def _validate_data_structure(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate basic data structure and completeness"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.validation_results.append(ValidationResult(
                check_name="data_structure",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing required columns: {missing_columns}",
                suggested_action="Ensure all OHLCV columns are present"
            ))
            return df

        # Check data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check minimum data requirements
        min_points = self.config["time_validation"]["min_data_points"]
        if len(df) < min_points:
            self.validation_results.append(ValidationResult(
                check_name="data_completeness",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Insufficient data: {len(df)} points (minimum: {min_points})",
                suggested_action="Fetch more historical data"
            ))

        # Check for completely missing data
        null_pct = df[required_columns].isnull().sum() / len(df)
        for col, pct in null_pct.items():
            if pct > 0.1:  # More than 10% null
                self.validation_results.append(ValidationResult(
                    check_name="null_data",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"High null percentage in {col}: {pct:.2%}",
                    affected_rows=int(pct * len(df)),
                    suggested_action="Check data source quality"
                ))

        return df

    def _validate_price_ranges(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate price ranges against expected market values"""
        if symbol not in self.config["price_ranges"]:
            logger.warning(f"No price range configuration for {symbol}")
            return df

        ranges = self.config["price_ranges"][symbol]
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            # Check minimum prices
            below_min = df[col] < ranges["min"]
            if below_min.any():
                count = below_min.sum()
                min_val = df.loc[below_min, col].min()

                self.validation_results.append(ValidationResult(
                    check_name=f"price_range_{col}",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"{col} below minimum: {count} values, lowest: {min_val:.4f} (min: {ranges['min']})",
                    affected_rows=count,
                    suggested_action="Check data source and symbol mapping"
                ))

                # Clamp to minimum
                df.loc[below_min, col] = ranges["min"]

            # Check maximum prices
            above_max = df[col] > ranges["max"]
            if above_max.any():
                count = above_max.sum()
                max_val = df.loc[above_max, col].max()

                self.validation_results.append(ValidationResult(
                    check_name=f"price_range_{col}",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"{col} above maximum: {count} values, highest: {max_val:.4f} (max: {ranges['max']})",
                    affected_rows=count,
                    suggested_action="Verify if price spike is legitimate"
                ))

                # Clamp to maximum
                df.loc[above_max, col] = ranges["max"]

        return df

    def _validate_ohlc_consistency(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLC bar consistency (High >= Low, etc.)"""
        tolerance = self.config["price_validation"]["ohlc_consistency_tolerance"]

        # High should be >= Open, Close, Low
        high_violations = (
            (df['high'] < df['open'] - tolerance) |
            (df['high'] < df['close'] - tolerance) |
            (df['high'] < df['low'] - tolerance)
        )

        if high_violations.any():
            count = high_violations.sum()
            self.validation_results.append(ValidationResult(
                check_name="ohlc_high_consistency",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"High price violations: {count} bars where high < open/close/low",
                affected_rows=count,
                suggested_action="Fix OHLC data consistency"
            ))

            # Fix by setting high = max(open, close, low, high)
            df.loc[high_violations, 'high'] = df.loc[high_violations, ['open', 'high', 'low', 'close']].max(axis=1)

        # Low should be <= Open, Close, High
        low_violations = (
            (df['low'] > df['open'] + tolerance) |
            (df['low'] > df['close'] + tolerance) |
            (df['low'] > df['high'] + tolerance)
        )

        if low_violations.any():
            count = low_violations.sum()
            self.validation_results.append(ValidationResult(
                check_name="ohlc_low_consistency",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Low price violations: {count} bars where low > open/close/high",
                affected_rows=count,
                suggested_action="Fix OHLC data consistency"
            ))

            # Fix by setting low = min(open, close, high, low)
            df.loc[low_violations, 'low'] = df.loc[low_violations, ['open', 'high', 'low', 'close']].min(axis=1)

        return df

    def _validate_volume_quality(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate volume data quality and detect anomalies"""
        vol_config = self.config["volume_validation"]

        # Check for negative volume
        negative_vol = df['volume'] < 0
        if negative_vol.any():
            count = negative_vol.sum()
            self.validation_results.append(ValidationResult(
                check_name="negative_volume",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Negative volume detected: {count} periods",
                affected_rows=count,
                suggested_action="Set negative volumes to zero"
            ))
            df.loc[negative_vol, 'volume'] = 0

        # Check for zero volume periods
        zero_vol = df['volume'] == 0
        zero_vol_pct = zero_vol.sum() / len(df)

        if zero_vol_pct > vol_config["zero_volume_threshold"]:
            self.validation_results.append(ValidationResult(
                check_name="zero_volume",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"High zero volume percentage: {zero_vol_pct:.2%}",
                affected_rows=zero_vol.sum(),
                suggested_action="Check market hours and trading activity"
            ))

        # Detect volume spikes
        median_volume = df['volume'].median()
        if median_volume > 0:
            volume_ratio = df['volume'] / median_volume
            spike_threshold = vol_config["spike_threshold"]

            volume_spikes = volume_ratio > spike_threshold
            if volume_spikes.any():
                count = volume_spikes.sum()
                max_spike = volume_ratio.max()

                self.validation_results.append(ValidationResult(
                    check_name="volume_spikes",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message=f"Volume spikes detected: {count} periods, max: {max_spike:.1f}x median",
                    affected_rows=count,
                    suggested_action="Monitor for data quality issues"
                ))

        return df

    def _validate_time_series_integrity(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate time series continuity and frequency"""
        time_config = self.config["time_validation"]

        if not isinstance(df.index, pd.DatetimeIndex):
            self.validation_results.append(ValidationResult(
                check_name="datetime_index",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message="Index is not DatetimeIndex",
                suggested_action="Convert index to DatetimeIndex"
            ))
            return df

        # Check for duplicate timestamps
        duplicates = df.index.duplicated()
        if duplicates.any():
            count = duplicates.sum()
            dup_pct = count / len(df)

            severity = ValidationSeverity.ERROR if dup_pct > time_config["duplicate_tolerance"] else ValidationSeverity.WARNING

            self.validation_results.append(ValidationResult(
                check_name="duplicate_timestamps",
                severity=severity,
                passed=False,
                message=f"Duplicate timestamps: {count} ({dup_pct:.2%})",
                affected_rows=count,
                suggested_action="Remove duplicate timestamps"
            ))

            # Remove duplicates, keeping first occurrence
            df = df[~duplicates]

        # Check time series gaps
        expected_freq = time_config["expected_frequency"]
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            expected_diff = pd.Timedelta(expected_freq)

            # Allow some tolerance for weekend gaps, etc.
            large_gaps = time_diffs > expected_diff * 3
            if large_gaps.any():
                gap_count = large_gaps.sum()
                max_gap = time_diffs.max()

                self.validation_results.append(ValidationResult(
                    check_name="time_gaps",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Large time gaps: {gap_count}, max gap: {max_gap}",
                    affected_rows=gap_count,
                    suggested_action="Check for missing data periods"
                ))

        return df

    def _validate_statistical_properties(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate statistical properties of price data"""
        stat_config = self.config["statistical_validation"]

        # Calculate returns for analysis
        returns = df['close'].pct_change().dropna()

        if len(returns) > 100:  # Need sufficient data for stats
            # Check skewness
            skewness = abs(stats.skew(returns))
            if skewness > stat_config["skewness_threshold"]:
                self.validation_results.append(ValidationResult(
                    check_name="return_skewness",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"High return skewness: {skewness:.2f} (threshold: {stat_config['skewness_threshold']})",
                    suggested_action="Check for data quality issues or extreme events"
                ))

            # Check kurtosis
            kurt = stats.kurtosis(returns)
            if kurt > stat_config["kurtosis_threshold"]:
                self.validation_results.append(ValidationResult(
                    check_name="return_kurtosis",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message=f"High return kurtosis: {kurt:.2f} (indicates fat tails)",
                    suggested_action="Normal for crypto markets"
                ))

        return df

    def _detect_and_handle_outliers(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Detect and handle statistical outliers in price data"""
        stat_config = self.config["statistical_validation"]
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = z_scores > stat_config["outlier_std_threshold"]

            if outliers.any():
                outlier_count = outliers.sum()
                outlier_pct = outlier_count / len(df)

                if outlier_pct > stat_config["max_outlier_pct"]:
                    severity = ValidationSeverity.ERROR
                    passed = False
                    action = "Investigate data source quality"
                else:
                    severity = ValidationSeverity.WARNING
                    passed = False
                    action = "Monitor outliers"

                self.validation_results.append(ValidationResult(
                    check_name=f"outliers_{col}",
                    severity=severity,
                    passed=passed,
                    message=f"Statistical outliers in {col}: {outlier_count} ({outlier_pct:.2%})",
                    affected_rows=outlier_count,
                    suggested_action=action
                ))

                # Cap extreme outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                # Only cap the most extreme values
                extreme_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                if extreme_outliers.any():
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound

        return df

    def _generate_validation_summary(self) -> bool:
        """Generate validation summary and determine overall pass/fail"""
        if not self.validation_results:
            logger.info("✅ All validation checks passed")
            return True

        # Count by severity
        severity_counts = {}
        for result in self.validation_results:
            severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1

        logger.info("📊 Validation Summary:")
        for severity, count in severity_counts.items():
            logger.info(f"  {severity.value.upper()}: {count}")

        # Log detailed results
        for result in self.validation_results:
            log_func = {
                ValidationSeverity.INFO: logger.info,
                ValidationSeverity.WARNING: logger.warning,
                ValidationSeverity.ERROR: logger.error,
                ValidationSeverity.CRITICAL: logger.critical
            }[result.severity]

            log_func(f"  {result.check_name}: {result.message}")
            if result.suggested_action:
                log_func(f"    → {result.suggested_action}")

        # Determine overall result - fail if any CRITICAL errors
        has_critical = any(r.severity == ValidationSeverity.CRITICAL for r in self.validation_results)
        return not has_critical

    def get_validation_report(self) -> Dict:
        """Get detailed validation report for monitoring/alerting"""
        return {
            "total_checks": len(self.validation_results),
            "passed_checks": sum(1 for r in self.validation_results if r.passed),
            "failed_checks": sum(1 for r in self.validation_results if not r.passed),
            "severity_breakdown": {
                severity.value: sum(1 for r in self.validation_results if r.severity == severity)
                for severity in ValidationSeverity
            },
            "detailed_results": [
                {
                    "check": r.check_name,
                    "severity": r.severity.value,
                    "passed": r.passed,
                    "message": r.message,
                    "affected_rows": r.affected_rows,
                    "suggested_action": r.suggested_action
                }
                for r in self.validation_results
            ]
        }