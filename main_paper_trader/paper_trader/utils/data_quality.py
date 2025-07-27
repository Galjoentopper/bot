"""Data quality monitoring and validation for the paper trader."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class DataQualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class DataQualityIssue:
    """Represents a data quality issue."""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'gaps', 'outliers', 'volume', 'completeness'
    description: str
    affected_rows: int = 0
    recommendation: str = ""


@dataclass
class DataQualityReport:
    """Complete data quality assessment report."""
    symbol: str
    timestamp: datetime
    overall_quality: DataQualityLevel
    quality_score: float  # 0-100
    issues: List[DataQualityIssue]
    metrics: Dict[str, Any]
    is_tradeable: bool


class DataQualityMonitor:
    """Monitors and validates data quality for trading decisions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds (adjusted for 15-minute data)
        self.thresholds = {
            'max_gap_minutes': 30,  # 30 minutes for 15-minute candles (2 periods)
            'min_volume_threshold': 1000,
            'max_price_deviation': 0.1,  # 10% single-candle move
            'max_spread_pct': 0.05,  # 5% bid-ask spread
            'min_data_completeness': 0.95,  # 95% data completeness
            'max_consecutive_gaps': 3,
            'min_price_variance': 1e-6,  # Detect stuck prices
            'max_volume_spike': 10.0,  # 10x volume spike
            'outlier_zscore_threshold': 4.0
        }
        
        # Feature requirements
        self.required_features = [
            'open', 'high', 'low', 'close', 'volume'
        ]
        
        # Historical quality tracking
        self.quality_history = {}
        self.max_history_length = 100
    
    def validate_data_quality(self, symbol: str, data: pd.DataFrame) -> DataQualityReport:
        """Perform comprehensive data quality validation."""
        
        issues = []
        metrics = {}
        quality_score = 100.0
        
        try:
            # Basic structure validation
            structure_issues, structure_score = self._validate_structure(data)
            issues.extend(structure_issues)
            quality_score = min(quality_score, structure_score)
            
            # Completeness validation
            completeness_issues, completeness_score = self._validate_completeness(data)
            issues.extend(completeness_issues)
            quality_score = min(quality_score, completeness_score)
            
            # Time series validation
            temporal_issues, temporal_score = self._validate_temporal_consistency(data)
            issues.extend(temporal_issues)
            quality_score = min(quality_score, temporal_score)
            
            # Price data validation
            price_issues, price_score = self._validate_price_data(data)
            issues.extend(price_issues)
            quality_score = min(quality_score, price_score)
            
            # Volume validation
            volume_issues, volume_score = self._validate_volume_data(data)
            issues.extend(volume_issues)
            quality_score = min(quality_score, volume_score)
            
            # Outlier detection
            outlier_issues, outlier_score = self._detect_outliers(data)
            issues.extend(outlier_issues)
            quality_score = min(quality_score, outlier_score)
            
            # Calculate metrics
            metrics = self._calculate_quality_metrics(data)
            
            # Determine overall quality level
            overall_quality = self._determine_quality_level(quality_score)
            
            # Determine if data is tradeable
            is_tradeable = self._is_data_tradeable(issues, quality_score)
            
            # Create report
            report = DataQualityReport(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_quality=overall_quality,
                quality_score=quality_score,
                issues=issues,
                metrics=metrics,
                is_tradeable=is_tradeable
            )
            
            # Store in history
            self._update_quality_history(symbol, report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error validating data quality for {symbol}: {e}")
            return DataQualityReport(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_quality=DataQualityLevel.CRITICAL,
                quality_score=0.0,
                issues=[DataQualityIssue(
                    severity='critical',
                    category='system',
                    description=f"Validation error: {str(e)}",
                    recommendation="Check data format and system integrity"
                )],
                metrics={},
                is_tradeable=False
            )
    
    def _validate_structure(self, data: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Validate basic data structure."""
        issues = []
        score = 100.0
        
        # Check if data exists
        if data is None or data.empty:
            issues.append(DataQualityIssue(
                severity='critical',
                category='completeness',
                description="No data available",
                recommendation="Check data collection process"
            ))
            return issues, 0.0
        
        # Check required columns
        missing_columns = [col for col in self.required_features if col not in data.columns]
        if missing_columns:
            issues.append(DataQualityIssue(
                severity='critical',
                category='completeness',
                description=f"Missing required columns: {missing_columns}",
                recommendation="Ensure data source provides all required fields"
            ))
            score -= 50
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                issues.append(DataQualityIssue(
                    severity='warning',
                    category='format',
                    description=f"Column {col} is not numeric",
                    recommendation=f"Convert {col} to numeric type"
                ))
                score -= 10
        
        # Check minimum data length
        if len(data) < 100:
            issues.append(DataQualityIssue(
                severity='warning',
                category='completeness',
                description=f"Insufficient data length: {len(data)} rows",
                recommendation="Collect more historical data"
            ))
            score -= 20
        
        return issues, score
    
    def _validate_completeness(self, data: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Validate data completeness."""
        issues = []
        score = 100.0
        
        # Check for null values
        null_counts = data.isnull().sum()
        total_cells = len(data) * len(data.columns)
        null_percentage = null_counts.sum() / total_cells
        
        if null_percentage > (1 - self.thresholds['min_data_completeness']):
            issues.append(DataQualityIssue(
                severity='critical',
                category='completeness',
                description=f"High null percentage: {null_percentage:.2%}",
                affected_rows=int(null_counts.sum()),
                recommendation="Investigate data source issues"
            ))
            score -= 40
        
        # Check for null values in critical columns
        critical_columns = ['close', 'volume']
        for col in critical_columns:
            if col in data.columns:
                col_nulls = data[col].isnull().sum()
                if col_nulls > 0:
                    issues.append(DataQualityIssue(
                        severity='warning',
                        category='completeness',
                        description=f"Null values in {col}: {col_nulls}",
                        affected_rows=col_nulls,
                        recommendation=f"Forward-fill or interpolate {col} values"
                    ))
                    score -= 15
        
        return issues, score
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Validate time series consistency."""
        issues = []
        score = 100.0
        
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append(DataQualityIssue(
                severity='warning',
                category='temporal',
                description="Index is not datetime type",
                recommendation="Convert index to datetime"
            ))
            score -= 10
            return issues, score
        
        # Check for time gaps
        time_diffs = data.index.to_series().diff().dropna()
        # Calculate expected interval (should be 15 minutes for our data)
        expected_interval = time_diffs.mode().iloc[0] if not time_diffs.empty else pd.Timedelta(minutes=15)
        
        large_gaps = time_diffs > expected_interval * 2
        if large_gaps.any():
            gap_count = large_gaps.sum()
            issues.append(DataQualityIssue(
                severity='warning',
                category='gaps',
                description=f"Time gaps detected: {gap_count} gaps",
                affected_rows=gap_count,
                recommendation="Fill gaps with interpolated data"
            ))
            score -= min(30, gap_count * 5)
        
        # Check for duplicate timestamps
        duplicates = data.index.duplicated().sum()
        if duplicates > 0:
            issues.append(DataQualityIssue(
                severity='warning',
                category='temporal',
                description=f"Duplicate timestamps: {duplicates}",
                affected_rows=duplicates,
                recommendation="Remove or aggregate duplicate entries"
            ))
            score -= min(20, duplicates * 2)
        
        # Check chronological order
        if not data.index.is_monotonic_increasing:
            issues.append(DataQualityIssue(
                severity='warning',
                category='temporal',
                description="Data is not chronologically ordered",
                recommendation="Sort data by timestamp"
            ))
            score -= 15
        
        return issues, score
    
    def _validate_price_data(self, data: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Validate price data consistency."""
        issues = []
        score = 100.0
        
        price_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in price_columns if col in data.columns]
        
        if not available_columns:
            return issues, score
        
        # Check OHLC relationships
        if all(col in data.columns for col in price_columns):
            # High should be >= open, close, low
            invalid_high = (data['high'] < data[['open', 'close', 'low']].max(axis=1)).sum()
            if invalid_high > 0:
                issues.append(DataQualityIssue(
                    severity='critical',
                    category='price',
                    description=f"Invalid high prices: {invalid_high} candles",
                    affected_rows=invalid_high,
                    recommendation="Verify price data source"
                ))
                score -= 30
            
            # Low should be <= open, close, high
            invalid_low = (data['low'] > data[['open', 'close', 'high']].min(axis=1)).sum()
            if invalid_low > 0:
                issues.append(DataQualityIssue(
                    severity='critical',
                    category='price',
                    description=f"Invalid low prices: {invalid_low} candles",
                    affected_rows=invalid_low,
                    recommendation="Verify price data source"
                ))
                score -= 30
        
        # Check for zero or negative prices
        for col in available_columns:
            invalid_prices = (data[col] <= 0).sum()
            if invalid_prices > 0:
                issues.append(DataQualityIssue(
                    severity='critical',
                    category='price',
                    description=f"Invalid {col} prices (<=0): {invalid_prices}",
                    affected_rows=invalid_prices,
                    recommendation=f"Remove or correct invalid {col} values"
                ))
                score -= 25
        
        # Check for extreme price movements
        if 'close' in data.columns:
            returns = data['close'].pct_change().abs()
            extreme_moves = (returns > self.thresholds['max_price_deviation']).sum()
            if extreme_moves > 0:
                issues.append(DataQualityIssue(
                    severity='warning',
                    category='outliers',
                    description=f"Extreme price movements: {extreme_moves} occurrences",
                    affected_rows=extreme_moves,
                    recommendation="Investigate extreme movements for data errors"
                ))
                score -= min(20, extreme_moves * 2)
        
        # Check for stuck prices (no variance)
        for col in available_columns:
            if data[col].var() < self.thresholds['min_price_variance']:
                issues.append(DataQualityIssue(
                    severity='warning',
                    category='price',
                    description=f"No price variance in {col}",
                    recommendation="Check if price feed is active"
                ))
                score -= 15
        
        return issues, score
    
    def _validate_volume_data(self, data: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Validate volume data."""
        issues = []
        score = 100.0
        
        if 'volume' not in data.columns:
            return issues, score
        
        volume = data['volume']
        
        # Check for negative volume
        negative_volume = (volume < 0).sum()
        if negative_volume > 0:
            issues.append(DataQualityIssue(
                severity='critical',
                category='volume',
                description=f"Negative volume values: {negative_volume}",
                affected_rows=negative_volume,
                recommendation="Correct negative volume values"
            ))
            score -= 30
        
        # Check for zero volume periods
        zero_volume = (volume == 0).sum()
        zero_pct = zero_volume / len(volume)
        if zero_pct > 0.1:  # More than 10% zero volume
            issues.append(DataQualityIssue(
                severity='warning',
                category='volume',
                description=f"High zero volume percentage: {zero_pct:.1%}",
                affected_rows=zero_volume,
                recommendation="Investigate market activity during zero volume periods"
            ))
            score -= 15
        
        # Check for volume spikes
        if len(volume) > 1:
            volume_median = volume.median()
            if volume_median > 0:
                volume_spikes = (volume > volume_median * self.thresholds['max_volume_spike']).sum()
                if volume_spikes > 0:
                    issues.append(DataQualityIssue(
                        severity='info',
                        category='volume',
                        description=f"Volume spikes detected: {volume_spikes}",
                        affected_rows=volume_spikes,
                        recommendation="Review volume spikes for potential market events"
                    ))
        
        return issues, score
    
    def _detect_outliers(self, data: pd.DataFrame) -> Tuple[List[DataQualityIssue], float]:
        """Detect statistical outliers in the data."""
        issues = []
        score = 100.0
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in data.columns and len(data[col].dropna()) > 10:
                # Z-score based outlier detection
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = (z_scores > self.thresholds['outlier_zscore_threshold']).sum()
                
                if outliers > 0:
                    outlier_pct = outliers / len(data)
                    if outlier_pct > 0.05:  # More than 5% outliers
                        issues.append(DataQualityIssue(
                            severity='warning',
                            category='outliers',
                            description=f"Statistical outliers in {col}: {outliers} ({outlier_pct:.1%})",
                            affected_rows=outliers,
                            recommendation=f"Review {col} outliers for data quality issues"
                        ))
                        score -= min(15, outliers)
        
        return issues, score
    
    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed quality metrics."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['total_rows'] = len(data)
            metrics['total_columns'] = len(data.columns)
            metrics['completeness_pct'] = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            
            # Time series metrics
            if isinstance(data.index, pd.DatetimeIndex):
                time_span = data.index.max() - data.index.min()
                metrics['time_span_hours'] = time_span.total_seconds() / 3600
                
                if len(data) > 1:
                    time_diffs = data.index.to_series().diff().dropna()
                    metrics['avg_interval_minutes'] = time_diffs.mean().total_seconds() / 60
                    metrics['max_gap_minutes'] = time_diffs.max().total_seconds() / 60
            
            # Price metrics
            if 'close' in data.columns:
                close_prices = data['close'].dropna()
                if len(close_prices) > 1:
                    returns = close_prices.pct_change().dropna()
                    metrics['price_volatility'] = returns.std() * 100
                    metrics['max_return'] = returns.max() * 100
                    metrics['min_return'] = returns.min() * 100
            
            # Volume metrics
            if 'volume' in data.columns:
                volume = data['volume'].dropna()
                if len(volume) > 0:
                    metrics['avg_volume'] = volume.mean()
                    metrics['median_volume'] = volume.median()
                    metrics['volume_std'] = volume.std()
                    metrics['zero_volume_pct'] = (volume == 0).sum() / len(volume) * 100
        
        except Exception as e:
            self.logger.warning(f"Error calculating quality metrics: {e}")
        
        return metrics
    
    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """Determine overall quality level from score."""
        if score >= 90:
            return DataQualityLevel.EXCELLENT
        elif score >= 75:
            return DataQualityLevel.GOOD
        elif score >= 60:
            return DataQualityLevel.FAIR
        elif score >= 40:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL
    
    def _is_data_tradeable(self, issues: List[DataQualityIssue], score: float) -> bool:
        """Determine if data quality is sufficient for trading."""
        # Critical issues make data non-tradeable
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        if critical_issues:
            return False
        
        # Score too low
        if score < 60:
            return False
        
        return True
    
    def _update_quality_history(self, symbol: str, report: DataQualityReport):
        """Update quality history for trend analysis."""
        if symbol not in self.quality_history:
            self.quality_history[symbol] = []
        
        self.quality_history[symbol].append({
            'timestamp': report.timestamp,
            'quality_score': report.quality_score,
            'is_tradeable': report.is_tradeable,
            'issue_count': len(report.issues)
        })
        
        # Keep only recent history
        if len(self.quality_history[symbol]) > self.max_history_length:
            self.quality_history[symbol] = self.quality_history[symbol][-self.max_history_length:]
    
    def get_quality_trends(self, symbol: str) -> Dict[str, Any]:
        """Get quality trends for a symbol."""
        if symbol not in self.quality_history or not self.quality_history[symbol]:
            return {}
        
        history = self.quality_history[symbol]
        scores = [h['quality_score'] for h in history]
        
        return {
            'current_score': scores[-1] if scores else 0,
            'avg_score_7d': np.mean(scores[-7:]) if len(scores) >= 7 else np.mean(scores),
            'score_trend': 'improving' if len(scores) >= 2 and scores[-1] > scores[-2] else 'declining',
            'tradeable_ratio': sum(h['is_tradeable'] for h in history) / len(history),
            'total_assessments': len(history)
        }
    
    def generate_quality_summary(self, symbol: str, report: DataQualityReport) -> str:
        """Generate a human-readable quality summary."""
        trends = self.get_quality_trends(symbol)
        
        summary = f"""
📊 Data Quality Report for {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Quality: {report.overall_quality.value.upper()} ({report.quality_score:.1f}/100)
Tradeable: {'✅ YES' if report.is_tradeable else '❌ NO'}
Issues Found: {len(report.issues)}

📈 Recent Trends:
- 7-day Average: {trends.get('avg_score_7d', 0):.1f}
- Trend: {trends.get('score_trend', 'unknown')}
- Tradeable Ratio: {trends.get('tradeable_ratio', 0):.1%}

"""
        
        if report.issues:
            summary += "⚠️  Issues Detected:\n"
            for issue in report.issues[:5]:  # Show top 5 issues
                summary += f"   • {issue.severity.upper()}: {issue.description}\n"
            
            if len(report.issues) > 5:
                summary += f"   ... and {len(report.issues) - 5} more issues\n"
        
        if report.metrics:
            summary += f"\n📋 Key Metrics:\n"
            if 'completeness_pct' in report.metrics:
                summary += f"   • Data Completeness: {report.metrics['completeness_pct']:.1f}%\n"
            if 'price_volatility' in report.metrics:
                summary += f"   • Price Volatility: {report.metrics['price_volatility']:.2f}%\n"
            if 'avg_volume' in report.metrics:
                summary += f"   • Average Volume: {report.metrics['avg_volume']:,.0f}\n"
        
        return summary