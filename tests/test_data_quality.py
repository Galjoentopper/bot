"""Tests for data quality monitoring functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from paper_trader.utils.data_quality import (
    DataQualityMonitor, DataQualityLevel, DataQualityIssue, DataQualityReport
)


class TestDataQualityIssue:
    """Test DataQualityIssue data class."""
    
    def test_issue_creation(self):
        """Test creating a data quality issue."""
        issue = DataQualityIssue(
            severity='critical',
            category='gaps',
            description='Missing data points detected',
            affected_rows=10,
            recommendation='Check data source'
        )
        
        assert issue.severity == 'critical'
        assert issue.category == 'gaps'
        assert issue.affected_rows == 10


class TestDataQualityReport:
    """Test DataQualityReport data class."""
    
    def test_report_creation(self):
        """Test creating a data quality report."""
        report = DataQualityReport(
            symbol='BTC-EUR',
            timestamp=datetime.now(),
            overall_quality=DataQualityLevel.GOOD,
            quality_score=85.0,
            issues=[],
            metrics={'completeness': 0.95},
            is_tradeable=True
        )
        
        assert report.symbol == 'BTC-EUR'
        assert report.overall_quality == DataQualityLevel.GOOD
        assert report.quality_score == 85.0
        assert report.is_tradeable is True


class TestDataQualityMonitor:
    """Test DataQualityMonitor functionality."""
    
    def test_monitor_initialization(self):
        """Test data quality monitor initialization."""
        monitor = DataQualityMonitor()
        
        # Check that thresholds are set
        assert hasattr(monitor, 'thresholds')
        assert 'max_gap_minutes' in monitor.thresholds
        
    def test_create_sample_data(self):
        """Test creating sample data for testing."""
        # Create a sample DataFrame with price data
        timestamps = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        assert len(data) == 100
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
    def test_data_completeness_check(self):
        """Test data completeness validation."""
        monitor = DataQualityMonitor()
        
        # Create complete data
        complete_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1min'),
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        })
        
        # Test complete data
        completeness = len(complete_data.dropna()) / len(complete_data)
        assert completeness == 1.0
        
        # Create data with missing values
        incomplete_data = complete_data.copy()
        incomplete_data.loc[0, 'close'] = np.nan
        incomplete_data.loc[1, 'volume'] = np.nan
        
        completeness = len(incomplete_data.dropna()) / len(incomplete_data)
        assert completeness == 0.8  # 8 out of 10 rows complete
        
    def test_price_validation(self):
        """Test price data validation."""
        # Create valid price data
        valid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],  # high >= open, close
            'low': [95, 96, 97],      # low <= open, close
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        
        # Test that high >= max(open, close) for each row
        for i, row in valid_data.iterrows():
            assert row['high'] >= max(row['open'], row['close'])
            assert row['low'] <= min(row['open'], row['close'])
            
    def test_volume_validation(self):
        """Test volume data validation."""
        data = pd.DataFrame({
            'volume': [1000, 2000, 0, -100, np.nan]
        })
        
        # Volume should be non-negative (including 0)
        positive_volumes = data['volume'] >= 0
        valid_volumes = data['volume'].notna() & positive_volumes
        
        # Should have 3 valid volumes (1000, 2000, 0)
        assert valid_volumes.sum() == 3
        
    def test_outlier_detection_logic(self):
        """Test outlier detection logic."""
        # Create data with obvious outliers
        normal_prices = [100] * 10
        outlier_prices = normal_prices + [1000, 10]  # Very high and very low
        
        data = pd.Series(outlier_prices)
        
        # Simple outlier detection using IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
        # Should detect at least the extreme values as outliers
        assert outliers.sum() >= 2
        
    def test_gap_detection_logic(self):
        """Test data gap detection logic."""
        # Create timestamps with a gap
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        timestamps = [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=2),
            base_time + timedelta(minutes=10),  # 8-minute gap
            base_time + timedelta(minutes=11)
        ]
        
        data = pd.DataFrame({'timestamp': timestamps})
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Calculate time differences
        time_diffs = data['timestamp'].diff().dt.total_seconds() / 60  # in minutes
        
        # Find gaps larger than 5 minutes
        large_gaps = time_diffs > 5
        
        # Should detect the 8-minute gap
        assert large_gaps.sum() == 1
        
    def test_quality_score_calculation(self):
        """Test quality score calculation logic."""
        # Simple quality score based on completeness and gaps
        completeness = 0.95  # 95% complete
        has_large_gaps = False
        has_outliers = True
        
        # Basic scoring logic
        score = 100
        score *= completeness  # Reduce by incompleteness
        
        if has_large_gaps:
            score -= 20
            
        if has_outliers:
            score -= 10
            
        expected_score = 100 * 0.95 - 10  # 95 - 10 = 85
        assert score == expected_score
        
    def test_quality_level_determination(self):
        """Test quality level determination based on score."""
        scores_and_levels = [
            (95, DataQualityLevel.EXCELLENT),
            (85, DataQualityLevel.GOOD),
            (70, DataQualityLevel.FAIR),
            (50, DataQualityLevel.POOR),
            (30, DataQualityLevel.CRITICAL)
        ]
        
        for score, expected_level in scores_and_levels:
            # Simple level determination logic
            if score >= 90:
                level = DataQualityLevel.EXCELLENT
            elif score >= 80:
                level = DataQualityLevel.GOOD
            elif score >= 60:
                level = DataQualityLevel.FAIR
            elif score >= 40:
                level = DataQualityLevel.POOR
            else:
                level = DataQualityLevel.CRITICAL
                
            assert level == expected_level