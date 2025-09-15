"""
Mathematical Stability Framework
===============================

Enterprise-grade mathematical operations with robust numerical stability,
outlier handling, and NaN/infinity prevention for financial calculations.
"""

import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class StableMath:
    """
    Mathematical operations framework with enterprise-grade numerical stability.
    Prevents common issues: division by zero, infinite values, extreme outliers.
    """

    # Global constants for numerical stability
    EPSILON = 1e-10  # Minimum denominator to prevent division by zero
    MAX_RATIO = 1000.0  # Maximum allowed ratio value
    MAX_ZSCORE = 10.0  # Maximum z-score before capping
    MAX_PERCENTAGE = 10.0  # Maximum percentage change (1000%)

    @classmethod
    def safe_divide(
        cls,
        numerator: Union[float, np.ndarray, pd.Series],
        denominator: Union[float, np.ndarray, pd.Series],
        max_value: Optional[float] = None,
        fill_value: float = 0.0
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Safe division with numerical stability guarantees.

        Args:
            numerator: Dividend values
            denominator: Divisor values
            max_value: Maximum allowed result (default: MAX_RATIO)
            fill_value: Value for zero/invalid denominators

        Returns:
            Stable division result
        """
        if max_value is None:
            max_value = cls.MAX_RATIO

        # Handle pandas Series
        if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
            # Ensure both are Series for consistent indexing
            if not isinstance(numerator, pd.Series):
                numerator = pd.Series(numerator, index=denominator.index if isinstance(denominator, pd.Series) else None)
            if not isinstance(denominator, pd.Series):
                denominator = pd.Series(denominator, index=numerator.index if isinstance(numerator, pd.Series) else None)

            # Safe division for Series
            safe_denom = denominator.copy()
            safe_denom = safe_denom.fillna(cls.EPSILON)
            safe_denom = np.where(np.abs(safe_denom) < cls.EPSILON, cls.EPSILON, safe_denom)

            result = numerator / safe_denom

            # Handle edge cases
            result = result.fillna(fill_value)
            result = np.where(np.isinf(result), fill_value, result)
            result = np.clip(result, -max_value, max_value)

            return result

        # Handle numpy arrays and scalars
        denominator = np.asarray(denominator)
        numerator = np.asarray(numerator)

        # Replace zeros and near-zeros with epsilon
        safe_denom = np.where(np.abs(denominator) < cls.EPSILON, cls.EPSILON, denominator)

        # Perform division
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / safe_denom

        # Handle NaN and infinity
        result = np.where(np.isnan(result) | np.isinf(result), fill_value, result)

        # Clip to reasonable bounds
        result = np.clip(result, -max_value, max_value)

        return result

    @classmethod
    def safe_log(
        cls,
        values: Union[float, np.ndarray, pd.Series],
        fill_value: float = 0.0
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Safe logarithm with handling for non-positive values.

        Args:
            values: Input values
            fill_value: Replacement for invalid results

        Returns:
            Safe logarithm result
        """
        if isinstance(values, pd.Series):
            # For Series, preserve index
            safe_values = np.where(values <= 0, cls.EPSILON, values)
            result = np.log(safe_values)
            result = pd.Series(result, index=values.index)
            return result.fillna(fill_value)

        # For arrays and scalars
        values = np.asarray(values)
        safe_values = np.where(values <= 0, cls.EPSILON, values)

        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.log(safe_values)

        result = np.where(np.isnan(result) | np.isinf(result), fill_value, result)
        return result

    @classmethod
    def safe_sqrt(
        cls,
        values: Union[float, np.ndarray, pd.Series],
        fill_value: float = 0.0
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Safe square root with handling for negative values.

        Args:
            values: Input values
            fill_value: Replacement for invalid results

        Returns:
            Safe square root result
        """
        if isinstance(values, pd.Series):
            safe_values = np.where(values < 0, 0, values)
            result = np.sqrt(safe_values)
            return result.fillna(fill_value)

        values = np.asarray(values)
        safe_values = np.where(values < 0, 0, values)

        with np.errstate(invalid='ignore'):
            result = np.sqrt(safe_values)

        result = np.where(np.isnan(result), fill_value, result)
        return result

    @classmethod
    def calculate_sharpe_ratio(
        cls,
        returns: pd.Series,
        window: int,
        risk_free_rate: float = 0.0,
        annualize: bool = True,
        periods_per_year: int = 17520  # 30-min periods per year
    ) -> pd.Series:
        """
        Calculate stable Sharpe ratio with robust numerical handling.

        Args:
            returns: Return series
            window: Rolling window size
            risk_free_rate: Risk-free rate (annual)
            annualize: Whether to annualize the ratio
            periods_per_year: Periods per year for annualization

        Returns:
            Stable Sharpe ratio series
        """
        # Calculate excess returns
        if annualize and risk_free_rate > 0:
            period_rf_rate = risk_free_rate / periods_per_year
            excess_returns = returns - period_rf_rate
        else:
            excess_returns = returns

        # Rolling mean and std with minimum periods
        min_periods = max(1, window // 2)
        mean_return = excess_returns.rolling(window=window, min_periods=min_periods).mean()
        volatility = excess_returns.rolling(window=window, min_periods=min_periods).std()

        # Safe Sharpe calculation
        sharpe = cls.safe_divide(mean_return, volatility, max_value=50.0, fill_value=0.0)

        # Annualize if requested
        if annualize:
            sharpe = sharpe * np.sqrt(periods_per_year)

        # Additional stability check - cap extreme values
        sharpe = np.clip(sharpe, -20.0, 20.0)

        return sharpe

    @classmethod
    def calculate_sortino_ratio(
        cls,
        returns: pd.Series,
        window: int,
        target_return: float = 0.0,
        annualize: bool = True,
        periods_per_year: int = 17520
    ) -> pd.Series:
        """
        Calculate stable Sortino ratio with robust downside deviation.

        Args:
            returns: Return series
            window: Rolling window size
            target_return: Target return threshold
            annualize: Whether to annualize the ratio
            periods_per_year: Periods per year for annualization

        Returns:
            Stable Sortino ratio series
        """
        # Calculate excess returns
        if annualize and target_return > 0:
            period_target = target_return / periods_per_year
            excess_returns = returns - period_target
        else:
            excess_returns = returns - target_return

        # Rolling mean
        min_periods = max(1, window // 2)
        mean_return = excess_returns.rolling(window=window, min_periods=min_periods).mean()

        # Downside deviation (only negative excess returns)
        downside_returns = excess_returns.where(excess_returns < 0, 0)
        downside_variance = downside_returns.rolling(window=window, min_periods=min_periods).var()
        downside_deviation = cls.safe_sqrt(downside_variance)

        # Safe Sortino calculation
        sortino = cls.safe_divide(mean_return, downside_deviation, max_value=100.0, fill_value=0.0)

        # Annualize if requested
        if annualize:
            sortino = sortino * np.sqrt(periods_per_year)

        # Cap extreme values
        sortino = np.clip(sortino, -50.0, 50.0)

        return sortino

    @classmethod
    def calculate_calmar_ratio(
        cls,
        returns: pd.Series,
        window: int,
        annualize: bool = True,
        periods_per_year: int = 17520
    ) -> pd.Series:
        """
        Calculate stable Calmar ratio with robust max drawdown calculation.

        Args:
            returns: Return series
            window: Rolling window size
            annualize: Whether to annualize the ratio
            periods_per_year: Periods per year for annualization

        Returns:
            Stable Calmar ratio series
        """
        # Calculate rolling mean return
        min_periods = max(1, window // 2)
        mean_return = returns.rolling(window=window, min_periods=min_periods).mean()

        # Calculate rolling maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=window, min_periods=min_periods).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(window=window, min_periods=min_periods).min()

        # Ensure max drawdown is not too close to zero
        max_drawdown = np.where(np.abs(max_drawdown) < 0.001, -0.001, max_drawdown)

        # Safe Calmar calculation (negative because max_drawdown is negative)
        calmar = cls.safe_divide(mean_return, -max_drawdown, max_value=200.0, fill_value=0.0)

        # Annualize if requested
        if annualize:
            calmar = calmar * periods_per_year

        # Cap extreme values
        calmar = np.clip(calmar, -100.0, 100.0)

        return calmar

    @classmethod
    def clean_extreme_values(
        cls,
        data: pd.Series,
        method: str = 'iqr',
        threshold: float = 3.0,
        cap_method: str = 'percentile'
    ) -> pd.Series:
        """
        Clean extreme values using robust statistical methods.

        Args:
            data: Input data series
            method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            cap_method: Method for handling outliers ('percentile', 'winsorize', 'clip')

        Returns:
            Cleaned data series
        """
        if len(data) < 10:  # Need minimum data for statistics
            return data

        cleaned_data = data.copy()

        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

        elif method == 'zscore':
            mean_val = data.mean()
            std_val = data.std()

            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val

        elif method == 'modified_zscore':
            median_val = data.median()
            mad = np.median(np.abs(data - median_val))
            modified_z_scores = 0.6745 * (data - median_val) / (mad + cls.EPSILON)

            outliers = np.abs(modified_z_scores) > threshold
            if cap_method == 'percentile':
                lower_bound = data.quantile(0.01)
                upper_bound = data.quantile(0.99)
            else:
                return cleaned_data  # Modified z-score with other methods not implemented

        else:
            return cleaned_data  # Unknown method

        # Apply capping method
        if cap_method == 'percentile':
            # Use percentiles instead of calculated bounds
            lower_bound = data.quantile(0.005)  # 0.5th percentile
            upper_bound = data.quantile(0.995)  # 99.5th percentile

        elif cap_method == 'winsorize':
            # Winsorize at specified percentiles
            lower_pct = 0.01
            upper_pct = 0.99
            lower_bound = data.quantile(lower_pct)
            upper_bound = data.quantile(upper_pct)

        # Apply bounds
        cleaned_data = np.clip(cleaned_data, lower_bound, upper_bound)

        return cleaned_data

    @classmethod
    def stabilize_features(
        cls,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
        max_zscore: float = 5.0
    ) -> pd.DataFrame:
        """
        Stabilize all features in a dataframe using robust methods.

        Args:
            df: Input dataframe
            feature_cols: Columns to stabilize (default: all numeric)
            max_zscore: Maximum z-score before capping

        Returns:
            Dataframe with stabilized features
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        stabilized_df = df.copy()

        for col in feature_cols:
            if col in df.columns:
                # Clean extreme values
                stabilized_df[col] = cls.clean_extreme_values(
                    df[col],
                    method='modified_zscore',
                    threshold=max_zscore,
                    cap_method='percentile'
                )

                # Handle remaining NaN/inf
                stabilized_df[col] = stabilized_df[col].fillna(0)
                stabilized_df[col] = np.where(
                    np.isinf(stabilized_df[col]), 0, stabilized_df[col]
                )

        return stabilized_df