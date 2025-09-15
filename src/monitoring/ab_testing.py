"""
A/B Testing Framework for Trading Models

Implements statistical A/B testing for comparing model versions,
with proper traffic splitting, significance testing, and automated
decision making for model deployment.
"""

import hashlib
import json
import logging
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind


class TestStatus(Enum):
    """A/B Test status"""

    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class TestConclusion(Enum):
    """A/B Test conclusion"""

    VARIANT_WINS = "variant_wins"
    CONTROL_WINS = "control_wins"
    NO_DIFFERENCE = "no_difference"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ABTestConfig:
    """Configuration for A/B test"""

    test_id: str
    name: str
    description: str

    # Models being tested
    control_model_name: str
    variant_model_name: str

    # Traffic allocation (should sum to 1.0)
    control_traffic: float = 0.5
    variant_traffic: float = 0.5

    # Test duration and criteria
    min_sample_size: int = 1000
    max_duration_days: int = 30
    significance_level: float = 0.05
    minimum_effect_size: float = 0.05  # 5% minimum improvement

    # Primary metric to optimize
    primary_metric: str = "sharpe_ratio"

    # Secondary metrics to track
    secondary_metrics: List[str] = field(
        default_factory=lambda: [
            "total_return",
            "max_drawdown",
            "volatility",
            "win_rate",
        ]
    )

    # Automated stopping conditions
    early_stopping_enabled: bool = True
    confidence_threshold: float = 0.99  # Stop early if very confident

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTestResult:
    """Results of A/B test analysis"""

    test_id: str
    status: TestStatus
    conclusion: TestConclusion

    # Sample sizes
    control_samples: int
    variant_samples: int

    # Primary metric results
    control_primary_value: float
    variant_primary_value: float
    improvement_pct: float
    p_value: float
    confidence_interval: Tuple[float, float]

    # Statistical power and effect size
    statistical_power: float
    effect_size: float

    # Secondary metrics
    secondary_results: Dict[str, Dict[str, Any]]

    # Recommendations
    recommendation: str
    confidence_level: float

    analyzed_at: datetime = field(default_factory=datetime.now)


class ABTestingFramework:
    """Comprehensive A/B testing framework for trading models"""

    def __init__(self, db_path: Optional[Path] = None, random_seed: int = 42):
        """
        Initialize A/B testing framework

        Args:
            db_path: SQLite database for storing test data
            random_seed: Random seed for consistent traffic splitting
        """
        self.db_path = db_path or Path("data/ab_testing.db")
        self.random_seed = random_seed
        self.logger = logging.getLogger(__name__)

        # Active tests cache
        self.active_tests = {}

        # Initialize database
        self._init_database()
        self._load_active_tests()

    def _init_database(self):
        """Initialize SQLite database for A/B testing"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Test configurations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ab_test_configs (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    control_model_name TEXT NOT NULL,
                    variant_model_name TEXT NOT NULL,
                    control_traffic REAL NOT NULL,
                    variant_traffic REAL NOT NULL,
                    min_sample_size INTEGER NOT NULL,
                    max_duration_days INTEGER NOT NULL,
                    significance_level REAL NOT NULL,
                    minimum_effect_size REAL NOT NULL,
                    primary_metric TEXT NOT NULL,
                    secondary_metrics TEXT NOT NULL,
                    early_stopping_enabled INTEGER NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Test observations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ab_test_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    variant TEXT NOT NULL,  -- 'control' or 'variant'
                    timestamp TEXT NOT NULL,
                    primary_metric_value REAL,
                    secondary_metrics TEXT,  -- JSON of secondary metrics
                    additional_data TEXT,    -- JSON of additional context
                    FOREIGN KEY (test_id) REFERENCES ab_test_configs (test_id)
                )
            """
            )

            # Test results table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    test_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    conclusion TEXT NOT NULL,
                    control_samples INTEGER NOT NULL,
                    variant_samples INTEGER NOT NULL,
                    control_primary_value REAL NOT NULL,
                    variant_primary_value REAL NOT NULL,
                    improvement_pct REAL NOT NULL,
                    p_value REAL NOT NULL,
                    confidence_interval_lower REAL NOT NULL,
                    confidence_interval_upper REAL NOT NULL,
                    statistical_power REAL NOT NULL,
                    effect_size REAL NOT NULL,
                    secondary_results TEXT NOT NULL,  -- JSON
                    recommendation TEXT NOT NULL,
                    confidence_level REAL NOT NULL,
                    analyzed_at TEXT NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES ab_test_configs (test_id)
                )
            """
            )

            # Create indices
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_test_variant ON ab_test_observations(test_id, variant)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_timestamp ON ab_test_observations(timestamp)"
            )

        self.logger.info("A/B testing database initialized")

    def create_test(self, config: ABTestConfig) -> bool:
        """Create new A/B test"""

        # Validate configuration
        if not self._validate_test_config(config):
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO ab_test_configs
                    (test_id, name, description, control_model_name, variant_model_name,
                     control_traffic, variant_traffic, min_sample_size, max_duration_days,
                     significance_level, minimum_effect_size, primary_metric,
                     secondary_metrics, early_stopping_enabled, confidence_threshold,
                     status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        config.test_id,
                        config.name,
                        config.description,
                        config.control_model_name,
                        config.variant_model_name,
                        config.control_traffic,
                        config.variant_traffic,
                        config.min_sample_size,
                        config.max_duration_days,
                        config.significance_level,
                        config.minimum_effect_size,
                        config.primary_metric,
                        json.dumps(config.secondary_metrics),
                        int(config.early_stopping_enabled),
                        config.confidence_threshold,
                        TestStatus.DRAFT.value,
                        config.created_at.isoformat(),
                        config.updated_at.isoformat(),
                    ),
                )

            self.logger.info(f"Created A/B test: {config.name} ({config.test_id})")
            return True

        except Exception as e:
            self.logger.error(f"Error creating A/B test: {e}")
            return False

    def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE ab_test_configs
                    SET status = ?, updated_at = ?
                    WHERE test_id = ?
                """,
                    (TestStatus.RUNNING.value, datetime.now().isoformat(), test_id),
                )

                if conn.total_changes == 0:
                    self.logger.error(f"Test {test_id} not found")
                    return False

            # Add to active tests cache
            self._load_active_tests()

            self.logger.info(f"Started A/B test: {test_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting test {test_id}: {e}")
            return False

    def assign_variant(self, test_id: str, user_id: str) -> Optional[str]:
        """
        Assign user to control or variant group

        Returns 'control' or 'variant' based on consistent hashing
        """
        if test_id not in self.active_tests:
            return None

        config = self.active_tests[test_id]

        # Use consistent hashing to assign variant
        hash_input = f"{test_id}_{user_id}_{self.random_seed}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        assignment_value = (hash_value % 10000) / 10000.0  # 0 to 1

        if assignment_value < config.control_traffic:
            return "control"
        elif assignment_value < (config.control_traffic + config.variant_traffic):
            return "variant"
        else:
            # User not in test (if traffic allocation < 1.0)
            return None

    def record_observation(
        self,
        test_id: str,
        user_id: str,
        variant: str,
        primary_metric_value: float,
        secondary_metrics: Optional[Dict[str, float]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record observation for A/B test"""

        if test_id not in self.active_tests:
            self.logger.warning(f"Attempted to record observation for inactive test: {test_id}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO ab_test_observations
                    (test_id, user_id, variant, timestamp, primary_metric_value,
                     secondary_metrics, additional_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        test_id,
                        user_id,
                        variant,
                        datetime.now().isoformat(),
                        primary_metric_value,
                        json.dumps(secondary_metrics) if secondary_metrics else None,
                        json.dumps(additional_data) if additional_data else None,
                    ),
                )

            return True

        except Exception as e:
            self.logger.error(f"Error recording observation for test {test_id}: {e}")
            return False

    def analyze_test(self, test_id: str, force_analysis: bool = False) -> Optional[ABTestResult]:
        """
        Analyze A/B test results and determine statistical significance

        Args:
            test_id: Test identifier
            force_analysis: Force analysis even if minimum criteria not met
        """

        if test_id not in self.active_tests:
            self.logger.error(f"Test {test_id} not found in active tests")
            return None

        config = self.active_tests[test_id]

        try:
            # Get observations
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT variant, primary_metric_value, secondary_metrics, timestamp
                    FROM ab_test_observations
                    WHERE test_id = ? AND primary_metric_value IS NOT NULL
                    ORDER BY timestamp
                """,
                    (test_id,),
                )

                observations = cursor.fetchall()

        except Exception as e:
            self.logger.error(f"Error fetching observations for test {test_id}: {e}")
            return None

        if not observations:
            self.logger.warning(f"No observations found for test {test_id}")
            return None

        # Separate control and variant observations
        control_values = []
        variant_values = []
        control_secondary = []
        variant_secondary = []

        for variant, primary_value, secondary_json, timestamp in observations:
            secondary_data = json.loads(secondary_json) if secondary_json else {}

            if variant == "control":
                control_values.append(primary_value)
                control_secondary.append(secondary_data)
            elif variant == "variant":
                variant_values.append(primary_value)
                variant_secondary.append(secondary_data)

        # Check minimum sample size
        if not force_analysis:
            if (
                len(control_values) < config.min_sample_size
                or len(variant_values) < config.min_sample_size
            ):
                self.logger.info(f"Test {test_id} has insufficient samples for analysis")
                return None

        if len(control_values) == 0 or len(variant_values) == 0:
            self.logger.warning(f"Test {test_id} missing control or variant data")
            return None

        # Perform statistical analysis
        control_mean = np.mean(control_values)
        variant_mean = np.mean(variant_values)
        improvement_pct = (
            (variant_mean - control_mean) / control_mean * 100 if control_mean != 0 else 0
        )

        # Two-sample t-test for primary metric
        t_stat, p_value = ttest_ind(variant_values, control_values, equal_var=False)

        # Calculate confidence interval for the difference
        pooled_std = np.sqrt((np.var(control_values) + np.var(variant_values)) / 2)
        se_diff = pooled_std * np.sqrt(1 / len(control_values) + 1 / len(variant_values))

        t_critical = stats.t.ppf(
            1 - config.significance_level / 2,
            len(control_values) + len(variant_values) - 2,
        )
        margin_of_error = t_critical * se_diff

        mean_diff = variant_mean - control_mean
        ci_lower = mean_diff - margin_of_error
        ci_upper = mean_diff + margin_of_error

        # Calculate effect size (Cohen's d)
        effect_size = mean_diff / pooled_std if pooled_std != 0 else 0

        # Estimate statistical power (simplified)
        statistical_power = self._estimate_power(
            len(control_values),
            len(variant_values),
            effect_size,
            config.significance_level,
        )

        # Analyze secondary metrics
        secondary_results = {}
        for metric_name in config.secondary_metrics:
            control_secondary_values = [
                s.get(metric_name) for s in control_secondary if s.get(metric_name) is not None
            ]
            variant_secondary_values = [
                s.get(metric_name) for s in variant_secondary if s.get(metric_name) is not None
            ]

            if len(control_secondary_values) > 5 and len(variant_secondary_values) > 5:
                try:
                    sec_t_stat, sec_p_value = ttest_ind(
                        variant_secondary_values,
                        control_secondary_values,
                        equal_var=False,
                    )
                    sec_control_mean = np.mean(control_secondary_values)
                    sec_variant_mean = np.mean(variant_secondary_values)
                    sec_improvement = (
                        (sec_variant_mean - sec_control_mean) / sec_control_mean * 100
                        if sec_control_mean != 0
                        else 0
                    )

                    secondary_results[metric_name] = {
                        "control_mean": sec_control_mean,
                        "variant_mean": sec_variant_mean,
                        "improvement_pct": sec_improvement,
                        "p_value": sec_p_value,
                        "significant": sec_p_value < config.significance_level,
                    }
                except Exception as e:
                    self.logger.warning(f"Error analyzing secondary metric {metric_name}: {e}")

        # Determine conclusion
        conclusion = self._determine_conclusion(
            p_value,
            improvement_pct,
            config.significance_level,
            config.minimum_effect_size,
        )

        # Generate recommendation
        recommendation, confidence_level = self._generate_recommendation(
            conclusion, improvement_pct, p_value, statistical_power, config
        )

        result = ABTestResult(
            test_id=test_id,
            status=TestStatus.RUNNING,
            conclusion=conclusion,
            control_samples=len(control_values),
            variant_samples=len(variant_values),
            control_primary_value=control_mean,
            variant_primary_value=variant_mean,
            improvement_pct=improvement_pct,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            statistical_power=statistical_power,
            effect_size=effect_size,
            secondary_results=secondary_results,
            recommendation=recommendation,
            confidence_level=confidence_level,
        )

        # Store results
        self._store_test_result(result)

        # Check for early stopping
        if config.early_stopping_enabled and self._should_stop_early(result, config):
            self.stop_test(test_id)
            result.status = TestStatus.COMPLETED

        return result

    def _validate_test_config(self, config: ABTestConfig) -> bool:
        """Validate A/B test configuration"""

        if abs(config.control_traffic + config.variant_traffic - 1.0) > 0.001:
            if config.control_traffic + config.variant_traffic > 1.0:
                self.logger.error("Traffic allocation cannot exceed 100%")
                return False

        if config.significance_level <= 0 or config.significance_level >= 1:
            self.logger.error("Significance level must be between 0 and 1")
            return False

        if config.min_sample_size < 10:
            self.logger.error("Minimum sample size too small (< 10)")
            return False

        return True

    def _determine_conclusion(
        self,
        p_value: float,
        improvement_pct: float,
        significance_level: float,
        minimum_effect_size: float,
    ) -> TestConclusion:
        """Determine test conclusion based on statistical results"""

        is_significant = p_value < significance_level
        has_practical_significance = abs(improvement_pct) >= minimum_effect_size * 100

        if is_significant and has_practical_significance:
            if improvement_pct > 0:
                return TestConclusion.VARIANT_WINS
            else:
                return TestConclusion.CONTROL_WINS
        elif is_significant and not has_practical_significance:
            return TestConclusion.NO_DIFFERENCE
        else:
            return TestConclusion.INCONCLUSIVE

    def _generate_recommendation(
        self,
        conclusion: TestConclusion,
        improvement_pct: float,
        p_value: float,
        statistical_power: float,
        config: ABTestConfig,
    ) -> Tuple[str, float]:
        """Generate recommendation and confidence level"""

        confidence_level = 1 - p_value

        recommendations = {
            TestConclusion.VARIANT_WINS: (
                f"Deploy variant model. Shows {improvement_pct:.1f}% improvement with "
                f"{confidence_level:.1%} confidence. Expected performance gain is statistically significant.",
                confidence_level,
            ),
            TestConclusion.CONTROL_WINS: (
                f"Keep control model. Variant shows {improvement_pct:.1f}% decrease with "
                f"{confidence_level:.1%} confidence. Do not deploy variant.",
                confidence_level,
            ),
            TestConclusion.NO_DIFFERENCE: (
                f"No practical difference detected. Improvement of {improvement_pct:.1f}% is below "
                f"minimum threshold of {config.minimum_effect_size:.1%}. Either model can be used.",
                confidence_level,
            ),
            TestConclusion.INCONCLUSIVE: (
                f"Inconclusive results. P-value: {p_value:.3f}, improvement: {improvement_pct:.1f}%. "
                f"Consider running test longer or increasing sample size. Power: {statistical_power:.1%}",
                min(confidence_level, 0.5),  # Low confidence for inconclusive
            ),
        }

        return recommendations.get(conclusion, ("No recommendation available.", 0.5))

    def _estimate_power(self, n1: int, n2: int, effect_size: float, alpha: float) -> float:
        """Estimate statistical power (simplified calculation)"""

        # Simplified power calculation using normal approximation
        pooled_n = 2 / (1 / n1 + 1 / n2)
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = abs(effect_size) * np.sqrt(pooled_n / 2) - z_alpha
        power = stats.norm.cdf(z_beta)

        return max(0.0, min(1.0, power))

    def _should_stop_early(self, result: ABTestResult, config: ABTestConfig) -> bool:
        """Check if test should stop early"""

        # Stop if very high confidence and practical significance
        if (
            result.confidence_level >= config.confidence_threshold
            and abs(result.improvement_pct) >= config.minimum_effect_size * 100
            and result.statistical_power >= 0.8
        ):
            return True

        # Stop if variant is clearly worse with high confidence
        if (
            result.confidence_level >= 0.95
            and result.improvement_pct < -config.minimum_effect_size * 100
        ):
            return True

        return False

    def _store_test_result(self, result: ABTestResult):
        """Store test result to database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ab_test_results
                    (test_id, status, conclusion, control_samples, variant_samples,
                     control_primary_value, variant_primary_value, improvement_pct,
                     p_value, confidence_interval_lower, confidence_interval_upper,
                     statistical_power, effect_size, secondary_results,
                     recommendation, confidence_level, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.test_id,
                        result.status.value,
                        result.conclusion.value,
                        result.control_samples,
                        result.variant_samples,
                        result.control_primary_value,
                        result.variant_primary_value,
                        result.improvement_pct,
                        result.p_value,
                        result.confidence_interval[0],
                        result.confidence_interval[1],
                        result.statistical_power,
                        result.effect_size,
                        json.dumps(result.secondary_results),
                        result.recommendation,
                        result.confidence_level,
                        result.analyzed_at.isoformat(),
                    ),
                )

        except Exception as e:
            self.logger.error(f"Error storing test result: {e}")

    def stop_test(self, test_id: str) -> bool:
        """Stop a running A/B test"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE ab_test_configs
                    SET status = ?, updated_at = ?
                    WHERE test_id = ?
                """,
                    (TestStatus.COMPLETED.value, datetime.now().isoformat(), test_id),
                )

            # Remove from active tests
            if test_id in self.active_tests:
                del self.active_tests[test_id]

            self.logger.info(f"Stopped A/B test: {test_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping test {test_id}: {e}")
            return False

    def _load_active_tests(self):
        """Load active test configurations"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT test_id, control_model_name, variant_model_name,
                           control_traffic, variant_traffic, min_sample_size,
                           max_duration_days, significance_level, minimum_effect_size,
                           primary_metric, secondary_metrics, early_stopping_enabled,
                           confidence_threshold
                    FROM ab_test_configs
                    WHERE status = ?
                """,
                    (TestStatus.RUNNING.value,),
                )

                self.active_tests = {}
                for row in cursor.fetchall():
                    test_id = row[0]
                    self.active_tests[test_id] = ABTestConfig(
                        test_id=test_id,
                        name=f"Test_{test_id}",
                        description="",
                        control_model_name=row[1],
                        variant_model_name=row[2],
                        control_traffic=row[3],
                        variant_traffic=row[4],
                        min_sample_size=row[5],
                        max_duration_days=row[6],
                        significance_level=row[7],
                        minimum_effect_size=row[8],
                        primary_metric=row[9],
                        secondary_metrics=json.loads(row[10]),
                        early_stopping_enabled=bool(row[11]),
                        confidence_threshold=row[12],
                    )

                self.logger.info(f"Loaded {len(self.active_tests)} active A/B tests")

        except Exception as e:
            self.logger.error(f"Error loading active tests: {e}")

    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of A/B test"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get test config
                cursor = conn.execute(
                    """
                    SELECT * FROM ab_test_configs WHERE test_id = ?
                """,
                    (test_id,),
                )
                config_row = cursor.fetchone()

                if not config_row:
                    return None

                # Get latest results
                cursor = conn.execute(
                    """
                    SELECT * FROM ab_test_results WHERE test_id = ?
                    ORDER BY analyzed_at DESC LIMIT 1
                """,
                    (test_id,),
                )
                result_row = cursor.fetchone()

                # Get observation counts
                cursor = conn.execute(
                    """
                    SELECT variant, COUNT(*) as count
                    FROM ab_test_observations
                    WHERE test_id = ?
                    GROUP BY variant
                """,
                    (test_id,),
                )
                observation_counts = dict(cursor.fetchall())

                return {
                    "test_id": test_id,
                    "status": config_row[16],  # status column
                    "control_model": config_row[3],
                    "variant_model": config_row[4],
                    "observation_counts": observation_counts,
                    "has_results": result_row is not None,
                    "last_analyzed": (result_row[17] if result_row else None),  # analyzed_at
                }

        except Exception as e:
            self.logger.error(f"Error getting test status: {e}")
            return None

    def list_tests(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tests with optional status filter"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT test_id, name, status, created_at FROM ab_test_configs"
                params = []

                if status_filter:
                    query += " WHERE status = ?"
                    params.append(status_filter)

                query += " ORDER BY created_at DESC"

                cursor = conn.execute(query, params)

                tests = []
                for row in cursor.fetchall():
                    test_status = self.get_test_status(row[0])
                    if test_status:
                        tests.append(
                            {
                                "test_id": row[0],
                                "name": row[1],
                                "status": row[2],
                                "created_at": row[3],
                                "details": test_status,
                            }
                        )

                return tests

        except Exception as e:
            self.logger.error(f"Error listing tests: {e}")
            return []
