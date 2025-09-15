"""
Comprehensive Test System for Enhanced Monitoring and Risk Management

Tests all new monitoring, risk management, and integration components
to ensure they work correctly together.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from integration import EnhancedTradingSystem
    from monitoring import (
        ABTestingFramework,
        DriftDetector,
        ModelPerformanceMonitor,
        PerformanceTracker,
    )
    from monitoring.ab_testing import ABTestConfig, TestStatus
    from risk_management import (
        DrawdownProtector,
        DynamicPositionSizer,
        PortfolioRiskManager,
        RiskCalculator,
    )
    from risk_management.portfolio_manager import PortfolioConstraints
    from risk_management.position_sizer import SizingMethod

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_SUCCESSFUL = False


class EnhancedSystemTester:
    """Comprehensive testing system for all new components"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.data_dir = Path("data/test")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Test data
        self.mock_returns = np.random.randn(252) * 0.02  # 1 year of daily returns
        self.mock_features = pd.DataFrame(
            np.random.randn(252, 15), columns=[f"feature_{i}" for i in range(15)]
        )
        self.mock_predictions = np.random.randn(252) * 0.01

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all test suites"""

        print("🧪 Starting Enhanced Trading System Tests")
        print("=" * 60)

        if not IMPORTS_SUCCESSFUL:
            print("❌ Cannot run tests due to import failures")
            return {"import_test": False}

        test_suites = [
            ("Model Performance Monitoring", self.test_model_monitoring),
            ("Drift Detection", self.test_drift_detection),
            ("Performance Tracking", self.test_performance_tracking),
            ("A/B Testing Framework", self.test_ab_testing),
            ("Risk Calculation", self.test_risk_calculation),
            ("Dynamic Position Sizing", self.test_position_sizing),
            ("Portfolio Risk Management", self.test_portfolio_management),
            ("Drawdown Protection", self.test_drawdown_protection),
            ("System Integration", self.test_system_integration),
        ]

        for test_name, test_func in test_suites:
            print(f"\\n🔍 Testing: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"   {status}")
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                self.test_results[test_name] = False

        # Print summary
        self.print_test_summary()

        return self.test_results

    def test_model_monitoring(self) -> bool:
        """Test model performance monitoring system"""

        try:
            # Initialize monitor
            monitor = ModelPerformanceMonitor(data_dir=self.data_dir)

            # Set reference data
            monitor.set_model_reference_data(
                model_name="test_model",
                features_df=self.mock_features,
                predictions=self.mock_predictions,
                targets=self.mock_returns,
            )

            # Monitor a prediction
            current_features = self.mock_features.tail(10)
            current_predictions = self.mock_predictions[-10:]
            current_targets = self.mock_returns[-10:]

            monitoring_result = monitor.monitor_model_prediction(
                model_name="test_model",
                features_df=current_features,
                predictions=current_predictions,
                targets=current_targets,
                user_id="test_user",
            )

            # Check results
            assert "status" in monitoring_result
            assert "alerts" in monitoring_result
            assert "timestamp" in monitoring_result

            # Generate report
            report = monitor.generate_comprehensive_report("test_model")
            assert report.model_name == "test_model"
            assert report.overall_status in ["healthy", "monitor", "investigate", "action_required"]

            print("     ✓ Monitor initialization successful")
            print("     ✓ Reference data setting successful")
            print("     ✓ Prediction monitoring successful")
            print("     ✓ Report generation successful")

            return True

        except Exception as e:
            print(f"     ✗ Model monitoring test failed: {e}")
            return False

    def test_drift_detection(self) -> bool:
        """Test drift detection system"""

        try:
            # Initialize drift detector
            drift_detector = DriftDetector()

            # Set reference data
            drift_detector.set_reference_data(
                features_df=self.mock_features,
                predictions=self.mock_predictions,
                targets=self.mock_returns,
            )

            # Test with slightly drifted data
            drifted_features = self.mock_features.tail(30) + np.random.randn(30, 15) * 0.5
            drifted_predictions = self.mock_predictions[-30:] + np.random.randn(30) * 0.02
            drifted_targets = self.mock_returns[-30:] + np.random.randn(30) * 0.01

            # Run comprehensive drift check
            drift_results = drift_detector.run_comprehensive_drift_check(
                current_features=drifted_features,
                current_predictions=drifted_predictions,
                current_targets=drifted_targets,
            )

            # Check results
            assert "data_drift" in drift_results
            assert "concept_drift" in drift_results
            assert "prediction_drift" in drift_results

            # Get drift summary
            summary = drift_detector.get_drift_summary(hours=24)
            assert "total_alerts" in summary
            assert "drift_types" in summary

            print("     ✓ Drift detector initialization successful")
            print("     ✓ Reference data setting successful")
            print("     ✓ Drift detection successful")
            print("     ✓ Summary generation successful")

            return True

        except Exception as e:
            print(f"     ✗ Drift detection test failed: {e}")
            return False

    def test_performance_tracking(self) -> bool:
        """Test performance tracking system"""

        try:
            # Initialize performance tracker
            tracker = PerformanceTracker(db_path=self.data_dir / "test_performance.db")

            # Record regression performance
            y_true = self.mock_returns[-50:]
            y_pred = self.mock_predictions[-50:]

            metrics = tracker.record_regression_performance(
                model_name="test_model", y_true=y_true, y_pred=y_pred, data_period="1h"
            )

            assert len(metrics) > 0
            assert metrics[0].model_name == "test_model"

            # Get performance summary
            summary = tracker.get_model_performance_summary("test_model", hours=24)
            assert summary["model_name"] == "test_model"
            assert "status" in summary
            assert "metrics" in summary

            # Record custom metrics
            custom_metrics = {"custom_metric_1": 0.85, "custom_metric_2": 0.92}

            custom_results = tracker.record_custom_metrics(
                model_name="test_model", metrics_dict=custom_metrics
            )

            assert len(custom_results) == 2

            print("     ✓ Performance tracker initialization successful")
            print("     ✓ Regression metrics recording successful")
            print("     ✓ Performance summary generation successful")
            print("     ✓ Custom metrics recording successful")

            return True

        except Exception as e:
            print(f"     ✗ Performance tracking test failed: {e}")
            return False

    def test_ab_testing(self) -> bool:
        """Test A/B testing framework"""

        try:
            # Initialize A/B testing framework
            ab_framework = ABTestingFramework(db_path=self.data_dir / "test_ab.db")

            # Create test configuration
            test_config = ABTestConfig(
                test_id="test_001",
                name="Model Comparison Test",
                description="Testing GRU vs LightGBM",
                control_model_name="gru_model",
                variant_model_name="lightgbm_model",
                min_sample_size=100,
                primary_metric="sharpe_ratio",
            )

            # Create test
            creation_result = ab_framework.create_test(test_config)
            assert creation_result == True

            # Start test
            start_result = ab_framework.start_test("test_001")
            assert start_result == True

            # Record observations
            for i in range(150):  # Enough for minimum sample size
                user_id = f"user_{i}"
                variant = ab_framework.assign_variant("test_001", user_id)

                if variant:
                    primary_metric_value = np.random.normal(
                        0.8 if variant == "variant" else 0.7, 0.1
                    )

                    ab_framework.record_observation(
                        test_id="test_001",
                        user_id=user_id,
                        variant=variant,
                        primary_metric_value=primary_metric_value,
                    )

            # Analyze test
            analysis_result = ab_framework.analyze_test("test_001", force_analysis=True)
            assert analysis_result is not None
            assert analysis_result.control_samples > 0
            assert analysis_result.variant_samples > 0

            # Get test status
            status = ab_framework.get_test_status("test_001")
            assert status is not None
            assert status["test_id"] == "test_001"

            print("     ✓ A/B framework initialization successful")
            print("     ✓ Test creation successful")
            print("     ✓ Test execution successful")
            print("     ✓ Statistical analysis successful")

            return True

        except Exception as e:
            print(f"     ✗ A/B testing test failed: {e}")
            return False

    def test_risk_calculation(self) -> bool:
        """Test risk calculation system"""

        try:
            # Initialize risk calculator
            risk_calc = RiskCalculator()

            # Calculate comprehensive risk
            risk_metrics = risk_calc.calculate_comprehensive_risk(returns=self.mock_returns)

            # Check risk metrics
            assert hasattr(risk_metrics, "var_1d_95")
            assert hasattr(risk_metrics, "cvar_1d_95")
            assert hasattr(risk_metrics, "realized_volatility")
            assert hasattr(risk_metrics, "max_drawdown")

            assert risk_metrics.var_1d_95 >= 0
            assert risk_metrics.cvar_1d_95 >= 0
            assert risk_metrics.realized_volatility >= 0

            # Test position risk calculation
            position_risk = risk_calc.calculate_position_risk(
                position_value=10000, asset_returns=self.mock_returns
            )

            assert "position_value" in position_risk
            assert "var_amount" in position_risk
            assert "cvar_amount" in position_risk

            # Test portfolio risk calculation
            positions = {"BTCEUR": 5000, "ETHEUR": 3000}
            asset_returns = {"BTCEUR": self.mock_returns, "ETHEUR": np.random.randn(252) * 0.03}

            portfolio_risk = risk_calc.calculate_portfolio_risk(
                positions=positions, asset_returns=asset_returns
            )

            assert "portfolio_metrics" in portfolio_risk
            assert "position_risks" in portfolio_risk
            assert "concentration_risk" in portfolio_risk

            print("     ✓ Risk calculator initialization successful")
            print("     ✓ Comprehensive risk calculation successful")
            print("     ✓ Position risk calculation successful")
            print("     ✓ Portfolio risk calculation successful")

            return True

        except Exception as e:
            print(f"     ✗ Risk calculation test failed: {e}")
            return False

    def test_position_sizing(self) -> bool:
        """Test dynamic position sizing system"""

        try:
            # Initialize position sizer
            sizer = DynamicPositionSizer()

            # Test different sizing methods
            sizing_methods = [
                SizingMethod.FIXED_FRACTIONAL,
                SizingMethod.KELLY_CRITERION,
                SizingMethod.VOLATILITY_ADJUSTED,
                SizingMethod.ADAPTIVE_VOLATILITY,
            ]

            for method in sizing_methods:
                result = sizer.calculate_position_size(
                    asset_returns=self.mock_returns,
                    predicted_return=0.01,  # 1% predicted return
                    current_portfolio_value=100000,
                    sizing_method=method,
                )

                assert hasattr(result, "recommended_size")
                assert hasattr(result, "confidence_score")
                assert hasattr(result, "sizing_method")
                assert result.sizing_method == method
                assert 0 <= result.recommended_size <= 1.0
                assert 0 <= result.confidence_score <= 1.0

            # Test portfolio sizing
            assets_data = {
                "BTCEUR": self.mock_returns,
                "ETHEUR": np.random.randn(252) * 0.03,
                "ADAEUR": np.random.randn(252) * 0.025,
            }

            portfolio_sizes = sizer.calculate_portfolio_sizes(
                assets_data=assets_data, sizing_method=SizingMethod.RISK_PARITY
            )

            assert len(portfolio_sizes) == 3
            for asset, result in portfolio_sizes.items():
                assert asset in assets_data
                assert hasattr(result, "recommended_size")

            print("     ✓ Position sizer initialization successful")
            print("     ✓ Individual sizing methods successful")
            print("     ✓ Portfolio sizing successful")

            return True

        except Exception as e:
            print(f"     ✗ Position sizing test failed: {e}")
            return False

    def test_portfolio_management(self) -> bool:
        """Test portfolio risk management system"""

        try:
            # Initialize portfolio manager
            constraints = PortfolioConstraints(max_single_position=0.25, max_portfolio_var_95=0.04)

            portfolio_mgr = PortfolioRiskManager(constraints=constraints)

            # Test portfolio risk analysis
            positions = {"BTCEUR": 20000, "ETHEUR": 15000, "ADAEUR": 10000}

            asset_returns = {
                "BTCEUR": self.mock_returns,
                "ETHEUR": np.random.randn(252) * 0.03,
                "ADAEUR": np.random.randn(252) * 0.025,
            }

            risk_analysis = portfolio_mgr.analyze_portfolio_risk(
                positions=positions, asset_returns=asset_returns, portfolio_value=100000
            )

            assert "portfolio_risk_metrics" in risk_analysis
            assert "risk_limits" in risk_analysis
            assert "correlation_analysis" in risk_analysis
            assert "overall_risk_status" in risk_analysis

            # Test trade compliance
            proposed_trade = {"BTCEUR": 5000}  # Add 5k to BTC

            compliance = portfolio_mgr.check_trade_compliance(
                proposed_trade=proposed_trade,
                current_positions=positions,
                portfolio_value=100000,
                asset_returns=asset_returns,
            )

            assert "compliance_status" in compliance
            assert compliance["compliance_status"] in ["approved", "rejected", "review_required"]

            print("     ✓ Portfolio manager initialization successful")
            print("     ✓ Portfolio risk analysis successful")
            print("     ✓ Trade compliance checking successful")

            return True

        except Exception as e:
            print(f"     ✗ Portfolio management test failed: {e}")
            return False

    def test_drawdown_protection(self) -> bool:
        """Test drawdown protection system"""

        try:
            # Initialize drawdown protector
            protector = DrawdownProtector(portfolio_high_water_mark=100000)

            # Simulate portfolio value changes
            test_values = [100000, 95000, 90000, 85000, 92000, 105000]

            for value in test_values:
                result = protector.update_portfolio_value(value)

                assert "portfolio_value" in result
                assert "current_drawdown_pct" in result
                assert "drawdown_level" in result
                assert "actions_taken" in result

            # Check trading permissions
            trading_check = protector.check_trading_allowed("new_position")
            assert "allowed" in trading_check
            assert "reason" in trading_check

            # Get protection status
            status = protector.get_protection_status()
            assert "protection_active" in status
            assert "trading_suspended" in status

            # Test drawdown statistics
            stats = protector.calculate_drawdown_statistics()
            assert "status" in stats

            print("     ✓ Drawdown protector initialization successful")
            print("     ✓ Portfolio value tracking successful")
            print("     ✓ Trading permission checking successful")
            print("     ✓ Statistics calculation successful")

            return True

        except Exception as e:
            print(f"     ✗ Drawdown protection test failed: {e}")
            return False

    def test_system_integration(self) -> bool:
        """Test integrated trading system"""

        try:
            # Initialize enhanced trading system
            trading_system = EnhancedTradingSystem(
                data_dir=self.data_dir, initial_portfolio_value=100000
            )

            # Start trading session
            session_result = trading_system.start_trading_session(
                symbols=["BTCEUR", "ETHEUR"], enable_live_trading=False
            )

            assert session_result["status"] == "success"
            assert "session_id" in session_result

            # Process trading opportunity
            market_data = {"price": 50000, "volume": 1000, "timestamp": datetime.now()}

            # Use asyncio.run to handle async function
            decision = asyncio.run(
                trading_system.process_trading_opportunity(
                    symbol="BTCEUR", market_data=market_data, user_id="test_user"
                )
            )

            assert hasattr(decision, "asset")
            assert hasattr(decision, "action")
            assert hasattr(decision, "confidence_score")
            assert decision.asset == "BTCEUR"

            # Get system status
            status = trading_system.get_system_status()
            assert "session_active" in status
            assert "portfolio_value" in status
            assert "monitoring" in status
            assert "risk_management" in status

            # Stop trading session
            stop_result = trading_system.stop_trading_session()
            assert stop_result["status"] == "success"

            print("     ✓ Trading system initialization successful")
            print("     ✓ Trading session management successful")
            print("     ✓ Trading opportunity processing successful")
            print("     ✓ System status reporting successful")

            return True

        except Exception as e:
            print(f"     ✗ System integration test failed: {e}")
            return False

    def print_test_summary(self):
        """Print comprehensive test summary"""

        print("\\n" + "=" * 60)
        print("🧪 TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<35} {status}")

        print("-" * 60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")

        if passed == total:
            print("\\n🎉 ALL TESTS PASSED! Enhanced trading system is ready.")
        else:
            print(f"\\n⚠️  {total - passed} tests failed. Please review and fix issues.")

        return passed == total


def main():
    """Run the comprehensive test suite"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and run tests
    tester = EnhancedSystemTester()
    success = tester.run_all_tests()

    # Save test results
    results_file = Path("data/test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "results": tester.test_results,
                "overall_success": all(tester.test_results.values()),
            },
            f,
            indent=2,
        )

    print(f"\\n📄 Test results saved to: {results_file}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
