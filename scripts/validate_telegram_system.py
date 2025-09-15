#!/usr/bin/env python3
"""
Telegram System Validation Script
Comprehensive validation and testing of the new unified Telegram system.
"""

import asyncio
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add bot directory to path
bot_dir = Path(__file__).parent.parent
sys.path.insert(0, str(bot_dir))


# Color codes for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"{Colors.GREEN}✓{Colors.END} {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"{Colors.RED}✗{Colors.END} {test_name}: {Colors.RED}{error}{Colors.END}")

    def add_warning(self, test_name: str, warning: str):
        self.warnings += 1
        print(f"{Colors.YELLOW}⚠{Colors.END} {test_name}: {Colors.YELLOW}{warning}{Colors.END}")

    def print_summary(self):
        total = self.passed + self.failed + self.warnings
        print(f"\n{Colors.BOLD}Validation Summary:{Colors.END}")
        print(f"  {Colors.GREEN}Passed:{Colors.END} {self.passed}/{total}")
        if self.failed > 0:
            print(f"  {Colors.RED}Failed:{Colors.END} {self.failed}/{total}")
        if self.warnings > 0:
            print(f"  {Colors.YELLOW}Warnings:{Colors.END} {self.warnings}/{total}")

        if self.failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All critical tests passed!{Colors.END}")
            return True
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}💥 {self.failed} critical test(s) failed{Colors.END}")
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  • {error}")
            return False


async def validate_imports():
    """Test that all new modules can be imported."""
    print(f"{Colors.BOLD}{Colors.BLUE}Testing Module Imports{Colors.END}")
    result = ValidationResult()

    imports_to_test = [
        ("src.security", "get_credential_manager, TelegramCredentials"),
        ("src.notifications.core", "get_telegram_client, MessageQueue, MessagePriority"),
        ("src.notifications.core", "get_command_registry, telegram_command"),
        ("src.notifications", "get_telegram_service, TelegramService"),
        ("src.notifications.integrations.trader_integration", "TradingBotIntegration"),
        ("src.adapters.telegram_adapter", "get_telegram_adapter, TelegramAdapter"),
        ("src.config.telegram_config_manager", "get_telegram_config_manager"),
        ("src.monitoring.telegram_monitor", "get_telegram_monitor"),
    ]

    for module_name, imports in imports_to_test:
        try:
            exec(f"from {module_name} import {imports}")
            result.add_pass(f"Import {module_name}")
        except ImportError as e:
            result.add_fail(f"Import {module_name}", str(e))
        except Exception as e:
            result.add_fail(f"Import {module_name}", f"Unexpected error: {e}")

    return result


async def validate_credential_system():
    """Test the credential management system."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Testing Credential System{Colors.END}")
    result = ValidationResult()

    try:
        from src.security import TelegramCredentials, get_credential_manager

        # Test credential manager creation
        manager = get_credential_manager()
        result.add_pass("Credential manager creation")

        # Test validation with empty environment
        with patch_dict_context(os.environ, {}, clear=True):
            validation = manager.validate_environment()
            if not validation["valid"]:
                result.add_pass("Environment validation (empty env)")
            else:
                result.add_fail("Environment validation", "Should fail with empty environment")

        # Test Telegram credentials validation
        try:
            valid_creds = TelegramCredentials(
                bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz", chat_id="987654321"
            )
            result.add_pass("Valid Telegram credentials")
        except Exception as e:
            result.add_fail("Valid Telegram credentials", str(e))

        # Test invalid credentials
        try:
            TelegramCredentials(bot_token="invalid", chat_id="123")
            result.add_fail("Invalid credentials validation", "Should have raised ValueError")
        except ValueError:
            result.add_pass("Invalid credentials validation")
        except Exception as e:
            result.add_fail("Invalid credentials validation", f"Wrong exception: {e}")

    except Exception as e:
        result.add_fail("Credential system test", f"Unexpected error: {e}")
        traceback.print_exc()

    return result


async def validate_message_queue():
    """Test the message queue system."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Testing Message Queue{Colors.END}")
    result = ValidationResult()

    try:
        from src.notifications.core import MessagePriority, MessageQueue

        # Create temporary queue file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            temp_queue_file = tmp_file.name

        try:
            # Test queue creation
            queue = MessageQueue(
                queue_file=temp_queue_file, max_queue_size=5, persistence_enabled=True
            )
            result.add_pass("Message queue creation")

            # Test enqueue/dequeue
            success = await queue.enqueue("Test message", MessagePriority.NORMAL)
            if success:
                result.add_pass("Message enqueue")
            else:
                result.add_fail("Message enqueue", "Enqueue returned False")

            message = await queue.dequeue()
            if message and message.message == "Test message":
                result.add_pass("Message dequeue")
            else:
                result.add_fail("Message dequeue", f"Got: {message}")

            # Test priority ordering
            await queue.enqueue("Low priority", MessagePriority.LOW)
            await queue.enqueue("High priority", MessagePriority.HIGH)
            await queue.enqueue("Critical priority", MessagePriority.CRITICAL)

            msg1 = await queue.dequeue()
            msg2 = await queue.dequeue()
            msg3 = await queue.dequeue()

            if (
                msg1.message == "Critical priority"
                and msg2.message == "High priority"
                and msg3.message == "Low priority"
            ):
                result.add_pass("Priority ordering")
            else:
                result.add_fail(
                    "Priority ordering",
                    f"Wrong order: {msg1.message}, {msg2.message}, {msg3.message}",
                )

            # Test queue status
            status = await queue.get_queue_status()
            if isinstance(status, dict) and "queue_size" in status:
                result.add_pass("Queue status")
            else:
                result.add_fail("Queue status", f"Invalid status format: {status}")

        finally:
            # Cleanup
            try:
                os.unlink(temp_queue_file)
            except:
                pass

    except Exception as e:
        result.add_fail("Message queue test", f"Unexpected error: {e}")
        traceback.print_exc()

    return result


async def validate_command_registry():
    """Test the command registry system."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Testing Command Registry{Colors.END}")
    result = ValidationResult()

    try:
        from src.notifications.core import CommandRegistry, get_command_registry

        # Test registry creation
        registry = CommandRegistry()
        result.add_pass("Command registry creation")

        # Test command registration
        async def test_command(update, context):
            pass

        success = registry.register_command(
            name="test", handler=test_command, description="Test command", admin_only=False
        )

        if success:
            result.add_pass("Command registration")
        else:
            result.add_fail("Command registration", "Registration returned False")

        # Test command list
        commands = registry.get_command_list()
        if len(commands) > 0 and commands[0]["name"] == "test":
            result.add_pass("Command listing")
        else:
            result.add_fail("Command listing", f"Wrong commands: {commands}")

        # Test statistics
        stats = registry.get_statistics()
        if isinstance(stats, dict) and "total_commands" in stats:
            result.add_pass("Registry statistics")
        else:
            result.add_fail("Registry statistics", f"Invalid stats: {stats}")

    except Exception as e:
        result.add_fail("Command registry test", f"Unexpected error: {e}")
        traceback.print_exc()

    return result


async def validate_configuration_system():
    """Test the configuration management system."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Testing Configuration System{Colors.END}")
    result = ValidationResult()

    try:
        from src.config.telegram_config_manager import get_telegram_config_manager

        # Test config manager creation
        config_manager = get_telegram_config_manager()
        result.add_pass("Config manager creation")

        # Test config loading
        config = config_manager.get_config()
        if config and hasattr(config, "service"):
            result.add_pass("Config loading")
        else:
            result.add_fail("Config loading", f"Invalid config: {config}")

        # Test config summary
        summary = config_manager.get_config_summary()
        if isinstance(summary, dict) and "service_enabled" in summary:
            result.add_pass("Config summary")
        else:
            result.add_fail("Config summary", f"Invalid summary: {summary}")

        # Test environment overrides
        overrides = config_manager.get_environment_overrides()
        if isinstance(overrides, dict):
            result.add_pass("Environment overrides")
        else:
            result.add_fail("Environment overrides", f"Invalid overrides: {overrides}")

    except Exception as e:
        result.add_fail("Configuration system test", f"Unexpected error: {e}")
        traceback.print_exc()

    return result


async def validate_integration_layer():
    """Test the trading integration layer."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Testing Integration Layer{Colors.END}")
    result = ValidationResult()

    try:
        from src.adapters.telegram_adapter import TelegramAdapter, get_telegram_adapter
        from src.notifications.integrations.trader_integration import (
            TradeSignificance,
            TradingBotIntegration,
        )

        # Test integration creation
        integration = TradingBotIntegration()
        result.add_pass("Trading integration creation")

        # Test trade significance assessment
        trade_data = {
            "symbol": "BTCEUR",
            "side": "BUY",
            "quantity": 0.1,
            "price": 40000.0,
            "confidence": 0.95,
            "realized_pnl": 0,
        }

        significance = integration._assess_trade_significance(trade_data)
        if significance == TradeSignificance.CRITICAL:
            result.add_pass("Trade significance assessment")
        else:
            result.add_fail(
                "Trade significance assessment", f"Expected CRITICAL, got {significance}"
            )

        # Test adapter creation
        adapter = TelegramAdapter()
        result.add_pass("Telegram adapter creation")

        # Test adapter configuration
        config = adapter.get_config()
        if isinstance(config, dict) and "enabled" in config:
            result.add_pass("Adapter configuration")
        else:
            result.add_fail("Adapter configuration", f"Invalid config: {config}")

        # Test singleton adapter
        adapter2 = get_telegram_adapter()
        if adapter is adapter2:
            result.add_pass("Adapter singleton")
        else:
            result.add_warning("Adapter singleton", "Different instances returned")

    except Exception as e:
        result.add_fail("Integration layer test", f"Unexpected error: {e}")
        traceback.print_exc()

    return result


async def validate_monitoring_system():
    """Test the monitoring system."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Testing Monitoring System{Colors.END}")
    result = ValidationResult()

    try:
        from src.monitoring.telegram_monitor import TelegramSystemMonitor, get_telegram_monitor

        # Test monitor creation
        monitor = get_telegram_monitor()
        result.add_pass("Monitor creation")

        # Test health check registration
        async def dummy_health_check():
            from src.monitoring.telegram_monitor import HealthCheck, HealthStatus

            return HealthCheck("test", HealthStatus.HEALTHY, "Test OK")

        monitor.register_health_check("test_check", dummy_health_check)
        result.add_pass("Health check registration")

        # Test performance recording
        monitor.record_message_sent()
        monitor.record_command_executed()
        result.add_pass("Performance recording")

        # Test monitoring status
        status = monitor.get_monitoring_status()
        if isinstance(status, dict) and "running" in status:
            result.add_pass("Monitoring status")
        else:
            result.add_fail("Monitoring status", f"Invalid status: {status}")

        # Test metrics summary
        metrics = monitor.get_metrics_summary()
        if isinstance(metrics, dict) and "performance_stats" in metrics:
            result.add_pass("Metrics summary")
        else:
            result.add_fail("Metrics summary", f"Invalid metrics: {metrics}")

    except Exception as e:
        result.add_fail("Monitoring system test", f"Unexpected error: {e}")
        traceback.print_exc()

    return result


async def validate_file_structure():
    """Validate unified Telegram file structure and absence of deprecated files."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Validating File Structure{Colors.END}")
    result = ValidationResult()

    required_files = [
        "src/security/credential_manager.py",
        "src/security/__init__.py",
        "src/notifications/core/telegram_client.py",
        "src/notifications/core/message_queue.py",
        "src/notifications/core/command_registry.py",
        "src/notifications/core/__init__.py",
        "src/notifications/telegram_service.py",
        "src/notifications/handlers/trading_commands.py",
        "src/notifications/handlers/system_commands.py",
        "src/notifications/handlers/admin_commands.py",
        "src/notifications/handlers/__init__.py",
        "src/notifications/integrations/trader_integration.py",
        "src/notifications/integrations/__init__.py",
        "src/notifications/__init__.py",
        "src/adapters/telegram_adapter.py",
        "src/config/telegram_config_manager.py",
        "src/monitoring/telegram_monitor.py",
        # Optional config and tests may be absent in production
        "bin/telegram_bot",
    ]

    for file_path in required_files:
        full_path = bot_dir / file_path
        if full_path.exists():
            result.add_pass(f"File exists: {file_path}")
        else:
            result.add_fail(f"File missing: {file_path}", "File not found")

    # Ensure deprecated files are fully removed
    removed_files = [
        "src/notifications/telegram_bot.py",
        "src/notifications/unified_telegram_system.py",
        "src/notifications/enhanced_telegram.py",
        "src/notifications/bot_singleton.py",
        "src/notifications/telegram_integration.py",
        "src/notifier/enhanced_telegram.py",
        "src/notifier/telegram_notifier.py",
    ]

    for file_path in removed_files:
        if (bot_dir / file_path).exists() or (bot_dir / f"{file_path}.deprecated").exists():
            result.add_fail(f"Deprecated file present: {file_path}", "Should be removed")
        else:
            result.add_pass(f"Deprecated file removed: {file_path}")

    return result


async def validate_launcher_script():
    """Test the updated launcher script."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Testing Launcher Script{Colors.END}")
    result = ValidationResult()

    try:
        launcher_path = bot_dir / "bin/telegram_bot"

        if launcher_path.exists():
            result.add_pass("Launcher script exists")

            # Check if it's executable
            if os.access(launcher_path, os.X_OK):
                result.add_pass("Launcher script is executable")
            else:
                result.add_warning("Launcher script", "Not executable")

            # Check content for new imports
            with open(launcher_path, "r") as f:
                content = f.read()

            if "from src.notifications import get_telegram_service" in content:
                result.add_pass("Launcher uses new unified system")
            else:
                result.add_fail("Launcher script", "Still uses old system")

            if "Unified Telegram Service" in content:
                result.add_pass("Launcher updated description")
            else:
                result.add_warning("Launcher script", "Description not updated")

        else:
            result.add_fail("Launcher script", "Script not found")

    except Exception as e:
        result.add_fail("Launcher script test", f"Unexpected error: {e}")
        traceback.print_exc()

    return result


# Helper context manager for patching os.environ
class patch_dict_context:
    def __init__(self, dict_obj, values, clear=False):
        self.dict_obj = dict_obj
        self.values = values
        self.clear = clear
        self.original = {}

    def __enter__(self):
        # Save original values
        for key in self.values:
            if key in self.dict_obj:
                self.original[key] = self.dict_obj[key]

        if self.clear:
            # Save all original values if clearing
            self.original = dict(self.dict_obj)
            self.dict_obj.clear()

        # Set new values
        self.dict_obj.update(self.values)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clear:
            # Restore everything
            self.dict_obj.clear()
            self.dict_obj.update(self.original)
        else:
            # Remove added values and restore original ones
            for key in self.values:
                if key in self.original:
                    self.dict_obj[key] = self.original[key]
                else:
                    self.dict_obj.pop(key, None)


async def main():
    """Run all validation tests."""
    print(f"{Colors.BOLD}{Colors.CYAN}🔍 Telegram System Validation{Colors.END}")
    print(f"{Colors.CYAN}Testing the new unified Telegram notification system{Colors.END}")
    print(f"Started at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Run all validation tests
    tests = [
        ("File Structure", validate_file_structure),
        ("Module Imports", validate_imports),
        ("Credential System", validate_credential_system),
        ("Message Queue", validate_message_queue),
        ("Command Registry", validate_command_registry),
        ("Configuration System", validate_configuration_system),
        ("Integration Layer", validate_integration_layer),
        ("Monitoring System", validate_monitoring_system),
        ("Launcher Script", validate_launcher_script),
    ]

    overall_result = ValidationResult()

    for test_name, test_func in tests:
        try:
            result = await test_func()
            overall_result.passed += result.passed
            overall_result.failed += result.failed
            overall_result.warnings += result.warnings
            overall_result.errors.extend(result.errors)
        except Exception as e:
            overall_result.add_fail(f"{test_name} (Exception)", str(e))
            traceback.print_exc()

    # Print final summary
    success = overall_result.print_summary()

    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Telegram System Validation PASSED{Colors.END}")
        print(f"{Colors.GREEN}The unified Telegram system is ready for deployment!{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Telegram System Validation FAILED{Colors.END}")
        print(f"{Colors.RED}Please fix the identified issues before deploying.{Colors.END}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with unexpected error: {e}{Colors.END}")
        traceback.print_exc()
        sys.exit(1)
