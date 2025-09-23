"""
Telegram System Monitor
Production-grade monitoring and health checking for the unified Telegram system.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

from src.config.telegram_config_manager import get_telegram_config_manager
from src.core.logging_manager import get_system_logger


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelegramSystemMonitor:
    """
    Comprehensive monitoring system for Telegram service.

    Features:
    - Health checks for all components
    - Performance metrics collection
    - Alerting and notification
    - Automatic recovery attempts
    - Metrics persistence and analysis
    """

    def __init__(self):
        self.logger = get_system_logger(__name__)
        self.config_manager = get_telegram_config_manager()

        # Monitoring state
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None

        # Health checks
        self._health_checks: Dict[str, Callable] = {}
        self._last_health_results: Dict[str, HealthCheck] = {}

        # Metrics storage
        self._metrics: Dict[str, List[MetricPoint]] = {}
        self._max_metric_points = 1000

        # Performance tracking
        self._performance_stats = {
            "messages_sent": 0,
            "messages_failed": 0,
            "commands_executed": 0,
            "errors_count": 0,
            "average_response_time": 0.0,
            "uptime_start": datetime.now(timezone.utc),
        }

        # Alerting
        self._alert_handlers: List[Callable] = []
        self._alert_history: List[Dict[str, Any]] = []
        self._max_alert_history = 100

        # Initialize built-in health checks
        self._setup_builtin_health_checks()

    def _setup_builtin_health_checks(self):
        """Setup built-in health checks."""
        self.register_health_check("telegram_client", self._check_telegram_client)
        self.register_health_check("message_queue", self._check_message_queue)
        self.register_health_check("command_registry", self._check_command_registry)
        self.register_health_check("credentials", self._check_credentials)
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("network_connectivity", self._check_network_connectivity)
        self.register_health_check("disk_space", self._check_disk_space)

    async def start_monitoring(self) -> bool:
        """
        Start the monitoring system.

        Returns:
            bool: True if monitoring started successfully
        """
        if self._running:
            self.logger.warning("Monitoring already running")
            return True

        try:
            self.logger.info("Starting Telegram system monitoring...")

            config = self.config_manager.get_config()
            if not config.monitoring.enabled:
                self.logger.info("Monitoring disabled in configuration")
                return False

            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("Telegram system monitoring started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self._running = False
            return False

    async def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self._running:
            return

        self.logger.info("Stopping Telegram system monitoring...")

        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Telegram system monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        config = self.config_manager.get_config()
        check_interval = config.service.health_check_interval

        self.logger.info(f"Monitoring loop started with {check_interval}s interval")

        try:
            while self._running:
                try:
                    # Run health checks
                    await self._run_health_checks()

                    # Collect metrics
                    await self._collect_metrics()

                    # Check for alerts
                    await self._process_alerts()

                    # Cleanup old data
                    await self._cleanup_old_data()

                    # Wait for next cycle
                    await asyncio.sleep(check_interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(min(check_interval, 60))  # Don't overwhelm on errors

        except Exception as e:
            self.logger.error(f"Monitoring loop crashed: {e}")
        finally:
            self.logger.info("Monitoring loop ended")

    async def _run_health_checks(self):
        """Execute all registered health checks."""
        try:
            for name, check_func in self._health_checks.items():
                try:
                    result = await check_func()
                    self._last_health_results[name] = result

                    # Log critical issues
                    if result.status == HealthStatus.CRITICAL:
                        self.logger.error(f"CRITICAL health check {name}: {result.message}")

                except Exception as e:
                    self.logger.error(f"Health check {name} failed: {e}")
                    self._last_health_results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.CRITICAL,
                        message=f"Health check failed: {e}",
                    )

        except Exception as e:
            self.logger.error(f"Error running health checks: {e}")

    async def _collect_metrics(self):
        """Collect system metrics."""
        try:
            timestamp = datetime.now(timezone.utc)

            # System metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            self._record_metric("cpu_usage_percent", cpu_percent, timestamp)
            self._record_metric("memory_usage_percent", memory.percent, timestamp)
            self._record_metric("disk_usage_percent", disk.used / disk.total * 100, timestamp)

            # Performance metrics
            uptime_seconds = (timestamp - self._performance_stats["uptime_start"]).total_seconds()
            self._record_metric("uptime_seconds", uptime_seconds, timestamp)
            self._record_metric(
                "messages_sent_total",
                self._performance_stats["messages_sent"],
                timestamp,
            )
            self._record_metric(
                "messages_failed_total",
                self._performance_stats["messages_failed"],
                timestamp,
            )
            self._record_metric(
                "commands_executed_total",
                self._performance_stats["commands_executed"],
                timestamp,
            )
            self._record_metric(
                "errors_count_total", self._performance_stats["errors_count"], timestamp
            )

            # Success rate metrics
            total_messages = (
                self._performance_stats["messages_sent"]
                + self._performance_stats["messages_failed"]
            )
            if total_messages > 0:
                success_rate = (self._performance_stats["messages_sent"] / total_messages) * 100
                self._record_metric("message_success_rate_percent", success_rate, timestamp)

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")

    def _record_metric(
        self,
        name: str,
        value: float,
        timestamp: datetime,
        metadata: Dict[str, Any] = None,
    ):
        """Record a metric point."""
        if name not in self._metrics:
            self._metrics[name] = []

        point = MetricPoint(timestamp=timestamp, value=value, metadata=metadata or {})
        self._metrics[name].append(point)

        # Keep only recent points
        if len(self._metrics[name]) > self._max_metric_points:
            self._metrics[name] = self._metrics[name][-self._max_metric_points :]

    async def _process_alerts(self):
        """Process and send alerts based on health checks and metrics."""
        try:
            config = self.config_manager.get_config()

            # Check for alert conditions
            alerts = []

            # Health-based alerts
            for name, health_check in self._last_health_results.items():
                if health_check.status == HealthStatus.CRITICAL:
                    alerts.append(
                        {
                            "type": "health_critical",
                            "component": name,
                            "message": health_check.message,
                            "details": health_check.details,
                        }
                    )
                elif health_check.status == HealthStatus.WARNING:
                    alerts.append(
                        {
                            "type": "health_warning",
                            "component": name,
                            "message": health_check.message,
                            "details": health_check.details,
                        }
                    )

            # Metric-based alerts
            if "memory_usage_percent" in self._metrics:
                latest_memory = self._metrics["memory_usage_percent"][-1].value
                if latest_memory > config.monitoring.max_memory_usage_warning:
                    alerts.append(
                        {
                            "type": "high_memory_usage",
                            "component": "system",
                            "message": f"Memory usage at {latest_memory:.1f}%",
                            "details": {"memory_percent": latest_memory},
                        }
                    )

            if "message_success_rate_percent" in self._metrics:
                latest_success_rate = self._metrics["message_success_rate_percent"][-1].value
                if latest_success_rate < 90:  # Alert if success rate drops below 90%
                    alerts.append(
                        {
                            "type": "low_success_rate",
                            "component": "telegram_client",
                            "message": f"Message success rate at {latest_success_rate:.1f}%",
                            "details": {"success_rate": latest_success_rate},
                        }
                    )

            # Send alerts
            for alert in alerts:
                await self._send_alert(alert)

        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")

    async def _send_alert(self, alert: Dict[str, Any]):
        """Send an alert notification."""
        try:
            alert["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Add to history
            self._alert_history.append(alert)
            if len(self._alert_history) > self._max_alert_history:
                self._alert_history = self._alert_history[-self._max_alert_history :]

            # Send to registered handlers
            for handler in self._alert_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")

            # Log alert
            alert_level = "ERROR" if alert["type"].endswith("critical") else "WARNING"
            self.logger.log(
                getattr(logging, alert_level),
                f"ALERT [{alert['type']}] {alert['component']}: {alert['message']}",
            )

        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")

    async def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

            # Clean up metrics
            for metric_name, points in self._metrics.items():
                self._metrics[metric_name] = [
                    point for point in points if point.timestamp > cutoff_time
                ]

            # Clean up old health results
            for name, health_check in list(self._last_health_results.items()):
                if health_check.timestamp < cutoff_time:
                    del self._last_health_results[name]

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    # Health check implementations

    async def _check_telegram_client(self) -> HealthCheck:
        """Check Telegram client health."""
        try:
            from src.notifications.core import get_telegram_client

            client = get_telegram_client()

            if not client.is_initialized:
                return HealthCheck(
                    name="telegram_client",
                    status=HealthStatus.CRITICAL,
                    message="Telegram client not initialized",
                )

            if not client.is_healthy:
                return HealthCheck(
                    name="telegram_client",
                    status=HealthStatus.WARNING,
                    message="Telegram client health issues detected",
                    details=client.get_health_status(),
                )

            return HealthCheck(
                name="telegram_client",
                status=HealthStatus.HEALTHY,
                message="Telegram client operational",
                details=client.get_health_status(),
            )

        except Exception as e:
            return HealthCheck(
                name="telegram_client",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check Telegram client: {e}",
            )

    async def _check_message_queue(self) -> HealthCheck:
        """Check message queue health."""
        try:
            from src.notifications import get_telegram_service

            service = get_telegram_service()
            queue_status = await service.message_queue.get_queue_status()

            config = self.config_manager.get_config()

            # Check queue size
            if queue_status["queue_size"] > config.monitoring.max_queue_size_warning:
                return HealthCheck(
                    name="message_queue",
                    status=HealthStatus.WARNING,
                    message=f"Queue size high: {queue_status['queue_size']} messages",
                    details=queue_status,
                )

            # Check for dead letters
            if queue_status["dead_letter_size"] > 10:
                return HealthCheck(
                    name="message_queue",
                    status=HealthStatus.WARNING,
                    message=f"High dead letter count: {queue_status['dead_letter_size']}",
                    details=queue_status,
                )

            return HealthCheck(
                name="message_queue",
                status=HealthStatus.HEALTHY,
                message="Message queue operational",
                details=queue_status,
            )

        except Exception as e:
            return HealthCheck(
                name="message_queue",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check message queue: {e}",
            )

    async def _check_command_registry(self) -> HealthCheck:
        """Check command registry health."""
        try:
            from src.notifications.core import get_command_registry

            registry = get_command_registry()
            stats = registry.get_statistics()

            if stats["enabled_commands"] == 0:
                return HealthCheck(
                    name="command_registry",
                    status=HealthStatus.WARNING,
                    message="No commands enabled",
                    details=stats,
                )

            return HealthCheck(
                name="command_registry",
                status=HealthStatus.HEALTHY,
                message=f"{stats['enabled_commands']} commands available",
                details=stats,
            )

        except Exception as e:
            return HealthCheck(
                name="command_registry",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check command registry: {e}",
            )

    async def _check_credentials(self) -> HealthCheck:
        """Check credential status."""
        try:
            from src.security import get_credential_manager

            cred_manager = get_credential_manager()
            validation = cred_manager.validate_environment()

            if not validation["valid"]:
                return HealthCheck(
                    name="credentials",
                    status=HealthStatus.CRITICAL,
                    message=f"Credential validation failed: {validation['errors']}",
                    details=validation,
                )

            if validation["warnings"]:
                return HealthCheck(
                    name="credentials",
                    status=HealthStatus.WARNING,
                    message=f"Credential warnings: {validation['warnings']}",
                    details=validation,
                )

            return HealthCheck(
                name="credentials",
                status=HealthStatus.HEALTHY,
                message="Credentials valid",
                details={"credentials_found": validation["credentials_found"]},
            )

        except Exception as e:
            return HealthCheck(
                name="credentials",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check credentials: {e}",
            )

    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            config = self.config_manager.get_config()

            issues = []
            status = HealthStatus.HEALTHY

            if memory.percent > config.monitoring.max_memory_usage_warning:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                status = HealthStatus.WARNING

            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = HealthStatus.WARNING

            # Check for system overload
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()[0]
                cpu_count = psutil.cpu_count()
                if load_avg > cpu_count * 2:
                    issues.append(f"System overloaded: load {load_avg:.2f}")
                    status = HealthStatus.CRITICAL

            message = "System resources normal" if not issues else "; ".join(issues)

            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "memory_percent": memory.percent,
                    "cpu_percent": cpu_percent,
                    "load_average": load_avg if "load_avg" in locals() else None,
                },
            )

        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}",
            )

    async def _check_network_connectivity(self) -> HealthCheck:
        """Check network connectivity."""
        try:
            import socket

            # Test connectivity to Telegram API
            try:
                socket.create_connection(("api.telegram.org", 443), timeout=10)
                telegram_reachable = True
            except:
                telegram_reachable = False

            # Test general internet connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=5)
                internet_reachable = True
            except:
                internet_reachable = False

            if not telegram_reachable:
                return HealthCheck(
                    name="network_connectivity",
                    status=HealthStatus.CRITICAL,
                    message="Cannot reach Telegram API",
                    details={
                        "telegram_reachable": telegram_reachable,
                        "internet_reachable": internet_reachable,
                    },
                )

            if not internet_reachable:
                return HealthCheck(
                    name="network_connectivity",
                    status=HealthStatus.WARNING,
                    message="Limited internet connectivity",
                    details={
                        "telegram_reachable": telegram_reachable,
                        "internet_reachable": internet_reachable,
                    },
                )

            return HealthCheck(
                name="network_connectivity",
                status=HealthStatus.HEALTHY,
                message="Network connectivity good",
                details={
                    "telegram_reachable": telegram_reachable,
                    "internet_reachable": internet_reachable,
                },
            )

        except Exception as e:
            return HealthCheck(
                name="network_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check network connectivity: {e}",
            )

    async def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            if disk_percent > 95:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.CRITICAL,
                    message=f"Disk space critical: {disk_percent:.1f}% used",
                    details={
                        "disk_percent": disk_percent,
                        "free_gb": disk.free / (1024**3),
                        "total_gb": disk.total / (1024**3),
                    },
                )

            if disk_percent > 85:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.WARNING,
                    message=f"Disk space low: {disk_percent:.1f}% used",
                    details={
                        "disk_percent": disk_percent,
                        "free_gb": disk.free / (1024**3),
                        "total_gb": disk.total / (1024**3),
                    },
                )

            return HealthCheck(
                name="disk_space",
                status=HealthStatus.HEALTHY,
                message=f"Disk space adequate: {disk_percent:.1f}% used",
                details={
                    "disk_percent": disk_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3),
                },
            )

        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {e}",
            )

    # Public API methods

    def register_health_check(self, name: str, check_func: Callable):
        """Register a custom health check."""
        self._health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")

    def register_alert_handler(self, handler: Callable):
        """Register an alert handler."""
        self._alert_handlers.append(handler)
        self.logger.info("Registered alert handler")

    def record_message_sent(self):
        """Record a successful message send."""
        self._performance_stats["messages_sent"] += 1

    def record_message_failed(self):
        """Record a failed message send."""
        self._performance_stats["messages_failed"] += 1

    def record_command_executed(self):
        """Record a command execution."""
        self._performance_stats["commands_executed"] += 1

    def record_error(self):
        """Record an error occurrence."""
        self._performance_stats["errors_count"] += 1

    def record_response_time(self, response_time_ms: float):
        """Record response time."""
        # Update moving average
        current_avg = self._performance_stats["average_response_time"]
        self._performance_stats["average_response_time"] = (current_avg + response_time_ms) / 2

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        overall_status = HealthStatus.HEALTHY
        critical_count = 0
        warning_count = 0

        for health_check in self._last_health_results.values():
            if health_check.status == HealthStatus.CRITICAL:
                critical_count += 1
                overall_status = HealthStatus.CRITICAL
            elif health_check.status == HealthStatus.WARNING:
                warning_count += 1
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING

        return {
            "overall_status": overall_status.value,
            "critical_issues": critical_count,
            "warnings": warning_count,
            "total_checks": len(self._last_health_results),
            "last_check": max(
                (hc.timestamp for hc in self._last_health_results.values()),
                default=None,
            ),
            "checks": {
                name: {
                    "status": hc.status.value,
                    "message": hc.message,
                    "timestamp": hc.timestamp.isoformat(),
                }
                for name, hc in self._last_health_results.items()
            },
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "performance_stats": self._performance_stats.copy(),
            "recent_metrics": {
                name: {
                    "latest_value": points[-1].value if points else None,
                    "data_points": len(points),
                    "latest_timestamp": (points[-1].timestamp.isoformat() if points else None),
                }
                for name, points in self._metrics.items()
            },
        }

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        return {
            "running": self._running,
            "uptime": (
                (
                    datetime.now(timezone.utc) - self._performance_stats["uptime_start"]
                ).total_seconds()
                if self._running
                else 0
            ),
            "health_checks_registered": len(self._health_checks),
            "alert_handlers_registered": len(self._alert_handlers),
            "recent_alerts": len(self._alert_history),
            "metrics_collected": len(self._metrics),
        }

    async def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        try:
            export_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "health_summary": self.get_health_summary(),
                "metrics_summary": self.get_metrics_summary(),
                "monitoring_status": self.get_monitoring_status(),
                "alert_history": self._alert_history[-50:],  # Last 50 alerts
                "raw_metrics": {
                    name: [
                        {
                            "timestamp": point.timestamp.isoformat(),
                            "value": point.value,
                            "metadata": point.metadata,
                        }
                        for point in points[-100:]  # Last 100 points per metric
                    ]
                    for name, points in self._metrics.items()
                },
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Metrics exported to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


# Global monitor instance
_telegram_monitor = None


def get_telegram_monitor() -> TelegramSystemMonitor:
    """Get singleton Telegram monitor instance."""
    global _telegram_monitor
    if _telegram_monitor is None:
        _telegram_monitor = TelegramSystemMonitor()
    return _telegram_monitor
