"""Health monitoring and alerting system for the trading bot."""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .interfaces import ILogger
from .error_handler import ErrorHandler, ErrorSeverity


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable
    interval: float = 60.0  # seconds
    timeout: float = 30.0   # seconds
    enabled: bool = True
    critical: bool = False  # If True, failure marks system as critical


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    alerts: List[Alert]
    last_updated: datetime = field(default_factory=datetime.now)
    uptime: float = 0.0


class HealthMonitor:
    """Health monitoring system."""
    
    def __init__(
        self,
        logger: Optional[ILogger] = None,
        error_handler: Optional[ErrorHandler] = None,
        alert_retention_hours: int = 24
    ):
        self.logger = logger
        self.error_handler = error_handler
        self.alert_retention_hours = alert_retention_hours
        
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics: Dict[str, HealthMetric] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        self.start_time = time.time()
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Metric history for trending
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        self.register_health_check(HealthCheck(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            interval=30.0
        ))
        
        self.register_health_check(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            interval=30.0
        ))
        
        self.register_health_check(HealthCheck(
            name="disk_usage",
            check_function=self._check_disk_usage,
            interval=60.0
        ))
        
        self.register_health_check(HealthCheck(
            name="system_load",
            check_function=self._check_system_load,
            interval=30.0
        ))
    
    async def _check_cpu_usage(self) -> HealthMetric:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
        elif cpu_percent > 70:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            status=status,
            threshold_warning=70.0,
            threshold_critical=90.0,
            unit="%",
            description="CPU usage percentage"
        )
    
    async def _check_memory_usage(self) -> HealthMetric:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > 90:
            status = HealthStatus.CRITICAL
        elif memory_percent > 80:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return HealthMetric(
            name="memory_usage",
            value=memory_percent,
            status=status,
            threshold_warning=80.0,
            threshold_critical=90.0,
            unit="%",
            description="Memory usage percentage"
        )
    
    async def _check_disk_usage(self) -> HealthMetric:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > 95:
            status = HealthStatus.CRITICAL
        elif disk_percent > 85:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return HealthMetric(
            name="disk_usage",
            value=disk_percent,
            status=status,
            threshold_warning=85.0,
            threshold_critical=95.0,
            unit="%",
            description="Disk usage percentage"
        )
    
    async def _check_system_load(self) -> HealthMetric:
        """Check system load average."""
        try:
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100
            
            if load_percent > 90:
                status = HealthStatus.CRITICAL
            elif load_percent > 70:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthMetric(
                name="system_load",
                value=load_avg,
                status=status,
                threshold_warning=cpu_count * 0.7,
                threshold_critical=cpu_count * 0.9,
                unit="",
                description="System load average (1 minute)"
            )
        except AttributeError:
            # getloadavg not available on Windows
            return HealthMetric(
                name="system_load",
                value=0.0,
                status=HealthStatus.UNKNOWN,
                unit="",
                description="System load (not available on Windows)"
            )
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check.
        
        Args:
            health_check: Health check configuration
        """
        self.health_checks[health_check.name] = health_check
        self._log_info(f"Registered health check: {health_check.name}")
    
    def unregister_health_check(self, name: str) -> None:
        """Unregister a health check.
        
        Args:
            name: Health check name
        """
        if name in self.health_checks:
            del self.health_checks[name]
            self._log_info(f"Unregistered health check: {name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert callback function.
        
        Args:
            callback: Function to call when alert is raised
        """
        self.alert_callbacks.append(callback)
    
    async def start(self) -> None:
        """Start health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self._log_info("Health monitor started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self._log_info("Health monitor stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        check_schedules = {name: 0.0 for name in self.health_checks.keys()}
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Run scheduled health checks
                for name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    if current_time >= check_schedules[name]:
                        try:
                            metric = await asyncio.wait_for(
                                health_check.check_function(),
                                timeout=health_check.timeout
                            )
                            
                            self._update_metric(metric)
                            check_schedules[name] = current_time + health_check.interval
                            
                        except asyncio.TimeoutError:
                            self._create_alert(
                                AlertSeverity.WARNING,
                                "health_monitor",
                                f"Health check {name} timed out",
                                {"check_name": name, "timeout": health_check.timeout}
                            )
                        except Exception as e:
                            self._create_alert(
                                AlertSeverity.WARNING,
                                "health_monitor",
                                f"Health check {name} failed: {str(e)}",
                                {"check_name": name, "error": str(e)}
                            )
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Sleep for a short interval
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self._log_error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5.0)
    
    def _update_metric(self, metric: HealthMetric) -> None:
        """Update a health metric.
        
        Args:
            metric: Health metric to update
        """
        self.metrics[metric.name] = metric
        self.metric_history[metric.name].append({
            'timestamp': metric.timestamp,
            'value': metric.value,
            'status': metric.status
        })
        
        # Check if metric status requires an alert
        if metric.status == HealthStatus.CRITICAL:
            self._create_alert(
                AlertSeverity.CRITICAL,
                "health_monitor",
                f"Critical health metric: {metric.name} = {metric.value}{metric.unit}",
                {
                    'metric_name': metric.name,
                    'value': metric.value,
                    'threshold': metric.threshold_critical,
                    'unit': metric.unit
                }
            )
        elif metric.status == HealthStatus.WARNING:
            self._create_alert(
                AlertSeverity.WARNING,
                "health_monitor",
                f"Warning health metric: {metric.name} = {metric.value}{metric.unit}",
                {
                    'metric_name': metric.name,
                    'value': metric.value,
                    'threshold': metric.threshold_warning,
                    'unit': metric.unit
                }
            )
    
    def _create_alert(
        self,
        severity: AlertSeverity,
        component: str,
        message: str,
        details: Dict[str, Any]
    ) -> Alert:
        """Create and process an alert.
        
        Args:
            severity: Alert severity
            component: Component that generated the alert
            message: Alert message
            details: Additional alert details
            
        Returns:
            Created alert
        """
        alert_id = f"{component}_{int(time.time())}_{len(self.alerts)}"
        alert = Alert(
            id=alert_id,
            severity=severity,
            component=component,
            message=message,
            details=details
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        log_message = f"ALERT [{severity.value.upper()}] {component}: {message}"
        if severity == AlertSeverity.CRITICAL:
            self._log_error(log_message)
        elif severity == AlertSeverity.WARNING:
            self._log_warning(log_message)
        else:
            self._log_info(log_message)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self._log_error(f"Error in alert callback: {e}")
        
        # Report to error handler if available
        if self.error_handler and severity == AlertSeverity.CRITICAL:
            from .error_handler import TradingBotException
            error = TradingBotException(
                message,
                component=component,
                operation="health_monitoring",
                details=details
            )
            self.error_handler.handle_error(error, attempt_recovery=False)
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            
        Returns:
            True if alert was resolved, False if not found
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            self._log_info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
        
        alerts_to_remove = []
        for alert_id, alert in self.alerts.items():
            if (alert.resolved and alert.resolved_at and 
                alert.resolved_at < cutoff_time):
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health status.
        
        Returns:
            System health status
        """
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        
        # Check for critical metrics
        for metric in self.metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif metric.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        # Check for unresolved critical alerts
        unresolved_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        for alert in unresolved_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
        
        uptime = time.time() - self.start_time
        
        return SystemHealth(
            status=overall_status,
            metrics=self.metrics.copy(),
            alerts=list(self.alerts.values()),
            uptime=uptime
        )
    
    def get_metric_history(self, metric_name: str, limit: int = 50) -> List[Dict]:
        """Get metric history.
        
        Args:
            metric_name: Name of the metric
            limit: Maximum number of history entries
            
        Returns:
            List of metric history entries
        """
        if metric_name in self.metric_history:
            history = list(self.metric_history[metric_name])
            return history[-limit:]
        return []
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active (unresolved) alerts.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of active alerts
        """
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        if severity:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            if hasattr(self.logger, 'logger'):
                self.logger.logger.info(message)
            else:
                self.logger.info(message)
    
    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            if hasattr(self.logger, 'logger'):
                self.logger.logger.warning(message)
            else:
                self.logger.warning(message)
    
    def _log_error(self, message: str) -> None:
        """Log error message."""
        if self.logger:
            if hasattr(self.logger, 'logger'):
                self.logger.logger.error(message)
            else:
                self.logger.error(message)