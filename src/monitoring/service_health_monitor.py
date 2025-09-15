"""
Service Health Monitoring System
Monitors the health and status of all trading bot services.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from src.core.lock_manager import ServiceLockManager, cleanup_all_stale_locks
from src.core.logging_manager import get_system_logger


@dataclass
class ServiceStatus:
    """Service status information."""

    name: str
    running: bool
    pid: Optional[int]
    cpu_percent: float
    memory_mb: float
    uptime_seconds: float
    last_check: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    lock_status: Optional[Dict[str, Any]] = None


class ServiceHealthMonitor:
    """Health monitor for trading bot services."""

    def __init__(self, check_interval: int = 30):
        """
        Initialize service health monitor.

        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self.logger = get_system_logger(__name__)

        # Service definitions
        self.services = {
            "telegram": {
                "process_name": "telegram_bot",
                "script_path": "bin/telegram_bot",
                "required": True,
                "restart_on_failure": True,
            },
            "trader": {
                "process_name": "trader",
                "script_path": "bin/trader",
                "required": True,
                "restart_on_failure": True,
            },
            "system_manager": {
                "process_name": "system_manager",
                "script_path": "bin/system_manager",
                "required": False,
                "restart_on_failure": False,
            },
        }

        # Status tracking
        self.service_statuses: Dict[str, ServiceStatus] = {}
        self.monitoring_enabled = True
        self.start_time = time.time()

        # Health check history
        self.health_history_file = Path("logs/health_history.json")
        self.health_history_file.parent.mkdir(parents=True, exist_ok=True)

    async def start_monitoring(self):
        """Start the health monitoring loop."""
        self.logger.info("Starting service health monitoring...")

        # Initial cleanup
        cleanup_all_stale_locks()

        while self.monitoring_enabled:
            try:
                await self._check_all_services()
                await self._save_health_snapshot()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_all_services(self):
        """Check health of all services."""
        for service_name, config in self.services.items():
            try:
                status = await self._check_service(service_name, config)
                self.service_statuses[service_name] = status

                if config.get("required") and not status.running:
                    self.logger.warning(f"Required service {service_name} is not running")

                    if config.get("restart_on_failure"):
                        await self._attempt_service_restart(service_name, config)

            except Exception as e:
                self.logger.error(f"Error checking service {service_name}: {e}")

    async def _check_service(self, service_name: str, config: Dict) -> ServiceStatus:
        """Check health of a specific service."""
        process_name = config["process_name"]
        current_time = datetime.now(timezone.utc)

        # Find running processes
        running_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if any(process_name in " ".join(proc.info["cmdline"] or [])):
                    running_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not running_processes:
            # Service not running
            lock_manager = ServiceLockManager(service_name)
            lock_info = lock_manager.manager.get_lock_info()

            return ServiceStatus(
                name=service_name,
                running=False,
                pid=None,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=0.0,
                last_check=current_time,
                lock_status=lock_info,
            )

        # Service is running - get metrics from primary process
        main_process = running_processes[0]  # Assume first is main process

        try:
            cpu_percent = main_process.cpu_percent()
            memory_info = main_process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            create_time = main_process.create_time()
            uptime_seconds = time.time() - create_time

            # Check lock status
            lock_manager = ServiceLockManager(service_name)
            lock_info = lock_manager.manager.get_lock_info()

            return ServiceStatus(
                name=service_name,
                running=True,
                pid=main_process.pid,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                uptime_seconds=uptime_seconds,
                last_check=current_time,
                lock_status=lock_info,
            )

        except Exception as e:
            self.logger.error(f"Error getting metrics for {service_name}: {e}")
            return ServiceStatus(
                name=service_name,
                running=True,
                pid=main_process.pid,
                cpu_percent=0.0,
                memory_mb=0.0,
                uptime_seconds=0.0,
                last_check=current_time,
                error_count=1,
                last_error=str(e),
            )

    async def _attempt_service_restart(self, service_name: str, config: Dict):
        """Attempt to restart a failed service."""
        script_path = config["script_path"]
        self.logger.info(f"Attempting to restart service: {service_name}")

        try:
            # Clean up stale locks first
            lock_manager = ServiceLockManager(service_name)
            lock_manager.cleanup_stale_locks()

            # Start the service (this is a simplified restart - in production you might want more sophisticated logic)
            import subprocess

            process = subprocess.Popen(
                [script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            self.logger.info(f"Started {service_name} restart with PID: {process.pid}")

            # Give it a moment to start
            await asyncio.sleep(5)

            # Check if it started successfully
            status = await self._check_service(service_name, config)
            if status.running:
                self.logger.info(f"Successfully restarted {service_name}")
            else:
                self.logger.error(f"Failed to restart {service_name}")

        except Exception as e:
            self.logger.error(f"Error restarting {service_name}: {e}")

    async def _save_health_snapshot(self):
        """Save current health status to file."""
        try:
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "monitor_uptime": time.time() - self.start_time,
                "services": {
                    name: asdict(status) for name, status in self.service_statuses.items()
                },
            }

            # Load existing history
            history = []
            if self.health_history_file.exists():
                with open(self.health_history_file, "r") as f:
                    history = json.load(f)

            # Add new snapshot
            history.append(snapshot)

            # Keep only last 100 snapshots
            history = history[-100:]

            # Save back to file
            with open(self.health_history_file, "w") as f:
                json.dump(history, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving health snapshot: {e}")

    def get_service_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current status of services."""
        if service_name:
            if service_name in self.service_statuses:
                return asdict(self.service_statuses[service_name])
            else:
                return {"error": f"Service {service_name} not found"}
        else:
            return {name: asdict(status) for name, status in self.service_statuses.items()}

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        total_services = len(self.services)
        running_services = sum(1 for status in self.service_statuses.values() if status.running)
        required_services = [
            name for name, config in self.services.items() if config.get("required")
        ]
        required_running = sum(
            1 for name in required_services if self.service_statuses.get(name, {}).running
        )

        overall_health = "healthy" if required_running == len(required_services) else "degraded"
        if running_services == 0:
            overall_health = "critical"

        return {
            "overall_health": overall_health,
            "total_services": total_services,
            "running_services": running_services,
            "required_services": len(required_services),
            "required_running": required_running,
            "monitor_uptime": time.time() - self.start_time,
            "last_check": datetime.now(timezone.utc).isoformat(),
            "services": self.get_service_status(),
        }

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_enabled = False
        self.logger.info("Health monitoring stopped")


# Global health monitor instance
_health_monitor: Optional[ServiceHealthMonitor] = None


def get_health_monitor() -> ServiceHealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ServiceHealthMonitor()
    return _health_monitor


async def start_health_monitoring():
    """Start health monitoring in background."""
    monitor = get_health_monitor()
    await monitor.start_monitoring()


def get_quick_health_status() -> Dict[str, Any]:
    """Get quick health status without starting monitoring."""
    monitor = ServiceHealthMonitor(check_interval=1)

    # Run one check cycle
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(monitor._check_all_services())
        return monitor.get_health_summary()
    finally:
        loop.close()
