"""Graceful shutdown and cleanup system for the trading bot."""

import asyncio
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .error_handler import ErrorHandler
from .interfaces import ILogger


class ShutdownReason(Enum):
    """Reasons for system shutdown."""

    USER_REQUEST = "user_request"
    SIGNAL = "signal"
    ERROR = "error"
    HEALTH_CHECK = "health_check"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ShutdownPhase(Enum):
    """Shutdown phases."""

    INITIATED = "initiated"
    STOPPING_SERVICES = "stopping_services"
    CLEANING_RESOURCES = "cleaning_resources"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ShutdownTask:
    """Task to execute during shutdown."""

    name: str
    function: Callable
    priority: int = 0  # Higher priority runs first
    timeout: float = 30.0
    critical: bool = False  # If True, failure stops shutdown process
    phase: ShutdownPhase = ShutdownPhase.STOPPING_SERVICES


@dataclass
class ShutdownStatus:
    """Current shutdown status."""

    is_shutting_down: bool = False
    reason: Optional[ShutdownReason] = None
    phase: ShutdownPhase = ShutdownPhase.INITIATED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tasks: int = 0
    current_task: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ShutdownHandler:
    """Handles graceful shutdown and cleanup procedures."""

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        error_handler: Optional[ErrorHandler] = None,
        shutdown_timeout: float = 120.0,
    ):
        self.logger = logger
        self.error_handler = error_handler
        self.shutdown_timeout = shutdown_timeout

        self.shutdown_tasks: Dict[str, ShutdownTask] = {}
        self.status = ShutdownStatus()
        self.shutdown_event = asyncio.Event()
        self.cleanup_callbacks: List[Callable] = []

        # Signal handlers
        self._setup_signal_handlers()

        # Resource tracking
        self.managed_resources: Set[Any] = set()
        self.active_connections: Set[Any] = set()
        self.background_tasks: Set[asyncio.Task] = set()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        if sys.platform != "win32":
            # Unix signals
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        else:
            # Windows signals
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGBREAK, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        self._log_info(f"Received signal {signal_name}, initiating graceful shutdown")

        # Create shutdown task if not already shutting down
        if not self.status.is_shutting_down:
            asyncio.create_task(self.shutdown(ShutdownReason.SIGNAL))

    def register_shutdown_task(self, task: ShutdownTask) -> None:
        """Register a shutdown task.

        Args:
            task: Shutdown task to register
        """
        self.shutdown_tasks[task.name] = task
        self._log_info(f"Registered shutdown task: {task.name} (priority: {task.priority})")

    def unregister_shutdown_task(self, name: str) -> None:
        """Unregister a shutdown task.

        Args:
            name: Task name to unregister
        """
        if name in self.shutdown_tasks:
            del self.shutdown_tasks[name]
            self._log_info(f"Unregistered shutdown task: {name}")

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add cleanup callback function.

        Args:
            callback: Function to call during cleanup
        """
        self.cleanup_callbacks.append(callback)

    def track_resource(self, resource: Any) -> None:
        """Track a resource for cleanup.

        Args:
            resource: Resource to track
        """
        self.managed_resources.add(resource)

    def untrack_resource(self, resource: Any) -> None:
        """Stop tracking a resource.

        Args:
            resource: Resource to stop tracking
        """
        self.managed_resources.discard(resource)

    def track_connection(self, connection: Any) -> None:
        """Track an active connection.

        Args:
            connection: Connection to track
        """
        self.active_connections.add(connection)

    def untrack_connection(self, connection: Any) -> None:
        """Stop tracking a connection.

        Args:
            connection: Connection to stop tracking
        """
        self.active_connections.discard(connection)

    def track_background_task(self, task: asyncio.Task) -> None:
        """Track a background task.

        Args:
            task: Task to track
        """
        self.background_tasks.add(task)

        # Remove task when it completes
        def remove_task(t):
            self.background_tasks.discard(t)

        task.add_done_callback(remove_task)

    async def shutdown(
        self,
        reason: ShutdownReason = ShutdownReason.USER_REQUEST,
        timeout: Optional[float] = None,
    ) -> bool:
        """Initiate graceful shutdown.

        Args:
            reason: Reason for shutdown
            timeout: Optional timeout override

        Returns:
            True if shutdown completed successfully, False otherwise
        """
        if self.status.is_shutting_down:
            self._log_warning("Shutdown already in progress")
            return False

        self.status.is_shutting_down = True
        self.status.reason = reason
        self.status.started_at = datetime.now()
        self.status.phase = ShutdownPhase.INITIATED

        shutdown_timeout = timeout or self.shutdown_timeout

        self._log_info(
            f"Initiating graceful shutdown (reason: {reason.value}, timeout: {shutdown_timeout}s)"
        )

        try:
            # Execute shutdown with timeout
            await asyncio.wait_for(self._execute_shutdown(), timeout=shutdown_timeout)

            self.status.phase = ShutdownPhase.COMPLETED
            self.status.completed_at = datetime.now()

            duration = (self.status.completed_at - self.status.started_at).total_seconds()
            self._log_info(f"Graceful shutdown completed in {duration:.2f}s")

            # Set shutdown event
            self.shutdown_event.set()

            return True

        except asyncio.TimeoutError:
            self.status.phase = ShutdownPhase.FAILED
            self.status.errors.append(f"Shutdown timed out after {shutdown_timeout}s")
            self._log_error(f"Shutdown timed out after {shutdown_timeout}s")
            return False

        except Exception as e:
            self.status.phase = ShutdownPhase.FAILED
            self.status.errors.append(f"Shutdown failed: {str(e)}")
            self._log_error(f"Shutdown failed: {e}")
            return False

    async def _execute_shutdown(self) -> None:
        """Execute the shutdown process."""
        # Phase 1: Stop services
        self.status.phase = ShutdownPhase.STOPPING_SERVICES
        await self._execute_shutdown_phase(ShutdownPhase.STOPPING_SERVICES)

        # Phase 2: Clean resources
        self.status.phase = ShutdownPhase.CLEANING_RESOURCES
        await self._execute_shutdown_phase(ShutdownPhase.CLEANING_RESOURCES)

        # Phase 3: Finalize
        self.status.phase = ShutdownPhase.FINALIZING
        await self._execute_shutdown_phase(ShutdownPhase.FINALIZING)

        # Additional cleanup
        await self._cleanup_background_tasks()
        await self._cleanup_connections()
        await self._cleanup_resources()
        await self._execute_cleanup_callbacks()

    async def _execute_shutdown_phase(self, phase: ShutdownPhase) -> None:
        """Execute shutdown tasks for a specific phase.

        Args:
            phase: Shutdown phase to execute
        """
        # Get tasks for this phase, sorted by priority (highest first)
        phase_tasks = [task for task in self.shutdown_tasks.values() if task.phase == phase]
        phase_tasks.sort(key=lambda x: x.priority, reverse=True)

        self.status.total_tasks += len(phase_tasks)

        self._log_info(f"Executing {len(phase_tasks)} tasks for phase {phase.value}")

        for task in phase_tasks:
            self.status.current_task = task.name

            try:
                self._log_info(f"Executing shutdown task: {task.name}")

                if asyncio.iscoroutinefunction(task.function):
                    await asyncio.wait_for(task.function(), timeout=task.timeout)
                else:
                    task.function()

                self.status.tasks_completed += 1
                self._log_info(f"Completed shutdown task: {task.name}")

            except asyncio.TimeoutError:
                error_msg = f"Shutdown task {task.name} timed out after {task.timeout}s"
                self.status.errors.append(error_msg)
                self.status.tasks_failed += 1

                if task.critical:
                    self._log_error(f"Critical task failed: {error_msg}")
                    raise
                else:
                    self._log_warning(f"Non-critical task failed: {error_msg}")

            except Exception as e:
                error_msg = f"Shutdown task {task.name} failed: {str(e)}"
                self.status.errors.append(error_msg)
                self.status.tasks_failed += 1

                if task.critical:
                    self._log_error(f"Critical task failed: {error_msg}")
                    raise
                else:
                    self._log_warning(f"Non-critical task failed: {error_msg}")

        self.status.current_task = None

    async def _cleanup_background_tasks(self) -> None:
        """Cancel and cleanup background tasks."""
        if not self.background_tasks:
            return

        self._log_info(f"Cancelling {len(self.background_tasks)} background tasks")

        # Cancel all tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete or timeout
        if self.background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.background_tasks, return_exceptions=True),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                self._log_warning("Some background tasks did not complete within timeout")

        self.background_tasks.clear()

    async def _cleanup_connections(self) -> None:
        """Close active connections."""
        if not self.active_connections:
            return

        self._log_info(f"Closing {len(self.active_connections)} active connections")

        for connection in list(self.active_connections):
            try:
                if hasattr(connection, "close"):
                    if asyncio.iscoroutinefunction(connection.close):
                        await connection.close()
                    else:
                        connection.close()
                elif hasattr(connection, "disconnect"):
                    if asyncio.iscoroutinefunction(connection.disconnect):
                        await connection.disconnect()
                    else:
                        connection.disconnect()
            except Exception as e:
                self._log_warning(f"Error closing connection: {e}")

        self.active_connections.clear()

    async def _cleanup_resources(self) -> None:
        """Cleanup managed resources."""
        if not self.managed_resources:
            return

        self._log_info(f"Cleaning up {len(self.managed_resources)} managed resources")

        for resource in list(self.managed_resources):
            try:
                if hasattr(resource, "cleanup"):
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        await resource.cleanup()
                    else:
                        resource.cleanup()
                elif hasattr(resource, "close"):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
            except Exception as e:
                self._log_warning(f"Error cleaning up resource: {e}")

        self.managed_resources.clear()

    async def _execute_cleanup_callbacks(self) -> None:
        """Execute cleanup callbacks."""
        if not self.cleanup_callbacks:
            return

        self._log_info(f"Executing {len(self.cleanup_callbacks)} cleanup callbacks")

        for callback in self.cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self._log_warning(f"Error in cleanup callback: {e}")

    def is_shutting_down(self) -> bool:
        """Check if system is shutting down.

        Returns:
            True if shutting down, False otherwise
        """
        return self.status.is_shutting_down

    def get_status(self) -> ShutdownStatus:
        """Get current shutdown status.

        Returns:
            Current shutdown status
        """
        return self.status

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self.shutdown_event.wait()

    @asynccontextmanager
    async def managed_resource(self, resource: Any):
        """Context manager for automatic resource tracking.

        Args:
            resource: Resource to manage
        """
        self.track_resource(resource)
        try:
            yield resource
        finally:
            self.untrack_resource(resource)

    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(message)
            else:
                self.logger.info(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.warning(message)
            else:
                self.logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log error message."""
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.error(message)
            else:
                self.logger.error(message)


# Global shutdown handler instance
_shutdown_handler: Optional[ShutdownHandler] = None


def get_shutdown_handler() -> Optional[ShutdownHandler]:
    """Get the global shutdown handler instance.

    Returns:
        Global shutdown handler or None if not initialized
    """
    return _shutdown_handler


def initialize_shutdown_handler(
    logger: Optional[ILogger] = None,
    error_handler: Optional[ErrorHandler] = None,
    shutdown_timeout: float = 120.0,
) -> ShutdownHandler:
    """Initialize the global shutdown handler.

    Args:
        logger: Logger instance
        error_handler: Error handler instance
        shutdown_timeout: Shutdown timeout in seconds

    Returns:
        Initialized shutdown handler
    """
    global _shutdown_handler
    _shutdown_handler = ShutdownHandler(logger, error_handler, shutdown_timeout)
    return _shutdown_handler
