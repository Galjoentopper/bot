"""
Enhanced Lock File Management System
Provides robust lock file handling with stale lock detection and cleanup.
"""

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import psutil

from src.core.logging_manager import get_system_logger


class LockManager:
    """Enhanced lock file manager with stale lock detection."""

    def __init__(self, lock_file: Union[str, Path], max_age_seconds: int = 3600):
        """
        Initialize lock manager.

        Args:
            lock_file: Path to lock file
            max_age_seconds: Maximum age of lock file before considering it stale
        """
        self.lock_file = Path(lock_file)
        self.max_age_seconds = max_age_seconds
        self.logger = get_system_logger(__name__)

        # Ensure lock directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def is_locked(self) -> bool:
        """Check if lock file exists and is valid."""
        if not self.lock_file.exists():
            return False

        # Check if lock is stale
        if self._is_stale_lock():
            self.logger.warning(f"Removing stale lock file: {self.lock_file}")
            self.release()
            return False

        # Check if process is still running
        pid = self._get_lock_pid()
        if pid and not self._is_process_running(pid):
            self.logger.warning(f"Lock file references dead process {pid}, removing")
            self.release()
            return False

        return True

    def acquire(self, timeout: int = 30) -> bool:
        """
        Acquire lock with timeout.

        Args:
            timeout: Maximum seconds to wait for lock

        Returns:
            True if lock acquired, False otherwise
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._try_acquire():
                return True
            time.sleep(0.1)

        return False

    def release(self) -> bool:
        """Release lock file."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
                self.logger.debug(f"Released lock: {self.lock_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to release lock {self.lock_file}: {e}")
            return False

    def _try_acquire(self) -> bool:
        """Try to acquire lock once."""
        try:
            if self.is_locked():
                return False

            # Create lock file with PID
            lock_data = {
                "pid": os.getpid(),
                "timestamp": time.time(),
                "hostname": os.uname().nodename,
            }

            # Use atomic write
            temp_file = self.lock_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                import json

                json.dump(lock_data, f)

            # Atomic rename
            temp_file.rename(self.lock_file)
            self.logger.debug(f"Acquired lock: {self.lock_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to acquire lock {self.lock_file}: {e}")
            return False

    def _is_stale_lock(self) -> bool:
        """Check if lock file is stale based on age."""
        try:
            lock_age = time.time() - self.lock_file.stat().st_mtime
            return lock_age > self.max_age_seconds
        except Exception:
            return True

    def _get_lock_pid(self) -> Optional[int]:
        """Get PID from lock file."""
        try:
            with open(self.lock_file, "r") as f:
                import json

                data = json.load(f)
                return data.get("pid")
        except Exception:
            return None

    def _is_process_running(self, pid: int) -> bool:
        """Check if process is still running."""
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False

    @contextmanager
    def lock(self, timeout: int = 30):
        """Context manager for lock acquisition."""
        if not self.acquire(timeout=timeout):
            raise RuntimeError(f"Failed to acquire lock {self.lock_file} within {timeout}s")

        try:
            yield
        finally:
            self.release()

    def force_release(self) -> bool:
        """Force release lock regardless of ownership."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
                self.logger.warning(f"Force released lock: {self.lock_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to force release lock {self.lock_file}: {e}")
            return False

    def get_lock_info(self) -> dict:
        """Get information about current lock."""
        if not self.lock_file.exists():
            return {"locked": False}

        try:
            with open(self.lock_file, "r") as f:
                import json

                data = json.load(f)

            # Add additional info
            data["locked"] = True
            data["age_seconds"] = time.time() - data.get("timestamp", 0)
            data["is_stale"] = self._is_stale_lock()

            pid = data.get("pid")
            if pid:
                data["process_running"] = self._is_process_running(pid)

            return data

        except Exception as e:
            self.logger.error(f"Failed to read lock info: {e}")
            return {"locked": True, "error": str(e)}


class ServiceLockManager:
    """Specialized lock manager for services."""

    def __init__(self, service_name: str, logs_dir: str = "logs"):
        """Initialize service lock manager."""
        self.service_name = service_name
        self.lock_file = Path(logs_dir) / f"{service_name}_service.lock"
        self.manager = LockManager(self.lock_file, max_age_seconds=1800)  # 30 min
        self.logger = get_system_logger(__name__)

    def acquire_service_lock(self, timeout: int = 10) -> bool:
        """Acquire service lock with service-specific logic."""
        self.logger.info(f"Acquiring lock for {self.service_name} service...")

        if self.manager.acquire(timeout=timeout):
            self.logger.info(f"{self.service_name} service lock acquired")
            return True
        else:
            lock_info = self.manager.get_lock_info()
            self.logger.error(
                f"Failed to acquire {self.service_name} service lock. " f"Lock info: {lock_info}"
            )
            return False

    def release_service_lock(self) -> bool:
        """Release service lock."""
        if self.manager.release():
            self.logger.info(f"{self.service_name} service lock released")
            return True
        return False

    @contextmanager
    def service_lock(self, timeout: int = 10):
        """Context manager for service lock."""
        if not self.acquire_service_lock(timeout=timeout):
            raise RuntimeError(f"Could not acquire {self.service_name} service lock")

        try:
            yield
        finally:
            self.release_service_lock()

    def cleanup_stale_locks(self):
        """Clean up stale locks for this service."""
        self.logger.info(f"Cleaning up stale locks for {self.service_name}")
        if self.manager.is_locked():
            lock_info = self.manager.get_lock_info()
            if lock_info.get("is_stale") or not lock_info.get("process_running", True):
                self.manager.force_release()
                self.logger.info(f"Cleaned up stale lock for {self.service_name}")


# Utility functions for common service locks
def get_telegram_lock_manager() -> ServiceLockManager:
    """Get lock manager for Telegram service."""
    return ServiceLockManager("telegram")


def get_trader_lock_manager() -> ServiceLockManager:
    """Get lock manager for trader service."""
    return ServiceLockManager("trader")


def cleanup_all_stale_locks():
    """Clean up stale locks for all services."""
    logger = get_system_logger(__name__)
    logger.info("Cleaning up stale locks for all services...")

    services = ["telegram", "trader", "monitor"]
    for service in services:
        try:
            manager = ServiceLockManager(service)
            manager.cleanup_stale_locks()
        except Exception as e:
            logger.error(f"Failed to cleanup locks for {service}: {e}")
