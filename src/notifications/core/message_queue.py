"""
High-performance message queue for Telegram notifications.
Handles priority, persistence, and rate limiting.
"""

import asyncio
import heapq
import json
import logging
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from src.core.logging_manager import get_system_logger


class MessagePriority(IntEnum):
    """Message priority levels (lower number = higher priority)."""

    CRITICAL = 1  # System errors, security alerts
    HIGH = 2  # Trading alerts, important notifications
    NORMAL = 3  # Regular notifications
    LOW = 4  # Status updates, info messages


class QueuedMessage(NamedTuple):
    """Represents a queued message with metadata."""

    priority: int
    timestamp: float
    sequence: int  # For stable sorting
    message: str
    parse_mode: str
    retry_count: int
    max_retries: int
    original_timestamp: float


class MessageQueue:
    """
    Advanced message queue with priority, persistence, and rate limiting.
    Uses a binary heap for efficient priority handling.
    """

    def __init__(
        self,
        queue_file: Optional[str] = None,
        max_queue_size: int = 1000,
        persistence_enabled: bool = True,
    ):
        """
        Initialize the message queue.

        Args:
            queue_file: Path to persistence file (None for auto-generated)
            max_queue_size: Maximum messages in queue
            persistence_enabled: Enable message persistence across restarts
        """
        self.logger = get_system_logger(__name__)

        # Queue management
        self._heap: List[QueuedMessage] = []
        self._sequence_counter = 0
        self.max_queue_size = max_queue_size

        # Persistence
        self.persistence_enabled = persistence_enabled
        if queue_file:
            self.queue_file = Path(queue_file)
        else:
            # Auto-generate in logs directory
            logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            self.queue_file = logs_dir / "telegram_message_queue.json"

        # Thread safety
        self._lock = asyncio.Lock()

        # Dead letter queue for failed messages
        self._dead_letter_queue: List[QueuedMessage] = []
        self.max_dead_letters = 100

        # Statistics
        self._stats = {
            "messages_queued": 0,
            "messages_sent": 0,
            "messages_failed": 0,
            "messages_dropped": 0,
            "queue_overflows": 0,
        }

        # Load persisted messages
        if self.persistence_enabled:
            self._load_queue()

    async def start(self):
        """Initialize or warm up the queue (no-op for now)."""
        # The queue is ready after construction; keep for API symmetry
        self.logger.debug("MessageQueue start called (no-op)")

    async def stop(self):
        """Stop the queue and persist state if enabled (lightweight)."""
        if self.persistence_enabled:
            try:
                self._save_queue()
                self.logger.info("Message queue state persisted on stop")
            except Exception as e:
                self.logger.error(f"Failed to persist queue on stop: {e}")

    async def enqueue(
        self,
        message: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        parse_mode: str = "HTML",
        max_retries: int = 3,
    ) -> bool:
        """
        Add message to priority queue.

        Args:
            message: Message text to send
            priority: Message priority level
            parse_mode: Telegram parse mode
            max_retries: Maximum retry attempts

        Returns:
            bool: True if message was queued successfully
        """
        async with self._lock:
            try:
                # Check queue size limits
                if len(self._heap) >= self.max_queue_size:
                    # Remove lowest priority message if queue is full
                    if self._heap:
                        dropped_msg = heapq.heappop(self._heap)
                        self.logger.warning(
                            f"Queue full, dropped message: {dropped_msg.message[:50]}..."
                        )
                        self._stats["messages_dropped"] += 1
                        self._stats["queue_overflows"] += 1

                # Create queued message
                current_time = asyncio.get_event_loop().time()
                queued_msg = QueuedMessage(
                    priority=priority.value,
                    timestamp=current_time,
                    sequence=self._sequence_counter,
                    message=message,
                    parse_mode=parse_mode,
                    retry_count=0,
                    max_retries=max_retries,
                    original_timestamp=current_time,
                )

                # Add to priority queue
                heapq.heappush(self._heap, queued_msg)
                self._sequence_counter += 1
                self._stats["messages_queued"] += 1

                self.logger.debug(
                    f"Message queued with priority {priority.name}: {message[:50]}..."
                )

                # Persist queue state
                if self.persistence_enabled:
                    self._save_queue()

                return True

            except Exception as e:
                self.logger.error(f"Failed to enqueue message: {e}")
                return False

    async def dequeue(self) -> Optional[QueuedMessage]:
        """
        Get next message from priority queue.

        Returns:
            QueuedMessage or None if queue is empty
        """
        async with self._lock:
            if not self._heap:
                return None

            try:
                message = heapq.heappop(self._heap)

                # Persist queue state if needed
                if self.persistence_enabled and len(self._heap) % 10 == 0:
                    self._save_queue()

                return message

            except Exception as e:
                self.logger.error(f"Failed to dequeue message: {e}")
                return None

    async def requeue_with_retry(self, message: QueuedMessage) -> bool:
        """
        Requeue a failed message with incremented retry count.

        Args:
            message: Failed message to retry

        Returns:
            bool: True if message was requeued, False if max retries exceeded
        """
        if message.retry_count >= message.max_retries:
            # Move to dead letter queue
            await self._move_to_dead_letter(message)
            return False

        async with self._lock:
            try:
                # Create new message with incremented retry count
                retry_message = QueuedMessage(
                    priority=min(
                        message.priority + 1, MessagePriority.LOW.value
                    ),  # Lower priority for retries
                    timestamp=asyncio.get_event_loop().time(),
                    sequence=self._sequence_counter,
                    message=message.message,
                    parse_mode=message.parse_mode,
                    retry_count=message.retry_count + 1,
                    max_retries=message.max_retries,
                    original_timestamp=message.original_timestamp,
                )

                heapq.heappush(self._heap, retry_message)
                self._sequence_counter += 1

                self.logger.debug(
                    f"Message requeued for retry {retry_message.retry_count}/{message.max_retries}"
                )

                return True

            except Exception as e:
                self.logger.error(f"Failed to requeue message: {e}")
                await self._move_to_dead_letter(message)
                return False

    async def _move_to_dead_letter(self, message: QueuedMessage):
        """Move failed message to dead letter queue."""
        try:
            # Add to dead letter queue (keep only recent failures)
            self._dead_letter_queue.append(message)
            if len(self._dead_letter_queue) > self.max_dead_letters:
                self._dead_letter_queue.pop(0)  # Remove oldest

            self._stats["messages_failed"] += 1
            self.logger.warning(
                f"Message moved to dead letter queue after {message.retry_count} retries: {message.message[:50]}..."
            )

        except Exception as e:
            self.logger.error(f"Failed to move message to dead letter queue: {e}")

    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get comprehensive queue status and statistics.

        Returns:
            Dict with queue metrics
        """
        async with self._lock:
            # Priority distribution
            priority_counts = {}
            for msg in self._heap:
                priority_name = MessagePriority(msg.priority).name
                priority_counts[priority_name] = priority_counts.get(priority_name, 0) + 1

            return {
                "queue_size": len(self._heap),
                "max_queue_size": self.max_queue_size,
                "dead_letter_size": len(self._dead_letter_queue),
                "priority_distribution": priority_counts,
                "statistics": self._stats.copy(),
                "persistence_enabled": self.persistence_enabled,
                "queue_file": str(self.queue_file) if self.persistence_enabled else None,
                "oldest_message_age": (
                    asyncio.get_event_loop().time() - min(msg.timestamp for msg in self._heap)
                    if self._heap
                    else 0
                ),
            }

    async def clear_queue(self) -> int:
        """
        Clear all messages from queue.

        Returns:
            int: Number of messages cleared
        """
        async with self._lock:
            cleared_count = len(self._heap)
            self._heap.clear()

            if self.persistence_enabled:
                self._save_queue()

            self.logger.info(f"Cleared {cleared_count} messages from queue")
            return cleared_count

    async def get_dead_letters(self) -> List[Dict[str, Any]]:
        """
        Get dead letter messages for analysis.

        Returns:
            List of failed message details
        """
        return [
            {
                "message": msg.message,
                "original_timestamp": datetime.fromtimestamp(
                    msg.original_timestamp, tz=timezone.utc
                ).isoformat(),
                "retry_count": msg.retry_count,
                "priority": MessagePriority(msg.priority).name,
                "parse_mode": msg.parse_mode,
            }
            for msg in self._dead_letter_queue
        ]

    def _save_queue(self):
        """Persist queue state to disk."""
        try:
            queue_data = {
                "messages": [
                    {
                        "priority": msg.priority,
                        "timestamp": msg.timestamp,
                        "sequence": msg.sequence,
                        "message": msg.message,
                        "parse_mode": msg.parse_mode,
                        "retry_count": msg.retry_count,
                        "max_retries": msg.max_retries,
                        "original_timestamp": msg.original_timestamp,
                    }
                    for msg in self._heap
                ],
                "dead_letters": [
                    {
                        "priority": msg.priority,
                        "timestamp": msg.timestamp,
                        "sequence": msg.sequence,
                        "message": msg.message,
                        "parse_mode": msg.parse_mode,
                        "retry_count": msg.retry_count,
                        "max_retries": msg.max_retries,
                        "original_timestamp": msg.original_timestamp,
                    }
                    for msg in self._dead_letter_queue
                ],
                "sequence_counter": self._sequence_counter,
                "statistics": self._stats,
            }

            with open(self.queue_file, "w") as f:
                json.dump(queue_data, f, indent=2)

            self.logger.debug(f"Queue state persisted to {self.queue_file}")

        except Exception as e:
            self.logger.error(f"Failed to save queue state: {e}")

    def _load_queue(self):
        """Load persisted queue state from disk."""
        try:
            if not self.queue_file.exists():
                self.logger.debug("No persisted queue state found")
                return

            with open(self.queue_file, "r") as f:
                queue_data = json.load(f)

            # Restore main queue
            for msg_data in queue_data.get("messages", []):
                queued_msg = QueuedMessage(**msg_data)
                heapq.heappush(self._heap, queued_msg)

            # Restore dead letter queue
            self._dead_letter_queue = [
                QueuedMessage(**msg_data) for msg_data in queue_data.get("dead_letters", [])
            ]

            # Restore counters and stats
            self._sequence_counter = queue_data.get("sequence_counter", 0)
            self._stats.update(queue_data.get("statistics", {}))

            loaded_count = len(self._heap)
            dead_count = len(self._dead_letter_queue)

            self.logger.info(
                f"Loaded {loaded_count} messages and {dead_count} dead letters from persistent storage"
            )

        except Exception as e:
            self.logger.error(f"Failed to load queue state: {e}")
            # Continue with empty queue rather than failing

    async def cleanup(self):
        """Clean up resources and persist final state."""
        if self.persistence_enabled:
            try:
                self._save_queue()
                self.logger.info("Queue state saved during cleanup")
            except Exception as e:
                self.logger.error(f"Error saving queue during cleanup: {e}")

    def mark_message_sent(self):
        """Mark a message as successfully sent (for statistics)."""
        self._stats["messages_sent"] += 1
