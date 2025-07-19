"""Enhanced notification system with intelligent rate limiting and priority queuing."""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class NotificationPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class NotificationType(Enum):
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    ERROR = "error"
    SYSTEM_STATUS = "system_status"
    PORTFOLIO_UPDATE = "portfolio_update"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_ALERT = "performance_alert"


@dataclass
class NotificationMessage:
    """Represents a notification message with metadata."""
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_per_minute: int = 10
    max_per_hour: int = 60
    cooldown_seconds: int = 60
    burst_allowance: int = 3


class SmartNotificationManager:
    """Intelligent notification manager with rate limiting and priority queuing."""
    
    def __init__(self, telegram_notifier=None):
        self.telegram_notifier = telegram_notifier
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting configuration
        self.rate_limits = {
            NotificationType.TRADE_OPENED: RateLimitConfig(max_per_minute=20, max_per_hour=100),
            NotificationType.TRADE_CLOSED: RateLimitConfig(max_per_minute=20, max_per_hour=100),
            NotificationType.ERROR: RateLimitConfig(max_per_minute=5, max_per_hour=30, cooldown_seconds=300),
            NotificationType.SYSTEM_STATUS: RateLimitConfig(max_per_minute=2, max_per_hour=10),
            NotificationType.PORTFOLIO_UPDATE: RateLimitConfig(max_per_minute=1, max_per_hour=24),
            NotificationType.HEALTH_CHECK: RateLimitConfig(max_per_minute=1, max_per_hour=12),
            NotificationType.PERFORMANCE_ALERT: RateLimitConfig(max_per_minute=2, max_per_hour=20)
        }
        
        # Priority queues
        self.queues = {
            NotificationPriority.CRITICAL: deque(),
            NotificationPriority.HIGH: deque(),
            NotificationPriority.NORMAL: deque(),
            NotificationPriority.LOW: deque()
        }
        
        # Rate limiting tracking
        self.notification_history = defaultdict(lambda: {
            'minute': deque(),
            'hour': deque(),
            'last_sent': None
        })
        
        # Processing state
        self.is_processing = False
        self.processing_task = None
        
        # Statistics
        self.stats = {
            'sent': defaultdict(int),
            'dropped': defaultdict(int),
            'queued': defaultdict(int),
            'errors': defaultdict(int)
        }
        
        # Start processing task
        self._start_processing()
    
    def _start_processing(self):
        """Start the background notification processing task."""
        async def process_notifications():
            while True:
                try:
                    await self._process_queue()
                    await asyncio.sleep(1)  # Check queue every second
                except Exception as e:
                    self.logger.error(f"Error in notification processing: {e}")
                    await asyncio.sleep(5)  # Wait longer on error
        
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(process_notifications())
    
    async def send_notification(self, 
                               notification_type: NotificationType,
                               title: str,
                               message: str,
                               priority: NotificationPriority = NotificationPriority.NORMAL,
                               data: Optional[Dict[str, Any]] = None) -> bool:
        """Send a notification with intelligent queuing and rate limiting."""
        
        notification = NotificationMessage(
            type=notification_type,
            priority=priority,
            title=title,
            message=message,
            data=data or {}
        )
        
        # Check rate limits
        if not self._check_rate_limit(notification_type):
            self.logger.warning(f"Rate limit exceeded for {notification_type.value}, dropping notification")
            self.stats['dropped'][notification_type] += 1
            return False
        
        # Add to appropriate priority queue
        self.queues[priority].append(notification)
        self.stats['queued'][notification_type] += 1
        
        self.logger.debug(f"Queued {priority.value} notification: {title}")
        return True
    
    async def send_critical_alert(self, title: str, message: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Send critical alert that bypasses most rate limiting."""
        return await self.send_notification(
            NotificationType.ERROR,
            title,
            message,
            NotificationPriority.CRITICAL,
            data
        )
    
    async def send_trade_notification(self, 
                                     action: str,
                                     symbol: str,
                                     price: float,
                                     quantity: float,
                                     pnl: Optional[float] = None,
                                     reason: Optional[str] = None) -> bool:
        """Send trade-specific notification."""
        
        if action.upper() == "BUY":
            notification_type = NotificationType.TRADE_OPENED
            title = f"🟢 Position Opened - {symbol}"
            message = f"Bought {quantity:.6f} {symbol} at ${price:.4f}"
        else:
            notification_type = NotificationType.TRADE_CLOSED
            pnl_emoji = "🟢" if pnl and pnl > 0 else "🔴"
            title = f"{pnl_emoji} Position Closed - {symbol}"
            message = f"Sold {quantity:.6f} {symbol} at ${price:.4f}"
            if pnl is not None:
                message += f"\nP&L: ${pnl:.2f} ({(pnl/price/quantity)*100:.2f}%)"
            if reason:
                message += f"\nReason: {reason}"
        
        return await self.send_notification(
            notification_type,
            title,
            message,
            NotificationPriority.HIGH,
            {
                'symbol': symbol,
                'action': action,
                'price': price,
                'quantity': quantity,
                'pnl': pnl,
                'reason': reason
            }
        )
    
    async def send_portfolio_update(self, 
                                   total_value: float,
                                   daily_pnl: float,
                                   positions_count: int,
                                   top_performers: List[Dict[str, Any]]) -> bool:
        """Send portfolio update notification."""
        
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        title = f"📊 Portfolio Update"
        
        message = f"""
Portfolio Value: ${total_value:,.2f}
Daily P&L: {pnl_emoji} ${daily_pnl:,.2f} ({(daily_pnl/total_value)*100:.2f}%)
Active Positions: {positions_count}
"""
        
        if top_performers:
            message += "\n🏆 Top Performers:\n"
            for i, performer in enumerate(top_performers[:3], 1):
                message += f"{i}. {performer.get('symbol', 'Unknown')}: {performer.get('pnl_pct', 0):.2f}%\n"
        
        return await self.send_notification(
            NotificationType.PORTFOLIO_UPDATE,
            title,
            message,
            NotificationPriority.NORMAL,
            {
                'total_value': total_value,
                'daily_pnl': daily_pnl,
                'positions_count': positions_count,
                'top_performers': top_performers
            }
        )
    
    async def send_health_alert(self, 
                               health_status: str,
                               health_score: float,
                               issues: List[str]) -> bool:
        """Send system health alert."""
        
        if health_score >= 80:
            priority = NotificationPriority.LOW
            emoji = "🟢"
        elif health_score >= 60:
            priority = NotificationPriority.NORMAL
            emoji = "🟡"
        else:
            priority = NotificationPriority.HIGH
            emoji = "🔴"
        
        title = f"{emoji} System Health: {health_status.upper()}"
        message = f"Health Score: {health_score:.1f}/100"
        
        if issues:
            message += f"\n\n⚠️ Issues:\n"
            for issue in issues[:5]:  # Limit to 5 issues
                message += f"• {issue}\n"
        
        return await self.send_notification(
            NotificationType.HEALTH_CHECK,
            title,
            message,
            priority,
            {
                'health_status': health_status,
                'health_score': health_score,
                'issues': issues
            }
        )
    
    async def send_performance_alert(self, 
                                   alert_type: str,
                                   metric_name: str,
                                   current_value: float,
                                   threshold: float,
                                   recommendation: str) -> bool:
        """Send performance-related alert."""
        
        title = f"⚡ Performance Alert: {metric_name}"
        message = f"""
Alert Type: {alert_type}
Current Value: {current_value:.3f}
Threshold: {threshold:.3f}

💡 Recommendation: {recommendation}
"""
        
        priority = NotificationPriority.HIGH if alert_type == "Critical" else NotificationPriority.NORMAL
        
        return await self.send_notification(
            NotificationType.PERFORMANCE_ALERT,
            title,
            message,
            priority,
            {
                'alert_type': alert_type,
                'metric_name': metric_name,
                'current_value': current_value,
                'threshold': threshold,
                'recommendation': recommendation
            }
        )
    
    def _check_rate_limit(self, notification_type: NotificationType) -> bool:
        """Check if notification type is within rate limits."""
        
        config = self.rate_limits.get(notification_type)
        if not config:
            return True  # No limits configured
        
        history = self.notification_history[notification_type]
        now = datetime.now()
        
        # Clean old entries
        cutoff_minute = now - timedelta(minutes=1)
        cutoff_hour = now - timedelta(hours=1)
        
        history['minute'] = deque([t for t in history['minute'] if t > cutoff_minute])
        history['hour'] = deque([t for t in history['hour'] if t > cutoff_hour])
        
        # Check minute limit
        if len(history['minute']) >= config.max_per_minute:
            return False
        
        # Check hour limit
        if len(history['hour']) >= config.max_per_hour:
            return False
        
        # Check cooldown period
        if history['last_sent']:
            time_since_last = (now - history['last_sent']).total_seconds()
            if time_since_last < config.cooldown_seconds:
                return False
        
        return True
    
    def _record_sent_notification(self, notification_type: NotificationType):
        """Record that a notification was sent for rate limiting."""
        history = self.notification_history[notification_type]
        now = datetime.now()
        
        history['minute'].append(now)
        history['hour'].append(now)
        history['last_sent'] = now
        
        self.stats['sent'][notification_type] += 1
    
    async def _process_queue(self):
        """Process notification queues in priority order."""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            # Process in priority order
            for priority in [NotificationPriority.CRITICAL, NotificationPriority.HIGH, 
                           NotificationPriority.NORMAL, NotificationPriority.LOW]:
                
                queue = self.queues[priority]
                
                # Process up to 3 notifications per cycle to avoid blocking
                processed = 0
                while queue and processed < 3:
                    notification = queue.popleft()
                    
                    # Check rate limits again (may have changed)
                    if not self._check_rate_limit(notification.type):
                        # Re-queue low priority notifications, drop others
                        if priority == NotificationPriority.LOW:
                            queue.append(notification)
                        else:
                            self.stats['dropped'][notification.type] += 1
                        break
                    
                    # Send notification
                    success = await self._send_notification(notification)
                    
                    if success:
                        self._record_sent_notification(notification.type)
                        processed += 1
                    else:
                        # Retry logic
                        notification.retry_count += 1
                        if notification.retry_count < notification.max_retries:
                            queue.append(notification)  # Re-queue for retry
                        else:
                            self.stats['errors'][notification.type] += 1
                            self.logger.error(f"Failed to send notification after {notification.max_retries} retries")
                
                # Don't process lower priority if we hit rate limits on higher priority
                if processed == 0 and queue:
                    break
        
        finally:
            self.is_processing = False
    
    async def _send_notification(self, notification: NotificationMessage) -> bool:
        """Actually send the notification via the configured channel."""
        
        try:
            if self.telegram_notifier and self.telegram_notifier.is_enabled():
                
                # Format message based on type
                formatted_message = self._format_telegram_message(notification)
                
                # Send via Telegram
                await self.telegram_notifier.send_message(formatted_message)
                
                self.logger.debug(f"Sent {notification.type.value} notification: {notification.title}")
                return True
            else:
                # Log if no notifier available
                self.logger.info(f"Notification (no sender): {notification.title} - {notification.message}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
    
    def _format_telegram_message(self, notification: NotificationMessage) -> str:
        """Format notification for Telegram."""
        
        # Add priority indicators
        priority_indicators = {
            NotificationPriority.CRITICAL: "🚨",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.NORMAL: "ℹ️",
            NotificationPriority.LOW: "💭"
        }
        
        indicator = priority_indicators.get(notification.priority, "")
        
        # Format timestamp
        timestamp = notification.timestamp.strftime("%H:%M:%S")
        
        message = f"{indicator} *{notification.title}*\n"
        message += f"🕐 {timestamp}\n\n"
        message += notification.message
        
        # Add technical details for certain types
        if notification.data and notification.type in [NotificationType.ERROR, NotificationType.PERFORMANCE_ALERT]:
            message += "\n\n_Technical Details:_\n"
            for key, value in list(notification.data.items())[:3]:  # Limit to 3 details
                message += f"• {key}: {value}\n"
        
        return message
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        
        # Calculate queue sizes
        queue_sizes = {
            priority.value: len(queue) 
            for priority, queue in self.queues.items()
        }
        
        # Calculate rate limit status
        rate_limit_status = {}
        for notification_type in NotificationType:
            config = self.rate_limits.get(notification_type)
            if config:
                history = self.notification_history[notification_type]
                rate_limit_status[notification_type.value] = {
                    'minute_count': len(history['minute']),
                    'minute_limit': config.max_per_minute,
                    'hour_count': len(history['hour']),
                    'hour_limit': config.max_per_hour,
                    'can_send': self._check_rate_limit(notification_type)
                }
        
        return {
            'queue_sizes': queue_sizes,
            'total_queued': sum(queue_sizes.values()),
            'stats': {
                'sent': dict(self.stats['sent']),
                'dropped': dict(self.stats['dropped']),
                'queued': dict(self.stats['queued']),
                'errors': dict(self.stats['errors'])
            },
            'rate_limits': rate_limit_status,
            'is_processing': self.is_processing
        }
    
    def clear_queues(self):
        """Clear all notification queues."""
        for queue in self.queues.values():
            queue.clear()
        self.logger.info("Cleared all notification queues")
    
    def adjust_rate_limits(self, notification_type: NotificationType, **kwargs):
        """Dynamically adjust rate limits for a notification type."""
        if notification_type in self.rate_limits:
            config = self.rate_limits[notification_type]
            
            if 'max_per_minute' in kwargs:
                config.max_per_minute = kwargs['max_per_minute']
            if 'max_per_hour' in kwargs:
                config.max_per_hour = kwargs['max_per_hour']
            if 'cooldown_seconds' in kwargs:
                config.cooldown_seconds = kwargs['cooldown_seconds']
            
            self.logger.info(f"Adjusted rate limits for {notification_type.value}")
    
    async def shutdown(self):
        """Gracefully shutdown the notification manager."""
        self.logger.info("Shutting down notification manager...")
        
        # Process remaining critical and high priority notifications
        critical_count = len(self.queues[NotificationPriority.CRITICAL])
        high_count = len(self.queues[NotificationPriority.HIGH])
        
        if critical_count > 0 or high_count > 0:
            self.logger.info(f"Processing {critical_count + high_count} priority notifications before shutdown")
            
            # Process critical notifications
            while self.queues[NotificationPriority.CRITICAL]:
                notification = self.queues[NotificationPriority.CRITICAL].popleft()
                await self._send_notification(notification)
            
            # Process high priority notifications
            while self.queues[NotificationPriority.HIGH]:
                notification = self.queues[NotificationPriority.HIGH].popleft()
                await self._send_notification(notification)
        
        # Cancel processing task
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Notification manager shutdown complete")