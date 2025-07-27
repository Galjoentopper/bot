"""Enhanced performance monitoring and intelligent caching for the paper trader."""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring system efficiency."""
    operation_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    data_hash: str = ""
    size_bytes: int = 0


class IntelligentCache:
    """Intelligent caching system with automatic cleanup and performance optimization."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300, max_memory_mb: int = 500):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.cleanup_count = 0
        
        # LRU tracking
        self.access_order = deque(maxlen=max_size * 2)
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Cleanup every minute
                    self._cleanup_expired()
                    self._cleanup_memory_pressure()
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
        
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if self._is_expired(entry):
            del self.cache[key]
            self.miss_count += 1
            return None
        
        # Update access statistics
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        self.access_order.append(key)
        self.hit_count += 1
        
        return entry.data
    
    def put(self, key: str, data: Any, custom_ttl: Optional[int] = None) -> bool:
        """Put item in cache."""
        try:
            # Calculate data size
            size_bytes = self._estimate_size(data)
            
            # Create cache entry
            entry = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                data_hash=self._calculate_hash(data),
                size_bytes=size_bytes
            )
            
            # Check if we need to make space
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Check memory pressure
            total_memory = sum(e.size_bytes for e in self.cache.values()) + size_bytes
            if total_memory > self.max_memory_bytes:
                self._cleanup_memory_pressure()
            
            self.cache[key] = entry
            self.access_order.append(key)
            return True
            
        except Exception as e:
            self.logger.error(f"Cache put error for key {key}: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        keys_to_remove = [key for key in self.cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self.cache[key]
        return len(keys_to_remove)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.now() - entry.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.cleanup_count += len(expired_keys)
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _cleanup_memory_pressure(self):
        """Clean up cache when memory pressure is high."""
        current_memory = sum(e.size_bytes for e in self.cache.values())
        
        if current_memory <= self.max_memory_bytes:
            return
        
        # Sort by access frequency and recency (LFU + LRU)
        entries_by_score = []
        for key, entry in self.cache.items():
            # Score: lower is worse (more likely to be evicted)
            recency_score = (datetime.now() - entry.last_accessed).total_seconds()
            frequency_score = 1.0 / max(1, entry.access_count)
            combined_score = recency_score * frequency_score
            entries_by_score.append((combined_score, key))
        
        # Sort by score (highest first = most likely to evict)
        entries_by_score.sort(reverse=True)
        
        # Remove entries until memory is under limit
        removed_count = 0
        for _, key in entries_by_score:
            if current_memory <= self.max_memory_bytes * 0.8:  # Leave 20% buffer
                break
            
            if key in self.cache:
                current_memory -= self.cache[key].size_bytes
                del self.cache[key]
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} entries due to memory pressure")
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find least recently accessed entry
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        
        del self.cache[oldest_key]
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate hash of data for integrity checking."""
        try:
            if isinstance(data, pd.DataFrame):
                return hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()[:16]
            elif isinstance(data, (dict, list)):
                return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
            else:
                return hashlib.md5(str(data).encode()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data).encode('utf-8'))
            else:
                return 1024  # Default estimate
        except Exception:
            return 1024
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        total_memory = sum(e.size_bytes for e in self.cache.values())
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cleanup_count': self.cleanup_count,
            'memory_usage_bytes': total_memory,
            'memory_usage_mb': total_memory / (1024 * 1024),
            'memory_limit_mb': self.max_memory_bytes / (1024 * 1024)
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()


class PerformanceMonitor:
    """Advanced performance monitoring with detailed analytics."""
    
    def __init__(self, max_history: int = 10000):
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # Operation-specific tracking
        self.operation_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'cache_hits': 0,
            'errors': 0
        })
        
        # Real-time metrics
        self.current_operations = {}
        
    async def monitor_operation(self, operation_name: str, func, *args, use_cache: bool = True, **kwargs):
        """Monitor an operation's performance."""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        cache_hit = False
        error = None
        
        try:
            # Check cache if enabled
            if use_cache and hasattr(self, 'cache'):
                cache_key = self._generate_cache_key(operation_name, args, kwargs)
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    cache_hit = True
                    result = cached_result
                else:
                    # Execute operation
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Cache result
                    self.cache.put(cache_key, result)
            else:
                # Execute operation without caching
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            error = str(e)
            self.logger.error(f"Error in monitored operation {operation_name}: {e}")
            raise
            
        finally:
            # Record metrics
            execution_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_before=memory_before,
                memory_after=memory_after,
                cache_hit=cache_hit,
                error=error
            )
            
            self._record_metrics(metrics)
    
    def _generate_cache_key(self, operation_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for operation."""
        try:
            # Create a deterministic key from operation name and parameters
            key_data = {
                'operation': operation_name,
                'args': str(args),
                'kwargs': sorted(kwargs.items()) if kwargs else []
            }
            key_str = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception:
            return f"{operation_name}_{hash(str(args))}_{hash(str(kwargs))}"
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        # Add to history
        self.metrics_history.append(metrics)
        
        # Trim history if needed
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        # Update operation statistics
        stats = self.operation_stats[metrics.operation_name]
        stats['count'] += 1
        stats['total_time'] += metrics.execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], metrics.execution_time)
        stats['max_time'] = max(stats['max_time'], metrics.execution_time)
        
        if metrics.cache_hit:
            stats['cache_hits'] += 1
        
        if metrics.error:
            stats['errors'] += 1
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_performance_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for all operations or specific operation."""
        if operation_name:
            if operation_name not in self.operation_stats:
                return {}
            
            stats = self.operation_stats[operation_name]
            cache_hit_rate = stats['cache_hits'] / max(1, stats['count'])
            error_rate = stats['errors'] / max(1, stats['count'])
            
            return {
                'operation': operation_name,
                'total_calls': stats['count'],
                'avg_execution_time': stats['avg_time'],
                'min_execution_time': stats['min_time'],
                'max_execution_time': stats['max_time'],
                'cache_hit_rate': cache_hit_rate,
                'error_rate': error_rate,
                'total_time': stats['total_time']
            }
        else:
            # Summary for all operations
            summary = {}
            for op_name, stats in self.operation_stats.items():
                cache_hit_rate = stats['cache_hits'] / max(1, stats['count'])
                error_rate = stats['errors'] / max(1, stats['count'])
                
                summary[op_name] = {
                    'calls': stats['count'],
                    'avg_time': stats['avg_time'],
                    'cache_hit_rate': cache_hit_rate,
                    'error_rate': error_rate
                }
            
            return summary
    
    def get_slowest_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest operations by average execution time."""
        operations = []
        for op_name, stats in self.operation_stats.items():
            operations.append({
                'operation': op_name,
                'avg_time': stats['avg_time'],
                'calls': stats['count'],
                'total_time': stats['total_time']
            })
        
        return sorted(operations, key=lambda x: x['avg_time'], reverse=True)[:limit]
    
    def get_recent_performance(self, minutes: int = 10) -> Dict[str, Any]:
        """Get performance metrics for recent time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        total_operations = len(recent_metrics)
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        errors = sum(1 for m in recent_metrics if m.error)
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / total_operations
        
        return {
            'time_period_minutes': minutes,
            'total_operations': total_operations,
            'cache_hit_rate': cache_hits / total_operations,
            'error_rate': errors / total_operations,
            'avg_execution_time': avg_execution_time,
            'operations_per_minute': total_operations / minutes
        }
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        summary = self.get_performance_summary()
        recent = self.get_recent_performance(10)
        slowest = self.get_slowest_operations(5)
        
        report = f"""
📊 Performance Monitor Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🕒 Recent Performance (10 minutes):
   • Total Operations: {recent.get('total_operations', 0)}
   • Avg Execution Time: {recent.get('avg_execution_time', 0):.3f}s
   • Cache Hit Rate: {recent.get('cache_hit_rate', 0):.1%}
   • Error Rate: {recent.get('error_rate', 0):.1%}
   • Ops/Minute: {recent.get('operations_per_minute', 0):.1f}

🔍 Operation Summary:
"""
        
        for op_name, stats in summary.items():
            report += f"""   • {op_name}:
     - Calls: {stats['calls']}
     - Avg Time: {stats['avg_time']:.3f}s
     - Cache Hit Rate: {stats['cache_hit_rate']:.1%}
     - Error Rate: {stats['error_rate']:.1%}
"""
        
        if slowest:
            report += "\n🐌 Slowest Operations:\n"
            for i, op in enumerate(slowest, 1):
                report += f"   {i}. {op['operation']}: {op['avg_time']:.3f}s ({op['calls']} calls)\n"
        
        return report


class EnhancedFeatureCache:
    """Enhanced caching system specifically for feature engineering."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.cache = IntelligentCache(max_size=500, ttl_seconds=600)  # 10 minute TTL
        self.performance_monitor = performance_monitor
        self.logger = logging.getLogger(__name__)
        
        # Feature-specific settings
        self.feature_dependencies = {
            'technical_indicators': ['close', 'high', 'low', 'volume'],
            'returns': ['close'],
            'volatility': ['close'],
            'volume_features': ['volume', 'close']
        }
    
    async def get_cached_features(self, symbol: str, data_hash: str, feature_type: str = 'all') -> Optional[pd.DataFrame]:
        """Get cached features for a symbol and data hash."""
        cache_key = f"features_{symbol}_{data_hash}_{feature_type}"
        
        return await self.performance_monitor.monitor_operation(
            f"feature_cache_get_{feature_type}",
            self.cache.get,
            cache_key
        )
    
    async def cache_features(self, symbol: str, data_hash: str, features: pd.DataFrame, feature_type: str = 'all') -> bool:
        """Cache features for a symbol and data hash."""
        cache_key = f"features_{symbol}_{data_hash}_{feature_type}"
        
        return await self.performance_monitor.monitor_operation(
            f"feature_cache_put_{feature_type}",
            self.cache.put,
            cache_key,
            features
        )
    
    def calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of input data for cache key."""
        try:
            # Use last 5 rows and key columns for hash
            key_columns = ['close', 'high', 'low', 'volume'] if all(col in data.columns for col in ['close', 'high', 'low', 'volume']) else data.columns[:4]
            hash_data = data[key_columns].tail(5)
            return hashlib.md5(str(hash_data.values.tobytes()).encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.warning(f"Error calculating data hash: {e}")
            return f"fallback_{int(time.time())}"
    
    def invalidate_symbol_cache(self, symbol: str) -> int:
        """Invalidate all cached features for a symbol."""
        return self.cache.invalidate_pattern(f"features_{symbol}_")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()