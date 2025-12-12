"""
Real-time Performance Monitoring for PULSE Trading Platform

This module provides comprehensive performance monitoring with metrics collection:
- Request rate tracking
- Response time histograms
- Error rate monitoring
- Resource usage (CPU, memory)
- Database connection pool stats
- Cache hit rates
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import json

# Try to import psutil for system metrics
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class PerformanceMonitor:
    """
    Real-time performance monitoring and metrics collection.
    
    This class tracks various performance metrics and provides
    real-time monitoring capabilities for the PULSE system.
    """
    
    def __init__(self, window_size: int = 300):
        """
        Initialize the performance monitor.
        
        Args:
            window_size: Size of the rolling window in seconds (default: 5 minutes)
        """
        self.window_size = window_size
        self.start_time = time.time()
        
        # Request metrics
        self.request_times = deque(maxlen=10000)
        self.response_times = deque(maxlen=10000)
        self.error_count = 0
        self.total_requests = 0
        
        # Endpoint metrics
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'errors': 0,
            'total_time': 0.0,
            'response_times': deque(maxlen=1000)
        })
        
        # Resource metrics
        self.cpu_samples = deque(maxlen=300)  # 5 minutes at 1 sample/sec
        self.memory_samples = deque(maxlen=300)
        
        # Database metrics
        self.db_query_times = deque(maxlen=1000)
        self.db_slow_queries = []
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start background monitoring if psutil available
        if HAS_PSUTIL:
            self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self._monitor_thread.start()
    
    def record_request(self, endpoint: str, duration: float, status_code: int):
        """
        Record a request and its metrics.
        
        Args:
            endpoint: The endpoint that was called
            duration: Request duration in seconds
            status_code: HTTP status code
        """
        with self._lock:
            current_time = time.time()
            
            # Update overall metrics
            self.total_requests += 1
            self.request_times.append(current_time)
            self.response_times.append(duration)
            
            # Check if error
            if status_code >= 400:
                self.error_count += 1
            
            # Update endpoint-specific metrics
            metrics = self.endpoint_metrics[endpoint]
            metrics['count'] += 1
            metrics['total_time'] += duration
            metrics['response_times'].append(duration)
            
            if status_code >= 400:
                metrics['errors'] += 1
    
    def record_db_query(self, duration: float, query_type: str = "unknown"):
        """
        Record a database query execution time.
        
        Args:
            duration: Query duration in seconds
            query_type: Type of query (SELECT, INSERT, etc.)
        """
        with self._lock:
            self.db_query_times.append(duration)
            
            # Track slow queries (>50ms)
            if duration > 0.05:
                self.db_slow_queries.append({
                    'duration': duration,
                    'type': query_type,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only last 100 slow queries
                if len(self.db_slow_queries) > 100:
                    self.db_slow_queries.pop(0)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        with self._lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        with self._lock:
            self.cache_misses += 1
    
    def get_request_rate(self) -> float:
        """
        Calculate current request rate (requests per second).
        
        Returns:
            Request rate in req/s
        """
        with self._lock:
            if not self.request_times:
                return 0.0
            
            current_time = time.time()
            cutoff_time = current_time - 60  # Last minute
            
            recent_requests = sum(1 for t in self.request_times if t > cutoff_time)
            return recent_requests / 60.0
    
    def get_error_rate(self) -> float:
        """
        Calculate error rate as percentage.
        
        Returns:
            Error rate as percentage
        """
        with self._lock:
            if self.total_requests == 0:
                return 0.0
            return (self.error_count / self.total_requests) * 100
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """
        Get response time statistics.
        
        Returns:
            Dictionary with min, max, avg, median, p95, p99
        """
        with self._lock:
            if not self.response_times:
                return {
                    'min': 0.0, 'max': 0.0, 'avg': 0.0,
                    'median': 0.0, 'p95': 0.0, 'p99': 0.0
                }
            
            sorted_times = sorted(self.response_times)
            count = len(sorted_times)
            
            return {
                'min': sorted_times[0] * 1000,  # Convert to ms
                'max': sorted_times[-1] * 1000,
                'avg': sum(sorted_times) / count * 1000,
                'median': sorted_times[count // 2] * 1000,
                'p95': sorted_times[int(count * 0.95)] * 1000,
                'p99': sorted_times[int(count * 0.99)] * 1000
            }
    
    def get_endpoint_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get per-endpoint statistics.
        
        Returns:
            Dictionary of endpoint stats
        """
        with self._lock:
            stats = {}
            
            for endpoint, metrics in self.endpoint_metrics.items():
                if metrics['count'] > 0:
                    avg_time = metrics['total_time'] / metrics['count']
                    error_rate = (metrics['errors'] / metrics['count']) * 100
                    
                    response_times = list(metrics['response_times'])
                    if response_times:
                        sorted_times = sorted(response_times)
                        count = len(sorted_times)
                        p95 = sorted_times[int(count * 0.95)] if count > 0 else 0
                    else:
                        p95 = 0
                    
                    stats[endpoint] = {
                        'count': metrics['count'],
                        'errors': metrics['errors'],
                        'error_rate': error_rate,
                        'avg_response_time_ms': avg_time * 1000,
                        'p95_response_time_ms': p95 * 1000
                    }
            
            return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        with self._lock:
            total = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total * 100) if total > 0 else 0.0
            
            return {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'total': total,
                'hit_rate': hit_rate
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database performance statistics.
        
        Returns:
            Dictionary with database metrics
        """
        with self._lock:
            if not self.db_query_times:
                return {
                    'total_queries': 0,
                    'avg_time_ms': 0.0,
                    'slow_queries': 0
                }
            
            sorted_times = sorted(self.db_query_times)
            count = len(sorted_times)
            
            return {
                'total_queries': count,
                'avg_time_ms': sum(sorted_times) / count * 1000,
                'median_time_ms': sorted_times[count // 2] * 1000,
                'p95_time_ms': sorted_times[int(count * 0.95)] * 1000,
                'slow_queries': len(self.db_slow_queries)
            }
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get system resource statistics.
        
        Returns:
            Dictionary with CPU and memory metrics
        """
        if not HAS_PSUTIL:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_mb': 0.0
            }
        
        with self._lock:
            if not self.cpu_samples or not self.memory_samples:
                return {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_mb': 0.0
                }
            
            return {
                'cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples),
                'memory_percent': sum(s[0] for s in self.memory_samples) / len(self.memory_samples),
                'memory_mb': sum(s[1] for s in self.memory_samples) / len(self.memory_samples)
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all metrics.
        
        Returns:
            Dictionary with all performance metrics
        """
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'timestamp': datetime.now().isoformat(),
            'requests': {
                'total': self.total_requests,
                'rate_per_second': self.get_request_rate(),
                'error_count': self.error_count,
                'error_rate': self.get_error_rate()
            },
            'response_times': self.get_response_time_stats(),
            'endpoints': self.get_endpoint_stats(),
            'database': self.get_database_stats(),
            'cache': self.get_cache_stats(),
            'resources': self.get_resource_stats()
        }
    
    def _monitor_resources(self):
        """Background thread for monitoring system resources."""
        import psutil
        process = psutil.Process()
        
        while True:
            try:
                with self._lock:
                    # CPU percentage
                    cpu_percent = process.cpu_percent(interval=1)
                    self.cpu_samples.append(cpu_percent)
                    
                    # Memory usage
                    mem_info = process.memory_info()
                    mem_percent = process.memory_percent()
                    mem_mb = mem_info.rss / 1024 / 1024
                    self.memory_samples.append((mem_percent, mem_mb))
                
                time.sleep(1)
            except Exception:
                time.sleep(5)
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.request_times.clear()
            self.response_times.clear()
            self.error_count = 0
            self.total_requests = 0
            self.endpoint_metrics.clear()
            self.db_query_times.clear()
            self.db_slow_queries.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.start_time = time.time()


# Global performance monitor instance
_monitor = None


def get_monitor() -> PerformanceMonitor:
    """
    Get or create the global performance monitor instance.
    
    Returns:
        Global PerformanceMonitor instance
    """
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


# Convenience functions
def record_request(endpoint: str, duration: float, status_code: int):
    """Record a request metric."""
    get_monitor().record_request(endpoint, duration, status_code)


def record_db_query(duration: float, query_type: str = "unknown"):
    """Record a database query metric."""
    get_monitor().record_db_query(duration, query_type)


def record_cache_hit():
    """Record a cache hit."""
    get_monitor().record_cache_hit()


def record_cache_miss():
    """Record a cache miss."""
    get_monitor().record_cache_miss()


def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of all performance metrics."""
    return get_monitor().get_summary()
