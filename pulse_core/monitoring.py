"""
PULSE Platform Monitoring Utilities
====================================

Comprehensive monitoring and logging utilities for the PULSE trading platform:
- Performance monitoring decorator
- Security event logging
- Health monitoring with metrics tracking
- Request/response tracking
- Error tracking and alerting

Author: jetgause
Created: 2025-12-11
Version: 1.0.0
"""

import logging
import time
from functools import wraps
from typing import Callable, Dict, Any, Optional
import json
from datetime import datetime
from collections import defaultdict
import threading


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

def monitor_performance(func: Callable) -> Callable:
    """
    Decorator to monitor function performance.
    
    Tracks execution time and logs performance metrics.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
        
    Example:
        @monitor_performance
        def slow_function():
            time.sleep(1)
            return "done"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(
                f"Performance: {function_name} completed in {duration:.3f}s",
                extra={
                    'function': function_name,
                    'duration_seconds': duration,
                    'status': 'success'
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                f"Error in {function_name} after {duration:.3f}s: {str(e)}",
                extra={
                    'function': function_name,
                    'duration_seconds': duration,
                    'status': 'error',
                    'error': str(e)
                }
            )
            
            raise
    
    return wrapper


def monitor_async_performance(func: Callable) -> Callable:
    """
    Decorator to monitor async function performance.
    
    Args:
        func: Async function to monitor
        
    Returns:
        Wrapped async function with performance monitoring
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(
                f"Performance: {function_name} completed in {duration:.3f}s",
                extra={
                    'function': function_name,
                    'duration_seconds': duration,
                    'status': 'success'
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                f"Error in {function_name} after {duration:.3f}s: {str(e)}",
                extra={
                    'function': function_name,
                    'duration_seconds': duration,
                    'status': 'error',
                    'error': str(e)
                }
            )
            
            raise
    
    return wrapper


# ============================================================================
# SECURITY EVENT LOGGING
# ============================================================================

def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "WARNING"):
    """
    Log security-related events.
    
    Args:
        event_type: Type of security event (e.g., 'authentication_failed', 'rate_limit_exceeded')
        details: Dictionary with event details
        severity: Log severity level (INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        log_security_event(
            'authentication_failed',
            {'user': 'admin', 'ip': '192.168.1.1'},
            severity='WARNING'
        )
    """
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details,
        "severity": severity
    }
    
    log_message = f"Security Event: {json.dumps(event)}"
    
    if severity == "CRITICAL":
        logger.critical(log_message)
    elif severity == "ERROR":
        logger.error(log_message)
    elif severity == "WARNING":
        logger.warning(log_message)
    else:
        logger.info(log_message)


def log_authentication_event(user_id: str, success: bool, details: Optional[Dict[str, Any]] = None):
    """
    Log authentication events.
    
    Args:
        user_id: User identifier
        success: Whether authentication succeeded
        details: Optional additional details
    """
    event_details = {
        "user_id": user_id,
        "success": success
    }
    
    # Safely merge additional details
    if details is not None:
        event_details.update(details)
    
    log_security_event(
        "authentication",
        event_details,
        severity="INFO" if success else "WARNING"
    )


def log_access_control_event(user_id: str, resource: str, action: str, granted: bool):
    """
    Log access control events.
    
    Args:
        user_id: User identifier
        resource: Resource being accessed
        action: Action being performed
        granted: Whether access was granted
    """
    log_security_event(
        "access_control",
        {
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "granted": granted
        },
        severity="INFO" if granted else "WARNING"
    )


# ============================================================================
# HEALTH MONITORING
# ============================================================================

class HealthMonitor:
    """
    System health monitoring with metrics tracking.
    
    Tracks various system metrics including:
    - Total requests
    - Failed requests
    - Security violations
    - Rate limit hits
    - Average response times
    
    Example:
        monitor = HealthMonitor()
        monitor.record_request(success=True)
        monitor.record_security_violation()
        metrics = monitor.get_metrics()
    """
    
    def __init__(self):
        """Initialize health monitor."""
        self.metrics = {
            "requests_total": 0,
            "requests_failed": 0,
            "security_violations": 0,
            "rate_limit_hits": 0,
            "response_times": []
        }
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_request(self, success: bool = True, response_time: Optional[float] = None):
        """
        Record an API request.
        
        Args:
            success: Whether the request succeeded
            response_time: Optional response time in seconds
        """
        with self._lock:
            self.metrics["requests_total"] += 1
            
            if not success:
                self.metrics["requests_failed"] += 1
            
            if response_time is not None:
                self.metrics["response_times"].append(response_time)
                
                # Keep only last 1000 response times to prevent memory issues
                if len(self.metrics["response_times"]) > 1000:
                    self.metrics["response_times"] = self.metrics["response_times"][-1000:]
    
    def record_security_violation(self, violation_type: Optional[str] = None):
        """
        Record a security violation.
        
        Args:
            violation_type: Optional type of violation
        """
        with self._lock:
            self.metrics["security_violations"] += 1
        
        if violation_type:
            log_security_event(
                "security_violation",
                {"type": violation_type},
                severity="ERROR"
            )
    
    def record_rate_limit(self, identifier: Optional[str] = None):
        """
        Record a rate limit hit.
        
        Args:
            identifier: Optional identifier that hit the rate limit
        """
        with self._lock:
            self.metrics["rate_limit_hits"] += 1
        
        if identifier:
            log_security_event(
                "rate_limit_exceeded",
                {"identifier": identifier},
                severity="WARNING"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary with current metrics
        """
        with self._lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            uptime_seconds = time.time() - self.start_time
            metrics["uptime_seconds"] = uptime_seconds
            
            if metrics["requests_total"] > 0:
                metrics["failure_rate"] = metrics["requests_failed"] / metrics["requests_total"]
            else:
                metrics["failure_rate"] = 0.0
            
            if metrics["response_times"]:
                metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
                metrics["min_response_time"] = min(metrics["response_times"])
                metrics["max_response_time"] = max(metrics["response_times"])
            else:
                metrics["avg_response_time"] = 0.0
                metrics["min_response_time"] = 0.0
                metrics["max_response_time"] = 0.0
            
            # Remove raw response times from output
            del metrics["response_times"]
            
            return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = {
                "requests_total": 0,
                "requests_failed": 0,
                "security_violations": 0,
                "rate_limit_hits": 0,
                "response_times": []
            }
            self.start_time = time.time()
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status.
        
        Returns:
            Dictionary with health status and metrics
        """
        metrics = self.get_metrics()
        
        # Determine health status
        if metrics["security_violations"] > 10:
            status = "critical"
        elif metrics["failure_rate"] > 0.1:
            status = "degraded"
        elif metrics["rate_limit_hits"] > 50:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }


# ============================================================================
# REQUEST TRACKING
# ============================================================================

class RequestTracker:
    """
    Track individual requests for detailed monitoring.
    
    Example:
        tracker = RequestTracker()
        
        with tracker.track_request("GET", "/api/users") as req:
            # Process request
            req.set_status(200)
            req.add_metadata("user_count", 42)
    """
    
    def __init__(self):
        """Initialize request tracker."""
        self.requests = defaultdict(list)
        self._lock = threading.Lock()
    
    def track_request(self, method: str, path: str):
        """
        Context manager for tracking a request.
        
        Args:
            method: HTTP method
            path: Request path
            
        Returns:
            Request context
        """
        return RequestContext(self, method, path)
    
    def record_request_data(self, data: Dict[str, Any]):
        """
        Record request data.
        
        Args:
            data: Request data dictionary
        """
        with self._lock:
            endpoint = f"{data['method']} {data['path']}"
            self.requests[endpoint].append(data)
            
            # Keep only last 100 requests per endpoint
            if len(self.requests[endpoint]) > 100:
                self.requests[endpoint] = self.requests[endpoint][-100:]
    
    def get_endpoint_stats(self, method: str, path: str) -> Dict[str, Any]:
        """
        Get statistics for a specific endpoint.
        
        Args:
            method: HTTP method
            path: Request path
            
        Returns:
            Statistics dictionary
        """
        endpoint = f"{method} {path}"
        
        with self._lock:
            requests = self.requests.get(endpoint, [])
            
            if not requests:
                return {
                    "total_requests": 0,
                    "avg_duration": 0.0,
                    "error_rate": 0.0
                }
            
            total = len(requests)
            errors = sum(1 for r in requests if r.get("status_code", 200) >= 400)
            durations = [r["duration"] for r in requests if "duration" in r]
            
            return {
                "total_requests": total,
                "avg_duration": sum(durations) / len(durations) if durations else 0.0,
                "error_rate": errors / total if total > 0 else 0.0
            }


class RequestContext:
    """Context for tracking a single request."""
    
    def __init__(self, tracker: RequestTracker, method: str, path: str):
        """
        Initialize request context.
        
        Args:
            tracker: Parent RequestTracker
            method: HTTP method
            path: Request path
        """
        self.tracker = tracker
        self.method = method
        self.path = path
        self.start_time = None
        self.data = {
            "method": method,
            "path": path,
            "metadata": {}
        }
    
    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record request."""
        duration = time.time() - self.start_time
        self.data["duration"] = duration
        
        if exc_type is not None:
            self.data["error"] = str(exc_val)
            self.data["status_code"] = 500
        
        self.tracker.record_request_data(self.data)
    
    def set_status(self, status_code: int):
        """Set response status code."""
        self.data["status_code"] = status_code
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to request."""
        self.data["metadata"][key] = value


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

# Global health monitor instance
health_monitor = HealthMonitor()

# Global request tracker instance
request_tracker = RequestTracker()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'monitor_performance',
    'monitor_async_performance',
    'log_security_event',
    'log_authentication_event',
    'log_access_control_event',
    'HealthMonitor',
    'RequestTracker',
    'RequestContext',
    'health_monitor',
    'request_tracker',
]


if __name__ == "__main__":
    # Example usage
    print("PULSE Monitoring Module")
    print("=" * 60)
    
    # Performance monitoring example
    @monitor_performance
    def example_function():
        time.sleep(0.1)
        return "done"
    
    print("\n1. Performance Monitoring:")
    result = example_function()
    print(f"Result: {result}")
    
    # Health monitoring example
    print("\n2. Health Monitoring:")
    monitor = HealthMonitor()
    monitor.record_request(success=True, response_time=0.1)
    monitor.record_request(success=True, response_time=0.15)
    monitor.record_request(success=False, response_time=0.5)
    monitor.record_security_violation("test_violation")
    
    health_status = monitor.get_health_status()
    print(f"Health Status: {health_status['status']}")
    print(f"Metrics: {json.dumps(health_status['metrics'], indent=2)}")
    
    # Security event logging
    print("\n3. Security Event Logging:")
    log_security_event("test_event", {"detail": "test"}, severity="INFO")
    
    print("\n" + "=" * 60)
    print("Monitoring module loaded successfully!")
