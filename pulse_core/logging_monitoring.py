"""
Comprehensive Logging and Monitoring Module
==========================================

Enterprise-grade logging, monitoring, and audit trail system:
- Structured logging with JSON format
- Security event logging
- Audit trail tracking
- Performance monitoring
- Error tracking and alerting
- Log rotation and retention
- Metrics collection
- Health check monitoring
- Real-time log streaming
- Integration with popular monitoring tools

Author: jetgause
Created: 2025-12-10
Version: 1.0.0
"""

import os
import sys
import json
import logging
import logging.handlers
import traceback
import time
import functools
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
from collections import defaultdict, deque
import hashlib

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventCategory(Enum):
    """Categories for audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SECURITY = "security"
    SYSTEM = "system"
    API = "api"
    USER_ACTION = "user_action"
    ADMIN_ACTION = "admin_action"


class SecurityEventType(Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    ACCESS_DENIED = "access_denied"
    PERMISSION_ESCALATION = "permission_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_ATTEMPT = "csrf_attempt"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AuditEvent:
    """Audit trail event."""
    event_id: str
    timestamp: str
    category: str
    action: str
    user_id: Optional[str]
    user_ip: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    status: str  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SecurityEvent:
    """Security-specific event."""
    event_id: str
    timestamp: str
    event_type: str
    severity: str  # low, medium, high, critical
    user_id: Optional[str]
    user_ip: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)


@dataclass
class PerformanceMetric:
    """Performance measurement."""
    name: str
    value: float
    unit: str
    timestamp: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
            extra=getattr(record, 'extra', {})
        )
        
        # Add exception info if present
        if record.exc_info:
            log_entry.extra['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return log_entry.to_json()


class StructuredLogger:
    """Structured logging with context."""
    
    def __init__(self, name: str, log_dir: Optional[str] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup log handlers."""
        # Console handler with color support
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # JSON file handler with rotation
        json_file = self.log_dir / "app.json.log"
        json_handler = logging.handlers.RotatingFileHandler(
            json_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
        
        # Error file handler
        error_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)
    
    def log(self, level: LogLevel, message: str, **kwargs):
        """Log with extra context."""
        extra = {'extra': kwargs}
        
        if level == LogLevel.DEBUG:
            self.logger.debug(message, extra=extra)
        elif level == LogLevel.INFO:
            self.logger.info(message, extra=extra)
        elif level == LogLevel.WARNING:
            self.logger.warning(message, extra=extra)
        elif level == LogLevel.ERROR:
            self.logger.error(message, extra=extra)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, exc_info=True, **kwargs):
        """Log exception with traceback."""
        extra = {'extra': kwargs}
        self.logger.exception(message, exc_info=exc_info, extra=extra)


class AuditLogger:
    """Audit trail logging."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs/audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup audit log file
        self.audit_file = self.log_dir / "audit.json.log"
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Rotating file handler for audit logs
        handler = logging.handlers.RotatingFileHandler(
            self.audit_file,
            maxBytes=50 * 1024 * 1024,  # 50 MB
            backupCount=10
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # In-memory recent events
        self.recent_events: deque = deque(maxlen=1000)
    
    def log_event(
        self,
        category: EventCategory,
        action: str,
        status: str,
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log an audit event.
        
        Returns:
            Event ID
        """
        # Generate event ID
        event_data = f"{datetime.utcnow().isoformat()}{user_id}{action}"
        event_id = hashlib.sha256(event_data.encode()).hexdigest()[:16]
        
        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            category=category.value,
            action=action,
            user_id=user_id,
            user_ip=user_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
            details=kwargs.get('details', {}),
            metadata=kwargs.get('metadata', {})
        )
        
        # Log to file
        self.logger.info(event.to_json())
        
        # Store in memory
        self.recent_events.append(event)
        
        return event_id
    
    def get_recent_events(self, limit: int = 100) -> List[AuditEvent]:
        """Get recent audit events."""
        return list(self.recent_events)[-limit:]


class SecurityLogger:
    """Security event logging."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize security logger.
        
        Args:
            log_dir: Directory for security logs
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs/security")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup security log file
        self.security_file = self.log_dir / "security.json.log"
        self.logger = logging.getLogger("security")
        self.logger.setLevel(logging.INFO)
        
        # Rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            self.security_file,
            maxBytes=50 * 1024 * 1024,
            backupCount=10
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Track security events
        self.events_by_ip: Dict[str, List] = defaultdict(list)
        self.events_by_user: Dict[str, List] = defaultdict(list)
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        severity: str,
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        **kwargs
    ) -> str:
        """Log a security event."""
        # Generate event ID
        event_data = f"{datetime.utcnow().isoformat()}{user_ip}{event_type.value}"
        event_id = hashlib.sha256(event_data.encode()).hexdigest()[:16]
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(event_type, user_ip, user_id)
        
        # Create event
        event = SecurityEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type.value,
            severity=severity,
            user_id=user_id,
            user_ip=user_ip,
            user_agent=user_agent,
            details=kwargs.get('details', {}),
            risk_score=risk_score
        )
        
        # Log to file
        self.logger.warning(event.to_json())
        
        # Track by IP and user
        if user_ip:
            self.events_by_ip[user_ip].append(event)
        if user_id:
            self.events_by_user[user_id].append(event)
        
        # Alert on high-risk events
        if risk_score > 0.7:
            self._send_alert(event)
        
        return event_id
    
    def _calculate_risk_score(
        self,
        event_type: SecurityEventType,
        user_ip: Optional[str],
        user_id: Optional[str]
    ) -> float:
        """Calculate risk score for event."""
        score = 0.0
        
        # Base score by event type
        high_risk_events = {
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventType.XSS_ATTEMPT,
            SecurityEventType.DATA_BREACH_ATTEMPT,
        }
        medium_risk_events = {
            SecurityEventType.BRUTE_FORCE_ATTEMPT,
            SecurityEventType.PERMISSION_ESCALATION,
            SecurityEventType.SUSPICIOUS_ACTIVITY,
        }
        
        if event_type in high_risk_events:
            score += 0.5
        elif event_type in medium_risk_events:
            score += 0.3
        else:
            score += 0.1
        
        # Increase score for repeated events
        if user_ip:
            recent_events = self.events_by_ip[user_ip][-10:]
            if len(recent_events) > 5:
                score += 0.2
        
        if user_id:
            recent_events = self.events_by_user[user_id][-10:]
            if len(recent_events) > 5:
                score += 0.2
        
        return min(score, 1.0)
    
    def _send_alert(self, event: SecurityEvent):
        """Send alert for high-risk event."""
        # In production, integrate with alerting system
        # (e.g., email, Slack, PagerDuty)
        self.logger.critical(f"SECURITY ALERT: {event.to_json()}")


class PerformanceMonitor:
    """Performance monitoring and metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: List[PerformanceMetric] = []
        self.endpoint_timings: Dict[str, List[float]] = defaultdict(list)
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        **tags
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow().isoformat(),
            tags=tags
        )
        self.metrics.append(metric)
    
    def record_request_time(self, endpoint: str, duration: float):
        """Record API request duration."""
        self.endpoint_timings[endpoint].append(duration)
        self.record_metric(
            name="request_duration",
            value=duration,
            unit="seconds",
            endpoint=endpoint
        )
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, float]:
        """Get statistics for an endpoint."""
        timings = self.endpoint_timings.get(endpoint, [])
        
        if not timings:
            return {}
        
        return {
            "count": len(timings),
            "mean": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
            "p50": self._percentile(timings, 50),
            "p95": self._percentile(timings, 95),
            "p99": self._percentile(timings, 99),
        }
    
    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_io": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
            }
        }


class RequestLogger:
    """HTTP request/response logging."""
    
    def __init__(self, logger: StructuredLogger):
        """
        Initialize request logger.
        
        Args:
            logger: StructuredLogger instance
        """
        self.logger = logger
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs
    ):
        """Log HTTP request."""
        self.logger.info(
            f"{method} {path} - {status_code}",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration * 1000,
            user_id=user_id,
            ip_address=ip_address,
            **kwargs
        )


def timed(logger: Optional[StructuredLogger] = None):
    """Decorator to measure function execution time."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                
                if logger:
                    logger.debug(
                        f"Function {func.__name__} completed",
                        duration_seconds=duration
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start
                
                if logger:
                    logger.error(
                        f"Function {func.__name__} failed",
                        duration_seconds=duration,
                        error=str(e)
                    )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                if logger:
                    logger.debug(
                        f"Function {func.__name__} completed",
                        duration_seconds=duration
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start
                
                if logger:
                    logger.error(
                        f"Function {func.__name__} failed",
                        duration_seconds=duration,
                        error=str(e)
                    )
                
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Global instances
app_logger = StructuredLogger("pulse_app")
audit_logger = AuditLogger()
security_logger = SecurityLogger()
performance_monitor = PerformanceMonitor()


# Example usage
if __name__ == "__main__":
    print("Logging and Monitoring Module")
    print("=" * 60)
    
    # Structured logging
    print("\n1. Structured Logging:")
    logger = StructuredLogger("demo")
    logger.info("Application started", version="1.0.0", environment="production")
    logger.warning("High memory usage detected", memory_percent=85.5)
    logger.error("Database connection failed", db_host="localhost", error_code=1234)
    
    # Audit logging
    print("\n2. Audit Logging:")
    audit = AuditLogger()
    event_id = audit.log_event(
        category=EventCategory.DATA_MODIFICATION,
        action="task_updated",
        status="success",
        user_id="user-123",
        user_ip="192.168.1.100",
        resource_type="task",
        resource_id="task-456",
        details={"field": "status", "old_value": "pending", "new_value": "completed"}
    )
    print(f"Logged audit event: {event_id}")
    
    # Security logging
    print("\n3. Security Event Logging:")
    security = SecurityLogger()
    sec_event_id = security.log_security_event(
        event_type=SecurityEventType.LOGIN_FAILURE,
        severity="medium",
        user_id="user-123",
        user_ip="192.168.1.100",
        details={"reason": "invalid_password", "attempt": 3}
    )
    print(f"Logged security event: {sec_event_id}")
    
    # Performance monitoring
    print("\n4. Performance Monitoring:")
    perf = PerformanceMonitor()
    perf.record_request_time("/api/tasks", 0.125)
    perf.record_request_time("/api/tasks", 0.145)
    perf.record_request_time("/api/tasks", 0.110)
    stats = perf.get_endpoint_stats("/api/tasks")
    print(f"Endpoint stats: {stats}")
    
    print("\n" + "=" * 60)
    print("Module loaded successfully!")


import asyncio
