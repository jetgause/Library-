"""
SmartTool Base Class Implementation
====================================

Core base class for all smart tools with built-in performance tracking,
optimization triggers, thread-safe execution, and global tool registry.

Author: jetgause
Created: 2025-12-09
"""

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class OptimizationTrigger(Enum):
    """Enumeration of optimization trigger types."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_THRESHOLD = "error_threshold"
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CUSTOM = "custom"


@dataclass
class PerformanceMetrics:
    """
    Tracks performance metrics for tool execution.
    Thread-safe implementation for concurrent access.
    """
    tool_id: str
    tool_name: str
    
    # Execution metrics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    
    # Timing metrics (in seconds)
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    
    # Error tracking
    error_count: int = 0
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_error: Optional[str] = None
    last_error_timestamp: Optional[datetime] = None
    
    # Optimization tracking
    optimization_count: int = 0
    last_optimization: Optional[datetime] = None
    optimization_triggers: List[Tuple[datetime, OptimizationTrigger, str]] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_execution: Optional[datetime] = None
    
    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    
    def record_execution(self, execution_time: float, success: bool, error: Optional[Exception] = None):
        """Record execution metrics in a thread-safe manner."""
        with self._lock:
            self.total_executions += 1
            self.last_execution = datetime.utcnow()
            
            if success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            
            # Update timing metrics
            self.total_execution_time += execution_time
            self.min_execution_time = min(self.min_execution_time, execution_time)
            self.max_execution_time = max(self.max_execution_time, execution_time)
            self.avg_execution_time = self.total_execution_time / self.total_executions
            
            # Record error if present
            if error:
                self.error_count += 1
                error_type = type(error).__name__
                self.error_types[error_type] += 1
                self.last_error = str(error)
                self.last_error_timestamp = datetime.utcnow()
    
    def record_optimization(self, trigger: OptimizationTrigger, reason: str):
        """Record an optimization event."""
        with self._lock:
            self.optimization_count += 1
            self.last_optimization = datetime.utcnow()
            self.optimization_triggers.append((datetime.utcnow(), trigger, reason))
    
    def get_success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        with self._lock:
            if self.total_executions == 0:
                return 0.0
            return (self.successful_executions / self.total_executions) * 100
    
    def get_error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        with self._lock:
            if self.total_executions == 0:
                return 0.0
            return (self.failed_executions / self.total_executions) * 100
    
    def should_optimize(self, 
                       error_threshold: float = 10.0,
                       avg_time_threshold: float = 5.0,
                       min_executions: int = 10) -> Tuple[bool, Optional[OptimizationTrigger], Optional[str]]:
        """
        Determine if optimization should be triggered based on metrics.
        
        Returns:
            Tuple of (should_optimize, trigger_type, reason)
        """
        with self._lock:
            if self.total_executions < min_executions:
                return False, None, None
            
            # Check error rate threshold
            error_rate = self.get_error_rate()
            if error_rate >= error_threshold:
                return (True, 
                       OptimizationTrigger.ERROR_THRESHOLD,
                       f"Error rate {error_rate:.2f}% exceeds threshold {error_threshold}%")
            
            # Check average execution time
            if self.avg_execution_time >= avg_time_threshold:
                return (True,
                       OptimizationTrigger.EXECUTION_TIME,
                       f"Average execution time {self.avg_execution_time:.2f}s exceeds threshold {avg_time_threshold}s")
            
            # Check for performance degradation (comparing recent vs historical)
            if self.total_executions >= 50:
                recent_threshold = self.avg_execution_time * 1.5
                if self.max_execution_time >= recent_threshold:
                    return (True,
                           OptimizationTrigger.PERFORMANCE_DEGRADATION,
                           f"Max execution time {self.max_execution_time:.2f}s indicates degradation")
            
            return False, None, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        with self._lock:
            return {
                "tool_id": self.tool_id,
                "tool_name": self.tool_name,
                "total_executions": self.total_executions,
                "successful_executions": self.successful_executions,
                "failed_executions": self.failed_executions,
                "success_rate": self.get_success_rate(),
                "error_rate": self.get_error_rate(),
                "timing": {
                    "total_time": self.total_execution_time,
                    "avg_time": self.avg_execution_time,
                    "min_time": self.min_execution_time if self.min_execution_time != float('inf') else 0.0,
                    "max_time": self.max_execution_time,
                },
                "errors": {
                    "count": self.error_count,
                    "types": dict(self.error_types),
                    "last_error": self.last_error,
                    "last_error_timestamp": self.last_error_timestamp.isoformat() if self.last_error_timestamp else None,
                },
                "optimizations": {
                    "count": self.optimization_count,
                    "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
                },
                "timestamps": {
                    "created_at": self.created_at.isoformat(),
                    "last_execution": self.last_execution.isoformat() if self.last_execution else None,
                }
            }


class ToolRegistry:
    """
    Global registry for managing all tool instances.
    Thread-safe singleton implementation.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._tools: Dict[str, 'SmartToolBase'] = {}
        self._tools_by_name: Dict[str, Set[str]] = defaultdict(set)
        self._registry_lock = threading.RLock()
        self._initialized = True
    
    def register(self, tool: 'SmartToolBase') -> None:
        """Register a tool instance."""
        with self._registry_lock:
            self._tools[tool.tool_id] = tool
            self._tools_by_name[tool.name].add(tool.tool_id)
    
    def unregister(self, tool_id: str) -> bool:
        """Unregister a tool instance."""
        with self._registry_lock:
            if tool_id in self._tools:
                tool = self._tools[tool_id]
                self._tools_by_name[tool.name].discard(tool_id)
                if not self._tools_by_name[tool.name]:
                    del self._tools_by_name[tool.name]
                del self._tools[tool_id]
                return True
            return False
    
    def get_tool(self, tool_id: str) -> Optional['SmartToolBase']:
        """Get a tool by its ID."""
        with self._registry_lock:
            return self._tools.get(tool_id)
    
    def get_tools_by_name(self, name: str) -> List['SmartToolBase']:
        """Get all tools with a specific name."""
        with self._registry_lock:
            tool_ids = self._tools_by_name.get(name, set())
            return [self._tools[tid] for tid in tool_ids if tid in self._tools]
    
    def get_all_tools(self) -> List['SmartToolBase']:
        """Get all registered tools."""
        with self._registry_lock:
            return list(self._tools.values())
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered tools."""
        with self._registry_lock:
            return {
                tool_id: tool.get_metrics().to_dict()
                for tool_id, tool in self._tools.items()
            }
    
    def clear(self) -> None:
        """Clear all registered tools (use with caution)."""
        with self._registry_lock:
            self._tools.clear()
            self._tools_by_name.clear()
    
    def __len__(self) -> int:
        """Return the number of registered tools."""
        with self._registry_lock:
            return len(self._tools)
    
    def __contains__(self, tool_id: str) -> bool:
        """Check if a tool is registered."""
        with self._registry_lock:
            return tool_id in self._tools


class SmartToolBase(ABC):
    """
    Abstract base class for all smart tools.
    
    Features:
    - Automatic performance tracking
    - Thread-safe execution
    - Optimization trigger detection
    - Global tool registry integration
    - Error handling and recovery
    """
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 auto_register: bool = True,
                 optimization_config: Optional[Dict[str, Any]] = None):
        """
        Initialize SmartTool base.
        
        Args:
            name: Tool name
            description: Tool description
            auto_register: Whether to automatically register with ToolRegistry
            optimization_config: Configuration for optimization triggers
        """
        self.tool_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        
        # Performance tracking
        self.metrics = PerformanceMetrics(
            tool_id=self.tool_id,
            tool_name=self.name
        )
        
        # Thread safety
        self._execution_lock = threading.RLock()
        
        # Optimization configuration
        self.optimization_config = optimization_config or {
            "error_threshold": 10.0,
            "avg_time_threshold": 5.0,
            "min_executions": 10,
            "auto_optimize": True
        }
        
        # State management
        self._initialized = False
        self._enabled = True
        
        # Registry
        self._registry = ToolRegistry()
        if auto_register:
            self._registry.register(self)
    
    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        """
        Core execution logic to be implemented by subclasses.
        This method should contain the actual tool functionality.
        """
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Thread-safe execution wrapper with performance tracking.
        
        This method wraps the core _execute method with:
        - Thread safety
        - Performance metrics tracking
        - Error handling
        - Optimization trigger detection
        """
        if not self._enabled:
            raise RuntimeError(f"Tool {self.name} is disabled")
        
        with self._execution_lock:
            start_time = time.time()
            success = False
            error = None
            result = None
            
            try:
                result = self._execute(*args, **kwargs)
                success = True
                return result
            
            except Exception as e:
                success = False
                error = e
                raise
            
            finally:
                # Record metrics
                execution_time = time.time() - start_time
                self.metrics.record_execution(execution_time, success, error)
                
                # Check for optimization triggers
                if self.optimization_config.get("auto_optimize", True):
                    should_opt, trigger, reason = self.metrics.should_optimize(
                        error_threshold=self.optimization_config.get("error_threshold", 10.0),
                        avg_time_threshold=self.optimization_config.get("avg_time_threshold", 5.0),
                        min_executions=self.optimization_config.get("min_executions", 10)
                    )
                    
                    if should_opt:
                        self._trigger_optimization(trigger, reason)
    
    def _trigger_optimization(self, trigger: OptimizationTrigger, reason: str) -> None:
        """
        Handle optimization trigger.
        Can be overridden by subclasses for custom optimization logic.
        """
        self.metrics.record_optimization(trigger, reason)
        self.on_optimization_triggered(trigger, reason)
    
    def on_optimization_triggered(self, trigger: OptimizationTrigger, reason: str) -> None:
        """
        Hook for subclasses to implement custom optimization logic.
        Called when optimization is triggered.
        """
        pass
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self._execution_lock:
            self.metrics = PerformanceMetrics(
                tool_id=self.tool_id,
                tool_name=self.name
            )
    
    def enable(self) -> None:
        """Enable the tool for execution."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable the tool from execution."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if the tool is enabled."""
        return self._enabled
    
    def unregister(self) -> bool:
        """Unregister this tool from the global registry."""
        return self._registry.unregister(self.tool_id)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.tool_id[:8]}, name={self.name}, enabled={self._enabled})>"
    
    def __str__(self) -> str:
        return f"{self.name} (ID: {self.tool_id[:8]})"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed."""
        return False


# Global registry instance
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry


def get_tool(tool_id: str) -> Optional[SmartToolBase]:
    """Convenience function to get a tool from the global registry."""
    return _global_registry.get_tool(tool_id)


def get_all_tools() -> List[SmartToolBase]:
    """Convenience function to get all tools from the global registry."""
    return _global_registry.get_all_tools()


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """Convenience function to get all metrics from the global registry."""
    return _global_registry.get_all_metrics()
