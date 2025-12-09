"""
Resource Manager - Manages computational resources and optimization
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import time


class ResourceManager:
    """Manages resource allocation, tracking, and optimization for tools"""
    
    def __init__(self, max_memory_mb: int = 1024, max_cpu_percent: float = 80.0):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.resource_usage: Dict[str, List[Dict[str, Any]]] = {}
        self.allocations: Dict[str, Dict[str, Any]] = {}
    
    def allocate_resources(self, tool_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate resources for a tool
        
        Requirements format:
        {
            "memory_mb": 256,
            "cpu_percent": 20.0,
            "duration_seconds": 60
        }
        """
        memory_needed = requirements.get("memory_mb", 100)
        cpu_needed = requirements.get("cpu_percent", 10.0)
        
        # Check availability
        available_memory = self._get_available_memory()
        available_cpu = self._get_available_cpu()
        
        if memory_needed > available_memory:
            return {
                "success": False,
                "reason": f"Insufficient memory. Need: {memory_needed}MB, Available: {available_memory}MB"
            }
        
        if cpu_needed > available_cpu:
            return {
                "success": False,
                "reason": f"Insufficient CPU. Need: {cpu_needed}%, Available: {available_cpu}%"
            }
        
        # Allocate
        allocation = {
            "tool_id": tool_id,
            "memory_mb": memory_needed,
            "cpu_percent": cpu_needed,
            "allocated_at": datetime.utcnow().isoformat(),
            "duration_seconds": requirements.get("duration_seconds", 60),
            "status": "active"
        }
        
        self.allocations[tool_id] = allocation
        
        return {
            "success": True,
            "allocation_id": tool_id,
            "allocation": allocation
        }
    
    def release_resources(self, tool_id: str) -> bool:
        """Release allocated resources"""
        if tool_id in self.allocations:
            self.allocations[tool_id]["status"] = "released"
            self.allocations[tool_id]["released_at"] = datetime.utcnow().isoformat()
            return True
        return False
    
    def track_usage(self, tool_id: str, usage: Dict[str, Any]):
        """
        Track actual resource usage
        
        Usage format:
        {
            "memory_used_mb": 180,
            "cpu_used_percent": 15.5,
            "execution_time_seconds": 2.3,
            "timestamp": "2025-12-09T12:00:00"
        }
        """
        if tool_id not in self.resource_usage:
            self.resource_usage[tool_id] = []
        
        usage["timestamp"] = usage.get("timestamp", datetime.utcnow().isoformat())
        self.resource_usage[tool_id].append(usage)
    
    def get_usage_stats(self, tool_id: str) -> Dict[str, Any]:
        """Get resource usage statistics for a tool"""
        usage_history = self.resource_usage.get(tool_id, [])
        
        if not usage_history:
            return {
                "tool_id": tool_id,
                "executions": 0,
                "avg_memory_mb": 0,
                "avg_cpu_percent": 0,
                "avg_execution_time": 0
            }
        
        total_memory = sum(u.get("memory_used_mb", 0) for u in usage_history)
        total_cpu = sum(u.get("cpu_used_percent", 0) for u in usage_history)
        total_time = sum(u.get("execution_time_seconds", 0) for u in usage_history)
        count = len(usage_history)
        
        return {
            "tool_id": tool_id,
            "executions": count,
            "avg_memory_mb": round(total_memory / count, 2),
            "avg_cpu_percent": round(total_cpu / count, 2),
            "avg_execution_time": round(total_time / count, 2),
            "total_memory_mb": total_memory,
            "total_cpu_percent": total_cpu,
            "total_time_seconds": round(total_time, 2)
        }
    
    def _get_available_memory(self) -> int:
        """Calculate available memory"""
        allocated = sum(
            a["memory_mb"] for a in self.allocations.values()
            if a.get("status") == "active"
        )
        return self.max_memory_mb - allocated
    
    def _get_available_cpu(self) -> float:
        """Calculate available CPU"""
        allocated = sum(
            a["cpu_percent"] for a in self.allocations.values()
            if a.get("status") == "active"
        )
        return self.max_cpu_percent - allocated
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system resource status"""
        active_allocations = [
            a for a in self.allocations.values()
            if a.get("status") == "active"
        ]
        
        return {
            "max_memory_mb": self.max_memory_mb,
            "available_memory_mb": self._get_available_memory(),
            "max_cpu_percent": self.max_cpu_percent,
            "available_cpu_percent": self._get_available_cpu(),
            "active_allocations": len(active_allocations),
            "total_allocations": len(self.allocations)
        }
    
    def optimize_allocation(self, tool_id: str) -> Dict[str, Any]:
        """
        Suggest optimized resource allocation based on usage history
        """
        stats = self.get_usage_stats(tool_id)
        
        if stats["executions"] == 0:
            return {
                "tool_id": tool_id,
                "recommendation": "No usage history available"
            }
        
        # Add 20% buffer to average usage
        recommended_memory = int(stats["avg_memory_mb"] * 1.2)
        recommended_cpu = round(stats["avg_cpu_percent"] * 1.2, 1)
        
        current_allocation = self.allocations.get(tool_id, {})
        current_memory = current_allocation.get("memory_mb", 0)
        current_cpu = current_allocation.get("cpu_percent", 0)
        
        return {
            "tool_id": tool_id,
            "current_allocation": {
                "memory_mb": current_memory,
                "cpu_percent": current_cpu
            },
            "recommended_allocation": {
                "memory_mb": recommended_memory,
                "cpu_percent": recommended_cpu
            },
            "savings": {
                "memory_mb": current_memory - recommended_memory,
                "cpu_percent": round(current_cpu - recommended_cpu, 1)
            }
        }
    
    def get_top_consumers(self, metric: str = "memory", limit: int = 5) -> List[Dict[str, Any]]:
        """Get top resource-consuming tools"""
        stats = []
        for tool_id in self.resource_usage.keys():
            stat = self.get_usage_stats(tool_id)
            stats.append(stat)
        
        sort_key = f"avg_{metric}_mb" if metric == "memory" else "avg_cpu_percent"
        sorted_stats = sorted(stats, key=lambda s: s.get(sort_key, 0), reverse=True)
        
        return sorted_stats[:limit]
