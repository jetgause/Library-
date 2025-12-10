"""
Session Optimizer
=================

Handles session-based optimizations for tools in the Smart Marketplace.

When a user inserts a tool into their session, the optimizer can:
1. Apply local optimizations specific to the session context
2. Track optimization performance
3. Promote successful local optimizations to permanent status

Author: jetgause
Created: 2025-12-10
"""

import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from .models import (
    OptimizationRecord,
    OptimizationScope,
    OptimizationStatus,
    ToolListing,
    TrainingMethod,
)


class SessionOptimizer:
    """
    Manages session-based optimizations for marketplace tools.
    
    Provides local optimization capabilities when tools are inserted into
    a session, with the ability to track and potentially promote
    successful optimizations to permanent status.
    """
    
    def __init__(self, standards_engine=None):
        """
        Initialize the SessionOptimizer.
        
        Args:
            standards_engine: OptimizationStandardsEngine for checking promotion eligibility
        """
        self._sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session data
        self._optimizations: Dict[str, OptimizationRecord] = {}  # record_id -> record
        self._session_optimizations: Dict[str, List[str]] = {}  # session_id -> [record_ids]
        self._lock = threading.RLock()
        self._standards_engine = standards_engine
        self._optimization_callbacks: Dict[str, Callable] = {}
    
    def create_session(
        self,
        user_id: str,
        session_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new optimization session.
        
        Args:
            user_id: ID of the user creating the session
            session_config: Optional session configuration
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "tools": {},  # tool_id -> tool state
                "optimizations_applied": [],
                "config": session_config or {},
                "is_active": True,
            }
            self._session_optimizations[session_id] = []
        
        return session_id
    
    def insert_tool(
        self,
        session_id: str,
        tool: ToolListing,
        auto_optimize: bool = True,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Insert a tool into a session with optional automatic optimization.
        
        This is the key entry point where local optimizations are applied
        when the user adds a tool to their session.
        
        Args:
            session_id: Active session ID
            tool: Tool to insert
            auto_optimize: Whether to apply automatic optimizations
            optimization_config: Configuration for optimization
            
        Returns:
            Insertion result with optimization details
        """
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
            
            if not session["is_active"]:
                raise ValueError(f"Session {session_id} is not active")
            
            # Store tool state in session
            tool_state = {
                "tool_id": tool.tool_id,
                "name": tool.name,
                "original_parameters": tool.parameters.copy(),
                "current_parameters": tool.parameters.copy(),
                "original_metrics": tool.performance_metrics.copy(),
                "current_metrics": tool.performance_metrics.copy(),
                "inserted_at": datetime.utcnow().isoformat(),
                "optimizations": [],
            }
            
            session["tools"][tool.tool_id] = tool_state
            
            result = {
                "session_id": session_id,
                "tool_id": tool.tool_id,
                "inserted": True,
                "auto_optimized": False,
                "optimizations": [],
            }
            
            # Apply automatic optimizations if enabled
            if auto_optimize:
                optimization_result = self.apply_local_optimization(
                    session_id=session_id,
                    tool=tool,
                    config=optimization_config or {}
                )
                
                result["auto_optimized"] = True
                result["optimizations"].append(optimization_result)
                tool_state["optimizations"].append(optimization_result["record_id"])
            
            return result
    
    def apply_local_optimization(
        self,
        session_id: str,
        tool: ToolListing,
        config: Dict[str, Any],
        method: TrainingMethod = TrainingMethod.OPTIMIZATION_ENGINE
    ) -> Dict[str, Any]:
        """
        Apply a local optimization to a tool within a session.
        
        Local optimizations are session-specific and don't affect the
        base tool until promoted to permanent status.
        
        Args:
            session_id: Session ID
            tool: Tool to optimize
            config: Optimization configuration
            method: Training method to use
            
        Returns:
            Optimization result
        """
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
            user_id = session["user_id"]
            
            # Create optimization record
            record = OptimizationRecord(
                record_id=str(uuid.uuid4()),
                tool_id=tool.tool_id,
                session_id=session_id,
                user_id=user_id,
                scope=OptimizationScope.LOCAL,
                method=method,
                status=OptimizationStatus.IN_PROGRESS,
                before_metrics=tool.performance_metrics.copy(),
                config=config,
            )
            
            # Store record
            self._optimizations[record.record_id] = record
            self._session_optimizations[session_id].append(record.record_id)
        
        # Perform optimization (outside lock to avoid blocking)
        try:
            optimized_params, new_metrics = self._run_optimization(
                tool=tool,
                method=method,
                config=config
            )
            
            with self._lock:
                # Update record with results
                record.status = OptimizationStatus.COMPLETED
                record.after_metrics = new_metrics
                record.calculate_improvement()
                
                # Update tool state in session
                if tool.tool_id in session["tools"]:
                    session["tools"][tool.tool_id]["current_parameters"] = optimized_params
                    session["tools"][tool.tool_id]["current_metrics"] = new_metrics
                
                # Check if optimization meets standards for permanent promotion
                if self._standards_engine:
                    meets_standards = self._standards_engine.evaluate(record)
                    record.meets_standards = meets_standards
                
                return {
                    "record_id": record.record_id,
                    "status": record.status.value,
                    "improvement_percentage": record.improvement_percentage,
                    "meets_standards": record.meets_standards,
                    "before_metrics": record.before_metrics,
                    "after_metrics": record.after_metrics,
                }
                
        except Exception as e:
            with self._lock:
                record.status = OptimizationStatus.FAILED
            
            return {
                "record_id": record.record_id,
                "status": OptimizationStatus.FAILED.value,
                "error": str(e),
            }
    
    def _run_optimization(
        self,
        tool: ToolListing,
        method: TrainingMethod,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute the actual optimization logic.
        
        This is a simulation of the optimization process. In production,
        this would integrate with actual training engines.
        
        Args:
            tool: Tool to optimize
            method: Training method
            config: Configuration
            
        Returns:
            Tuple of (optimized_parameters, new_metrics)
        """
        # Get optimization callback if registered
        callback = self._optimization_callbacks.get(method.value)
        
        if callback:
            return callback(tool, config)
        
        # Default optimization simulation
        optimized_params = tool.parameters.copy()
        new_metrics = tool.performance_metrics.copy()
        
        # Apply method-specific improvements (simulation)
        improvement_factor = self._get_improvement_factor(method, config)
        
        for key in new_metrics:
            if key in ["success_rate", "accuracy", "precision"]:
                # For rate metrics, improve towards 1.0
                current = new_metrics[key]
                improvement = (1.0 - current) * improvement_factor
                new_metrics[key] = min(1.0, current + improvement)
            elif key in ["avg_execution_time", "latency"]:
                # For time metrics, reduce
                new_metrics[key] = new_metrics[key] * (1.0 - improvement_factor * 0.5)
            else:
                # For other metrics, apply general improvement
                new_metrics[key] = new_metrics[key] * (1.0 + improvement_factor)
        
        return optimized_params, new_metrics
    
    def _get_improvement_factor(
        self,
        method: TrainingMethod,
        config: Dict[str, Any]
    ) -> float:
        """
        Calculate improvement factor based on method and config.
        
        Args:
            method: Training method
            config: Configuration
            
        Returns:
            Improvement factor (0.0 to 1.0)
        """
        base_factors = {
            TrainingMethod.DEEP_LEARNING: 0.15,
            TrainingMethod.Q_LEARNING: 0.12,
            TrainingMethod.OPTIMIZATION_ENGINE: 0.10,
            TrainingMethod.HYBRID: 0.18,
        }
        
        base = base_factors.get(method, 0.10)
        
        # Adjust based on config
        epochs = config.get("epochs", 100)
        epoch_bonus = min(epochs / 1000, 0.05)  # Up to 5% bonus for more epochs
        
        return base + epoch_bonus
    
    def register_optimization_callback(
        self,
        method: str,
        callback: Callable[[ToolListing, Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, float]]]
    ):
        """
        Register a custom optimization callback for a training method.
        
        Args:
            method: Training method name
            callback: Function that takes (tool, config) and returns (params, metrics)
        """
        self._optimization_callbacks[method] = callback
    
    def promote_optimization(
        self,
        record_id: str
    ) -> Dict[str, Any]:
        """
        Promote a local optimization to permanent status.
        
        This should only be called after the optimization has been
        verified to meet standards.
        
        Args:
            record_id: Optimization record ID
            
        Returns:
            Promotion result
        """
        with self._lock:
            if record_id not in self._optimizations:
                raise ValueError(f"Optimization record {record_id} not found")
            
            record = self._optimizations[record_id]
            
            if record.scope == OptimizationScope.PERMANENT:
                return {
                    "success": False,
                    "message": "Optimization is already permanent"
                }
            
            if record.status != OptimizationStatus.COMPLETED:
                return {
                    "success": False,
                    "message": f"Cannot promote optimization with status {record.status.value}"
                }
            
            if not record.meets_standards:
                return {
                    "success": False,
                    "message": "Optimization does not meet standards for permanent promotion"
                }
            
            # Promote to permanent
            record.scope = OptimizationScope.PERMANENT
            record.status = OptimizationStatus.APPROVED
            record.approved_at = datetime.utcnow()
            
            return {
                "success": True,
                "record_id": record_id,
                "tool_id": record.tool_id,
                "improvement_percentage": record.improvement_percentage,
                "approved_at": record.approved_at.isoformat()
            }
    
    def get_session_optimizations(
        self,
        session_id: str
    ) -> List[OptimizationRecord]:
        """
        Get all optimizations for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of optimization records
        """
        with self._lock:
            record_ids = self._session_optimizations.get(session_id, [])
            return [
                self._optimizations[rid]
                for rid in record_ids
                if rid in self._optimizations
            ]
    
    def get_tool_session_state(
        self,
        session_id: str,
        tool_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a tool in a session.
        
        Args:
            session_id: Session ID
            tool_id: Tool ID
            
        Returns:
            Tool state or None
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            return session["tools"].get(tool_id)
    
    def end_session(
        self,
        session_id: str,
        promote_successful: bool = False
    ) -> Dict[str, Any]:
        """
        End a session and optionally promote successful optimizations.
        
        Args:
            session_id: Session ID
            promote_successful: Whether to promote optimizations that meet standards
            
        Returns:
            Session end summary
        """
        with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self._sessions[session_id]
            session["is_active"] = False
            session["ended_at"] = datetime.utcnow().isoformat()
            
            promoted = []
            
            if promote_successful:
                for record_id in self._session_optimizations.get(session_id, []):
                    record = self._optimizations.get(record_id)
                    if record and record.meets_standards:
                        result = self.promote_optimization(record_id)
                        if result.get("success"):
                            promoted.append(record_id)
            
            return {
                "session_id": session_id,
                "ended_at": session["ended_at"],
                "tools_used": list(session["tools"].keys()),
                "optimizations_applied": len(self._session_optimizations.get(session_id, [])),
                "optimizations_promoted": promoted,
            }
    
    def get_optimization_stats(
        self,
        tool_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Args:
            tool_id: Optional tool ID to filter by
            
        Returns:
            Statistics dictionary
        """
        with self._lock:
            total = 0
            completed = 0
            promoted = 0
            avg_improvement = 0.0
            improvements = []
            
            for record in self._optimizations.values():
                if tool_id and record.tool_id != tool_id:
                    continue
                
                total += 1
                
                if record.status == OptimizationStatus.COMPLETED:
                    completed += 1
                    improvements.append(record.improvement_percentage)
                
                if record.status == OptimizationStatus.APPROVED:
                    promoted += 1
            
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
            
            return {
                "total_optimizations": total,
                "completed": completed,
                "promoted_to_permanent": promoted,
                "average_improvement_percentage": avg_improvement,
                "tool_id": tool_id,
            }
