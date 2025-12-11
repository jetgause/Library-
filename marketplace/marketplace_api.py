"""
Marketplace API
===============

FastAPI endpoints for the Smart Marketplace system.

Provides REST API for:
- Tool listing and discovery
- Rental subscription management
- Session-based optimization
- Creator training tools

Author: jetgause
Created: 2025-12-10
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from .models import (
    OptimizationRecord,
    OptimizationScope,
    OptimizationStandard,
    OptimizationStatus,
    RentalSubscription,
    SubscriptionTier,
    ToolListing,
    TrainingConfig,
    TrainingMethod,
)
from .optimization_standards import OptimizationStandardsEngine
from .rental_manager import RentalManager
from .session_optimizer import SessionOptimizer
from .training_engines import TrainingEngineManager


class MarketplaceAPI:
    """
    API handler for the Smart Marketplace.
    
    Provides a unified interface for all marketplace operations including
    tool management, rentals, optimization, and training.
    """
    
    def __init__(self):
        """Initialize the Marketplace API."""
        self._tools: Dict[str, ToolListing] = {}
        self._rental_manager = RentalManager()
        self._standards_engine = OptimizationStandardsEngine()
        self._session_optimizer = SessionOptimizer(self._standards_engine)
        self._training_manager = TrainingEngineManager()
    
    # ==================== Tool Management ====================
    
    def create_tool(
        self,
        creator_id: str,
        name: str,
        description: str,
        category: str,
        monthly_price: float,
        tier: str = "basic",
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new tool listing in the marketplace.
        
        Args:
            creator_id: ID of the tool creator
            name: Tool name
            description: Tool description
            category: Tool category
            monthly_price: Monthly rental price
            tier: Required subscription tier
            parameters: Tool parameters
            metadata: Additional metadata
            
        Returns:
            Created tool data
        """
        tool = ToolListing(
            tool_id=str(uuid.uuid4()),
            creator_id=creator_id,
            name=name,
            description=description,
            category=category,
            monthly_price=monthly_price,
            tier=SubscriptionTier(tier),
            parameters=parameters or {},
            metadata=metadata or {},
            performance_metrics={
                "success_rate": 0.75,
                "avg_execution_time": 1.0,
                "accuracy": 0.80,
            },
        )
        
        self._tools[tool.tool_id] = tool
        
        return {
            "success": True,
            "tool": tool.to_dict()
        }
    
    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get a tool by ID."""
        tool = self._tools.get(tool_id)
        return tool.to_dict() if tool else None
    
    def list_tools(
        self,
        category: Optional[str] = None,
        tier: Optional[str] = None,
        max_price: Optional[float] = None,
        creator_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List tools with optional filters.
        
        Args:
            category: Filter by category
            tier: Filter by tier
            max_price: Maximum monthly price
            creator_id: Filter by creator
            
        Returns:
            List of tool data
        """
        tools = list(self._tools.values())
        
        # Apply filters
        if category:
            tools = [t for t in tools if t.category == category]
        
        if tier:
            tools = [t for t in tools if t.tier.value == tier]
        
        if max_price is not None:
            tools = [t for t in tools if t.monthly_price <= max_price]
        
        if creator_id:
            tools = [t for t in tools if t.creator_id == creator_id]
        
        # Only active tools
        tools = [t for t in tools if t.is_active]
        
        return [t.to_dict() for t in tools]
    
    def update_tool(
        self,
        tool_id: str,
        creator_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a tool listing.
        
        Args:
            tool_id: Tool ID
            creator_id: Creator ID (for authorization)
            updates: Fields to update
            
        Returns:
            Update result
        """
        tool = self._tools.get(tool_id)
        
        if not tool:
            return {"success": False, "error": "Tool not found"}
        
        if tool.creator_id != creator_id:
            return {"success": False, "error": "Not authorized to update this tool"}
        
        # Apply allowed updates
        allowed_fields = ["name", "description", "monthly_price", "parameters", "metadata", "is_active"]
        
        for field in allowed_fields:
            if field in updates:
                setattr(tool, field, updates[field])
        
        tool.updated_at = datetime.utcnow()
        
        return {
            "success": True,
            "tool": tool.to_dict()
        }
    
    def delete_tool(self, tool_id: str, creator_id: str) -> Dict[str, Any]:
        """
        Delete (deactivate) a tool.
        
        Args:
            tool_id: Tool ID
            creator_id: Creator ID (for authorization)
            
        Returns:
            Deletion result
        """
        tool = self._tools.get(tool_id)
        
        if not tool:
            return {"success": False, "error": "Tool not found"}
        
        if tool.creator_id != creator_id:
            return {"success": False, "error": "Not authorized to delete this tool"}
        
        # Soft delete
        tool.is_active = False
        tool.updated_at = datetime.utcnow()
        
        return {"success": True, "message": "Tool deactivated"}
    
    # ==================== Rental Management ====================
    
    def subscribe_to_tool(
        self,
        user_id: str,
        tool_id: str,
        tier: str = "basic",
        months: int = 1,
        auto_renew: bool = True
    ) -> Dict[str, Any]:
        """
        Subscribe to a tool for monthly rental.
        
        Args:
            user_id: Subscribing user ID
            tool_id: Tool to subscribe to
            tier: Subscription tier
            months: Number of months
            auto_renew: Enable auto-renewal
            
        Returns:
            Subscription result
        """
        tool = self._tools.get(tool_id)
        
        if not tool:
            return {"success": False, "error": "Tool not found"}
        
        if not tool.is_active:
            return {"success": False, "error": "Tool is not available"}
        
        try:
            subscription = self._rental_manager.create_subscription(
                user_id=user_id,
                tool=tool,
                tier=SubscriptionTier(tier),
                months=months,
                auto_renew=auto_renew
            )
            
            return {
                "success": True,
                "subscription": subscription.to_dict()
            }
            
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    def cancel_subscription(
        self,
        subscription_id: str,
        user_id: str,
        immediate: bool = False
    ) -> Dict[str, Any]:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: Subscription to cancel
            user_id: User ID (for authorization)
            immediate: Cancel immediately
            
        Returns:
            Cancellation result
        """
        subscription = self._rental_manager.get_subscription(subscription_id)
        
        if not subscription:
            return {"success": False, "error": "Subscription not found"}
        
        if subscription.user_id != user_id:
            return {"success": False, "error": "Not authorized"}
        
        try:
            updated = self._rental_manager.cancel_subscription(subscription_id, immediate)
            return {
                "success": True,
                "subscription": updated.to_dict()
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    def get_user_subscriptions(
        self,
        user_id: str,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all subscriptions for a user."""
        subscriptions = self._rental_manager.get_user_subscriptions(user_id, active_only)
        return [s.to_dict() for s in subscriptions]
    
    def check_tool_access(
        self,
        user_id: str,
        tool_id: str
    ) -> Dict[str, Any]:
        """
        Check if user has access to a tool.
        
        Args:
            user_id: User ID
            tool_id: Tool ID
            
        Returns:
            Access check result
        """
        tool = self._tools.get(tool_id)
        
        if not tool:
            return {"has_access": False, "reason": "tool_not_found"}
        
        return self._rental_manager.check_access(user_id, tool_id, tool.tier)
    
    # ==================== Session Optimization ====================
    
    def create_session(
        self,
        user_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new optimization session.
        
        Args:
            user_id: User creating the session
            config: Session configuration
            
        Returns:
            Session creation result
        """
        session_id = self._session_optimizer.create_session(user_id, config)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Session created. Insert tools to begin optimization."
        }
    
    def insert_tool_into_session(
        self,
        session_id: str,
        user_id: str,
        tool_id: str,
        auto_optimize: bool = True,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Insert a tool into a session with automatic optimization.
        
        This is the key endpoint where local optimizations are applied
        when users add tools to their session.
        
        Args:
            session_id: Session ID
            user_id: User ID
            tool_id: Tool to insert
            auto_optimize: Apply automatic optimizations
            optimization_config: Optimization configuration
            
        Returns:
            Insertion result with optimization details
        """
        # Check access
        access = self.check_tool_access(user_id, tool_id)
        if not access.get("has_access"):
            return {
                "success": False,
                "error": f"Access denied: {access.get('reason')}"
            }
        
        tool = self._tools.get(tool_id)
        if not tool:
            return {"success": False, "error": "Tool not found"}
        
        try:
            result = self._session_optimizer.insert_tool(
                session_id=session_id,
                tool=tool,
                auto_optimize=auto_optimize,
                optimization_config=optimization_config
            )
            
            return {
                "success": True,
                "result": result,
                "message": "Tool inserted into session" + (
                    " with optimizations applied" if result.get("auto_optimized") else ""
                )
            }
            
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    def apply_optimization(
        self,
        session_id: str,
        tool_id: str,
        method: str = "optimization_engine",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply a specific optimization to a tool in session.
        
        Args:
            session_id: Session ID
            tool_id: Tool ID
            method: Optimization method
            config: Optimization configuration
            
        Returns:
            Optimization result
        """
        tool = self._tools.get(tool_id)
        if not tool:
            return {"success": False, "error": "Tool not found"}
        
        try:
            result = self._session_optimizer.apply_local_optimization(
                session_id=session_id,
                tool=tool,
                config=config or {},
                method=TrainingMethod(method)
            )
            
            return {
                "success": True,
                "optimization": result
            }
            
        except (ValueError, KeyError) as e:
            return {"success": False, "error": str(e)}
    
    def promote_optimization(
        self,
        record_id: str,
        creator_id: str
    ) -> Dict[str, Any]:
        """
        Promote a local optimization to permanent status.
        
        Only optimizations that meet standards can be promoted.
        
        Args:
            record_id: Optimization record ID
            creator_id: Creator ID (for authorization)
            
        Returns:
            Promotion result
        """
        # Get optimization record
        optimizations = list(self._session_optimizer._optimizations.values())
        record = next((o for o in optimizations if o.record_id == record_id), None)
        
        if not record:
            return {"success": False, "error": "Optimization not found"}
        
        # Check authorization
        tool = self._tools.get(record.tool_id)
        if not tool or tool.creator_id != creator_id:
            return {"success": False, "error": "Not authorized"}
        
        result = self._session_optimizer.promote_optimization(record_id)
        
        # If promoted successfully, update the base tool
        if result.get("success"):
            # Apply optimized metrics to the tool
            tool.performance_metrics.update(record.after_metrics)
            tool.optimization_history.append(record_id)
            tool.updated_at = datetime.utcnow()
        
        return result
    
    def end_session(
        self,
        session_id: str,
        promote_successful: bool = False
    ) -> Dict[str, Any]:
        """
        End an optimization session.
        
        Args:
            session_id: Session ID
            promote_successful: Promote successful optimizations
            
        Returns:
            Session end summary
        """
        try:
            result = self._session_optimizer.end_session(session_id, promote_successful)
            return {
                "success": True,
                "summary": result
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # ==================== Creator Training ====================
    
    def train_tool(
        self,
        creator_id: str,
        tool_id: str,
        method: str = "optimization_engine",
        config: Optional[Dict[str, Any]] = None,
        training_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a tool using specified method.
        
        Creators can optimize their tools using:
        - Deep Learning
        - Q-Learning
        - Optimization Engines
        - Hybrid approaches
        
        Args:
            creator_id: Creator ID (for authorization)
            tool_id: Tool to train
            method: Training method
            config: Training configuration
            training_data: Optional training data
            
        Returns:
            Training result
        """
        tool = self._tools.get(tool_id)
        
        if not tool:
            return {"success": False, "error": "Tool not found"}
        
        if tool.creator_id != creator_id:
            return {"success": False, "error": "Not authorized to train this tool"}
        
        # Create training config
        training_config = TrainingConfig.from_dict(config) if config else TrainingConfig()
        
        try:
            result = self._training_manager.train_tool(
                tool=tool,
                method=TrainingMethod(method),
                config=training_config,
                training_data=training_data
            )
            
            # If training successful, evaluate against standards
            if result.success:
                # Create a temporary optimization record for evaluation
                temp_record = OptimizationRecord(
                    tool_id=tool_id,
                    before_metrics=tool.performance_metrics.copy(),
                    after_metrics=result.new_metrics,
                    improvement_percentage=result.improvement_percentage,
                    status=OptimizationStatus.COMPLETED,
                )
                
                meets_standards = self._standards_engine.evaluate(temp_record)
                
                return {
                    "success": True,
                    "training_result": result.to_dict(),
                    "meets_standards": meets_standards,
                    "message": (
                        "Training completed. Optimization meets standards for permanent application."
                        if meets_standards else
                        "Training completed. Optimization does not meet standards for permanent application."
                    )
                }
            
            return {
                "success": False,
                "error": result.error or "Training failed"
            }
            
        except (ValueError, KeyError) as e:
            return {"success": False, "error": str(e)}
    
    def apply_training_results(
        self,
        creator_id: str,
        tool_id: str,
        training_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply training results to permanently update a tool.
        
        Args:
            creator_id: Creator ID
            tool_id: Tool ID
            training_result: Training result to apply
            
        Returns:
            Application result
        """
        tool = self._tools.get(tool_id)
        
        if not tool:
            return {"success": False, "error": "Tool not found"}
        
        if tool.creator_id != creator_id:
            return {"success": False, "error": "Not authorized"}
        
        # Apply new metrics
        new_metrics = training_result.get("new_metrics", {})
        new_params = training_result.get("optimized_parameters", {})
        
        tool.performance_metrics.update(new_metrics)
        tool.parameters.update(new_params)
        tool.updated_at = datetime.utcnow()
        
        return {
            "success": True,
            "tool": tool.to_dict(),
            "message": "Training results applied to tool"
        }
    
    def get_training_recommendations(
        self,
        creator_id: str,
        tool_id: str
    ) -> Dict[str, Any]:
        """
        Get recommended training method for a tool.
        
        Args:
            creator_id: Creator ID
            tool_id: Tool ID
            
        Returns:
            Recommendations
        """
        tool = self._tools.get(tool_id)
        
        if not tool:
            return {"success": False, "error": "Tool not found"}
        
        if tool.creator_id != creator_id:
            return {"success": False, "error": "Not authorized"}
        
        recommended = self._training_manager.get_best_method_for_tool(tool)
        
        return {
            "success": True,
            "tool_id": tool_id,
            "recommended_method": recommended.value if recommended else None,
            "available_methods": [m.value for m in TrainingMethod],
            "training_stats": self._training_manager.get_training_stats()
        }
    
    # ==================== Statistics & Analytics ====================
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get overall marketplace statistics."""
        active_tools = sum(1 for t in self._tools.values() if t.is_active)
        total_tools = len(self._tools)
        
        return {
            "total_tools": total_tools,
            "active_tools": active_tools,
            "optimization_stats": self._session_optimizer.get_optimization_stats(),
            "training_stats": self._training_manager.get_training_stats(),
            "standards": self._standards_engine.get_standards_summary(),
        }
    
    def get_creator_stats(self, creator_id: str) -> Dict[str, Any]:
        """Get statistics for a specific creator."""
        creator_tools = [t for t in self._tools.values() if t.creator_id == creator_id]
        
        return {
            "total_tools": len(creator_tools),
            "active_tools": sum(1 for t in creator_tools if t.is_active),
            "tools": [t.to_dict() for t in creator_tools],
            "revenue": self._rental_manager.get_revenue_stats(),
        }
    
    def get_tool_analytics(self, tool_id: str) -> Dict[str, Any]:
        """Get analytics for a specific tool."""
        tool = self._tools.get(tool_id)
        
        if not tool:
            return {"error": "Tool not found"}
        
        subscriptions = self._rental_manager.get_tool_subscriptions(tool_id)
        optimization_stats = self._session_optimizer.get_optimization_stats(tool_id)
        
        return {
            "tool": tool.to_dict(),
            "active_subscriptions": len([s for s in subscriptions if s.is_active]),
            "total_subscriptions": len(subscriptions),
            "optimization_stats": optimization_stats,
            "revenue": self._rental_manager.get_revenue_stats(tool_id),
        }
