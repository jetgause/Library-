"""
Marketplace Data Models
=======================

Core data models for the Smart Marketplace system including:
- Tool listings and metadata
- Rental subscriptions
- Optimization records
- Training configurations

Author: jetgause
Created: 2025-12-10
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class SubscriptionTier(Enum):
    """Subscription tier levels for tool rental."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class OptimizationStatus(Enum):
    """Status of an optimization attempt."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class TrainingMethod(Enum):
    """Available training methods for tool optimization."""
    DEEP_LEARNING = "deep_learning"
    Q_LEARNING = "q_learning"
    OPTIMIZATION_ENGINE = "optimization_engine"
    HYBRID = "hybrid"


class OptimizationScope(Enum):
    """Scope of optimization - local (session) or permanent."""
    LOCAL = "local"
    PERMANENT = "permanent"


@dataclass
class ToolListing:
    """
    Represents a tool listed in the marketplace.
    
    Attributes:
        tool_id: Unique identifier for the tool
        creator_id: ID of the user who created the tool
        name: Display name of the tool
        description: Detailed description of tool functionality
        category: Tool category (e.g., 'analysis', 'trading', 'optimization')
        version: Current version string
        monthly_price: Monthly rental price in USD
        tier: Required subscription tier to access
        parameters: Default tool parameters
        metadata: Additional tool metadata
        performance_metrics: Tracked performance statistics
        optimization_history: List of optimization record IDs
        is_active: Whether the tool is available for rental
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
    """
    tool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creator_id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    version: str = "1.0.0"
    monthly_price: float = 0.0
    tier: SubscriptionTier = SubscriptionTier.FREE
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    optimization_history: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert listing to dictionary format."""
        return {
            "tool_id": self.tool_id,
            "creator_id": self.creator_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "monthly_price": self.monthly_price,
            "tier": self.tier.value,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "optimization_history": self.optimization_history,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolListing":
        """Create a ToolListing from dictionary data."""
        return cls(
            tool_id=data.get("tool_id", str(uuid.uuid4())),
            creator_id=data.get("creator_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            version=data.get("version", "1.0.0"),
            monthly_price=data.get("monthly_price", 0.0),
            tier=SubscriptionTier(data.get("tier", "free")),
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {}),
            performance_metrics=data.get("performance_metrics", {}),
            optimization_history=data.get("optimization_history", []),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )


@dataclass
class RentalSubscription:
    """
    Represents a rental subscription for a tool.
    
    Attributes:
        subscription_id: Unique subscription identifier
        user_id: ID of the subscribing user
        tool_id: ID of the rented tool
        tier: Subscription tier level
        monthly_price: Price paid per month
        start_date: When the subscription started
        end_date: When the subscription expires
        is_active: Whether the subscription is currently active
        auto_renew: Whether to auto-renew the subscription
        payment_history: List of payment records
    """
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tool_id: str = ""
    tier: SubscriptionTier = SubscriptionTier.BASIC
    monthly_price: float = 0.0
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    is_active: bool = True
    auto_renew: bool = True
    payment_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if the subscription has expired."""
        if self.end_date is None:
            return False
        return datetime.utcnow() > self.end_date
    
    def days_remaining(self) -> int:
        """Calculate days remaining in subscription."""
        if self.end_date is None:
            return -1  # Unlimited
        delta = self.end_date - datetime.utcnow()
        return max(0, delta.days)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert subscription to dictionary format."""
        return {
            "subscription_id": self.subscription_id,
            "user_id": self.user_id,
            "tool_id": self.tool_id,
            "tier": self.tier.value,
            "monthly_price": self.monthly_price,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_active": self.is_active,
            "auto_renew": self.auto_renew,
            "payment_history": self.payment_history,
            "is_expired": self.is_expired(),
            "days_remaining": self.days_remaining(),
        }


@dataclass
class OptimizationRecord:
    """
    Records an optimization attempt or result.
    
    Attributes:
        record_id: Unique record identifier
        tool_id: ID of the tool being optimized
        session_id: Session in which optimization occurred
        user_id: User who triggered the optimization
        scope: Whether optimization is local or permanent
        method: Training method used
        status: Current status of the optimization
        before_metrics: Performance metrics before optimization
        after_metrics: Performance metrics after optimization
        improvement_percentage: Calculated improvement
        meets_standards: Whether it meets permanent optimization standards
        applied_at: When the optimization was applied
        approved_at: When it was approved (if permanent)
        config: Training configuration used
    """
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str = ""
    session_id: str = ""
    user_id: str = ""
    scope: OptimizationScope = OptimizationScope.LOCAL
    method: TrainingMethod = TrainingMethod.OPTIMIZATION_ENGINE
    status: OptimizationStatus = OptimizationStatus.PENDING
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)
    improvement_percentage: float = 0.0
    meets_standards: bool = False
    applied_at: datetime = field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_improvement(self) -> float:
        """Calculate overall improvement percentage."""
        if not self.before_metrics or not self.after_metrics:
            return 0.0
        
        improvements = []
        for key in self.before_metrics:
            if key in self.after_metrics:
                before = self.before_metrics[key]
                after = self.after_metrics[key]
                if before > 0:
                    improvement = ((after - before) / before) * 100
                    improvements.append(improvement)
        
        if not improvements:
            return 0.0
        
        self.improvement_percentage = sum(improvements) / len(improvements)
        return self.improvement_percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary format."""
        return {
            "record_id": self.record_id,
            "tool_id": self.tool_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "scope": self.scope.value,
            "method": self.method.value,
            "status": self.status.value,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "improvement_percentage": self.improvement_percentage,
            "meets_standards": self.meets_standards,
            "applied_at": self.applied_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "config": self.config,
        }


@dataclass
class TrainingConfig:
    """
    Configuration for tool training/optimization.
    
    Attributes:
        config_id: Unique configuration identifier
        method: Training method to use
        epochs: Number of training epochs (for deep learning)
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        discount_factor: Discount factor (gamma) for Q-learning
        exploration_rate: Epsilon for exploration vs exploitation
        optimization_targets: Metrics to optimize
        constraints: Constraints on optimization
        hyperparameters: Additional hyperparameters
    """
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: TrainingMethod = TrainingMethod.OPTIMIZATION_ENGINE
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    optimization_targets: List[str] = field(default_factory=lambda: ["performance", "accuracy"])
    constraints: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            "config_id": self.config_id,
            "method": self.method.value,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "optimization_targets": self.optimization_targets,
            "constraints": self.constraints,
            "hyperparameters": self.hyperparameters,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create a TrainingConfig from dictionary data."""
        return cls(
            config_id=data.get("config_id", str(uuid.uuid4())),
            method=TrainingMethod(data.get("method", "optimization_engine")),
            epochs=data.get("epochs", 100),
            learning_rate=data.get("learning_rate", 0.001),
            batch_size=data.get("batch_size", 32),
            discount_factor=data.get("discount_factor", 0.99),
            exploration_rate=data.get("exploration_rate", 0.1),
            optimization_targets=data.get("optimization_targets", ["performance", "accuracy"]),
            constraints=data.get("constraints", {}),
            hyperparameters=data.get("hyperparameters", {}),
        )


@dataclass
class OptimizationStandard:
    """
    Defines standards that optimizations must meet for permanent approval.
    
    Attributes:
        standard_id: Unique standard identifier
        name: Human-readable name for the standard
        description: Detailed description of the standard
        min_improvement_threshold: Minimum improvement percentage required
        required_metrics: Metrics that must improve
        stability_threshold: Maximum variance allowed in results
        min_sample_size: Minimum number of test runs required
        regression_tolerance: Maximum allowed regression in any metric
        is_active: Whether this standard is currently enforced
    """
    standard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    min_improvement_threshold: float = 15.0  # Default 15% improvement required
    required_metrics: List[str] = field(default_factory=lambda: ["success_rate", "avg_execution_time"])
    stability_threshold: float = 5.0  # Max 5% variance
    min_sample_size: int = 100
    regression_tolerance: float = 2.0  # Max 2% regression allowed
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert standard to dictionary format."""
        return {
            "standard_id": self.standard_id,
            "name": self.name,
            "description": self.description,
            "min_improvement_threshold": self.min_improvement_threshold,
            "required_metrics": self.required_metrics,
            "stability_threshold": self.stability_threshold,
            "min_sample_size": self.min_sample_size,
            "regression_tolerance": self.regression_tolerance,
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationStandard":
        """Create an OptimizationStandard from dictionary data."""
        return cls(
            standard_id=data.get("standard_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            min_improvement_threshold=data.get("min_improvement_threshold", 15.0),
            required_metrics=data.get("required_metrics", ["success_rate", "avg_execution_time"]),
            stability_threshold=data.get("stability_threshold", 5.0),
            min_sample_size=data.get("min_sample_size", 100),
            regression_tolerance=data.get("regression_tolerance", 2.0),
            is_active=data.get("is_active", True),
        )
