"""
Smart Marketplace Module
=========================

A comprehensive marketplace system for user-created tools with:
- Monthly rental/subscription management
- Session-based optimization with automatic local improvements
- Optimization standards for permanent approval
- Creator training integration (Deep Learning, Q-Learning, Optimization Engines)

Author: jetgause
Created: 2025-12-10
"""

from .models import (
    ToolListing,
    RentalSubscription,
    OptimizationRecord,
    TrainingConfig,
    OptimizationStandard,
)
from .rental_manager import RentalManager
from .session_optimizer import SessionOptimizer
from .optimization_standards import OptimizationStandardsEngine
from .training_engines import TrainingEngineManager
from .marketplace_api import MarketplaceAPI

__all__ = [
    # Models
    "ToolListing",
    "RentalSubscription",
    "OptimizationRecord",
    "TrainingConfig",
    "OptimizationStandard",
    # Managers
    "RentalManager",
    "SessionOptimizer",
    "OptimizationStandardsEngine",
    "TrainingEngineManager",
    "MarketplaceAPI",
]

__version__ = "1.0.0"
