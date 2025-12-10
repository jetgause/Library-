"""
Rental Manager
==============

Manages tool rentals, subscriptions, and billing for the Smart Marketplace.

Features:
- Monthly subscription management
- Tier-based access control
- Payment processing integration
- Subscription lifecycle management

Author: jetgause
Created: 2025-12-10
"""

import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import uuid

from .models import (
    RentalSubscription,
    SubscriptionTier,
    ToolListing,
)


class RentalManager:
    """
    Manages tool rentals and subscriptions in the marketplace.
    
    Handles subscription creation, renewal, cancellation, and access control
    based on subscription status and tier.
    """
    
    # Price multipliers by tier
    TIER_MULTIPLIERS = {
        SubscriptionTier.FREE: 0.0,
        SubscriptionTier.BASIC: 1.0,
        SubscriptionTier.PRO: 2.5,
        SubscriptionTier.ENTERPRISE: 5.0,
    }
    
    def __init__(self):
        """Initialize the RentalManager."""
        self._subscriptions: Dict[str, RentalSubscription] = {}
        self._user_subscriptions: Dict[str, List[str]] = {}  # user_id -> [subscription_ids]
        self._tool_subscriptions: Dict[str, List[str]] = {}  # tool_id -> [subscription_ids]
        self._lock = threading.RLock()
    
    def create_subscription(
        self,
        user_id: str,
        tool: ToolListing,
        tier: SubscriptionTier = SubscriptionTier.BASIC,
        months: int = 1,
        auto_renew: bool = True
    ) -> RentalSubscription:
        """
        Create a new rental subscription for a tool.
        
        Args:
            user_id: ID of the subscribing user
            tool: ToolListing to subscribe to
            tier: Subscription tier level
            months: Number of months for the subscription
            auto_renew: Whether to enable auto-renewal
            
        Returns:
            Created RentalSubscription
            
        Raises:
            ValueError: If user already has an active subscription for the tool
        """
        with self._lock:
            # Check for existing active subscription
            existing = self.get_active_subscription(user_id, tool.tool_id)
            if existing:
                raise ValueError(f"User {user_id} already has an active subscription for tool {tool.tool_id}")
            
            # Check tier requirements
            if self._tier_level(tier) < self._tier_level(tool.tier):
                raise ValueError(f"Tool requires {tool.tier.value} tier or higher")
            
            # Calculate price
            base_price = tool.monthly_price
            multiplier = self.TIER_MULTIPLIERS.get(tier, 1.0)
            monthly_price = base_price * multiplier if base_price > 0 else 0.0
            
            # Create subscription
            subscription = RentalSubscription(
                subscription_id=str(uuid.uuid4()),
                user_id=user_id,
                tool_id=tool.tool_id,
                tier=tier,
                monthly_price=monthly_price,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=30 * months),
                is_active=True,
                auto_renew=auto_renew,
                payment_history=[
                    {
                        "date": datetime.utcnow().isoformat(),
                        "amount": monthly_price * months,
                        "months": months,
                        "status": "completed"
                    }
                ]
            )
            
            # Store subscription
            self._subscriptions[subscription.subscription_id] = subscription
            
            # Update user index
            if user_id not in self._user_subscriptions:
                self._user_subscriptions[user_id] = []
            self._user_subscriptions[user_id].append(subscription.subscription_id)
            
            # Update tool index
            if tool.tool_id not in self._tool_subscriptions:
                self._tool_subscriptions[tool.tool_id] = []
            self._tool_subscriptions[tool.tool_id].append(subscription.subscription_id)
            
            return subscription
    
    def renew_subscription(
        self,
        subscription_id: str,
        months: int = 1
    ) -> RentalSubscription:
        """
        Renew an existing subscription.
        
        Args:
            subscription_id: ID of the subscription to renew
            months: Number of months to extend
            
        Returns:
            Updated RentalSubscription
            
        Raises:
            ValueError: If subscription not found
        """
        with self._lock:
            if subscription_id not in self._subscriptions:
                raise ValueError(f"Subscription {subscription_id} not found")
            
            subscription = self._subscriptions[subscription_id]
            
            # Calculate new end date
            current_end = subscription.end_date or datetime.utcnow()
            if current_end < datetime.utcnow():
                current_end = datetime.utcnow()
            
            subscription.end_date = current_end + timedelta(days=30 * months)
            subscription.is_active = True
            
            # Record payment
            subscription.payment_history.append({
                "date": datetime.utcnow().isoformat(),
                "amount": subscription.monthly_price * months,
                "months": months,
                "status": "completed"
            })
            
            return subscription
    
    def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False
    ) -> RentalSubscription:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: ID of the subscription to cancel
            immediate: If True, cancel immediately; otherwise, cancel at end of period
            
        Returns:
            Updated RentalSubscription
            
        Raises:
            ValueError: If subscription not found
        """
        with self._lock:
            if subscription_id not in self._subscriptions:
                raise ValueError(f"Subscription {subscription_id} not found")
            
            subscription = self._subscriptions[subscription_id]
            subscription.auto_renew = False
            
            if immediate:
                subscription.is_active = False
                subscription.end_date = datetime.utcnow()
            
            return subscription
    
    def get_subscription(self, subscription_id: str) -> Optional[RentalSubscription]:
        """Get a subscription by ID."""
        with self._lock:
            return self._subscriptions.get(subscription_id)
    
    def get_active_subscription(
        self,
        user_id: str,
        tool_id: str
    ) -> Optional[RentalSubscription]:
        """
        Get the active subscription for a user and tool.
        
        Args:
            user_id: User ID
            tool_id: Tool ID
            
        Returns:
            Active subscription or None
        """
        with self._lock:
            user_subs = self._user_subscriptions.get(user_id, [])
            
            for sub_id in user_subs:
                sub = self._subscriptions.get(sub_id)
                if sub and sub.tool_id == tool_id and sub.is_active and not sub.is_expired():
                    return sub
            
            return None
    
    def get_user_subscriptions(
        self,
        user_id: str,
        active_only: bool = True
    ) -> List[RentalSubscription]:
        """
        Get all subscriptions for a user.
        
        Args:
            user_id: User ID
            active_only: If True, return only active subscriptions
            
        Returns:
            List of subscriptions
        """
        with self._lock:
            sub_ids = self._user_subscriptions.get(user_id, [])
            subscriptions = []
            
            for sub_id in sub_ids:
                sub = self._subscriptions.get(sub_id)
                if sub:
                    if not active_only or (sub.is_active and not sub.is_expired()):
                        subscriptions.append(sub)
            
            return subscriptions
    
    def get_tool_subscriptions(
        self,
        tool_id: str,
        active_only: bool = True
    ) -> List[RentalSubscription]:
        """
        Get all subscriptions for a tool.
        
        Args:
            tool_id: Tool ID
            active_only: If True, return only active subscriptions
            
        Returns:
            List of subscriptions
        """
        with self._lock:
            sub_ids = self._tool_subscriptions.get(tool_id, [])
            subscriptions = []
            
            for sub_id in sub_ids:
                sub = self._subscriptions.get(sub_id)
                if sub:
                    if not active_only or (sub.is_active and not sub.is_expired()):
                        subscriptions.append(sub)
            
            return subscriptions
    
    def check_access(
        self,
        user_id: str,
        tool_id: str,
        required_tier: Optional[SubscriptionTier] = None
    ) -> Dict[str, Any]:
        """
        Check if a user has access to a tool.
        
        Args:
            user_id: User ID
            tool_id: Tool ID
            required_tier: Minimum tier required (optional)
            
        Returns:
            Access check result with details
        """
        with self._lock:
            subscription = self.get_active_subscription(user_id, tool_id)
            
            if not subscription:
                return {
                    "has_access": False,
                    "reason": "no_subscription",
                    "message": "No active subscription found"
                }
            
            if subscription.is_expired():
                return {
                    "has_access": False,
                    "reason": "expired",
                    "message": "Subscription has expired"
                }
            
            if required_tier and self._tier_level(subscription.tier) < self._tier_level(required_tier):
                return {
                    "has_access": False,
                    "reason": "insufficient_tier",
                    "message": f"Requires {required_tier.value} tier or higher",
                    "current_tier": subscription.tier.value,
                    "required_tier": required_tier.value
                }
            
            return {
                "has_access": True,
                "subscription_id": subscription.subscription_id,
                "tier": subscription.tier.value,
                "days_remaining": subscription.days_remaining()
            }
    
    def process_renewals(self) -> List[Dict[str, Any]]:
        """
        Process automatic renewals for expiring subscriptions.
        
        Returns:
            List of renewal results
        """
        results = []
        
        with self._lock:
            for sub_id, subscription in self._subscriptions.items():
                if not subscription.auto_renew:
                    continue
                
                if not subscription.is_active:
                    continue
                
                # Check if subscription expires within 24 hours
                if subscription.end_date:
                    time_remaining = subscription.end_date - datetime.utcnow()
                    if time_remaining <= timedelta(hours=24):
                        try:
                            self.renew_subscription(sub_id, months=1)
                            results.append({
                                "subscription_id": sub_id,
                                "status": "renewed",
                                "new_end_date": subscription.end_date.isoformat()
                            })
                        except Exception as e:
                            results.append({
                                "subscription_id": sub_id,
                                "status": "failed",
                                "error": str(e)
                            })
        
        return results
    
    def get_revenue_stats(self, tool_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get revenue statistics.
        
        Args:
            tool_id: Optional tool ID to filter by
            
        Returns:
            Revenue statistics
        """
        with self._lock:
            total_revenue = 0.0
            active_subscriptions = 0
            monthly_recurring = 0.0
            
            for subscription in self._subscriptions.values():
                if tool_id and subscription.tool_id != tool_id:
                    continue
                
                # Sum payment history
                for payment in subscription.payment_history:
                    if payment.get("status") == "completed":
                        total_revenue += payment.get("amount", 0.0)
                
                # Count active
                if subscription.is_active and not subscription.is_expired():
                    active_subscriptions += 1
                    monthly_recurring += subscription.monthly_price
            
            return {
                "total_revenue": total_revenue,
                "active_subscriptions": active_subscriptions,
                "monthly_recurring_revenue": monthly_recurring,
                "tool_id": tool_id
            }
    
    @staticmethod
    def _tier_level(tier: SubscriptionTier) -> int:
        """Get numeric level for tier comparison."""
        levels = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.BASIC: 1,
            SubscriptionTier.PRO: 2,
            SubscriptionTier.ENTERPRISE: 3,
        }
        return levels.get(tier, 0)
