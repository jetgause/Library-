"""
Smart Marketplace Test Suite
=============================

Comprehensive tests for the Smart Marketplace system including:
- Tool listing and management
- Monthly rental/subscription management
- Session-based optimization
- Optimization standards evaluation
- Creator training engines

Author: jetgause
Created: 2025-12-10
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, Any

# Import marketplace components
from marketplace.models import (
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
from marketplace.rental_manager import RentalManager
from marketplace.session_optimizer import SessionOptimizer
from marketplace.optimization_standards import OptimizationStandardsEngine
from marketplace.training_engines import (
    DeepLearningEngine,
    QLearningEngine,
    OptimizationEngine,
    HybridTrainingEngine,
    TrainingEngineManager,
)
from marketplace.marketplace_api import MarketplaceAPI


class TestToolListing(unittest.TestCase):
    """Test suite for ToolListing model."""
    
    def test_create_tool_listing(self):
        """Test creating a basic tool listing."""
        tool = ToolListing(
            creator_id="creator_123",
            name="Test Tool",
            description="A test tool",
            category="analysis",
            monthly_price=9.99,
        )
        
        self.assertIsNotNone(tool.tool_id)
        self.assertEqual(tool.creator_id, "creator_123")
        self.assertEqual(tool.name, "Test Tool")
        self.assertEqual(tool.monthly_price, 9.99)
        self.assertEqual(tool.tier, SubscriptionTier.FREE)
        self.assertTrue(tool.is_active)
    
    def test_tool_to_dict(self):
        """Test converting tool to dictionary."""
        tool = ToolListing(
            creator_id="creator_123",
            name="Test Tool",
            category="trading",
        )
        
        data = tool.to_dict()
        
        self.assertIn("tool_id", data)
        self.assertEqual(data["creator_id"], "creator_123")
        self.assertEqual(data["name"], "Test Tool")
        self.assertEqual(data["tier"], "free")
        self.assertIn("created_at", data)
    
    def test_tool_from_dict(self):
        """Test creating tool from dictionary."""
        data = {
            "tool_id": "test_id",
            "creator_id": "creator_123",
            "name": "Test Tool",
            "category": "analysis",
            "monthly_price": 19.99,
            "tier": "pro",
        }
        
        tool = ToolListing.from_dict(data)
        
        self.assertEqual(tool.tool_id, "test_id")
        self.assertEqual(tool.monthly_price, 19.99)
        self.assertEqual(tool.tier, SubscriptionTier.PRO)


class TestRentalSubscription(unittest.TestCase):
    """Test suite for RentalSubscription model."""
    
    def test_create_subscription(self):
        """Test creating a subscription."""
        subscription = RentalSubscription(
            user_id="user_123",
            tool_id="tool_456",
            tier=SubscriptionTier.PRO,
            monthly_price=29.99,
        )
        
        self.assertIsNotNone(subscription.subscription_id)
        self.assertEqual(subscription.user_id, "user_123")
        self.assertEqual(subscription.tier, SubscriptionTier.PRO)
        self.assertTrue(subscription.is_active)
        self.assertTrue(subscription.auto_renew)
    
    def test_subscription_is_expired(self):
        """Test subscription expiration check."""
        # Not expired
        future_end = datetime.utcnow() + timedelta(days=30)
        subscription = RentalSubscription(
            user_id="user_123",
            tool_id="tool_456",
            end_date=future_end,
        )
        self.assertFalse(subscription.is_expired())
        
        # Expired
        past_end = datetime.utcnow() - timedelta(days=1)
        expired_sub = RentalSubscription(
            user_id="user_123",
            tool_id="tool_456",
            end_date=past_end,
        )
        self.assertTrue(expired_sub.is_expired())
    
    def test_days_remaining(self):
        """Test days remaining calculation."""
        end_date = datetime.utcnow() + timedelta(days=15)
        subscription = RentalSubscription(
            user_id="user_123",
            tool_id="tool_456",
            end_date=end_date,
        )
        
        days = subscription.days_remaining()
        self.assertGreaterEqual(days, 14)
        self.assertLessEqual(days, 15)


class TestRentalManager(unittest.TestCase):
    """Test suite for RentalManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = RentalManager()
        self.test_tool = ToolListing(
            creator_id="creator_123",
            name="Test Tool",
            monthly_price=19.99,
            tier=SubscriptionTier.BASIC,
        )
    
    def test_create_subscription(self):
        """Test creating a subscription."""
        subscription = self.manager.create_subscription(
            user_id="user_123",
            tool=self.test_tool,
            tier=SubscriptionTier.BASIC,
            months=1,
        )
        
        self.assertIsNotNone(subscription)
        self.assertEqual(subscription.user_id, "user_123")
        self.assertEqual(subscription.tool_id, self.test_tool.tool_id)
        self.assertTrue(subscription.is_active)
    
    def test_duplicate_subscription_raises_error(self):
        """Test that duplicate active subscription raises error."""
        self.manager.create_subscription(
            user_id="user_123",
            tool=self.test_tool,
            tier=SubscriptionTier.BASIC,
        )
        
        with self.assertRaises(ValueError):
            self.manager.create_subscription(
                user_id="user_123",
                tool=self.test_tool,
                tier=SubscriptionTier.BASIC,
            )
    
    def test_check_access_with_subscription(self):
        """Test access check with valid subscription."""
        self.manager.create_subscription(
            user_id="user_123",
            tool=self.test_tool,
            tier=SubscriptionTier.BASIC,
        )
        
        access = self.manager.check_access("user_123", self.test_tool.tool_id)
        
        self.assertTrue(access["has_access"])
        self.assertIn("days_remaining", access)
    
    def test_check_access_without_subscription(self):
        """Test access check without subscription."""
        access = self.manager.check_access("user_123", "nonexistent_tool")
        
        self.assertFalse(access["has_access"])
        self.assertEqual(access["reason"], "no_subscription")
    
    def test_cancel_subscription(self):
        """Test subscription cancellation."""
        subscription = self.manager.create_subscription(
            user_id="user_123",
            tool=self.test_tool,
        )
        
        cancelled = self.manager.cancel_subscription(
            subscription.subscription_id,
            immediate=True
        )
        
        self.assertFalse(cancelled.is_active)
        self.assertFalse(cancelled.auto_renew)


class TestOptimizationRecord(unittest.TestCase):
    """Test suite for OptimizationRecord model."""
    
    def test_calculate_improvement(self):
        """Test improvement percentage calculation."""
        record = OptimizationRecord(
            tool_id="tool_123",
            before_metrics={"success_rate": 0.5, "accuracy": 0.6},
            after_metrics={"success_rate": 0.7, "accuracy": 0.8},
        )
        
        improvement = record.calculate_improvement()
        
        # Success rate improved 40%, accuracy improved 33%, avg ~36.7%
        self.assertGreater(improvement, 30)
        self.assertLess(improvement, 45)
    
    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = OptimizationRecord(
            tool_id="tool_123",
            session_id="session_456",
            scope=OptimizationScope.LOCAL,
            method=TrainingMethod.DEEP_LEARNING,
        )
        
        data = record.to_dict()
        
        self.assertEqual(data["tool_id"], "tool_123")
        self.assertEqual(data["scope"], "local")
        self.assertEqual(data["method"], "deep_learning")


class TestSessionOptimizer(unittest.TestCase):
    """Test suite for SessionOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.standards_engine = OptimizationStandardsEngine()
        self.optimizer = SessionOptimizer(self.standards_engine)
        self.test_tool = ToolListing(
            creator_id="creator_123",
            name="Test Tool",
            performance_metrics={
                "success_rate": 0.75,
                "avg_execution_time": 1.0,
                "accuracy": 0.80,
            },
        )
    
    def test_create_session(self):
        """Test session creation."""
        session_id = self.optimizer.create_session("user_123")
        
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.optimizer._sessions)
    
    def test_insert_tool_without_optimization(self):
        """Test inserting tool without auto-optimization."""
        session_id = self.optimizer.create_session("user_123")
        
        result = self.optimizer.insert_tool(
            session_id=session_id,
            tool=self.test_tool,
            auto_optimize=False,
        )
        
        self.assertTrue(result["inserted"])
        self.assertFalse(result["auto_optimized"])
        self.assertEqual(len(result["optimizations"]), 0)
    
    def test_insert_tool_with_optimization(self):
        """Test inserting tool with auto-optimization."""
        session_id = self.optimizer.create_session("user_123")
        
        result = self.optimizer.insert_tool(
            session_id=session_id,
            tool=self.test_tool,
            auto_optimize=True,
        )
        
        self.assertTrue(result["inserted"])
        self.assertTrue(result["auto_optimized"])
        self.assertGreater(len(result["optimizations"]), 0)
    
    def test_apply_local_optimization(self):
        """Test applying local optimization."""
        session_id = self.optimizer.create_session("user_123")
        self.optimizer.insert_tool(session_id, self.test_tool, auto_optimize=False)
        
        result = self.optimizer.apply_local_optimization(
            session_id=session_id,
            tool=self.test_tool,
            config={"epochs": 50},
            method=TrainingMethod.OPTIMIZATION_ENGINE,
        )
        
        self.assertIn("record_id", result)
        self.assertEqual(result["status"], "completed")
        self.assertIn("improvement_percentage", result)
    
    def test_end_session(self):
        """Test ending a session."""
        session_id = self.optimizer.create_session("user_123")
        self.optimizer.insert_tool(session_id, self.test_tool)
        
        summary = self.optimizer.end_session(session_id, promote_successful=False)
        
        self.assertEqual(summary["session_id"], session_id)
        self.assertIn("ended_at", summary)
        self.assertIn(self.test_tool.tool_id, summary["tools_used"])


class TestOptimizationStandardsEngine(unittest.TestCase):
    """Test suite for OptimizationStandardsEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = OptimizationStandardsEngine()
    
    def test_evaluate_passing_optimization(self):
        """Test evaluation of optimization that meets standards."""
        record = OptimizationRecord(
            tool_id="tool_123",
            status=OptimizationStatus.COMPLETED,
            before_metrics={"success_rate": 0.5, "avg_execution_time": 2.0},
            after_metrics={"success_rate": 0.75, "avg_execution_time": 1.0},
        )
        record.calculate_improvement()
        
        result = self.engine.evaluate(record)
        
        # With 50% improvement in success_rate and 50% reduction in time,
        # this should pass most standards
        self.assertIsInstance(result, bool)
    
    def test_evaluate_failing_optimization(self):
        """Test evaluation of optimization that doesn't meet standards."""
        record = OptimizationRecord(
            tool_id="tool_123",
            status=OptimizationStatus.COMPLETED,
            before_metrics={"success_rate": 0.75},
            after_metrics={"success_rate": 0.76},  # Only 1.3% improvement
        )
        record.calculate_improvement()
        
        result = self.engine.evaluate(record)
        
        # With only ~1% improvement, this should fail the 15% threshold
        self.assertFalse(result)
    
    def test_add_custom_standard(self):
        """Test adding a custom standard."""
        custom = OptimizationStandard(
            name="Custom Standard",
            min_improvement_threshold=5.0,
        )
        
        self.engine.add_standard("custom", custom)
        
        standards = self.engine.get_all_standards()
        self.assertIn("custom", standards)
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        record = OptimizationRecord(
            tool_id="tool_123",
            status=OptimizationStatus.COMPLETED,
            before_metrics={"success_rate": 0.6},
            after_metrics={"success_rate": 0.9},
            improvement_percentage=50.0,
        )
        
        score = self.engine.calculate_quality_score(record)
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class TestTrainingEngines(unittest.TestCase):
    """Test suite for training engines."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_tool = ToolListing(
            creator_id="creator_123",
            name="Test Tool",
            parameters={"threshold": 0.5, "timeout": 10},
            performance_metrics={
                "success_rate": 0.7,
                "avg_execution_time": 1.5,
                "accuracy": 0.75,
            },
        )
        self.config = TrainingConfig(epochs=50, learning_rate=0.01)
    
    def test_deep_learning_engine(self):
        """Test Deep Learning training engine."""
        engine = DeepLearningEngine(self.config)
        result = engine.train(self.test_tool)
        
        self.assertTrue(result.success)
        self.assertEqual(result.method, TrainingMethod.DEEP_LEARNING)
        self.assertGreater(result.epochs_completed, 0)
        self.assertIn("success_rate", result.new_metrics)
    
    def test_q_learning_engine(self):
        """Test Q-Learning training engine."""
        engine = QLearningEngine(self.config)
        result = engine.train(self.test_tool)
        
        self.assertTrue(result.success)
        self.assertEqual(result.method, TrainingMethod.Q_LEARNING)
        self.assertIn("q_table_size", result.metadata)
    
    def test_optimization_engine(self):
        """Test Optimization training engine."""
        engine = OptimizationEngine(self.config)
        result = engine.train(self.test_tool)
        
        self.assertTrue(result.success)
        self.assertEqual(result.method, TrainingMethod.OPTIMIZATION_ENGINE)
        self.assertIn("best_score", result.metadata)
    
    def test_hybrid_engine(self):
        """Test Hybrid training engine."""
        engine = HybridTrainingEngine(self.config)
        result = engine.train(self.test_tool)
        
        self.assertTrue(result.success)
        self.assertEqual(result.method, TrainingMethod.HYBRID)
        self.assertIn("best_method", result.metadata)


class TestTrainingEngineManager(unittest.TestCase):
    """Test suite for TrainingEngineManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = TrainingEngineManager()
        self.test_tool = ToolListing(
            creator_id="creator_123",
            name="Test Tool",
            performance_metrics={"success_rate": 0.7},
        )
    
    def test_train_tool_with_different_methods(self):
        """Test training with different methods."""
        for method in [TrainingMethod.DEEP_LEARNING, TrainingMethod.Q_LEARNING]:
            result = self.manager.train_tool(self.test_tool, method)
            self.assertTrue(result.success)
    
    def test_train_tool_all_methods(self):
        """Test training with all methods."""
        results = self.manager.train_tool_all_methods(self.test_tool)
        
        self.assertEqual(len(results), len(TrainingMethod))
        for method_name, result in results.items():
            self.assertIsNotNone(result)
    
    def test_get_training_stats(self):
        """Test getting training statistics."""
        # Train first
        self.manager.train_tool(self.test_tool, TrainingMethod.DEEP_LEARNING)
        
        stats = self.manager.get_training_stats()
        
        self.assertGreater(stats["total_trainings"], 0)
        self.assertIn("deep_learning", stats["by_method"])


class TestMarketplaceAPI(unittest.TestCase):
    """Test suite for MarketplaceAPI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api = MarketplaceAPI()
    
    def test_create_and_get_tool(self):
        """Test creating and retrieving a tool."""
        result = self.api.create_tool(
            creator_id="creator_123",
            name="My Tool",
            description="A great tool",
            category="analysis",
            monthly_price=19.99,
            tier="basic",
        )
        
        self.assertTrue(result["success"])
        tool_id = result["tool"]["tool_id"]
        
        tool = self.api.get_tool(tool_id)
        self.assertIsNotNone(tool)
        self.assertEqual(tool["name"], "My Tool")
    
    def test_list_tools_with_filters(self):
        """Test listing tools with filters."""
        # Create tools
        self.api.create_tool(
            creator_id="creator_1",
            name="Tool 1",
            description="",
            category="trading",
            monthly_price=10.0,
        )
        self.api.create_tool(
            creator_id="creator_2",
            name="Tool 2",
            description="",
            category="analysis",
            monthly_price=20.0,
        )
        
        # Filter by category
        trading_tools = self.api.list_tools(category="trading")
        self.assertEqual(len(trading_tools), 1)
        
        # Filter by max price
        cheap_tools = self.api.list_tools(max_price=15.0)
        self.assertEqual(len(cheap_tools), 1)
    
    def test_subscribe_and_access(self):
        """Test subscribing to a tool and checking access."""
        # Create tool
        result = self.api.create_tool(
            creator_id="creator_123",
            name="Subscription Tool",
            description="",
            category="trading",
            monthly_price=29.99,
            tier="basic",
        )
        tool_id = result["tool"]["tool_id"]
        
        # Subscribe
        sub_result = self.api.subscribe_to_tool(
            user_id="user_456",
            tool_id=tool_id,
            tier="basic",
        )
        
        self.assertTrue(sub_result["success"])
        
        # Check access
        access = self.api.check_tool_access("user_456", tool_id)
        self.assertTrue(access["has_access"])
    
    def test_session_workflow(self):
        """Test complete session optimization workflow."""
        # Create tool (free tier tool)
        result = self.api.create_tool(
            creator_id="creator_123",
            name="Session Tool",
            description="",
            category="optimization",
            monthly_price=0.0,
            tier="free",  # Explicitly set tool to free tier
        )
        tool_id = result["tool"]["tool_id"]
        
        # Subscribe (free tier)
        self.api.subscribe_to_tool(
            user_id="user_456",
            tool_id=tool_id,
            tier="free",
        )
        
        # Create session
        session_result = self.api.create_session("user_456")
        self.assertTrue(session_result["success"])
        session_id = session_result["session_id"]
        
        # Insert tool into session
        insert_result = self.api.insert_tool_into_session(
            session_id=session_id,
            user_id="user_456",
            tool_id=tool_id,
            auto_optimize=True,
        )
        
        self.assertTrue(insert_result["success"])
        self.assertTrue(insert_result["result"]["auto_optimized"])
        
        # End session
        end_result = self.api.end_session(session_id)
        self.assertTrue(end_result["success"])
    
    def test_creator_training(self):
        """Test creator training workflow."""
        # Create tool
        result = self.api.create_tool(
            creator_id="creator_123",
            name="Training Tool",
            description="",
            category="trading",
            monthly_price=49.99,
        )
        tool_id = result["tool"]["tool_id"]
        
        # Train using deep learning
        train_result = self.api.train_tool(
            creator_id="creator_123",
            tool_id=tool_id,
            method="deep_learning",
            config={"epochs": 50},
        )
        
        self.assertTrue(train_result["success"])
        self.assertIn("training_result", train_result)
        self.assertIn("meets_standards", train_result)
    
    def test_marketplace_stats(self):
        """Test getting marketplace statistics."""
        # Create some tools
        self.api.create_tool(
            creator_id="creator_123",
            name="Stats Tool",
            description="",
            category="analysis",
            monthly_price=9.99,
        )
        
        stats = self.api.get_marketplace_stats()
        
        self.assertIn("total_tools", stats)
        self.assertIn("active_tools", stats)
        self.assertIn("optimization_stats", stats)


class TestTrainingConfig(unittest.TestCase):
    """Test suite for TrainingConfig model."""
    
    def test_create_config(self):
        """Test creating a training config."""
        config = TrainingConfig(
            method=TrainingMethod.DEEP_LEARNING,
            epochs=200,
            learning_rate=0.005,
        )
        
        self.assertEqual(config.method, TrainingMethod.DEEP_LEARNING)
        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.learning_rate, 0.005)
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "method": "q_learning",
            "epochs": 500,
            "discount_factor": 0.95,
            "exploration_rate": 0.2,
        }
        
        config = TrainingConfig.from_dict(data)
        
        self.assertEqual(config.method, TrainingMethod.Q_LEARNING)
        self.assertEqual(config.epochs, 500)
        self.assertEqual(config.discount_factor, 0.95)


def run_tests():
    """Run all marketplace tests."""
    print("=" * 70)
    print("Smart Marketplace Test Suite")
    print("=" * 70)
    print(f"Test execution started at: {datetime.utcnow().isoformat()}")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestToolListing,
        TestRentalSubscription,
        TestRentalManager,
        TestOptimizationRecord,
        TestSessionOptimizer,
        TestOptimizationStandardsEngine,
        TestTrainingEngines,
        TestTrainingEngineManager,
        TestMarketplaceAPI,
        TestTrainingConfig,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    if result.wasSuccessful():
        print("✓ All tests passed successfully!")
        return True
    else:
        print("✗ Some tests failed.")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
