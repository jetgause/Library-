# Smart Marketplace

A comprehensive marketplace system for user-created tools with monthly rental, session-based optimization, and creator training capabilities.

## Overview

The Smart Marketplace enables:

1. **Tool Rental** - Users can put their tools up for monthly rent
2. **Session Optimization** - Local optimizations are applied when users insert tools into their sessions
3. **Permanent Optimization** - Local optimizations that meet quality standards are promoted to permanent status
4. **Creator Training** - Tool creators can optimize their tools using Deep Learning, Q-Learning, and Optimization Engines

## Architecture

```
marketplace/
├── __init__.py                 # Module exports
├── models.py                   # Data models (ToolListing, Subscription, etc.)
├── rental_manager.py           # Subscription and rental management
├── session_optimizer.py        # Session-based optimization engine
├── optimization_standards.py   # Quality standards for permanent optimization
├── training_engines.py         # Deep Learning, Q-Learning, Optimization engines
└── marketplace_api.py          # Unified API interface
```

## Quick Start

### 1. Create a Tool Listing

```python
from marketplace import MarketplaceAPI

api = MarketplaceAPI()

# Create a tool
result = api.create_tool(
    creator_id="creator_123",
    name="Advanced Analytics Tool",
    description="ML-powered analytics for trading signals",
    category="analysis",
    monthly_price=29.99,
    tier="pro",
    parameters={"threshold": 0.5, "lookback_period": 20}
)

tool_id = result["tool"]["tool_id"]
```

### 2. Subscribe to a Tool (Monthly Rental)

```python
# User subscribes to the tool
subscription = api.subscribe_to_tool(
    user_id="user_456",
    tool_id=tool_id,
    tier="pro",
    months=1,
    auto_renew=True
)
```

### 3. Create a Session and Insert Tool with Optimization

```python
# Create an optimization session
session = api.create_session("user_456")
session_id = session["session_id"]

# Insert tool - automatic optimization is applied
result = api.insert_tool_into_session(
    session_id=session_id,
    user_id="user_456",
    tool_id=tool_id,
    auto_optimize=True,
    optimization_config={"epochs": 100}
)

# Session-local optimizations are now applied
print(f"Optimizations applied: {result['result']['optimizations']}")
```

### 4. Promote Optimization to Permanent

```python
# If optimization meets standards, promote it
optimization_id = result['result']['optimizations'][0]['record_id']

promote_result = api.promote_optimization(
    record_id=optimization_id,
    creator_id="creator_123"  # Only creator can promote
)

if promote_result["success"]:
    print("Optimization is now permanent!")
```

### 5. Creator Training (Deep Learning, Q-Learning, Optimization Engines)

```python
# Train using Deep Learning
dl_result = api.train_tool(
    creator_id="creator_123",
    tool_id=tool_id,
    method="deep_learning",
    config={
        "epochs": 500,
        "learning_rate": 0.001,
        "batch_size": 32
    }
)

# Train using Q-Learning
ql_result = api.train_tool(
    creator_id="creator_123",
    tool_id=tool_id,
    method="q_learning",
    config={
        "epochs": 1000,
        "discount_factor": 0.99,
        "exploration_rate": 0.1
    }
)

# Train using Optimization Engine
opt_result = api.train_tool(
    creator_id="creator_123",
    tool_id=tool_id,
    method="optimization_engine",
    config={"epochs": 200}
)

# Or use Hybrid (combines all methods)
hybrid_result = api.train_tool(
    creator_id="creator_123",
    tool_id=tool_id,
    method="hybrid"
)
```

## Key Concepts

### Subscription Tiers

| Tier | Description | Price Multiplier |
|------|-------------|------------------|
| Free | Basic access | 0.0x |
| Basic | Standard features | 1.0x |
| Pro | Advanced features | 2.5x |
| Enterprise | Full access | 5.0x |

### Optimization Flow

```
User inserts tool into session
         ↓
    Local optimization applied
         ↓
    Performance metrics updated
         ↓
    Check against standards
         ↓
  ┌──────┴──────┐
  │             │
Meets       Doesn't Meet
Standards   Standards
  │             │
  ↓             ↓
Can be      Stays as
promoted    session-local
to permanent
```

### Optimization Standards

For a local optimization to become permanent, it must meet these standards:

1. **Minimum Improvement Threshold** (default: 15%)
   - Overall performance improvement must exceed threshold

2. **Required Metrics**
   - Key metrics (success_rate, execution_time) must improve

3. **Regression Tolerance** (default: 2%)
   - No metric can regress by more than tolerance

4. **Stability**
   - Results must be consistent across multiple runs

### Training Methods

#### Deep Learning
- Neural network-based optimization
- Gradient descent with early stopping
- Best for complex pattern learning

```python
config = TrainingConfig(
    method=TrainingMethod.DEEP_LEARNING,
    epochs=500,
    learning_rate=0.001,
    batch_size=32
)
```

#### Q-Learning
- Reinforcement learning approach
- State-action-reward framework
- Epsilon-greedy exploration

```python
config = TrainingConfig(
    method=TrainingMethod.Q_LEARNING,
    epochs=1000,
    discount_factor=0.99,
    exploration_rate=0.1
)
```

#### Optimization Engine
- Classical optimization techniques
- Bayesian optimization (simulated)
- Parameter space exploration

```python
config = TrainingConfig(
    method=TrainingMethod.OPTIMIZATION_ENGINE,
    epochs=200
)
```

#### Hybrid
- Combines all three methods
- Selects best result automatically
- Most comprehensive but slower

## API Reference

### Tool Management

```python
# Create tool
api.create_tool(creator_id, name, description, category, monthly_price, tier, parameters, metadata)

# Get tool
api.get_tool(tool_id)

# List tools with filters
api.list_tools(category=None, tier=None, max_price=None, creator_id=None)

# Update tool
api.update_tool(tool_id, creator_id, updates)

# Delete tool
api.delete_tool(tool_id, creator_id)
```

### Subscription Management

```python
# Subscribe to tool
api.subscribe_to_tool(user_id, tool_id, tier, months, auto_renew)

# Cancel subscription
api.cancel_subscription(subscription_id, user_id, immediate)

# Get user subscriptions
api.get_user_subscriptions(user_id, active_only)

# Check tool access
api.check_tool_access(user_id, tool_id)
```

### Session Optimization

```python
# Create session
api.create_session(user_id, config)

# Insert tool into session
api.insert_tool_into_session(session_id, user_id, tool_id, auto_optimize, optimization_config)

# Apply specific optimization
api.apply_optimization(session_id, tool_id, method, config)

# Promote optimization to permanent
api.promote_optimization(record_id, creator_id)

# End session
api.end_session(session_id, promote_successful)
```

### Creator Training

```python
# Train tool with specific method
api.train_tool(creator_id, tool_id, method, config, training_data)

# Apply training results
api.apply_training_results(creator_id, tool_id, training_result)

# Get training recommendations
api.get_training_recommendations(creator_id, tool_id)
```

### Analytics

```python
# Get marketplace stats
api.get_marketplace_stats()

# Get creator stats
api.get_creator_stats(creator_id)

# Get tool analytics
api.get_tool_analytics(tool_id)
```

## Running Tests

```bash
python -m unittest tests.test_marketplace -v
```

## Design Decisions

### Why Session-Based Optimization?

Session-based optimization allows users to benefit from improvements without permanently changing the base tool. This:
- Protects the tool creator's original design
- Allows experimentation without risk
- Creates a pathway to permanent improvements through standards

### Why Multiple Training Methods?

Different methods excel in different scenarios:
- **Deep Learning**: Best for complex patterns with lots of data
- **Q-Learning**: Best for sequential decision problems
- **Optimization Engine**: Best for parameter tuning
- **Hybrid**: When you're not sure which method will work best

### Why Optimization Standards?

Standards ensure quality control:
- Prevents regressions that could harm users
- Ensures improvements are significant and stable
- Creates a gate for permanent changes

## Future Enhancements

1. **Real ML Integration** - Connect to actual training frameworks (PyTorch, TensorFlow)
2. **A/B Testing** - Compare tool versions with statistical significance
3. **Market Analytics** - Revenue forecasting, demand prediction
4. **Tool Reviews** - User ratings and feedback system
5. **Version History** - Track and rollback tool versions
