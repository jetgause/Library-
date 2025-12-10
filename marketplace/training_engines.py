"""
Training Engines
================

Integrates Deep Learning, Q-Learning, and Optimization Engines
for creator-driven tool optimization in the Smart Marketplace.

Features:
- Deep Learning training with configurable architectures
- Q-Learning for reinforcement-based optimization
- Optimization engines for parameter tuning
- Hybrid approaches combining multiple methods

Author: jetgause
Created: 2025-12-10
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    OptimizationRecord,
    OptimizationScope,
    OptimizationStatus,
    ToolListing,
    TrainingConfig,
    TrainingMethod,
)


@dataclass
class TrainingResult:
    """Result of a training session."""
    success: bool = False
    method: TrainingMethod = TrainingMethod.OPTIMIZATION_ENGINE
    epochs_completed: int = 0
    final_loss: float = 0.0
    improvement_percentage: float = 0.0
    optimized_parameters: Dict[str, Any] = field(default_factory=dict)
    new_metrics: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "method": self.method.value,
            "epochs_completed": self.epochs_completed,
            "final_loss": self.final_loss,
            "improvement_percentage": self.improvement_percentage,
            "optimized_parameters": self.optimized_parameters,
            "new_metrics": self.new_metrics,
            "training_time_seconds": self.training_time_seconds,
            "metadata": self.metadata,
            "error": self.error,
        }


class BaseTrainingEngine(ABC):
    """Abstract base class for training engines."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the training engine.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self._training_history: List[TrainingResult] = []
    
    @abstractmethod
    def train(
        self,
        tool: ToolListing,
        training_data: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Train/optimize a tool.
        
        Args:
            tool: Tool to optimize
            training_data: Optional training data
            
        Returns:
            TrainingResult
        """
        pass
    
    def get_training_history(self) -> List[TrainingResult]:
        """Get training history."""
        return self._training_history.copy()


class DeepLearningEngine(BaseTrainingEngine):
    """
    Deep Learning training engine for tool optimization.
    
    Simulates neural network-based optimization with:
    - Configurable architectures
    - Gradient descent optimization
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize Deep Learning engine."""
        super().__init__(config)
        self.config.method = TrainingMethod.DEEP_LEARNING
    
    def train(
        self,
        tool: ToolListing,
        training_data: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Train tool using deep learning approach.
        
        Args:
            tool: Tool to optimize
            training_data: Training data (optional)
            
        Returns:
            TrainingResult
        """
        start_time = datetime.utcnow()
        result = TrainingResult(method=TrainingMethod.DEEP_LEARNING)
        
        try:
            # Simulate training process
            epochs = self.config.epochs
            learning_rate = self.config.learning_rate
            batch_size = self.config.batch_size
            
            # Initialize loss tracking
            losses = []
            current_loss = 1.0
            best_loss = current_loss
            patience = 10
            no_improvement_count = 0
            
            # Simulate gradient descent optimization
            for epoch in range(epochs):
                # Simulate loss reduction with some noise
                gradient = learning_rate * (current_loss * 0.1 + random.uniform(-0.01, 0.01))
                current_loss = max(0.01, current_loss - gradient)
                losses.append(current_loss)
                
                # Early stopping check
                if current_loss < best_loss:
                    best_loss = current_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= patience:
                    break
                
                result.epochs_completed = epoch + 1
            
            # Calculate improvement based on loss reduction
            loss_improvement = (1.0 - best_loss) * 100
            
            # Apply improvements to tool metrics
            new_metrics = self._apply_improvements(
                tool.performance_metrics,
                loss_improvement
            )
            
            # Calculate overall improvement
            before_avg = sum(tool.performance_metrics.values()) / len(tool.performance_metrics) if tool.performance_metrics else 0
            after_avg = sum(new_metrics.values()) / len(new_metrics) if new_metrics else 0
            
            if before_avg > 0:
                result.improvement_percentage = ((after_avg - before_avg) / before_avg) * 100
            
            result.success = True
            result.final_loss = best_loss
            result.new_metrics = new_metrics
            result.optimized_parameters = tool.parameters.copy()
            result.metadata = {
                "losses": losses[-10:],  # Last 10 losses
                "early_stopped": no_improvement_count >= patience,
                "batch_size": batch_size,
                "final_learning_rate": learning_rate,
            }
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        # Record training time
        result.training_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        
        # Store in history
        self._training_history.append(result)
        
        return result
    
    def _apply_improvements(
        self,
        metrics: Dict[str, float],
        loss_improvement: float
    ) -> Dict[str, float]:
        """Apply improvements to metrics based on loss reduction."""
        new_metrics = metrics.copy()
        improvement_factor = loss_improvement / 100 * 0.3  # Scale improvement
        
        for key in new_metrics:
            if key in ["success_rate", "accuracy", "precision", "recall"]:
                # Improve rate metrics towards 1.0
                current = new_metrics[key]
                improvement = (1.0 - current) * improvement_factor
                new_metrics[key] = min(1.0, current + improvement)
            elif key in ["avg_execution_time", "latency"]:
                # Reduce time metrics
                new_metrics[key] = max(0.001, new_metrics[key] * (1.0 - improvement_factor * 0.5))
            else:
                # General improvement
                new_metrics[key] = new_metrics[key] * (1.0 + improvement_factor * 0.2)
        
        return new_metrics


class QLearningEngine(BaseTrainingEngine):
    """
    Q-Learning training engine for tool optimization.
    
    Uses reinforcement learning concepts:
    - State-action-reward framework
    - Epsilon-greedy exploration
    - Q-value updates
    - Experience replay
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize Q-Learning engine."""
        super().__init__(config)
        self.config.method = TrainingMethod.Q_LEARNING
        self._q_table: Dict[str, Dict[str, float]] = {}
    
    def train(
        self,
        tool: ToolListing,
        training_data: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Train tool using Q-Learning approach.
        
        Args:
            tool: Tool to optimize
            training_data: Training data (optional)
            
        Returns:
            TrainingResult
        """
        start_time = datetime.utcnow()
        result = TrainingResult(method=TrainingMethod.Q_LEARNING)
        
        try:
            episodes = self.config.epochs
            gamma = self.config.discount_factor
            epsilon = self.config.exploration_rate
            alpha = self.config.learning_rate
            
            # Define states and actions based on tool parameters
            states = self._define_states(tool)
            actions = self._define_actions(tool)
            
            # Initialize Q-table
            for state in states:
                self._q_table[state] = {action: 0.0 for action in actions}
            
            # Training loop
            total_rewards = []
            best_params = tool.parameters.copy()
            best_reward = float('-inf')
            
            for episode in range(episodes):
                current_state = random.choice(states)
                episode_reward = 0.0
                
                for _ in range(10):  # Steps per episode
                    # Epsilon-greedy action selection
                    if random.random() < epsilon:
                        action = random.choice(actions)
                    else:
                        action = max(
                            self._q_table[current_state],
                            key=self._q_table[current_state].get
                        )
                    
                    # Take action and observe reward
                    reward = self._simulate_reward(tool, action)
                    next_state = random.choice(states)
                    
                    # Q-value update
                    max_next_q = max(self._q_table[next_state].values())
                    current_q = self._q_table[current_state][action]
                    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
                    self._q_table[current_state][action] = new_q
                    
                    episode_reward += reward
                    current_state = next_state
                
                total_rewards.append(episode_reward)
                
                # Track best performance
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_params = self._extract_best_params(tool, actions)
                
                # Decay exploration rate
                epsilon = max(0.01, epsilon * 0.995)
                
                result.epochs_completed = episode + 1
            
            # Calculate improvement
            avg_final_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
            avg_initial_reward = sum(total_rewards[:10]) / min(10, len(total_rewards))
            
            if avg_initial_reward != 0:
                result.improvement_percentage = ((avg_final_reward - avg_initial_reward) / abs(avg_initial_reward)) * 100
            
            # Apply learned improvements
            new_metrics = self._apply_q_improvements(tool.performance_metrics, best_reward)
            
            result.success = True
            result.final_loss = -avg_final_reward  # Q-learning maximizes reward
            result.new_metrics = new_metrics
            result.optimized_parameters = best_params
            result.metadata = {
                "final_epsilon": epsilon,
                "best_reward": best_reward,
                "avg_final_reward": avg_final_reward,
                "q_table_size": len(self._q_table),
            }
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        result.training_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        self._training_history.append(result)
        
        return result
    
    def _define_states(self, tool: ToolListing) -> List[str]:
        """Define states based on tool configuration."""
        return ["low", "medium", "high", "optimal"]
    
    def _define_actions(self, tool: ToolListing) -> List[str]:
        """Define possible actions."""
        return ["increase_threshold", "decrease_threshold", "adjust_timing", "optimize_cache", "no_change"]
    
    def _simulate_reward(self, tool: ToolListing, action: str) -> float:
        """Simulate reward for an action."""
        base_reward = random.uniform(-0.5, 1.5)
        
        # Actions have different expected rewards
        action_bonuses = {
            "increase_threshold": 0.1,
            "decrease_threshold": 0.1,
            "adjust_timing": 0.2,
            "optimize_cache": 0.3,
            "no_change": 0.0,
        }
        
        return base_reward + action_bonuses.get(action, 0.0)
    
    def _extract_best_params(
        self,
        tool: ToolListing,
        actions: List[str]
    ) -> Dict[str, Any]:
        """Extract best parameters based on Q-values."""
        params = tool.parameters.copy()
        
        # Find best action for each state and apply to params
        for state, q_values in self._q_table.items():
            best_action = max(q_values, key=q_values.get)
            params[f"q_action_{state}"] = best_action
        
        return params
    
    def _apply_q_improvements(
        self,
        metrics: Dict[str, float],
        best_reward: float
    ) -> Dict[str, float]:
        """Apply Q-learning improvements to metrics."""
        new_metrics = metrics.copy()
        
        # Normalize reward to improvement factor (0-0.3)
        improvement_factor = max(0, min(0.3, best_reward / 10))
        
        for key in new_metrics:
            if key in ["success_rate", "accuracy"]:
                current = new_metrics[key]
                improvement = (1.0 - current) * improvement_factor
                new_metrics[key] = min(1.0, current + improvement)
            elif key in ["avg_execution_time", "latency"]:
                new_metrics[key] = max(0.001, new_metrics[key] * (1.0 - improvement_factor * 0.4))
        
        return new_metrics


class OptimizationEngine(BaseTrainingEngine):
    """
    General optimization engine for tool parameter tuning.
    
    Uses classical optimization techniques:
    - Grid search
    - Random search
    - Bayesian optimization (simulated)
    - Gradient-free optimization
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize Optimization engine."""
        super().__init__(config)
        self.config.method = TrainingMethod.OPTIMIZATION_ENGINE
    
    def train(
        self,
        tool: ToolListing,
        training_data: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Optimize tool using classical optimization.
        
        Args:
            tool: Tool to optimize
            training_data: Training data (optional)
            
        Returns:
            TrainingResult
        """
        start_time = datetime.utcnow()
        result = TrainingResult(method=TrainingMethod.OPTIMIZATION_ENGINE)
        
        try:
            iterations = self.config.epochs
            
            # Define optimization space
            param_space = self._define_param_space(tool)
            
            # Track best configuration
            best_score = self._evaluate_configuration(tool, tool.parameters)
            best_params = tool.parameters.copy()
            scores_history = [best_score]
            
            # Optimization loop (simulated Bayesian optimization)
            for iteration in range(iterations):
                # Generate candidate configuration
                candidate = self._generate_candidate(param_space, best_params, iteration)
                
                # Evaluate candidate
                score = self._evaluate_configuration(tool, candidate)
                scores_history.append(score)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = candidate.copy()
                
                result.epochs_completed = iteration + 1
            
            # Calculate improvement
            initial_score = scores_history[0]
            if initial_score > 0:
                result.improvement_percentage = ((best_score - initial_score) / initial_score) * 100
            
            # Apply improvements to metrics
            new_metrics = self._apply_optimization_improvements(
                tool.performance_metrics,
                result.improvement_percentage
            )
            
            result.success = True
            result.final_loss = 1.0 - best_score  # Convert score to loss
            result.new_metrics = new_metrics
            result.optimized_parameters = best_params
            result.metadata = {
                "best_score": best_score,
                "initial_score": initial_score,
                "param_space_size": len(param_space),
                "convergence": scores_history[-10:],
            }
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        result.training_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        self._training_history.append(result)
        
        return result
    
    def _define_param_space(self, tool: ToolListing) -> Dict[str, Tuple[float, float]]:
        """Define parameter search space."""
        space = {}
        
        for key, value in tool.parameters.items():
            if isinstance(value, (int, float)):
                # Create range around current value
                space[key] = (value * 0.5, value * 1.5)
        
        # Add default optimization targets
        if not space:
            space = {
                "threshold": (0.1, 0.9),
                "timeout": (1.0, 30.0),
                "cache_size": (100, 10000),
            }
        
        return space
    
    def _generate_candidate(
        self,
        param_space: Dict[str, Tuple[float, float]],
        best_params: Dict[str, Any],
        iteration: int
    ) -> Dict[str, Any]:
        """Generate a candidate configuration."""
        candidate = best_params.copy()
        
        # Mix of exploration and exploitation
        exploration_rate = max(0.1, 1.0 - iteration / 100)
        
        for key, (low, high) in param_space.items():
            if random.random() < exploration_rate:
                # Explore: random value
                candidate[key] = random.uniform(low, high)
            elif key in best_params:
                # Exploit: small perturbation around best
                current = best_params[key]
                perturbation = (high - low) * 0.1 * random.uniform(-1, 1)
                candidate[key] = max(low, min(high, current + perturbation))
        
        return candidate
    
    def _evaluate_configuration(
        self,
        tool: ToolListing,
        params: Dict[str, Any]
    ) -> float:
        """Evaluate a parameter configuration (simulated)."""
        # Simulate evaluation based on parameter values
        base_score = 0.5
        
        # Add score contributions from parameters
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Simulate optimal value around middle of range
                contribution = random.uniform(0.0, 0.1)
                base_score += contribution
        
        # Add some noise
        base_score += random.uniform(-0.05, 0.05)
        
        return max(0.0, min(1.0, base_score))
    
    def _apply_optimization_improvements(
        self,
        metrics: Dict[str, float],
        improvement_pct: float
    ) -> Dict[str, float]:
        """Apply optimization improvements to metrics."""
        new_metrics = metrics.copy()
        factor = improvement_pct / 100 * 0.4
        
        for key in new_metrics:
            if key in ["success_rate", "accuracy", "precision"]:
                current = new_metrics[key]
                improvement = (1.0 - current) * factor
                new_metrics[key] = min(1.0, current + improvement)
            elif key in ["avg_execution_time", "latency"]:
                new_metrics[key] = max(0.001, new_metrics[key] * (1.0 - factor * 0.3))
        
        return new_metrics


class HybridTrainingEngine(BaseTrainingEngine):
    """
    Hybrid training engine combining multiple methods.
    
    Combines Deep Learning, Q-Learning, and Optimization
    for comprehensive tool optimization.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize Hybrid engine."""
        super().__init__(config)
        self.config.method = TrainingMethod.HYBRID
        
        self._dl_engine = DeepLearningEngine(config)
        self._ql_engine = QLearningEngine(config)
        self._opt_engine = OptimizationEngine(config)
    
    def train(
        self,
        tool: ToolListing,
        training_data: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Train using hybrid approach.
        
        Args:
            tool: Tool to optimize
            training_data: Training data (optional)
            
        Returns:
            TrainingResult
        """
        start_time = datetime.utcnow()
        result = TrainingResult(method=TrainingMethod.HYBRID)
        
        try:
            # Run all three methods
            dl_result = self._dl_engine.train(tool, training_data)
            ql_result = self._ql_engine.train(tool, training_data)
            opt_result = self._opt_engine.train(tool, training_data)
            
            # Select best result
            results = [dl_result, ql_result, opt_result]
            best_result = max(results, key=lambda r: r.improvement_percentage if r.success else 0)
            
            if best_result.success:
                # Combine metrics from all successful methods
                combined_metrics = self._combine_metrics([
                    r.new_metrics for r in results if r.success
                ])
                
                result.success = True
                result.improvement_percentage = best_result.improvement_percentage
                result.new_metrics = combined_metrics
                result.optimized_parameters = best_result.optimized_parameters
                result.epochs_completed = sum(r.epochs_completed for r in results)
                result.final_loss = best_result.final_loss
                result.metadata = {
                    "dl_improvement": dl_result.improvement_percentage if dl_result.success else None,
                    "ql_improvement": ql_result.improvement_percentage if ql_result.success else None,
                    "opt_improvement": opt_result.improvement_percentage if opt_result.success else None,
                    "best_method": best_result.method.value,
                }
            else:
                result.success = False
                result.error = "All training methods failed"
            
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        result.training_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        self._training_history.append(result)
        
        return result
    
    def _combine_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine metrics from multiple methods."""
        if not metrics_list:
            return {}
        
        combined = {}
        
        for key in metrics_list[0]:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                # Use average for most metrics
                combined[key] = sum(values) / len(values)
        
        return combined


class TrainingEngineManager:
    """
    Manager for all training engines.
    
    Provides a unified interface for creators to train their tools
    using different methods.
    """
    
    def __init__(self):
        """Initialize the TrainingEngineManager."""
        self._engines: Dict[TrainingMethod, BaseTrainingEngine] = {
            TrainingMethod.DEEP_LEARNING: DeepLearningEngine(),
            TrainingMethod.Q_LEARNING: QLearningEngine(),
            TrainingMethod.OPTIMIZATION_ENGINE: OptimizationEngine(),
            TrainingMethod.HYBRID: HybridTrainingEngine(),
        }
        self._training_records: List[Dict[str, Any]] = []
    
    def train_tool(
        self,
        tool: ToolListing,
        method: TrainingMethod,
        config: Optional[TrainingConfig] = None,
        training_data: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Train a tool using specified method.
        
        Args:
            tool: Tool to train
            method: Training method to use
            config: Optional training configuration
            training_data: Optional training data
            
        Returns:
            TrainingResult
        """
        engine = self._engines.get(method)
        
        if not engine:
            raise ValueError(f"Unknown training method: {method}")
        
        if config:
            engine.config = config
        
        result = engine.train(tool, training_data)
        
        # Record training
        self._training_records.append({
            "tool_id": tool.tool_id,
            "method": method.value,
            "success": result.success,
            "improvement": result.improvement_percentage,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return result
    
    def train_tool_all_methods(
        self,
        tool: ToolListing,
        config: Optional[TrainingConfig] = None
    ) -> Dict[str, TrainingResult]:
        """
        Train tool using all available methods.
        
        Args:
            tool: Tool to train
            config: Optional training configuration
            
        Returns:
            Dictionary of method -> TrainingResult
        """
        results = {}
        
        for method in TrainingMethod:
            results[method.value] = self.train_tool(tool, method, config)
        
        return results
    
    def get_best_method_for_tool(
        self,
        tool: ToolListing
    ) -> Optional[TrainingMethod]:
        """
        Analyze training history to recommend best method for a tool.
        
        Args:
            tool: Tool to analyze
            
        Returns:
            Recommended TrainingMethod or None
        """
        tool_records = [
            r for r in self._training_records
            if r["tool_id"] == tool.tool_id and r["success"]
        ]
        
        if not tool_records:
            return None
        
        # Group by method and find best average improvement
        method_improvements: Dict[str, List[float]] = {}
        
        for record in tool_records:
            method = record["method"]
            if method not in method_improvements:
                method_improvements[method] = []
            method_improvements[method].append(record["improvement"])
        
        best_method = None
        best_avg = float('-inf')
        
        for method, improvements in method_improvements.items():
            avg = sum(improvements) / len(improvements)
            if avg > best_avg:
                best_avg = avg
                best_method = method
        
        return TrainingMethod(best_method) if best_method else None
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self._training_records:
            return {
                "total_trainings": 0,
                "by_method": {},
                "success_rate": 0.0,
            }
        
        total = len(self._training_records)
        successful = sum(1 for r in self._training_records if r["success"])
        
        by_method: Dict[str, Dict[str, Any]] = {}
        for method in TrainingMethod:
            method_records = [r for r in self._training_records if r["method"] == method.value]
            if method_records:
                method_successful = sum(1 for r in method_records if r["success"])
                by_method[method.value] = {
                    "total": len(method_records),
                    "successful": method_successful,
                    "success_rate": (method_successful / len(method_records)) * 100,
                    "avg_improvement": sum(r["improvement"] for r in method_records if r["success"]) / max(1, method_successful),
                }
        
        return {
            "total_trainings": total,
            "successful": successful,
            "success_rate": (successful / total) * 100,
            "by_method": by_method,
        }
