#!/usr/bin/env python3
"""
Comprehensive Batch Optimization Script for All 52 Trading Tools
================================================================

This script provides automated optimization for all trading tools in the library
with advanced features including:
- Tier and category filtering
- Parallel processing for faster optimization
- Detailed reporting and analytics
- Progress tracking and error handling
- Export capabilities for results

Author: jetgause
Created: 2025-12-10
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# TRADING TOOLS CATALOG - All 52 Tools Organized by Tier and Category
# ============================================================================

TRADING_TOOLS = {
    # TIER 1: FOUNDATION TOOLS (6 tools)
    "tier1_foundation": {
        "category": "Foundation",
        "tier": 1,
        "tools": [
            {
                "id": 1,
                "name": "Moving Average Crossover",
                "module": "moving_average_crossover",
                "description": "Dual MA crossover strategy",
                "params": ["fast_period", "slow_period"],
                "optimization_range": {"fast_period": (5, 50), "slow_period": (20, 200)}
            },
            {
                "id": 2,
                "name": "RSI Strategy",
                "module": "rsi_strategy",
                "description": "Relative Strength Index momentum strategy",
                "params": ["rsi_period", "oversold", "overbought"],
                "optimization_range": {"rsi_period": (7, 21), "oversold": (20, 35), "overbought": (65, 80)}
            },
            {
                "id": 3,
                "name": "MACD Strategy",
                "module": "macd_strategy",
                "description": "Moving Average Convergence Divergence",
                "params": ["fast_period", "slow_period", "signal_period"],
                "optimization_range": {"fast_period": (8, 16), "slow_period": (20, 30), "signal_period": (7, 12)}
            },
            {
                "id": 4,
                "name": "Bollinger Bands",
                "module": "bollinger_bands",
                "description": "Volatility-based mean reversion",
                "params": ["period", "std_dev"],
                "optimization_range": {"period": (15, 30), "std_dev": (1.5, 3.0)}
            },
            {
                "id": 5,
                "name": "Support/Resistance",
                "module": "support_resistance",
                "description": "Key level identification and trading",
                "params": ["lookback_period", "touch_tolerance"],
                "optimization_range": {"lookback_period": (20, 100), "touch_tolerance": (0.001, 0.01)}
            },
            {
                "id": 6,
                "name": "Volume Profile",
                "module": "volume_profile",
                "description": "Volume-weighted price analysis",
                "params": ["profile_period", "value_area_pct"],
                "optimization_range": {"profile_period": (20, 100), "value_area_pct": (60, 80)}
            }
        ]
    },
    
    # TIER 2: INTERMEDIATE TOOLS (12 tools)
    "tier2_intermediate": {
        "category": "Intermediate",
        "tier": 2,
        "tools": [
            {
                "id": 7,
                "name": "Ichimoku Cloud",
                "module": "ichimoku_cloud",
                "description": "Comprehensive trend and momentum system",
                "params": ["tenkan_period", "kijun_period", "senkou_span_b"],
                "optimization_range": {"tenkan_period": (7, 12), "kijun_period": (20, 30), "senkou_span_b": (44, 60)}
            },
            {
                "id": 8,
                "name": "Stochastic Oscillator",
                "module": "stochastic_oscillator",
                "description": "Momentum oscillator for overbought/oversold",
                "params": ["k_period", "d_period", "overbought", "oversold"],
                "optimization_range": {"k_period": (10, 20), "d_period": (2, 5), "overbought": (70, 85), "oversold": (15, 30)}
            },
            {
                "id": 9,
                "name": "ADX Trend Strength",
                "module": "adx_trend",
                "description": "Average Directional Index trend filter",
                "params": ["adx_period", "trend_threshold"],
                "optimization_range": {"adx_period": (10, 20), "trend_threshold": (20, 30)}
            },
            {
                "id": 10,
                "name": "Fibonacci Retracement",
                "module": "fibonacci_retracement",
                "description": "Fibonacci-based support/resistance",
                "params": ["lookback_period", "key_levels"],
                "optimization_range": {"lookback_period": (20, 100)}
            },
            {
                "id": 11,
                "name": "Parabolic SAR",
                "module": "parabolic_sar",
                "description": "Stop and reverse trend following",
                "params": ["acceleration", "maximum"],
                "optimization_range": {"acceleration": (0.015, 0.025), "maximum": (0.15, 0.25)}
            },
            {
                "id": 12,
                "name": "Keltner Channels",
                "module": "keltner_channels",
                "description": "ATR-based volatility channels",
                "params": ["ema_period", "atr_period", "multiplier"],
                "optimization_range": {"ema_period": (15, 25), "atr_period": (8, 15), "multiplier": (1.5, 3.0)}
            },
            {
                "id": 13,
                "name": "CCI Strategy",
                "module": "cci_strategy",
                "description": "Commodity Channel Index momentum",
                "params": ["cci_period", "overbought", "oversold"],
                "optimization_range": {"cci_period": (15, 25), "overbought": (100, 150), "oversold": (-150, -100)}
            },
            {
                "id": 14,
                "name": "Williams %R",
                "module": "williams_r",
                "description": "Momentum indicator for reversals",
                "params": ["period", "overbought", "oversold"],
                "optimization_range": {"period": (10, 20), "overbought": (-20, -10), "oversold": (-90, -80)}
            },
            {
                "id": 15,
                "name": "Money Flow Index",
                "module": "money_flow_index",
                "description": "Volume-weighted RSI",
                "params": ["period", "overbought", "oversold"],
                "optimization_range": {"period": (10, 20), "overbought": (70, 85), "oversold": (15, 30)}
            },
            {
                "id": 16,
                "name": "Donchian Channels",
                "module": "donchian_channels",
                "description": "Price channel breakout system",
                "params": ["period"],
                "optimization_range": {"period": (15, 30)}
            },
            {
                "id": 17,
                "name": "Aroon Indicator",
                "module": "aroon_indicator",
                "description": "Trend identification and strength",
                "params": ["period", "threshold"],
                "optimization_range": {"period": (20, 30), "threshold": (60, 80)}
            },
            {
                "id": 18,
                "name": "Chaikin Money Flow",
                "module": "chaikin_money_flow",
                "description": "Volume-weighted momentum",
                "params": ["period", "threshold"],
                "optimization_range": {"period": (15, 25), "threshold": (0.0, 0.1)}
            }
        ]
    },
    
    # TIER 3: ADVANCED TOOLS (16 tools)
    "tier3_advanced": {
        "category": "Advanced",
        "tier": 3,
        "tools": [
            {
                "id": 19,
                "name": "Elliott Wave Detector",
                "module": "elliott_wave",
                "description": "Automated Elliott Wave pattern recognition",
                "params": ["wave_degree", "correction_tolerance"],
                "optimization_range": {"wave_degree": (3, 7), "correction_tolerance": (0.15, 0.35)}
            },
            {
                "id": 20,
                "name": "Harmonic Pattern Scanner",
                "module": "harmonic_patterns",
                "description": "Gartley, Butterfly, Bat patterns",
                "params": ["pattern_tolerance", "min_pattern_size"],
                "optimization_range": {"pattern_tolerance": (0.05, 0.15), "min_pattern_size": (20, 50)}
            },
            {
                "id": 21,
                "name": "Order Flow Imbalance",
                "module": "order_flow",
                "description": "Bid/ask imbalance detection",
                "params": ["window_size", "imbalance_threshold"],
                "optimization_range": {"window_size": (10, 50), "imbalance_threshold": (0.5, 0.8)}
            },
            {
                "id": 22,
                "name": "Market Profile",
                "module": "market_profile",
                "description": "TPO-based market analysis",
                "params": ["profile_period", "value_area_pct"],
                "optimization_range": {"profile_period": (20, 60), "value_area_pct": (65, 75)}
            },
            {
                "id": 23,
                "name": "Delta Footprint",
                "module": "delta_footprint",
                "description": "Price level delta analysis",
                "params": ["aggregation_level", "delta_threshold"],
                "optimization_range": {"aggregation_level": (5, 20), "delta_threshold": (100, 500)}
            },
            {
                "id": 24,
                "name": "Volume Weighted MACD",
                "module": "vwmacd",
                "description": "MACD with volume weighting",
                "params": ["fast_period", "slow_period", "signal_period"],
                "optimization_range": {"fast_period": (8, 16), "slow_period": (20, 30), "signal_period": (7, 12)}
            },
            {
                "id": 25,
                "name": "Adaptive Moving Average",
                "module": "adaptive_ma",
                "description": "Kaufman's Adaptive MA",
                "params": ["period", "fast_ema", "slow_ema"],
                "optimization_range": {"period": (8, 15), "fast_ema": (2, 4), "slow_ema": (25, 35)}
            },
            {
                "id": 26,
                "name": "Z-Score Mean Reversion",
                "module": "zscore_reversion",
                "description": "Statistical mean reversion",
                "params": ["lookback_period", "entry_threshold", "exit_threshold"],
                "optimization_range": {"lookback_period": (15, 40), "entry_threshold": (1.5, 3.0), "exit_threshold": (0.0, 0.5)}
            },
            {
                "id": 27,
                "name": "Volatility Breakout",
                "module": "volatility_breakout",
                "description": "ATR-based breakout system",
                "params": ["atr_period", "breakout_multiplier"],
                "optimization_range": {"atr_period": (10, 20), "breakout_multiplier": (1.5, 3.5)}
            },
            {
                "id": 28,
                "name": "Heikin Ashi Strategy",
                "module": "heikin_ashi",
                "description": "Smoothed candlestick patterns",
                "params": ["confirmation_bars", "trend_strength"],
                "optimization_range": {"confirmation_bars": (2, 5), "trend_strength": (0.5, 1.5)}
            },
            {
                "id": 29,
                "name": "Renko Brick Strategy",
                "module": "renko_strategy",
                "description": "Time-independent brick charts",
                "params": ["brick_size", "reversal_bricks"],
                "optimization_range": {"brick_size": (10, 50), "reversal_bricks": (2, 5)}
            },
            {
                "id": 30,
                "name": "Elder Ray Index",
                "module": "elder_ray",
                "description": "Bull/Bear power analysis",
                "params": ["ema_period", "power_threshold"],
                "optimization_range": {"ema_period": (10, 20), "power_threshold": (0.0, 0.5)}
            },
            {
                "id": 31,
                "name": "Schaff Trend Cycle",
                "module": "schaff_trend",
                "description": "Enhanced MACD oscillator",
                "params": ["cycle_period", "fast_period", "slow_period"],
                "optimization_range": {"cycle_period": (8, 12), "fast_period": (20, 27), "slow_period": (45, 55)}
            },
            {
                "id": 32,
                "name": "Choppiness Index",
                "module": "choppiness_index",
                "description": "Market directional filter",
                "params": ["period", "trending_threshold", "ranging_threshold"],
                "optimization_range": {"period": (10, 20), "trending_threshold": (38, 45), "ranging_threshold": (55, 62)}
            },
            {
                "id": 33,
                "name": "Vortex Indicator",
                "module": "vortex_indicator",
                "description": "Trend direction and strength",
                "params": ["period", "crossover_confirmation"],
                "optimization_range": {"period": (10, 20), "crossover_confirmation": (1, 3)}
            },
            {
                "id": 34,
                "name": "Linear Regression Channel",
                "module": "linear_regression_channel",
                "description": "Statistical price channel",
                "params": ["period", "std_dev"],
                "optimization_range": {"period": (30, 100), "std_dev": (1.5, 3.0)}
            }
        ]
    },
    
    # TIER 4: EXPERT TOOLS (18 tools)
    "tier4_expert": {
        "category": "Expert",
        "tier": 4,
        "tools": [
            {
                "id": 35,
                "name": "Machine Learning Predictor",
                "module": "ml_predictor",
                "description": "Random Forest price prediction",
                "params": ["n_estimators", "max_depth", "lookback_period"],
                "optimization_range": {"n_estimators": (50, 200), "max_depth": (5, 15), "lookback_period": (20, 60)}
            },
            {
                "id": 36,
                "name": "LSTM Neural Network",
                "module": "lstm_network",
                "description": "Deep learning time series",
                "params": ["units", "layers", "dropout", "lookback"],
                "optimization_range": {"units": (32, 128), "layers": (1, 3), "dropout": (0.1, 0.3), "lookback": (30, 90)}
            },
            {
                "id": 37,
                "name": "Ensemble Strategy Combiner",
                "module": "ensemble_combiner",
                "description": "Multi-strategy voting system",
                "params": ["min_agreement", "strategy_weights"],
                "optimization_range": {"min_agreement": (0.5, 0.8)}
            },
            {
                "id": 38,
                "name": "Market Regime Detector",
                "module": "regime_detector",
                "description": "HMM-based regime identification",
                "params": ["n_regimes", "transition_penalty"],
                "optimization_range": {"n_regimes": (2, 5), "transition_penalty": (0.5, 2.0)}
            },
            {
                "id": 39,
                "name": "Cointegration Pairs",
                "module": "cointegration_pairs",
                "description": "Statistical arbitrage pairs",
                "params": ["lookback_period", "entry_zscore", "exit_zscore"],
                "optimization_range": {"lookback_period": (30, 90), "entry_zscore": (1.5, 3.0), "exit_zscore": (0.0, 0.5)}
            },
            {
                "id": 40,
                "name": "Options Greeks Strategy",
                "module": "options_greeks",
                "description": "Delta-neutral portfolio",
                "params": ["delta_threshold", "gamma_threshold", "rebalance_frequency"],
                "optimization_range": {"delta_threshold": (0.05, 0.15), "gamma_threshold": (0.01, 0.05), "rebalance_frequency": (1, 5)}
            },
            {
                "id": 41,
                "name": "Volatility Surface Arbitrage",
                "module": "vol_surface_arb",
                "description": "IV skew exploitation",
                "params": ["skew_threshold", "vega_exposure"],
                "optimization_range": {"skew_threshold": (0.1, 0.3), "vega_exposure": (100, 500)}
            },
            {
                "id": 42,
                "name": "Statistical Arbitrage",
                "module": "stat_arb",
                "description": "Multi-asset mean reversion",
                "params": ["basket_size", "entry_threshold", "exit_threshold"],
                "optimization_range": {"basket_size": (3, 10), "entry_threshold": (1.5, 3.0), "exit_threshold": (0.0, 0.5)}
            },
            {
                "id": 43,
                "name": "Kalman Filter Tracker",
                "module": "kalman_filter",
                "description": "Adaptive price tracking",
                "params": ["process_variance", "measurement_variance"],
                "optimization_range": {"process_variance": (0.001, 0.01), "measurement_variance": (0.1, 1.0)}
            },
            {
                "id": 44,
                "name": "Wavelet Transform",
                "module": "wavelet_transform",
                "description": "Multi-resolution analysis",
                "params": ["wavelet_type", "decomposition_level"],
                "optimization_range": {"decomposition_level": (3, 7)}
            },
            {
                "id": 45,
                "name": "Fractal Dimension",
                "module": "fractal_dimension",
                "description": "Market efficiency measure",
                "params": ["window_size", "fractal_threshold"],
                "optimization_range": {"window_size": (20, 60), "fractal_threshold": (1.4, 1.6)}
            },
            {
                "id": 46,
                "name": "Markov Chain Monte Carlo",
                "module": "mcmc_strategy",
                "description": "Probabilistic state transitions",
                "params": ["n_states", "burn_in_period", "samples"],
                "optimization_range": {"n_states": (3, 7), "burn_in_period": (100, 500), "samples": (500, 2000)}
            },
            {
                "id": 47,
                "name": "Reinforcement Learning Agent",
                "module": "rl_agent",
                "description": "Q-learning trading agent",
                "params": ["learning_rate", "discount_factor", "epsilon"],
                "optimization_range": {"learning_rate": (0.001, 0.01), "discount_factor": (0.9, 0.99), "epsilon": (0.05, 0.2)}
            },
            {
                "id": 48,
                "name": "Genetic Algorithm Optimizer",
                "module": "genetic_optimizer",
                "description": "Evolutionary parameter optimization",
                "params": ["population_size", "mutation_rate", "generations"],
                "optimization_range": {"population_size": (20, 100), "mutation_rate": (0.01, 0.1), "generations": (50, 200)}
            },
            {
                "id": 49,
                "name": "Sentiment Analysis Trader",
                "module": "sentiment_trader",
                "description": "News and social media sentiment",
                "params": ["sentiment_threshold", "decay_factor", "lookback_hours"],
                "optimization_range": {"sentiment_threshold": (0.5, 0.8), "decay_factor": (0.8, 0.95), "lookback_hours": (6, 48)}
            },
            {
                "id": 50,
                "name": "Limit Order Book Imbalance",
                "module": "lob_imbalance",
                "description": "Deep order book analysis",
                "params": ["depth_levels", "imbalance_threshold", "time_window"],
                "optimization_range": {"depth_levels": (5, 20), "imbalance_threshold": (0.6, 0.85), "time_window": (10, 60)}
            },
            {
                "id": 51,
                "name": "Cross-Asset Correlation",
                "module": "cross_asset_corr",
                "description": "Multi-asset correlation trading",
                "params": ["correlation_window", "correlation_threshold", "n_assets"],
                "optimization_range": {"correlation_window": (20, 90), "correlation_threshold": (0.6, 0.9), "n_assets": (3, 10)}
            },
            {
                "id": 52,
                "name": "Bayesian Network Strategy",
                "module": "bayesian_network",
                "description": "Probabilistic graphical model",
                "params": ["n_nodes", "prior_strength", "update_frequency"],
                "optimization_range": {"n_nodes": (4, 10), "prior_strength": (0.5, 2.0), "update_frequency": (5, 20)}
            }
        ]
    }
}

# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

class OptimizationResult:
    """Container for optimization results"""
    def __init__(self, tool_id: int, tool_name: str, tier: int, category: str):
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.tier = tier
        self.category = category
        self.status = "pending"
        self.best_params = {}
        self.performance_metrics = {}
        self.optimization_time = 0.0
        self.error_message = None
        self.iterations = 0
        
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "tier": self.tier,
            "category": self.category,
            "status": self.status,
            "best_params": self.best_params,
            "performance_metrics": self.performance_metrics,
            "optimization_time": self.optimization_time,
            "error_message": self.error_message,
            "iterations": self.iterations
        }


def simulate_optimization(tool: Dict, config: Dict) -> OptimizationResult:
    """
    Simulate optimization for a single tool
    In production, this would call actual optimization algorithms
    """
    result = OptimizationResult(
        tool_id=tool["id"],
        tool_name=tool["name"],
        tier=config.get("tier", 0),
        category=config.get("category", "Unknown")
    )
    
    start_time = time.time()
    
    try:
        logger.info(f"Optimizing Tool #{tool['id']}: {tool['name']}")
        
        # Simulate optimization process
        iterations = config.get("max_iterations", 100)
        
        # Mock optimization - in reality, this would run actual backtests
        import random
        time.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
        
        # Generate mock best parameters
        best_params = {}
        for param, (min_val, max_val) in tool.get("optimization_range", {}).items():
            if isinstance(min_val, int):
                best_params[param] = random.randint(min_val, max_val)
            else:
                best_params[param] = round(random.uniform(min_val, max_val), 3)
        
        # Generate mock performance metrics
        performance_metrics = {
            "sharpe_ratio": round(random.uniform(0.5, 3.0), 2),
            "total_return": round(random.uniform(-10, 50), 2),
            "max_drawdown": round(random.uniform(-25, -5), 2),
            "win_rate": round(random.uniform(0.4, 0.7), 2),
            "profit_factor": round(random.uniform(1.0, 2.5), 2),
            "total_trades": random.randint(50, 500)
        }
        
        result.status = "success"
        result.best_params = best_params
        result.performance_metrics = performance_metrics
        result.iterations = iterations
        
        logger.info(f"✓ Tool #{tool['id']} optimized successfully - Sharpe: {performance_metrics['sharpe_ratio']}")
        
    except Exception as e:
        result.status = "failed"
        result.error_message = str(e)
        logger.error(f"✗ Tool #{tool['id']} optimization failed: {str(e)}")
        logger.debug(traceback.format_exc())
    
    result.optimization_time = time.time() - start_time
    return result


def optimize_tool_wrapper(args: Tuple[Dict, Dict]) -> OptimizationResult:
    """Wrapper function for parallel processing"""
    tool, config = args
    return simulate_optimization(tool, config)


# ============================================================================
# BATCH OPTIMIZATION MANAGER
# ============================================================================

class BatchOptimizationManager:
    """Manages batch optimization of all trading tools"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results: List[OptimizationResult] = []
        self.start_time = None
        self.end_time = None
        
    def filter_tools(self) -> List[Tuple[Dict, Dict]]:
        """Filter tools based on tier and category"""
        filtered = []
        
        tier_filter = self.config.get("tier_filter", None)
        category_filter = self.config.get("category_filter", None)
        tool_ids_filter = self.config.get("tool_ids", None)
        
        for tier_key, tier_data in TRADING_TOOLS.items():
            tier = tier_data["tier"]
            category = tier_data["category"]
            
            # Apply filters
            if tier_filter and tier not in tier_filter:
                continue
            if category_filter and category not in category_filter:
                continue
                
            for tool in tier_data["tools"]:
                if tool_ids_filter and tool["id"] not in tool_ids_filter:
                    continue
                    
                # Pass tier and category info to optimization
                config = self.config.copy()
                config["tier"] = tier
                config["category"] = category
                
                filtered.append((tool, config))
        
        return filtered
    
    def run_parallel(self, tools: List[Tuple[Dict, Dict]]) -> List[OptimizationResult]:
        """Run optimizations in parallel"""
        max_workers = self.config.get("max_workers", 4)
        results = []
        
        logger.info(f"Starting parallel optimization with {max_workers} workers")
        logger.info(f"Total tools to optimize: {len(tools)}")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_tool = {
                executor.submit(optimize_tool_wrapper, tool_config): tool_config[0]
                for tool_config in tools
            }
            
            # Process completed jobs
            completed = 0
            for future in as_completed(future_to_tool):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Progress: {completed}/{len(tools)} tools completed")
                except Exception as e:
                    tool = future_to_tool[future]
                    logger.error(f"Tool {tool['name']} generated an exception: {e}")
                    
        return results
    
    def run_sequential(self, tools: List[Tuple[Dict, Dict]]) -> List[OptimizationResult]:
        """Run optimizations sequentially"""
        results = []
        
        logger.info(f"Starting sequential optimization")
        logger.info(f"Total tools to optimize: {len(tools)}")
        
        for idx, (tool, config) in enumerate(tools, 1):
            logger.info(f"Progress: {idx}/{len(tools)} tools")
            result = simulate_optimization(tool, config)
            results.append(result)
            
        return results
    
    def optimize_all(self) -> List[OptimizationResult]:
        """Main optimization execution"""
        self.start_time = time.time()
        
        logger.info("="*80)
        logger.info("BATCH OPTIMIZATION STARTED")
        logger.info("="*80)
        
        # Filter tools
        tools = self.filter_tools()
        logger.info(f"Filtered to {len(tools)} tools based on configuration")
        
        if not tools:
            logger.warning("No tools match the filter criteria!")
            return []
        
        # Run optimization
        if self.config.get("parallel", True) and len(tools) > 1:
            self.results = self.run_parallel(tools)
        else:
            self.results = self.run_sequential(tools)
        
        self.end_time = time.time()
        
        logger.info("="*80)
        logger.info("BATCH OPTIMIZATION COMPLETED")
        logger.info("="*80)
        
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive optimization report"""
        if not self.results:
            return {"error": "No results to report"}
        
        # Calculate statistics
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status == "failed"]
        
        total_time = self.end_time - self.start_time if self.end_time else 0
        avg_time_per_tool = total_time / len(self.results) if self.results else 0
        
        # Performance statistics from successful optimizations
        sharpe_ratios = [r.performance_metrics.get("sharpe_ratio", 0) for r in successful]
        returns = [r.performance_metrics.get("total_return", 0) for r in successful]
        
        report = {
            "summary": {
                "total_tools": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(self.results) * 100 if self.results else 0,
                "total_time_seconds": total_time,
                "avg_time_per_tool": avg_time_per_tool
            },
            "performance": {
                "avg_sharpe_ratio": sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0,
                "max_sharpe_ratio": max(sharpe_ratios) if sharpe_ratios else 0,
                "min_sharpe_ratio": min(sharpe_ratios) if sharpe_ratios else 0,
                "avg_return": sum(returns) / len(returns) if returns else 0,
                "max_return": max(returns) if returns else 0,
                "min_return": min(returns) if returns else 0
            },
            "by_tier": self._group_by_tier(),
            "by_category": self._group_by_category(),
            "top_performers": self._get_top_performers(10),
            "failed_tools": [{"id": r.tool_id, "name": r.tool_name, "error": r.error_message} 
                           for r in failed],
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _group_by_tier(self) -> Dict:
        """Group results by tier"""
        by_tier = {}
        for result in self.results:
            tier = result.tier
            if tier not in by_tier:
                by_tier[tier] = {"count": 0, "successful": 0, "avg_sharpe": []}
            
            by_tier[tier]["count"] += 1
            if result.status == "success":
                by_tier[tier]["successful"] += 1
                by_tier[tier]["avg_sharpe"].append(
                    result.performance_metrics.get("sharpe_ratio", 0)
                )
        
        # Calculate averages
        for tier in by_tier:
            sharpes = by_tier[tier]["avg_sharpe"]
            by_tier[tier]["avg_sharpe"] = sum(sharpes) / len(sharpes) if sharpes else 0
        
        return by_tier
    
    def _group_by_category(self) -> Dict:
        """Group results by category"""
        by_category = {}
        for result in self.results:
            category = result.category
            if category not in by_category:
                by_category[category] = {"count": 0, "successful": 0, "avg_sharpe": []}
            
            by_category[category]["count"] += 1
            if result.status == "success":
                by_category[category]["successful"] += 1
                by_category[category]["avg_sharpe"].append(
                    result.performance_metrics.get("sharpe_ratio", 0)
                )
        
        # Calculate averages
        for category in by_category:
            sharpes = by_category[category]["avg_sharpe"]
            by_category[category]["avg_sharpe"] = sum(sharpes) / len(sharpes) if sharpes else 0
        
        return by_category
    
    def _get_top_performers(self, n: int = 10) -> List[Dict]:
        """Get top N performing tools by Sharpe ratio"""
        successful = [r for r in self.results if r.status == "success"]
        sorted_results = sorted(
            successful,
            key=lambda r: r.performance_metrics.get("sharpe_ratio", 0),
            reverse=True
        )[:n]
        
        return [
            {
                "rank": idx + 1,
                "tool_id": r.tool_id,
                "tool_name": r.tool_name,
                "tier": r.tier,
                "category": r.category,
                "sharpe_ratio": r.performance_metrics.get("sharpe_ratio", 0),
                "total_return": r.performance_metrics.get("total_return", 0),
                "best_params": r.best_params
            }
            for idx, r in enumerate(sorted_results)
        ]
    
    def export_results(self, output_dir: str = "optimization_results"):
        """Export results to files"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export individual results
        results_file = Path(output_dir) / f"all_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        logger.info(f"Exported individual results to {results_file}")
        
        # Export summary report
        report = self.generate_report()
        report_file = Path(output_dir) / f"summary_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Exported summary report to {report_file}")
        
        # Export human-readable summary
        summary_file = Path(output_dir) / f"summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            self._write_text_summary(f, report)
        logger.info(f"Exported text summary to {summary_file}")
        
        return {
            "results_file": str(results_file),
            "report_file": str(report_file),
            "summary_file": str(summary_file)
        }
    
    def _write_text_summary(self, f, report: Dict):
        """Write human-readable text summary"""
        f.write("="*80 + "\n")
        f.write("BATCH OPTIMIZATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        summary = report["summary"]
        f.write(f"Total Tools Optimized: {summary['total_tools']}\n")
        f.write(f"Successful: {summary['successful']}\n")
        f.write(f"Failed: {summary['failed']}\n")
        f.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
        f.write(f"Total Time: {summary['total_time_seconds']:.1f} seconds\n")
        f.write(f"Avg Time per Tool: {summary['avg_time_per_tool']:.2f} seconds\n\n")
        
        # Performance statistics
        f.write("PERFORMANCE STATISTICS\n")
        f.write("-"*80 + "\n")
        perf = report["performance"]
        f.write(f"Average Sharpe Ratio: {perf['avg_sharpe_ratio']:.2f}\n")
        f.write(f"Max Sharpe Ratio: {perf['max_sharpe_ratio']:.2f}\n")
        f.write(f"Min Sharpe Ratio: {perf['min_sharpe_ratio']:.2f}\n")
        f.write(f"Average Return: {perf['avg_return']:.2f}%\n")
        f.write(f"Max Return: {perf['max_return']:.2f}%\n")
        f.write(f"Min Return: {perf['min_return']:.2f}%\n\n")
        
        # Top performers
        f.write("TOP 10 PERFORMERS (by Sharpe Ratio)\n")
        f.write("-"*80 + "\n")
        for performer in report["top_performers"]:
            f.write(f"{performer['rank']}. {performer['tool_name']} (ID: {performer['tool_id']})\n")
            f.write(f"   Tier: {performer['tier']} | Category: {performer['category']}\n")
            f.write(f"   Sharpe: {performer['sharpe_ratio']:.2f} | Return: {performer['total_return']:.2f}%\n")
            f.write(f"   Best Params: {json.dumps(performer['best_params'])}\n\n")
        
        # By tier
        f.write("RESULTS BY TIER\n")
        f.write("-"*80 + "\n")
        for tier, data in sorted(report["by_tier"].items()):
            f.write(f"Tier {tier}: {data['successful']}/{data['count']} successful ")
            f.write(f"(Avg Sharpe: {data['avg_sharpe']:.2f})\n")
        f.write("\n")
        
        # Failed tools
        if report["failed_tools"]:
            f.write("FAILED OPTIMIZATIONS\n")
            f.write("-"*80 + "\n")
            for failed in report["failed_tools"]:
                f.write(f"- {failed['name']} (ID: {failed['id']}): {failed['error']}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Batch optimize all 52 trading tools with advanced filtering and reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize all tools with default settings
  python optimize_all_tools.py
  
  # Optimize only Tier 1 and 2 tools
  python optimize_all_tools.py --tiers 1 2
  
  # Optimize specific tools by ID
  python optimize_all_tools.py --tool-ids 1 5 10 20
  
  # Use 8 parallel workers
  python optimize_all_tools.py --workers 8
  
  # Run sequentially (no parallelization)
  python optimize_all_tools.py --no-parallel
  
  # Filter by category
  python optimize_all_tools.py --categories Foundation Intermediate
        """
    )
    
    parser.add_argument(
        '--tiers', 
        type=int, 
        nargs='+', 
        choices=[1, 2, 3, 4],
        help='Filter by tier (1-4). Can specify multiple tiers.'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        choices=['Foundation', 'Intermediate', 'Advanced', 'Expert'],
        help='Filter by category. Can specify multiple categories.'
    )
    
    parser.add_argument(
        '--tool-ids',
        type=int,
        nargs='+',
        help='Optimize specific tools by ID (1-52). Can specify multiple IDs.'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Maximum optimization iterations per tool (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='optimization_results',
        help='Output directory for results (default: optimization_results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build configuration
    config = {
        'tier_filter': args.tiers,
        'category_filter': args.categories,
        'tool_ids': args.tool_ids,
        'max_workers': args.workers,
        'parallel': not args.no_parallel,
        'max_iterations': args.iterations,
        'output_dir': args.output_dir
    }
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Tier Filter: {config['tier_filter'] or 'All tiers'}")
    logger.info(f"  Category Filter: {config['category_filter'] or 'All categories'}")
    logger.info(f"  Tool IDs: {config['tool_ids'] or 'All tools'}")
    logger.info(f"  Parallel Processing: {config['parallel']}")
    if config['parallel']:
        logger.info(f"  Max Workers: {config['max_workers']}")
    logger.info(f"  Max Iterations: {config['max_iterations']}")
    logger.info(f"  Output Directory: {config['output_dir']}")
    logger.info("")
    
    # Create manager and run optimization
    manager = BatchOptimizationManager(config)
    results = manager.optimize_all()
    
    # Generate and display report
    report = manager.generate_report()
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total Tools: {report['summary']['total_tools']}")
    logger.info(f"Successful: {report['summary']['successful']}")
    logger.info(f"Failed: {report['summary']['failed']}")
    logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    logger.info(f"Total Time: {report['summary']['total_time_seconds']:.1f} seconds")
    logger.info(f"Avg Sharpe Ratio: {report['performance']['avg_sharpe_ratio']:.2f}")
    logger.info(f"Avg Return: {report['performance']['avg_return']:.2f}%")
    
    # Export results
    logger.info("\n" + "="*80)
    logger.info("EXPORTING RESULTS")
    logger.info("="*80)
    exported_files = manager.export_results(config['output_dir'])
    logger.info(f"Results exported to: {config['output_dir']}/")
    logger.info(f"  - Detailed results: {Path(exported_files['results_file']).name}")
    logger.info(f"  - Summary report: {Path(exported_files['report_file']).name}")
    logger.info(f"  - Text summary: {Path(exported_files['summary_file']).name}")
    
    logger.info("\n" + "="*80)
    logger.info("BATCH OPTIMIZATION COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
