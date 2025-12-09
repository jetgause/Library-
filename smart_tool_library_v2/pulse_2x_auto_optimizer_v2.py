"""
PULSE 2x AUTO-OPTIMIZER v2.0
Advanced Market Simulation & Optimization Engine

Features:
- Batch market generation with vectorized operations
- Ultra-fast backtesting with numpy vectorization
- Adaptive optimization with momentum tracking
- 2x win rate targeting with intelligent parameter adjustment
- Real-time performance monitoring and analytics

Author: jetgause
Created: 2025-12-09
Version: 2.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MarketConfig:
    """Configuration for market simulation parameters"""
    num_markets: int = 1000
    lookback_period: int = 50
    trend_strength: float = 0.65
    volatility: float = 0.02
    noise_level: float = 0.01
    seasonal_factor: float = 0.15
    regime_change_prob: float = 0.05
    

@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters"""
    target_win_rate: float = 0.66  # 2x win rate (from 0.33 baseline)
    min_trades: int = 100
    max_iterations: int = 500
    convergence_threshold: float = 0.001
    learning_rate: float = 0.05
    momentum_factor: float = 0.9
    adaptive_step_size: bool = True
    population_size: int = 50
    

@dataclass
class StrategyParams:
    """Trading strategy parameters"""
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    ma_fast: int = 10
    ma_slow: int = 30
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    atr_multiplier: float = 1.5
    trend_threshold: float = 0.001
    volume_factor: float = 1.2
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to numpy array"""
        return np.array([
            self.rsi_period, self.rsi_oversold, self.rsi_overbought,
            self.ma_fast, self.ma_slow, self.bb_period, self.bb_std,
            self.atr_period, self.atr_multiplier, self.trend_threshold,
            self.volume_factor
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'StrategyParams':
        """Create parameters from numpy array"""
        return cls(
            rsi_period=int(arr[0]),
            rsi_oversold=float(arr[1]),
            rsi_overbought=float(arr[2]),
            ma_fast=int(arr[3]),
            ma_slow=int(arr[4]),
            bb_period=int(arr[5]),
            bb_std=float(arr[6]),
            atr_period=int(arr[7]),
            atr_multiplier=float(arr[8]),
            trend_threshold=float(arr[9]),
            volume_factor=float(arr[10])
        )


@dataclass
class BacktestResults:
    """Results from backtesting"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))


class VectorizedMarketGenerator:
    """Ultra-fast market data generation using vectorized operations"""
    
    def __init__(self, config: MarketConfig):
        self.config = config
        
    def generate_batch_markets(self, num_markets: int = None) -> np.ndarray:
        """
        Generate multiple market scenarios simultaneously using vectorization
        
        Returns:
            Array of shape (num_markets, num_bars, 6) where last dim is [O, H, L, C, V, T]
        """
        if num_markets is None:
            num_markets = self.config.num_markets
            
        num_bars = 1000  # Generate 1000 bars per market
        
        # Initialize price matrices
        markets = np.zeros((num_markets, num_bars, 6))
        
        # Generate base trends
        trends = np.random.choice(
            [-1, 0, 1], 
            size=(num_markets, 1), 
            p=[0.3, 0.4, 0.3]
        )
        
        # Generate trend strengths
        trend_strengths = np.random.uniform(
            0.5, self.config.trend_strength * 1.5, 
            size=(num_markets, 1)
        )
        
        # Generate base prices
        base_prices = np.random.uniform(50, 200, size=(num_markets, 1))
        
        # Time array for all markets
        time = np.arange(num_bars).reshape(1, -1)
        
        # Generate trend components (vectorized)
        trend_component = trends * trend_strengths * time * 0.001
        
        # Generate seasonal components
        seasonal = self.config.seasonal_factor * np.sin(2 * np.pi * time / 50)
        
        # Generate volatility (time-varying)
        vol = self.config.volatility * (1 + 0.3 * np.sin(2 * np.pi * time / 100))
        
        # Generate returns with regime changes
        regime_changes = np.random.random((num_markets, num_bars)) < self.config.regime_change_prob
        regime_volatility = np.where(regime_changes, vol * 2, vol)
        
        returns = np.random.normal(0, 1, (num_markets, num_bars)) * regime_volatility
        returns += trend_component + seasonal
        
        # Add noise
        noise = np.random.normal(0, self.config.noise_level, (num_markets, num_bars))
        returns += noise
        
        # Calculate cumulative prices
        cumulative_returns = np.cumsum(returns, axis=1)
        prices = base_prices * np.exp(cumulative_returns)
        
        # Generate OHLC data
        intrabar_volatility = vol * 0.5
        
        # Close prices
        markets[:, :, 3] = prices
        
        # High/Low with realistic intrabar movement
        high_dev = np.abs(np.random.normal(0, intrabar_volatility, (num_markets, num_bars)))
        low_dev = np.abs(np.random.normal(0, intrabar_volatility, (num_markets, num_bars)))
        
        markets[:, :, 1] = prices * (1 + high_dev)  # High
        markets[:, :, 2] = prices * (1 - low_dev)   # Low
        
        # Open prices (previous close with gap)
        markets[:, 1:, 0] = markets[:, :-1, 3] * (1 + np.random.normal(0, vol[:, 1:] * 0.3, (num_markets, num_bars - 1)))
        markets[:, 0, 0] = prices[:, 0]
        
        # Volume (correlated with price movement)
        price_change = np.abs(np.diff(markets[:, :, 3], axis=1, prepend=markets[:, :1, 3]))
        base_volume = np.random.lognormal(10, 1, (num_markets, num_bars))
        volume_multiplier = 1 + 2 * (price_change / (prices + 1e-10))
        markets[:, :, 4] = base_volume * volume_multiplier
        
        # Timestamp
        markets[:, :, 5] = time
        
        return markets


class VectorizedBacktester:
    """Ultra-fast backtesting engine using numpy vectorization"""
    
    def __init__(self):
        self.results_cache = {}
        
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized RSI calculation"""
        deltas = np.diff(prices, axis=-1, prepend=prices[..., :1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use exponential moving average
        alpha = 1.0 / period
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        avg_gains[..., period] = np.mean(gains[..., :period], axis=-1)
        avg_losses[..., period] = np.mean(losses[..., :period], axis=-1)
        
        for i in range(period + 1, prices.shape[-1]):
            avg_gains[..., i] = alpha * gains[..., i] + (1 - alpha) * avg_gains[..., i - 1]
            avg_losses[..., i] = alpha * losses[..., i] + (1 - alpha) * avg_losses[..., i - 1]
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Vectorized SMA calculation"""
        kernel = np.ones(period) / period
        # Pad to maintain shape
        padded = np.pad(prices, ((0, 0), (period - 1, 0)), mode='edge')
        return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), -1, padded)
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized Bollinger Bands calculation"""
        sma = VectorizedBacktester.calculate_sma(prices, period)
        
        # Rolling standard deviation
        rolling_std = np.zeros_like(prices)
        for i in range(period - 1, prices.shape[-1]):
            rolling_std[:, i] = np.std(prices[:, i - period + 1:i + 1], axis=-1)
        
        upper = sma + std_dev * rolling_std
        lower = sma - std_dev * rolling_std
        
        return upper, sma, lower
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized ATR calculation"""
        prev_close = np.roll(close, 1, axis=-1)
        prev_close[..., 0] = close[..., 0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Exponential moving average of TR
        alpha = 1.0 / period
        atr = np.zeros_like(close)
        atr[..., period - 1] = np.mean(tr[..., :period], axis=-1)
        
        for i in range(period, close.shape[-1]):
            atr[..., i] = alpha * tr[..., i] + (1 - alpha) * atr[..., i - 1]
        
        return atr
    
    def backtest_vectorized(self, markets: np.ndarray, params: StrategyParams) -> List[BacktestResults]:
        """
        Vectorized backtesting across multiple markets simultaneously
        
        Args:
            markets: Array of shape (num_markets, num_bars, 6)
            params: Strategy parameters
            
        Returns:
            List of BacktestResults for each market
        """
        num_markets = markets.shape[0]
        num_bars = markets.shape[1]
        
        # Extract OHLCV
        opens = markets[:, :, 0]
        highs = markets[:, :, 1]
        lows = markets[:, :, 2]
        closes = markets[:, :, 3]
        volumes = markets[:, :, 4]
        
        # Calculate indicators
        rsi = self.calculate_rsi(closes, params.rsi_period)
        ma_fast = self.calculate_sma(closes, params.ma_fast)
        ma_slow = self.calculate_sma(closes, params.ma_slow)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes, params.bb_period, params.bb_std)
        atr = self.calculate_atr(highs, lows, closes, params.atr_period)
        
        # Generate signals (vectorized)
        # Long signals
        rsi_long = rsi < params.rsi_oversold
        ma_long = ma_fast > ma_slow
        bb_long = closes < bb_lower
        
        long_signals = rsi_long & ma_long & bb_long
        
        # Short signals
        rsi_short = rsi > params.rsi_overbought
        ma_short = ma_fast < ma_slow
        bb_short = closes > bb_upper
        
        short_signals = rsi_short & ma_short & bb_short
        
        # Position tracking
        positions = np.zeros((num_markets, num_bars))
        entry_prices = np.zeros((num_markets, num_bars))
        stop_losses = np.zeros((num_markets, num_bars))
        take_profits = np.zeros((num_markets, num_bars))
        
        # Trade tracking
        trades_list = [[] for _ in range(num_markets)]
        
        # Simulate trading
        for i in range(1, num_bars):
            # Check for entries
            long_entry = long_signals[:, i] & (positions[:, i - 1] == 0)
            short_entry = short_signals[:, i] & (positions[:, i - 1] == 0)
            
            # Open long positions
            positions[:, i] = np.where(long_entry, 1, positions[:, i - 1])
            entry_prices[:, i] = np.where(long_entry, closes[:, i], entry_prices[:, i - 1])
            stop_losses[:, i] = np.where(long_entry, closes[:, i] - params.atr_multiplier * atr[:, i], stop_losses[:, i - 1])
            take_profits[:, i] = np.where(long_entry, closes[:, i] + 2 * params.atr_multiplier * atr[:, i], take_profits[:, i - 1])
            
            # Open short positions
            positions[:, i] = np.where(short_entry, -1, positions[:, i])
            entry_prices[:, i] = np.where(short_entry, closes[:, i], entry_prices[:, i])
            stop_losses[:, i] = np.where(short_entry, closes[:, i] + params.atr_multiplier * atr[:, i], stop_losses[:, i])
            take_profits[:, i] = np.where(short_entry, closes[:, i] - 2 * params.atr_multiplier * atr[:, i], take_profits[:, i])
            
            # Check exits for existing positions
            in_long = positions[:, i - 1] == 1
            in_short = positions[:, i - 1] == -1
            
            # Long exits
            long_stop_hit = in_long & (lows[:, i] <= stop_losses[:, i - 1])
            long_profit_hit = in_long & (highs[:, i] >= take_profits[:, i - 1])
            long_signal_exit = in_long & short_signals[:, i]
            
            long_exit = long_stop_hit | long_profit_hit | long_signal_exit
            
            # Short exits
            short_stop_hit = in_short & (highs[:, i] >= stop_losses[:, i - 1])
            short_profit_hit = in_short & (lows[:, i] <= take_profits[:, i - 1])
            short_signal_exit = in_short & long_signals[:, i]
            
            short_exit = short_stop_hit | short_profit_hit | short_signal_exit
            
            # Calculate PnL for exits
            long_pnl = (closes[:, i] - entry_prices[:, i - 1]) / entry_prices[:, i - 1]
            short_pnl = (entry_prices[:, i - 1] - closes[:, i]) / entry_prices[:, i - 1]
            
            # Record trades
            for m in range(num_markets):
                if long_exit[m]:
                    trades_list[m].append({
                        'entry': entry_prices[m, i - 1],
                        'exit': closes[m, i],
                        'direction': 'long',
                        'pnl': long_pnl[m],
                        'bars': i
                    })
                    positions[m, i] = 0
                    
                if short_exit[m]:
                    trades_list[m].append({
                        'entry': entry_prices[m, i - 1],
                        'exit': closes[m, i],
                        'direction': 'short',
                        'pnl': short_pnl[m],
                        'bars': i
                    })
                    positions[m, i] = 0
        
        # Calculate results for each market
        results = []
        for m in range(num_markets):
            trades = trades_list[m]
            
            if len(trades) == 0:
                results.append(BacktestResults())
                continue
            
            pnls = np.array([t['pnl'] for t in trades])
            winning = pnls > 0
            losing = pnls < 0
            
            total_trades = len(trades)
            winning_trades = np.sum(winning)
            losing_trades = np.sum(losing)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_return = np.sum(pnls)
            sharpe_ratio = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
            
            # Calculate max drawdown
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            avg_win = np.mean(pnls[winning]) if winning_trades > 0 else 0
            avg_loss = np.mean(pnls[losing]) if losing_trades > 0 else 0
            
            total_wins = np.sum(pnls[winning])
            total_losses = abs(np.sum(pnls[losing]))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            results.append(BacktestResults(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                trades=trades,
                equity_curve=cumulative
            ))
        
        return results


class AdaptiveOptimizer:
    """Adaptive optimization engine with momentum and intelligent parameter adjustment"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_params = None
        self.best_score = -np.inf
        self.iteration_history = []
        self.momentum_velocity = None
        
    def calculate_fitness(self, results: List[BacktestResults]) -> float:
        """
        Calculate fitness score with emphasis on 2x win rate achievement
        
        Scoring components:
        - Win rate (primary): 50%
        - Sharpe ratio: 20%
        - Profit factor: 15%
        - Total return: 10%
        - Trade count: 5%
        """
        valid_results = [r for r in results if r.total_trades >= self.config.min_trades]
        
        if len(valid_results) == 0:
            return -1000
        
        # Average metrics across markets
        avg_win_rate = np.mean([r.win_rate for r in valid_results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in valid_results])
        avg_profit_factor = np.mean([r.profit_factor for r in valid_results])
        avg_return = np.mean([r.total_return for r in valid_results])
        avg_trades = np.mean([r.total_trades for r in valid_results])
        
        # Normalize components
        win_rate_score = (avg_win_rate - 0.33) / (self.config.target_win_rate - 0.33) * 100
        win_rate_score = min(win_rate_score, 100)  # Cap at 100
        
        sharpe_score = min(avg_sharpe * 10, 100)
        profit_factor_score = min(avg_profit_factor * 20, 100)
        return_score = min(avg_return * 100, 100)
        trade_score = min(avg_trades / self.config.min_trades * 20, 100)
        
        # Weighted combination with emphasis on win rate
        fitness = (
            0.50 * win_rate_score +
            0.20 * sharpe_score +
            0.15 * profit_factor_score +
            0.10 * return_score +
            0.05 * trade_score
        )
        
        # Bonus for achieving target win rate
        if avg_win_rate >= self.config.target_win_rate:
            fitness *= 1.2
        
        # Consistency bonus (lower std = better)
        win_rate_std = np.std([r.win_rate for r in valid_results])
        consistency_bonus = max(0, (0.1 - win_rate_std) * 50)
        fitness += consistency_bonus
        
        return fitness
    
    def mutate_params(self, params: StrategyParams, iteration: int, adaptive: bool = True) -> StrategyParams:
        """
        Mutate parameters with adaptive step size based on iteration and momentum
        """
        params_array = params.to_array()
        
        # Adaptive learning rate
        if adaptive:
            progress = iteration / self.config.max_iterations
            step_size = self.config.learning_rate * (1 - 0.8 * progress)  # Decay over time
        else:
            step_size = self.config.learning_rate
        
        # Parameter bounds
        bounds = np.array([
            [7, 30],      # rsi_period
            [20, 40],     # rsi_oversold
            [60, 80],     # rsi_overbought
            [5, 20],      # ma_fast
            [20, 50],     # ma_slow
            [10, 30],     # bb_period
            [1.5, 3.0],   # bb_std
            [7, 21],      # atr_period
            [1.0, 3.0],   # atr_multiplier
            [0.0005, 0.005],  # trend_threshold
            [1.0, 2.0]    # volume_factor
        ])
        
        # Apply mutation with momentum
        mutation = np.random.randn(len(params_array)) * step_size
        
        if self.momentum_velocity is not None:
            mutation = self.config.momentum_factor * self.momentum_velocity + (1 - self.config.momentum_factor) * mutation
            self.momentum_velocity = mutation
        else:
            self.momentum_velocity = mutation
        
        # Scale mutation by parameter range
        param_ranges = bounds[:, 1] - bounds[:, 0]
        scaled_mutation = mutation * param_ranges * 0.1
        
        new_params = params_array + scaled_mutation
        
        # Clip to bounds
        new_params = np.clip(new_params, bounds[:, 0], bounds[:, 1])
        
        # Ensure integer constraints
        new_params[[0, 3, 4, 5, 7]] = np.round(new_params[[0, 3, 4, 5, 7]])
        
        return StrategyParams.from_array(new_params)
    
    def optimize(self, markets: np.ndarray, initial_params: StrategyParams = None) -> Tuple[StrategyParams, Dict]:
        """
        Optimize strategy parameters using adaptive genetic algorithm with momentum
        
        Returns:
            Best parameters and optimization history
        """
        print(f"\n{'=' * 60}")
        print("PULSE 2x AUTO-OPTIMIZER v2.0 - Optimization Started")
        print(f"{'=' * 60}")
        print(f"Target Win Rate: {self.config.target_win_rate:.1%}")
        print(f"Markets: {markets.shape[0]}")
        print(f"Max Iterations: {self.config.max_iterations}")
        print(f"{'=' * 60}\n")
        
        backtester = VectorizedBacktester()
        
        # Initialize population
        if initial_params is None:
            initial_params = StrategyParams()
        
        population = [initial_params]
        for _ in range(self.config.population_size - 1):
            population.append(self.mutate_params(initial_params, 0, adaptive=False))
        
        best_iteration = 0
        no_improvement_count = 0
        
        for iteration in range(self.config.max_iterations):
            # Evaluate population
            fitness_scores = []
            
            for params in population:
                results = backtester.backtest_vectorized(markets, params)
                fitness = self.calculate_fitness(results)
                fitness_scores.append((fitness, params, results))
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            current_best_score = fitness_scores[0][0]
            current_best_params = fitness_scores[0][1]
            current_best_results = fitness_scores[0][2]
            
            # Update global best
            if current_best_score > self.best_score:
                improvement = current_best_score - self.best_score
                self.best_score = current_best_score
                self.best_params = current_best_params
                best_iteration = iteration
                no_improvement_count = 0
                
                # Calculate average metrics
                valid_results = [r for r in current_best_results if r.total_trades >= self.config.min_trades]
                avg_win_rate = np.mean([r.win_rate for r in valid_results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in valid_results])
                avg_return = np.mean([r.total_return for r in valid_results])
                
                print(f"Iteration {iteration + 1}: NEW BEST! 🚀")
                print(f"  Fitness Score: {self.best_score:.2f} (+{improvement:.2f})")
                print(f"  Win Rate: {avg_win_rate:.2%} (Target: {self.config.target_win_rate:.2%})")
                print(f"  Sharpe Ratio: {avg_sharpe:.2f}")
                print(f"  Avg Return: {avg_return:.2%}")
                
                if avg_win_rate >= self.config.target_win_rate:
                    print(f"  ✓ TARGET WIN RATE ACHIEVED!")
                
                print()
            else:
                no_improvement_count += 1
                
                if (iteration + 1) % 50 == 0:
                    print(f"Iteration {iteration + 1}: Score = {current_best_score:.2f} (Best: {self.best_score:.2f})")
            
            # Store history
            self.iteration_history.append({
                'iteration': iteration + 1,
                'best_score': self.best_score,
                'current_score': current_best_score,
                'avg_win_rate': np.mean([r.win_rate for r in current_best_results if r.total_trades >= self.config.min_trades])
            })
            
            # Check convergence
            if no_improvement_count >= 100:
                print(f"\nConverged after {iteration + 1} iterations (no improvement for 100 iterations)")
                break
            
            if self.best_score > 95 and iteration > 100:  # Near optimal
                print(f"\nNear-optimal solution found at iteration {iteration + 1}")
                break
            
            # Create next generation
            # Keep top 20% (elitism)
            elite_size = self.config.population_size // 5
            next_population = [params for _, params, _ in fitness_scores[:elite_size]]
            
            # Generate offspring with mutation
            while len(next_population) < self.config.population_size:
                # Select parent from top 50%
                parent_idx = np.random.randint(0, self.config.population_size // 2)
                parent = fitness_scores[parent_idx][1]
                
                # Mutate
                offspring = self.mutate_params(parent, iteration, adaptive=self.config.adaptive_step_size)
                next_population.append(offspring)
            
            population = next_population
        
        # Final summary
        print(f"\n{'=' * 60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Best Iteration: {best_iteration + 1}")
        print(f"Best Fitness Score: {self.best_score:.2f}")
        
        # Run final backtest with best parameters
        final_results = backtester.backtest_vectorized(markets, self.best_params)
        valid_results = [r for r in final_results if r.total_trades >= self.config.min_trades]
        
        if valid_results:
            print(f"\nFinal Performance Metrics:")
            print(f"  Win Rate: {np.mean([r.win_rate for r in valid_results]):.2%}")
            print(f"  Sharpe Ratio: {np.mean([r.sharpe_ratio for r in valid_results]):.2f}")
            print(f"  Profit Factor: {np.mean([r.profit_factor for r in valid_results]):.2f}")
            print(f"  Avg Return: {np.mean([r.total_return for r in valid_results]):.2%}")
            print(f"  Avg Trades: {np.mean([r.total_trades for r in valid_results]):.0f}")
        
        print(f"\nOptimized Parameters:")
        for key, value in self.best_params.__dict__.items():
            print(f"  {key}: {value}")
        print(f"{'=' * 60}\n")
        
        optimization_summary = {
            'best_score': self.best_score,
            'best_iteration': best_iteration,
            'total_iterations': len(self.iteration_history),
            'history': self.iteration_history,
            'final_results': valid_results
        }
        
        return self.best_params, optimization_summary


class PULSE2xAutoOptimizer:
    """Main class orchestrating the entire optimization pipeline"""
    
    def __init__(self, 
                 market_config: MarketConfig = None,
                 optimization_config: OptimizationConfig = None):
        self.market_config = market_config or MarketConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        self.market_generator = VectorizedMarketGenerator(self.market_config)
        self.optimizer = AdaptiveOptimizer(self.optimization_config)
        
    def run_full_optimization(self, 
                            num_markets: int = None,
                            initial_params: StrategyParams = None,
                            save_results: bool = True) -> Dict:
        """
        Run the complete optimization pipeline
        
        Args:
            num_markets: Number of markets to generate (default from config)
            initial_params: Starting parameters (default values if None)
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing optimized parameters and performance metrics
        """
        start_time = datetime.now()
        
        print("\n" + "=" * 60)
        print("PULSE 2x AUTO-OPTIMIZER v2.0")
        print("Ultra-Fast Market Simulation & Strategy Optimization")
        print("=" * 60)
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Generate markets
        print("Step 1: Generating Synthetic Markets...")
        if num_markets is None:
            num_markets = self.market_config.num_markets
        
        markets = self.market_generator.generate_batch_markets(num_markets)
        print(f"✓ Generated {num_markets} market scenarios")
        print(f"  Shape: {markets.shape}")
        print()
        
        # Optimize
        print("Step 2: Running Adaptive Optimization...")
        best_params, optimization_summary = self.optimizer.optimize(markets, initial_params)
        print("✓ Optimization complete")
        print()
        
        # Compile results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'market_config': self.market_config.__dict__,
            'optimization_config': self.optimization_config.__dict__,
            'best_parameters': best_params.__dict__,
            'optimization_summary': {
                'best_score': optimization_summary['best_score'],
                'best_iteration': optimization_summary['best_iteration'],
                'total_iterations': optimization_summary['total_iterations']
            },
            'performance_metrics': {
                'avg_win_rate': np.mean([r.win_rate for r in optimization_summary['final_results']]),
                'avg_sharpe_ratio': np.mean([r.sharpe_ratio for r in optimization_summary['final_results']]),
                'avg_profit_factor': np.mean([r.profit_factor for r in optimization_summary['final_results']]),
                'avg_return': np.mean([r.total_return for r in optimization_summary['final_results']]),
                'target_achieved': np.mean([r.win_rate for r in optimization_summary['final_results']]) >= self.optimization_config.target_win_rate
            }
        }
        
        print(f"Total Duration: {duration:.2f} seconds")
        print(f"Markets per Second: {num_markets / duration:.2f}")
        print()
        
        # Save results
        if save_results:
            filename = f"pulse_2x_optimization_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x))
                json.dump(json_results, f, indent=2)
            print(f"✓ Results saved to: {filename}")
        
        return results
    
    def quick_test(self, num_markets: int = 100, max_iterations: int = 100):
        """Run a quick test with reduced parameters"""
        print("\n🔥 PULSE 2x Quick Test Mode 🔥\n")
        
        self.market_config.num_markets = num_markets
        self.optimization_config.max_iterations = max_iterations
        
        return self.run_full_optimization(num_markets=num_markets, save_results=False)


def main():
    """Example usage and demonstration"""
    
    # Configuration
    market_config = MarketConfig(
        num_markets=500,
        lookback_period=50,
        trend_strength=0.65,
        volatility=0.02
    )
    
    optimization_config = OptimizationConfig(
        target_win_rate=0.66,  # 2x target (from 0.33 baseline)
        max_iterations=300,
        population_size=50,
        learning_rate=0.05,
        momentum_factor=0.9
    )
    
    # Initialize optimizer
    optimizer = PULSE2xAutoOptimizer(
        market_config=market_config,
        optimization_config=optimization_config
    )
    
    # Run optimization
    results = optimizer.run_full_optimization()
    
    # Display summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Target Win Rate: {optimization_config.target_win_rate:.1%}")
    print(f"Achieved Win Rate: {results['performance_metrics']['avg_win_rate']:.1%}")
    print(f"Target Achieved: {'✓ YES' if results['performance_metrics']['target_achieved'] else '✗ NO'}")
    print(f"Sharpe Ratio: {results['performance_metrics']['avg_sharpe_ratio']:.2f}")
    print(f"Profit Factor: {results['performance_metrics']['avg_profit_factor']:.2f}")
    print(f"Optimization Score: {results['optimization_summary']['best_score']:.2f}/100")
    print("=" * 60)


if __name__ == "__main__":
    # Uncomment to run full optimization
    # main()
    
    # Quick test mode
    optimizer = PULSE2xAutoOptimizer()
    results = optimizer.quick_test(num_markets=100, max_iterations=50)
