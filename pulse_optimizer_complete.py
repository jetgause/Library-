#!/usr/bin/env python3
"""
PULSE TOOLS OPTIMIZER - Complete 0DTE Options Trading Strategy System

A self-contained backtest and calibration system for 0DTE options trading strategies.

Features:
- 100K+ simulation backtesting
- Systematic threshold optimization
- Automatic target win rate convergence
- RL training data export capabilities
- Comprehensive performance analytics

Author: jetgause
Created: 2025-12-09
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeResult:
    """Container for individual trade results"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    trade_type: str  # 'long' or 'short'
    entry_signal: Dict
    exit_reason: str
    hold_duration: float  # in minutes
    
    def to_dict(self):
        return {
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'trade_type': self.trade_type,
            'entry_signal': self.entry_signal,
            'exit_reason': self.exit_reason,
            'hold_duration': self.hold_duration
        }


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_hold_time: float
    total_return_pct: float
    
    def to_dict(self):
        return asdict(self)


class PulseSignalGenerator:
    """Generate trading signals based on PULSE methodology"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rsi_period = config.get('rsi_period', 14)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.volume_ma_period = config.get('volume_ma_period', 20)
        
    def calculate_rsi(self, prices: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate RSI indicator"""
        if period is None:
            period = self.rsi_period
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.divide(avg_gains, avg_losses, where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50  # neutral for initial period
        
        return rsi
    
    def calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=self.bb_period).mean().values
        std = pd.Series(prices).rolling(window=self.bb_period).std().values
        
        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)
        
        return upper_band, sma, lower_band
    
    def calculate_volume_profile(self, volumes: np.ndarray) -> np.ndarray:
        """Calculate volume moving average"""
        return pd.Series(volumes).rolling(window=self.volume_ma_period).mean().values
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data"""
        df = data.copy()
        
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df['close'].values)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'].values)
        df['volume_ma'] = self.calculate_volume_profile(df['volume'].values)
        
        # Price position relative to BB
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume relative to average
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_momentum'] = df['price_change'].rolling(window=5).mean()
        
        # Generate signal scores
        df['signal_score'] = 0.0
        
        # Long signals (oversold conditions)
        long_conditions = (
            (df['rsi'] < self.config.get('rsi_oversold', 30)) * 0.3 +
            (df['bb_position'] < self.config.get('bb_lower_threshold', 0.2)) * 0.3 +
            (df['volume_ratio'] > self.config.get('volume_threshold', 1.2)) * 0.2 +
            (df['price_momentum'] < 0) * 0.2
        )
        
        # Short signals (overbought conditions)
        short_conditions = (
            (df['rsi'] > self.config.get('rsi_overbought', 70)) * 0.3 +
            (df['bb_position'] > self.config.get('bb_upper_threshold', 0.8)) * 0.3 +
            (df['volume_ratio'] > self.config.get('volume_threshold', 1.2)) * 0.2 +
            (df['price_momentum'] > 0) * 0.2
        )
        
        df['signal_score'] = long_conditions - short_conditions
        df['signal_type'] = np.where(df['signal_score'] > self.config.get('signal_threshold', 0.5), 'long',
                                     np.where(df['signal_score'] < -self.config.get('signal_threshold', 0.5), 'short', 'neutral'))
        
        return df


class PositionManager:
    """Manage trading positions and risk"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.position_size = config.get('position_size', 1.0)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.04)
        self.max_hold_minutes = config.get('max_hold_minutes', 240)  # 4 hours for 0DTE
        
    def can_enter(self, current_time: datetime) -> bool:
        """Check if we can enter a new position"""
        if self.position is not None:
            return False
        
        # Don't enter positions too close to market close (last 30 minutes)
        market_close = current_time.replace(hour=16, minute=0, second=0)
        time_to_close = (market_close - current_time).total_seconds() / 60
        
        return time_to_close > 30
    
    def enter_position(self, trade_type: str, price: float, time: datetime, signal: Dict) -> None:
        """Enter a new position"""
        self.position = trade_type
        self.entry_price = price
        self.entry_time = time
        self.entry_signal = signal
    
    def should_exit(self, current_price: float, current_time: datetime) -> Tuple[bool, str]:
        """Check if position should be exited"""
        if self.position is None:
            return False, ""
        
        # Calculate P&L
        if self.position == 'long':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # short
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, "stop_loss"
        
        # Take profit
        if pnl_pct >= self.take_profit_pct:
            return True, "take_profit"
        
        # Max hold time
        hold_time = (current_time - self.entry_time).total_seconds() / 60
        if hold_time >= self.max_hold_minutes:
            return True, "max_hold_time"
        
        # End of day exit (15 minutes before close)
        market_close = current_time.replace(hour=16, minute=0, second=0)
        time_to_close = (market_close - current_time).total_seconds() / 60
        if time_to_close <= 15:
            return True, "end_of_day"
        
        return False, ""
    
    def exit_position(self, exit_price: float, exit_time: datetime, exit_reason: str) -> TradeResult:
        """Exit current position and return trade result"""
        if self.position is None:
            return None
        
        if self.position == 'long':
            pnl = (exit_price - self.entry_price) * self.position_size
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            pnl = (self.entry_price - exit_price) * self.position_size
            pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        hold_duration = (exit_time - self.entry_time).total_seconds() / 60
        
        trade = TradeResult(
            entry_time=self.entry_time,
            exit_time=exit_time,
            entry_price=self.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_pct,
            trade_type=self.position,
            entry_signal=self.entry_signal,
            exit_reason=exit_reason,
            hold_duration=hold_duration
        )
        
        # Reset position
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.entry_signal = None
        
        return trade


class BacktestEngine:
    """Core backtesting engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.signal_generator = PulseSignalGenerator(config)
        self.position_manager = PositionManager(config)
        self.trades: List[TradeResult] = []
        
    def run_backtest(self, data: pd.DataFrame) -> List[TradeResult]:
        """Run backtest on historical data"""
        self.trades = []
        
        # Generate signals
        df = self.signal_generator.generate_signals(data)
        
        # Iterate through data
        for idx in range(len(df)):
            row = df.iloc[idx]
            current_time = row.name if isinstance(row.name, datetime) else datetime.now()
            current_price = row['close']
            
            # Check if we should exit current position
            if self.position_manager.position is not None:
                should_exit, exit_reason = self.position_manager.should_exit(current_price, current_time)
                if should_exit:
                    trade = self.position_manager.exit_position(current_price, current_time, exit_reason)
                    self.trades.append(trade)
            
            # Check if we should enter a new position
            elif self.position_manager.can_enter(current_time):
                if row['signal_type'] in ['long', 'short']:
                    signal_info = {
                        'score': row['signal_score'],
                        'rsi': row['rsi'],
                        'bb_position': row['bb_position'],
                        'volume_ratio': row['volume_ratio']
                    }
                    self.position_manager.enter_position(
                        row['signal_type'],
                        current_price,
                        current_time,
                        signal_info
                    )
        
        # Close any remaining position at the end
        if self.position_manager.position is not None:
            last_row = df.iloc[-1]
            last_time = last_row.name if isinstance(last_row.name, datetime) else datetime.now()
            trade = self.position_manager.exit_position(last_row['close'], last_time, "backtest_end")
            self.trades.append(trade)
        
        return self.trades
    
    def calculate_metrics(self, initial_capital: float = 100000.0) -> PerformanceMetrics:
        """Calculate performance metrics from trades"""
        if not self.trades:
            return None
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        largest_win = max([t.pnl for t in wins]) if wins else 0
        largest_loss = min([t.pnl for t in losses]) if losses else 0
        
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio
        returns = [t.pnl_percent for t in self.trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum([t.pnl for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        avg_hold_time = np.mean([t.hold_duration for t in self.trades])
        total_return_pct = (total_pnl / initial_capital) * 100
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_hold_time=avg_hold_time,
            total_return_pct=total_return_pct
        )


class ThresholdOptimizer:
    """Optimize strategy thresholds to achieve target win rate"""
    
    def __init__(self, target_win_rate: float = 0.70, tolerance: float = 0.02):
        self.target_win_rate = target_win_rate
        self.tolerance = tolerance
        
    def optimize(self, data: pd.DataFrame, base_config: Dict, max_iterations: int = 50) -> Dict:
        """Optimize thresholds to converge on target win rate"""
        print(f"\n{'='*60}")
        print(f"THRESHOLD OPTIMIZATION - Target Win Rate: {self.target_win_rate:.1%}")
        print(f"{'='*60}\n")
        
        # Parameter ranges to optimize
        param_ranges = {
            'signal_threshold': (0.3, 0.8),
            'rsi_oversold': (20, 35),
            'rsi_overbought': (65, 80),
            'stop_loss_pct': (0.01, 0.03),
            'take_profit_pct': (0.02, 0.06),
        }
        
        best_config = base_config.copy()
        best_diff = float('inf')
        
        for iteration in range(max_iterations):
            # Try random variations
            test_config = best_config.copy()
            
            # Randomly adjust one parameter
            param_to_adjust = np.random.choice(list(param_ranges.keys()))
            min_val, max_val = param_ranges[param_to_adjust]
            test_config[param_to_adjust] = np.random.uniform(min_val, max_val)
            
            # Run backtest with test config
            engine = BacktestEngine(test_config)
            engine.run_backtest(data)
            metrics = engine.calculate_metrics()
            
            if metrics and metrics.total_trades > 10:
                diff = abs(metrics.win_rate - self.target_win_rate)
                
                if diff < best_diff:
                    best_diff = diff
                    best_config = test_config.copy()
                    
                    print(f"Iteration {iteration + 1:3d}: Win Rate = {metrics.win_rate:.2%}, "
                          f"Trades = {metrics.total_trades}, "
                          f"Profit Factor = {metrics.profit_factor:.2f}")
                    
                    # Check if we've converged
                    if diff < self.tolerance:
                        print(f"\n✓ Converged! Final win rate: {metrics.win_rate:.2%}")
                        break
        
        print(f"\nOptimization complete. Best win rate achieved: {metrics.win_rate:.2%}")
        print(f"Difference from target: {abs(metrics.win_rate - self.target_win_rate):.2%}\n")
        
        return best_config


class DataGenerator:
    """Generate synthetic market data for backtesting"""
    
    @staticmethod
    def generate_0dte_data(num_days: int = 252, 
                          minutes_per_day: int = 390,
                          base_price: float = 450.0,
                          volatility: float = 0.02) -> pd.DataFrame:
        """Generate synthetic 0DTE options market data"""
        print(f"\nGenerating {num_days} days of synthetic 0DTE market data...")
        
        all_data = []
        current_date = datetime.now() - timedelta(days=num_days)
        
        for day in range(num_days):
            # Market hours: 9:30 AM - 4:00 PM (390 minutes)
            day_start = current_date.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Generate intraday prices with mean reversion and volatility clustering
            prices = [base_price]
            for minute in range(minutes_per_day):
                # Mean reversion component
                mean_reversion = (base_price - prices[-1]) * 0.001
                
                # Volatility clustering
                vol = volatility * (1 + 0.5 * np.random.randn())
                
                # Random walk with drift
                change = mean_reversion + vol * np.random.randn()
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            prices = prices[1:]  # Remove initial price
            
            # Generate corresponding volumes
            volumes = np.random.lognormal(mean=12, sigma=1, size=minutes_per_day)
            
            # Create timestamps
            timestamps = [day_start + timedelta(minutes=i) for i in range(minutes_per_day)]
            
            # Create daily dataframe
            day_data = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            all_data.append(day_data)
            current_date += timedelta(days=1)
        
        df = pd.concat(all_data, ignore_index=True)
        df.set_index('timestamp', inplace=True)
        
        print(f"Generated {len(df):,} data points across {num_days} trading days")
        return df


class RLDataExporter:
    """Export training data for reinforcement learning"""
    
    @staticmethod
    def export_trades(trades: List[TradeResult], filename: str = "rl_training_data.json"):
        """Export trades in format suitable for RL training"""
        export_data = {
            'metadata': {
                'total_trades': len(trades),
                'export_time': datetime.now().isoformat(),
                'data_format': 'pulse_optimizer_v1'
            },
            'trades': [t.to_dict() for t in trades]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Exported {len(trades)} trades to {filename}")
    
    @staticmethod
    def export_features(data: pd.DataFrame, filename: str = "rl_features.csv"):
        """Export feature data for RL training"""
        data.to_csv(filename)
        print(f"✓ Exported {len(data)} feature rows to {filename}")


class PulseOptimizer:
    """Main optimizer class orchestrating all components"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
    @staticmethod
    def _default_config() -> Dict:
        """Default configuration"""
        return {
            # Signal generation
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_lower_threshold': 0.2,
            'bb_upper_threshold': 0.8,
            'volume_ma_period': 20,
            'volume_threshold': 1.2,
            'signal_threshold': 0.5,
            
            # Position management
            'position_size': 1.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'max_hold_minutes': 240,
            
            # Backtest settings
            'initial_capital': 100000.0,
        }
    
    def run_full_simulation(self, 
                           num_simulations: int = 100000,
                           num_days: int = 252,
                           optimize_thresholds: bool = True,
                           target_win_rate: float = 0.70,
                           export_rl_data: bool = True) -> Dict:
        """Run complete simulation with optimization"""
        
        print("\n" + "="*60)
        print(" PULSE TOOLS OPTIMIZER - 0DTE Strategy System")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Simulations: {num_simulations:,}")
        print(f"  Trading Days: {num_days}")
        print(f"  Optimize Thresholds: {optimize_thresholds}")
        print(f"  Target Win Rate: {target_win_rate:.1%}")
        print(f"  Export RL Data: {export_rl_data}")
        
        # Generate synthetic market data
        data = DataGenerator.generate_0dte_data(num_days=num_days)
        
        # Optimize thresholds if requested
        if optimize_thresholds:
            optimizer = ThresholdOptimizer(target_win_rate=target_win_rate)
            self.config = optimizer.optimize(data, self.config)
        
        # Run large-scale simulation
        print(f"\n{'='*60}")
        print(f"RUNNING {num_simulations:,} SIMULATIONS")
        print(f"{'='*60}\n")
        
        all_metrics = []
        all_trades = []
        
        # Run simulations in batches
        batch_size = 1000
        num_batches = num_simulations // batch_size
        
        for batch in range(num_batches):
            batch_metrics = []
            
            for sim in range(batch_size):
                # Add small random variations to config for robustness testing
                sim_config = self.config.copy()
                sim_config['signal_threshold'] *= np.random.uniform(0.95, 1.05)
                sim_config['stop_loss_pct'] *= np.random.uniform(0.9, 1.1)
                sim_config['take_profit_pct'] *= np.random.uniform(0.9, 1.1)
                
                # Run backtest
                engine = BacktestEngine(sim_config)
                trades = engine.run_backtest(data)
                metrics = engine.calculate_metrics()
                
                if metrics:
                    batch_metrics.append(metrics)
                    if len(all_trades) < 10000:  # Store first 10k trades for analysis
                        all_trades.extend(trades)
            
            all_metrics.extend(batch_metrics)
            
            # Progress update
            completed = (batch + 1) * batch_size
            if completed % 10000 == 0:
                avg_wr = np.mean([m.win_rate for m in all_metrics[-batch_size:]])
                avg_pf = np.mean([m.profit_factor for m in all_metrics[-batch_size:]])
                print(f"  Completed {completed:,} simulations | "
                      f"Avg Win Rate: {avg_wr:.2%} | "
                      f"Avg Profit Factor: {avg_pf:.2f}")
        
        # Aggregate results
        results = self._aggregate_results(all_metrics)
        
        # Export RL training data if requested
        if export_rl_data and all_trades:
            RLDataExporter.export_trades(all_trades)
            signal_generator = PulseSignalGenerator(self.config)
            featured_data = signal_generator.generate_signals(data)
            RLDataExporter.export_features(featured_data)
        
        # Print final results
        self._print_results(results)
        
        return {
            'config': self.config,
            'results': results,
            'sample_trades': all_trades[:100]  # Return sample of trades
        }
    
    def _aggregate_results(self, metrics_list: List[PerformanceMetrics]) -> Dict:
        """Aggregate results from multiple simulations"""
        if not metrics_list:
            return {}
        
        return {
            'num_simulations': len(metrics_list),
            'win_rate': {
                'mean': np.mean([m.win_rate for m in metrics_list]),
                'std': np.std([m.win_rate for m in metrics_list]),
                'min': np.min([m.win_rate for m in metrics_list]),
                'max': np.max([m.win_rate for m in metrics_list]),
                'median': np.median([m.win_rate for m in metrics_list])
            },
            'profit_factor': {
                'mean': np.mean([m.profit_factor for m in metrics_list if m.profit_factor != float('inf')]),
                'std': np.std([m.profit_factor for m in metrics_list if m.profit_factor != float('inf')]),
                'min': np.min([m.profit_factor for m in metrics_list if m.profit_factor != float('inf')]),
                'max': np.max([m.profit_factor for m in metrics_list if m.profit_factor != float('inf')]),
                'median': np.median([m.profit_factor for m in metrics_list if m.profit_factor != float('inf')])
            },
            'total_return_pct': {
                'mean': np.mean([m.total_return_pct for m in metrics_list]),
                'std': np.std([m.total_return_pct for m in metrics_list]),
                'min': np.min([m.total_return_pct for m in metrics_list]),
                'max': np.max([m.total_return_pct for m in metrics_list]),
                'median': np.median([m.total_return_pct for m in metrics_list])
            },
            'sharpe_ratio': {
                'mean': np.mean([m.sharpe_ratio for m in metrics_list]),
                'std': np.std([m.sharpe_ratio for m in metrics_list]),
                'min': np.min([m.sharpe_ratio for m in metrics_list]),
                'max': np.max([m.sharpe_ratio for m in metrics_list]),
                'median': np.median([m.sharpe_ratio for m in metrics_list])
            },
            'avg_trades_per_sim': np.mean([m.total_trades for m in metrics_list]),
            'max_drawdown': {
                'mean': np.mean([m.max_drawdown for m in metrics_list]),
                'worst': np.min([m.max_drawdown for m in metrics_list])
            }
        }
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print(" FINAL RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Total Simulations: {results['num_simulations']:,}\n")
        
        print("Win Rate Statistics:")
        print(f"  Mean:   {results['win_rate']['mean']:.2%}")
        print(f"  Median: {results['win_rate']['median']:.2%}")
        print(f"  Std:    {results['win_rate']['std']:.2%}")
        print(f"  Range:  {results['win_rate']['min']:.2%} - {results['win_rate']['max']:.2%}\n")
        
        print("Profit Factor Statistics:")
        print(f"  Mean:   {results['profit_factor']['mean']:.2f}")
        print(f"  Median: {results['profit_factor']['median']:.2f}")
        print(f"  Std:    {results['profit_factor']['std']:.2f}")
        print(f"  Range:  {results['profit_factor']['min']:.2f} - {results['profit_factor']['max']:.2f}\n")
        
        print("Return Statistics:")
        print(f"  Mean:   {results['total_return_pct']['mean']:.2f}%")
        print(f"  Median: {results['total_return_pct']['median']:.2f}%")
        print(f"  Std:    {results['total_return_pct']['std']:.2f}%")
        print(f"  Range:  {results['total_return_pct']['min']:.2f}% - {results['total_return_pct']['max']:.2f}%\n")
        
        print("Sharpe Ratio Statistics:")
        print(f"  Mean:   {results['sharpe_ratio']['mean']:.2f}")
        print(f"  Median: {results['sharpe_ratio']['median']:.2f}")
        print(f"  Range:  {results['sharpe_ratio']['min']:.2f} - {results['sharpe_ratio']['max']:.2f}\n")
        
        print(f"Average Trades per Simulation: {results['avg_trades_per_sim']:.1f}")
        print(f"Average Max Drawdown: ${results['max_drawdown']['mean']:,.2f}")
        print(f"Worst Max Drawdown: ${results['max_drawdown']['worst']:,.2f}")


def main():
    """Main entry point"""
    # Initialize optimizer with custom config (optional)
    optimizer = PulseOptimizer()
    
    # Run full simulation
    results = optimizer.run_full_simulation(
        num_simulations=100000,  # 100K simulations
        num_days=252,  # 1 year of trading days
        optimize_thresholds=True,  # Enable threshold optimization
        target_win_rate=0.70,  # Target 70% win rate
        export_rl_data=True  # Export data for RL training
    )
    
    # Save configuration
    with open('optimized_config.json', 'w') as f:
        json.dump(results['config'], f, indent=2)
    
    print("\n✓ Optimization complete!")
    print("✓ Configuration saved to optimized_config.json")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
