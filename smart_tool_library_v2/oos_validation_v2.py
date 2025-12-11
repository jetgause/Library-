"""
PULSE Out-of-Sample Validation v2.0

A comprehensive framework for validating trading strategies with temporal structure,
bootstrap confidence intervals, walk-forward analysis, and statistical edge verification.

Author: PULSE Trading System
Version: 2.0
Date: 2025-12-11
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import warnings


# ============================================================================
# Configuration and Status Classes
# ============================================================================

@dataclass
class OOSConfig:
    """Configuration for Out-of-Sample validation."""
    
    train_samples: int = 50000
    test_samples: int = 20000
    min_edge_threshold: float = 0.02
    confidence_level: float = 0.95
    bootstrap_iterations: int = 10000
    walk_forward_windows: int = 5
    min_trades_per_window: int = 100
    max_drawdown_threshold: float = 0.25
    sharpe_threshold: float = 1.0
    temporal_correlation_max: float = 0.3
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.train_samples <= 0 or self.test_samples <= 0:
            raise ValueError("Sample sizes must be positive")
        if not 0 < self.min_edge_threshold < 1:
            raise ValueError("Edge threshold must be between 0 and 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")


class ValidationStatus(Enum):
    """Status outcomes for validation process."""
    
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"
    OVERFITTING = "overfitting"
    WARNING = "warning"
    
    def __str__(self):
        return self.value


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class EdgeMetrics:
    """Metrics for statistical edge computation."""
    
    mean_edge: float
    std_edge: float
    confidence_lower: float
    confidence_upper: float
    t_statistic: float
    p_value: float
    sample_size: int
    sharpe_ratio: float
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if edge is statistically significant."""
        return self.p_value < alpha and self.confidence_lower > 0


@dataclass
class ValidationResult:
    """Complete validation result."""
    
    status: ValidationStatus
    train_metrics: EdgeMetrics
    test_metrics: EdgeMetrics
    overfitting_ratio: float
    walk_forward_results: List[Dict[str, Any]]
    bootstrap_edges: np.ndarray
    temporal_stability: float
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Validation Status: {self.status}",
            f"Train Edge: {self.train_metrics.mean_edge:.4f} ± {self.train_metrics.std_edge:.4f}",
            f"Test Edge: {self.test_metrics.mean_edge:.4f} ± {self.test_metrics.std_edge:.4f}",
            f"Overfitting Ratio: {self.overfitting_ratio:.4f}",
            f"Temporal Stability: {self.temporal_stability:.4f}",
            f"Train Sharpe: {self.train_metrics.sharpe_ratio:.2f}",
            f"Test Sharpe: {self.test_metrics.sharpe_ratio:.2f}",
        ]
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)


# ============================================================================
# Real Edge Computation
# ============================================================================

class RealEdgeComputer:
    """
    Computes real backtested edge from trading results.
    
    The edge represents the expected profit per trade after accounting for
    transaction costs, slippage, and realistic market conditions.
    """
    
    def __init__(self, transaction_cost: float = 0.001, slippage: float = 0.0005):
        """
        Initialize edge computer.
        
        Args:
            transaction_cost: Cost per trade as fraction (e.g., 0.001 = 0.1%)
            slippage: Expected slippage as fraction
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.total_cost = transaction_cost + slippage
    
    def compute(
        self,
        returns: np.ndarray,
        positions: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute real edge from trading returns.
        
        Args:
            returns: Array of trade returns
            positions: Optional position sizes (default: all 1.0)
            prices: Optional price data for additional analysis
        
        Returns:
            Tuple of (mean_edge, edge_per_trade)
        """
        if len(returns) == 0:
            return 0.0, np.array([])
        
        # Apply transaction costs
        if positions is None:
            positions = np.ones(len(returns))
        
        # Calculate net returns after costs
        gross_returns = returns * positions
        costs = np.abs(positions) * self.total_cost
        net_returns = gross_returns - costs
        
        # Compute edge
        mean_edge = np.mean(net_returns)
        
        return mean_edge, net_returns
    
    def compute_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive edge metrics.
        
        Args:
            returns: Array of trade returns
        
        Returns:
            Dictionary of metrics
        """
        mean_edge, net_returns = self.compute(returns)
        
        if len(net_returns) == 0:
            return {
                'mean_edge': 0.0,
                'std_edge': 0.0,
                'sharpe': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }
        
        # Basic statistics
        std_edge = np.std(net_returns, ddof=1) if len(net_returns) > 1 else 0.0
        sharpe = mean_edge / std_edge if std_edge > 0 else 0.0
        
        # Win rate
        win_rate = np.mean(net_returns > 0)
        
        # Profit factor
        profits = net_returns[net_returns > 0]
        losses = -net_returns[net_returns < 0]
        profit_factor = (np.sum(profits) / np.sum(losses) 
                        if len(losses) > 0 and np.sum(losses) > 0 else 0.0)
        
        # Maximum drawdown
        cumulative = np.cumsum(net_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'mean_edge': mean_edge,
            'std_edge': std_edge,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }


# ============================================================================
# Temporal Market Data Generator
# ============================================================================

class TemporalMarketGenerator:
    """
    Generates synthetic market data with realistic temporal structure.
    
    This includes:
    - Autocorrelation in returns
    - Volatility clustering
    - Regime changes
    - Fat-tailed distributions
    """
    
    def __init__(
        self,
        volatility: float = 0.02,
        autocorr: float = 0.05,
        regime_change_prob: float = 0.01,
        fat_tail_df: int = 5,
        seed: Optional[int] = None
    ):
        """
        Initialize market generator.
        
        Args:
            volatility: Base volatility level
            autocorr: Autocorrelation coefficient for returns
            regime_change_prob: Probability of regime change per period
            fat_tail_df: Degrees of freedom for Student's t distribution
            seed: Random seed for reproducibility
        """
        self.volatility = volatility
        self.autocorr = autocorr
        self.regime_change_prob = regime_change_prob
        self.fat_tail_df = fat_tail_df
        self.rng = np.random.RandomState(seed)
        self.current_regime = 1.0
    
    def generate(
        self,
        n_samples: int,
        start_price: float = 100.0,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate temporal market data.
        
        Args:
            n_samples: Number of data points to generate
            start_price: Starting price level
            start_date: Starting date for index
        
        Returns:
            DataFrame with columns: timestamp, price, returns, volume
        """
        # Initialize
        prices = np.zeros(n_samples)
        returns = np.zeros(n_samples)
        volumes = np.zeros(n_samples)
        prices[0] = start_price
        
        # Generate with temporal structure
        for i in range(1, n_samples):
            # Regime changes
            if self.rng.random() < self.regime_change_prob:
                self.current_regime = self.rng.choice([0.5, 1.0, 1.5, 2.0])
            
            # Autocorrelated returns with fat tails
            innovation = self.rng.standard_t(self.fat_tail_df)
            returns[i] = (self.autocorr * returns[i-1] + 
                         self.volatility * self.current_regime * innovation)
            
            # Update price
            prices[i] = prices[i-1] * (1 + returns[i])
            
            # Volume with correlation to volatility
            volumes[i] = (1000000 * (1 + abs(returns[i]) * 10) * 
                         self.rng.lognormal(0, 0.3))
        
        # Create DataFrame
        if start_date is None:
            start_date = datetime(2020, 1, 1)
        
        timestamps = [start_date + timedelta(minutes=i) for i in range(n_samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'returns': returns,
            'volume': volumes
        })
        
        return df
    
    def add_microstructure_noise(self, df: pd.DataFrame, noise_level: float = 0.0001) -> pd.DataFrame:
        """Add market microstructure noise to prices."""
        df = df.copy()
        noise = self.rng.normal(0, noise_level, len(df))
        df['price'] = df['price'] * (1 + noise)
        df['returns'] = df['price'].pct_change()
        return df


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

class BootstrapAnalyzer:
    """Compute bootstrap confidence intervals for edge estimates."""
    
    def __init__(self, n_iterations: int = 10000, seed: Optional[int] = None):
        """
        Initialize bootstrap analyzer.
        
        Args:
            n_iterations: Number of bootstrap iterations
            seed: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.rng = np.random.RandomState(seed)
    
    def compute_confidence_interval(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = 'percentile'
    ) -> Tuple[float, float, np.ndarray]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'percentile' or 'bca' (bias-corrected and accelerated)
        
        Returns:
            Tuple of (lower_bound, upper_bound, bootstrap_samples)
        """
        n = len(returns)
        bootstrap_means = np.zeros(self.n_iterations)
        
        # Generate bootstrap samples
        for i in range(self.n_iterations):
            sample = self.rng.choice(returns, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        
        if method == 'percentile':
            lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        elif method == 'bca':
            # Bias-corrected and accelerated
            lower, upper = self._bca_interval(returns, bootstrap_means, confidence_level)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return lower, upper, bootstrap_means
    
    def _bca_interval(
        self,
        returns: np.ndarray,
        bootstrap_means: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Compute bias-corrected and accelerated (BCa) interval."""
        # Original statistic
        theta_hat = np.mean(returns)
        
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_means < theta_hat))
        
        # Acceleration (jackknife)
        n = len(returns)
        jackknife_means = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(returns, i)
            jackknife_means[i] = np.mean(jackknife_sample)
        
        mean_jackknife = np.mean(jackknife_means)
        numerator = np.sum((mean_jackknife - jackknife_means) ** 3)
        denominator = 6 * (np.sum((mean_jackknife - jackknife_means) ** 2) ** 1.5)
        a = numerator / denominator if denominator != 0 else 0
        
        # Adjusted percentiles
        alpha = 1 - confidence_level
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1alpha = stats.norm.ppf(1 - alpha / 2)
        
        lower_percentile = 100 * stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        upper_percentile = 100 * stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))
        
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)
        
        return lower, upper


# ============================================================================
# Walk-Forward Validation
# ============================================================================

class WalkForwardValidator:
    """
    Perform walk-forward validation to assess temporal stability.
    
    This splits the data into multiple sequential windows and validates
    that the edge persists across time periods.
    """
    
    def __init__(self, n_windows: int = 5, train_ratio: float = 0.7):
        """
        Initialize walk-forward validator.
        
        Args:
            n_windows: Number of walk-forward windows
            train_ratio: Ratio of data used for training in each window
        """
        self.n_windows = n_windows
        self.train_ratio = train_ratio
    
    def validate(
        self,
        returns: np.ndarray,
        edge_computer: RealEdgeComputer
    ) -> List[Dict[str, Any]]:
        """
        Perform walk-forward validation.
        
        Args:
            returns: Full array of returns
            edge_computer: Edge computer instance
        
        Returns:
            List of results for each window
        """
        n_total = len(returns)
        window_size = n_total // self.n_windows
        results = []
        
        for i in range(self.n_windows):
            # Define window
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, n_total)
            window_returns = returns[start_idx:end_idx]
            
            # Split into train/test
            n_train = int(len(window_returns) * self.train_ratio)
            train_returns = window_returns[:n_train]
            test_returns = window_returns[n_train:]
            
            # Compute metrics
            train_edge, _ = edge_computer.compute(train_returns)
            test_edge, _ = edge_computer.compute(test_returns)
            train_metrics = edge_computer.compute_metrics(train_returns)
            test_metrics = edge_computer.compute_metrics(test_returns)
            
            results.append({
                'window': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'train_edge': train_edge,
                'test_edge': test_edge,
                'train_sharpe': train_metrics['sharpe'],
                'test_sharpe': test_metrics['sharpe'],
                'overfitting_ratio': test_edge / train_edge if train_edge != 0 else 0.0,
                'train_samples': len(train_returns),
                'test_samples': len(test_returns)
            })
        
        return results
    
    def compute_stability_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute temporal stability score from walk-forward results.
        
        Args:
            results: Walk-forward results
        
        Returns:
            Stability score (0 to 1, higher is better)
        """
        if not results:
            return 0.0
        
        test_edges = [r['test_edge'] for r in results]
        
        # Consistency: how many windows have positive edge
        positive_ratio = np.mean([e > 0 for e in test_edges])
        
        # Stability: coefficient of variation (lower is better)
        mean_edge = np.mean(test_edges)
        std_edge = np.std(test_edges)
        cv = std_edge / abs(mean_edge) if mean_edge != 0 else float('inf')
        stability = 1 / (1 + cv)  # Convert to 0-1 scale
        
        # Combined score
        score = 0.6 * positive_ratio + 0.4 * stability
        
        return score


# ============================================================================
# Main Validation Framework
# ============================================================================

class OOSValidator:
    """
    Main Out-of-Sample Validation Framework.
    
    Orchestrates all validation components to provide comprehensive
    strategy validation with statistical rigor.
    """
    
    def __init__(self, config: Optional[OOSConfig] = None):
        """
        Initialize validator.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or OOSConfig()
        self.edge_computer = RealEdgeComputer()
        self.bootstrap_analyzer = BootstrapAnalyzer(
            n_iterations=self.config.bootstrap_iterations
        )
        self.walk_forward = WalkForwardValidator(
            n_windows=self.config.walk_forward_windows
        )
    
    def validate(
        self,
        returns: np.ndarray,
        train_ratio: float = 0.7
    ) -> ValidationResult:
        """
        Perform complete validation.
        
        Args:
            returns: Array of trade returns
            train_ratio: Ratio of data for training (rest for testing)
        
        Returns:
            Complete validation result
        """
        warnings_list = []
        
        # Check data sufficiency
        required_samples = self.config.train_samples + self.config.test_samples
        if len(returns) < required_samples:
            return ValidationResult(
                status=ValidationStatus.INSUFFICIENT_DATA,
                train_metrics=self._empty_metrics(),
                test_metrics=self._empty_metrics(),
                overfitting_ratio=0.0,
                walk_forward_results=[],
                bootstrap_edges=np.array([]),
                temporal_stability=0.0,
                warnings=[f"Insufficient data: {len(returns)} < {required_samples}"]
            )
        
        # Split data
        n_train = int(len(returns) * train_ratio)
        train_returns = returns[:n_train]
        test_returns = returns[n_train:]
        
        # Compute train metrics
        train_metrics = self._compute_edge_metrics(train_returns, "train")
        
        # Compute test metrics
        test_metrics = self._compute_edge_metrics(test_returns, "test")
        
        # Bootstrap confidence intervals
        _, _, bootstrap_edges = self.bootstrap_analyzer.compute_confidence_interval(
            test_returns,
            confidence_level=self.config.confidence_level
        )
        
        # Walk-forward validation
        wf_results = self.walk_forward.validate(returns, self.edge_computer)
        temporal_stability = self.walk_forward.compute_stability_score(wf_results)
        
        # Compute overfitting ratio
        overfitting_ratio = (test_metrics.mean_edge / train_metrics.mean_edge 
                            if train_metrics.mean_edge != 0 else 0.0)
        
        # Determine validation status
        status = self._determine_status(
            train_metrics,
            test_metrics,
            overfitting_ratio,
            temporal_stability,
            warnings_list
        )
        
        return ValidationResult(
            status=status,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            overfitting_ratio=overfitting_ratio,
            walk_forward_results=wf_results,
            bootstrap_edges=bootstrap_edges,
            temporal_stability=temporal_stability,
            warnings=warnings_list,
            metadata={
                'total_samples': len(returns),
                'train_samples': len(train_returns),
                'test_samples': len(test_returns),
                'config': self.config
            }
        )
    
    def _compute_edge_metrics(
        self,
        returns: np.ndarray,
        dataset_name: str
    ) -> EdgeMetrics:
        """Compute comprehensive edge metrics for a dataset."""
        mean_edge, net_returns = self.edge_computer.compute(returns)
        
        if len(net_returns) == 0:
            return self._empty_metrics()
        
        # Statistical measures
        std_edge = np.std(net_returns, ddof=1) if len(net_returns) > 1 else 0.0
        n = len(net_returns)
        
        # t-test
        if std_edge > 0 and n > 1:
            t_stat = mean_edge / (std_edge / np.sqrt(n))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        else:
            t_stat = 0.0
            p_value = 1.0
        
        # Bootstrap confidence interval
        ci_lower, ci_upper, _ = self.bootstrap_analyzer.compute_confidence_interval(
            net_returns,
            confidence_level=self.config.confidence_level
        )
        
        # Sharpe ratio
        sharpe = mean_edge / std_edge if std_edge > 0 else 0.0
        
        return EdgeMetrics(
            mean_edge=mean_edge,
            std_edge=std_edge,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            t_statistic=t_stat,
            p_value=p_value,
            sample_size=n,
            sharpe_ratio=sharpe
        )
    
    def _empty_metrics(self) -> EdgeMetrics:
        """Return empty metrics for insufficient data."""
        return EdgeMetrics(
            mean_edge=0.0,
            std_edge=0.0,
            confidence_lower=0.0,
            confidence_upper=0.0,
            t_statistic=0.0,
            p_value=1.0,
            sample_size=0,
            sharpe_ratio=0.0
        )
    
    def _determine_status(
        self,
        train_metrics: EdgeMetrics,
        test_metrics: EdgeMetrics,
        overfitting_ratio: float,
        temporal_stability: float,
        warnings_list: List[str]
    ) -> ValidationStatus:
        """Determine final validation status."""
        
        # Check test edge significance
        if not test_metrics.is_significant():
            warnings_list.append("Test edge is not statistically significant")
            return ValidationStatus.FAILED
        
        # Check edge threshold
        if test_metrics.mean_edge < self.config.min_edge_threshold:
            warnings_list.append(
                f"Test edge {test_metrics.mean_edge:.4f} below threshold "
                f"{self.config.min_edge_threshold:.4f}"
            )
            return ValidationStatus.FAILED
        
        # Check Sharpe ratio
        if test_metrics.sharpe_ratio < self.config.sharpe_threshold:
            warnings_list.append(
                f"Test Sharpe ratio {test_metrics.sharpe_ratio:.2f} below threshold "
                f"{self.config.sharpe_threshold:.2f}"
            )
        
        # Check overfitting
        if overfitting_ratio < 0.7:
            warnings_list.append(
                f"Significant overfitting detected: ratio {overfitting_ratio:.2f}"
            )
            return ValidationStatus.OVERFITTING
        
        # Check temporal stability
        if temporal_stability < 0.6:
            warnings_list.append(
                f"Low temporal stability: {temporal_stability:.2f}"
            )
        
        # Passed all critical checks
        if warnings_list:
            return ValidationStatus.WARNING
        
        return ValidationStatus.PASSED
    
    def validate_with_market_data(
        self,
        strategy_returns: np.ndarray,
        market_data: pd.DataFrame,
        check_correlation: bool = True
    ) -> ValidationResult:
        """
        Validate with market data context.
        
        Args:
            strategy_returns: Strategy returns
            market_data: Market data DataFrame with 'returns' column
            check_correlation: Whether to check temporal correlation
        
        Returns:
            Validation result with market context
        """
        result = self.validate(strategy_returns)
        
        # Check temporal correlation with market
        if check_correlation and 'returns' in market_data.columns:
            n = min(len(strategy_returns), len(market_data))
            corr = np.corrcoef(
                strategy_returns[:n],
                market_data['returns'].values[:n]
            )[0, 1]
            
            result.metadata['market_correlation'] = corr
            
            if abs(corr) > self.config.temporal_correlation_max:
                result.warnings.append(
                    f"High market correlation detected: {corr:.2f}"
                )
        
        return result


# ============================================================================
# Utility Functions
# ============================================================================

def generate_sample_returns(
    n_samples: int,
    edge: float = 0.02,
    volatility: float = 0.05,
    autocorr: float = 0.05,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate sample returns for testing.
    
    Args:
        n_samples: Number of samples
        edge: Expected edge per trade
        volatility: Return volatility
        autocorr: Autocorrelation coefficient
        seed: Random seed
    
    Returns:
        Array of returns
    """
    rng = np.random.RandomState(seed)
    returns = np.zeros(n_samples)
    
    for i in range(n_samples):
        innovation = rng.normal(edge, volatility)
        if i > 0:
            returns[i] = autocorr * returns[i-1] + innovation
        else:
            returns[i] = innovation
    
    return returns


def plot_validation_results(result: ValidationResult, save_path: Optional[str] = None):
    """
    Plot validation results (requires matplotlib).
    
    Args:
        result: Validation result to plot
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bootstrap distribution
    axes[0, 0].hist(result.bootstrap_edges, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(result.test_metrics.mean_edge, color='red', 
                       linestyle='--', label='Mean Edge')
    axes[0, 0].axvline(result.test_metrics.confidence_lower, color='green',
                       linestyle=':', label='95% CI')
    axes[0, 0].axvline(result.test_metrics.confidence_upper, color='green',
                       linestyle=':')
    axes[0, 0].set_xlabel('Edge')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Bootstrap Edge Distribution')
    axes[0, 0].legend()
    
    # Walk-forward results
    if result.walk_forward_results:
        windows = [r['window'] for r in result.walk_forward_results]
        train_edges = [r['train_edge'] for r in result.walk_forward_results]
        test_edges = [r['test_edge'] for r in result.walk_forward_results]
        
        axes[0, 1].plot(windows, train_edges, 'o-', label='Train Edge')
        axes[0, 1].plot(windows, test_edges, 's-', label='Test Edge')
        axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('Window')
        axes[0, 1].set_ylabel('Edge')
        axes[0, 1].set_title('Walk-Forward Edge Stability')
        axes[0, 1].legend()
    
    # Train vs Test comparison
    categories = ['Edge', 'Sharpe', 'Std']
    train_values = [
        result.train_metrics.mean_edge,
        result.train_metrics.sharpe_ratio,
        result.train_metrics.std_edge
    ]
    test_values = [
        result.test_metrics.mean_edge,
        result.test_metrics.sharpe_ratio,
        result.test_metrics.std_edge
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    axes[1, 0].bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    axes[1, 0].bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Train vs Test Metrics')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()
    
    # Summary text
    summary_text = result.summary()
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Validation Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("PULSE Out-of-Sample Validation v2.0")
    print("=" * 60)
    
    # Configuration
    config = OOSConfig(
        train_samples=50000,
        test_samples=20000,
        min_edge_threshold=0.02,
        confidence_level=0.95,
        bootstrap_iterations=10000,
        walk_forward_windows=5
    )
    
    # Generate sample data
    print("\nGenerating sample market data...")
    market_gen = TemporalMarketGenerator(
        volatility=0.02,
        autocorr=0.05,
        regime_change_prob=0.01,
        seed=42
    )
    
    market_data = market_gen.generate(
        n_samples=70000,
        start_price=100.0
    )
    
    # Generate strategy returns with edge
    print("Generating strategy returns with edge...")
    strategy_returns = generate_sample_returns(
        n_samples=70000,
        edge=0.025,
        volatility=0.05,
        autocorr=0.05,
        seed=123
    )
    
    # Perform validation
    print("\nPerforming out-of-sample validation...")
    validator = OOSValidator(config)
    result = validator.validate_with_market_data(
        strategy_returns,
        market_data,
        check_correlation=True
    )
    
    # Display results
    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)
    
    # Additional details
    print("\nWalk-Forward Analysis:")
    for wf_result in result.walk_forward_results:
        print(f"  Window {wf_result['window']}: "
              f"Train={wf_result['train_edge']:.4f}, "
              f"Test={wf_result['test_edge']:.4f}, "
              f"Ratio={wf_result['overfitting_ratio']:.2f}")
    
    print(f"\nMarket Correlation: {result.metadata.get('market_correlation', 'N/A'):.4f}")
    
    # Optional: Plot results (requires matplotlib)
    # plot_validation_results(result)
    
    print("\nValidation complete!")
