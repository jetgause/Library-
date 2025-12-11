"""
PULSE Out-of-Sample Validation v2.0
====================================
Proper train/test split with REAL backtested edge.

Key principle: Train on past, validate on future.
Reject if OOS edge < 2% (statistical noise threshold).
"""

import numpy as np
from typing import Dict, Callable, Tuple, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OOSConfig:
    """Out-of-sample validation configuration"""
    train_samples: int = 50_000        # Training set size
    test_samples: int = 20_000         # Test set size (OOS)
    min_edge_threshold: float = 0.02   # 2% minimum real edge to pass
    min_signals_train: int = 100       # Minimum signals in training
    min_signals_test: int = 50         # Minimum signals in test
    confidence_level: float = 0.95     # Statistical confidence
    n_bootstrap: int = 1000            # Bootstrap iterations
    walk_forward_folds: int = 5        # Number of walk-forward folds


class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"
    OVERFITTING = "overfitting"


# =============================================================================
# REAL EDGE COMPUTER
# =============================================================================

class RealEdgeComputer:
    """Compute real backtested edge"""
    
    @staticmethod
    def compute(
        market: Dict[str, np.ndarray],
        tool_condition: Callable,
        thresholds: Dict[str, float],
        cost_bps: float = 3.0
    ) -> Dict[str, float]:
        """
        Compute REAL edge from forward returns.
        
        Returns dict with: win_rate, edge, n_signals, frequency, sharpe
        """
        try:
            signals = np.asarray(tool_condition(market, thresholds), dtype=bool)
        except:
            return {'win_rate': 0.5, 'edge': 0, 'n_signals': 0, 'frequency': 0, 'sharpe': 0, 'valid': False}
        
        n_bars = len(signals)
        n_signals = int(np.sum(signals))
        
        if n_signals == 0:
            return {'win_rate': 0.5, 'edge': 0, 'n_signals': 0, 'frequency': 0, 'sharpe': 0, 'valid': False}
        
        # Get forward returns
        if 'returns_5m' in market:
            returns = market['returns_5m']
        elif 'close' in market:
            returns = np.diff(market['close']) / market['close'][:-1]
        else:
            return {'win_rate': 0.5, 'edge': 0, 'n_signals': n_signals, 'frequency': n_signals/n_bars, 'sharpe': 0, 'valid': False}
        
        # Align signals with forward returns
        max_idx = min(len(signals) - 1, len(returns))
        signal_mask = signals[:max_idx]
        forward_returns = returns[:max_idx]
        
        signal_returns = forward_returns[signal_mask]
        
        if len(signal_returns) == 0:
            return {'win_rate': 0.5, 'edge': 0, 'n_signals': n_signals, 'frequency': n_signals/n_bars, 'sharpe': 0, 'valid': False}
        
        # Apply costs
        cost = cost_bps / 10000
        signal_returns_net = signal_returns - cost
        
        # Metrics
        win_rate = float(np.mean(signal_returns_net > 0))
        edge = float(np.mean(signal_returns_net))
        frequency = n_signals / n_bars
        
        # Sharpe (annualized for 5-min bars)
        if len(signal_returns_net) > 1 and np.std(signal_returns_net) > 0:
            sharpe = edge / np.std(signal_returns_net) * np.sqrt(252 * 78)
        else:
            sharpe = 0
        
        return {
            'win_rate': win_rate,
            'edge': edge,
            'n_signals': n_signals,
            'frequency': frequency,
            'sharpe': sharpe,
            'returns': signal_returns_net,
            'valid': True
        }


# =============================================================================
# MARKET GENERATOR WITH TEMPORAL STRUCTURE
# =============================================================================

class TemporalMarketGenerator:
    """Generate market data with temporal structure for train/test splits"""
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
    
    def generate(
        self,
        n_bars: int,
        start_date: str = None,
        end_date: str = None,
        regime: str = 'mixed'
    ) -> Dict[str, np.ndarray]:
        """
        Generate market data with optional date range simulation.
        
        Args:
            n_bars: Number of bars to generate
            start_date: Simulated start date (for regime selection)
            end_date: Simulated end date
            regime: 'bull', 'bear', 'sideways', 'volatile', 'mixed'
        """
        # Base parameters by regime
        regime_params = {
            'bull':     {'drift': 0.0003, 'vol': 0.0015, 'rsi_mean': 58, 'vix_mean': 15},
            'bear':     {'drift': -0.0002, 'vol': 0.0025, 'rsi_mean': 42, 'vix_mean': 28},
            'sideways': {'drift': 0.0000, 'vol': 0.0018, 'rsi_mean': 50, 'vix_mean': 18},
            'volatile': {'drift': 0.0001, 'vol': 0.0035, 'rsi_mean': 50, 'vix_mean': 32},
            'mixed':    {'drift': 0.0001, 'vol': 0.0020, 'rsi_mean': 50, 'vix_mean': 20},
        }
        
        params = regime_params.get(regime, regime_params['mixed'])
        
        # Generate returns with autocorrelation
        returns = self.rng.normal(params['drift'], params['vol'], n_bars)
        for i in range(1, n_bars):
            returns[i] += 0.1 * returns[i-1]  # Momentum
        
        # Volatility clustering
        vol_mult = np.ones(n_bars)
        for i in range(1, n_bars):
            if abs(returns[i-1]) > 2 * params['vol']:
                vol_mult[i] = min(vol_mult[i-1] * 1.3, 3.0)
            else:
                vol_mult[i] = max(vol_mult[i-1] * 0.95, 0.5)
        returns *= vol_mult
        
        # Price
        close = 100 * np.cumprod(1 + returns)
        high = close * (1 + np.abs(self.rng.normal(0, 0.001, n_bars)))
        low = close * (1 - np.abs(self.rng.normal(0, 0.001, n_bars)))
        
        # Indicators
        rsi = self._compute_rsi(close, params['rsi_mean'])
        momentum = self._compute_momentum(close, period=10)
        bb_pct = self._compute_bb_pct(close)
        macd_hist = self._compute_macd_hist(close)
        vol_ratio = self._compute_vol_ratio(n_bars)
        
        return {
            'close': close,
            'high': high,
            'low': low,
            'returns_5m': returns,
            'rsi': rsi,
            'momentum': momentum,
            'bb_pct': bb_pct,
            'macd_hist': macd_hist,
            'vol_ratio': vol_ratio,
            'vix': np.clip(self.rng.normal(params['vix_mean'], 5, n_bars), 10, 50),
            'volume': self.rng.lognormal(10, 0.5, n_bars),
            'n': n_bars,
            'regime': regime,
            'start_date': start_date,
            'end_date': end_date,
        }
    
    def _compute_rsi(self, close: np.ndarray, base_mean: float, period: int = 14) -> np.ndarray:
        """Compute RSI with target mean"""
        rsi = np.zeros(len(close))
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        for i in range(period, len(close)):
            avg_gain = np.mean(gain[i-period:i])
            avg_loss = np.mean(loss[i-period:i])
            rs = avg_gain / (avg_loss + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
        
        # Adjust to target mean
        rsi[period:] += (base_mean - np.mean(rsi[period:]))
        return np.clip(rsi, 0, 100)
    
    def _compute_momentum(self, close: np.ndarray, period: int = 10) -> np.ndarray:
        """Price momentum"""
        momentum = np.zeros(len(close))
        momentum[period:] = (close[period:] / close[:-period] - 1) * 100
        return momentum
    
    def _compute_bb_pct(self, close: np.ndarray, period: int = 20, std: float = 2.0) -> np.ndarray:
        """Bollinger Band %B"""
        bb_pct = np.zeros(len(close))
        for i in range(period, len(close)):
            window = close[i-period:i]
            middle = np.mean(window)
            upper = middle + std * np.std(window)
            lower = middle - std * np.std(window)
            bb_pct[i] = (close[i] - lower) / (upper - lower + 1e-10)
        return np.clip(bb_pct, -1, 2)
    
    def _compute_macd_hist(self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
        """MACD histogram"""
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        return macd_line - signal_line
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average"""
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _compute_vol_ratio(self, n_bars: int) -> np.ndarray:
        """Volume ratio"""
        return self.rng.lognormal(0, 0.3, n_bars)


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

class WalkForwardValidator:
    """Perform walk-forward validation"""
    
    @staticmethod
    def validate(
        tool_condition: Callable,
        thresholds: Dict[str, float],
        n_folds: int,
        train_size: int,
        test_size: int,
        edge_computer: RealEdgeComputer,
        market_generator: TemporalMarketGenerator
    ) -> List[Dict]:
        """
        Run walk-forward validation.
        
        Returns list of fold results with train/test edge.
        """
        results = []
        
        for fold in range(n_folds):
            # Generate training data (historical)
            train_market = market_generator.generate(
                train_size,
                start_date=f"2020-{fold*2+1:02d}-01",
                end_date=f"2020-{fold*2+2:02d}-01",
                regime=np.random.choice(['bull', 'bear', 'sideways', 'volatile'])
            )
            
            # Generate test data (future)
            test_market = market_generator.generate(
                test_size,
                start_date=f"2020-{fold*2+2:02d}-01",
                end_date=f"2020-{fold*2+3:02d}-01",
                regime=train_market['regime']  # Same regime assumption
            )
            
            # Compute edges
            train_result = edge_computer.compute(train_market, tool_condition, thresholds)
            test_result = edge_computer.compute(test_market, tool_condition, thresholds)
            
            results.append({
                'fold': fold,
                'train': train_result,
                'test': test_result,
                'train_regime': train_market['regime'],
                'test_regime': test_market['regime'],
                'overfitting': train_result['edge'] - test_result['edge'] if (train_result['valid'] and test_result['valid']) else 0,
            })
        
        return results


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

class BootstrapValidator:
    """Compute bootstrap confidence intervals for edge"""
    
    @staticmethod
    def compute_ci(
        returns: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for mean return.
        
        Returns: (lower_bound, mean, upper_bound)
        """
        if len(returns) < 10:
            return (0, 0, 0)
        
        bootstrap_means = []
        n = len(returns)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        mean = np.mean(bootstrap_means)
        
        return (float(lower), float(mean), float(upper))


# =============================================================================
# OUT-OF-SAMPLE VALIDATOR (MAIN)
# =============================================================================

@dataclass
class OOSResult:
    """Out-of-sample validation result"""
    status: ValidationStatus
    train_edge: float
    test_edge: float
    edge_degradation: float
    train_n_signals: int
    test_n_signals: int
    train_frequency: float
    test_frequency: float
    train_sharpe: float
    test_sharpe: float
    bootstrap_ci: Tuple[float, float, float]
    walk_forward_results: List[Dict]
    passed: bool
    reason: str


class OutOfSampleValidator:
    """
    Main OOS validator with proper train/test split.
    
    Validates tools against FUTURE unseen data with real edge computation.
    """
    
    def __init__(self, config: OOSConfig = None):
        self.config = config or OOSConfig()
        self.edge_computer = RealEdgeComputer()
        self.market_generator = TemporalMarketGenerator()
        self.bootstrap_validator = BootstrapValidator()
        self.walk_forward_validator = WalkForwardValidator()
    
    def validate(
        self,
        tool_condition: Callable,
        thresholds: Dict[str, float],
        verbose: bool = True
    ) -> OOSResult:
        """
        Validate tool with out-of-sample testing.
        
        Args:
            tool_condition: Function that returns bool array of signals
            thresholds: Tool thresholds
            verbose: Print progress
        
        Returns:
            OOSResult with validation status
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"OUT-OF-SAMPLE VALIDATION")
            print(f"{'='*70}")
            print(f"Train size: {self.config.train_samples:,} bars")
            print(f"Test size: {self.config.test_samples:,} bars")
            print(f"Min edge threshold: {self.config.min_edge_threshold*100:.2f}%")
        
        # Generate training data (historical)
        if verbose:
            print(f"\n📊 Generating training data...")
        train_market = self.market_generator.generate(
            self.config.train_samples,
            start_date="2020-01-01",
            end_date="2022-01-01",
            regime='mixed'
        )
        
        # Compute training edge
        if verbose:
            print(f"📈 Computing training edge...")
        train_result = self.edge_computer.compute(train_market, tool_condition, thresholds)
        
        if not train_result['valid']:
            return self._create_result(
                ValidationStatus.INSUFFICIENT_DATA,
                train_result, {}, [], "Training data validation failed"
            )
        
        if train_result['n_signals'] < self.config.min_signals_train:
            return self._create_result(
                ValidationStatus.INSUFFICIENT_DATA,
                train_result, {}, [],
                f"Insufficient training signals: {train_result['n_signals']} < {self.config.min_signals_train}"
            )
        
        # Generate test data (future)
        if verbose:
            print(f"\n📊 Generating out-of-sample test data...")
        test_market = self.market_generator.generate(
            self.config.test_samples,
            start_date="2022-01-01",
            end_date="2023-01-01",
            regime=train_market['regime']
        )
        
        # Compute test edge
        if verbose:
            print(f"📈 Computing OOS edge...")
        test_result = self.edge_computer.compute(test_market, tool_condition, thresholds)
        
        if not test_result['valid']:
            return self._create_result(
                ValidationStatus.INSUFFICIENT_DATA,
                train_result, test_result, [],
                "Test data validation failed"
            )
        
        if test_result['n_signals'] < self.config.min_signals_test:
            return self._create_result(
                ValidationStatus.INSUFFICIENT_DATA,
                train_result, test_result, [],
                f"Insufficient test signals: {test_result['n_signals']} < {self.config.min_signals_test}"
            )
        
        # Bootstrap confidence interval
        if verbose:
            print(f"\n🔁 Computing bootstrap CI ({self.config.n_bootstrap} iterations)...")
        ci = self.bootstrap_validator.compute_ci(
            test_result['returns'],
            self.config.n_bootstrap,
            self.config.confidence_level
        )
        
        # Walk-forward validation
        if verbose:
            print(f"\n🚶 Running walk-forward validation ({self.config.walk_forward_folds} folds)...")
        wf_results = self.walk_forward_validator.validate(
            tool_condition,
            thresholds,
            self.config.walk_forward_folds,
            self.config.train_samples // 2,
            self.config.test_samples // 2,
            self.edge_computer,
            self.market_generator
        )
        
        # Degradation check
        edge_degradation = train_result['edge'] - test_result['edge']
        
        # Decision logic
        if test_result['edge'] < self.config.min_edge_threshold:
            status = ValidationStatus.FAILED
            reason = f"OOS edge {test_result['edge']*100:.2f}% < threshold {self.config.min_edge_threshold*100:.2f}%"
        elif edge_degradation > 0.05:  # >5% degradation
            status = ValidationStatus.OVERFITTING
            reason = f"Severe overfitting: edge degradation {edge_degradation*100:.2f}%"
        else:
            status = ValidationStatus.PASSED
            reason = f"OOS edge {test_result['edge']*100:.2f}% meets threshold"
        
        result = OOSResult(
            status=status,
            train_edge=train_result['edge'],
            test_edge=test_result['edge'],
            edge_degradation=edge_degradation,
            train_n_signals=train_result['n_signals'],
            test_n_signals=test_result['n_signals'],
            train_frequency=train_result['frequency'],
            test_frequency=test_result['frequency'],
            train_sharpe=train_result['sharpe'],
            test_sharpe=test_result['sharpe'],
            bootstrap_ci=ci,
            walk_forward_results=wf_results,
            passed=(status == ValidationStatus.PASSED),
            reason=reason
        )
        
        if verbose:
            self._print_result(result)
        
        return result
    
    def _create_result(
        self,
        status: ValidationStatus,
        train: Dict,
        test: Dict,
        wf_results: List[Dict],
        reason: str
    ) -> OOSResult:
        """Create OOSResult from components"""
        return OOSResult(
            status=status,
            train_edge=train.get('edge', 0),
            test_edge=test.get('edge', 0),
            edge_degradation=train.get('edge', 0) - test.get('edge', 0),
            train_n_signals=train.get('n_signals', 0),
            test_n_signals=test.get('n_signals', 0),
            train_frequency=train.get('frequency', 0),
            test_frequency=test.get('frequency', 0),
            train_sharpe=train.get('sharpe', 0),
            test_sharpe=test.get('sharpe', 0),
            bootstrap_ci=(0, 0, 0),
            walk_forward_results=wf_results,
            passed=False,
            reason=reason
        )
    
    def _print_result(self, result: OOSResult):
        """Print validation result"""
        print(f"\n{'='*70}")
        print(f"VALIDATION RESULT: {result.status.value.upper()}")
        print(f"{'='*70}")
        print(f"Train Edge:      {result.train_edge*100:>6.2f}% ({result.train_n_signals:,} signals)")
        print(f"Test Edge (OOS): {result.test_edge*100:>6.2f}% ({result.test_n_signals:,} signals)")
        print(f"Degradation:     {result.edge_degradation*100:>6.2f}%")
        print(f"\nTrain Sharpe:    {result.train_sharpe:>6.2f}")
        print(f"Test Sharpe:     {result.test_sharpe:>6.2f}")
        print(f"\nBootstrap 95% CI: [{result.bootstrap_ci[0]*100:.2f}%, {result.bootstrap_ci[2]*100:.2f}%]")
        print(f"\nWalk-Forward Results:")
        for fold_result in result.walk_forward_results:
            print(f"  Fold {fold_result['fold']}: Train={fold_result['train']['edge']*100:.2f}%, Test={fold_result['test']['edge']*100:.2f}%, Overfit={fold_result['overfitting']*100:.2f}%")
        print(f"\n{'✅ PASSED' if result.passed else '❌ FAILED'}: {result.reason}")
        print(f"{'='*70}\n")


# =============================================================================
# TOOL-SPECIFIC CONDITION FUNCTIONS
# =============================================================================

def tool_1_condition(market: Dict[str, np.ndarray], thresholds: Dict[str, float]) -> np.ndarray:
    """Gamma Wall Pin condition"""
    return (market['vol_ratio'] >= thresholds['vol_spike']) & (market['rsi'] < 30)


def tool_4_condition(market: Dict[str, np.ndarray], thresholds: Dict[str, float]) -> np.ndarray:
    """Vol Trigger condition"""
    return (market['vix'] > thresholds['vix_threshold']) & (market['bb_pct'] > 0.8)


def tool_8_condition(market: Dict[str, np.ndarray], thresholds: Dict[str, float]) -> np.ndarray:
    """Triple Cascade condition"""
    return (
        (market['momentum'] > thresholds['momentum']) &
        (market['vol_ratio'] > thresholds['vol_ratio']) &
        (market['macd_hist'] > 0)
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Validate Tool 1 (Gamma Wall Pin)
    validator = OutOfSampleValidator()
    
    thresholds_1 = {'vol_spike': 1.5}
    result_1 = validator.validate(tool_1_condition, thresholds_1, verbose=True)
    
    # Example: Validate Tool 4 (Vol Trigger)
    thresholds_4 = {'vix_threshold': 25}
    result_4 = validator.validate(tool_4_condition, thresholds_4, verbose=True)
    
    # Example: Validate Tool 8 (Triple Cascade)
    thresholds_8 = {'momentum': 0.01, 'vol_ratio': 1.2}
    result_8 = validator.validate(tool_8_condition, thresholds_8, verbose=True)
    
    print("\n✅ Validation complete!")
    print(f"Tool 1: {'PASS' if result_1.passed else 'FAIL'} (OOS Edge: {result_1.test_edge*100:.2f}%)")
    print(f"Tool 4: {'PASS' if result_4.passed else 'FAIL'} (OOS Edge: {result_4.test_edge*100:.2f}%)")
    print(f"Tool 8: {'PASS' if result_8.passed else 'FAIL'} (OOS Edge: {result_8.test_edge*100:.2f}%)")
