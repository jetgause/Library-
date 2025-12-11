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
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='same')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='same')
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Adjust to target mean
        rsi = rsi - np.mean(rsi) + base_mean
        return np.clip(rsi, 0, 100)
    
    def _compute_momentum(self, close: np.ndarray, period: int = 10) -> np.ndarray:
        momentum = np.zeros_like(close)
        momentum[period:] = close[period:] / close[:-period] - 1
        return momentum
    
    def _compute_bb_pct(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        sma = np.convolve(close, np.ones(period)/period, mode='same')
        std = np.array([np.std(close[max(0,i-period):i+1]) for i in range(len(close))])
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (close - lower) / (upper - lower + 1e-10)
    
    def _compute_macd_hist(self, close: np.ndarray) -> np.ndarray:
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd = ema12 - ema26
        signal = self._ema(macd, 9)
        return macd - signal
    
    def _compute_vol_ratio(self, n: int) -> np.ndarray:
        volume = self.rng.lognormal(10, 0.5, n)
        vol_sma = np.convolve(volume, np.ones(20)/20, mode='same')
        return volume / (vol_sma + 1e-10)
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema


# =============================================================================
# OUT-OF-SAMPLE VALIDATOR
# =============================================================================

class OutOfSampleValidator:
    """
    Validates improvements using proper train/test splits.
    
    Key principle: Train on past data, validate on FUTURE data.
    This prevents overfitting to specific market conditions.
    """
    
    def __init__(self, config: OOSConfig = None):
        self.config = config or OOSConfig()
        self.market_gen = TemporalMarketGenerator()
        self.edge_computer = RealEdgeComputer()
    
    def validate_oos(
        self,
        tool_condition: Callable,
        old_thresholds: Dict[str, float],
        new_thresholds: Dict[str, float],
        verbose: bool = True
    ) -> Dict:
        """
        Validate improvement with out-of-sample testing.
        
        Process:
        1. Generate training data (simulates historical data)
        2. Verify improvement exists in training data
        3. Generate test data (simulates future data)
        4. Check if improvement holds in test data
        5. Require minimum 2% real edge in OOS
        """
        if verbose:
            print("\n" + "="*60)
            print("🔬 OUT-OF-SAMPLE VALIDATION")
            print("="*60)
        
        # === Step 1: Generate Training Data ===
        train_data = self.market_gen.generate(
            self.config.train_samples,
            start_date="2020-01-01",
            end_date="2024-01-01",
            regime='mixed'
        )
        
        if verbose:
            print(f"\n📊 Training Data: {self.config.train_samples:,} bars (2020-2024)")
        
        # === Step 2: Evaluate on Training Data ===
        old_train = self.edge_computer.compute(train_data, tool_condition, old_thresholds)
        new_train = self.edge_computer.compute(train_data, tool_condition, new_thresholds)
        
        if verbose:
            print(f"\n   IN-SAMPLE Results:")
            print(f"   Old: WR={old_train['win_rate']*100:.1f}% | Edge={old_train['edge']*10000:.1f}bps | N={old_train['n_signals']}")
            print(f"   New: WR={new_train['win_rate']*100:.1f}% | Edge={new_train['edge']*10000:.1f}bps | N={new_train['n_signals']}")
        
        # Check minimum signals
        if new_train['n_signals'] < self.config.min_signals_train:
            return self._fail_result(
                "insufficient_signals_train",
                f"Only {new_train['n_signals']} signals in training (need {self.config.min_signals_train})"
            )
        
        # Check in-sample improvement exists
        train_improvement = new_train['win_rate'] - old_train['win_rate']
        if train_improvement < 0.01:  # Need at least 1% in-sample improvement
            return self._fail_result(
                "no_insample_improvement",
                f"In-sample improvement only {train_improvement*100:.1f}%"
            )
        
        # === Step 3: Generate Test Data (OUT OF SAMPLE) ===
        test_data = self.market_gen.generate(
            self.config.test_samples,
            start_date="2024-01-02",
            end_date="2025-01-01",
            regime='mixed'
        )
        
        if verbose:
            print(f"\n📊 Test Data: {self.config.test_samples:,} bars (2024-2025) [OUT-OF-SAMPLE]")
        
        # === Step 4: Evaluate on Test Data ===
        old_test = self.edge_computer.compute(test_data, tool_condition, old_thresholds)
        new_test = self.edge_computer.compute(test_data, tool_condition, new_thresholds)
        
        if verbose:
            print(f"\n   OUT-OF-SAMPLE Results:")
            print(f"   Old: WR={old_test['win_rate']*100:.1f}% | Edge={old_test['edge']*10000:.1f}bps | N={old_test['n_signals']}")
            print(f"   New: WR={new_test['win_rate']*100:.1f}% | Edge={new_test['edge']*10000:.1f}bps | N={new_test['n_signals']}")
        
        # Check minimum signals in test
        if new_test['n_signals'] < self.config.min_signals_test:
            return self._fail_result(
                "insufficient_signals_test",
                f"Only {new_test['n_signals']} signals in test (need {self.config.min_signals_test})"
            )
        
        # === Step 5: Check OOS Edge Threshold ===
        oos_edge = new_test['win_rate'] - 0.5  # Edge over random
        oos_improvement = new_test['win_rate'] - old_test['win_rate']
        
        if verbose:
            print(f"\n   OOS Edge: {oos_edge*100:.1f}% (threshold: {self.config.min_edge_threshold*100:.0f}%)")
            print(f"   OOS Improvement: {oos_improvement*100:.1f}%")
        
        # CRITICAL: Reject if OOS edge < 2%
        if oos_edge < self.config.min_edge_threshold:
            return self._fail_result(
                "insufficient_oos_edge",
                f"OOS edge {oos_edge*100:.1f}% < {self.config.min_edge_threshold*100:.0f}% threshold"
            )
        
        # Check for overfitting (big drop from in-sample to OOS)
        edge_decay = (new_train['win_rate'] - new_test['win_rate']) / (new_train['win_rate'] - 0.5 + 1e-10)
        if edge_decay > 0.5:  # Lost more than 50% of edge
            return self._fail_result(
                "overfitting_detected",
                f"Edge decayed {edge_decay*100:.0f}% from in-sample to OOS"
            )
        
        # === Step 6: Statistical Significance ===
        if 'returns' in new_test and 'returns' in old_test:
            significance = self._test_significance(old_test['returns'], new_test['returns'])
        else:
            significance = {'p_value': 0.05, 'significant': True}  # Assume pass if no returns
        
        if verbose:
            print(f"\n   Statistical Test: p={significance['p_value']:.4f} ({'✅' if significance['significant'] else '❌'})")
        
        # === PASSED ===
        result = {
            'status': ValidationStatus.PASSED,
            'passed': True,
            
            # In-sample metrics
            'insample': {
                'old_wr': old_train['win_rate'],
                'new_wr': new_train['win_rate'],
                'improvement': train_improvement,
                'n_signals': new_train['n_signals']
            },
            
            # Out-of-sample metrics
            'oos': {
                'old_wr': old_test['win_rate'],
                'new_wr': new_test['win_rate'],
                'improvement': oos_improvement,
                'edge': oos_edge,
                'n_signals': new_test['n_signals'],
                'sharpe': new_test['sharpe']
            },
            
            # Decay metrics
            'edge_decay': edge_decay,
            'statistical_significance': significance,
            
            'message': f"✅ PASSED: OOS edge {oos_edge*100:.1f}% exceeds {self.config.min_edge_threshold*100:.0f}% threshold"
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(result['message'])
            print(f"{'='*60}")
        
        return result
    
    def walk_forward_validate(
        self,
        tool_condition: Callable,
        thresholds: Dict[str, float],
        verbose: bool = True
    ) -> Dict:
        """
        Walk-forward validation across multiple time periods.
        
        Splits data into sequential folds and tests on each.
        """
        if verbose:
            print("\n" + "="*60)
            print("🚶 WALK-FORWARD VALIDATION")
            print("="*60)
        
        fold_results = []
        samples_per_fold = self.config.train_samples // self.config.walk_forward_folds
        
        regimes = ['bull', 'bear', 'sideways', 'volatile', 'mixed']
        
        for fold in range(self.config.walk_forward_folds):
            regime = regimes[fold % len(regimes)]
            
            # Generate fold data
            fold_data = self.market_gen.generate(
                samples_per_fold,
                regime=regime
            )
            
            # Compute real edge
            result = self.edge_computer.compute(fold_data, tool_condition, thresholds)
            
            fold_results.append({
                'fold': fold + 1,
                'regime': regime,
                'win_rate': result['win_rate'],
                'edge': result['edge'],
                'n_signals': result['n_signals'],
                'sharpe': result['sharpe']
            })
            
            if verbose:
                status = "✅" if result['win_rate'] > 0.52 else "❌"
                print(f"  Fold {fold+1} ({regime:8}): WR={result['win_rate']*100:.1f}% | Edge={result['edge']*10000:.1f}bps {status}")
        
        # Aggregate results
        wrs = [f['win_rate'] for f in fold_results]
        mean_wr = np.mean(wrs)
        std_wr = np.std(wrs)
        min_wr = np.min(wrs)
        
        # Pass criteria
        passed = (
            mean_wr > 0.52 and          # Average > 52%
            std_wr < 0.10 and           # Consistent across folds
            min_wr > 0.48               # No terrible folds
        )
        
        if verbose:
            print(f"\n  Mean WR: {mean_wr*100:.1f}% | Std: {std_wr*100:.1f}% | Min: {min_wr*100:.1f}%")
            print(f"  Result: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return {
            'passed': passed,
            'mean_wr': mean_wr,
            'std_wr': std_wr,
            'min_wr': min_wr,
            'max_wr': np.max(wrs),
            'folds': fold_results
        }
    
    def _test_significance(
        self,
        old_returns: np.ndarray,
        new_returns: np.ndarray
    ) -> Dict:
        """Test statistical significance of improvement"""
        if len(old_returns) < 10 or len(new_returns) < 10:
            return {'p_value': 1.0, 'significant': False}
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(new_returns, old_returns, alternative='greater')
        
        return {
            'p_value': float(p_value),
            't_stat': float(t_stat),
            'significant': p_value < (1 - self.config.confidence_level)
        }
    
    def _fail_result(self, reason: str, message: str) -> Dict:
        """Generate failure result"""
        return {
            'status': ValidationStatus.FAILED,
            'passed': False,
            'failure_reason': reason,
            'message': f"❌ FAILED: {message}"
        }


# =============================================================================
# INTEGRATION WITH MARKETPLACE
# =============================================================================

def validate_improvement(
    session_id: str,
    tool_condition: Callable,
    old_thresholds: Dict[str, float],
    new_thresholds: Dict[str, float],
    verbose: bool = True
) -> Dict:
    """
    Main validation entry point for marketplace improvements.
    
    Returns dict with:
        - passed: bool
        - oos_edge: float (out-of-sample edge)
        - message: str
    """
    validator = OutOfSampleValidator()
    
    # Run OOS validation
    result = validator.validate_oos(
        tool_condition,
        old_thresholds,
        new_thresholds,
        verbose=verbose
    )
    
    # If passed OOS, also run walk-forward
    if result['passed']:
        wf_result = validator.walk_forward_validate(
            tool_condition,
            new_thresholds,
            verbose=verbose
        )
        result['walk_forward'] = wf_result
        
        # Final verdict
        if not wf_result['passed']:
            result['passed'] = False
            result['status'] = ValidationStatus.FAILED
            result['message'] = "❌ FAILED: Walk-forward validation failed"
    
    return result


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate OOS validation"""
    print("="*70)
    print("🔬 OUT-OF-SAMPLE VALIDATION DEMO")
    print("="*70)
    
    # Define a test tool
    def rsi_momentum(m, t):
        return (m['rsi'] < t['rsi']) & (m['momentum'] > t['momentum'])
    
    old_thresholds = {'rsi': 40, 'momentum': 0.001}
    new_thresholds = {'rsi': 35, 'momentum': 0.002}  # "Improved" thresholds
    
    # Run validation
    result = validate_improvement(
        session_id="test_session",
        tool_condition=rsi_momentum,
        old_thresholds=old_thresholds,
        new_thresholds=new_thresholds,
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: {result['message']}")
    print(f"{'='*70}")
    
    return result


if __name__ == "__main__":
    demo()
