"""
God-Tier Alpha Signal Engines
Advanced Rare-Event Detection System

This module contains elite alpha signal engines designed to detect
ultra-rare market events and anomalies that generate significant alpha.

Author: jetgause
Created: 2025-12-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.signal import find_peaks
from collections import deque


class SignalStrength(Enum):
    """Signal strength classification"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4
    GOD_TIER = 5


@dataclass
class AlphaSignal:
    """Container for alpha signal output"""
    timestamp: pd.Timestamp
    signal_type: str
    strength: SignalStrength
    confidence: float
    direction: int  # 1 for long, -1 for short, 0 for neutral
    metadata: Dict
    expected_duration: Optional[int] = None
    risk_level: Optional[float] = None


class BaseAlphaEngine:
    """Base class for all alpha signal engines"""
    
    def __init__(self, lookback: int = 100, sensitivity: float = 0.95):
        """
        Initialize base engine
        
        Args:
            lookback: Historical window size
            sensitivity: Detection sensitivity (0-1)
        """
        self.lookback = lookback
        self.sensitivity = sensitivity
        self.signal_history = deque(maxlen=1000)
        
    def _calculate_z_score(self, data: np.ndarray, value: float) -> float:
        """Calculate z-score for anomaly detection"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (value - mean) / std
    
    def _detect_regime_change(self, data: np.ndarray, threshold: float = 2.5) -> bool:
        """Detect statistical regime changes"""
        if len(data) < 20:
            return False
        
        mid = len(data) // 2
        first_half = data[:mid]
        second_half = data[mid:]
        
        # Perform Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(first_half, second_half)
        return p_value < (1 - self.sensitivity)
    
    def generate_signal(self, market_data: pd.DataFrame) -> Optional[AlphaSignal]:
        """Generate alpha signal - to be implemented by subclasses"""
        raise NotImplementedError


class VolTriggerEngine(BaseAlphaEngine):
    """
    Volatility Trigger Engine
    
    Detects extreme volatility regime changes and vol compression/expansion events.
    These events often precede significant price moves.
    """
    
    def __init__(self, lookback: int = 100, vol_threshold: float = 3.0, 
                 compression_ratio: float = 0.3):
        """
        Initialize VolTrigger Engine
        
        Args:
            lookback: Historical window for volatility calculation
            vol_threshold: Z-score threshold for extreme vol detection
            compression_ratio: Ratio threshold for vol compression detection
        """
        super().__init__(lookback)
        self.vol_threshold = vol_threshold
        self.compression_ratio = compression_ratio
        self.vol_history = deque(maxlen=lookback)
        
    def _calculate_realized_vol(self, returns: np.ndarray, window: int = 20) -> float:
        """Calculate realized volatility"""
        if len(returns) < window:
            return np.std(returns) * np.sqrt(252)
        return np.std(returns[-window:]) * np.sqrt(252)
    
    def _detect_vol_compression(self, vol_series: np.ndarray) -> Tuple[bool, float]:
        """
        Detect volatility compression events
        
        Returns:
            Tuple of (is_compressed, compression_score)
        """
        if len(vol_series) < 30:
            return False, 0.0
        
        recent_vol = np.mean(vol_series[-10:])
        historical_vol = np.mean(vol_series[:-10])
        
        if historical_vol == 0:
            return False, 0.0
        
        compression_ratio = recent_vol / historical_vol
        
        if compression_ratio < self.compression_ratio:
            return True, 1.0 - compression_ratio
        
        return False, 0.0
    
    def _detect_vol_spike(self, vol_series: np.ndarray, current_vol: float) -> Tuple[bool, float]:
        """Detect volatility spikes"""
        z_score = self._calculate_z_score(vol_series, current_vol)
        
        if abs(z_score) > self.vol_threshold:
            return True, abs(z_score)
        
        return False, abs(z_score)
    
    def generate_signal(self, market_data: pd.DataFrame) -> Optional[AlphaSignal]:
        """
        Generate volatility-based alpha signals
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            AlphaSignal or None
        """
        if len(market_data) < self.lookback:
            return None
        
        # Calculate returns
        returns = market_data['close'].pct_change().dropna().values
        
        if len(returns) < 30:
            return None
        
        # Calculate realized volatility
        current_vol = self._calculate_realized_vol(returns)
        self.vol_history.append(current_vol)
        
        if len(self.vol_history) < 30:
            return None
        
        vol_array = np.array(self.vol_history)
        
        # Check for compression
        is_compressed, compression_score = self._detect_vol_compression(vol_array)
        
        # Check for spike
        is_spike, spike_score = self._detect_vol_spike(vol_array[:-1], current_vol)
        
        # Generate signal
        if is_compressed and compression_score > 0.5:
            return AlphaSignal(
                timestamp=market_data.index[-1],
                signal_type="VOL_COMPRESSION",
                strength=SignalStrength.STRONG if compression_score > 0.7 else SignalStrength.MODERATE,
                confidence=min(compression_score, 0.95),
                direction=0,  # Neutral - awaiting breakout
                metadata={
                    'compression_ratio': compression_score,
                    'current_vol': current_vol,
                    'avg_historical_vol': np.mean(vol_array[:-10])
                },
                expected_duration=20,
                risk_level=0.3
            )
        
        elif is_spike:
            direction = 1 if returns[-1] > 0 else -1
            strength = SignalStrength.EXTREME if spike_score > 4.0 else SignalStrength.STRONG
            
            return AlphaSignal(
                timestamp=market_data.index[-1],
                signal_type="VOL_SPIKE",
                strength=strength,
                confidence=min(spike_score / 5.0, 0.98),
                direction=direction,
                metadata={
                    'z_score': spike_score,
                    'current_vol': current_vol,
                    'historical_mean': np.mean(vol_array[:-1])
                },
                expected_duration=5,
                risk_level=0.8
            )
        
        return None


class GammaFlipEngine(BaseAlphaEngine):
    """
    Gamma Flip Detection Engine
    
    Detects when market makers flip from long to short gamma or vice versa.
    These transitions create powerful directional moves.
    """
    
    def __init__(self, lookback: int = 50, flip_threshold: float = 0.0):
        """
        Initialize GammaFlip Engine
        
        Args:
            lookback: Historical window
            flip_threshold: Gamma level threshold for flip detection
        """
        super().__init__(lookback)
        self.flip_threshold = flip_threshold
        self.gamma_history = deque(maxlen=lookback)
        self.position_history = deque(maxlen=lookback)
        
    def _estimate_gamma_exposure(self, market_data: pd.DataFrame) -> float:
        """
        Estimate gamma exposure from price action
        
        This is a simplified model that uses price acceleration and
        volume patterns as proxy for gamma positioning.
        """
        if len(market_data) < 10:
            return 0.0
        
        # Calculate price acceleration
        returns = market_data['close'].pct_change().values[-10:]
        acceleration = np.diff(returns)
        
        # Calculate volume-weighted price pressure
        volume_normalized = market_data['volume'].values[-10:] / market_data['volume'].mean()
        price_pressure = returns[-9:] * volume_normalized[-9:]
        
        # Gamma proxy: negative correlation between acceleration and volume
        # High volume with decelerating moves = long gamma (dampening)
        # High volume with accelerating moves = short gamma (amplifying)
        gamma_proxy = -np.corrcoef(np.abs(acceleration), volume_normalized[-9:])[0, 1]
        
        if np.isnan(gamma_proxy):
            return 0.0
        
        return gamma_proxy
    
    def _detect_flip(self, gamma_series: np.ndarray) -> Tuple[bool, str, float]:
        """
        Detect gamma flip events
        
        Returns:
            Tuple of (is_flip, flip_type, flip_strength)
        """
        if len(gamma_series) < 20:
            return False, "NONE", 0.0
        
        recent_gamma = np.mean(gamma_series[-5:])
        historical_gamma = np.mean(gamma_series[-20:-5])
        
        # Check for significant gamma change
        gamma_change = recent_gamma - historical_gamma
        
        if abs(gamma_change) > 0.5:  # Significant change
            if recent_gamma > self.flip_threshold and historical_gamma <= self.flip_threshold:
                return True, "LONG_GAMMA", abs(gamma_change)
            elif recent_gamma <= self.flip_threshold and historical_gamma > self.flip_threshold:
                return True, "SHORT_GAMMA", abs(gamma_change)
        
        return False, "NONE", 0.0
    
    def generate_signal(self, market_data: pd.DataFrame) -> Optional[AlphaSignal]:
        """
        Generate gamma flip signals
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            AlphaSignal or None
        """
        if len(market_data) < self.lookback:
            return None
        
        # Estimate current gamma exposure
        current_gamma = self._estimate_gamma_exposure(market_data)
        self.gamma_history.append(current_gamma)
        
        if len(self.gamma_history) < 20:
            return None
        
        gamma_array = np.array(self.gamma_history)
        
        # Detect flip
        is_flip, flip_type, flip_strength = self._detect_flip(gamma_array)
        
        if is_flip:
            # Long gamma = market makers will dampen moves = fade extremes
            # Short gamma = market makers will amplify moves = follow momentum
            
            if flip_type == "LONG_GAMMA":
                direction = -np.sign(market_data['close'].pct_change().iloc[-1])
                strategy = "FADE"
            else:  # SHORT_GAMMA
                direction = np.sign(market_data['close'].pct_change().iloc[-1])
                strategy = "MOMENTUM"
            
            return AlphaSignal(
                timestamp=market_data.index[-1],
                signal_type=f"GAMMA_FLIP_{flip_type}",
                strength=SignalStrength.EXTREME if flip_strength > 1.0 else SignalStrength.STRONG,
                confidence=min(flip_strength / 1.5, 0.92),
                direction=int(direction),
                metadata={
                    'flip_type': flip_type,
                    'flip_strength': flip_strength,
                    'current_gamma': current_gamma,
                    'strategy': strategy
                },
                expected_duration=10,
                risk_level=0.7
            )
        
        return None


class SilenceEngine(BaseAlphaEngine):
    """
    Market Silence Detection Engine
    
    Detects periods of abnormal market silence (low volatility, low volume)
    which often precede explosive moves.
    """
    
    def __init__(self, lookback: int = 100, silence_threshold: float = 0.25):
        """
        Initialize Silence Engine
        
        Args:
            lookback: Historical window
            silence_threshold: Percentile threshold for silence detection
        """
        super().__init__(lookback)
        self.silence_threshold = silence_threshold
        self.activity_history = deque(maxlen=lookback)
        
    def _calculate_market_activity(self, market_data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate market activity score combining volatility and volume
        """
        if len(market_data) < window:
            return 0.0
        
        recent_data = market_data.tail(window)
        
        # Volatility component
        returns = recent_data['close'].pct_change().dropna()
        vol_score = returns.std()
        
        # Volume component (normalized)
        avg_volume = market_data['volume'].tail(self.lookback).mean()
        if avg_volume == 0:
            vol_component = 0
        else:
            vol_component = recent_data['volume'].mean() / avg_volume
        
        # Range component
        range_pct = (recent_data['high'] - recent_data['low']) / recent_data['close']
        range_score = range_pct.mean()
        
        # Combined activity score
        activity = (vol_score * 0.4 + vol_component * 0.3 + range_score * 0.3)
        
        return activity
    
    def _detect_silence(self, activity_series: np.ndarray) -> Tuple[bool, float, int]:
        """
        Detect market silence events
        
        Returns:
            Tuple of (is_silent, silence_score, duration)
        """
        if len(activity_series) < 30:
            return False, 0.0, 0
        
        current_activity = activity_series[-1]
        percentile = stats.percentileofscore(activity_series[:-1], current_activity)
        
        # Check if current activity is in bottom percentile
        if percentile < (self.silence_threshold * 100):
            # Calculate duration of silence
            duration = 1
            for i in range(len(activity_series) - 2, -1, -1):
                if stats.percentileofscore(activity_series[:i], activity_series[i]) < (self.silence_threshold * 100):
                    duration += 1
                else:
                    break
            
            silence_score = 1.0 - (percentile / 100.0)
            return True, silence_score, duration
        
        return False, 0.0, 0
    
    def generate_signal(self, market_data: pd.DataFrame) -> Optional[AlphaSignal]:
        """
        Generate silence-based alpha signals
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            AlphaSignal or None
        """
        if len(market_data) < self.lookback:
            return None
        
        # Calculate current market activity
        current_activity = self._calculate_market_activity(market_data)
        self.activity_history.append(current_activity)
        
        if len(self.activity_history) < 30:
            return None
        
        activity_array = np.array(self.activity_history)
        
        # Detect silence
        is_silent, silence_score, duration = self._detect_silence(activity_array)
        
        if is_silent and duration >= 3:  # At least 3 periods of silence
            # Longer silence = stronger signal
            strength_multiplier = min(duration / 10.0, 2.0)
            
            if duration >= 10:
                strength = SignalStrength.GOD_TIER
            elif duration >= 7:
                strength = SignalStrength.EXTREME
            elif duration >= 5:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
            
            return AlphaSignal(
                timestamp=market_data.index[-1],
                signal_type="MARKET_SILENCE",
                strength=strength,
                confidence=min(silence_score * strength_multiplier, 0.96),
                direction=0,  # Neutral - awaiting breakout direction
                metadata={
                    'silence_score': silence_score,
                    'duration': duration,
                    'current_activity': current_activity,
                    'activity_percentile': stats.percentileofscore(activity_array[:-1], current_activity)
                },
                expected_duration=duration // 2,  # Typically breaks in half the accumulation time
                risk_level=0.4
            )
        
        return None


class LiquidityCrisisEngine(BaseAlphaEngine):
    """
    Liquidity Crisis Detection Engine
    
    Detects liquidity droughts and flash crash conditions by analyzing
    order book imbalances and bid-ask dynamics.
    """
    
    def __init__(self, lookback: int = 50, crisis_threshold: float = 2.5):
        """
        Initialize LiquidityCrisis Engine
        
        Args:
            lookback: Historical window
            crisis_threshold: Z-score threshold for crisis detection
        """
        super().__init__(lookback)
        self.crisis_threshold = crisis_threshold
        self.liquidity_history = deque(maxlen=lookback)
        
    def _estimate_liquidity(self, market_data: pd.DataFrame) -> float:
        """
        Estimate market liquidity from price and volume data
        
        Uses Amihud illiquidity measure as proxy
        """
        if len(market_data) < 2:
            return 0.0
        
        recent_data = market_data.tail(20)
        
        # Calculate returns
        returns = recent_data['close'].pct_change().dropna()
        
        # Amihud illiquidity: |return| / volume
        illiquidity = (returns.abs() / recent_data['volume'].iloc[1:]).mean()
        
        if np.isnan(illiquidity) or np.isinf(illiquidity):
            return 0.0
        
        # Return inverse for liquidity measure
        return 1.0 / (illiquidity + 1e-10)
    
    def _calculate_price_impact(self, market_data: pd.DataFrame) -> float:
        """Calculate price impact as measure of liquidity"""
        if len(market_data) < 5:
            return 0.0
        
        recent = market_data.tail(5)
        
        # Price impact = price change per unit volume
        price_changes = recent['close'].diff().abs()
        volume_normalized = recent['volume'] / recent['volume'].mean()
        
        impact = (price_changes / (volume_normalized + 1e-10)).mean()
        
        return impact if not np.isnan(impact) else 0.0
    
    def _detect_crisis(self, liquidity_series: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect liquidity crisis events
        
        Returns:
            Tuple of (is_crisis, crisis_severity, crisis_type)
        """
        if len(liquidity_series) < 20:
            return False, 0.0, "NONE"
        
        current_liquidity = liquidity_series[-1]
        z_score = self._calculate_z_score(liquidity_series[:-1], current_liquidity)
        
        # Negative z-score = below average liquidity = crisis
        if z_score < -self.crisis_threshold:
            severity = abs(z_score)
            
            # Check if it's sudden (flash crash) or gradual (drought)
            recent_change = liquidity_series[-1] - liquidity_series[-5]
            gradual_change = liquidity_series[-5] - liquidity_series[-20]
            
            if abs(recent_change) > abs(gradual_change):
                crisis_type = "FLASH_CRASH"
            else:
                crisis_type = "LIQUIDITY_DROUGHT"
            
            return True, severity, crisis_type
        
        return False, 0.0, "NONE"
    
    def generate_signal(self, market_data: pd.DataFrame) -> Optional[AlphaSignal]:
        """
        Generate liquidity crisis signals
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            AlphaSignal or None
        """
        if len(market_data) < self.lookback:
            return None
        
        # Estimate current liquidity
        current_liquidity = self._estimate_liquidity(market_data)
        current_impact = self._calculate_price_impact(market_data)
        
        self.liquidity_history.append(current_liquidity)
        
        if len(self.liquidity_history) < 20:
            return None
        
        liquidity_array = np.array(self.liquidity_history)
        
        # Detect crisis
        is_crisis, severity, crisis_type = self._detect_crisis(liquidity_array)
        
        if is_crisis:
            # Liquidity crisis = opportunity to fade panic or catch recovery
            if crisis_type == "FLASH_CRASH":
                direction = 1  # Buy the crash
                strategy = "CRASH_RECOVERY"
            else:  # LIQUIDITY_DROUGHT
                direction = -1  # Avoid or short
                strategy = "AVOID_ILLIQUID"
            
            if severity > 4.0:
                strength = SignalStrength.GOD_TIER
            elif severity > 3.5:
                strength = SignalStrength.EXTREME
            else:
                strength = SignalStrength.STRONG
            
            return AlphaSignal(
                timestamp=market_data.index[-1],
                signal_type=f"LIQUIDITY_CRISIS_{crisis_type}",
                strength=strength,
                confidence=min(severity / 5.0, 0.94),
                direction=direction,
                metadata={
                    'crisis_type': crisis_type,
                    'severity': severity,
                    'current_liquidity': current_liquidity,
                    'price_impact': current_impact,
                    'strategy': strategy
                },
                expected_duration=15 if crisis_type == "FLASH_CRASH" else 30,
                risk_level=0.9
            )
        
        return None


class DarkPoolGammaEngine(BaseAlphaEngine):
    """
    Dark Pool Gamma Detection Engine
    
    Detects large institutional positioning and dark pool activity
    through volume and price anomalies.
    """
    
    def __init__(self, lookback: int = 100, block_threshold: float = 3.0):
        """
        Initialize DarkPoolGamma Engine
        
        Args:
            lookback: Historical window
            block_threshold: Z-score threshold for block trade detection
        """
        super().__init__(lookback)
        self.block_threshold = block_threshold
        self.block_history = deque(maxlen=lookback)
        self.volume_profile = deque(maxlen=lookback)
        
    def _detect_block_trades(self, market_data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Detect potential block trades from volume anomalies
        
        Returns:
            Tuple of (is_block, block_size_zscore, direction)
        """
        if len(market_data) < 20:
            return False, 0.0, "NONE"
        
        current_volume = market_data['volume'].iloc[-1]
        historical_volume = market_data['volume'].iloc[-20:-1]
        
        z_score = self._calculate_z_score(historical_volume.values, current_volume)
        
        if z_score > self.block_threshold:
            # Determine direction from price action
            price_change = market_data['close'].iloc[-1] - market_data['close'].iloc[-2]
            
            if abs(price_change) / market_data['close'].iloc[-2] < 0.001:
                # Large volume with minimal price change = dark pool
                direction = "DARK_POOL"
            elif price_change > 0:
                direction = "BULLISH_BLOCK"
            else:
                direction = "BEARISH_BLOCK"
            
            return True, z_score, direction
        
        return False, 0.0, "NONE"
    
    def _analyze_gamma_positioning(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze institutional gamma positioning from price levels
        """
        if len(market_data) < 50:
            return {}
        
        # Identify key price levels with high volume
        volume_by_price = {}
        recent_data = market_data.tail(50)
        
        for idx, row in recent_data.iterrows():
            price_level = round(row['close'] / row['close'] * 100) * row['close'] / 100  # Round to significant levels
            if price_level not in volume_by_price:
                volume_by_price[price_level] = 0
            volume_by_price[price_level] += row['volume']
        
        # Find gamma walls (high volume price levels)
        sorted_levels = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
        
        current_price = market_data['close'].iloc[-1]
        
        resistance_levels = [p for p, v in sorted_levels if p > current_price][:3]
        support_levels = [p for p, v in sorted_levels if p < current_price][:3]
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'current_price': current_price
        }
    
    def generate_signal(self, market_data: pd.DataFrame) -> Optional[AlphaSignal]:
        """
        Generate dark pool gamma signals
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            AlphaSignal or None
        """
        if len(market_data) < self.lookback:
            return None
        
        # Detect block trades
        is_block, block_size, direction = self._detect_block_trades(market_data)
        
        if is_block:
            # Analyze gamma positioning
            gamma_levels = self._analyze_gamma_positioning(market_data)
            
            # Determine signal direction
            if direction == "DARK_POOL":
                # Dark pool activity = smart money positioning
                signal_direction = 0  # Wait for directional confirmation
                strategy = "FOLLOW_SMART_MONEY"
            elif direction == "BULLISH_BLOCK":
                signal_direction = 1
                strategy = "INSTITUTIONAL_LONG"
            else:  # BEARISH_BLOCK
                signal_direction = -1
                strategy = "INSTITUTIONAL_SHORT"
            
            # Strength based on block size
            if block_size > 5.0:
                strength = SignalStrength.GOD_TIER
            elif block_size > 4.0:
                strength = SignalStrength.EXTREME
            else:
                strength = SignalStrength.STRONG
            
            return AlphaSignal(
                timestamp=market_data.index[-1],
                signal_type=f"DARK_POOL_{direction}",
                strength=strength,
                confidence=min(block_size / 6.0, 0.95),
                direction=signal_direction,
                metadata={
                    'block_type': direction,
                    'block_size_zscore': block_size,
                    'volume': market_data['volume'].iloc[-1],
                    'avg_volume': market_data['volume'].iloc[-20:-1].mean(),
                    'gamma_levels': gamma_levels,
                    'strategy': strategy
                },
                expected_duration=20,
                risk_level=0.6
            )
        
        return None


class GodTierAlphaAggregator:
    """
    Aggregates signals from all God-Tier engines and generates
    unified alpha recommendations.
    """
    
    def __init__(self):
        """Initialize the aggregator with all engines"""
        self.engines = {
            'vol_trigger': VolTriggerEngine(),
            'gamma_flip': GammaFlipEngine(),
            'silence': SilenceEngine(),
            'liquidity_crisis': LiquidityCrisisEngine(),
            'dark_pool': DarkPoolGammaEngine()
        }
        self.signal_log = []
        
    def generate_all_signals(self, market_data: pd.DataFrame) -> Dict[str, Optional[AlphaSignal]]:
        """
        Generate signals from all engines
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping engine names to their signals
        """
        signals = {}
        
        for engine_name, engine in self.engines.items():
            try:
                signal = engine.generate_signal(market_data)
                signals[engine_name] = signal
                
                if signal is not None:
                    self.signal_log.append({
                        'timestamp': signal.timestamp,
                        'engine': engine_name,
                        'signal': signal
                    })
            except Exception as e:
                warnings.warn(f"Engine {engine_name} failed: {str(e)}")
                signals[engine_name] = None
        
        return signals
    
    def get_consensus_signal(self, signals: Dict[str, Optional[AlphaSignal]]) -> Optional[AlphaSignal]:
        """
        Generate consensus signal from multiple engine signals
        
        Args:
            signals: Dictionary of signals from each engine
            
        Returns:
            Consensus AlphaSignal or None
        """
        active_signals = [s for s in signals.values() if s is not None]
        
        if not active_signals:
            return None
        
        # Weight signals by strength and confidence
        weights = []
        directions = []
        
        for signal in active_signals:
            weight = signal.strength.value * signal.confidence
            weights.append(weight)
            directions.append(signal.direction)
        
        # Calculate weighted direction
        weighted_direction = np.average(directions, weights=weights)
        consensus_direction = int(np.sign(weighted_direction))
        
        # Calculate consensus strength
        avg_strength_value = np.average([s.strength.value for s in active_signals], weights=weights)
        
        if avg_strength_value >= 4.5:
            consensus_strength = SignalStrength.GOD_TIER
        elif avg_strength_value >= 3.5:
            consensus_strength = SignalStrength.EXTREME
        elif avg_strength_value >= 2.5:
            consensus_strength = SignalStrength.STRONG
        elif avg_strength_value >= 1.5:
            consensus_strength = SignalStrength.MODERATE
        else:
            consensus_strength = SignalStrength.WEAK
        
        # Calculate consensus confidence
        consensus_confidence = np.average([s.confidence for s in active_signals], weights=weights)
        
        # Aggregate metadata
        consensus_metadata = {
            'num_signals': len(active_signals),
            'contributing_engines': [s.signal_type for s in active_signals],
            'individual_confidences': [s.confidence for s in active_signals]
        }
        
        return AlphaSignal(
            timestamp=active_signals[0].timestamp,
            signal_type="CONSENSUS_GOD_TIER",
            strength=consensus_strength,
            confidence=min(consensus_confidence, 0.98),
            direction=consensus_direction,
            metadata=consensus_metadata,
            expected_duration=int(np.mean([s.expected_duration for s in active_signals if s.expected_duration])),
            risk_level=np.max([s.risk_level for s in active_signals if s.risk_level])
        )
    
    def get_signal_report(self) -> pd.DataFrame:
        """
        Generate a report of all historical signals
        
        Returns:
            DataFrame with signal history
        """
        if not self.signal_log:
            return pd.DataFrame()
        
        report_data = []
        for entry in self.signal_log:
            signal = entry['signal']
            report_data.append({
                'timestamp': entry['timestamp'],
                'engine': entry['engine'],
                'signal_type': signal.signal_type,
                'strength': signal.strength.name,
                'confidence': signal.confidence,
                'direction': signal.direction,
                'risk_level': signal.risk_level
            })
        
        return pd.DataFrame(report_data)


# Example usage and testing
if __name__ == "__main__":
    print("God-Tier Alpha Signal Engines initialized successfully!")
    print("\nAvailable Engines:")
    print("1. VolTriggerEngine - Volatility regime detection")
    print("2. GammaFlipEngine - Gamma positioning analysis")
    print("3. SilenceEngine - Market silence detection")
    print("4. LiquidityCrisisEngine - Liquidity crisis detection")
    print("5. DarkPoolGammaEngine - Institutional positioning detection")
    print("\nUse GodTierAlphaAggregator to combine all signals!")
