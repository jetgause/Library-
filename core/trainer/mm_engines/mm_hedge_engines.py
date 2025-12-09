"""
Market Maker Hedging Detection Engines
Detect real dealer hedging flows from options positioning
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class GammaExposureEngine:
    """
    Calculate Gamma Exposure (GEX) profiles to detect dealer hedging pressure
    
    Dealers are short gamma from selling options to retail/institutions.
    When spot moves, they must hedge by trading the underlying:
    - Positive GEX zone: Dealers sell into rallies, buy into dips (stabilizing)
    - Negative GEX zone: Dealers buy rallies, sell dips (destabilizing)
    """
    
    @staticmethod
    def calculate_gex_profile(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
        """
        Calculate GEX at each strike
        
        Args:
            chain: Options chain with columns ['strike', 'gamma', 'oi_call', 'oi_put']
            spot: Current spot price
        
        Returns:
            DataFrame with GEX profile indexed by strike
        """
        result = chain.copy()
        
        # Dealers are SHORT options, so signs flip:
        # - Short calls = negative gamma for dealers (they're long stock)
        # - Short puts = positive gamma for dealers (they're short stock)
        
        result['gex_call'] = -result['gamma'] * result['oi_call'] * 100 * spot ** 2 / 100
        result['gex_put'] = result['gamma'] * result['oi_put'] * 100 * spot ** 2 / 100
        result['net_gex'] = result['gex_call'] + result['gex_put']
        
        return result.set_index('strike')[['gex_call', 'gex_put', 'net_gex']]
    
    @staticmethod
    def find_gamma_flip_point(gex_profile: pd.DataFrame, spot: float) -> Tuple[float, str]:
        """
        Find the strike where GEX flips from positive to negative
        
        Above flip point: Dealers stabilize (positive gamma)
        Below flip point: Dealers destabilize (negative gamma)
        
        Returns:
            (flip_strike, regime) where regime is 'long_gamma' or 'short_gamma'
        """
        # Find zero-crossing
        gex_series = gex_profile['net_gex'].sort_index()
        
        for i in range(len(gex_series) - 1):
            if gex_series.iloc[i] < 0 and gex_series.iloc[i + 1] >= 0:
                flip_strike = gex_series.index[i]
                regime = 'short_gamma' if spot < flip_strike else 'long_gamma'
                return flip_strike, regime
        
        # If no flip found, determine regime from total GEX
        total_gex = gex_series.sum()
        regime = 'long_gamma' if total_gex > 0 else 'short_gamma'
        return spot, regime
    
    @staticmethod
    def calculate_gex_concentration(gex_profile: pd.DataFrame, spot: float, window: float = 5.0) -> float:
        """
        Measure GEX concentration near spot
        
        High concentration = strong pinning force
        
        Args:
            gex_profile: GEX profile from calculate_gex_profile
            spot: Current spot price
            window: Strike window around spot (dollars)
        
        Returns:
            Concentration ratio (0 to 1)
        """
        near_spot = gex_profile[
            (gex_profile.index >= spot - window) & 
            (gex_profile.index <= spot + window)
        ]
        
        if len(near_spot) == 0:
            return 0.0
        
        concentration = near_spot['net_gex'].abs().sum() / gex_profile['net_gex'].abs().sum()
        return concentration


class CharmFlowEngine:
    """
    Detect charm-driven hedging flows
    
    Charm (delta decay) forces dealers to constantly rehedge as expiration approaches.
    For 0DTE options, charm flow dominates market microstructure in final hours.
    """
    
    @staticmethod
    def calculate_charm_pressure(chain: pd.DataFrame) -> float:
        """
        Calculate total charm pressure from options chain
        
        Positive charm = dealers must sell stock
        Negative charm = dealers must buy stock
        
        Args:
            chain: Options chain with columns ['charm', 'oi_call', 'oi_put']
        
        Returns:
            Net charm pressure (notional value)
        """
        # Dealers are short options, so flip signs
        charm_from_calls = -(chain['charm'] * chain['oi_call']).sum()
        charm_from_puts = -(chain['charm'] * chain['oi_put']).sum()
        
        total_charm = (charm_from_calls + charm_from_puts) * 100  # 100 shares per contract
        
        return total_charm
    
    @staticmethod
    def estimate_hourly_hedging_flow(chain: pd.DataFrame, spot: float) -> float:
        """
        Estimate shares dealers must trade per hour due to charm
        
        Returns:
            Estimated shares per hour
        """
        total_charm = CharmFlowEngine.calculate_charm_pressure(chain)
        
        # Charm is reported per day, convert to per hour
        # For 0DTE in final hours, this accelerates
        hours_remaining = chain['tte'].mean() * 24 * 365  # Convert years to hours
        
        if hours_remaining < 1:
            hours_remaining = 1
        
        hourly_flow = total_charm / hours_remaining
        
        return hourly_flow
    
    @staticmethod
    def detect_charm_nuclear_event(chain: pd.DataFrame, threshold: float = 1e9) -> bool:
        """
        Detect "charm nuclear" event - massive forced hedging
        
        Args:
            chain: Options chain
            threshold: Minimum charm pressure for nuclear event (default 1 billion shares)
        
        Returns:
            True if nuclear charm event detected
        """
        total_charm = abs(CharmFlowEngine.calculate_charm_pressure(chain))
        return total_charm > threshold


class VannaExposureEngine:
    """
    Detect vanna-driven spot/vol dynamics
    
    Vanna links volatility and delta. When IV changes, dealers must rehedge delta:
    - Positive vanna + rising IV = dealers buy spot (gamma squeeze fuel)
    - Negative vanna + rising IV = dealers sell spot (crash accelerator)
    """
    
    @staticmethod
    def calculate_vanna_exposure(chain: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate net vanna exposure
        
        Returns:
            (net_vanna, abs_vanna) where net can be positive/negative, abs is magnitude
        """
        # Dealers are short options
        vanna_from_calls = -(chain['vanna'] * chain['oi_call']).sum()
        vanna_from_puts = -(chain['vanna'] * chain['oi_put']).sum()
        
        net_vanna = (vanna_from_calls + vanna_from_puts) * 100
        abs_vanna = abs(net_vanna)
        
        return net_vanna, abs_vanna
    
    @staticmethod
    def predict_spot_move_from_vol_change(chain: pd.DataFrame, spot: float, 
                                          vol_change_pct: float) -> float:
        """
        Predict spot price move from volatility change via vanna
        
        Args:
            chain: Options chain
            spot: Current spot price
            vol_change_pct: Percentage change in IV (e.g., 0.10 for +10%)
        
        Returns:
            Expected spot price change (dollars)
        """
        net_vanna, _ = VannaExposureEngine.calculate_vanna_exposure(chain)
        
        # Vanna measures delta change per vol change
        # Convert to spot impact
        delta_change = net_vanna * vol_change_pct
        
        # Assume dealers hedge 80% of delta change
        hedge_ratio = 0.80
        
        spot_impact = (delta_change * hedge_ratio) / (spot * 0.01)  # Normalize
        
        return spot_impact
    
    @staticmethod
    def detect_vanna_shock_risk(chain: pd.DataFrame, threshold: float = 2e9) -> bool:
        """
        Detect potential for vanna-driven shock
        
        Returns:
            True if vanna exposure exceeds threshold
        """
        _, abs_vanna = VannaExposureEngine.calculate_vanna_exposure(chain)
        return abs_vanna > threshold


class DealerPositionEngine:
    """
    Estimate overall dealer positioning and hedging requirements
    """
    
    @staticmethod
    def estimate_net_dealer_delta(chain: pd.DataFrame, spot: float) -> float:
        """
        Estimate dealers' net delta position
        
        Returns:
            Net delta (positive = long, negative = short)
        """
        # Dealers are short options to customers
        call_delta = -(chain['delta'] * chain['oi_call']).sum()
        put_delta = -(chain['delta'] * chain['oi_put']).sum()
        
        net_delta = (call_delta + put_delta) * 100
        
        return net_delta
    
    @staticmethod
    def calculate_hedging_requirement(chain: pd.DataFrame, spot: float, 
                                     spot_move: float) -> float:
        """
        Calculate shares dealers must trade to maintain delta-neutral
        
        Args:
            chain: Options chain
            spot: Current spot price
            spot_move: Expected spot price move (dollars)
        
        Returns:
            Shares to trade (positive = buy, negative = sell)
        """
        # Current gamma exposure
        total_gamma = (chain['gamma'] * (chain['oi_call'] + chain['oi_put'])).sum() * 100
        
        # Dealers are short gamma, so flip sign
        dealer_gamma = -total_gamma
        
        # Delta change from spot move
        delta_change = dealer_gamma * spot_move
        
        return delta_change
    
    @staticmethod
    def assess_dealer_pressure(chain: pd.DataFrame, spot: float) -> Dict[str, any]:
        """
        Comprehensive dealer pressure assessment
        
        Returns:
            Dictionary with all pressure metrics
        """
        gex_engine = GammaExposureEngine()
        charm_engine = CharmFlowEngine()
        vanna_engine = VannaExposureEngine()
        
        gex_profile = gex_engine.calculate_gex_profile(chain, spot)
        flip_strike, regime = gex_engine.find_gamma_flip_point(gex_profile, spot)
        gex_concentration = gex_engine.calculate_gex_concentration(gex_profile, spot)
        
        charm_pressure = charm_engine.calculate_charm_pressure(chain)
        hourly_flow = charm_engine.estimate_hourly_hedging_flow(chain, spot)
        
        net_vanna, abs_vanna = vanna_engine.calculate_vanna_exposure(chain)
        
        net_delta = DealerPositionEngine.estimate_net_dealer_delta(chain, spot)
        
        return {
            'gamma_regime': regime,
            'gamma_flip_strike': flip_strike,
            'gex_concentration': gex_concentration,
            'charm_pressure': charm_pressure,
            'charm_hourly_flow': hourly_flow,
            'net_vanna': net_vanna,
            'abs_vanna': abs_vanna,
            'net_dealer_delta': net_delta,
            'total_open_interest': (chain['oi_call'].sum() + chain['oi_put'].sum())
        }


# Convenience function
def analyze_dealer_hedging(chain: pd.DataFrame, spot: float) -> Dict[str, any]:
    """
    Complete dealer hedging analysis
    
    Args:
        chain: Options chain with required columns
        spot: Current spot price
    
    Returns:
        Complete analysis dictionary
    """
    return DealerPositionEngine.assess_dealer_pressure(chain, spot)
