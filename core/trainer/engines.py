"""
PULSE Trading Platform - Greek Calculation Engines
Complete implementation of 13 Greek engines for options analysis
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import pandas as pd


class GreekEngines:
    """
    Complete Greek calculation engine suite.
    Implements all 13 Greeks: Delta, Gamma, Vega, Theta, Rho, Charm, Vanna, Vomma, Volga, Speed, Zomma, Color, Ultima
    """
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)
    
    # ==================================================================
    # FIRST-ORDER GREEKS
    # ==================================================================
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Delta (∂V/∂S) - Rate of change of option value with respect to underlying price
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Delta value
        """
        if T <= 0:
            return 1.0 if (S > K and option_type == 'call') else 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Gamma (∂²V/∂S²) - Rate of change of delta with respect to underlying price
        
        High gamma = large delta changes = high risk/reward for dealers
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vega (∂V/∂σ) - Sensitivity to volatility changes
        
        Returns vega per 1% change in volatility
        """
        if T <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Theta (∂V/∂t) - Time decay
        
        Returns theta per day
        """
        if T <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        d2 = GreekEngines._d2(S, K, T, r, sigma)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type.lower() == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            return (term1 + term2) / 365
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            return (term1 + term2) / 365
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Rho (∂V/∂r) - Sensitivity to interest rate changes
        
        Returns rho per 1% change in interest rate
        """
        if T <= 0:
            return 0.0
        
        d2 = GreekEngines._d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    # ==================================================================
    # SECOND-ORDER GREEKS
    # ==================================================================
    
    @staticmethod
    def charm(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Charm (∂Δ/∂t) - Delta decay
        
        Critical for 0DTE: measures how delta changes as time passes
        Dealers must hedge charm flow constantly near expiration
        
        Returns charm per day
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        d2 = GreekEngines._d2(S, K, T, r, sigma)
        
        term1 = norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T))
        term2 = 2 * T * sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            return -(term1 / term2) / 365
        else:
            return -(term1 / term2) / 365
    
    @staticmethod
    def vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vanna (∂Δ/∂σ = ∂V/∂S∂σ) - Delta sensitivity to volatility
        
        Critical for 0DTE: when vol spikes, vanna forces dealers to hedge
        Positive vanna + rising vol = dealers buy spot
        Negative vanna + rising vol = dealers sell spot
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        d2 = GreekEngines._d2(S, K, T, r, sigma)
        
        return -norm.pdf(d1) * d2 / (sigma * 100)
    
    @staticmethod
    def vomma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vomma/Volga (∂²V/∂σ²) - Vega sensitivity to volatility
        
        Measures convexity of vega
        High vomma = vega changes rapidly with vol changes
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        d2 = GreekEngines._d2(S, K, T, r, sigma)
        
        vega_val = GreekEngines.vega(S, K, T, r, sigma)
        return vega_val * (d1 * d2) / sigma
    
    @staticmethod
    def volga(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Alias for vomma"""
        return GreekEngines.vomma(S, K, T, r, sigma)
    
    # ==================================================================
    # THIRD-ORDER GREEKS
    # ==================================================================
    
    @staticmethod
    def speed(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Speed (∂³V/∂S³ = ∂Γ/∂S) - Gamma sensitivity to spot
        
        Measures how quickly gamma changes
        Critical for large moves in 0DTE
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        gamma_val = GreekEngines.gamma(S, K, T, r, sigma)
        
        return -gamma_val * (d1 / (S * sigma * np.sqrt(T)) + 1 / S)
    
    @staticmethod
    def zomma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Zomma (∂²Δ/∂S∂σ = ∂Γ/∂σ) - Gamma sensitivity to volatility
        
        Critical for vol-driven gamma squeezes
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        d2 = GreekEngines._d2(S, K, T, r, sigma)
        gamma_val = GreekEngines.gamma(S, K, T, r, sigma)
        
        return gamma_val * (d1 * d2 - 1) / sigma
    
    @staticmethod
    def color(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Color (∂³V/∂S²∂t = ∂Γ/∂t) - Gamma decay
        
        Critical for 0DTE: measures how gamma changes with time
        Dealers face massive color exposure near expiration
        
        Returns color per day
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        d2 = GreekEngines._d2(S, K, T, r, sigma)
        
        term1 = 2 * r * T + 1
        term2 = d1 * (2 * r * T - d2 * sigma * np.sqrt(T))
        
        gamma_val = GreekEngines.gamma(S, K, T, r, sigma)
        
        return -(gamma_val / (2 * T)) * (term1 + term2) / 365
    
    @staticmethod
    def ultima(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Ultima (∂³V/∂σ³) - Vomma sensitivity to volatility
        
        Third derivative of option value with respect to volatility
        Measures vol convexity changes
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = GreekEngines._d1(S, K, T, r, sigma)
        d2 = GreekEngines._d2(S, K, T, r, sigma)
        
        vega_val = GreekEngines.vega(S, K, T, r, sigma)
        
        term = d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2
        return -vega_val * term / (sigma ** 2)
    
    # ==================================================================
    # BATCH CALCULATION
    # ==================================================================
    
    @staticmethod
    def calculate_all_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                            option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all 13 Greeks at once
        
        Returns:
            Dictionary with all Greek values
        """
        return {
            # First-order
            'delta': GreekEngines.delta(S, K, T, r, sigma, option_type),
            'gamma': GreekEngines.gamma(S, K, T, r, sigma),
            'vega': GreekEngines.vega(S, K, T, r, sigma),
            'theta': GreekEngines.theta(S, K, T, r, sigma, option_type),
            'rho': GreekEngines.rho(S, K, T, r, sigma, option_type),
            
            # Second-order
            'charm': GreekEngines.charm(S, K, T, r, sigma, option_type),
            'vanna': GreekEngines.vanna(S, K, T, r, sigma),
            'vomma': GreekEngines.vomma(S, K, T, r, sigma),
            'volga': GreekEngines.volga(S, K, T, r, sigma),
            
            # Third-order
            'speed': GreekEngines.speed(S, K, T, r, sigma),
            'zomma': GreekEngines.zomma(S, K, T, r, sigma),
            'color': GreekEngines.color(S, K, T, r, sigma),
            'ultima': GreekEngines.ultima(S, K, T, r, sigma)
        }
    
    @staticmethod
    def calculate_chain_greeks(chain: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
        """
        Calculate all Greeks for an entire options chain
        
        Args:
            chain: DataFrame with columns ['strike', 'iv', 'tte', 'option_type']
            spot: Current spot price
            r: Risk-free rate
        
        Returns:
            DataFrame with all Greek columns added
        """
        result = chain.copy()
        
        for greek in ['delta', 'gamma', 'vega', 'theta', 'rho', 'charm', 'vanna', 
                     'vomma', 'speed', 'zomma', 'color', 'ultima']:
            result[greek] = 0.0
        
        for idx, row in result.iterrows():
            greeks = GreekEngines.calculate_all_greeks(
                S=spot,
                K=row['strike'],
                T=row['tte'],
                r=r,
                sigma=row['iv'],
                option_type=row.get('option_type', 'call')
            )
            
            for greek, value in greeks.items():
                result.at[idx, greek] = value
        
        return result


# ==================================================================
# CONVENIENCE FUNCTIONS
# ==================================================================

def calculate_portfolio_greeks(positions: list, spot: float, r: float = 0.05) -> Dict[str, float]:
    """
    Calculate net Greeks for a portfolio of options
    
    Args:
        positions: List of dicts with keys: strike, tte, iv, option_type, quantity
        spot: Current spot price
        r: Risk-free rate
    
    Returns:
        Dictionary with net Greek exposures
    """
    net_greeks = {
        'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0,
        'charm': 0.0, 'vanna': 0.0, 'vomma': 0.0, 
        'speed': 0.0, 'zomma': 0.0, 'color': 0.0, 'ultima': 0.0
    }
    
    for pos in positions:
        greeks = GreekEngines.calculate_all_greeks(
            S=spot,
            K=pos['strike'],
            T=pos['tte'],
            r=r,
            sigma=pos['iv'],
            option_type=pos['option_type']
        )
        
        quantity = pos.get('quantity', 1)
        for greek, value in greeks.items():
            net_greeks[greek] += value * quantity
    
    return net_greeks