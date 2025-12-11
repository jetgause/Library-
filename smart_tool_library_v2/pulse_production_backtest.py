import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
import random

# Current date for simulation
CURRENT_DATE = pd.Timestamp('2025-12-11 21:58:18')

class OptionsChainGenerator:
    def __init__(self, underlying_price, volatility, time_to_expiry_days, risk_free_rate=0.02):
        self.S = underlying_price
        self.vol = volatility
        self.T = time_to_expiry_days / 365  # Time in years
        self.r = risk_free_rate
        self.strikes = self._generate_strikes()

    def _generate_strikes(self):
        # Generate strikes around underlying price, with more density near ATM
        strikes = []
        for i in np.arange(-5, 6, 0.5):
            strike = self.S * (1 + i * 0.01)
            strikes.append(round(strike, 2))
        return sorted(strikes)

    def _black_scholes_price(self, K, option_type='call'):
        d1 = (np.log(self.S / K) + (self.r + 0.5 * self.vol**2) * self.T) / (self.vol * np.sqrt(self.T))
        d2 = d1 - self.vol * np.sqrt(self.T)
        if option_type == 'call':
            return self.S * norm.cdf(d1) - K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def _calculate_oi(self, strike, option_type, is_extreme_event=False):
        # Base OI model: Gaussian around ATM, skewed for puts
        distance = abs(np.log(strike / self.S)) / (self.vol * np.sqrt(self.T))
        base_oi = np.exp(-0.5 * distance**2) * 1000  # Base OI

        # For puts, slightly higher OI for OTM
        if option_type == 'put' and strike < self.S:
            base_oi *= 1.2

        # Extreme event adjustment: Reduce OI for OTM options, increase for ATM
        if is_extreme_event:
            if abs(strike - self.S) / self.S > 0.05:  # OTM
                base_oi *= 0.3  # Much lower OI for OTM in extreme events
            else:
                base_oi *= 2  # Higher OI for ATM

        return max(int(base_oi), 1)  # At least 1

    def generate_chain(self, is_extreme_event=False):
        chain = []
        for strike in self.strikes:
            call_price = self._black_scholes_price(strike, 'call')
            put_price = self._black_scholes_price(strike, 'put')
            call_oi = self._calculate_oi(strike, 'call', is_extreme_event)
            put_oi = self._calculate_oi(strike, 'put', is_extreme_event)
            chain.append({
                'strike': strike,
                'call_price': round(call_price, 2),
                'put_price': round(put_price, 2),
                'call_oi': call_oi,
                'put_oi': put_oi
            })
        return pd.DataFrame(chain)

class ProductionBacktest:
    def __init__(self, initial_capital=100000, underlying_price=100, volatility=0.2, time_to_expiry_days=30):
        self.capital = initial_capital
        self.underlying_price = underlying_price
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry_days
        self.generator = OptionsChainGenerator(underlying_price, volatility, time_to_expiry_days)
        self.positions = []  # List of positions: {'type': 'call'/'put', 'strike': , 'quantity': , 'entry_price': }
        self.results = []

    def generate_extreme_event(self):
        # Simulate extreme event: sudden volatility spike and price jump
        return random.choice([True, False])  # 50% chance

    def run_backtest(self, days=30):
        for day in range(days):
            is_extreme = self.generate_extreme_event()
            chain = self.generator.generate_chain(is_extreme)
            
            # Simple strategy: Buy ATM call if extreme event
            if is_extreme:
                atm_strike = min(chain['strike'], key=lambda x: abs(x - self.underlying_price))
                atm_call = chain[chain['strike'] == atm_strike].iloc[0]
                if self.capital > atm_call['call_price'] * 100:  # 1 lot
                    self.positions.append({
                        'type': 'call',
                        'strike': atm_strike,
                        'quantity': 1,
                        'entry_price': atm_call['call_price']
                    })
                    self.capital -= atm_call['call_price'] * 100
            
            # Update underlying price (random walk with drift)
            drift = 0.01 if is_extreme else 0.0
            shock = np.random.normal(0, self.volatility * np.sqrt(1/365)) + drift
            self.underlying_price *= (1 + shock)
            
            # Update generator
            self.generator = OptionsChainGenerator(self.underlying_price, self.volatility, self.time_to_expiry - day)
            
            # Calculate P&L at end of day (simplified, assuming no exit)
            daily_pnl = 0
            # For simplicity, no P&L calculation yet
            
            self.results.append({
                'day': day,
                'underlying_price': self.underlying_price,
                'capital': self.capital,
                'is_extreme': is_extreme,
                'positions_count': len(self.positions)
            })
        
        return pd.DataFrame(self.results)

# Example usage
if __name__ == "__main__":
    backtest = ProductionBacktest()
    results = backtest.run_backtest()
    print(results.head())
    print("Backtest completed.")
