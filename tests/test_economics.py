"""
PULSE Economics Testing Module

Comprehensive test suite for economic calculations including:
- Time savings analysis
- ROI (Return on Investment) calculations
- Payback period analysis
- Efficiency gains measurement
- Break-even analysis
- Decimal precision validation

Author: jetgause
Created: 2025-12-10
"""

import unittest
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple


class EconomicsCalculator:
    """
    Calculator for PULSE economic metrics and financial analysis.
    
    Provides methods for calculating time savings, ROI, payback periods,
    efficiency gains, and break-even points with high decimal precision.
    """
    
    def __init__(self, precision: int = 10):
        """
        Initialize the EconomicsCalculator.
        
        Args:
            precision: Decimal precision for calculations (default: 10)
        """
        self.precision = precision
        getcontext().prec = precision
    
    def calculate_time_savings(
        self,
        manual_time: float,
        automated_time: float,
        frequency: int = 1
    ) -> Dict[str, Decimal]:
        """
        Calculate time savings from automation.
        
        Args:
            manual_time: Time for manual process (hours)
            automated_time: Time for automated process (hours)
            frequency: Number of times process runs per period
            
        Returns:
            Dictionary with savings metrics
        """
        manual = Decimal(str(manual_time))
        automated = Decimal(str(automated_time))
        freq = Decimal(str(frequency))
        
        savings_per_run = manual - automated
        total_savings = savings_per_run * freq
        percentage = (savings_per_run / manual * 100) if manual > 0 else Decimal('0')
        
        return {
            'savings_per_run': savings_per_run,
            'total_savings': total_savings,
            'percentage_saved': percentage,
            'efficiency_ratio': manual / automated if automated > 0 else Decimal('0')
        }
    
    def calculate_roi(
        self,
        initial_investment: float,
        returns: float,
        time_period: float = 1.0
    ) -> Dict[str, Decimal]:
        """
        Calculate Return on Investment (ROI).
        
        Args:
            initial_investment: Initial cost/investment
            returns: Total returns/benefits
            time_period: Time period in years
            
        Returns:
            Dictionary with ROI metrics
        """
        investment = Decimal(str(initial_investment))
        ret = Decimal(str(returns))
        period = Decimal(str(time_period))
        
        net_return = ret - investment
        roi_percentage = (net_return / investment * 100) if investment > 0 else Decimal('0')
        annualized_roi = roi_percentage / period if period > 0 else Decimal('0')
        
        return {
            'net_return': net_return,
            'roi_percentage': roi_percentage,
            'annualized_roi': annualized_roi,
            'return_multiple': ret / investment if investment > 0 else Decimal('0')
        }
    
    def calculate_payback_period(
        self,
        initial_investment: float,
        periodic_savings: float,
        discount_rate: float = 0.0
    ) -> Dict[str, Decimal]:
        """
        Calculate payback period for investment.
        
        Args:
            initial_investment: Initial cost
            periodic_savings: Savings per period (e.g., monthly)
            discount_rate: Discount rate for NPV calculation
            
        Returns:
            Dictionary with payback metrics
        """
        investment = Decimal(str(initial_investment))
        savings = Decimal(str(periodic_savings))
        rate = Decimal(str(discount_rate))
        
        if savings <= 0:
            return {
                'simple_payback': Decimal('inf'),
                'discounted_payback': Decimal('inf'),
                'break_even_point': Decimal('inf')
            }
        
        simple_payback = investment / savings
        
        # Calculate discounted payback
        if rate > 0:
            cumulative = Decimal('0')
            period = 0
            while cumulative < investment and period < 1000:
                period += 1
                discount_factor = (Decimal('1') + rate) ** period
                cumulative += savings / discount_factor
            discounted_payback = Decimal(str(period)) if cumulative >= investment else Decimal('inf')
        else:
            discounted_payback = simple_payback
        
        return {
            'simple_payback': simple_payback,
            'discounted_payback': discounted_payback,
            'break_even_point': simple_payback
        }
    
    def calculate_efficiency_gains(
        self,
        baseline_output: float,
        improved_output: float,
        baseline_input: float = 1.0,
        improved_input: float = 1.0
    ) -> Dict[str, Decimal]:
        """
        Calculate efficiency improvements.
        
        Args:
            baseline_output: Output before improvement
            improved_output: Output after improvement
            baseline_input: Input before improvement
            improved_input: Input after improvement
            
        Returns:
            Dictionary with efficiency metrics
        """
        base_out = Decimal(str(baseline_output))
        imp_out = Decimal(str(improved_output))
        base_in = Decimal(str(baseline_input))
        imp_in = Decimal(str(improved_input))
        
        baseline_efficiency = base_out / base_in if base_in > 0 else Decimal('0')
        improved_efficiency = imp_out / imp_in if imp_in > 0 else Decimal('0')
        
        efficiency_gain = improved_efficiency - baseline_efficiency
        percentage_improvement = (
            (efficiency_gain / baseline_efficiency * 100)
            if baseline_efficiency > 0 else Decimal('0')
        )
        
        return {
            'baseline_efficiency': baseline_efficiency,
            'improved_efficiency': improved_efficiency,
            'efficiency_gain': efficiency_gain,
            'percentage_improvement': percentage_improvement,
            'productivity_ratio': improved_efficiency / baseline_efficiency if baseline_efficiency > 0 else Decimal('0')
        }
    
    def calculate_break_even(
        self,
        fixed_costs: float,
        variable_cost_per_unit: float,
        price_per_unit: float
    ) -> Dict[str, Decimal]:
        """
        Calculate break-even point.
        
        Args:
            fixed_costs: Total fixed costs
            variable_cost_per_unit: Variable cost per unit
            price_per_unit: Selling price per unit
            
        Returns:
            Dictionary with break-even metrics
        """
        fixed = Decimal(str(fixed_costs))
        variable = Decimal(str(variable_cost_per_unit))
        price = Decimal(str(price_per_unit))
        
        contribution_margin = price - variable
        
        if contribution_margin <= 0:
            return {
                'break_even_units': Decimal('inf'),
                'break_even_revenue': Decimal('inf'),
                'contribution_margin': contribution_margin,
                'contribution_margin_ratio': Decimal('0')
            }
        
        break_even_units = fixed / contribution_margin
        break_even_revenue = break_even_units * price
        contribution_margin_ratio = contribution_margin / price if price > 0 else Decimal('0')
        
        return {
            'break_even_units': break_even_units,
            'break_even_revenue': break_even_revenue,
            'contribution_margin': contribution_margin,
            'contribution_margin_ratio': contribution_margin_ratio
        }
    
    def calculate_net_present_value(
        self,
        cash_flows: List[float],
        discount_rate: float
    ) -> Decimal:
        """
        Calculate Net Present Value (NPV) of cash flows.
        
        Args:
            cash_flows: List of cash flows (first is initial investment)
            discount_rate: Discount rate
            
        Returns:
            Net present value
        """
        rate = Decimal(str(discount_rate))
        npv = Decimal('0')
        
        for period, cash_flow in enumerate(cash_flows):
            cf = Decimal(str(cash_flow))
            discount_factor = (Decimal('1') + rate) ** period
            npv += cf / discount_factor
        
        return npv
    
    def calculate_total_cost_of_ownership(
        self,
        acquisition_cost: float,
        annual_operating_cost: float,
        years: int,
        salvage_value: float = 0.0,
        discount_rate: float = 0.0
    ) -> Dict[str, Decimal]:
        """
        Calculate Total Cost of Ownership (TCO).
        
        Args:
            acquisition_cost: Initial acquisition cost
            annual_operating_cost: Annual operating/maintenance cost
            years: Number of years
            salvage_value: Residual value at end of period
            discount_rate: Discount rate for present value calculation
            
        Returns:
            Dictionary with TCO metrics
        """
        acquisition = Decimal(str(acquisition_cost))
        operating = Decimal(str(annual_operating_cost))
        salvage = Decimal(str(salvage_value))
        rate = Decimal(str(discount_rate))
        
        # Calculate nominal TCO
        total_operating = operating * years
        nominal_tco = acquisition + total_operating - salvage
        
        # Calculate present value TCO
        if rate > 0:
            pv_operating = Decimal('0')
            for year in range(1, years + 1):
                pv_operating += operating / ((Decimal('1') + rate) ** year)
            pv_salvage = salvage / ((Decimal('1') + rate) ** years)
            present_value_tco = acquisition + pv_operating - pv_salvage
        else:
            present_value_tco = nominal_tco
        
        annual_equivalent = nominal_tco / years if years > 0 else Decimal('0')
        
        return {
            'nominal_tco': nominal_tco,
            'present_value_tco': present_value_tco,
            'annual_equivalent': annual_equivalent,
            'total_operating_cost': total_operating
        }


class TestEconomicsCalculator(unittest.TestCase):
    """
    Comprehensive test suite for EconomicsCalculator.
    
    Tests 27 scenarios covering:
    - Time savings calculations
    - ROI analysis
    - Payback period calculations
    - Efficiency gains
    - Break-even analysis
    - Decimal precision
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = EconomicsCalculator(precision=10)
    
    # Time Savings Tests (Test 1-5)
    
    def test_01_basic_time_savings(self):
        """Test basic time savings calculation."""
        result = self.calculator.calculate_time_savings(
            manual_time=10.0,
            automated_time=2.0,
            frequency=1
        )
        self.assertEqual(result['savings_per_run'], Decimal('8.0'))
        self.assertEqual(result['total_savings'], Decimal('8.0'))
        self.assertEqual(result['percentage_saved'], Decimal('80.0'))
    
    def test_02_time_savings_with_frequency(self):
        """Test time savings with multiple frequencies."""
        result = self.calculator.calculate_time_savings(
            manual_time=5.0,
            automated_time=1.0,
            frequency=100
        )
        self.assertEqual(result['savings_per_run'], Decimal('4.0'))
        self.assertEqual(result['total_savings'], Decimal('400.0'))
    
    def test_03_time_savings_efficiency_ratio(self):
        """Test efficiency ratio calculation."""
        result = self.calculator.calculate_time_savings(
            manual_time=20.0,
            automated_time=5.0
        )
        self.assertEqual(result['efficiency_ratio'], Decimal('4.0'))
    
    def test_04_time_savings_zero_manual(self):
        """Test time savings with zero manual time."""
        result = self.calculator.calculate_time_savings(
            manual_time=0.0,
            automated_time=5.0
        )
        self.assertEqual(result['percentage_saved'], Decimal('0'))
    
    def test_05_time_savings_decimal_precision(self):
        """Test time savings with high decimal precision."""
        result = self.calculator.calculate_time_savings(
            manual_time=3.333333,
            automated_time=1.111111
        )
        self.assertAlmostEqual(
            float(result['savings_per_run']),
            2.222222,
            places=5
        )
    
    # ROI Tests (Test 6-10)
    
    def test_06_basic_roi_positive(self):
        """Test basic positive ROI calculation."""
        result = self.calculator.calculate_roi(
            initial_investment=10000.0,
            returns=15000.0
        )
        self.assertEqual(result['net_return'], Decimal('5000.0'))
        self.assertEqual(result['roi_percentage'], Decimal('50.0'))
    
    def test_07_roi_negative(self):
        """Test negative ROI calculation."""
        result = self.calculator.calculate_roi(
            initial_investment=10000.0,
            returns=8000.0
        )
        self.assertEqual(result['net_return'], Decimal('-2000.0'))
        self.assertEqual(result['roi_percentage'], Decimal('-20.0'))
    
    def test_08_roi_annualized(self):
        """Test annualized ROI calculation."""
        result = self.calculator.calculate_roi(
            initial_investment=10000.0,
            returns=15000.0,
            time_period=2.0
        )
        self.assertEqual(result['annualized_roi'], Decimal('25.0'))
    
    def test_09_roi_return_multiple(self):
        """Test return multiple calculation."""
        result = self.calculator.calculate_roi(
            initial_investment=5000.0,
            returns=20000.0
        )
        self.assertEqual(result['return_multiple'], Decimal('4.0'))
    
    def test_10_roi_zero_investment(self):
        """Test ROI with zero investment."""
        result = self.calculator.calculate_roi(
            initial_investment=0.0,
            returns=5000.0
        )
        self.assertEqual(result['roi_percentage'], Decimal('0'))
    
    # Payback Period Tests (Test 11-15)
    
    def test_11_simple_payback_period(self):
        """Test simple payback period calculation."""
        result = self.calculator.calculate_payback_period(
            initial_investment=10000.0,
            periodic_savings=1000.0
        )
        self.assertEqual(result['simple_payback'], Decimal('10.0'))
    
    def test_12_payback_period_with_discount(self):
        """Test discounted payback period."""
        result = self.calculator.calculate_payback_period(
            initial_investment=10000.0,
            periodic_savings=2000.0,
            discount_rate=0.1
        )
        self.assertGreater(result['discounted_payback'], result['simple_payback'])
    
    def test_13_payback_period_zero_savings(self):
        """Test payback period with zero savings."""
        result = self.calculator.calculate_payback_period(
            initial_investment=10000.0,
            periodic_savings=0.0
        )
        self.assertEqual(result['simple_payback'], Decimal('inf'))
    
    def test_14_payback_period_negative_savings(self):
        """Test payback period with negative savings."""
        result = self.calculator.calculate_payback_period(
            initial_investment=10000.0,
            periodic_savings=-500.0
        )
        self.assertEqual(result['simple_payback'], Decimal('inf'))
    
    def test_15_payback_break_even_point(self):
        """Test break-even point in payback calculation."""
        result = self.calculator.calculate_payback_period(
            initial_investment=5000.0,
            periodic_savings=500.0
        )
        self.assertEqual(result['break_even_point'], Decimal('10.0'))
    
    # Efficiency Gains Tests (Test 16-20)
    
    def test_16_basic_efficiency_gains(self):
        """Test basic efficiency gains calculation."""
        result = self.calculator.calculate_efficiency_gains(
            baseline_output=100.0,
            improved_output=150.0
        )
        self.assertEqual(result['baseline_efficiency'], Decimal('100.0'))
        self.assertEqual(result['improved_efficiency'], Decimal('150.0'))
        self.assertEqual(result['efficiency_gain'], Decimal('50.0'))
    
    def test_17_efficiency_percentage_improvement(self):
        """Test efficiency percentage improvement."""
        result = self.calculator.calculate_efficiency_gains(
            baseline_output=200.0,
            improved_output=300.0
        )
        self.assertEqual(result['percentage_improvement'], Decimal('50.0'))
    
    def test_18_efficiency_with_varying_inputs(self):
        """Test efficiency with different inputs."""
        result = self.calculator.calculate_efficiency_gains(
            baseline_output=100.0,
            improved_output=120.0,
            baseline_input=10.0,
            improved_input=8.0
        )
        self.assertEqual(result['baseline_efficiency'], Decimal('10.0'))
        self.assertEqual(result['improved_efficiency'], Decimal('15.0'))
    
    def test_19_efficiency_productivity_ratio(self):
        """Test productivity ratio calculation."""
        result = self.calculator.calculate_efficiency_gains(
            baseline_output=50.0,
            improved_output=100.0
        )
        self.assertEqual(result['productivity_ratio'], Decimal('2.0'))
    
    def test_20_efficiency_zero_baseline(self):
        """Test efficiency with zero baseline."""
        result = self.calculator.calculate_efficiency_gains(
            baseline_output=0.0,
            improved_output=100.0
        )
        self.assertEqual(result['percentage_improvement'], Decimal('0'))
    
    # Break-Even Analysis Tests (Test 21-24)
    
    def test_21_basic_break_even(self):
        """Test basic break-even calculation."""
        result = self.calculator.calculate_break_even(
            fixed_costs=10000.0,
            variable_cost_per_unit=5.0,
            price_per_unit=15.0
        )
        self.assertEqual(result['break_even_units'], Decimal('1000.0'))
        self.assertEqual(result['contribution_margin'], Decimal('10.0'))
    
    def test_22_break_even_revenue(self):
        """Test break-even revenue calculation."""
        result = self.calculator.calculate_break_even(
            fixed_costs=5000.0,
            variable_cost_per_unit=10.0,
            price_per_unit=20.0
        )
        self.assertEqual(result['break_even_revenue'], Decimal('10000.0'))
    
    def test_23_break_even_contribution_margin_ratio(self):
        """Test contribution margin ratio."""
        result = self.calculator.calculate_break_even(
            fixed_costs=8000.0,
            variable_cost_per_unit=6.0,
            price_per_unit=10.0
        )
        self.assertEqual(result['contribution_margin_ratio'], Decimal('0.4'))
    
    def test_24_break_even_negative_margin(self):
        """Test break-even with negative contribution margin."""
        result = self.calculator.calculate_break_even(
            fixed_costs=5000.0,
            variable_cost_per_unit=20.0,
            price_per_unit=15.0
        )
        self.assertEqual(result['break_even_units'], Decimal('inf'))
    
    # Advanced Calculations Tests (Test 25-27)
    
    def test_25_net_present_value(self):
        """Test NPV calculation with multiple cash flows."""
        cash_flows = [-10000.0, 3000.0, 4000.0, 5000.0, 6000.0]
        npv = self.calculator.calculate_net_present_value(
            cash_flows=cash_flows,
            discount_rate=0.1
        )
        self.assertGreater(npv, Decimal('0'))
        self.assertLess(npv, Decimal('8000.0'))
    
    def test_26_total_cost_of_ownership(self):
        """Test TCO calculation."""
        result = self.calculator.calculate_total_cost_of_ownership(
            acquisition_cost=50000.0,
            annual_operating_cost=5000.0,
            years=5,
            salvage_value=10000.0
        )
        self.assertEqual(result['nominal_tco'], Decimal('65000.0'))
        self.assertEqual(result['total_operating_cost'], Decimal('25000.0'))
        self.assertEqual(result['annual_equivalent'], Decimal('13000.0'))
    
    def test_27_tco_with_discount_rate(self):
        """Test TCO with discount rate for present value."""
        result = self.calculator.calculate_total_cost_of_ownership(
            acquisition_cost=100000.0,
            annual_operating_cost=10000.0,
            years=5,
            salvage_value=20000.0,
            discount_rate=0.05
        )
        self.assertLess(
            result['present_value_tco'],
            result['nominal_tco']
        )


def run_test_suite():
    """Run the complete test suite with verbose output."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEconomicsCalculator)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    # Run tests
    print("=" * 70)
    print("PULSE Economics Testing Module - Comprehensive Test Suite")
    print("=" * 70)
    print(f"\nRunning 27 test cases...")
    print("-" * 70)
    
    result = run_test_suite()
    
    print("\n" + "=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
