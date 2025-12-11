"""
Optimization Standards Engine
=============================

Defines and evaluates standards that local optimizations must meet
to be promoted to permanent status in the Smart Marketplace.

Standards ensure quality and prevent regressions in tool performance.

Author: jetgause
Created: 2025-12-10
"""

import statistics
from typing import Any, Dict, List, Optional

from .models import (
    OptimizationRecord,
    OptimizationStandard,
    OptimizationStatus,
)


class OptimizationStandardsEngine:
    """
    Engine for defining and evaluating optimization standards.
    
    Local optimizations can only become permanent if they meet
    the defined standards for improvement, stability, and
    non-regression.
    """
    
    # Default standards configuration
    DEFAULT_STANDARDS = {
        "minimum_improvement": OptimizationStandard(
            name="Minimum Improvement Threshold",
            description="Optimizations must achieve at least 15% improvement",
            min_improvement_threshold=15.0,
            required_metrics=["success_rate"],
            stability_threshold=5.0,
            min_sample_size=100,
            regression_tolerance=2.0,
        ),
        "performance_standard": OptimizationStandard(
            name="Performance Standard",
            description="Execution time must not increase significantly",
            min_improvement_threshold=10.0,
            required_metrics=["avg_execution_time"],
            stability_threshold=10.0,
            min_sample_size=50,
            regression_tolerance=5.0,
        ),
        "stability_standard": OptimizationStandard(
            name="Stability Standard",
            description="Results must be stable across runs",
            min_improvement_threshold=5.0,
            required_metrics=["success_rate", "avg_execution_time"],
            stability_threshold=3.0,
            min_sample_size=200,
            regression_tolerance=1.0,
        ),
    }
    
    def __init__(self, custom_standards: Optional[Dict[str, OptimizationStandard]] = None):
        """
        Initialize the OptimizationStandardsEngine.
        
        Args:
            custom_standards: Optional custom standards to use instead of defaults
        """
        self._standards: Dict[str, OptimizationStandard] = {}
        self._evaluation_history: List[Dict[str, Any]] = []
        
        # Load standards
        if custom_standards:
            self._standards = custom_standards
        else:
            self._standards = self.DEFAULT_STANDARDS.copy()
    
    def add_standard(self, standard_id: str, standard: OptimizationStandard):
        """
        Add a new optimization standard.
        
        Args:
            standard_id: Unique identifier for the standard
            standard: OptimizationStandard instance
        """
        self._standards[standard_id] = standard
    
    def remove_standard(self, standard_id: str) -> bool:
        """
        Remove an optimization standard.
        
        Args:
            standard_id: ID of standard to remove
            
        Returns:
            True if removed, False if not found
        """
        if standard_id in self._standards:
            del self._standards[standard_id]
            return True
        return False
    
    def get_standard(self, standard_id: str) -> Optional[OptimizationStandard]:
        """Get a standard by ID."""
        return self._standards.get(standard_id)
    
    def get_all_standards(self, active_only: bool = True) -> Dict[str, OptimizationStandard]:
        """
        Get all standards.
        
        Args:
            active_only: If True, return only active standards
            
        Returns:
            Dictionary of standards
        """
        if active_only:
            return {k: v for k, v in self._standards.items() if v.is_active}
        return self._standards.copy()
    
    def evaluate(
        self,
        optimization: OptimizationRecord,
        standards_to_check: Optional[List[str]] = None
    ) -> bool:
        """
        Evaluate if an optimization meets standards for permanent promotion.
        
        Args:
            optimization: OptimizationRecord to evaluate
            standards_to_check: Optional list of specific standard IDs to check
            
        Returns:
            True if optimization meets all required standards
        """
        if optimization.status != OptimizationStatus.COMPLETED:
            return False
        
        # Calculate improvement if not already done
        if optimization.improvement_percentage == 0.0:
            optimization.calculate_improvement()
        
        standards = self._get_standards_to_check(standards_to_check)
        evaluation_results = []
        
        for standard_id, standard in standards.items():
            result = self._evaluate_standard(optimization, standard)
            evaluation_results.append({
                "standard_id": standard_id,
                "standard_name": standard.name,
                "passed": result["passed"],
                "details": result,
            })
        
        # Record evaluation
        self._evaluation_history.append({
            "record_id": optimization.record_id,
            "tool_id": optimization.tool_id,
            "results": evaluation_results,
            "overall_passed": all(r["passed"] for r in evaluation_results),
        })
        
        # All standards must pass
        return all(r["passed"] for r in evaluation_results)
    
    def _get_standards_to_check(
        self,
        standards_to_check: Optional[List[str]]
    ) -> Dict[str, OptimizationStandard]:
        """Get the standards to evaluate."""
        if standards_to_check:
            return {
                k: v for k, v in self._standards.items()
                if k in standards_to_check and v.is_active
            }
        return {k: v for k, v in self._standards.items() if v.is_active}
    
    def _evaluate_standard(
        self,
        optimization: OptimizationRecord,
        standard: OptimizationStandard
    ) -> Dict[str, Any]:
        """
        Evaluate an optimization against a single standard.
        
        Args:
            optimization: Optimization to evaluate
            standard: Standard to check against
            
        Returns:
            Evaluation result dictionary
        """
        result = {
            "passed": True,
            "checks": [],
        }
        
        # Check 1: Minimum improvement threshold
        improvement_check = self._check_improvement_threshold(
            optimization, standard
        )
        result["checks"].append(improvement_check)
        if not improvement_check["passed"]:
            result["passed"] = False
        
        # Check 2: Required metrics improved
        metrics_check = self._check_required_metrics(
            optimization, standard
        )
        result["checks"].append(metrics_check)
        if not metrics_check["passed"]:
            result["passed"] = False
        
        # Check 3: No significant regressions
        regression_check = self._check_regression(
            optimization, standard
        )
        result["checks"].append(regression_check)
        if not regression_check["passed"]:
            result["passed"] = False
        
        return result
    
    def _check_improvement_threshold(
        self,
        optimization: OptimizationRecord,
        standard: OptimizationStandard
    ) -> Dict[str, Any]:
        """Check if improvement meets minimum threshold."""
        threshold = standard.min_improvement_threshold
        improvement = optimization.improvement_percentage
        
        return {
            "check": "improvement_threshold",
            "passed": improvement >= threshold,
            "threshold": threshold,
            "actual": improvement,
            "message": f"Improvement {improvement:.2f}% vs required {threshold:.2f}%"
        }
    
    def _check_required_metrics(
        self,
        optimization: OptimizationRecord,
        standard: OptimizationStandard
    ) -> Dict[str, Any]:
        """Check if required metrics have improved."""
        before = optimization.before_metrics
        after = optimization.after_metrics
        required = standard.required_metrics
        
        improved_metrics = []
        failed_metrics = []
        
        for metric in required:
            if metric not in before or metric not in after:
                # Metric not available, skip
                continue
            
            before_val = before[metric]
            after_val = after[metric]
            
            # Determine if improvement is positive or negative direction
            if metric in ["avg_execution_time", "latency", "error_rate"]:
                # Lower is better
                improved = after_val < before_val
            else:
                # Higher is better
                improved = after_val > before_val
            
            if improved:
                improved_metrics.append(metric)
            else:
                failed_metrics.append(metric)
        
        return {
            "check": "required_metrics",
            "passed": len(failed_metrics) == 0,
            "improved": improved_metrics,
            "failed": failed_metrics,
            "message": f"Improved: {improved_metrics}, Failed: {failed_metrics}"
        }
    
    def _check_regression(
        self,
        optimization: OptimizationRecord,
        standard: OptimizationStandard
    ) -> Dict[str, Any]:
        """Check for significant regressions in any metric."""
        before = optimization.before_metrics
        after = optimization.after_metrics
        tolerance = standard.regression_tolerance
        
        regressions = []
        
        for metric, before_val in before.items():
            if metric not in after:
                continue
            
            after_val = after[metric]
            
            # Determine if this is a regression based on metric type
            # For time-based metrics (lower is better), regression = increase
            # For rate metrics (higher is better), regression = decrease
            if metric in ["avg_execution_time", "latency", "error_rate"]:
                # Lower is better - regression when after > before
                if after_val > before_val and before_val > 0:
                    regression_pct = ((after_val - before_val) / before_val) * 100
                    if regression_pct > tolerance:
                        regressions.append({
                            "metric": metric,
                            "regression_percentage": regression_pct
                        })
            else:
                # Higher is better - regression when after < before
                if after_val < before_val and before_val > 0:
                    regression_pct = ((before_val - after_val) / before_val) * 100
                    if regression_pct > tolerance:
                        regressions.append({
                            "metric": metric,
                            "regression_percentage": regression_pct
                        })
        
        return {
            "check": "regression",
            "passed": len(regressions) == 0,
            "tolerance": tolerance,
            "regressions": regressions,
            "message": f"Regressions found: {len(regressions)}"
        }
    
    def evaluate_batch(
        self,
        optimizations: List[OptimizationRecord]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple optimizations and provide summary.
        
        Args:
            optimizations: List of optimization records
            
        Returns:
            Batch evaluation summary
        """
        results = []
        passed_count = 0
        failed_count = 0
        
        for opt in optimizations:
            passed = self.evaluate(opt)
            results.append({
                "record_id": opt.record_id,
                "tool_id": opt.tool_id,
                "passed": passed,
                "improvement": opt.improvement_percentage,
            })
            
            if passed:
                passed_count += 1
            else:
                failed_count += 1
        
        return {
            "total": len(optimizations),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": (passed_count / len(optimizations) * 100) if optimizations else 0,
            "results": results,
        }
    
    def get_evaluation_history(
        self,
        tool_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation history.
        
        Args:
            tool_id: Optional filter by tool ID
            limit: Maximum records to return
            
        Returns:
            List of evaluation records
        """
        history = self._evaluation_history
        
        if tool_id:
            history = [h for h in history if h.get("tool_id") == tool_id]
        
        return history[-limit:]
    
    def get_standards_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all standards.
        
        Returns:
            Standards summary
        """
        active_count = sum(1 for s in self._standards.values() if s.is_active)
        
        return {
            "total_standards": len(self._standards),
            "active_standards": active_count,
            "standards": [
                {
                    "id": sid,
                    "name": s.name,
                    "min_improvement": s.min_improvement_threshold,
                    "is_active": s.is_active,
                }
                for sid, s in self._standards.items()
            ],
        }
    
    def update_standard_threshold(
        self,
        standard_id: str,
        new_threshold: float
    ) -> bool:
        """
        Update the minimum improvement threshold for a standard.
        
        Args:
            standard_id: Standard ID
            new_threshold: New threshold value
            
        Returns:
            True if updated, False if not found
        """
        if standard_id not in self._standards:
            return False
        
        self._standards[standard_id].min_improvement_threshold = new_threshold
        return True
    
    def calculate_quality_score(
        self,
        optimization: OptimizationRecord
    ) -> float:
        """
        Calculate a quality score for an optimization.
        
        Score ranges from 0 to 100 based on how well the optimization
        performs across all standards.
        
        Args:
            optimization: Optimization to score
            
        Returns:
            Quality score (0-100)
        """
        if optimization.status != OptimizationStatus.COMPLETED:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        for standard_id, standard in self._standards.items():
            if not standard.is_active:
                continue
            
            weight = 1.0
            standard_score = 0.0
            
            # Score based on improvement vs threshold
            if standard.min_improvement_threshold > 0:
                ratio = optimization.improvement_percentage / standard.min_improvement_threshold
                standard_score = min(50, ratio * 50)  # Up to 50 points for improvement
            
            # Check for regressions
            regression_result = self._check_regression(optimization, standard)
            if regression_result["passed"]:
                standard_score += 25  # 25 points for no regressions
            
            # Check required metrics
            metrics_result = self._check_required_metrics(optimization, standard)
            if metrics_result["passed"]:
                standard_score += 25  # 25 points for required metrics
            
            score += standard_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return score / total_weight
        
        return 0.0
